"""后台复现流水线执行器 + 进度事件存储（无 Streamlit 依赖，便于单元测试）。

用户点击前端"开始复现"后，Streamlit 主线程启动后台线程执行
run_pipeline_core()，把每个阶段的进度事件实时追加写入进度文件
(JSON Lines)。前端通过 ProgressStore.read_snapshot() 轮询聚合，
实现"后台运行 + 用户实时看到当前进度"：

事件类型：
  state    -> {"type": "state", "state": <FSM状态>, "agent": <显示名>,
               "status": "running"|"success"|"error"|"waiting"}
  log      -> {"type": "log", "log": <审计日志条目>}
  done     -> {"type": "done", "result": <完整流水线结果>}
  error    -> {"type": "error", "error": <后台线程捕获的异常消息>}

read_snapshot() 聚合以上事件为前端可直接渲染的视图：
  {state, agent_status, logs, result, done, error, running, updated_at}
"""
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger

# ---------------- 流水线 Agent 定义（与前端卡片一致） ----------------

AGENTS = [
    ("READ_PAPER", "📖 PaperReader", "reader"),
    ("FIND_RESOURCES", "🔍 ResourceFinder", "finder"),
    ("BUILD_ENV", "🔧 EnvBuilder", "builder"),
    ("EXECUTE_CODE", "⚡ CodeExecutor", "executor"),
    ("VALIDATE", "✅ ResultValidator", "validator"),
]

OPTIMIZER_NAME = "🧪 Optimizer"
REPORTER_NAME = "📝 ReportGenerator"
VERIFIER_NAME = "🛡️ Verifier"


# ---------------- 进度事件存储 ----------------

class ProgressStore:
    """将流水线进度事件写入 JSON Lines 文件，供前端轮询读取。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 清空旧内容（每次复现从零开始）
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._start_ts = time.time()

    def emit(self, event: Dict[str, Any]) -> None:
        """追加一条进度事件（线程安全）。"""
        record = {
            "ts": round(time.time() - self._start_ts, 2),
            "at": datetime.now().isoformat(),
            **event,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def read_snapshot(path: str) -> Dict[str, Any]:
        """读取并聚合进度文件为前端渲染视图（不可用时返回空视图）。"""
        p = Path(path)
        view: Dict[str, Any] = {
            "state": "INIT", "agent_status": {}, "logs": [],
            "result": None, "done": False, "error": None,
            "running": True, "updated_at": "",
        }
        if not p.exists():
            return view
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            return view
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "state":
                view["state"] = ev.get("state", view["state"])
                agent = ev.get("agent", "")
                if agent:
                    view["agent_status"][agent] = ev.get("status", "waiting")
            elif etype == "log":
                lg = ev.get("log")
                if isinstance(lg, dict):
                    view["logs"].append(lg)
            elif etype == "done":
                view["done"] = True
                view["running"] = False
                view["result"] = ev.get("result")
                view["state"] = (ev.get("result") or {}).get(
                    "state", view["state"]) or view["state"]
            elif etype == "error":
                view["error"] = ev.get("error")
                view["running"] = False
            view["updated_at"] = ev.get("at", view["updated_at"])
        return view


# ---------------- 后台流水线执行 ----------------

def _emit_state(store: ProgressStore, state: str, agent: str,
                status: str) -> None:
    store.emit({"type": "state", "state": state,
                "agent": agent, "status": status})


def run_pipeline_core(progress_path: str,
                      paper_title: str = "", pdf_path: str = "",
                      corpus_paper: Optional[str] = None,
                      model_name: str = "", base_url: str = "",
                      api_key: str = "", mock_mode: bool = True,
                      max_trials: int = 10,
                      use_docker: bool = False,
                      workspace_dir: Optional[str] = None) -> Dict[str, Any]:
    """后台执行完整复现流水线（复现 -> 验证 -> 优化 -> 报告）。

    与前端解耦：不触碰 st.session_state，进度实时写入 progress_path；
    返回最终 result（与 app.py 旧 run_pipeline 结构一致，供 done 事件与
    前端结果摘要复用）。临时 PDF 由调用方负责清理。
    """
    store = ProgressStore(progress_path)
    state: Dict[str, Any] = {"current": "INIT"}
    running: Dict[str, Any] = {"value": True}
    pdf_path = pdf_path or ""

    def _current() -> str:
        return state["current"]

    def _set_current(s: str) -> None:
        state["current"] = s

    def _mark_result(result: Dict[str, Any]) -> Dict[str, Any]:
        running["value"] = False
        store.emit({"type": "done", "result": result})
        return result

    logger = AuditLogger()
    error_msg: Optional[str] = None
    emitted = 0

    def _emit_new_logs() -> None:
        """只把新增的审计日志追加进进度文件，避免每次全量重发造成重复。"""
        nonlocal emitted
        entries = logger.get_summary()
        for entry in entries[emitted:]:
            store.emit({"type": "log", "log": entry})
        emitted = len(entries)

    llm = LLMClient(
        mock_mode=mock_mode,
        model="" if mock_mode else model_name,
        base_url=base_url,
        api_key=api_key,
    )
    orchestrator = Orchestrator(llm_client=llm, mock_mode=mock_mode,
                                logger=logger, max_trials=max_trials,
                                use_docker=use_docker,
                                workspace_dir=workspace_dir)

    data = {
        "paper_title": paper_title,
        "pdf_path": pdf_path,
        "corpus_paper": corpus_paper,
    }

    for stage_name, display_name, agent_key in AGENTS:
        _set_current(stage_name)
        _emit_state(store, stage_name, display_name, "running")
        agent = orchestrator.agents[agent_key]
        try:
            result = agent.run(data)

            if stage_name == "READ_PAPER":
                data["paper_info"] = result.get("paper_info", {})
                data["raw_text"] = result.get("raw_text", "")
            elif stage_name == "FIND_RESOURCES":
                data["resources"] = result.get("resources", {})
            elif stage_name == "BUILD_ENV":
                data["env_config"] = result.get("env_config", {})
            elif stage_name == "EXECUTE_CODE":
                data["execution"] = result
            elif stage_name == "VALIDATE":
                data["validation"] = result

            # Prompt-Free 验证（与前端旧逻辑一致）
            verif = orchestrator.agents["verifier"].run({
                "agent_name": agent.name,
                "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
                "output": result,
            })
            data.setdefault("verifications", []).append(
                {"state": stage_name, "agent": agent.name, **verif})

            data["total_llm_calls"] = data.get("total_llm_calls", 0) + \
                int(result.get("llm_calls", 0) or 0) + \
                int(verif.get("llm_calls", 0) or 0)
            logger.add_llm_calls(int(result.get("llm_calls", 0) or 0) + int(verif.get("llm_calls", 0) or 0))

            _emit_state(store, stage_name, display_name, "success")
            _emit_new_logs()
        except Exception as e:
            error_msg = f"{stage_name} 阶段异常: {e}"
            logger.log(stage_name, "run", "ERROR", f"异常: {e}")
            _emit_state(store, stage_name, display_name, "error")
            _set_current("ERROR")
            _emit_new_logs()
            break

    _emit_state(store, _current(), VERIFIER_NAME, "success")

    # 优化阶段：仅在复现成功后触发
    if _current() != "ERROR":
        if data.get("validation", {}).get("is_reproduced"):
            _set_current("OPTIMIZING")
            _emit_state(store, "OPTIMIZING", OPTIMIZER_NAME, "running")
            try:
                data["optimization"] = orchestrator.agents["optimizer"].run(data)
                _set_current("OPTIMIZED")
                _emit_state(store, "OPTIMIZED", OPTIMIZER_NAME, "success")
            except Exception as e:
                error_msg = f"优化阶段异常: {e}"
                logger.log(OPTIMIZER_NAME, "optimize", "ERROR", f"异常: {e}")
                _set_current("ERROR")
                _emit_state(store, "ERROR", OPTIMIZER_NAME, "error")
        else:
            validation = data.get("validation", {}) or {}
            if validation.get("status") == "not_runnable":
                reason = ("代码未能运行，无法优化（"
                          f"{validation.get('reason', '未运行')}）")
            else:
                reason = "复现未成功,跳过优化"
            data["optimization"] = {"optimized": False, "reason": reason}
            _emit_state(store, _current(), OPTIMIZER_NAME, "waiting")
        _emit_new_logs()

    # 报告生成（合并复现 + 优化）
    if _current() != "ERROR":
        _set_current("GENERATE_REPORT")
        _emit_state(store, "GENERATE_REPORT", REPORTER_NAME, "running")
        # 在报告生成前注入审计统计，确保报告能展示
        data["audit_stats"] = logger.get_stats()
        try:
            data["report"] = orchestrator.agents["reporter"].run(data) \
                .get("report", "")
            _emit_state(store, "GENERATE_REPORT", REPORTER_NAME, "success")
        except Exception as e:
            error_msg = f"报告生成阶段异常: {e}"
            logger.log(REPORTER_NAME, "generate_report", "ERROR", f"异常: {e}")
            _set_current("ERROR")
            _emit_state(store, "ERROR", REPORTER_NAME, "error")
        _emit_new_logs()

    if _current() != "ERROR":
        _set_current("COMPLETED")
        data["audit_stats"] = logger.get_stats()
        logger.log("Orchestrator", "finish_pipeline", "SUCCESS",
                   "流水线完成", data.get("audit_stats"))

# 报告落盘（供历史记录与下载）
    report_path = ""
    report_text = data.get("report", "")
    if report_text:
        try:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            # 使用流水线 session_id 作为文件名时间戳，与 ledger 文件名保持一致
            ts = logger.session_id
            paper_title_safe = (data.get("paper_info", {}).get("title", "") or data.get("paper_title", "") or "report")[:30]
            # 去除非法字符
            paper_title_safe = "".join(c for c in paper_title_safe if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
            report_file = reports_dir / f"{paper_title_safe}_{ts}.md"
            report_file.write_text(report_text, encoding="utf-8")
            report_path = str(report_file)

            # 完整执行输出附件：报告内仅截断展示，完整 code/stdout/stderr 落盘
            # 供需要全文时查看（修复「输出一半」问题：附件永远完整）。
            execution_raw = data.get("execution", {}) or {}
            final_raw = execution_raw.get("final", {}) or {}
            attach_lines = ["# 完整执行输出（未被截断）",
                            "", "## 生成代码", "```python",
                            execution_raw.get("code", "（无）"), "```",
                            "", "## 标准输出 (full)", "```",
                            final_raw.get("stdout", "（无输出）"), "```",
                            "", "## 错误输出 (stderr)", "```",
                            final_raw.get("stderr", "（无）"), "```", ""]
            attach_file = reports_dir / f"{paper_title_safe}_{ts}_execution.txt"
            attach_file.write_text("\n".join(attach_lines), encoding="utf-8")
            # 在报告末尾补充附件说明，保证用户知道完整输出的位置
            report_text += (f"\n\n---\n\n> 📎 **完整执行输出附件**: "
                            f"`{attach_file.name}`（同一目录下，未被截断）\n")
            report_file.write_text(report_text, encoding="utf-8")
        except Exception:
            pass  # 落盘失败不阻断主流程

    # 终态落盘：无论 COMPLETED 还是 ERROR，都在 ledger 末条写终态信息，
    # 供历史列表回填 state / duration_sec / llm_calls。
    stats = logger.get_stats()
    logger.log_experiment(
        "FINISH", "流水线终止",
        inputs={"paper_title": paper_title},
        outputs={},
        result={"state": _current(),
                "duration_sec": stats["duration_sec"],
                "llm_calls": stats["llm_calls"]})

    result = {
        "state": _current(),
        "error": error_msg,
        "data": data,
        "audit_logs": logger.get_summary(),
        "audit_stats": stats,
        "report_path": report_path,
        "session_id": logger.session_id,
    }
    _emit_state(store, _current(), "", "success")
    return _mark_result(result)


def run_pipeline_background(progress_path: str, *,
                            paper_title: str = "", pdf_path: str = "",
                            corpus_paper: Optional[str] = None,
                            model_name: str = "", base_url: str = "",
                            api_key: str = "", mock_mode: bool = True,
                            max_trials: int = 10,
                            use_docker: bool = False,
                            workspace_dir: Optional[str] = None,
                            cleanup_pdf: bool = True,
                            on_done=None) -> threading.Thread:
    """启动后台线程执行流水线；返回守护线程句柄。

    线程内捕获一切异常并写入 error 事件；临时 PDF 默认在线程结束时删除
    （cleanup_pdf）。on_done(result) 在线程收尾时回调（可选）。
    """
    def _worker() -> None:
        store = ProgressStore(progress_path)
        try:
            result = run_pipeline_core(
                progress_path, paper_title=paper_title, pdf_path=pdf_path,
                corpus_paper=corpus_paper, model_name=model_name,
                base_url=base_url, api_key=api_key, mock_mode=mock_mode,
                max_trials=max_trials, use_docker=use_docker,
                workspace_dir=workspace_dir)
            if on_done:
                try:
                    on_done(result)
                except Exception:
                    pass
        except Exception as e:      # 兜底：流水线外层异常也写入进度
            store.emit({"type": "error", "error": str(e)})
            if on_done:
                try:
                    on_done(None)
                except Exception:
                    pass
        finally:
            if cleanup_pdf and pdf_path:
                try:
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                except OSError:
                    pass

    t = threading.Thread(target=_worker, name="autorepro-pipeline",
                         daemon=True)
    t.start()
    return t