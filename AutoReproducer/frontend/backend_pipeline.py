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

架构说明（驱动器统一，修复 UI 路径漏拉资源的缺陷）：
run_pipeline_core 是 Orchestrator（唯一 FSM 驱动器）的薄封装——通过
progress_cb 回调把阶段状态实时转写为进度事件。此前这里自持一份
流水线循环，直接驱动各 Agent，漏掉了 Orchestrator 的资源懒加载
（git clone 官方仓库）、Docker 镜像构建与验证修正闭环，导致
「发现了官方仓库却从未下载、复现跑的是 LLM 占位脚本」。
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger
from frontend.report_store import schedule_progress_cleanup

# ---------------- 流水线 Agent 定义（与前端卡片一致） ----------------

AGENTS = [
    ("READ_PAPER", "PaperReader", "reader"),
    ("FIND_RESOURCES", "ResourceFinder", "finder"),
    ("BUILD_ENV", "EnvBuilder", "builder"),
    ("PLAN_EXECUTION", "ExecutionPlanner", "planner"),
    ("EXECUTE_CODE", "CodeExecutor", "executor"),
    ("VALIDATE", "ResultValidator", "validator"),
]

# 展示名映射已收敛到 Orchestrator.STAGE_DISPLAY（单一事实来源），
# 此处仅保留进度事件需要独立使用的常量。
VERIFIER_NAME = "Verifier"


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
                      workspace_dir: Optional[str] = None,
                      experiment_profile: str = "",
                      use_llm_pipeline: bool = False) -> Dict[str, Any]:
    """后台执行完整复现流水线（复现 -> 验证 -> 优化 -> 报告）。

    Orchestrator（唯一 FSM 驱动器）的薄封装：通过 progress_cb 把阶段
    状态实时转写为进度事件。与前端解耦：不触碰 st.session_state，
    进度实时写入 progress_path；返回最终 result（供 done 事件与前端
    结果摘要复用）。临时 PDF 由调用方负责清理。
    """
    store = ProgressStore(progress_path)
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

    def _cb(state_name: str, display_name: str, status: str) -> None:
        _emit_state(store, state_name, display_name, status)
        # 阶段状态变化时同步增量推送审计日志，保持 UI 实时可见
        _emit_new_logs()

    llm = LLMClient(
        mock_mode=mock_mode,
        model="" if mock_mode else model_name,
        base_url=base_url,
        api_key=api_key,
    )
    orchestrator = Orchestrator(llm_client=llm, mock_mode=mock_mode,
                                logger=logger, max_trials=max_trials,
                                use_docker=use_docker,
                                workspace_dir=workspace_dir,
                                progress_cb=_cb)

    try:
        orch_result = orchestrator.run({
            "paper_title": paper_title,
            "pdf_path": pdf_path,
            "corpus_paper": corpus_paper,
            "experiment_profile": "" if use_llm_pipeline else experiment_profile,
            "use_llm_pipeline": use_llm_pipeline,
        })
    except Exception as e:
        error_msg = f"流水线异常: {e}"
        logger.log("Orchestrator", "run", "ERROR", f"异常: {e}")
        store.emit({"type": "error", "error": error_msg})
        orch_result = {
            "state": "ERROR", "error": error_msg, "data": {},
            "audit_logs": logger.get_summary(),
            "audit_stats": logger.get_stats(),
        }
    else:
        error_msg = orch_result.get("error")

    data = orch_result.get("data", {}) or {}
    state = orch_result.get("state", "ERROR")
    # Verifier 卡片状态（与旧事件序列保持一致）
    _emit_state(store, "VALIDATE" if state != "ERROR" else state,
                VERIFIER_NAME, "success")
    _emit_new_logs()

    # 报告默认只用于当前会话展示，用户选择保存后才写入 reports/。
    report_path = ""

    # 终态落盘：无论 COMPLETED 还是 ERROR，都在 ledger 末条写终态信息，
    # 供历史列表回填 state / duration_sec / llm_calls。
    stats = logger.get_stats()
    logger.log_experiment(
        "FINISH", "流水线终止",
        inputs={"paper_title": paper_title},
        outputs={},
        result={"state": state,
                "duration_sec": stats["duration_sec"],
                "llm_calls": stats["llm_calls"]})

    result = {
        "state": state,
        "error": error_msg,
        "data": data,
        "audit_logs": logger.get_summary(),
        "audit_stats": stats,
        "report_path": report_path,
        "report_saved": False,
        "session_id": logger.session_id,
    }
    _emit_state(store, state, "", "success")
    _emit_new_logs()
    store.emit({"type": "done", "result": result})
    schedule_progress_cleanup(progress_path)
    return result


def run_pipeline_background(progress_path: str, *,
                            paper_title: str = "", pdf_path: str = "",
                            corpus_paper: Optional[str] = None,
                            model_name: str = "", base_url: str = "",
                            api_key: str = "", mock_mode: bool = True,
                            max_trials: int = 10,
                            use_docker: bool = False,
                            workspace_dir: Optional[str] = None,
                            experiment_profile: str = "",
                            use_llm_pipeline: bool = False,
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
                workspace_dir=workspace_dir, experiment_profile=experiment_profile,
                use_llm_pipeline=use_llm_pipeline)
            if on_done:
                try:
                    on_done(result)
                except Exception:
                    pass
        except Exception as e:      # 兜底：流水线外层异常也写入进度
            store.emit({"type": "error", "error": str(e)})
            schedule_progress_cleanup(progress_path)
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
