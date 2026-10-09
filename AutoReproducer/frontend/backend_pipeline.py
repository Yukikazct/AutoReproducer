"""后台复现流水线执行器 + 进度事件存储（无 Streamlit 依赖，便于单元测试）。

用户点击前端"开始复现"后，Streamlit 主线程启动后台线程执行
run_pipeline_core()，把每个阶段的进度事件实时追加写入进度文件
(JSON Lines)。前端通过 ProgressStore.read_snapshot() 轮询聚合，
实现"后台运行 + 用户实时看到当前进度"：

事件类型：
  state    -> {"type": "state", "state": <FSM状态>, "agent": <显示名>,
               "phase_id": <本轮阶段ID>, "status": <执行状态>}
  pipeline_plan -> {"type": "pipeline_plan", "stages": <有序阶段列表>}
  log      -> {"type": "log", "log": <审计日志条目>}
  done     -> {"type": "done", "result": <完整流水线结果>}
  error    -> {"type": "error", "error": <后台线程捕获的异常消息>}

read_snapshot() 聚合以上事件为前端可直接渲染的视图：
  {state, agent_status, pipeline_stages, logs, result, done, error, running, updated_at}
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
            "execution_output": "", "repository_step": "",
            "execution_context": {}, "repository_step_label": "",
            "active_executions": [],
            "pipeline_stages": [],
        }
        stages: Dict[str, Dict[str, Any]] = {}
        stage_order: List[str] = []
        legacy_latest: Dict[tuple, str] = {}
        has_plan = False
        legacy_repository = False
        execution_numbers: Dict[str, int] = {}
        executions: Dict[tuple, Dict[str, Any]] = {}
        output_key = None
        labeled_output = False

        def execution_context(event: Dict[str, Any]) -> Dict[str, Any]:
            identifier = event.get("execution_id")
            identifier = identifier if isinstance(identifier, str) else ""
            index = event.get("step_index")
            index = index if type(index) is int and index > 0 else None
            previous = executions.get((identifier, index if identifier else None), {})
            step_id = event.get("step_id", previous.get("step_id", ""))
            step_id = step_id if isinstance(step_id, str) else ""
            count = event.get("step_count", previous.get("step_count"))
            count = count if type(count) is int and count > 0 and (index is None or count >= index) else None
            number = None
            label = "代码执行"
            if identifier:
                number = execution_numbers.setdefault(identifier, len(execution_numbers) + 1)
                label += f" · 第 {number} 轮"
                if index is not None:
                    label += f" · 步骤 {index}" + (f"/{count}" if count is not None else "")
            elif step_id:
                label += " · " + step_id
            return {"execution_id": identifier, "round_number": number, "step_id": step_id,
                    "step_index": index, "step_count": count, "label": label,
                    "phase_id": event.get("phase_id", previous.get("phase_id", "")),
                    "status": event.get("status", previous.get("status", ""))}

        def output_boundary(context: Dict[str, Any]) -> None:
            nonlocal output_key, labeled_output
            identifier = context["execution_id"]
            key = (identifier, context["step_index"] if identifier else context["step_id"])
            # Pure legacy streams keep their original text. A mixed stream gets
            # an explicit anonymous boundary instead of borrowing a newer ID.
            if key != output_key and (identifier or labeled_output):
                view["execution_output"] = (view["execution_output"] + f"\n── {context['label']} ──\n")[-16000:]
            output_key = key
            labeled_output = labeled_output or bool(identifier)

        def apply_stage(identifier: str, event: Dict[str, Any]) -> None:
            if identifier not in stages:
                stages[identifier] = {"id": identifier, "agent": event.get("agent", ""),
                                      "title": event.get("title", event.get("agent", identifier)),
                                      "description": event.get("description", ""), "status": "waiting"}
                stage_order.append(identifier)
            row = stages[identifier]
            status = event.get("status", "waiting")
            if status == "running":
                # A bounded correction starts a fresh attempt on the same row.
                row.pop("reason", None)
                row.pop("outcome", None)
            row["status"] = status
            for key in ("agent", "state", "attempt", "calls", "reason", "outcome"):
                if key in event:
                    row[key] = event[key]

        def finish_pending(reason: str, failed: bool) -> None:
            known_failure = any(row["status"] == "error" for row in stages.values())
            for row in stages.values():
                if row["status"] == "waiting":
                    row.update(status="blocked", reason=reason)
                elif row["status"] == "running":
                    # A pipeline-level failure cannot identify a second failed
                    # phase when an explicit phase error is already available.
                    # Running also means it started, so never label it unexecuted.
                    row.update(status="error" if failed and not known_failure else "blocked",
                               reason="流水线已终止，本阶段未正常结束，未收到完成结果")
                    agent = row.get("agent", "")
                    if agent and view["agent_status"].get(agent) == "running":
                        view["agent_status"][agent] = row["status"]

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
            if etype == "pipeline_plan":
                planned = ev.get("stages")
                if not isinstance(planned, list):
                    continue
                has_plan = True
                planned_order = []
                for stage in planned:
                    if not isinstance(stage, dict) or not isinstance(stage.get("id"), str):
                        continue
                    identifier = stage["id"]
                    if identifier in planned_order:
                        continue
                    # select_experiment may already have finished before this
                    # profile-specific plan is available; keep observed state.
                    observed = stages.get(identifier, {})
                    stages[identifier] = {**stage, **observed}
                    stages[identifier].setdefault("status", "waiting")
                    for key in ("agent", "title", "description"):
                        if key in stage:
                            stages[identifier][key] = stage[key]
                    planned_order.append(identifier)
                stage_order = planned_order + [identifier for identifier in stage_order
                                               if identifier not in planned_order]
            elif etype == "state":
                view["state"] = ev.get("state", view["state"])
                agent = ev.get("agent", "")
                if agent:
                    view["agent_status"][agent] = ev.get("status", "waiting")
                    identifier = ev.get("phase_id")
                    if identifier:
                        apply_stage(identifier, ev)
                    elif not has_plan and ev.get("status") != "waiting":
                        # Historical files have no phase IDs. Preserve event
                        # order and each running-after-terminal occurrence,
                        # rather than pretending that role cards are stages.
                        pair = (ev.get("state", ""), agent)
                        identifier = legacy_latest.get(pair)
                        previous = stages.get(identifier, {}) if identifier else {}
                        if not identifier or (ev.get("status") == "running" and
                                              previous.get("status") in {"success", "error", "skipped", "blocked"}):
                            identifier = f"legacy_{len(stages) + 1}"
                            legacy_latest[pair] = identifier
                            title = {"READ_PAPER": "论文解析", "FIND_RESOURCES": "资源准备",
                                     "BUILD_ENV": "环境分析与准备", "EXECUTE_CODE": "代码执行",
                                     "VALIDATE": "结果验证", "GENERATE_REPORT": "报告生成"}.get(pair[0], pair[0] or agent)
                            if agent.endswith("Optimizer"):
                                title = "智能优化（未启用）" if ev.get("status") == "skipped" else "智能优化"
                            elif agent.endswith("Verifier"):
                                title += "质量核验"
                            elif agent.endswith("SourceLoader"):
                                title = "加载公开证据"
                                legacy_repository = True
                            apply_stage(identifier, {**ev, "title": title,
                                "description": "依据历史进度事件的真实出现顺序恢复"})
                        else:
                            apply_stage(identifier, ev)
            elif etype == "log":
                lg = ev.get("log")
                if isinstance(lg, dict):
                    view["logs"].append(lg)
            elif etype in {"repository_step", "execution_step"}:
                if not view["running"]:
                    continue  # A delayed worker cannot reopen a terminal pipeline.
                view["repository_step"] = ev.get("step_id", "")
                context = execution_context(ev)
                key = (context["execution_id"], context["step_index"] if context["execution_id"] else None)
                executions[key] = context
                if context["status"] == "running":
                    output_boundary(context)
                view["execution_context"] = context
                view["repository_step_label"] = context["label"]
                legacy_repository = legacy_repository or etype == "repository_step"
            elif etype in {"repository_output", "execution_output"}:
                # A bounded live viewport; complete streams remain in run logs/report.
                context = execution_context(ev)
                text = ev.get("text", "")
                if text:
                    output_boundary(context)
                    view["execution_output"] = (view["execution_output"] + text)[-16000:]
            elif etype == "done":
                view["done"] = True
                view["running"] = False
                view["result"] = ev.get("result")
                legacy_repository = legacy_repository or bool(
                    ((view["result"] or {}).get("data") or {}).get("experiment_spec"))
                view["state"] = (ev.get("result") or {}).get(
                    "state", view["state"]) or view["state"]
                failed = view["state"] == "ERROR"
                finish_pending("上游阶段失败，本阶段未执行" if failed else
                               "流水线已结束，本阶段未执行", failed)
            elif etype == "error":
                view["error"] = ev.get("error")
                view["running"] = False
                view["state"] = "ERROR"
                finish_pending("流水线异常终止，本阶段未执行", True)
            view["updated_at"] = ev.get("at", view["updated_at"])
        view["pipeline_stages"] = [stages[identifier] for identifier in stage_order]
        if not view["running"]:
            for context in executions.values():
                if context["status"] == "running":
                    context["status"] = "interrupted"
        view["active_executions"] = [context for context in executions.values() if context["status"] == "running"]
        if not has_plan and legacy_repository:
            for row in view["pipeline_stages"]:
                if row.get("agent", "").endswith("Verifier"):
                    if row.get("state") == "BUILD_ENV":
                        row["title"] = "训练前证据预审"
                    elif row.get("state") == "VALIDATE":
                        row["title"] = "训练后核验"
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
                      api_key: str = "", mock_mode: bool = False,
                      max_trials: int = 10,
                      use_docker: bool = False,
                      workspace_dir: Optional[str] = None,
                      experiment_profile: Optional[str] = None,
                      use_llm_review: bool = False,
                      allow_result_summary_review: bool = False,
                      enable_optimization: bool = False,
                      prepare_environment: bool = False,
                      optimization_mode: str = "off",
                      max_candidates: int = 3,
                      budget_seconds: int = 7200,
                      prepare_only: bool = False,
                      offline: bool = False) -> Dict[str, Any]:
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

    display_names = {"PaperReader": "📖 PaperReader", "ResourceFinder": "🔍 ResourceFinder",
                     "EnvBuilder": "🔧 EnvBuilder", "CodeExecutor": "⚡ CodeExecutor",
                     "ResultValidator": "✅ ResultValidator", "Optimizer": OPTIMIZER_NAME,
                     "ReportGenerator": REPORTER_NAME, "Verifier": VERIFIER_NAME}

    def _on_event(event):
        if event.get("type") == "state":
            _set_current(event["state"])
            store.emit({**event, "agent": display_names.get(event.get("agent"), event.get("agent", ""))})
        elif event.get("type") in {"repository_step", "repository_output", "execution_step", "execution_output"}:
            store.emit(event)
        elif event.get("type") == "pipeline_plan":
            store.emit({**event, "stages": [
                {**stage, "agent": display_names.get(stage.get("agent"), stage.get("agent", ""))}
                for stage in event.get("stages", [])]})
        _emit_new_logs()

    try:
        outcome = orchestrator.run({
            "paper_title": paper_title, "pdf_path": pdf_path,
            "corpus_paper": corpus_paper, "experiment_profile": experiment_profile,
            "prepare_only": prepare_only, "offline": offline,
            "use_llm_review": use_llm_review,
            "allow_result_summary_review": allow_result_summary_review,
            "enable_optimization": enable_optimization,
            "prepare_environment": prepare_environment, "optimization_mode": optimization_mode,
            "max_candidates": max_candidates, "budget_seconds": budget_seconds,
        }, on_event=_on_event)
        data = outcome["data"]
        error_msg = outcome.get("error")
        _set_current(outcome["state"])
    except Exception as exc:
        error_msg = f"流水线异常: {exc}"
        logger.log("Orchestrator", "run", "ERROR", error_msg)
        data = getattr(orchestrator, "data", {})
        _set_current("ERROR")
    _emit_new_logs()

# 报告落盘（供历史记录与下载）
    report_path = ""
    report_text = data.get("report", "")
    if report_text:
        try:
            reports_dir = Path(__file__).resolve().parents[1] / "data" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            # 使用流水线 session_id 作为文件名时间戳，与 ledger 文件名保持一致
            ts = logger.session_id
            paper_title_safe = (data.get("paper_info", {}).get("title", "") or data.get("paper_title", "") or "report")[:30]
            # 去除非法字符
            paper_title_safe = "".join(c for c in paper_title_safe if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
            report_file = reports_dir / f"{paper_title_safe}_{ts}.md"
            from src.agents.report_generator import ReportGeneratorAgent
            data["report_path"] = str(report_file.resolve())
            report_text = ReportGeneratorAgent(logger).run(data, report_path=report_file)["report"]
            report_file.write_text(report_text, encoding="utf-8")
            report_path = str(report_file)

            # 完整执行输出附件：报告现已**全文内嵌** code/stdout/stderr，
            # 附件保留作为可直接下载/归档的纯文本旁路（终端里 wc/grep、
            # 或想脱离 Markdown 单独留档时用）。
            execution_raw = data.get("execution", {}) or {}
            final_raw = execution_raw.get("final", {}) or {}
            attach_lines = ["# 完整执行输出（未被截断）",
                            "", "## 生成代码", "```python",
                            execution_raw.get("code", "（无）"), "```",
                            "", "## 标准输出 (full)", "```",
                            final_raw.get("stdout", "（无输出）"), "```",
                            "", "## 诊断输出 (stderr)", "```",
                            final_raw.get("stderr", "（无）"), "```", ""]
            attach_file = reports_dir / f"{paper_title_safe}_{ts}_execution.txt"
            attach_file.write_text("\n".join(attach_lines), encoding="utf-8")
            # 在报告末尾补充附件说明，保证用户知道完整输出的位置
            report_text += (f"\n\n---\n\n> 📎 **完整执行输出附件**: "
                            f"`{attach_file.name}`（同一目录下，未被截断）\n")
            report_file.write_text(report_text, encoding="utf-8")
            data["report"] = report_text
        except Exception:
            pass  # 落盘失败不阻断主流程

    # 终态落盘：无论 COMPLETED 还是 ERROR，都在 ledger 末条写终态信息，
    # 供历史列表回填 state / duration_sec / llm_calls。
    stats = logger.get_stats()
    logger.log_experiment(
        "FINISH", "流水线终止",
        inputs={"paper_title": paper_title},
        outputs={"title": (data.get("paper_info") or {}).get("title") or paper_title},
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
                            api_key: str = "", mock_mode: bool = False,
                            max_trials: int = 10,
                            use_docker: bool = False,
                            workspace_dir: Optional[str] = None,
                            experiment_profile: Optional[str] = None,
                            use_llm_review: bool = False,
                            allow_result_summary_review: bool = False,
                            enable_optimization: bool = False,
                            prepare_environment: bool = False,
                            optimization_mode: str = "off",
                            max_candidates: int = 3,
                            budget_seconds: int = 7200,
                            prepare_only: bool = False,
                            offline: bool = False,
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
                prepare_only=prepare_only, offline=offline, use_llm_review=use_llm_review,
                allow_result_summary_review=allow_result_summary_review,
                enable_optimization=enable_optimization, prepare_environment=prepare_environment,
                optimization_mode=optimization_mode, max_candidates=max_candidates, budget_seconds=budget_seconds)
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
