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
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger


# Contract checked by the web loader before starting a new background thread.
BACKEND_API_VERSION = 4


def __getattr__(name):
    # Recovery must remain importable when the host is missing app dependencies
    # such as filelock. Preserve the exported class for subclassing and callers
    # that explicitly request it, while deferring the full agent graph import.
    if name != "Orchestrator":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from src.orchestrator import Orchestrator as implementation
    globals()[name] = implementation
    return implementation


def _create_orchestrator(*args, **kwargs):
    implementation = globals().get("Orchestrator") or __getattr__("Orchestrator")
    return implementation(*args, **kwargs)

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

    def __init__(self, path: str, *, reset: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 清空旧内容（每次复现从零开始）
        if reset:
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
        from frontend.progress_state import read_progress_snapshot
        return read_progress_snapshot(path)


# ---------------- 后台流水线执行 ----------------

def _emit_state(store: ProgressStore, state: str, agent: str,
                status: str) -> None:
    store.emit({"type": "state", "state": state,
                "agent": agent, "status": status})


def _reject_pdf_input(store: ProgressStore, reason: str) -> Dict[str, Any]:
    """Reject unreadable evidence before resource search or code generation."""
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "read_paper", "agent": "📖 PaperReader", "title": "论文解析", "status": "waiting"}]})
    store.emit({"type": "state", "state": "READ_PAPER", "agent": "📖 PaperReader",
                "phase_id": "read_paper", "status": "error", "reason": reason})
    result = {"state": "ERROR", "error": reason,
              "data": {"pdf_input": {"status": "rejected", "reason": reason}},
              "audit_logs": [], "audit_stats": {}, "report_path": "", "session_id": ""}
    store.emit({"type": "done", "result": result})
    return result


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
                      offline: bool = False,
                      title_resolution: Optional[dict] = None,
                      pdf_resolution: Optional[dict] = None,
                      _managed_runtime: bool = False,
                      _append_progress: bool = False,
                      _runtime_preparation: Optional[dict] = None) -> Dict[str, Any]:
    """后台执行完整复现流水线（复现 -> 验证 -> 优化 -> 报告）。

    与前端解耦：不触碰 st.session_state，进度实时写入 progress_path；
    返回最终 result（与 app.py 旧 run_pipeline 结构一致，供 done 事件与
    前端结果摘要复用）。临时 PDF 由调用方负责清理。
    """
    store = ProgressStore(progress_path, reset=not _append_progress)
    from src.title_routing import resolve_title_request
    resolved = {"paper_title": paper_title, "pdf_path": pdf_path,
                "corpus_paper": corpus_paper, "mock_mode": mock_mode,
                "experiment_profile": experiment_profile}
    pdf_parser_reason = None
    if pdf_path:
        from src.pdf_input import resolve_pdf_request, PDFInputError, PDFParserUnavailable
        try:
            resolved = resolve_pdf_request(resolved)
            paper_title = resolved.get("paper_title", paper_title)
            pdf_resolution = resolved.get("pdf_resolution")
        except PDFParserUnavailable as exc:
            if not _managed_runtime and not mock_mode and not use_docker:
                pdf_parser_reason = str(exc)
            else:
                return _reject_pdf_input(store, str(exc))
        except PDFInputError as exc:
            return _reject_pdf_input(store, str(exc))
    resolved = resolve_title_request(resolved)
    experiment_profile = resolved.get("experiment_profile")
    title_resolution = title_resolution or resolved.get("title_resolution")
    if title_resolution:
        store.emit({"type": "title_resolution", **title_resolution})
    if pdf_resolution:
        store.emit({"type": "pdf_resolution", **pdf_resolution})
    if not mock_mode and not use_docker and not _managed_runtime:
        from src.runtime_preparation import runtime_requirement
        reason = pdf_parser_reason or runtime_requirement()
        if reason:
            request = {
                "progress_path": str(Path(progress_path).resolve()),
                "paper_title": paper_title, "pdf_path": pdf_path,
                "corpus_paper": corpus_paper, "model_name": model_name,
                "base_url": base_url, "api_key": api_key, "mock_mode": mock_mode,
                "max_trials": max_trials, "use_docker": use_docker,
                "workspace_dir": workspace_dir, "experiment_profile": experiment_profile,
                "use_llm_review": use_llm_review,
                "allow_result_summary_review": allow_result_summary_review,
                "enable_optimization": enable_optimization,
                "prepare_environment": prepare_environment, "optimization_mode": optimization_mode,
                "max_candidates": max_candidates, "budget_seconds": budget_seconds,
                "prepare_only": prepare_only, "offline": offline,
                "title_resolution": title_resolution,
                "pdf_resolution": pdf_resolution,
            }
            return _run_prepared_pipeline(store, request, reason)
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
    orchestrator = _create_orchestrator(llm_client=llm, mock_mode=mock_mode,
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
        elif event.get("type") in {"repository_step", "repository_output", "execution_step", "execution_output", "execution_run", "repository_environment"}:
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
            "title_resolution": title_resolution,
            "pdf_resolution": pdf_resolution,
        }, on_event=_on_event)
        data = outcome["data"]
        error_msg = outcome.get("error")
        _set_current(outcome["state"])
    except Exception as exc:
        error_msg = f"流水线异常: {exc}"
        logger.log("Orchestrator", "run", "ERROR", error_msg)
        data = getattr(orchestrator, "data", {})
        _set_current("ERROR")
    if _runtime_preparation:
        data["runtime_preparation"] = _runtime_preparation
    if title_resolution:
        data["title_resolution"] = title_resolution
    if pdf_resolution:
        data["pdf_resolution"] = pdf_resolution
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


def _run_prepared_pipeline(store: ProgressStore, request: dict, reason: str) -> dict:
    """Recover the host, then continue this reproduction in an owned interpreter.

    Only the request pipe carries credentials. The existing progress file remains
    the single source of truth for the page throughout recovery and training.
    """
    from src.runtime_preparation import (
        prepare_runtime, run_owned_process, safe_runtime_diagnostic,
        PRESET_WORKER_PREPARATION_ALLOWANCE_S,
    )
    stage = {"id": "prepare_runtime", "agent": "🔧 EnvBuilder",
             "title": "自动准备运行环境", "description": reason, "status": "waiting"}
    store.emit({"type": "pipeline_plan", "pipeline": "repository" if request.get("experiment_profile") else "generic", "stages": [stage]})
    def state(status, message):
        store.emit({"type": "state", "state": "BUILD_ENV", "agent": stage["agent"],
                    "phase_id": stage["id"], "status": status, "reason": message})
    state("running", reason)
    metadata = None
    worker_id = "worker_" + uuid.uuid4().hex
    worker_started = False
    worker_cleaned = False
    worker_finished = False

    def confirmed_cleanup():
        nonlocal worker_cleaned
        worker_cleaned = True

    def finish_worker(status):
        nonlocal worker_finished
        if not worker_started or worker_finished:
            return
        # Only the process owner can confirm cleanup after its retained job has
        # closed. A stale child's log cannot prove that its processes exited.
        if worker_cleaned:
            snapshot = ProgressStore.read_snapshot(str(store.path))
            for run in snapshot.get("active_runs", []):
                store.emit({"type": "execution_run", "execution_id": run["execution_id"],
                            "phase_id": run.get("phase_id", ""), "status": "interrupted",
                            "cleanup_confirmed": True,
                            "reason": "持有者进程已退出并清理，但未返回本轮执行终态"})
        store.emit({"type": "pipeline_worker", "worker_id": worker_id,
                    "status": status, "cleanup_confirmed": worker_cleaned})
        worker_finished = worker_cleaned

    effective_key = request.get("api_key") or os.environ.get("LLM_API_KEY", "")
    try:
        prepared = prepare_runtime(Path(__file__).resolve().parents[1],
                                   offline=request.get("offline", False),
                                   progress_callback=store.emit)
        metadata = asdict(prepared)
        state("success", "兼容运行环境已就绪，继续本次复现")
        store.emit({"type": "runtime_preparation", "status": "success", **metadata})
        payload = {**request, "api_key": effective_key,
                   "_managed_runtime": True, "_append_progress": True,
                   "_runtime_preparation": metadata}
        worker_env = os.environ.copy()
        worker_env.pop("LLM_API_KEY", None)
        worker_started = True
        store.emit({"type": "pipeline_worker", "worker_id": worker_id, "status": "running"})
        completed = run_owned_process(
            [prepared.executable, "-X", "utf8", "-m", "frontend.preset_worker"],
            cwd=Path(__file__).resolve().parents[1], input=json.dumps(payload), env=worker_env,
            timeout_s=max(7200, int(request.get("budget_seconds", 7200))) + PRESET_WORKER_PREPARATION_ALLOWANCE_S,
            on_cleanup=confirmed_cleanup,
        )
        # Normal return also carries the run_owned_process cleanup contract.
        worker_cleaned = True
        finish_worker("success" if completed.returncode == 0 else "error")
        snapshot = ProgressStore.read_snapshot(str(store.path))
        if snapshot.get("done") and isinstance(snapshot.get("result"), dict):
            return snapshot["result"]
        if snapshot.get("error"):
            raise RuntimeError(snapshot["error"])
        diagnostic = safe_runtime_diagnostic(completed.stderr or completed.stdout)
        raise RuntimeError(f"安全运行环境进程未返回完整结果（退出码 {completed.returncode}）：{diagnostic[-1600:]}")
    except (Exception, KeyboardInterrupt) as exc:
        finish_worker("interrupted" if isinstance(exc, KeyboardInterrupt) else "error")
        message = safe_runtime_diagnostic(exc)
        if effective_key:
            message = message.replace(effective_key, "[REDACTED]")
        if metadata is None:
            state("error", message)
        else:
            store.emit({"type": "error", "error": message})
        result = {"state": "ERROR", "error": message,
                  "data": {"runtime_preparation": metadata or {"status": "failed", "reason": message}},
                  "audit_logs": [], "audit_stats": {}, "report_path": "", "session_id": ""}
        store.emit({"type": "done", "result": result})
        if isinstance(exc, KeyboardInterrupt):
            raise
        return result


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
                            title_resolution: Optional[dict] = None,
                            pdf_resolution: Optional[dict] = None,
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
                optimization_mode=optimization_mode, max_candidates=max_candidates, budget_seconds=budget_seconds,
                title_resolution=title_resolution, pdf_resolution=pdf_resolution)
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
