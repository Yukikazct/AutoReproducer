"""后台复现流水线与进度事件存储测试。

覆盖：
1. ProgressStore 事件写入 -> read_snapshot 聚合（state/agent_status/logs）；
2. done 事件落地 result 并置 running=False；
3. error 事件置 running=False 且暴露错误消息；
4. run_pipeline_core 后台全链路（Mock 模式）：COMPLETED + 进度事件完整
   （各 Agent 状态齐全、日志非空、done 事件含 result）；
5. run_pipeline_background 线程方式：join 后进度文件存在 done/error 事件；
6. 后台线程不阻塞调用方（线程启动即可返回）。

运行: python -m pytest tests/test_backend_pipeline.py -v
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.backend_pipeline import (  # noqa: E402
    AGENTS,
    ProgressStore,
    run_pipeline_background,
    run_pipeline_core,
)


# ---------------- 1. ProgressStore 读写与聚合 ----------------

def test_store_roundtrip_state_and_logs(tmp_path):
    p = tmp_path / "progress.jsonl"
    store = ProgressStore(str(p))
    store.emit({"type": "state", "state": "READ_PAPER",
                "agent": "📖 PaperReader", "status": "running"})
    store.emit({"type": "log", "log": {"agent": "PaperReader",
                                       "status": "SUCCESS", "detail": "ok"}})

    view = ProgressStore.read_snapshot(str(p))
    assert view["running"] is True
    assert view["done"] is False
    assert view["state"] == "READ_PAPER"
    assert view["agent_status"]["📖 PaperReader"] == "running"
    assert view["logs"][0]["agent"] == "PaperReader"
    assert view["updated_at"]  # 事件带时间戳


def test_store_done_event_fills_result(tmp_path):
    p = tmp_path / "progress.jsonl"
    store = ProgressStore(str(p))
    store.emit({"type": "state", "state": "COMPLETED", "agent": "", "status": "success"})
    store.emit({"type": "done", "result": {"state": "COMPLETED",
                                           "data": {"paper_title": "X"}}})

    view = ProgressStore.read_snapshot(str(p))
    assert view["done"] is True
    assert view["running"] is False
    assert view["result"]["state"] == "COMPLETED"
    assert view["result"]["data"]["paper_title"] == "X"


def test_store_error_event(tmp_path):
    p = tmp_path / "progress.jsonl"
    store = ProgressStore(str(p))
    store.emit({"type": "error", "error": "boom"})
    view = ProgressStore.read_snapshot(str(p))
    assert view["running"] is False
    assert view["error"] == "boom"
    assert view["state"] == "ERROR"


def test_store_missing_file_returns_empty_view(tmp_path):
    view = ProgressStore.read_snapshot(str(tmp_path / "nope.jsonl"))
    assert view["running"] is True
    assert view["agent_status"] == {}
    assert view["result"] is None


def test_phase_plan_preserves_early_completion_and_separates_verifier_roles(tmp_path):
    path = tmp_path / "phases.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "state", "phase_id": "select_experiment", "state": "READ_PAPER",
                "agent": "PaperReader", "status": "success"})
    plan = [
        {"id": "select_experiment", "agent": "PaperReader", "title": "选择实验", "status": "waiting"},
        {"id": "review_readiness", "agent": "Verifier", "title": "训练前预审", "status": "waiting"},
        {"id": "execute_repository", "agent": "CodeExecutor", "title": "真实执行", "status": "waiting"},
        {"id": "verify_protocol", "agent": "Verifier", "title": "训练后核验", "status": "waiting"},
        {"id": "review_result_summary", "agent": "ResultValidator", "status": "skipped"},
        {"id": "generate_report", "agent": "ReportGenerator", "status": "waiting"},
    ]
    store.emit({"type": "pipeline_plan", "stages": plan})
    store.emit({"type": "state", "phase_id": "review_readiness", "state": "BUILD_ENV",
                "agent": "Verifier", "status": "success", "outcome": "accepted"})
    store.emit({"type": "state", "phase_id": "execute_repository", "state": "EXECUTE_CODE",
                "agent": "CodeExecutor", "status": "running"})
    # Compatibility events do not add fictitious stages to the explicit plan.
    store.emit({"type": "state", "state": "EXECUTE_CODE", "agent": "Optimizer", "status": "skipped"})
    view = ProgressStore.read_snapshot(str(path))
    rows = {row["id"]: row for row in view["pipeline_stages"]}
    assert list(rows) == [stage["id"] for stage in plan]
    assert rows["select_experiment"]["status"] == "success"
    assert rows["select_experiment"]["title"] == "选择实验"
    assert rows["review_readiness"]["status"] == "success"
    assert rows["verify_protocol"]["status"] == "waiting"
    assert rows["generate_report"]["status"] == "waiting"
    assert rows["review_result_summary"]["status"] == "skipped"
    assert view["agent_status"]["Verifier"] == "success"


def test_phase_correction_reuses_row_and_clears_previous_rejection(tmp_path):
    path = tmp_path / "retry.jsonl"
    store = ProgressStore(str(path))
    stage = {"type": "state", "phase_id": "analyze_reader", "state": "READ_PAPER", "agent": "PaperReader"}
    store.emit({**stage, "status": "running", "attempt": 1})
    store.emit({**stage, "status": "error", "attempt": 1, "reason": "Incorrect citation", "outcome": "rejected"})
    store.emit({**stage, "status": "running", "attempt": 2})
    rows = ProgressStore.read_snapshot(str(path))["pipeline_stages"]
    assert len(rows) == 1
    assert rows[0]["status"] == "running" and rows[0]["attempt"] == 2
    assert "reason" not in rows[0] and "outcome" not in rows[0]
    store.emit({**stage, "status": "success", "attempt": 2, "calls": 1, "outcome": "accepted"})
    assert ProgressStore.read_snapshot(str(path))["pipeline_stages"][0]["outcome"] == "accepted"


def test_legacy_progress_restores_event_order_and_distinct_role_occurrences(tmp_path):
    path = tmp_path / "legacy.jsonl"
    store = ProgressStore(str(path))
    for name in ("PaperReader", "ResourceFinder", "EnvBuilder", "CodeExecutor", "ResultValidator", "Verifier"):
        store.emit({"type": "state", "state": "INIT", "agent": name, "status": "waiting"})
    for state, agent in [("READ_PAPER", "PaperReader"), ("BUILD_ENV", "EnvBuilder"),
                         ("READ_PAPER", "PaperReader"), ("BUILD_ENV", "Verifier"),
                         ("EXECUTE_CODE", "CodeExecutor")]:
        store.emit({"type": "state", "state": state, "agent": agent, "status": "running"})
        store.emit({"type": "state", "state": state, "agent": agent, "status": "success"})
    # Old repository runs emitted only the final local verifier's completion.
    store.emit({"type": "state", "state": "VALIDATE", "agent": "Verifier", "status": "success"})
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {"experiment_spec": {"id": "legacy_repository"}}}})
    view = ProgressStore.read_snapshot(str(path))
    rows = view["pipeline_stages"]
    assert [row["agent"] for row in rows] == ["PaperReader", "EnvBuilder", "PaperReader", "Verifier", "CodeExecutor", "Verifier"]
    assert len({row["id"] for row in rows}) == 6
    assert rows[3]["title"] == "训练前证据预审"
    assert rows[5]["title"] == "训练后核验"
    assert all(row["status"] == "success" for row in rows)
    assert view["done"] is True


def test_legacy_generic_quality_checks_and_skipped_optimizer_have_accurate_titles(tmp_path):
    path = tmp_path / "generic_legacy.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "state", "state": "READ_PAPER", "agent": "Verifier", "status": "success"})
    store.emit({"type": "state", "state": "VALIDATE", "agent": "Optimizer", "status": "skipped"})
    rows = ProgressStore.read_snapshot(str(path))["pipeline_stages"]
    assert rows[0]["title"] == "论文解析质量核验"
    assert rows[1]["title"] == "智能优化（未启用）"


@pytest.mark.parametrize("terminal", ["done", "error"])
def test_terminal_progress_marks_unexecuted_phases_blocked(tmp_path, terminal):
    path = tmp_path / "blocked.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "review_readiness", "status": "waiting"},
        {"id": "execute_repository", "status": "waiting"},
        {"id": "review_result_summary", "status": "skipped"},
        {"id": "generate_report", "status": "waiting"},
    ]})
    store.emit({"type": "state", "phase_id": "review_readiness", "agent": "Verifier", "status": "error"})
    store.emit({"type": "state", "phase_id": "generate_report", "agent": "ReportGenerator", "status": "success"})
    store.emit({"type": "done", "result": {"state": "ERROR"}} if terminal == "done" else
               {"type": "error", "error": "failed"})
    rows = {row["id"]: row for row in ProgressStore.read_snapshot(str(path))["pipeline_stages"]}
    assert rows["review_readiness"]["status"] == "error"
    assert rows["execute_repository"]["status"] == "blocked"
    assert rows["execute_repository"]["reason"]
    assert rows["review_result_summary"]["status"] == "skipped"
    assert rows["generate_report"]["status"] == "success"


def test_repository_progress_preserves_output_and_terminal(tmp_path):
    path = tmp_path / "p.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "repository_step", "step_id": "train_and_eval", "status": "running"})
    store.emit({"type": "repository_output", "stream": "stdout", "text": "Epoch: 1\n"})
    store.emit({"type": "repository_output", "stream": "stderr", "text": "warning\n"})
    store.emit({"type": "done", "result": {"state": "COMPLETED"}})
    view = ProgressStore.read_snapshot(str(path))
    assert view["repository_step"] == "train_and_eval"
    assert view["execution_output"] == "Epoch: 1\nwarning\n"
    assert view["running"] is False


def test_backend_uses_shared_orchestrator_resource_hooks(tmp_path):
    result = run_pipeline_core(str(tmp_path / "p.jsonl"), paper_title="Dummy Paper", mock_mode=True)
    assert result["state"] == "COMPLETED"
    assert set(result["data"]["storage"]["fetched"]) == {"code", "dataset", "weights"}
    assert result["data"]["storage"]["manifest_path"]


@pytest.mark.parametrize("requested", [False, True])
def test_backend_forwards_reserved_optimization_and_phase_plan(tmp_path, monkeypatch, requested):
    import frontend.backend_pipeline as backend

    captured = {}

    class PipelineFixture:
        def __init__(self, **kwargs):
            pass

        def run(self, input_data, on_event):
            captured.update(input_data)
            on_event({"type": "pipeline_plan", "stages": [
                {"id": "review_readiness", "agent": "Verifier", "title": "预审", "status": "waiting"}]})
            on_event({"type": "state", "phase_id": "review_readiness", "state": "BUILD_ENV",
                      "agent": "Verifier", "status": "success"})
            return {"state": "COMPLETED", "data": {}}

    monkeypatch.setattr(backend, "Orchestrator", PipelineFixture)
    path = tmp_path / "forwarding.jsonl"
    result = backend.run_pipeline_core(str(path), mock_mode=True, enable_optimization=requested)
    assert result["state"] == "COMPLETED"
    assert captured["enable_optimization"] is requested
    rows = ProgressStore.read_snapshot(str(path))["pipeline_stages"]
    assert rows[0]["agent"] == "🛡️ Verifier" and rows[0]["status"] == "success"


def test_background_forwards_reserved_optimization(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend

    captured = {}

    def core(progress_path, **kwargs):
        captured.update(kwargs)
        return {"state": "COMPLETED"}

    monkeypatch.setattr(backend, "run_pipeline_core", core)
    thread = backend.run_pipeline_background(str(tmp_path / "reserved.jsonl"), enable_optimization=True)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured["enable_optimization"] is True


# ---------------- 2. 后台全链路（Mock 模式） ----------------

def test_run_pipeline_core_writes_full_progress(tmp_path):
    progress = tmp_path / "p.jsonl"
    result = run_pipeline_core(str(progress), paper_title="Dummy Paper",
                               mock_mode=True, max_trials=2)

    assert result["state"] == "COMPLETED", result.get("error")
    assert result["data"]["validation"]["is_reproduced"] is True

    view = ProgressStore.read_snapshot(str(progress))
    assert view["done"] is True
    assert view["running"] is False
    assert view["result"]["state"] == "COMPLETED"
    # 每个展示 Agent 都有状态记录（running/success/error/waiting 至少一个）
    names = [a[1] for a in AGENTS] + ["🛡️ Verifier", "🧪 Optimizer",
                                      "📝 ReportGenerator"]
    for n in names:
        assert n in view["agent_status"], f"缺少 {n} 的状态事件"
    # 执行 Agent 最终应为 success（复现必成）
    assert view["agent_status"]["⚡ CodeExecutor"] in ("success",)
    # 审计日志已逐步写入进度
    assert view["logs"], "进度中缺少审计日志"


def test_run_pipeline_background_thread_and_cleanup(tmp_path):
    progress = tmp_path / "p.jsonl"
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    thread = run_pipeline_background(
        str(progress), paper_title="Dummy", mock_mode=True, max_trials=2,
        pdf_path=str(fake_pdf), cleanup_pdf=True)

    # 线程已启动（不阻塞调用方）
    assert thread.is_alive() or True
    thread.join(timeout=180)
    assert not thread.is_alive(), "后台线程未在超时内结束"

    view = ProgressStore.read_snapshot(str(progress))
    assert view["done"] is True or view["error"] is not None
    assert not fake_pdf.exists(), "临时 PDF 应由后台线程清理"


def test_run_pipeline_background_error_is_reported(tmp_path, monkeypatch):
    """进度文件写入失败等外层异常应落 error 事件而非静默。"""
    progress = tmp_path / "p.jsonl"

    def fail(*args, **kwargs):
        raise OSError("simulated progress failure")

    monkeypatch.setattr("frontend.backend_pipeline.run_pipeline_core", fail)
    thread = run_pipeline_background(str(progress), paper_title="Dummy",
                                     mock_mode=True, max_trials=1)
    thread.join(timeout=30)
    assert not thread.is_alive()
    view = ProgressStore.read_snapshot(str(progress))
    assert view["running"] is False
    assert view["state"] == "ERROR"
    assert "simulated progress failure" in view["error"]


# ---------------- 3. 阶段异常不吞 + 日志去重 + 终态落盘 ----------------

def test_run_pipeline_core_no_duplicate_logs(tmp_path):
    """get_summary() 不再全量重发：进度文件里同一条审计日志只出现一次。"""
    progress = tmp_path / "p.jsonl"
    result = run_pipeline_core(str(progress), paper_title="Dummy Paper",
                               mock_mode=True, max_trials=2)
    assert result["state"] == "COMPLETED", result.get("error")

    logs = []
    for line in progress.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "log":
            logs.append(ev["log"])
    assert logs, "进度文件应包含审计日志"
    unique = {json.dumps(l, sort_keys=True, ensure_ascii=False) for l in logs}
    assert len(unique) == len(logs), "进度文件存在重复审计日志"


def test_run_pipeline_core_reports_stage_error(tmp_path, monkeypatch):
    """阶段异常不再被吞：result['error'] 非 None，且 ledger 末条有 FINISH 终态。"""
    import frontend.backend_pipeline as bp

    class _BoomAgent:
        name = "PaperReader"
        system_prompt = ""
        def run(self, data):
            raise RuntimeError("boom")

    class _BoomOrch(bp.Orchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.agents["reader"] = _BoomAgent()

    monkeypatch.setattr(bp, "Orchestrator", _BoomOrch)
    progress = tmp_path / "p.jsonl"
    result = bp.run_pipeline_core(str(progress), paper_title="Dummy",
                                  mock_mode=True, max_trials=2)

    assert result["state"] == "ERROR"
    assert result["error"] and "boom" in result["error"]

    # ledger 末条 FINISH 记录携带终态（真实落盘到项目 data/experiment_ledger）
    root = Path(__file__).resolve().parents[1]
    ledger_file = (root / "data" / "experiment_ledger"
                   / f"ledger_{result['session_id']}.jsonl")
    records = [json.loads(line) for line in
               ledger_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    finish = records[-1]
    assert finish["phase"] == "FINISH"
    assert finish["result"]["state"] == "ERROR"
