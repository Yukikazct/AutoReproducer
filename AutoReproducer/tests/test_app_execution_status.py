"""Distinguish completed baseline phases from live additional execution."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import frontend.history_manager as history
from frontend.backend_pipeline import ProgressStore


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    instance = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    instance.session_state["docker_probe"] = (True, None)
    instance.run()
    assert not instance.exception
    return instance


def state(store, phase, agent, status):
    store.emit({"type": "state", "phase_id": phase, "agent": agent,
                "state": "OPTIMIZE" if phase == "method_advice" else "EXECUTE_CODE", "status": status})


def step(store, identifier, index, status, *, phase="method_advice", count=3):
    store.emit({"type": "repository_step", "execution_id": identifier, "phase_id": phase,
                "step_id": f"arbitrary_step_{index}", "step_index": index, "step_count": count, "status": status})


def output(store, identifier, index, text, *, phase="method_advice", count=3):
    store.emit({"type": "repository_output", "execution_id": identifier, "phase_id": phase,
                "step_id": f"arbitrary_step_{index}", "step_index": index, "step_count": count,
                "stream": "stdout", "text": text})


def method_progress(path):
    store = ProgressStore(str(path))
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "execute_repository", "agent": "⚡ CodeExecutor", "title": "代码执行", "status": "waiting"},
        {"id": "verify_protocol", "agent": "🛡️ Verifier", "title": "核验本次协议与产物", "status": "waiting"},
        {"id": "method_advice", "agent": "🧪 Optimizer", "title": "智能建议与参数试验", "status": "waiting"},
        {"id": "generate_report", "agent": "📝 ReportGenerator", "title": "生成报告", "status": "waiting"},
    ]})
    # The invocation number includes environment probes. It is independent of
    # training epochs, candidate numbers and the owning pipeline phase.
    step(store, "environment", 1, "success", phase="prepare_environment", count=1)
    step(store, "baseline", 3, "success", phase="execute_repository")
    state(store, "execute_repository", "⚡ CodeExecutor", "success")
    state(store, "verify_protocol", "🛡️ Verifier", "success")
    state(store, "method_advice", "🧪 Optimizer", "running")
    return store


def load(app, path):
    app.session_state["progress_file"] = str(path)
    app.run()
    assert not app.exception
    return {row["id"]: row for row in app.session_state["pipeline_stages"]}


def card(app, identifier):
    grid = next(item.value for item in app.markdown if 'data-stage-id="' in item.value)
    return next(chunk for chunk in grid.split('<div class="agent-card ')
                if f'data-stage-id="{identifier}"' in chunk)


def active_caption(app, label):
    return any(f"正在执行：{label} · 进行中" in item.value for item in app.caption)


def test_baseline_completion_and_optimizer_execution_have_separate_cards(app, tmp_path):
    path = tmp_path / "method.jsonl"
    store = method_progress(path)
    step(store, "study", 2, "running")
    output(store, "study", 2, "500/500 logged while the process remains active\n")
    rows = load(app, path)
    assert rows["execute_repository"]["status"] == "success"
    assert rows["verify_protocol"]["status"] == "success"
    assert rows["method_advice"]["status"] == "running"
    baseline = card(app, "execute_repository")
    assert "基线代码执行" in baseline and "基线执行完成" in baseline
    assert "当前执行：" not in baseline
    verification = card(app, "verify_protocol")
    assert "基线协议与产物核验" in verification and "基线核验完成" in verification
    study = card(app, "method_advice")
    assert "running" in study and "优化试验：代码执行" in study and "代码执行中" in study
    assert "🧪 Optimizer · ⚡ CodeExecutor" in study
    label = "代码执行 · 第 3 轮 · 步骤 2/3"
    assert "当前执行：" + label in study and active_caption(app, label)
    assert any("代码仍在执行" in notice.value and "整体任务" in notice.value for notice in app.info)
    assert "阶段完成: 2/4" in app.get("progress")[0].proto.text
    assert app.get("progress")[0].proto.value == 50
    assert app.session_state["running"] is True


def test_delayed_other_step_completion_does_not_hide_active_optimizer_step(app, tmp_path):
    path = tmp_path / "delayed.jsonl"
    store = method_progress(path)
    step(store, "study", 1, "running")
    step(store, "study", 2, "running")
    output(store, "study", 2, "live second-step output\n")
    step(store, "study", 1, "success")
    snapshot = ProgressStore.read_snapshot(str(path))
    assert snapshot["execution_context"]["step_index"] == 1
    assert snapshot["execution_context"]["status"] == "success"
    assert [row["step_index"] for row in snapshot["active_executions"]] == [2]
    load(app, path)
    assert active_caption(app, "代码执行 · 第 3 轮 · 步骤 2/3")
    assert not any("最近执行：" in item.value for item in app.caption)
    assert "代码执行中" in card(app, "method_advice")
    assert "步骤 2/3" in card(app, "method_advice")
    assert "阶段完成: 2/4" in app.get("progress")[0].proto.text


def test_step_completion_waits_for_task_terminal_and_next_execution_reactivates_display(app, tmp_path):
    path = tmp_path / "continued.jsonl"
    store = method_progress(path)
    step(store, "study", 2, "running")
    output(store, "study", 2, "current trial\n")
    step(store, "study", 2, "success")
    rows = load(app, path)
    assert rows["method_advice"]["status"] == "running"
    assert app.session_state["running"] is True
    assert any("整体任务仍在运行" in notice.value for notice in app.info)
    assert not any("代码仍在执行" in notice.value for notice in app.info)
    assert not any("正在执行：" in item.value for item in app.caption)
    assert any("本步骤完成" in item.value for item in app.caption)
    assert "代码执行结束" not in " ".join(item.value for item in app.markdown)
    assert "代码执行中" not in card(app, "method_advice")
    assert "阶段完成: 2/4" in app.get("progress")[0].proto.text

    step(store, "next-study", 2, "running")
    output(store, "next-study", 2, "additional execution\n")
    load(app, path)
    assert active_caption(app, "代码执行 · 第 4 轮 · 步骤 2/3")
    assert "代码执行中" in card(app, "method_advice")
    assert any("代码仍在执行" in notice.value for notice in app.info)
    assert "基线执行完成" in card(app, "execute_repository")

    step(store, "next-study", 2, "success")
    state(store, "method_advice", "🧪 Optimizer", "success")
    state(store, "generate_report", "📝 ReportGenerator", "success")
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {}}})
    rows = load(app, path)
    assert all(row["status"] == "success" for row in rows.values())
    assert app.session_state["running"] is False
    assert ProgressStore.read_snapshot(str(path))["active_executions"] == []
    assert not any("代码仍在执行" in notice.value for notice in app.info)
    assert not any("正在执行：" in item.value for item in app.caption)
    assert "代码执行中" not in card(app, "method_advice")
    assert "阶段完成: 4/4" in app.get("progress")[0].proto.text


def test_generic_completed_code_phase_does_not_claim_active_later_phase_is_done(app, tmp_path):
    path = tmp_path / "generic.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "previous", "agent": "CodeExecutor", "title": "代码执行", "status": "waiting"},
        {"id": "additional", "agent": "CustomWorker", "title": "额外计算", "status": "waiting"},
    ]})
    state(store, "previous", "CodeExecutor", "success")
    state(store, "additional", "CustomWorker", "running")
    step(store, "additional-operation", 2, "running", phase="additional", count=7)
    output(store, "additional-operation", 2, "live calculation\n", phase="additional", count=7)
    rows = load(app, path)
    assert rows["previous"]["status"] == "success" and rows["additional"]["status"] == "running"
    prior = card(app, "previous")
    assert "代码执行（此前轮次）" in prior and "此前轮次完成" in prior
    assert "本阶段完成" not in prior
    assert "当前执行：" not in prior
    assert "步骤 2/7" in card(app, "additional")
    assert any("代码仍在执行" in notice.value for notice in app.info)
    assert "阶段完成: 1/2" in app.get("progress")[0].proto.text
    assert app.session_state["running"] is True


def test_early_phase_and_task_completion_remain_pending_until_invocation_cleanup(app, tmp_path):
    path = tmp_path / "early.jsonl"
    store = method_progress(path)
    store.emit({"type": "execution_run", "execution_id": "owned-study",
                "phase_id": "method_advice", "status": "running"})
    step(store, "owned-study", 1, "success")
    state(store, "method_advice", "🧪 Optimizer", "success")
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {}}})
    rows = load(app, path)
    assert rows["method_advice"]["status"] == "running"
    assert rows["method_advice"]["completion_pending"]
    assert "结束确认中" in card(app, "method_advice")
    assert app.session_state["running"] is True
    assert app.session_state["result"] is None
    assert any("尚未确认任务结束" in notice.value for notice in app.warning)
    assert any("执行轮次尚未结束" in notice.value for notice in app.info)
    assert not any("任务完成" in notice.value for notice in app.success)
    store.emit({"type": "execution_run", "execution_id": "owned-study",
                "phase_id": "method_advice", "status": "success", "cleanup_confirmed": True})
    rows = load(app, path)
    assert rows["method_advice"]["completion_confirmed"]
    assert app.session_state["running"] is False


def test_child_done_is_not_rendered_complete_before_worker_reaping(app, tmp_path):
    path = tmp_path / "worker.jsonl"
    store = ProgressStore(str(path))
    state(store, "generate_report", "ReportGenerator", "success")
    store.emit({"type": "pipeline_worker", "worker_id": "owned", "status": "running"})
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {}}})
    load(app, path)
    assert app.session_state["running"] and app.session_state["result"] is None
    assert any("等待后台进程返回并完成清理" in notice.value for notice in app.info)
    assert any("尚未确认任务结束" in notice.value for notice in app.warning)
    store.emit({"type": "pipeline_worker", "worker_id": "owned", "status": "success", "cleanup_confirmed": True})
    load(app, path)
    assert app.session_state["running"] is False


@pytest.mark.parametrize(("status", "text"), [
    ("insufficient_evidence", "论文证据不足"),
    ("best_effort", "不能判定论文复现"),
    ("inconclusive", "复现结论无法验收"),
    ("no_reference_metrics", "复现结论无法验收"),
    ("not_runnable", "论文实验未完成"),
    ("execution_incomplete", "论文实验未完成"),
    ("execution_failed", "论文实验未完成"),
])
def test_completed_generated_work_does_not_display_paper_reproduction_success(app, tmp_path, status, text):
    path = tmp_path / "inconclusive.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {
        "validation": {"status": status, "is_reproduced": False, "reason": "Missing evidence"}}}})
    load(app, path)
    assert any(text in notice.value for notice in app.warning)
    assert not any("论文数值验收通过" in notice.value for notice in app.success)
