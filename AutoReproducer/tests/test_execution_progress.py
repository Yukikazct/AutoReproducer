"""Generic execution progress follows producer identities, not experiment names."""
import pytest

from frontend.backend_pipeline import ProgressStore


@pytest.fixture
def progress(tmp_path):
    path = tmp_path / "progress.jsonl"
    return ProgressStore(str(path)), lambda: ProgressStore.read_snapshot(str(path))


def step(store, identifier="opaque-a", index=1, status="running", prefix="repository", **extra):
    store.emit({"type": prefix + "_step", "execution_id": identifier,
                "step_id": "arbitrary_action", "step_index": index, "status": status, **extra})


def output(store, text, identifier="opaque-a", index=1, prefix="repository", **extra):
    store.emit({"type": prefix + "_output", "execution_id": identifier,
                "step_id": "arbitrary_action", "step_index": index, "text": text, "stream": "stdout", **extra})


@pytest.mark.parametrize("prefix", ["repository", "execution"])
@pytest.mark.parametrize("index,count,label", [(1, 7, "步骤 1/7"), (4, 7, "步骤 4/7"),
                                               (2, None, "步骤 2"), (9, None, "步骤 9")])
def test_any_producer_and_step_count_have_generic_labels(progress, prefix, index, count, label):
    store, snapshot = progress
    step(store, index=index, prefix=prefix, step_count=count, phase_id="arbitrary_phase",
         cwd="D:/runs/repository_example/trials/candidate_3_2021/repo", argv=["evaluate.py", "holdout"])
    view = snapshot()
    context = view["execution_context"]
    assert context["label"] == "代码执行 · 第 1 轮 · " + label
    assert context["step_index"] == index and context["step_count"] == count
    assert context["phase_id"] == "arbitrary_phase"
    assert context["execution_id"] == "opaque-a" and context["round_number"] == 1
    assert context["status"] == "running" and view["active_executions"] == [context]
    assert view["repository_step_label"] == context["label"]
    assert view["repository_step"] == "arbitrary_action"
    assert not any(key in context for key in ("seed", "candidate_index", "cwd", "split", "trial_label"))


def test_rounds_follow_first_id_appearance_including_output(progress):
    store, snapshot = progress
    output(store, "", identifier="z-first")
    step(store, "a-second")
    assert snapshot()["execution_context"]["round_number"] == 2
    step(store, "z-first", index=3)
    assert snapshot()["execution_context"]["round_number"] == 1
    assert [row["execution_id"] for row in snapshot()["active_executions"]] == ["a-second", "z-first"]


def test_delayed_output_uses_its_own_identity_without_changing_current_step(progress):
    store, snapshot = progress
    step(store, "first", step_count=3)
    output(store, "first early", "first")
    step(store, "first", status="success")
    step(store, "second", step_count=5, phase_id="later_phase")
    output(store, "second early", "second")
    output(store, "first delayed", "first", stream="stderr")
    output(store, "second resumed", "second")
    view = snapshot()
    assert view["execution_context"]["execution_id"] == "second"
    assert view["execution_context"]["phase_id"] == "later_phase"
    assert [row["execution_id"] for row in view["active_executions"]] == ["second"]
    text = view["execution_output"]
    first = "── 代码执行 · 第 1 轮 · 步骤 1/3 ──"
    second = "── 代码执行 · 第 2 轮 · 步骤 1/5 ──"
    assert text.count(first) == 2 and text.count(second) == 2
    assert text.index("second early") < text.rindex(first) < text.index("first delayed")
    assert text.index("first delayed") < text.rindex(second) < text.index("second resumed")


def test_parallel_steps_and_out_of_order_completion_keep_other_steps_active(progress):
    store, snapshot = progress
    step(store, "same", index=1)
    step(store, "same", index=2)
    step(store, "other", index=1)
    step(store, "same", index=1, status="success")
    view = snapshot()
    assert view["execution_context"]["status"] == "success"
    assert [(row["execution_id"], row["step_index"]) for row in view["active_executions"]] == [("same", 2), ("other", 1)]
    output(store, "late step one", "same", index=1)
    assert snapshot()["active_executions"] == view["active_executions"]


def test_running_event_marks_boundary_before_output_and_terminal_does_not_duplicate(progress):
    store, snapshot = progress
    step(store, "first")
    output(store, "x" * 17000, "first")
    step(store, "first", status="success")
    step(store, "second", index=2, step_count=4)
    before = snapshot()["execution_output"]
    header = "── 代码执行 · 第 2 轮 · 步骤 2/4 ──\n"
    assert len(before) == 16000 and before.endswith(header)
    output(store, "hello", "second", index=2)
    step(store, "second", index=2, status="success")
    after = snapshot()
    assert after["execution_output"].count(header) == 1
    assert after["active_executions"] == []


def test_old_steps_remain_generic_and_pure_output_stays_unchanged(progress):
    store, snapshot = progress
    store.emit({"type": "repository_step", "step_id": "train_and_eval", "status": "running",
                "cwd": "D:/runs/repository_example/trials/candidate_3_2021/repo",
                "argv": ["evaluate.py", "holdout"]})
    store.emit({"type": "repository_output", "text": "Epoch 1\n"})
    store.emit({"type": "repository_output", "stream": "stderr", "text": "warning\n"})
    view = snapshot()
    assert view["execution_output"] == "Epoch 1\nwarning\n"
    assert view["execution_context"]["label"] == "代码执行 · train_and_eval"
    assert view["execution_context"]["round_number"] is None
    step(store, "new")
    assert snapshot()["execution_context"]["round_number"] == 1


def test_unidentified_output_never_borrows_current_identified_round(progress):
    store, snapshot = progress
    step(store, "new")
    output(store, "identified", "new")
    store.emit({"type": "repository_output", "step_id": "legacy_action", "text": "unknown owner"})
    view = snapshot()
    assert view["execution_context"]["execution_id"] == "new"
    assert view["execution_output"].endswith("── 代码执行 · legacy_action ──\nunknown owner")


@pytest.mark.parametrize("count", [0, -1, True, "3", 1])
def test_invalid_or_inconsistent_count_is_not_invented(progress, count):
    store, snapshot = progress
    step(store, index=2, step_count=count)
    assert snapshot()["execution_context"]["label"] == "代码执行 · 第 1 轮 · 步骤 2"


@pytest.mark.parametrize("terminal", ["done", "error"])
def test_pipeline_terminal_closes_all_unfinished_steps(progress, terminal):
    store, snapshot = progress
    step(store, "first")
    step(store, "second", status="success")
    step(store, "third")
    store.emit({"type": "done", "result": {"state": "ERROR"}} if terminal == "done" else
               {"type": "error", "error": "stopped"})
    view = snapshot()
    assert view["execution_context"]["status"] == "interrupted"
    assert view["active_executions"] == []
    assert view["running"] is False


def test_successful_last_step_stays_success_when_pipeline_finishes(progress):
    store, snapshot = progress
    step(store)
    step(store, status="success")
    store.emit({"type": "done", "result": {"state": "COMPLETED"}})
    assert snapshot()["execution_context"]["status"] == "success"
    assert snapshot()["active_executions"] == []


@pytest.mark.parametrize("terminal", ["done", "error"])
def test_late_worker_events_cannot_replace_terminal_context_or_reactivate_steps(progress, terminal):
    store, snapshot = progress
    step(store, "completed", status="success")
    store.emit({"type": "done", "result": {"state": "COMPLETED"}} if terminal == "done" else
               {"type": "error", "error": "pipeline stopped"})
    original = snapshot()["execution_context"]
    step(store, "completed", status="running")
    step(store, "late-worker", index=3)
    output(store, "late output", "late-worker", index=3)
    view = snapshot()
    assert view["execution_context"] == original
    assert view["execution_context"]["status"] == "success"
    assert view["active_executions"] == [] and view["running"] is False
    assert view["execution_output"].endswith("── 代码执行 · 第 2 轮 · 步骤 3 ──\nlate output")


def test_backend_forwards_generic_execution_events(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend

    class PipelineFixture:
        def __init__(self, **kwargs):
            pass

        def run(self, input_data, on_event):
            base = {"execution_id": "generic-opaque", "step_id": "any_script", "step_index": 1,
                    "step_count": 2, "phase_id": "custom_phase"}
            on_event({"type": "execution_step", **base, "status": "running"})
            on_event({"type": "execution_output", **base, "text": "streamed output", "stream": "stdout"})
            on_event({"type": "execution_step", **base, "status": "success"})
            return {"state": "COMPLETED", "data": {}}

    monkeypatch.setattr(backend, "Orchestrator", PipelineFixture)
    path = tmp_path / "forwarded.jsonl"
    result = backend.run_pipeline_core(str(path), mock_mode=True)
    view = ProgressStore.read_snapshot(str(path))
    assert result["state"] == "COMPLETED"
    assert view["execution_context"]["phase_id"] == "custom_phase"
    assert view["execution_context"]["label"] == "代码执行 · 第 1 轮 · 步骤 1/2"
    assert view["execution_context"]["status"] == "success"
    assert "streamed output" in view["execution_output"]
