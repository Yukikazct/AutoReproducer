"""Progress completion requires owner results and confirmed scope cleanup."""
import json

import pytest

from frontend.progress_state import read_progress_snapshot


PHASE = "method_advice"


def planned(status="waiting", phase=PHASE):
    return {"type": "pipeline_plan", "stages": [
        {"id": phase, "agent": "Worker", "title": "Reviewed work", "status": status}]}


def owner(status, phase=PHASE, **extra):
    return {"type": "state", "phase_id": phase, "agent": "Worker",
            "state": "EXECUTE_CODE", "status": status, **extra}


def run(status, identifier="run-a", phase=PHASE, **extra):
    return {"type": "execution_run", "execution_id": identifier,
            "phase_id": phase, "status": status, **extra}


def step(status, index=1, identifier="run-a", phase=PHASE, **extra):
    return {"type": "execution_step", "execution_id": identifier,
            "phase_id": phase, "step_id": f"step_{index}", "step_index": index,
            "status": status, **extra}


def worker(status, identifier="parent", **extra):
    return {"type": "pipeline_worker", "worker_id": identifier, "status": status, **extra}


def terminal(kind="done", state="COMPLETED"):
    return ({"type": "done", "result": {"state": state, "data": {}}}
            if kind == "done" else {"type": "error", "error": "pipeline failed"})


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "completion.jsonl"

    def read(events):
        path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
        return read_progress_snapshot(str(path))

    return read


def row(view):
    return next(stage for stage in view["pipeline_stages"] if stage["id"] == PHASE)


@pytest.mark.parametrize("index,count", [(1, 1), (7, 7), (99, None)])
def test_last_step_and_logged_counts_cannot_close_invocation(snapshot, index, count):
    events = [planned(), owner("running"), run("running"),
              step("running", index=index, step_count=count),
              {"type": "execution_output", "execution_id": "run-a", "step_index": index,
               "text": "500/500; Training completed successfully"},
              step("success", index=index, step_count=count)]
    view = snapshot(events)
    assert view["active_executions"] == []
    assert [scope["execution_id"] for scope in view["active_runs"]] == ["run-a"]
    assert view["running"] and not view["done"]
    assert row(view)["status"] == "running" and not row(view)["completion_confirmed"]
    view = snapshot(events + [step("running", index=index + 1)])
    assert [context["step_index"] for context in view["active_executions"]] == [index + 1]
    assert len(view["active_runs"]) == 1


@pytest.mark.parametrize("kind", ["done", "error"])
def test_early_owner_and_pipeline_terminal_wait_for_scope_cleanup(snapshot, kind):
    events = [planned(), owner("running"), run("running"),
              step("running"), step("success"), owner("success"), terminal(kind)]
    view = snapshot(events)
    assert row(view)["status"] == "running" and row(view)["reported_status"] == "success"
    assert row(view)["completion_pending"] and not row(view)["completion_confirmed"]
    assert view["running"] and not view["done"] and view["result"] is None
    assert view["finalization_pending"] and "execution:run-a" in view["completion_blockers"]
    view = snapshot(events + [run("success")])
    assert not view["running"] and not view["finalization_pending"]
    assert view["active_runs"] == [] and view["completion_blockers"] == []
    assert row(view)["completion_confirmed"]
    assert view["done"] is (kind == "done")
    assert view["state"] == ("COMPLETED" if kind == "done" else "ERROR")


def test_two_invocations_require_both_cleanup_confirmations(snapshot):
    events = [planned(), run("running", "first"), run("running", "second"),
              owner("success"), terminal(), run("success", "first")]
    view = snapshot(events)
    assert view["running"] and view["finalization_pending"] and not view["done"]
    assert [scope["execution_id"] for scope in view["active_runs"]] == ["second"]
    assert view["completion_blockers"] == ["execution:second"]
    assert row(view)["completion_pending"]
    view = snapshot(events + [run("success", "second")])
    assert view["done"] and not view["running"]
    assert row(view)["status"] == "success" and row(view)["completion_confirmed"]


@pytest.mark.parametrize("scope", ["execution", "worker"])
@pytest.mark.parametrize("status", ["success", "error", "interrupted"])
def test_terminal_without_cleanup_confirmation_keeps_scope_open(snapshot, scope, status):
    emit = run if scope == "execution" else worker
    events = [planned(), emit("running"), owner("success"), terminal(),
              emit(status, cleanup_confirmed=False)]
    view = snapshot(events)
    active = view["active_runs"] if scope == "execution" else view["active_workers"]
    assert len(active) == 1 and active[0]["status"] == "running"
    assert view["running"] and view["finalization_pending"] and not view["done"]
    expected = "execution:run-a" if scope == "execution" else "worker:parent"
    assert expected in view["completion_blockers"]
    view = snapshot(events + [emit(status, cleanup_confirmed=True)])
    assert view["done"] and not view["running"] and not view["finalization_pending"]
    assert view["state"] == ("COMPLETED" if status == "success" else "ERROR")


@pytest.mark.parametrize("status", ["success", "error", "interrupted"])
def test_plan_terminal_status_is_not_an_owner_result(snapshot, status):
    view = snapshot([planned(status)])
    assert row(view)["status"] == "waiting"
    assert not row(view)["completion_confirmed"]
    assert view["running"]
    view = snapshot([planned(status), owner(status)])
    assert row(view)["status"] == status and row(view)["completion_confirmed"]


def test_explicit_plan_skip_needs_no_process_or_owner_result(snapshot):
    view = snapshot([planned("skipped")])
    assert row(view)["status"] == "skipped" and row(view)["completion_confirmed"]


def test_run_end_without_owner_result_cannot_confirm_phase(snapshot):
    view = snapshot([planned(), owner("running"), run("running"), run("success")])
    assert view["active_runs"] == []
    assert row(view)["status"] == "running" and not row(view)["completion_confirmed"]


def test_closed_run_delayed_running_step_cannot_hold_done_forever(snapshot):
    events = [planned(), run("running"), step("running"), step("success"),
              run("success"), owner("success"), step("running", index=8), terminal()]
    view = snapshot(events)
    assert view["done"] and not view["running"] and not view["finalization_pending"]
    assert view["active_runs"] == [] and view["active_executions"] == []
    assert view["execution_context"]["step_index"] == 1
    assert view["execution_context"]["status"] == "success"


def test_delayed_closed_starts_cannot_hide_current_live_execution(snapshot):
    events = [planned(), run("running", "closed"), step("success", identifier="closed"),
              run("success", "closed"), run("running", "live"),
              step("running", identifier="live", index=2),
              run("running", "closed"), step("running", identifier="closed", index=9)]
    view = snapshot(events)
    assert [scope["execution_id"] for scope in view["active_runs"]] == ["live"]
    assert [(context["execution_id"], context["step_index"]) for context in view["active_executions"]] == [("live", 2)]
    assert view["execution_context"]["execution_id"] == "live"


def test_delayed_step_start_cannot_reopen_finished_step_inside_open_run(snapshot):
    events = [run("running"), step("running"), step("success"),
              step("running", index=2), step("running")]
    view = snapshot(events)
    assert [context["step_index"] for context in view["active_executions"]] == [2]
    assert view["execution_context"]["step_index"] == 2


@pytest.mark.parametrize("late_failure", [run("error"), run("interrupted"), step("error")])
def test_later_execution_failure_rejects_early_owner_success(snapshot, late_failure):
    events = [planned(), run("running"), step("running"), step("success"),
              owner("success"), late_failure, run("success"), terminal()]
    view = snapshot(events)
    assert row(view)["status"] == "error" and not row(view)["completion_confirmed"]
    assert view["done"] and not view["running"] and view["state"] == "ERROR"
    assert view["result"]["state"] == "ERROR"


def test_later_failed_scope_verdict_can_correct_closed_success(snapshot):
    view = snapshot([planned(), run("running"), step("success"), run("success"),
                     owner("success"), run("error"), terminal()])
    assert row(view)["status"] == "error" and view["state"] == "ERROR"


def test_duplicate_owner_success_cannot_hide_failure_after_original_result(snapshot):
    view = snapshot([planned(), run("running"), owner("success"), run("error"),
                     owner("success"), terminal()])
    assert row(view)["status"] == "error" and not row(view)["completion_confirmed"]
    assert view["state"] == "ERROR"


def test_a_new_attempt_can_recover_prior_failure_but_stale_events_cannot_reopen_it(snapshot):
    events = [planned(), owner("error", attempt=1, reason="first failure"),
              owner("running", attempt=2), owner("success", attempt=1)]
    view = snapshot(events)
    assert row(view)["status"] == "running" and row(view)["attempt"] == 2
    assert "reason" not in row(view)
    view = snapshot(events + [owner("success", attempt=2), owner("running", attempt=1)])
    assert row(view)["status"] == "success" and row(view)["completion_confirmed"]
    assert view["agent_status"]["Worker"] == "success"


def test_delayed_same_attempt_phase_start_cannot_erase_owner_terminal(snapshot):
    view = snapshot([planned(), owner("success", attempt=1), owner("running", attempt=1)])
    assert row(view)["status"] == "success" and row(view)["completion_confirmed"]
    assert view["agent_status"]["Worker"] == "success"


def test_worker_scope_is_distinct_from_code_rounds_and_blocks_pipeline_finalization(snapshot):
    events = [planned(), worker("running"), run("running"), step("success"),
              run("success"), owner("success"), terminal()]
    view = snapshot(events)
    assert view["execution_context"]["round_number"] == 1
    assert view["active_runs"] == [] and view["active_executions"] == []
    assert [item["worker_id"] for item in view["active_workers"]] == ["parent"]
    assert view["completion_blockers"] == ["worker:parent"]
    assert view["running"] and not view["done"] and view["finalization_pending"]
    view = snapshot(events + [worker("success", cleanup_confirmed=True)])
    assert view["done"] and not view["running"] and view["active_workers"] == []


@pytest.mark.parametrize("status", ["error", "interrupted"])
def test_failed_worker_rejects_early_completed_pipeline_result(snapshot, status):
    view = snapshot([worker("running"), terminal(), worker(status, cleanup_confirmed=True)])
    assert view["done"] and not view["running"] and view["state"] == "ERROR"
    assert view["result"]["state"] == "ERROR" and view["error"]
    assert view["active_workers"] == []


def test_multiple_workers_and_error_request_cannot_be_overridden_by_late_success(snapshot):
    events = [worker("running", "first"), worker("running", "second"), terminal("error"),
              terminal(), worker("success", "first")]
    view = snapshot(events)
    assert view["running"] and view["finalization_pending"]
    assert view["completion_blockers"] == ["worker:second"]
    view = snapshot(events + [worker("success", "second")])
    assert not view["done"] and not view["running"] and view["state"] == "ERROR"
    assert view["error"] == "pipeline failed"


def test_delayed_closed_worker_start_cannot_reactivate_parent(snapshot):
    view = snapshot([worker("running"), worker("success"), worker("running"), terminal()])
    assert view["active_workers"] == [] and view["active_runs"] == []
    assert view["done"] and not view["running"]


def test_failure_before_cleanup_is_retained_when_cleanup_finishes_successfully(snapshot):
    view = snapshot([worker("running"), terminal(), worker("error", cleanup_confirmed=False),
                     worker("success", cleanup_confirmed=True)])
    assert view["done"] and not view["running"] and view["state"] == "ERROR"


def test_worker_terminal_does_not_imply_dangling_code_scope_was_cleaned(snapshot):
    events = [worker("running"), run("running"), terminal(), worker("error")]
    view = snapshot(events)
    assert view["running"] and view["finalization_pending"]
    assert view["active_workers"] == [] and view["completion_blockers"] == ["execution:run-a"]
    view = snapshot(events + [run("interrupted", cleanup_confirmed=True)])
    assert view["done"] and not view["running"] and view["state"] == "ERROR"


def test_worker_owned_by_phase_must_close_before_phase_is_confirmed(snapshot):
    events = [planned(), worker("running", phase_id=PHASE), owner("success")]
    view = snapshot(events)
    assert row(view)["completion_pending"] and not row(view)["completion_confirmed"]
    view = snapshot(events + [worker("error", phase_id=PHASE), terminal()])
    assert row(view)["status"] == "error" and not row(view)["completion_confirmed"]
    assert view["state"] == "ERROR"


def test_terminal_error_remains_authoritative_after_late_done_success(snapshot):
    view = snapshot([terminal("error"), terminal()])
    assert view["state"] == "ERROR" and not view["done"]
    assert view["result"] is None and view["error"] == "pipeline failed"


def test_late_pipeline_error_invalidates_already_published_result(snapshot):
    view = snapshot([terminal(), terminal("error")])
    assert view["state"] == "ERROR" and view["result"]["state"] == "ERROR"
    assert view["result"]["error"] == "pipeline failed"


@pytest.mark.parametrize("kind", ["done", "error"])
def test_legacy_stream_without_scope_events_keeps_historical_terminal_behavior(snapshot, kind):
    events = [planned(), owner("running"), step("running"), terminal(kind)]
    view = snapshot(events)
    assert not view["running"] and not view["finalization_pending"]
    assert view["active_runs"] == [] and view["active_workers"] == []
    assert view["active_executions"] == [] and view["completion_blockers"] == []
    assert view["execution_context"]["status"] == "interrupted"


def test_legacy_phase_and_plan_refresh_preserve_observed_owner_completion(snapshot):
    view = snapshot([owner("success"), planned("waiting")])
    assert row(view)["status"] == "success" and row(view)["completion_confirmed"]
    view = snapshot([{"type": "state", "agent": "CodeExecutor", "state": "EXECUTE_CODE", "status": "success"}, terminal()])
    assert view["pipeline_stages"][0]["completion_confirmed"]
    assert view["done"] and not view["running"]


def test_pdf_resolution_exposes_only_source_fields_and_defaults_to_empty(snapshot):
    assert snapshot([])["pdf_resolution"] == {}
    event = {"type": "pdf_resolution", "sha256": "a" * 64, "title": "Reviewed title",
             "source": "uploaded_pdf", "profile": "profile-a", "scope": "baseline",
             "pages": 12, "bytes": 345, "private_key": "omit"}
    view = snapshot([event])
    assert view["pdf_resolution"] == {key: value for key, value in event.items()
                                      if key not in {"type", "private_key"}}
