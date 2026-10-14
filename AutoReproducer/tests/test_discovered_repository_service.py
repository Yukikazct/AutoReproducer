"""Actual author scripts, owned processes, capture/evaluation and final report.

Only public network inputs and dependency-cache discovery are injected. Native
CPU fixtures are synthetic unit-test documents, never a paper-reproduction claim.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock

from filelock import FileLock
import pytest

from discovered_fixtures import PlannedLLM, accepted_review, make_experiment
from src import discovered_repository_reproduction as service
from src.audit.audit_logger import AuditLogger
from src.discovered_repository_runtime import CAPTURE_DIR
from src.method_adapters import digest, read_json
from src.runtime_platform import runtime_fingerprint
from src.dependency_versions import compatibility_pins


def setup_service(tmp_path, monkeypatch, family="signals", review=None):
    fixture = make_experiment(tmp_path, family)
    snapshot = {**fixture["snapshot"], "resolved_sha": fixture["snapshot"]["revision"]}

    def export(root, profile, workspace, **kwargs):
        assert profile["repository"] == fixture["proposal"]["repository"]
        shutil.copytree(fixture["workspace"], workspace)
        return deepcopy(snapshot)

    monkeypatch.setattr(service, "export_repository", export)
    # A complete supplied SHA takes the service's actual offline resolver path.
    llm = PlannedLLM(fixture["proposal"], review)
    logger = AuditLogger(log_dir=str(tmp_path / "logs"), ledger_dir=str(tmp_path / "ledger"))
    runner = service.DiscoveredRepositoryReproduction(tmp_path / "service", logger, llm=llm)
    request = {"pdf_path": str(fixture["pdf"]), "offline": True,
               "paper_info": {"title": "Invented protocol integration fixture"},
               "resources": {"code_repo_url": snapshot["url"],
                             "repo_discovery": {"selected_repo": snapshot["url"],
                                                "pinned_revision": snapshot["revision"]}}}
    return fixture, runner, request, llm


@pytest.fixture
def cpu_dependency_cache():
    """Reuse an already installed exact CPU build; never install during tests."""
    root = Path(__file__).resolve().parents[1] / "data" / "deps" / "repository" / runtime_fingerprint()
    torch_version = compatibility_pins()["torch"]
    caches = [path for path in root.glob("*") if (path / ".ready").is_file()
              and (path / f"torch-{torch_version}.dist-info").is_dir()
              and (path / "numpy-1.26.4.dist-info").is_dir()]
    if not caches:
        pytest.skip(f"Native fixture requires a preinstalled torch {torch_version} / NumPy 1.26.4 cache")
    return caches[0]


def use_preinstalled_cache(monkeypatch, runner, cache):
    import src.repository_runner as repository
    # This fixture intentionally uses a real preinstalled environment. Its
    # ownership lock and path boundary must refer to that same managed root.
    monkeypatch.setattr(repository, 'DEPS_CACHE_ROOT', cache.parents[2])
    executor = runner.runner.executor

    def use_cache(_directory):
        error = executor._check_local_dependency_health(cache, executor.env_config["requirements_txt"], "cached")
        if not error:
            executor._deps_dir = str(cache)
        return error

    monkeypatch.setattr(executor, "_ensure_local_deps", use_cache)
    monkeypatch.setattr(runner.runner, "cached_environment", lambda _config: cache)


@pytest.mark.parametrize("family,steps", [("signals", 40), ("pairs", 60)])
def test_two_actual_author_entrypoints_complete_service_and_report(tmp_path, monkeypatch, cpu_dependency_cache, family, steps):
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch, family)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    processes, events = [], []
    popen = subprocess.Popen
    def remember_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", remember_process)
    result = runner.run(request, on_event=events.append)
    assert result["state"] == "COMPLETED", result["error"]
    data = result["data"]
    run_dir = Path(data["run_dir"])
    workspace = run_dir / "repo"
    validation, execution = data["validation"], data["execution"]
    assert validation["is_reproduced"] is None
    assert validation["status"] == "author_code_completed"
    assert data["repository_executed"] is True
    assert all(validation["completion_checks"].values())
    assert validation["author_execution"][0]["optimizer_steps"] == steps
    assert [record["id"] for record in execution["steps"]] == ["train"]
    assert all(record["executed"] and record["success"] and record["exit_code"] == 0
               and not record["timed_out"] and record["timeout_s"] is None for record in execution["steps"])
    assert all(process.poll() is not None for process in processes)
    assert len(processes) >= 2
    for name, sha in fixture["snapshot"]["files"].items():
        assert digest(workspace / name) == sha
    phases = {event["phase_id"]: event["status"] for event in events if event.get("type") == "state"}
    assert phases["execute_repository"] == phases["finish_execution"] == "success"
    assert not any(key.startswith(("review_", "verify_")) for key in phases)
    assert read_json(run_dir / "run_status.json")["status"] == "completed"
    with FileLock(str(run_dir / ".run.lock"), timeout=0):
        pass
    assert {options["task"] for _, options in llm.calls} == {"author_execution_plan", "author_claim_comparison"}
    assert read_json(run_dir / "experiment_spec.json")["spec"] == data["grounded_repository_plan"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report == data["report"] and "| accuracy |" in report
    assert str(steps) in report


def test_dependency_warning_cannot_skip_failed_runtime_preflight(tmp_path, monkeypatch):
    review = accepted_review()
    review['checks']['dependency_compatibility'].update(passed=False, severity='warning',
        reason='Reviewed modern versions require actual runtime API verification.')
    _, runner, request, _ = setup_service(tmp_path, monkeypatch, review=review)
    runner.runner.run = Mock(return_value={'success': False, 'reason': 'Actual native operation failed'})
    result = runner.run(request)
    assert result['state'] == 'ERROR'
    assert result['data']['repository_executed'] is False
    assert result['data']['validation']['is_reproduced'] is None
    runner.runner.run.assert_called_once()
    assert runner.runner.run.call_args.args[1][0]['id'] == 'native_preflight'


def test_old_semantic_review_does_not_block_preparation(tmp_path, monkeypatch):
    review = accepted_review()
    review["accepted"] = False
    _, runner, request, llm = setup_service(tmp_path, monkeypatch, review=review)
    runner.runner.run = Mock(side_effect=AssertionError("Prepare-only cannot execute"))
    result = runner.run({**request, "prepare_only": True, "author_execution_only": False})
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "prepared"
    assert {options["task"] for _, options in llm.calls} == {"author_execution_plan"}
    runner.runner.run.assert_not_called()


def test_prepare_only_is_prepared_and_marks_training_stages_skipped(tmp_path, monkeypatch):
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    request["prepare_only"] = True
    runner.runner.run = Mock(side_effect=AssertionError("Preparation cannot execute"))
    events = []
    result = runner.run(request, on_event=events.append)
    assert result["error"] is None
    assert result["data"]["validation"]["status"] == "prepared"
    assert result["data"]["validation"]["is_reproduced"] is None
    assert not result["data"]["repository_executed"]
    last = {event["phase_id"]: event["status"] for event in events if event.get("type") == "state"}
    assert last["execute_repository"] == last["finish_execution"] == "skipped"
    runner.runner.run.assert_not_called()


def test_pdf_changed_during_repository_export_cannot_start_planning(tmp_path, monkeypatch):
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch)
    request.update(prepare_only=True, pdf_input={"sha256": digest(fixture["pdf"])})
    export = service.export_repository

    def replace_pdf(*args, **kwargs):
        snapshot = export(*args, **kwargs)
        # A valid trailing PDF comment changes identity without hiding links.
        with fixture["pdf"].open("ab") as stream:
            stream.write(b"\n% replaced during repository export\n")
        return snapshot

    monkeypatch.setattr(service, "export_repository", replace_pdf)
    runner.runner.run = Mock(side_effect=AssertionError("Changed PDF cannot execute"))
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == "prepare_repository"
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert "grounded_repository_plan" not in result["data"]
    assert llm.get_call_count() == 0
    runner.runner.run.assert_not_called()


def test_pdf_changed_after_planning_cannot_claim_prepared(tmp_path, monkeypatch):
    from src import author_execution_plan as planner
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    request["prepare_only"] = True
    propose = planner.propose_author_execution_plan

    def replace_pdf_after_plan(*args, **kwargs):
        plan = propose(*args, **kwargs)
        with fixture["pdf"].open("ab") as stream:
            stream.write(b"\n% changed after planning\n")
        return plan

    monkeypatch.setattr(planner, "propose_author_execution_plan", replace_pdf_after_plan)
    runner.runner.run = Mock(side_effect=AssertionError("Changed PDF cannot execute"))
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == "prepare_environment"
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert result["data"]["validation"]["is_reproduced"] is None
    runner.runner.run.assert_not_called()


@pytest.mark.parametrize("change_during,expected_phase", [
    ("preflight", "prepare_environment"), ("execution", "finish_execution"),
])
def test_pdf_identity_rechecked_after_native_preparation_and_execution(
        tmp_path, monkeypatch, change_during, expected_phase):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    calls = []

    def completed_runtime(workspace, steps, environment, **kwargs):
        stage = "execution" if isinstance(steps, dict) else "preflight"
        calls.append(stage)
        if stage == "preflight":
            service.write_json(Path(workspace) / CAPTURE_DIR / "preflight.json",
                               {"native_probe_pass": True, "device": "cpu"})
        else:
            config = read_json(Path(workspace) / CAPTURE_DIR / "config.json")
            service.write_json(Path(workspace) / CAPTURE_DIR / "author_train.json", {
                "spec_sha256": config["spec_sha256"], "step": config["steps"][0],
                "natural_return": True, "optimizer_steps": 40, "numeric_compatibility": []})
        if stage == change_during:
            with fixture["pdf"].open("ab") as stream:
                stream.write(b"\n% changed during observed process phase\n")
        # A claimed completed boundary cannot bypass the host's PDF check.
        return {"success": True, "steps": [{"id": "train", "executed": True,
                                             "success": True, "exit_code": 0}], "final": {}}

    monkeypatch.setattr(runner.runner, "run", completed_runtime)
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == expected_phase
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert result["data"]["validation"]["is_reproduced"] is None
    assert calls == (["preflight"] if change_during == "preflight" else ["preflight", "execution"])


@pytest.mark.parametrize("persistence_failure", [False, True])
def test_requested_sources_survive_planning_rejection_without_hiding_reason(tmp_path, monkeypatch, persistence_failure):
    from src import author_execution_plan as planner
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    reason = "Original entrypoint is not runnable"
    def reject(llm, packet, workspace):
        packet["requested_source_record"] = {"path": "author_model.py"}
        raise ValueError(reason)
    monkeypatch.setattr(planner, "propose_author_execution_plan", reject)
    if persistence_failure:
        persist = service.write_json
        writes = 0
        def fail_final_packet(path, value):
            nonlocal writes
            if Path(path).name == "source_packet.json":
                writes += 1
                if writes == 2:
                    raise OSError("fixture packet storage unavailable")
            return persist(path, value)
        monkeypatch.setattr(service, "write_json", fail_final_packet)
    runner.runner.run = Mock(side_effect=AssertionError("Failed plan cannot execute"))
    result = runner.run(request)
    assert result["state"] == "ERROR" and reason in result["error"]
    assert result["data"]["validation"]["failure_phase"] == "plan_repository"
    if persistence_failure:
        assert "storage unavailable" in result["data"]["source_packet_persistence_error"]
    else:
        packet = read_json(Path(result["data"]["run_dir"]) / "source_packet.json")
        assert packet["requested_source_record"]["path"] == "author_model.py"
    runner.runner.run.assert_not_called()


def test_exception_without_message_stays_error_and_cannot_report_completed(tmp_path, monkeypatch):
    from src import author_execution_plan
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    monkeypatch.setattr(author_execution_plan, 'propose_author_execution_plan', Mock(side_effect=ValueError()))
    runner.runner.run = Mock(side_effect=AssertionError('Rejected protocol cannot execute'))
    result = runner.run(request)
    assert result['state'] == 'ERROR' and result['error'] == 'ValueError'
    assert result['data']['validation']['failure_phase'] == 'plan_repository'
    runner.runner.run.assert_not_called()


def test_report_interruption_retains_completed_native_execution_summary(
        tmp_path, monkeypatch, cpu_dependency_cache):
    from src.agents.report_generator import ReportGeneratorAgent
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    completed, events = {}, []

    def interrupt_report(_agent, data, **kwargs):
        completed["data"] = deepcopy(data)
        raise KeyboardInterrupt("fixture interruption during report")

    monkeypatch.setattr(ReportGeneratorAgent, "run", interrupt_report)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption during report"):
        runner.run(request, on_event=events.append)
    before = completed["data"]
    assert before["execution"]["success"] is True
    assert before["validation"]["is_reproduced"] is None
    run_dir = Path(before["run_dir"])
    result = read_json(run_dir / "result.json")
    assert result["state"] == "ERROR" and result["error"] == "interrupted"
    assert result["data"]["interrupted"] is True
    assert result["data"]["execution"] == before["execution"]
    assert result["data"]["validation"] == before["validation"]
    assert result["data"]["report_error"] == {
        "phase": "generate_report", "reason": "fixture interruption during report"}
    assert read_json(run_dir / "run_status.json")["status"] == "interrupted"
    phases = {event["phase_id"]: event["status"] for event in events if event.get("type") == "state"}
    assert phases["execute_repository"] == phases["finish_execution"] == "success"
    assert phases["generate_report"] == "error"
    with FileLock(str(run_dir / ".run.lock"), timeout=0):
        pass
