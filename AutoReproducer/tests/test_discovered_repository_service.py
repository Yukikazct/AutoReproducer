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
    caches = [path for path in root.glob("*") if (path / ".ready").is_file()
              and (path / "torch-2.5.1+cpu.dist-info").is_dir()
              and (path / "numpy-1.26.4.dist-info").is_dir()]
    if not caches:
        pytest.skip("Native fixture requires a preinstalled torch 2.5.1+cpu / NumPy 1.26.4 cache")
    return caches[0]


def use_preinstalled_cache(monkeypatch, runner, cache):
    executor = runner.runner.executor

    def use_cache(_directory):
        error = executor._check_local_dependency_health(cache, executor.env_config["requirements_txt"], "cached")
        if not error:
            executor._deps_dir = str(cache)
        return error

    monkeypatch.setattr(executor, "_ensure_local_deps", use_cache)
    monkeypatch.setattr(runner.runner, "cached_environment", lambda _config: cache)


@pytest.mark.parametrize("family,steps,samples", [("signals", 40, 4), ("pairs", 60, 8)])
def test_two_actual_author_entrypoints_complete_service_and_report(
        tmp_path, monkeypatch, cpu_dependency_cache, family, steps, samples):
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
    assert validation["is_reproduced"] is True
    assert data["repository_executed"] is True
    assert validation["training_summary"] == {"optimizer_steps": steps, "steps_completed": steps,
                                               "test_samples": samples, "complete": True}
    assert validation["metrics_comparison"]["paper"]["accuracy"] == 100.0
    assert validation["metrics_comparison"]["actual"]["accuracy"] == 100.0
    assert validation["independent_metrics"]["numpy_torch_crosscheck"] is True
    assert validation["independent_metrics"]["evaluation"] == "fresh_process_torchscript_forward_numpy_metrics"
    assert read_json(workspace / CAPTURE_DIR / "capture.json")["optimizer_steps"] == steps
    assert data["native_preflight"]["native_probe_pass"] is True
    assert data["native_preflight"]["versions"] == {"torch": "2.5.1+cpu", "numpy": "1.26.4"}
    assert [record["id"] for record in execution["steps"]] == ["train", "independent_evaluation"]
    assert all(record["executed"] and record["success"] and record["exit_code"] == 0
               and not record["timed_out"] for record in execution["steps"])
    assert all(process.poll() is not None for process in processes)
    assert len(processes) >= 3  # native probe, author training, fresh evaluation
    for name, sha in fixture["snapshot"]["files"].items():
        assert digest(workspace / name) == sha
    last_phases = {event["phase_id"]: event["status"] for event in events
                   if event.get("type") == "state"}
    assert len(last_phases) == 7 and set(last_phases.values()) == {"success"}
    assert read_json(run_dir / "run_status.json")["status"] == "completed"
    with FileLock(str(run_dir / ".run.lock"), timeout=0):
        pass
    assert llm.get_call_count() == 2
    frozen = read_json(run_dir / "experiment_spec.json")["spec"]
    assert frozen == data["grounded_repository_plan"]
    assert frozen["semantic_review"]["accepted"] is True
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report == data["report"]
    assert "| accuracy |" in report
    metric_rows = [line for line in report.splitlines() if line.startswith("| accuracy |")]
    assert any(row.count("100") == 2 and "N/A" not in row for row in metric_rows)
    assert "fixed test split" in report
    assert str(samples) in report and str(steps) in report
    assert "最多10轮" not in report


def test_rejected_semantics_retains_audit_and_never_starts_runtime(tmp_path, monkeypatch):
    review = accepted_review()
    review["accepted"] = False
    review["checks"]["same_experiment"].update(passed=False, reason="The quoted baseline row uses train data.")
    _, runner, request, _ = setup_service(tmp_path, monkeypatch, review=review)
    runner.runner.run = Mock(side_effect=AssertionError("Unverified plan cannot execute"))
    events = []
    result = runner.run(request, on_event=events.append)
    assert result["state"] == "ERROR"
    data = result["data"]
    assert data["repository_executed"] is False and data["validation"]["is_reproduced"] is None
    assert data["validation"]["failure_phase"] == "plan_repository"
    rejection = read_json(Path(data["run_dir"]) / "plan_rejection.json")
    assert rejection["candidate"]["repository"] == {
        key: data["repository"][key] for key in ("url", "revision")}
    assert rejection["semantic_review"]["checks"]["same_experiment"]["passed"] is False
    assert "baseline row uses train data" in data["report"]
    assert not (Path(data["run_dir"]) / "repo" / CAPTURE_DIR).exists()
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
    assert last["execute_repository"] == last["verify_protocol"] == "skipped"
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
    assert result["data"]["validation"]["failure_phase"] == "plan_repository"
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert "grounded_repository_plan" not in result["data"]
    assert llm.get_call_count() == 0
    runner.runner.run.assert_not_called()


def test_pdf_changed_after_planning_cannot_claim_prepared(tmp_path, monkeypatch):
    from src import discovered_repository_plan as planner
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    request["prepare_only"] = True
    propose = planner.propose_plan

    def replace_pdf_after_plan(*args, **kwargs):
        plan = propose(*args, **kwargs)
        with fixture["pdf"].open("ab") as stream:
            stream.write(b"\n% changed after planning\n")
        return plan

    monkeypatch.setattr(planner, "propose_plan", replace_pdf_after_plan)
    runner.runner.run = Mock(side_effect=AssertionError("Changed PDF cannot execute"))
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == "prepare_environment"
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert result["data"]["validation"]["is_reproduced"] is None
    runner.runner.run.assert_not_called()


@pytest.mark.parametrize("change_during,expected_phase", [
    ("preflight", "prepare_environment"), ("execution", "verify_protocol"),
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
        if stage == change_during:
            with fixture["pdf"].open("ab") as stream:
                stream.write(b"\n% changed during observed process phase\n")
        # A claimed completed boundary cannot bypass the host's PDF check.
        return {"success": True, "steps": [{"id": "train", "executed": True}], "final": {}}

    monkeypatch.setattr(runner.runner, "run", completed_runtime)
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == expected_phase
    assert "PDF" in result["error"] and "发生变化" in result["error"]
    assert result["data"]["validation"]["is_reproduced"] is None
    assert calls == (["preflight"] if change_during == "preflight" else ["preflight", "execution"])


@pytest.mark.parametrize("persistence_failure", [False, True])
def test_requested_sources_survive_planning_rejection_without_hiding_reason(
        tmp_path, monkeypatch, persistence_failure):
    from src import discovered_repository_plan as planner
    review = accepted_review()
    review["accepted"] = False
    reason = "The paper row is a different experiment; original semantic rejection."
    review["checks"]["same_experiment"].update(passed=False, reason=reason)
    _, runner, request, llm = setup_service(tmp_path, monkeypatch, review=review)
    llm.prefixes = [{"request_files": ["author_model.py"]}]
    build_packet = planner.build_evidence_packet

    def omit_model_initially(*args, **kwargs):
        packet = build_packet(*args, **kwargs)
        del packet["repository"]["files"]["author_model.py"]
        return packet

    monkeypatch.setattr(planner, "build_evidence_packet", omit_model_initially)
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
    runner.runner.run = Mock(side_effect=AssertionError("Rejected plan cannot execute"))
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert reason in result["error"]
    assert result["data"]["validation"]["failure_phase"] == "plan_repository"
    assert llm.get_call_count() == 3
    run_dir = Path(result["data"]["run_dir"])
    rejection = read_json(run_dir / "plan_rejection.json")
    assert rejection["semantic_review"]["checks"]["same_experiment"]["reason"] == reason
    if persistence_failure:
        assert "storage unavailable" in result["data"]["source_packet_persistence_error"]
    else:
        packet = read_json(run_dir / "source_packet.json")
        assert "author_model.py" in packet["repository"]["files"]
        assert "SignalModel" in packet["repository"]["files"]["author_model.py"]["text"]
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
    assert before["validation"]["is_reproduced"] is True
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
    assert phases["execute_repository"] == phases["verify_protocol"] == "success"
    assert phases["generate_report"] == "error"
    with FileLock(str(run_dir / ".run.lock"), timeout=0):
        pass
