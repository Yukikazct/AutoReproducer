"""Method setup must recover before budgets or training start."""
from copy import deepcopy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.method_reproduction as method
from src.repository_profiles import get_profile


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = {"now": 1000.}
    monkeypatch.setattr(method, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time))
    exported = []
    def export(root, profile, workspace, *, offline=False):
        exported.append(offline)
        clock["now"] += 1500
        workspace.mkdir()
        return {"url": profile["repository"]["url"], "path": str(workspace), "files": {}}
    monkeypatch.setattr("src.repository_reproduction.export_repository", export)
    probe = {"id": "import_check", "argv": ["python", "-c", "print('PROBE')"]}
    train_step = {"id": "train", "argv": ["python", "-c", "print('TRAIN')"], "depends_on": ["import_check"]}
    evaluate = {"id": "evaluate", "argv": ["python", "-c", "print('METRICS')"], "depends_on": ["train"]}
    adapter = SimpleNamespace(
        prepare_dataset=Mock(return_value={"url": "https://example.org/pinned/dataset"}),
        materialize=Mock(return_value={"files": {}}), public_sources=Mock(return_value=[]),
        steps=Mock(side_effect=lambda profile, train=True: deepcopy([probe, train_step, evaluate] if train else [probe])),
        verify_environment=Mock(), verify=Mock(return_value={
            "status": "method_experiment_completed", "result_level": "method_experiment_completed",
            "is_reproduced": None, "quality_pass": True}))
    monkeypatch.setattr(method, "get_adapter", lambda profile: adapter)
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run", lambda *a, **k: {"report": "verified method report"})
    calls = []
    def run(workspace, steps, env, on_event=None):
        calls.append({"steps": deepcopy(steps), "env": deepcopy(env), "at": clock["now"]})
        clock["now"] += 500 if len(calls) == 1 else 10
        return {"success": True, "mode": "repository", "executed": True,
                "environment": {"preparation_elapsed_s": 500 if len(calls) == 1 else 3},
                "artifacts": [], "final": {"stdout": "", "stderr": ""}}
    runner = SimpleNamespace(run=Mock(side_effect=run))
    logger = Mock()
    logger.get_stats.return_value = {}
    service = method.MethodReproduction(tmp_path, logger, runner=runner)
    return SimpleNamespace(service=service, adapter=adapter, runner=runner, calls=calls,
                           clock=clock, exported=exported, root=tmp_path)


def test_full_method_prepares_online_without_spending_baseline_or_optimization_budget(setup, monkeypatch):
    optimization_starts = []
    def optimize(service, profile, data, request, started, emit):
        optimization_starts.append(started)
        return {"status": "no_eligible_candidates", "optimized": False}
    monkeypatch.setattr("src.method_optimization.validate_candidates", optimize)
    frozen = get_profile("neural_ode_spiral")
    result = setup.service.run({"experiment_profile": frozen["id"], "optimization_mode": "validate"})
    assert result["state"] == "COMPLETED"
    assert setup.exported == [False]
    assert setup.adapter.prepare_dataset.call_args.kwargs["offline"] is False
    assert [step["id"] for step in setup.calls[0]["steps"]] == ["import_check"]
    assert "deadline_monotonic" not in setup.calls[0]["env"]
    assert setup.calls[0]["env"]["auto_prepare"] and setup.calls[0]["env"]["dependency_health_check"]
    assert setup.calls[0]["env"]["cache_lock_timeout_s"] > 0
    assert [step["id"] for step in setup.calls[1]["steps"]] == ["import_check", "train", "evaluate"]
    assert setup.calls[1]["env"]["deadline_monotonic"] - setup.calls[1]["at"] == 1200
    assert result["data"]["preparation_elapsed_s"] == 2000
    assert result["data"]["baseline_elapsed_s"] == 7
    assert optimization_starts == [3003]
    assert result["data"]["experiment_spec"] == frozen
    setup.adapter.verify_environment.assert_called_once()
    saved = json.loads((Path(result["data"]["run_dir"]) / "environment_preparation_execution.json").read_text(encoding="utf-8"))
    assert saved["success"] and "deadline_monotonic" not in setup.calls[0]["env"]


def test_explicit_offline_request_is_preserved(setup):
    result = setup.service.run({"experiment_profile": "neural_ode_spiral", "offline": True})
    assert result["state"] == "COMPLETED"
    assert setup.exported == [True]
    assert setup.adapter.prepare_dataset.call_args.kwargs["offline"] is True
    assert all(call["env"]["offline"] is True for call in setup.calls)


@pytest.mark.parametrize("failure", ["dependencies", "device"])
def test_environment_failure_stops_training_and_preserves_preparation_evidence(setup, failure):
    failed = {"success": False, "executed": False, "mode": "repository", "artifacts": [],
              "environment": {"dependency_cache_repairs": [{"status": "failed"}]},
              "final": {"stderr": "native dependency cannot be repaired"}}
    if failure == "dependencies":
        setup.runner.run.side_effect = lambda *a, **k: deepcopy(failed)
    else:
        setup.adapter.verify_environment.side_effect = ValueError("requested CUDA device is unavailable")
    result = setup.service.run({"experiment_profile": "neural_ode_spiral"})
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == "prepare_environment"
    setup.runner.run.assert_called_once()
    assert [step["id"] for step in setup.runner.run.call_args.args[1]] == ["import_check"]
    setup.adapter.verify.assert_not_called()
    assert result["data"]["environment_preparation_execution"] == result["data"]["execution"]
    evidence = Path(result["data"]["run_dir"]) / "environment_preparation_execution.json"
    assert evidence.is_file()


def test_source_only_preparation_never_invokes_native_executor(setup):
    result = setup.service.run({"experiment_profile": "neural_ode_spiral", "prepare_only": True})
    assert result["state"] == "COMPLETED"
    setup.runner.run.assert_not_called()
    setup.adapter.verify_environment.assert_not_called()
    assert result["data"]["validation"]["status"] == "prepared"


def test_environment_only_preparation_checks_device_without_training_deadline(setup):
    result = setup.service.run({"experiment_profile": "neural_ode_spiral", "prepare_environment": True})
    assert result["state"] == "COMPLETED"
    assert len(setup.calls) == 1 and [step["id"] for step in setup.calls[0]["steps"]] == ["import_check"]
    assert "deadline_monotonic" not in setup.calls[0]["env"]
    setup.adapter.verify_environment.assert_called_once()
    setup.adapter.verify.assert_not_called()
    assert result["data"]["validation"]["status"] == "environment_prepared"


def test_interrupted_preparation_is_saved_without_starting_training(setup):
    interrupted = KeyboardInterrupt("cancel during setup")
    interrupted.execution = {"success": False, "mode": "repository", "executed": False,
                             "cancelled": True, "final": {"exit_code": 130, "cancelled": True}}
    setup.runner.run.side_effect = interrupted
    with pytest.raises(KeyboardInterrupt):
        setup.service.run({"experiment_profile": "neural_ode_spiral"})
    setup.runner.run.assert_called_once()
    assert [step["id"] for step in setup.runner.run.call_args.args[1]] == ["import_check"]
    run_dir = next((setup.root / "runs").iterdir())
    saved = json.loads((run_dir / "environment_preparation_execution.json").read_text(encoding="utf-8"))
    assert saved["cancelled"] and saved["final"]["exit_code"] == 130
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    setup.adapter.verify.assert_not_called()
