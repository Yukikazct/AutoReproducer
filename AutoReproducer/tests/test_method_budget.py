"""Standalone optimization budgets remain bounded; author execution ignores them."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.method_budget import optimization_budget_error
from src.method_reproduction import MethodReproduction


@pytest.mark.parametrize("budget", [True, False, None, "7200", [], -1, 0,
                                   float("nan"), float("inf"), float("-inf"), 7201, 10 ** 1000])
def test_budget_requires_a_finite_number_inside_total_limit(budget):
    assert optimization_budget_error(budget)


@pytest.mark.parametrize("budget", [1, 2399, 2400, 2400.0])
def test_budget_must_leave_time_before_frozen_confirmation_reserve(budget):
    reason = optimization_budget_error(budget)
    assert "大于 40 分钟" in reason and "120 分钟" in reason
    assert all(stage in reason for stage in ("在线分析", "基线", "候选", "确认"))


@pytest.mark.parametrize("budget", [2400.1, 2401, 3600, 7200, 7200.0])
def test_valid_budget_is_not_silently_extended(budget):
    assert optimization_budget_error(budget) == ""


@pytest.mark.parametrize("budget", [True, None, "7200", float("nan"), 0, 2399, 2400, 7201])
@pytest.mark.parametrize("mode_request", [{"optimization_mode": "validate"}, {"enable_optimization": True}])
def test_legacy_optimization_budget_does_not_reject_author_execution(tmp_path, monkeypatch, budget, mode_request):
    recover = Mock()
    monkeypatch.setattr("src.method_reproduction.recover_interrupted_optimizations", recover)
    service = MethodReproduction(tmp_path, Mock(), runner=SimpleNamespace(run=Mock()))
    execute = Mock(return_value={"state": "COMPLETED", "data": {}, "error": None})
    monkeypatch.setattr(service, "_run", execute)
    result = service.run({"experiment_profile": "siren_camera_quick", "use_llm_review": True,
                          "budget_seconds": budget, **mode_request})
    assert result["state"] == "COMPLETED"
    request = execute.call_args.args[0]
    assert request["optimization_mode"] == "off"
    assert request["enable_optimization"] is False
    assert request["use_llm_review"] is False
    assert request["budget_seconds"] is budget
    recover.assert_called_once()
    execute.assert_called_once()


@pytest.mark.parametrize("extra", [{"prepare_only": True}, {"prepare_environment": True},
                                   {"optimization_mode": "suggest"}, {"optimization_mode": "off"},
                                   {"budget_seconds": 2401}, {"budget_seconds": 7200}])
def test_legacy_budget_allows_existing_run_recovery_before_execution(tmp_path, monkeypatch, extra):
    class PassedPreflight(Exception):
        pass
    recover = Mock(side_effect=PassedPreflight)
    monkeypatch.setattr("src.method_reproduction.recover_interrupted_optimizations", recover)
    service = MethodReproduction(tmp_path, Mock(), runner=SimpleNamespace(run=Mock()))
    with pytest.raises(PassedPreflight):
        service.run({"experiment_profile": "siren_camera_quick", "optimization_mode": "validate",
                     "budget_seconds": 2400, **extra})
    recover.assert_called_once()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("prepare_flag", ["prepare_only", "prepare_environment"])
@pytest.mark.parametrize("budget", [2400, None])
def test_preparation_does_not_use_optimization_budget(tmp_path, monkeypatch, prepare_flag, budget):
    adapter = SimpleNamespace(prepare_dataset=Mock(return_value={}), materialize=Mock(return_value={}),
                              public_sources=Mock(return_value=[]),
                              steps=Mock(return_value=[{"id": "train", "kind": "train", "argv": ["python", "-c", "print(1)"]}]),
                              verify_environment=Mock())
    monkeypatch.setattr("src.method_reproduction.get_adapter", lambda profile: adapter)
    monkeypatch.setattr("src.repository_reproduction.export_repository",
                        lambda root, profile, workspace, **kwargs:
                        {"url": "https://example.org/repository", "path": str(workspace)})
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run",
                        lambda *args, **kwargs: {"report": "preparation complete"})
    runner = SimpleNamespace(run=Mock(return_value={"success": True, "artifacts": [], "final": {}}))
    logger = Mock()
    logger.get_stats.return_value = {}
    service = MethodReproduction(tmp_path, logger, runner=runner)
    result = service.run({"experiment_profile": "siren_camera_quick", "optimization_mode": "validate",
                          prepare_flag: True, "budget_seconds": budget})
    assert result["state"] == "COMPLETED", result.get("error")
    assert result["data"]["validation"]["result_level"] == "prepared"
    adapter.steps.assert_called_once()
    assert adapter.steps.call_args.kwargs["train"] is True
    if prepare_flag == "prepare_only":
        runner.run.assert_not_called()
    else:
        runner.run.assert_called_once()
        assert "deadline_monotonic" not in runner.run.call_args.args[2]
