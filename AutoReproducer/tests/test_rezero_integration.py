"""Registration and orchestration of the strict ReZero paper experiment."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.method_reproduction as method
from src.experiments.rezero_runtime import REFERENCE_PARAMETERS
from src.repository_adapters import get_adapter
from src.repository_profiles import PROFILE_LABELS, get_profile
from src.repository_reproduction import RepositoryReproduction
from src.rezero_adapter import ReZeroAdapter, SOURCE_SHA256
from src.rezero_profiles import REZERO_PROFILE_ID


def test_registered_paper_protocol_has_complete_inputs_and_an_independent_reference():
    profile = get_profile(REZERO_PROFILE_ID)
    assert REZERO_PROFILE_ID in PROFILE_LABELS
    assert isinstance(get_adapter(profile), ReZeroAdapter)
    assert profile["parameters"] == REFERENCE_PARAMETERS
    assert profile["repository"] == {
        "url": "https://github.com/tbachlechner/ReZero-Superconvergence",
        "revision": "6c0212669ac8c23d3db6f2b99255bcf6e3c5e6e6"}
    assert profile["source_sha256s"] == SOURCE_SHA256
    assert profile["dataset"]["sha256"] == "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
    assert profile["dataset"]["train_samples"] == 50000
    assert profile["dataset"]["test_samples"] == 10000
    assert profile["paper"]["metrics"] == {"top1_accuracy_pct": 94.0}
    assert profile["paper"]["required_metrics"] == ["top1_accuracy_pct", "cross_entropy"]
    assert profile["validation"]["scope"] == "selected_paper_experiment"
    steps = get_adapter(profile).steps(profile)
    assert [s["id"] for s in steps] == ["import_check", "train", "evaluate"]
    assert steps[-1]["argv"] == ["python", "-u", "evaluate.py", "test"]
    assert profile["budget"]["baseline_s"] >= sum(s["timeout_s"] for s in steps)
    assert profile["environment"]["requirements_txt"] == (
        "--extra-index-url https://download.pytorch.org/whl/cu121\n"
        "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\n"
        "numpy==1.26.4\nscipy==1.14.1\nmatplotlib==3.9.2\nPillow==10.4.0\n")
    profile["parameters"]["momentum_range"][0] = 0
    profile["source_sha256s"].clear()
    assert get_profile(REZERO_PROFILE_ID)["parameters"] == REFERENCE_PARAMETERS
    assert get_profile(REZERO_PROFILE_ID)["source_sha256s"] == SOURCE_SHA256


@pytest.mark.parametrize("options", [
    {"optimization_mode": "suggest"}, {"optimization_mode": "validate"},
    {"enable_optimization": True}, {"prepare_only": True, "optimization_mode": "validate"},
])
def test_rezero_rejects_protocol_search_before_creating_run(tmp_path, options):
    service = method.MethodReproduction(tmp_path, Mock(), runner=Mock())
    with pytest.raises(ValueError, match="冻结协议"):
        service.run({"experiment_profile": REZERO_PROFILE_ID, **options})
    assert not (tmp_path / "runs").exists()
    service.runner.run.assert_not_called()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(method, "time", SimpleNamespace(monotonic=lambda: now["value"], time=lambda: now["value"]))
    def export(root, profile, workspace, **kwargs):
        workspace.mkdir()
        return {**profile["repository"], "resolved_sha": profile["repository"]["revision"],
                "path": str(workspace), "files": {}}
    monkeypatch.setattr("src.repository_reproduction.export_repository", export)
    verdict = {"status": "reproduced", "result_level": "reproduced", "is_reproduced": True,
               "scope": "selected_paper_experiment", "quality_pass": True,
               "protocol_pass": True, "independent_metrics_pass": True,
               "optimization_eligible": False}
    adapter = SimpleNamespace(
        prepare_dataset=Mock(return_value={"url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"}),
        materialize=Mock(return_value={"files": {}}), public_sources=Mock(return_value=[]),
        steps=ReZeroAdapter().steps, verify_environment=Mock(), verify=Mock(return_value=verdict))
    monkeypatch.setattr(method, "get_adapter", lambda profile: adapter)
    monkeypatch.setattr("src.method_advice.review_sources", Mock(return_value={"status": "accepted"}))
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run", lambda *a, **k: {"report": "test"})
    calls = []
    def run(workspace, steps, env, on_event=None):
        calls.append({"at": now["value"], "steps": deepcopy(steps), "env": deepcopy(env)})
        now["value"] += 10 if len(calls) == 1 else 2000
        return {"mode": "repository", "success": True, "executed": True, "artifacts": [],
                "environment": {}, "final": {"stderr": ""}}
    runner, logger = SimpleNamespace(run=Mock(side_effect=run)), Mock()
    logger.get_stats.return_value = {}
    service = RepositoryReproduction(tmp_path, logger, runner=runner)
    return SimpleNamespace(service=service, adapter=adapter, runner=runner, calls=calls, root=tmp_path)


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("quality", [False, True])
def test_repository_dispatch_preserves_strict_verdict_and_long_training_budget(pipeline, review, quality):
    if not quality:
        pipeline.adapter.verify.return_value.update(status="reference_not_met", result_level="experiment_completed",
                                                    is_reproduced=False, quality_pass=False)
    events = []
    result = pipeline.service.run({"experiment_profile": REZERO_PROFILE_ID,
                                   "use_llm_review": review}, on_event=events.append)
    assert result["state"] == "COMPLETED" and result["error"] is None
    assert [step["id"] for step in pipeline.calls[0]["steps"]] == ["import_check"]
    assert "deadline_monotonic" not in pipeline.calls[0]["env"]
    training = pipeline.calls[1]
    assert [step["id"] for step in training["steps"]] == ["import_check", "train", "evaluate"]
    assert training["env"]["deadline_monotonic"] - training["at"] == 7800
    assert result["data"]["validation"] == pipeline.adapter.verify.return_value
    assert result["data"]["validation"]["is_reproduced"] is quality
    assert result["data"]["optimization"] == {
        "optimized": False, "status": "disabled", "available": False, "mode": "off"}
    assert not result["data"]["quick_target"]["validated_on_this_run"]
    assert all(event.get("status") != "running" for event in events if event.get("phase_id") == "method_advice")
    assert Path(result["data"]["run_dir"], "validation.json").is_file()


def pdf_resolution():
    return {"profile": REZERO_PROFILE_ID, "source": "pdf_author_code_repository", "sha256": "a" * 64,
            "pages": 14, "bytes": 1314235,
            "discovery_repository_url": "https://github.com/majumderb/rezero",
            "training_repository_url": "https://github.com/tbachlechner/ReZero-Superconvergence",
            "repository_relationship": {"source": "reviewed_author_readme"},
            "evidence": {"url": "https://github.com/majumderb/rezero", "page": 4,
                         "context": "Code for ReZero is available at https://github.com/majumderb/rezero",
                         "source": "pdf_text", "evidence_type": "author_code_statement"},
            "repository_links": [{"url": "https://github.com/majumderb/rezero"}]}


def test_preparation_transports_original_pdf_evidence_and_actual_training_repo(pipeline):
    resolution = pdf_resolution()
    result = pipeline.service.run({"experiment_profile": REZERO_PROFILE_ID, "prepare_only": True,
                                   "pdf_resolution": resolution})
    data = result["data"]
    assert result["state"] == "COMPLETED"
    assert data["resources"]["code_repo_url"] == resolution["training_repository_url"]
    assert data["resources"]["discovery_repository_url"] == resolution["discovery_repository_url"]
    assert data["resources"]["selection_evidence"] == resolution["evidence"]
    assert data["pdf_input"]["sha256"] == resolution["sha256"]
    assert data["validation"]["is_reproduced"] is None
    assert data["pdf_resolution"] == resolution
    data["resources"]["selection_evidence"]["page"] = 7
    assert resolution["evidence"]["page"] == 4
    pipeline.runner.run.assert_not_called()
    pipeline.adapter.verify.assert_not_called()


def test_mismatched_pdf_training_repo_stops_before_any_execution(pipeline):
    resolution = pdf_resolution()
    resolution["training_repository_url"] = "https://github.com/example/unrelated"
    result = pipeline.service.run({"experiment_profile": REZERO_PROFILE_ID, "pdf_resolution": resolution})
    assert result["state"] == "ERROR"
    assert result["data"]["validation"]["failure_phase"] == "prepare_repository"
    assert "冻结训练仓库不一致" in result["error"]
    pipeline.runner.run.assert_not_called()
