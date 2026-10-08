"""Method experiments must not turn untested advice or fit metrics into evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.method_advice import suggest, validate_suggestions
from src.method_profiles import method_profile
from src.method_adapters import SirenAdapter, write_json, download_dataset
from src.repository_adapters import get_adapter
from src.repository_runner import RepositoryRunner
from src.agents.code_executor import CodeExecutorAgent


def advice():
    return {"parameter": "learning_rate", "value": .0002, "hypothesis": "固定预算内收敛更快",
            "expected_effect": "待验证", "cost": "可能不稳定", "validation_plan": "只比较固定验证像素",
            "evidence": [{"source_id": "author", "quote": "lr=1e-4"}]}


SOURCES = [{"source_id": "author", "content": "optim = Adam(lr=1e-4)", "url": "https://example.org/author", "locator": "L1"}]


@pytest.mark.parametrize("change", [
    {"value": True}, {"value": .1}, {"value": .0001}, {"parameter": "steps"},
    {"evidence": []}, {"evidence": [{"source_id": "author", "quote": "invented quote"}]},
    {"hypothesis": ""},
])
def test_advice_rejects_unsafe_or_unsupported_claims(change):
    with pytest.raises(ValueError):
        validate_suggestions(json.dumps({"suggestions": [{**advice(), **change}]}), method_profile("siren_camera_quick"), SOURCES)


def test_valid_advice_remains_untested():
    items = validate_suggestions(json.dumps({"suggestions": [advice()]}), method_profile("siren_camera_quick"), SOURCES)
    assert items[0]["status"] == "untested"
    assert items[0]["baseline_value"] == .0001


def test_advice_timeout_is_bounded_and_does_not_claim_gain(monkeypatch):
    llm = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test", api_key="private", call_count=0)
    def timeout(*args, **kwargs):
        assert "private" not in str(args)
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(args[0], 2)
    monkeypatch.setattr("src.method_advice.subprocess.run", timeout)
    result = suggest(llm, method_profile("siren_camera_quick"), {
        "metrics_comparison": {"actual": {"psnr": 28}}, "training_summary": {},
        "protocol_pass": True, "independent_metrics_pass": True}, SOURCES, 2)
    assert result["status"] == "advice_timeout"
    assert result["optimized"] is False
    assert "private" not in json.dumps(result)


def test_missing_environment_never_installs_or_trains(tmp_path, monkeypatch):
    monkeypatch.setattr("src.repository_runner.DEPS_CACHE_ROOT", tmp_path / "deps")
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(side_effect=AssertionError("must not install"))
    runner = RepositoryRunner(executor=executor)
    repo = tmp_path / "repo"; repo.mkdir()
    result = runner.run(repo, [{"id": "train", "argv": ["python", "-c", "raise AssertionError('must not train')"]}],
                        {"require_prepared": True, "requirements_txt": "example==1", "cache_lock_timeout_s": 0})
    assert not result["success"] and not result["executed"]
    assert "尚未准备" in result["final"]["stderr"]
    executor._ensure_local_deps.assert_not_called()


def test_overall_deadline_stops_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("src.repository_runner.DEPS_CACHE_ROOT", tmp_path / "deps")
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(return_value=None)
    runner = RepositoryRunner(executor=executor)
    repo = tmp_path / "repo"; repo.mkdir()
    result = runner.run(repo, [{"id": "train", "argv": ["python", "-c", "raise AssertionError('must not train')"]}],
                        {"deadline_monotonic": time.monotonic()-1})
    assert result["final"]["exit_code"] == 124
    assert not result["executed"]


def test_offline_dataset_corruption_cannot_fall_back(tmp_path):
    spec = method_profile("siren_camera_quick")["dataset"]
    with pytest.raises(RuntimeError, match="校验"):
        download_dataset(tmp_path, spec, tmp_path, offline=True)


def test_unknown_adapter_does_not_use_dlinear():
    with pytest.raises(ValueError):
        get_adapter({"adapter_id": "unreviewed"})
    assert isinstance(get_adapter(method_profile("siren_camera_quick")), SirenAdapter)


def test_independent_pixel_evaluation_only_uses_selected_split(tmp_path):
    np = pytest.importorskip("numpy")
    from PIL import Image
    image_path = tmp_path / "camera.png"
    Image.fromarray(np.full((10,10), 100, dtype=np.uint8)).save(image_path)
    cfg = {"parameters": {"sidelength": 10, "protocol": "pixel_holdout"},
           "dataset_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "spec_sha256": "test"}
    write_json(tmp_path / "experiment.json", cfg)
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()
    pred = np.full((10,10), 100/255.)
    indices = np.random.default_rng(1729).permutation(100)
    pred.ravel()[indices[80:90]] += .1
    pred.ravel()[indices[90:]] += .3
    np.save(artifacts / "prediction.npy", pred)
    script = Path(__file__).resolve().parents[1] / "src/experiments/evaluate_siren.py"
    result = subprocess.run([sys.executable, str(script), "validation"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metric = json.loads(result.stdout)
    assert metric["metrics"]["psnr"] == pytest.approx(20)
    assert metric["samples"] == 10
    assert not (artifacts / "metrics_holdout.json").exists()
    assert not list(artifacts.glob("*.png"))


def test_siren_fixed_contract_has_no_paper_reference():
    p = method_profile("siren_camera_quick")
    assert p["paper"]["metrics"] == {}
    assert p["parameters"]["steps"] == 500
    assert p["parameters"]["sidelength"] == 256
    assert p["parameters"]["hidden_layers"] == 3
    assert p["budget"] == {"total_s": 300, "baseline_s": 240, "advice_s": 45}


def test_neural_ode_registers_its_own_frozen_protocol():
    from src.method_adapters import NeuralODEAdapter
    p = method_profile("neural_ode_spiral")
    assert isinstance(get_adapter(p), NeuralODEAdapter)
    assert p["parameters"]["steps"] == 2000
    assert p["parameters"]["solver"] == "dopri5"
    assert p["parameters"]["batch_time"] == 10
    assert p["parameters"]["learning_rate"] == .001
    assert p["dataset"]["kind"] == "analytic"
    assert p["paper"]["metrics"] == {}
