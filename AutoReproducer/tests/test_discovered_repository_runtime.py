"""Mechanics tests use invented fixture datasets, never a paper reproduction."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from discovered_fixtures import PlannedLLM, make_experiment
from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.discovered_repository_runtime import materialize_capture, validate_capture, verify_capture
from src.method_adapters import read_json, write_json
from src.repository_reproduction import spec_digest


@pytest.fixture
def experiment(tmp_path):
    result = make_experiment(tmp_path)
    result["packet"] = build_evidence_packet(result["pdf"], result["workspace"], result["snapshot"])
    return result


@pytest.mark.parametrize("field,value", [
    ("model", "model.eval()"), ("labels", "labels[:1]"), ("forward_args", ["__import__('os')"]),
    ("test_indices", "missing_names"), ("expected_optimizer_steps", 0),
    ("expected_test_samples", True), ("reported_metric", {"label": "\n", "unit": "fraction"}),
])
def test_capture_schema_rejects_expressions_and_incomplete_counts(experiment, field, value):
    capture = deepcopy(experiment["proposal"]["capture"])
    capture[field] = value
    with pytest.raises(ValueError):
        validate_capture(capture, experiment["packet"], experiment["workspace"])


def test_capture_does_not_overwrite_author_directory(experiment):
    (experiment["workspace"] / "_autorepro_capture").mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        materialize_capture(experiment["workspace"], experiment["proposal"], "a" * 64)


def test_changed_capture_source_rejected(experiment):
    (experiment["workspace"] / "author_model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_capture(experiment["proposal"]["capture"], experiment["packet"], experiment["workspace"])


def native_environment():
    environment = dict(os.environ)
    dependencies = environment.get("AUTOREPRO_TEST_CPU_DEPS")
    if dependencies:
        environment["PYTHONPATH"] = dependencies
    elif importlib.util.find_spec("torch") is None:
        pytest.skip("Native runtime test needs CPU torch; set AUTOREPRO_TEST_CPU_DEPS to an existing verified cache")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def invoke(workspace, action, environment):
    runtime = str(workspace / "_autorepro_capture" / "runtime.py")
    # These fixed fixtures create no child processes. Execute the runtime file
    # directly, as RepositoryRunner does; do not introduce a second runpy layer
    # around native TorchScript extension teardown just to inject test wheels.
    return subprocess.run([sys.executable, runtime, action], cwd=workspace, env=environment,
                          timeout=90, capture_output=True, text=True, encoding="utf-8")


def run_fixture(root, family="signals", mutate=None):
    environment = native_environment()
    fixture = make_experiment(root, family)
    if mutate:
        mutate(fixture)
    packet = build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    plan = propose_plan(PlannedLLM(fixture["proposal"]), packet, fixture["workspace"])
    digest = spec_digest(plan)
    manifest = materialize_capture(fixture["workspace"], plan, digest, snapshot=fixture["snapshot"])
    records, log_root = [], root / "owned_logs"
    log_root.mkdir()
    for step in manifest["steps"]:
        action = "capture" if step["kind"] == "train" else "evaluate"
        result = invoke(fixture["workspace"], action, environment)
        stdout_path = log_root / (step["id"] + ".stdout.log")
        stdout_path.write_text(result.stdout, encoding="utf-8")
        records.append({"id": step["id"], "success": result.returncode == 0, "executed": True,
                        "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                        "stdout_path": str(stdout_path), "timed_out": False, "cancelled": False})
        assert result.returncode == 0, result.stderr
    execution = {"success": True, "steps": records, "skipped": [], "run_dir": str(log_root)}
    return fixture, plan, digest, manifest, execution


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    return run_fixture(tmp_path_factory.mktemp("native_signals"))


def test_real_cpu_author_training_trace_and_fresh_numpy_evaluation(captured):
    fixture, plan, digest, manifest, execution = captured
    result = verify_capture(plan, fixture["workspace"], execution, digest)
    assert result["is_reproduced"] is True
    assert result["metrics_comparison"]["actual"]["accuracy"] == 100.0
    assert result["capture"]["optimizer_steps"] == 40
    assert result["capture"]["test_samples"] == 4
    assert result["independent_metrics"]["numpy_torch_crosscheck"] is True
    assert plan["semantic_review"]["accepted"] is True
    assert all(hashlib.sha256((fixture["workspace"] / name).read_bytes()).hexdigest() == sha
               for name, sha in fixture["snapshot"]["files"].items())
    assert all(hashlib.sha256((fixture["workspace"] / name).read_bytes()).hexdigest() == sha
               for name, sha in manifest["files"].items())


def test_second_unregistered_layout_multiple_inputs_complete_labels(tmp_path):
    fixture, plan, digest, _, execution = run_fixture(tmp_path, "pairs")
    result = verify_capture(plan, fixture["workspace"], execution, digest)
    assert result["is_reproduced"] is True
    assert result["capture"]["optimizer_steps"] == 60
    assert result["capture"]["test_samples"] == 8


@pytest.mark.parametrize("layout", ["entrypoint_class", "torch_builtin"])
def test_author_model_class_location_does_not_require_an_import_preset(tmp_path, layout):
    def change_model(fixture):
        path = fixture["workspace"] / "train.py"
        source = path.read_text(encoding="utf-8")
        if layout == "entrypoint_class":
            source = source.replace("from author_model import SignalModel\n",
                                    (fixture["workspace"] / "author_model.py").read_text(encoding="utf-8"))
        else:
            source = source.replace("model = SignalModel()", "model = torch.nn.Linear(1, 2)")
        path.write_bytes(source.encode("utf-8"))
        fixture["snapshot"]["files"]["train.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
        next(item for item in fixture["proposal"]["citations"] if item["id"] == "script")["quote"] = source

    fixture, plan, digest, _, execution = run_fixture(tmp_path, mutate=change_model)
    result = verify_capture(plan, fixture["workspace"], execution, digest)
    assert result["capture"]["model_source"] == "train.py"
    assert result["capture"]["native_torch_model"] == (layout == "torch_builtin")
    assert result["is_reproduced"] is True


@pytest.mark.parametrize("change", [
    lambda e: e.update(success=False),
    lambda e: e["steps"].pop(),
    lambda e: e["steps"][0].update(timed_out=True),
    lambda e: e["steps"][0].update(cancelled=True),
    lambda e: e["steps"][0].update(exit_code=1),
    lambda e: e["steps"][0].update(executed=False),
    lambda e: e["skipped"].append({"id": "missing"}),
])
def test_native_receipts_never_override_incomplete_processes(captured, change):
    fixture, plan, digest, _, execution = captured
    changed = deepcopy(execution)
    change(changed)
    with pytest.raises(ValueError, match="Every required"):
        verify_capture(plan, fixture["workspace"], changed, digest)


def test_complete_stdout_file_controls_author_metric_match(captured):
    fixture, plan, digest, _, execution = captured
    changed = deepcopy(execution)
    changed["steps"][0]["stdout"] = "truncated live display"
    assert verify_capture(plan, fixture["workspace"], changed, digest)["is_reproduced"] is True
    path = Path(execution["steps"][0]["stdout_path"])
    saved = path.read_bytes()
    try:
        path.write_text("Test Accuracy: 0.500000\n", encoding="utf-8")
        with pytest.raises(ValueError, match="disagree"):
            verify_capture(plan, fixture["workspace"], execution, digest)
    finally:
        path.write_bytes(saved)


def test_changed_training_artifact_rejected(captured):
    fixture, plan, digest, _, execution = captured
    path = fixture["workspace"] / "_autorepro_capture" / "model.pt"
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b"changed")
        with pytest.raises(ValueError, match="artifact changed"):
            verify_capture(plan, fixture["workspace"], execution, digest)
    finally:
        path.write_bytes(saved)


@pytest.mark.parametrize("field,value,match", [
    ("expected_optimizer_steps", 41, "optimizer steps"),
    ("expected_test_samples", 3, "sample count"),
])
def test_runtime_observes_updates_and_full_split_instead_of_trusting_plan(tmp_path, field, value, match):
    environment = native_environment()
    fixture = make_experiment(tmp_path)
    plan = fixture["proposal"]
    plan["capture"][field] = value
    materialize_capture(fixture["workspace"], plan, "a" * 64, snapshot=fixture["snapshot"])
    result = invoke(fixture["workspace"], "capture", environment)
    assert result.returncode != 0
    assert match in result.stderr
    assert not (fixture["workspace"] / "_autorepro_capture" / "capture.json").exists()
