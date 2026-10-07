"""Full-paper protocol evidence tests; no NumPy, author imports, or training."""
import hashlib
import math
import struct
from pathlib import Path

import pytest

from src.repository_profiles import get_profile
from src.repository_validation import verify_dlinear_protocol


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prediction(path, shape=(2785, 96, 7), *, version=1, dtype="<f4", truncate=False):
    # A sparse, zero-valued data section keeps these fixtures independent of ML
    # packages while producing a complete valid numeric NPY file.
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode()
    prefix = b"\x93NUMPY" + bytes([version, 0])
    length = struct.pack("<H" if version == 1 else "<I", len(header) + 1)
    with path.open("wb") as stream:
        stream.write(prefix + length + header + b"\n")
        if not truncate:
            stream.truncate(stream.tell() + math.prod(shape) * int(dtype[-1]))


def training_log(arguments, epochs=7, *, early_stop=True):
    namespace = "Namespace(" + ", ".join(f"{key}={value!r}" for key, value in arguments.items()) + ")"
    lines = [namespace, "Use CPU", ">>>>>>>start training : current_run>>>>>>>>",
             "train 8209", "val 2785", "test 2785"]
    for epoch in range(1, epochs + 1):
        lines.append(f"Epoch: {epoch} cost time: 0.1")
        lines.append(f"Epoch: {epoch}, Steps: 256 | Train Loss: 0.4 Vali Loss: 0.6 Test Loss: 0.38")
    if early_stop:
        lines.extend(["EarlyStopping counter: 3 out of 3", "Early stopping"])
    lines.extend([">>>>>>>testing : current_run<<<<<<<<", "test 2785", "mse:0.384, mae:0.405"])
    return "\n".join(lines) + "\n"


@pytest.fixture
def experiment(tmp_path):
    profile = get_profile("dlinear_etth1_reference")
    root = tmp_path / "repo"
    root.mkdir()
    entry = (
        "import random\nimport numpy as np\nimport torch\n"
        "fix_seed = 2021\nrandom.seed(fix_seed)\n"
        "torch.manual_seed(fix_seed)\nnp.random.seed(fix_seed)\n"
    )
    files = {"run_longExp.py", *profile["repository_map"].values()}
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry if relative == "run_longExp.py" else "# frozen author file\n")
    repository = {
        **profile["repository"], "resolved_sha": profile["repository"]["revision"],
        "path": str(root), "files": {relative: digest(root / relative) for relative in files},
    }
    data_path = root / "dataset/ETTh1.csv"
    data_path.parent.mkdir()
    data_path.write_bytes(b"date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\nfixture,1,2,3,4,5,6,7\n")
    profile["dataset"].update(sha256=digest(data_path), bytes=data_path.stat().st_size)
    dataset = {**profile["dataset"], "path": str(data_path), "verified": True}
    checkpoint = root / "checkpoints/current_run/checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture; never unpickle")
    predictions = root / "results/current_run/pred.npy"
    predictions.parent.mkdir(parents=True)
    write_prediction(predictions)
    arguments = {
        "is_training": 1, "model": "DLinear", "data": "ETTh1", "features": "M",
        "seq_len": 336, "pred_len": 96, "enc_in": 7, "batch_size": 32,
        "learning_rate": 0.005, "train_epochs": 10, "patience": 3, "itr": 1,
        "individual": False, "train_only": False, "use_amp": False,
        "use_gpu": False, "embed": "timeF", "lradj": "type1",
        "root_path": "./dataset/", "data_path": "ETTh1.csv",
    }
    final = {"id": "train_and_eval", "success": True, "exit_code": 0,
             "executed": True, "stdout": training_log(arguments)}
    execution = {"success": True, "executed": True, "final": final,
                 "steps": [{"id": "import_check", "success": True, "exit_code": 0}, final]}
    return {"profile": profile, "workspace": root, "repository": repository,
            "dataset": dataset, "execution": execution, "arguments": arguments,
            "checkpoint": checkpoint, "predictions": predictions}


def verify(experiment):
    return verify_dlinear_protocol(*(experiment[key] for key in
                                    ("profile", "execution", "workspace", "repository", "dataset")))


def assert_failed(experiment, check_name):
    result = verify(experiment)
    assert result["pass"] is False, result
    assert any(check["name"] == check_name and check["pass"] is False for check in result["checks"])
    return result


@pytest.mark.parametrize("epochs,early_stop", [(7, True), (10, False)])
def test_full_author_budget_and_official_early_stop_both_pass(experiment, epochs, early_stop):
    experiment["execution"]["final"]["stdout"] = training_log(experiment["arguments"], epochs, early_stop=early_stop)
    result = verify(experiment)
    assert result["pass"] is True, result["reason"]
    assert result["scope"] == "selected_paper_experiment"
    assert result["epochs_completed"] == epochs
    assert result["artifacts"]["pred.npy"]["shape"] == [2785, 96, 7]
    assert result["artifacts"]["pred.npy"]["sha256"] == digest(experiment["predictions"])
    assert result["artifacts"]["checkpoint.pth"]["sha256"] == digest(experiment["checkpoint"])
    # Protocol verification establishes facts even when the stored profile's
    # historical protocol_verified flag is false.
    assert all(check["pass"] for check in result["checks"])


@pytest.mark.parametrize("epochs,early_stop", [(1, False), (1, True), (7, False), (11, False)])
def test_shortened_or_excessive_schedule_is_not_full_reproduction(experiment, epochs, early_stop):
    experiment["execution"]["final"]["stdout"] = training_log(experiment["arguments"], epochs, early_stop=early_stop)
    assert_failed(experiment, "full_training_trace")


@pytest.mark.parametrize("old,new", [
    ("Epoch: 4, Steps: 256", "Epoch: 5, Steps: 256"),
    ("Epoch: 4, Steps: 256", "Epoch: 4, Steps: 100"),
    ("train 8209", "train 8000"), ("val 2785", "val 2000"),
    ("test 2785", "test 2000"),
    ("EarlyStopping counter: 3 out of 3", "EarlyStopping counter: 2 out of 3"),
])
def test_trace_must_match_real_windows_batches_and_continuous_epochs(experiment, old, new):
    final = experiment["execution"]["final"]
    final["stdout"] = final["stdout"].replace(old, new)
    assert_failed(experiment, "full_training_trace")


@pytest.mark.parametrize("name,value", [
    ("seq_len", 96), ("pred_len", 192), ("features", "MS"),
    ("train_epochs", 1), ("learning_rate", 0.01), ("patience", 10),
    ("batch_size", 16), ("itr", 5), ("individual", True), ("is_training", True),
])
def test_printed_actual_parameters_must_match_the_selected_paper_experiment(experiment, name, value):
    experiment["arguments"][name] = value
    experiment["execution"]["final"]["stdout"] = training_log(experiment["arguments"])
    assert_failed(experiment, "actual_parameters_and_seed")


def test_namespace_ast_never_executes_python_from_log(experiment, tmp_path):
    marker = tmp_path / "must_not_exist"
    final = experiment["execution"]["final"]
    expression = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    final["stdout"] = final["stdout"].replace("seq_len=336", f"seq_len={expression}")
    assert_failed(experiment, "actual_parameters_and_seed")
    assert not marker.exists()


def test_multiple_namespaces_cannot_mix_two_run_configurations(experiment):
    final = experiment["execution"]["final"]
    final["stdout"] += final["stdout"].splitlines()[0] + "\n"
    assert_failed(experiment, "actual_parameters_and_seed")


def test_seed_comes_from_verified_source_not_pythonhashseed_or_profile_only(experiment):
    entry = experiment["workspace"] / "run_longExp.py"
    entry.write_text(entry.read_text().replace("fix_seed = 2021", "fix_seed = 2022"))
    # Even if a new snapshot honestly records the changed file, its seed must
    # still agree with the frozen author experiment.
    experiment["repository"]["files"]["run_longExp.py"] = digest(entry)
    assert_failed(experiment, "actual_parameters_and_seed")


def test_changed_evaluator_source_invalidates_protocol(experiment):
    metrics = experiment["workspace"] / "utils/metrics.py"
    metrics.write_text("# replaced evaluation function\n")
    assert_failed(experiment, "fixed_source")


@pytest.mark.parametrize("problem", ["failed-import", "missing-import", "timeout"])
def test_matching_final_output_cannot_hide_a_failed_required_step(experiment, problem):
    steps = experiment["execution"]["steps"]
    if problem == "failed-import":
        steps[0].update(success=False, exit_code=1)
    elif problem == "missing-import":
        del steps[0]
    else:
        steps[0]["timed_out"] = True
    assert_failed(experiment, "required_steps")


def test_actual_dataset_corruption_is_detected_after_preparation(experiment):
    path = Path(experiment["dataset"]["path"])
    path.write_bytes(path.read_bytes().replace(b",1,", b",9,"))
    assert_failed(experiment, "real_dataset")


@pytest.mark.parametrize("problem", ["missing", "empty", "duplicate"])
def test_checkpoint_must_be_unique_and_nonempty(experiment, problem):
    path = experiment["checkpoint"]
    if problem == "missing":
        path.unlink()
    elif problem == "empty":
        path.write_bytes(b"")
    else:
        (experiment["workspace"] / "checkpoint.pth").write_bytes(b"stale checkpoint")
    assert_failed(experiment, "checkpoint_and_predictions")


@pytest.mark.parametrize("shape", [(2785, 96, 1), (100, 96, 7), (2785, 192, 7)])
def test_prediction_shape_must_cover_every_official_test_window(experiment, shape):
    write_prediction(experiment["predictions"], shape)
    assert_failed(experiment, "checkpoint_and_predictions")


def test_npy_header_alone_is_not_a_complete_prediction_artifact(experiment):
    write_prediction(experiment["predictions"], truncate=True)
    assert_failed(experiment, "checkpoint_and_predictions")


@pytest.mark.parametrize("version", [2, 3])
def test_standard_library_npy_reader_handles_supported_header_versions(experiment, version):
    write_prediction(experiment["predictions"], version=version)
    assert verify(experiment)["pass"] is True
