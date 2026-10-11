"""Fixed paper protocol and independently measured ReZero artifact contracts."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import rezero_runtime as runtime
from src.experiments.evaluate_rezero import metrics_from_logits


@pytest.fixture
def cfg():
    return {"parameters": deepcopy(runtime.REFERENCE_PARAMETERS), "spec_sha256": "a" * 64}


@pytest.fixture
def completed(cfg):
    history, batches = [], []
    for epoch in range(1, 46):
        test_correct = 9400 if epoch == 44 else 9391
        history.append({"epoch": epoch,
            "train": {"samples": 50000, "batches": 98, "correct": 98,
                      "accuracy_pct": 100.0 * 98 / 50000, "loss": .2 * 98 / 97,
                      "sample_cross_entropy": .2},
            "test": {"samples": 10000, "batches": 20, "correct": test_correct,
                     "accuracy_pct": 100.0 * test_correct / 10000,
                     "loss": .25, "sample_cross_entropy": .24}})
        for batch in range(1, 99):
            lr, momentum = runtime.reference_schedule((epoch - 1) * 98 + batch - 1)
            batches.append({"epoch": epoch, "batch": batch,
                            "samples": 512 if batch < 98 else 336, "correct": 1,
                            "loss": .2, "lr": lr, "momentum": momentum})
    return {"status": "completed", "epochs_completed": 45, "steps_completed": 4410,
            "parameters": cfg["parameters"], "spec_sha256": cfg["spec_sha256"],
            "history": history, "batch_history": batches,
            "model_parameters": 11171154, "numerics": dict(runtime.NUMERICS),
            "optimizer_groups": {"ordinary": {"names": ["conv.weight"]}},
            "optimizer_state": {"scheduler_last_step": 4410,
                                "sgd_parameter_tensors": 1, "sgd_momentum_buffers": 1,
                                "adagrad_steps": {name: 4410 for name in runtime.AUTHOR_RESIDUAL_NAMES}},
            "best_epoch": 44, "best_accuracy_pct": 94.0}


@pytest.mark.parametrize("key,value", [("epochs", 1), ("batch_size", 256),
    ("precision", "float16"), ("resweight_lr", .01), ("seed", 0),
    ("device", "cpu"), ("num_workers", 0), ("init_lr", True)])
def test_reference_protocol_cannot_be_shortened_or_retuned(cfg, key, value):
    runtime.validate_protocol(cfg["parameters"])
    cfg["parameters"][key] = value
    with pytest.raises(ValueError, match=key):
        runtime.validate_protocol(cfg["parameters"])


def test_author_schedule_key_points_are_exact():
    assert runtime.reference_schedule(0) == (.032, .95)
    assert runtime.reference_schedule(441) == pytest.approx((1.2, .85))
    assert runtime.reference_schedule(3969) == pytest.approx((.032, .95))
    assert runtime.reference_schedule(4410) == pytest.approx((.001, .95))
    with pytest.raises(ValueError):
        runtime.reference_schedule(4411)


def test_full_45_epoch_gate_accepts_original_best_checkpoint_rule(cfg, completed):
    assert runtime.validate_training_record(completed, cfg) == 4410
    # The author's checkpoint is the first strict maximum, not the last epoch.
    completed["history"][44]["test"].update(correct=9400, accuracy_pct=94.0)
    assert runtime.validate_training_record(completed, cfg) == 4410


@pytest.mark.parametrize("field,value", [("status", "running"),
    ("epochs_completed", 44), ("steps_completed", 4409),
    ("best_epoch", 45), ("best_accuracy_pct", 99.0), ("spec_sha256", "b" * 64)])
def test_incomplete_or_invented_training_cannot_pass(cfg, completed, field, value):
    completed[field] = value
    with pytest.raises(ValueError):
        runtime.validate_training_record(completed, cfg)


@pytest.mark.parametrize("split,field,value", [("train", "samples", 1000),
    ("test", "samples", 100), ("test", "accuracy_pct", 99.99),
    ("train", "sample_cross_entropy", float("nan"))])
def test_real_dataset_coverage_and_measured_counts_are_required(cfg, completed, split, field, value):
    completed["history"][0][split][field] = value
    with pytest.raises(ValueError):
        runtime.validate_training_record(completed, cfg)


@pytest.mark.parametrize("field,value", [("samples", 1), ("lr", .001),
    ("momentum", .5), ("batch", 98), ("loss", float("inf"))])
def test_optimizer_history_cannot_hide_changed_batches_or_schedule(cfg, completed, field, value):
    completed["batch_history"][0][field] = value
    with pytest.raises(ValueError):
        runtime.validate_training_record(completed, cfg)


@pytest.mark.parametrize("change", ["scheduler", "residual_steps", "momentum", "tf32", "model"])
def test_completion_requires_actual_optimizer_state_and_full_precision(cfg, completed, change):
    if change == "scheduler":
        completed["optimizer_state"]["scheduler_last_step"] = 4409
    elif change == "residual_steps":
        completed["optimizer_state"]["adagrad_steps"]["layer1.0.resweight"] = 4409
    elif change == "momentum":
        completed["optimizer_state"]["sgd_momentum_buffers"] = 0
    elif change == "tf32":
        completed["numerics"]["matmul_allow_tf32"] = True
    else:
        completed["model_parameters"] = 100
    with pytest.raises(ValueError):
        runtime.validate_training_record(completed, cfg)


def test_unknown_protocol_parameters_are_rejected(cfg):
    cfg["parameters"]["subset_samples"] = 1000
    with pytest.raises(ValueError, match="keys"):
        runtime.validate_protocol(cfg["parameters"])


class TransformProbe:
    def __getattr__(self, name):
        return lambda *args, **kwargs: (name, args, kwargs)


def test_test_split_never_receives_training_augmentation():
    train, test = runtime.build_transforms(TransformProbe())
    assert [item[0] for item in train[1][0]] == ["RandomCrop", "ColorJitter",
        "RandomRotation", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert [item[0] for item in test[1][0]] == ["ToTensor", "Normalize"]
    assert test[1][0][-1][2] == {"mean": runtime.NORMALIZE_MEAN, "std": runtime.NORMALIZE_STD}


def test_dataset_loader_uses_full_official_splits_without_network():
    calls = []
    def cifar(**kwargs):
        calls.append(kwargs)
        return range(50000 if kwargs["train"] else 10000)
    runtime.build_datasets(SimpleNamespace(CIFAR10=cifar), TransformProbe())
    assert [call["train"] for call in calls] == [True, False]
    assert all(call["root"] == "dataset" and call["download"] is False for call in calls)
    with pytest.raises(ValueError, match="50000"):
        runtime.build_datasets(SimpleNamespace(CIFAR10=lambda **k: range(2)), TransformProbe())


def test_data_identity_requires_both_official_md5_and_frozen_sha(tmp_path, monkeypatch):
    folder = tmp_path / "cifar-10-batches-py"
    folder.mkdir()
    files, md5s = {}, {}
    for name in runtime.CIFAR_FILE_MD5:
        raw = (name + "-test-fixture").encode()
        (folder / name).write_bytes(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
        md5s[name] = hashlib.md5(raw).hexdigest()
    cfg = {"dataset_sha256": "archive", "dataset_files_sha256": files,
           "dataset_manifest_sha256": hashlib.sha256(json.dumps(files, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()}
    with pytest.raises(ValueError, match="Official"):
        runtime.dataset_fingerprints(cfg, tmp_path)
    monkeypatch.setattr(runtime, "CIFAR_FILE_MD5", md5s)
    record = runtime.dataset_fingerprints(cfg, tmp_path)
    assert record["files_sha256"] == files
    assert record["train_samples"] == 50000 and record["test_samples"] == 10000
    cfg["dataset_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest"):
        runtime.dataset_fingerprints(cfg, tmp_path)
    cfg["dataset_files_sha256"] = {**files, "test_batch": "changed"}
    with pytest.raises(ValueError, match="SHA-256"):
        runtime.dataset_fingerprints(cfg, tmp_path)


def test_accuracy_is_calculated_from_actual_logits_not_a_preset_target():
    logits = np.zeros((10000, 10), dtype=np.float32)
    targets = np.zeros(10000, dtype=np.int64)
    logits[:, 0] = 1.0
    logits[9399:, 1] = 2.0
    metrics, correct, _ = metrics_from_logits(logits, targets)
    assert correct == 9399 and metrics["top1_accuracy_pct"] == 93.99
    assert metrics["top1_accuracy_pct"] < runtime.TARGET_ACCURACY_PCT
    assert metrics["cross_entropy"] > 0
    logits[9399, 1] = 0
    assert metrics_from_logits(logits, targets)[0]["top1_accuracy_pct"] == 94.0


@pytest.mark.parametrize("case", ["subset", "nan", "bad_labels", "wrong_classes"])
def test_independent_metrics_reject_invalid_predictions(case):
    logits, targets = np.zeros((10000, 10)), np.zeros(10000, dtype=np.int64)
    if case == "subset":
        logits, targets = logits[:10], targets[:10]
    elif case == "nan":
        logits[0, 0] = np.nan
    elif case == "bad_labels":
        targets[0] = 10
    else:
        logits = logits[:, :9]
    with pytest.raises(ValueError):
        metrics_from_logits(logits, targets)


def test_optimizer_groups_preserve_author_two_optimizer_algorithm(cfg):
    torch = pytest.importorskip("torch")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.resweights = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(8)])
        def named_parameters(self, **kwargs):
            yield "weight", self.weight
            for i, module in enumerate(self.resweights):
                yield f"layer.{i}.resweight", module.weight
    class SchedulerProbe:
        def __init__(self, optimizer, **kwargs):
            self.kwargs = kwargs
            optimizer.param_groups[0].update(lr=kwargs["init_lr"], momentum=.95)
    optimizer, residual, scheduler, groups = runtime.build_optimizers(
        torch, Model(), SchedulerProbe, cfg["parameters"], 98)
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(residual, torch.optim.Adagrad)
    assert optimizer.param_groups[0]["weight_decay"] == .0002
    assert optimizer.param_groups[0]["momentum"] == .95
    assert residual.param_groups[0]["lr"] == .1
    assert residual.param_groups[0]["weight_decay"] == 0
    assert len(groups["residual"]["names"]) == 8
    assert scheduler.kwargs == {"num_steps": 4410, "init_lr": .032,
        "point_1": (.1, 1.2), "point_2": (.9, .032), "end_lr": .001,
        "momentum_range": (.85, .95), "exp_decay": False, "param_group": 0}


def test_cpu_inference_exports_actual_model_logits_and_labels(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(runtime, "TEST_SAMPLES", 4)
    model = torch.nn.Linear(2, 10)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.bias[3] = 2
    inputs, targets = torch.zeros(4, 2), torch.tensor([3, 3, 0, 1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=2)
    measured, logits, labels = runtime.infer_test(
        torch, loader, model, torch.nn.CrossEntropyLoss(), "cpu", export=True)
    metrics, correct, predicted = metrics_from_logits(logits, labels, expected_samples=4)
    assert correct == measured["correct"] == 2
    assert metrics["top1_accuracy_pct"] == measured["accuracy_pct"] == 50.0
    assert list(predicted) == [3, 3, 3, 3]
    assert np.array_equal(labels, targets.numpy())
    assert metrics["cross_entropy"] == pytest.approx(measured["sample_cross_entropy"], rel=1e-6)
    monkeypatch.setattr(runtime, "TEST_SAMPLES", 10000)
    with pytest.raises(ValueError, match="10000"):
        runtime.infer_test(torch, loader, model, torch.nn.CrossEntropyLoss(), "cpu")


def test_cpu_training_loop_updates_both_author_optimizer_groups(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(runtime, "TRAIN_SAMPLES", 4)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 10))
            self.resweight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return x @ self.weight + self.resweight * torch.arange(10, dtype=x.dtype)[None, :]
    model = Model()
    optimizer = torch.optim.SGD([model.weight], lr=.032, momentum=.95, weight_decay=.0002)
    residual = torch.optim.Adagrad([model.resweight], lr=.1)
    scheduler = SimpleNamespace(step=lambda: None)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.ones(4, 2), torch.tensor([9, 9, 9, 9])), batch_size=2)
    measured, history = runtime.train_epoch(torch, loader, model,
        torch.nn.CrossEntropyLoss(), optimizer, residual, scheduler, "cpu", 1)
    assert measured["samples"] == 4 and measured["batches"] == len(history) == 2
    assert not torch.equal(model.weight.detach(), torch.zeros_like(model.weight))
    assert model.resweight.item() > 0
    assert all(step["samples"] == 2 and step["lr"] == .032 for step in history)


def test_optimizer_evidence_reads_actual_adagrad_counters_and_sgd_state():
    torch = pytest.importorskip("torch")
    ordinary = torch.nn.Parameter(torch.ones(1))
    residual = {name: torch.nn.Parameter(torch.zeros(1)) for name in runtime.AUTHOR_RESIDUAL_NAMES}
    model = SimpleNamespace(named_parameters=lambda: iter([("ordinary.weight", ordinary), *residual.items()]))
    optimizer = torch.optim.SGD([ordinary], lr=.032, momentum=.95)
    adagrad = torch.optim.Adagrad(list(residual.values()), lr=.1)
    before = runtime.optimizer_evidence(model, optimizer, adagrad, SimpleNamespace(last_step=0))
    assert before["sgd_momentum_buffers"] == 0
    assert set(before["adagrad_steps"].values()) == {0}
    for _ in range(2):
        optimizer.zero_grad()
        adagrad.zero_grad()
        (ordinary.sum() + sum(parameter.sum() for parameter in residual.values())).backward()
        optimizer.step()
        adagrad.step()
    after = runtime.optimizer_evidence(model, optimizer, adagrad, SimpleNamespace(last_step=2))
    assert after["scheduler_last_step"] == 2
    assert after["sgd_momentum_buffers"] == after["sgd_parameter_tensors"] == 1
    assert after["adagrad_steps"] == {name: 2 for name in runtime.AUTHOR_RESIDUAL_NAMES}


def test_runtime_rejects_historical_or_unpinned_torch_before_training():
    torch = SimpleNamespace(__version__="1.2.0")
    with pytest.raises(RuntimeError, match="frozen torch"):
        runtime.configure_runtime(torch, SimpleNamespace(__version__="0.20.1+cu121"))


def test_execution_evidence_rejects_mixed_precision_and_tf32(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    model = torch.nn.Linear(2, 10)
    assert runtime.execution_numerics(torch, model) == runtime.NUMERICS
    with pytest.raises(ValueError, match="FP32"):
        runtime.execution_numerics(torch, model.double())
    model.float()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(ValueError, match="FP32"):
        runtime.execution_numerics(torch, model)
