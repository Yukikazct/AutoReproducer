"""Synthetic unit fixtures test evidence rejection, never real reproduction claims."""
import copy
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

import src.rezero_adapter as module
from src.method_adapters import digest, read_json, write_json
from src.rezero_adapter import ReZeroAdapter
from src.experiments.rezero_runtime import AUTHOR_RESIDUAL_NAMES, NUMERICS


@pytest.fixture
def unit_files(monkeypatch):
    """Small bytes replace canonical hashes only inside these isolated tests."""
    labels = np.arange(10000, dtype=np.int64) % 10
    files = {name: (pickle.dumps({"labels": labels.tolist()}) if name == "test_batch"
                    else f"synthetic unit CIFAR file: {name}".encode())
             for name in module.CIFAR_FILE_MD5}
    monkeypatch.setattr(module, "CIFAR_FILE_MD5", {name: hashlib.md5(raw).hexdigest()
                                                 for name, raw in files.items()})
    sources = {name: (b'{"cells": [{"source": ["author code"], "outputs": [{"text": ["historic accuracy"]}]}]}'
                     if name.endswith(".ipynb") else f"# synthetic unit source {name}\n".encode())
               for name in module.SOURCE_SHA256}
    monkeypatch.setattr(module, "SOURCE_SHA256", {name: hashlib.sha256(raw).hexdigest()
                                                 for name, raw in sources.items()})
    return files, sources, labels


def profile_for_unit_test(sources, archive=b"placeholder"):
    return {"id": "rezero_cifar10_reference", "adapter_id": "rezero",
            "parameters": copy.deepcopy(module.REFERENCE_PARAMETERS),
            "paper": {"reference_source": "https://github.com/tbachlechner/ReZero-Superconvergence/blob/pinned/train_faster_superc.py",
                      "required_metrics": ["top1_accuracy_pct", "cross_entropy"]},
            "repository": {"url": "https://github.com/tbachlechner/ReZero-Superconvergence", "revision": "1" * 40},
            "required_files": list(sources), "source_sha256s": dict(module.SOURCE_SHA256),
            "dataset": {"name": "unit-cifar", "sha256": hashlib.sha256(archive).hexdigest(),
                        "md5": hashlib.md5(archive).hexdigest(), "bytes": len(archive),
                        "target": "cifar-10-python.tar.gz", "url": "https://example.org/unit-cifar",
                        "mirrors": ["https://example.org/unit-cifar-mirror"]}}


def tar_bytes(files, extra=()):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        directory = tarfile.TarInfo("cifar-10-batches-py")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, raw in files.items():
            member = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        for member, raw in extra:
            archive.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


def materialized(tmp_path, unit_files):
    files, sources, labels = unit_files
    root = tmp_path / "repo"
    root.mkdir()
    for name, raw in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "README.md").write_text("Synthetic author documentation", encoding="utf-8")
    folder = root / module.DATASET_FOLDER
    folder.mkdir(parents=True)
    for name, raw in files.items():
        (folder / name).write_bytes(raw)
    profile = profile_for_unit_test(sources)
    hashes = module.checked_dataset_files(root)
    write_json(root / "dataset_manifest.json", {"verified": True,
               "archive_sha256": profile["dataset"]["sha256"], "files_sha256": hashes,
               "manifest_sha256": module.dataset_manifest_hash(hashes)})
    snapshot = {"files": {name: digest(root / name) for name in sources}}
    manifest = ReZeroAdapter().materialize(root, profile, "unit-spec")
    return root, profile, snapshot, manifest, labels


def valid_evidence(tmp_path, unit_files, accuracy=95.0):
    root, profile, snapshot, manifest, labels = materialized(tmp_path, unit_files)
    out = root / "artifacts"
    out.mkdir()
    logits = np.full((10000, 10), -2, dtype=np.float32)
    choices = labels.copy()
    wrong = int(round(10000 * (1 - accuracy / 100)))
    choices[:wrong] = (choices[:wrong] + 1) % 10
    logits[np.arange(10000), choices] = 2
    for name, array in (("prediction.npy", logits), ("prediction_recomputed.npy", logits),
                        ("targets.npy", labels), ("targets_recomputed.npy", labels)):
        np.save(out / name, array, allow_pickle=False)
    (out / "checkpoint.pt").write_bytes(b"synthetic checkpoint evidence for rejection tests only")
    cfg = read_json(root / "experiment.json")
    files = cfg["dataset_files_sha256"]
    dataset = {"archive_sha256": cfg["dataset_sha256"], "files_sha256": files,
               "manifest_sha256": cfg["dataset_manifest_sha256"], "train_samples": 50000,
               "test_samples": 10000, "split": "official_train_and_test"}
    history, batches = [], []
    test_correct = int(accuracy * 100)
    for epoch in range(1, 46):
        for index in range(98):
            samples = 512 if index < 97 else 336
            lr, momentum = module._schedule((epoch - 1) * 98 + index, profile["parameters"])
            batches.append({"epoch": epoch, "batch": index + 1, "samples": samples,
                            "correct": samples // 2, "loss": 1., "lr": lr, "momentum": momentum})
        lr, momentum = module._schedule(epoch * 98, profile["parameters"])
        history.append({"epoch": epoch,
                        "train": {"samples": 50000, "batches": 98, "correct": 25000,
                                  "loss": 98 / 97, "sample_cross_entropy": 1., "accuracy_pct": 50.},
                        "test": {"samples": 10000, "batches": 20, "correct": test_correct,
                                 "loss": .4, "sample_cross_entropy": .38, "accuracy_pct": accuracy},
                        "lr_after_epoch": lr, "momentum_after_epoch": momentum,
                        "epoch_elapsed_s": 1., "gpu_peak_bytes": 1024})
    training = {"status": "completed", "parameters": profile["parameters"], "spec_sha256": "unit-spec",
                "dataset": dataset, "seed": 6892, "device": "cuda", "precision": "float32",
                "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "cuda": "12.1",
                "epochs_completed": 45, "steps_completed": 4410, "history": history, "batch_history": batches,
                "best_epoch": 1, "best_accuracy_pct": accuracy, "training_elapsed_s": 45.,
                "model_parameters": 11171154, "numerics": dict(NUMERICS),
                "optimizer_state": {"scheduler_last_step": 4410,
                                    "sgd_parameter_tensors": 1, "sgd_momentum_buffers": 1,
                                    "adagrad_steps": {name: 4410 for name in AUTHOR_RESIDUAL_NAMES}},
                "best_test_export": {"correct": test_correct},
                "source_files_sha256": {name: digest(root / name) for name in
                                        ("models/rezero_preact_resnet.py", "customonecycle.py")},
                "optimizer_groups": {"ordinary": {"optimizer": "SGD", "weight_decay": .0002,
                                                  "nesterov": False, "names": ["conv.weight"]},
                                     "residual": {"optimizer": "Adagrad", "lr": .1, "weight_decay": 0.,
                                                  "initial_accumulator_value": 0., "eps": 1e-10,
                                                  "names": list(AUTHOR_RESIDUAL_NAMES)}}}
    for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("prediction.npy", "prediction_sha256"),
                      ("targets.npy", "targets_sha256")):
        training[key] = digest(out / name)
    from src.experiments.evaluate_rezero import metrics_from_logits
    values, correct, _ = metrics_from_logits(logits, labels)
    metrics = {"pass": True, "protocol_pass": True, "independent_metrics_pass": True,
               "quality_pass": accuracy >= 94, "target_accuracy_pct": 94., "metrics": values,
               "split": "test", "samples": 10000, "correct": correct, "numpy_correct": correct,
               "torch_correct": correct, "epochs_completed": 45, "steps_completed": 4410,
               "best_epoch": 1, "spec_sha256": "unit-spec", "dataset": dataset,
               **{key: training[key] for key in ("numerics", "torch", "torchvision", "cuda")},
               **{key: training[key] for key in ("checkpoint_sha256", "prediction_sha256", "targets_sha256")},
               "recomputed_prediction_sha256": digest(out / "prediction_recomputed.npy"),
               "recomputed_targets_sha256": digest(out / "targets_recomputed.npy")}
    write_json(out / "training.json", training)
    write_json(out / "metrics_test.json", metrics)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    imported = {"torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "cuda": "12.1",
                "cuda_available": True,
                "modules": {name: str(dependencies / name / "__init__.py") for name in
                            ("torch", "torchvision", "numpy", "scipy", "PIL", "matplotlib")},
                "author_components": {"models.rezero_preact_resnet": str(root / "models/rezero_preact_resnet.py"),
                                      "customonecycle": str(root / "customonecycle.py")}}
    write_json(root / "import_provenance.json", imported)
    train_summary = {key: training[key] for key in
                     ("epochs_completed", "steps_completed", "best_accuracy_pct", "training_elapsed_s")}
    steps = [{"id": name, "success": True, "exit_code": 0, "executed": True} for name in
             ("import_check", "train", "evaluate")]
    steps[1]["stdout"] = json.dumps(train_summary)
    steps[2]["stdout"] = json.dumps(metrics)
    execution = {"success": True, "executed": True, "steps": steps, "final": dict(steps[2]),
                 "environment": {"dependencies_path": str(dependencies)}}
    return root, profile, snapshot, manifest, execution


def verify_unit(fixture):
    root, profile, snapshot, manifest, execution = fixture
    return ReZeroAdapter().verify(profile, execution, root, snapshot, manifest, "unit-spec")


def sync_metrics(fixture, changes):
    root, _, _, _, execution = fixture
    metrics = read_json(root / "artifacts/metrics_test.json")
    metrics.update(changes)
    write_json(root / "artifacts/metrics_test.json", metrics)
    execution["steps"][-1]["stdout"] = json.dumps(metrics)
    execution["final"]["stdout"] = json.dumps(metrics)


def test_dataset_preparation_checks_identity_before_extracting(tmp_path, monkeypatch, unit_files):
    files, sources, _ = unit_files
    archive = tar_bytes(files)
    profile = profile_for_unit_test(sources, archive)
    calls = []
    def download(url, **kwargs):
        calls.append(url)
        assert kwargs["max_bytes"] == len(archive)
        return archive
    monkeypatch.setattr(module, "download_bytes", download)
    result = ReZeroAdapter().prepare_dataset(tmp_path / "cache", profile, tmp_path / "repo")
    assert result["verified"] is True
    assert result["canonical_url"] == profile["dataset"]["url"]
    assert result["download_source"] == profile["dataset"]["url"]
    assert result["cache_hit"] is False
    assert result["train_samples"] == 50000 and result["test_samples"] == 10000
    assert result["files_sha256"] == {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    assert len(calls) == 1
    # Reuse of verified bytes works without any subsequent networking.
    monkeypatch.setattr(module, "download_bytes", lambda *args, **kwargs: pytest.fail("cache must be offline"))
    cached = ReZeroAdapter().prepare_dataset(tmp_path / "cache", profile, tmp_path / "repo2", offline=True)
    assert cached["verified"] and cached["cache_hit"] is True
    assert cached["download_source"] is None
    assert cached["canonical_url"] == profile["dataset"]["url"]


def test_transport_error_uses_declared_mirror(tmp_path, monkeypatch, unit_files):
    files, sources, _ = unit_files
    archive = tar_bytes(files)
    profile = profile_for_unit_test(sources, archive)
    calls = []
    def download(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("transport failure")
        return archive
    monkeypatch.setattr(module, "download_bytes", download)
    result = ReZeroAdapter().prepare_dataset(tmp_path / "cache", profile, tmp_path / "repo")
    assert result["download_source"] == profile["dataset"]["mirrors"][0]
    assert result["canonical_url"] == profile["dataset"]["url"]
    assert result["cache_hit"] is False
    assert calls == [profile["dataset"]["url"], profile["dataset"]["mirrors"][0]]
    monkeypatch.setattr(module, "download_bytes", lambda *args, **kwargs: pytest.fail("verified mirror cache must be reused"))
    cached = ReZeroAdapter().prepare_dataset(tmp_path / "cache", profile, tmp_path / "repo2", offline=True)
    assert cached["cache_hit"] is True and cached["download_source"] is None
    assert cached["canonical_url"] == profile["dataset"]["url"]
    assert cached["archive_sha256"] == result["archive_sha256"]


def test_wrong_download_is_never_published_or_extracted(tmp_path, monkeypatch, unit_files):
    _, sources, _ = unit_files
    profile = profile_for_unit_test(sources)
    monkeypatch.setattr(module, "download_bytes", lambda *args, **kwargs: b"bad")
    with pytest.raises(ValueError, match="identity"):
        ReZeroAdapter().prepare_dataset(tmp_path / "cache", profile, tmp_path / "repo")
    assert not (tmp_path / "repo").exists()
    assert not list(tmp_path.rglob("cifar-10-python.tar.gz"))


def test_offline_missing_archive_never_downloads(tmp_path, monkeypatch, unit_files):
    _, sources, _ = unit_files
    monkeypatch.setattr(module, "download_bytes", lambda *args, **kwargs: pytest.fail("offline download"))
    with pytest.raises(RuntimeError, match="Offline"):
        ReZeroAdapter().prepare_dataset(tmp_path, profile_for_unit_test(sources), tmp_path / "repo", offline=True)


@pytest.mark.parametrize("kind", ["escape", "link", "duplicate", "unexpected", "missing"])
def test_archive_rejects_noncanonical_paths_before_any_write(tmp_path, unit_files, kind):
    files, _, _ = unit_files
    extra = []
    if kind == "missing":
        files = {name: raw for name, raw in files.items() if name != "test_batch"}
    else:
        name = {"escape": "../escaped", "link": "cifar-10-batches-py/link",
                "duplicate": "cifar-10-batches-py/data_batch_1", "unexpected": "cifar-10-batches-py/runner.py"}[kind]
        member = tarfile.TarInfo(name)
        if kind == "link":
            member.type, member.linkname = tarfile.SYMTYPE, "../../escaped"
        extra = [(member, b"")]
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(tar_bytes(files, extra))
    with pytest.raises(ValueError):
        module._extract_cifar_archive(archive, tmp_path / "workspace")
    assert not (tmp_path / "workspace").exists()
    assert not (tmp_path / "escaped").exists()


def test_materialize_preserves_author_source_and_freezes_local_dataset(tmp_path, unit_files):
    root, profile, _, manifest, _ = materialized(tmp_path, unit_files)
    assert all((root / name).read_bytes() == raw for name, raw in unit_files[1].items())
    cfg = read_json(root / "experiment.json")
    assert cfg["parameters"] == module.REFERENCE_PARAMETERS
    assert cfg["dataset_sha256"] == profile["dataset"]["sha256"]
    assert cfg["dataset_manifest_sha256"] == module.dataset_manifest_hash(cfg["dataset_files_sha256"])
    assert {"run_experiment.py", "evaluate.py", "experiment.json", "dataset_manifest.json"} <= set(manifest["files"])
    assert (root / "run_experiment.py").read_bytes() == (module.RUNTIMES / "rezero_runtime.py").read_bytes()


def test_source_evidence_excludes_historic_notebook_results(tmp_path, unit_files):
    root, profile, _, _, _ = materialized(tmp_path, unit_files)
    sources = ReZeroAdapter().public_sources(root, profile)
    assert any(source["locator"] == "train_faster_superc.py" for source in sources)
    notebook = next(source for source in sources if source["locator"].endswith(".ipynb"))
    assert notebook["content"] == "author code"
    assert "historic accuracy" not in notebook["content"]


def test_author_source_checks_tolerate_git_eol_conversion_but_not_content_changes(tmp_path, unit_files):
    _, sources, _ = unit_files
    profile = profile_for_unit_test(sources)
    for name, source in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.replace(b"\n", b"\r\n"))
    assert module._verify_source_files(tmp_path, profile) == module.SOURCE_SHA256
    with (tmp_path / "customonecycle.py").open("ab") as stream:
        stream.write(b"# changed scheduler\r\n")
    with pytest.raises(ValueError, match="source checksum"):
        module._verify_source_files(tmp_path, profile)


def test_steps_keep_full_protocol_and_independent_evaluation(unit_files):
    profile = profile_for_unit_test(unit_files[1])
    steps = ReZeroAdapter().steps(profile)
    assert [step["id"] for step in steps] == ["import_check", "train", "evaluate"]
    assert steps[1]["timeout_s"] == 7200
    assert steps[2]["argv"] == ["python", "-u", "evaluate.py", "test"]
    assert steps[2]["depends_on"] == ["train"]
    assert {item["path"] for item in steps[2]["requires"]} == {
        "artifacts/training.json", "artifacts/checkpoint.pt", "artifacts/prediction.npy", "artifacts/targets.npy"}
    assert len(ReZeroAdapter().steps(profile, train=False)) == 1
    with pytest.raises(ValueError, match="test split"):
        ReZeroAdapter().steps(profile, split="validation")


@pytest.mark.parametrize("change", [{"epochs": 2}, {"batch_size": 64}, {"device": "cpu"},
                                    {"precision": "float16"}, {"seed": 1}, {"epochs": True}])
def test_shortened_or_changed_protocol_is_rejected(unit_files, change):
    profile = profile_for_unit_test(unit_files[1])
    profile["parameters"].update(change)
    with pytest.raises(ValueError, match="parameter"):
        ReZeroAdapter().steps(profile)


@pytest.mark.parametrize("accuracy,expected", [(94.0, True), (95.0, True), (93.99, False)])
def test_acceptance_requires_full_protocol_and_fixed_paper_reference(tmp_path, unit_files, accuracy, expected):
    result = verify_unit(valid_evidence(tmp_path, unit_files, accuracy))
    assert result["is_reproduced"] is expected
    assert result["status"] == ("reproduced" if expected else "reference_not_met")
    assert result["scope"] == "selected_paper_experiment"
    assert result["optimization_eligible"] is False
    assert result["protocol_pass"] is True and result["independent_metrics_pass"] is True
    assert result["metrics_comparison"]["actual"]["top1_accuracy_pct"] == accuracy
    assert result["training_summary"]["epochs_completed"] == 45


@pytest.mark.parametrize("change", ["epochs", "steps", "batch_count", "batch_order", "learning_rate",
                                    "momentum", "count", "loss", "best_epoch", "optimizer", "source"])
def test_forged_or_partial_training_never_passes(tmp_path, unit_files, change):
    fixture = valid_evidence(tmp_path, unit_files)
    path = fixture[0] / "artifacts/training.json"
    training = read_json(path)
    if change == "epochs": training["epochs_completed"] = 44
    elif change == "steps": training["steps_completed"] = 4312
    elif change == "batch_count": training["batch_history"][-1]["samples"] = 128
    elif change == "batch_order": training["batch_history"][1]["batch"] = 1
    elif change == "learning_rate": training["batch_history"][100]["lr"] = .03
    elif change == "momentum": training["batch_history"][100]["momentum"] = .99
    elif change == "count": training["batch_history"][100]["correct"] = 0
    elif change == "loss": training["batch_history"][100]["loss"] = .01
    elif change == "best_epoch": training["best_epoch"] = 45
    elif change == "optimizer": training["optimizer_groups"]["residual"]["optimizer"] = "Adam"
    elif change == "source": training["source_files_sha256"]["customonecycle.py"] = "bad"
    write_json(path, training)
    with pytest.raises(ValueError):
        verify_unit(fixture)


@pytest.mark.parametrize("name", ["checkpoint.pt", "prediction.npy", "targets.npy", "prediction_recomputed.npy",
                                  "targets_recomputed.npy"])
def test_artifact_tampering_is_rejected(tmp_path, unit_files, name):
    fixture = valid_evidence(tmp_path, unit_files)
    with (fixture[0] / "artifacts" / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_unit(fixture)


def test_even_self_consistent_reported_metrics_are_recomputed(tmp_path, unit_files):
    fixture = valid_evidence(tmp_path, unit_files)
    sync_metrics(fixture, {"metrics": {"top1_accuracy_pct": 99., "cross_entropy": .01}})
    with pytest.raises(ValueError, match="measurement"):
        verify_unit(fixture)


def test_low_accuracy_cannot_be_declared_quality_pass(tmp_path, unit_files):
    fixture = valid_evidence(tmp_path, unit_files, 93.)
    sync_metrics(fixture, {"quality_pass": True})
    with pytest.raises(ValueError, match="quality verdict"):
        verify_unit(fixture)


@pytest.mark.parametrize("change", [{"torch": "1.2.0"},
                                    {"numerics": {**NUMERICS, "matmul_allow_tf32": True}}])
def test_independent_evaluation_requires_the_same_precision_and_runtime(tmp_path, unit_files, change):
    fixture = valid_evidence(tmp_path, unit_files)
    sync_metrics(fixture, change)
    with pytest.raises(ValueError, match="evaluation precision or runtime"):
        verify_unit(fixture)


@pytest.mark.parametrize("change", ["missing_step", "failed_step", "versions", "external_model", "external_dependency"])
def test_execution_and_import_provenance_are_required(tmp_path, unit_files, change):
    fixture = valid_evidence(tmp_path, unit_files)
    root, _, _, _, execution = fixture
    if change == "missing_step":
        execution["steps"].pop(1)
    elif change == "failed_step":
        execution["steps"][1]["success"] = False
    else:
        imported = read_json(root / "import_provenance.json")
        if change == "versions": imported["torchvision"] = "0.19.0"
        elif change == "external_model": imported["author_components"]["models.rezero_preact_resnet"] = str(tmp_path / "external.py")
        elif change == "external_dependency": imported["modules"]["numpy"] = str(tmp_path / "numpy/__init__.py")
        write_json(root / "import_provenance.json", imported)
    with pytest.raises(ValueError):
        verify_unit(fixture)


@pytest.mark.parametrize("file", ["train_faster_superc.py", "customonecycle.py", "experiment.json",
                                  "dataset/cifar-10-batches-py/test_batch"])
def test_source_dataset_and_protocol_changes_cannot_be_hidden(tmp_path, unit_files, file):
    fixture = valid_evidence(tmp_path, unit_files)
    with (fixture[0] / file).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_unit(fixture)
