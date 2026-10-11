"""Frozen ReZero CIFAR-10 paper experiment and evidence-based acceptance.

Dataset downloads occur during controlled preparation. The training and
independent evaluation scripts consume checked local files and never download
data or change the author's model and learning-rate scheduler.
"""
import hashlib
import json
import math
import pickle
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from src.method_adapters import RUNTIMES, SirenAdapter, digest, read_json, write_json
from src.preset_downloads import atomic_cache_bytes, download_bytes
from src.safety.paths import workspace_path
from src.experiments.rezero_runtime import CIFAR_FILE_MD5, REFERENCE_PARAMETERS


CIFAR_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR_ARCHIVE_BYTES = 170498071
# The pinned repository stores these files with CRLF. Git exports may use LF,
# so pin the LF-normalized contents; snapshot hashes still bind exact run bytes.
SOURCE_SHA256 = {
    "train_faster_superc.py": "d0bd56a7eeea3baea3b81ed89ec2fc3008650324c8ae0b37549576eff584a3e3",
    "models/rezero_preact_resnet.py": "7a89b858ffee933aec702844bfd600a9c81f6360289f72e7697da8a052cd004a",
    "customonecycle.py": "369cb8114d405a0a451a4847d2acdc8c291fd31ba47ebe7cf363caff5a404719",
    "Faster_SuperC.ipynb": "bc05b44f91f69cff0e0754dc0aecbd926eb6c2a338e103c8af75670874f0dbc6",
}
DATASET_FOLDER = "dataset/cifar-10-batches-py"
STEPS_PER_EPOCH = 98
TOTAL_STEPS = 4410
TARGET_ACCURACY_PCT = 94.0


def file_hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def dataset_manifest_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def checked_dataset_files(workspace):
    files = {}
    for name, expected in CIFAR_FILE_MD5.items():
        path = workspace_path(workspace, f"{DATASET_FOLDER}/{name}",
                              "CIFAR-10 file", must_exist=True)
        sha, md5 = file_hashes(path)
        if md5 != expected:
            raise ValueError(f"Official CIFAR-10 checksum mismatch: {name}")
        files[name] = sha
    return files


def checked_parameters(profile):
    parameters = profile["parameters"]
    if set(parameters) != set(REFERENCE_PARAMETERS):
        raise ValueError("ReZero parameter keys differ from the frozen paper protocol")
    for key, expected in REFERENCE_PARAMETERS.items():
        value = parameters[key]
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"Frozen ReZero parameter mismatch: {key}")
    return parameters


def _source_hashes(profile):
    supplied = profile.get("source_sha256s", profile.get("source_sha256", SOURCE_SHA256))
    if not isinstance(supplied, dict):
        raise ValueError("ReZero requires individually frozen source checksums")
    normalized = {name.replace("\\", "/"): value for name, value in supplied.items()}
    if any(normalized.get(name) != sha for name, sha in SOURCE_SHA256.items()):
        raise ValueError("ReZero author source contract has changed")
    return normalized


def _verify_source_files(root, profile):
    expected = _source_hashes(profile)
    for name, sha in expected.items():
        path = workspace_path(root, name, "author source", must_exist=True)
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != sha:
            raise ValueError(f"Pinned ReZero author source checksum mismatch: {name}")
    return expected


def _archive_valid(path, spec):
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    sha, md5 = file_hashes(path)
    return sha == spec["sha256"] and md5 == spec.get("md5", CIFAR_ARCHIVE_MD5)


def _extract_cifar_archive(archive, workspace):
    expected = {f"cifar-10-batches-py/{name}" for name in CIFAR_FILE_MD5}
    allowed = expected | {"cifar-10-batches-py/readme.html"}
    with tarfile.open(archive, "r:gz") as tar:
        members, seen = tar.getmembers(), set()
        total = 0
        for member in members:
            name = member.name.rstrip("/")
            workspace_path(workspace, f"dataset/{name}", "CIFAR-10 archive")
            if name in seen or "\\" in name or str(PurePosixPath(name)) != name:
                raise ValueError("Duplicate or ambiguous CIFAR-10 archive path")
            seen.add(name)
            if member.isdir() and name == "cifar-10-batches-py":
                continue
            if not member.isfile() or name not in allowed:
                raise ValueError(f"Unexpected CIFAR-10 archive entry: {name}")
            if member.size < 0 or member.size > 32 * 1024 * 1024:
                raise ValueError("Unexpected CIFAR-10 archive member size")
            total += member.size
        if not expected.issubset(seen) or total > 190 * 1024 * 1024:
            raise ValueError("Incomplete or oversized CIFAR-10 archive")
        # No tarfile.extract/extractall: only known regular files are written.
        for member in members:
            if not member.isfile():
                continue
            name = member.name
            target = workspace_path(workspace, f"dataset/{name}", "CIFAR-10 archive")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return checked_dataset_files(workspace)


def _finite(value, name, *, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"Invalid measured ReZero value: {name}")
    return value


def _close(actual, expected, name):
    _finite(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"ReZero measurement mismatch: {name}")


def _schedule(step, p):
    first, second = round(p["point_1_step"] * TOTAL_STEPS), round(p["point_2_step"] * TOTAL_STEPS)
    low, high = p["momentum_range"]
    if step <= first:
        scale = step / first
        return p["init_lr"] + scale * (p["point_1_lr"] - p["init_lr"]), high + scale * (low - high)
    if step <= second:
        scale = (step - first) / (second - first)
        return p["point_1_lr"] + scale * (p["point_2_lr"] - p["point_1_lr"]), low - scale * (low - high)
    scale = (step - second) / (TOTAL_STEPS - second)
    return p["point_2_lr"] + scale * (p["end_lr"] - p["point_2_lr"]), high


def _verify_training(record, cfg):
    from src.experiments.rezero_runtime import validate_training_record
    validate_training_record(record, cfg)
    p = cfg["parameters"]
    if record.get("seed") != p["seed"] or record.get("device") != "cuda" or record.get("precision") != "float32":
        raise ValueError("ReZero training seed, device, or precision changed")
    if (record.get("torch") != "2.5.1+cu121" or record.get("torchvision") != "0.20.1+cu121"
            or record.get("cuda") != "12.1"):
        raise ValueError("Training did not use the frozen CUDA environment")
    _finite(record.get("training_elapsed_s"), "training_elapsed_s")
    batch_history = record["batch_history"]
    for epoch, history in enumerate(record["history"], 1):
        steps = batch_history[(epoch - 1) * STEPS_PER_EPOCH:epoch * STEPS_PER_EPOCH]
        correct, weighted, displayed = 0, 0.0, 0.0
        for index, item in enumerate(steps):
            samples = 512 if index < 97 else 336
            if (item.get("epoch") != epoch or item.get("batch") != index + 1
                    or item.get("samples") != samples):
                raise ValueError("ReZero batch history is missing, shortened, or reordered")
            count = item.get("correct")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= samples:
                raise ValueError("Invalid ReZero training batch count")
            loss = _finite(item.get("loss"), "batch_loss")
            lr, momentum = _schedule((epoch - 1) * STEPS_PER_EPOCH + index, p)
            _close(item.get("lr"), lr, "author_learning_rate")
            _close(item.get("momentum"), momentum, "author_momentum")
            correct += count
            weighted += loss * samples
            displayed += loss
        if correct != history["train"]["correct"]:
            raise ValueError("Training epoch accuracy differs from actual batch counts")
        _close(history["train"]["sample_cross_entropy"], weighted / 50000, "train_cross_entropy")
        _close(history["train"]["loss"], displayed / 97, "author_displayed_loss")
        next_step = epoch * STEPS_PER_EPOCH
        lr, momentum = _schedule(next_step, p)
        _close(history.get("lr_after_epoch"), lr, "learning_rate_after_epoch")
        _close(history.get("momentum_after_epoch"), momentum, "momentum_after_epoch")
        _finite(history.get("epoch_elapsed_s"), "epoch_elapsed_s")
    groups = record.get("optimizer_groups", {})
    ordinary, residual = groups.get("ordinary", {}), groups.get("residual", {})
    from src.experiments.rezero_runtime import AUTHOR_RESIDUAL_NAMES
    if (ordinary.get("optimizer") != "SGD" or ordinary.get("weight_decay") != .0002
            or ordinary.get("nesterov") is not False or residual.get("optimizer") != "Adagrad"
            or residual.get("lr") != .1 or residual.get("weight_decay") != 0.0
            or residual.get("initial_accumulator_value") != 0.0 or residual.get("eps") != 1e-10
            or residual.get("names") != AUTHOR_RESIDUAL_NAMES or not ordinary.get("names")
            or any("resweight" in name for name in ordinary["names"])
            or any("resweight" not in name for name in residual["names"])
            or set(ordinary["names"]) & set(residual["names"])):
        raise ValueError("ReZero optimizer groups differ from the author protocol")


class ReZeroAdapter:
    runtime = "rezero_runtime.py"
    evaluator = "evaluate_rezero.py"

    def required_files(self, profile):
        return profile["required_files"]

    def prepare_dataset(self, root, profile, workspace, *, offline=False):
        spec = profile["dataset"]
        cache = workspace_path(root, f"dataset_cache/{spec['name']}/{spec['sha256']}/{spec.get('target', 'cifar-10-python.tar.gz')}",
                               "CIFAR-10 cache")
        # A checksum identifies cached bytes, not the transport that originally
        # fetched them. Report only a download performed by this preparation.
        source_url = None
        cache_hit = _archive_valid(cache, spec)
        if not cache_hit:
            if offline:
                raise RuntimeError("Offline CIFAR-10 archive is missing or its fixed checksum is invalid")
            urls = list(dict.fromkeys([spec["url"], *spec.get("mirrors", [])]))
            for url in urls:
                try:
                    content = download_bytes(url, max_bytes=spec["bytes"], timeout_s=60)
                except RuntimeError:
                    if url == urls[-1]:
                        raise RuntimeError("All declared CIFAR-10 download sources failed") from None
                    continue
                if (len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]
                        or hashlib.md5(content).hexdigest() != spec.get("md5", CIFAR_ARCHIVE_MD5)):
                    raise ValueError("Downloaded CIFAR-10 archive identity verification failed")
                atomic_cache_bytes(cache, content)
                source_url = url
                break
        files = _extract_cifar_archive(cache, workspace)
        result = {**spec, "path": str(workspace_path(workspace, DATASET_FOLDER, "CIFAR-10 dataset", must_exist=True)),
                  "cache_path": str(cache), "canonical_url": spec["url"],
                  "cache_hit": cache_hit, "download_source": source_url, "verified": True,
                  "archive_sha256": spec["sha256"], "files_sha256": files,
                  "manifest_sha256": dataset_manifest_hash(files),
                  "train_samples": 50000, "test_samples": 10000, "split": "official_train_and_test"}
        write_json(Path(workspace) / "dataset_manifest.json", result)
        return result

    def materialize(self, workspace, profile, spec_hash):
        root = Path(workspace)
        checked_parameters(profile)
        _verify_source_files(root, profile)
        prepared = read_json(workspace_path(root, "dataset_manifest.json", "dataset manifest", must_exist=True))
        files = checked_dataset_files(root)
        if (prepared.get("verified") is not True or prepared.get("files_sha256") != files
                or prepared.get("manifest_sha256") != dataset_manifest_hash(files)
                or prepared.get("archive_sha256") != profile["dataset"]["sha256"]):
            raise ValueError("Prepared CIFAR-10 dataset manifest differs from the frozen bytes")
        for source, target in ((self.runtime, "run_experiment.py"), (self.evaluator, "evaluate.py")):
            shutil.copyfile(RUNTIMES / source, workspace_path(root, target, "adapter runtime"))
        write_json(root / "experiment.json", {"parameters": profile["parameters"], "spec_sha256": spec_hash,
            "dataset_sha256": profile["dataset"]["sha256"], "dataset_files_sha256": files,
            "dataset_manifest_sha256": dataset_manifest_hash(files)})
        paths = ["run_experiment.py", "evaluate.py", "experiment.json", "dataset_manifest.json",
                 *[f"{DATASET_FOLDER}/{name}" for name in files]]
        manifest = {"files": {name: digest(root / name) for name in paths},
                    "source": profile["paper"]["reference_source"], "source_files_sha256": _source_hashes(profile),
                    "scope": "selected_paper_experiment",
                    "note": "Unmodified pinned author model and scheduler; full CIFAR-10 45-epoch FP32 run and independent checkpoint evaluation."}
        write_json(root / "adapter_manifest.json", manifest)
        return manifest

    def public_sources(self, workspace, profile):
        root = Path(workspace)
        _verify_source_files(root, profile)
        repository = profile["repository"]
        prefix = f"{repository['url']}/blob/{repository['revision']}"
        sources = []
        for name in ["README.md", *SOURCE_SHA256]:
            path = workspace_path(root, name, "public source")
            if not path.is_file():
                continue
            raw = path.read_bytes().replace(b"\r\n", b"\n")
            content = raw.decode("utf-8")
            if name.endswith(".ipynb"):
                notebook = json.loads(content)
                # Code/markdown are evidence; historic cell outputs cannot stand
                # in for this run's actual training or independent evaluation.
                content = "\n\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
            sources.append({"source_id": "author_rezero_" + name.replace("/", "_").replace(".", "_"),
                            "url": f"{prefix}/{name}", "locator": name, "content": content,
                            "origin": "official_repository", "revision": repository["revision"],
                            "sha256": hashlib.sha256(raw).hexdigest()})
        return sources

    def steps(self, profile, *, train=True, split=None):
        checked_parameters(profile)
        steps = SirenAdapter().steps(profile, train=False)
        probe = steps[0]["argv"][-1]
        probe += ("; import importlib; p['author_components']={n:str(Path(importlib.import_module(n).__file__).resolve()) "
                  "for n in ('models.rezero_preact_resnet','customonecycle')}; "
                  "p['cuda_available']=torch.cuda.is_available(); p['device_name']=torch.cuda.get_device_name(0); "
                  "Path('import_provenance.json').write_text(json.dumps(p),encoding='utf-8'); print(json.dumps(p))")
        steps[0]["argv"][-1] = probe
        if train:
            env = dict(steps[0]["env"])
            steps.append({"id": "train", "kind": "train", "depends_on": ["import_check"],
                          "argv": ["python", "-u", "run_experiment.py"], "timeout_s": 7200, "env": env,
                          "artifacts": [{"path": f"artifacts/{name}"} for name in
                                        ("training.json", "checkpoint.pt", "prediction.npy", "targets.npy")]})
            step = self.evaluation_step(profile, split)
            step.update(kind="eval", depends_on=["train"],
                        requires=[{"path": f"artifacts/{name}"} for name in
                                  ("training.json", "checkpoint.pt", "prediction.npy", "targets.npy")],
                        artifacts=[{"path": f"artifacts/{name}"} for name in
                                   ("metrics_test.json", "prediction_recomputed.npy", "targets_recomputed.npy")])
            steps.append(step)
        return steps

    def evaluation_step(self, profile, split=None):
        if split not in (None, "test"):
            raise ValueError("ReZero paper acceptance requires the official test split")
        return {"id": "evaluate", "argv": ["python", "-u", "evaluate.py", "test"], "timeout_s": 300,
                "env": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg",
                        "PYTHONHASHSEED": str(profile["parameters"]["seed"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}

    def verify_environment(self, profile, execution, workspace):
        SirenAdapter().verify_environment(profile, execution, workspace)
        imported = read_json(workspace_path(workspace, "import_provenance.json", "import provenance", must_exist=True))
        required = {"torch", "torchvision", "numpy", "scipy", "PIL", "matplotlib"}
        if set(imported.get("modules", {})) != required or imported.get("cuda_available") is not True:
            raise ValueError("Incomplete frozen CUDA import provenance")
        expected = {"models.rezero_preact_resnet": "models/rezero_preact_resnet.py", "customonecycle": "customonecycle.py"}
        if set(imported.get("author_components", {})) != set(expected):
            raise ValueError("Missing pinned ReZero author import provenance")
        for module, locator in expected.items():
            if Path(imported["author_components"][module]).resolve() != workspace_path(workspace, locator, "author component", must_exist=True):
                raise ValueError("ReZero model or scheduler was imported outside the pinned workspace")

    def verify(self, profile, execution, workspace, snapshot, manifest, spec_hash, split="test"):
        import numpy as np
        from src.experiments.evaluate_rezero import metrics_from_logits
        from src.repository_reproduction import execution_succeeded
        root, out = Path(workspace), Path(workspace) / "artifacts"
        checked_parameters(profile)
        if split != "test" or not execution_succeeded(execution):
            raise ValueError("Full ReZero training and independent test evaluation must succeed")
        actual_steps = execution.get("steps", execution.get("stages", []))
        if [step.get("id") for step in actual_steps] != ["import_check", "train", "evaluate"]:
            raise ValueError("ReZero execution evidence is missing a required step")
        for name, expected in {**snapshot["files"], **manifest["files"]}.items():
            if digest(workspace_path(root, name, "protocol file", must_exist=True)) != expected:
                raise ValueError(f"Experiment source or frozen protocol was changed: {name}")
        _verify_source_files(root, profile)
        self.verify_environment(profile, execution, workspace)
        cfg = read_json(root / "experiment.json")
        files = checked_dataset_files(root)
        dataset = {"archive_sha256": profile["dataset"]["sha256"], "files_sha256": files,
                   "manifest_sha256": dataset_manifest_hash(files), "train_samples": 50000,
                   "test_samples": 10000, "split": "official_train_and_test"}
        if cfg != {"parameters": profile["parameters"], "spec_sha256": spec_hash,
                   "dataset_sha256": profile["dataset"]["sha256"], "dataset_files_sha256": files,
                   "dataset_manifest_sha256": dataset_manifest_hash(files)}:
            raise ValueError("Materialized ReZero protocol differs from the frozen experiment")
        training, metrics = read_json(out / "training.json"), read_json(out / "metrics_test.json")
        _verify_training(training, cfg)
        if any(metrics.get(key) != training.get(key) for key in ("numerics", "torch", "torchvision", "cuda")):
            raise ValueError("Independent evaluation precision or runtime differs from training")
        for document in (training, metrics):
            if document.get("dataset") != dataset or document.get("spec_sha256") != spec_hash:
                raise ValueError("ReZero evidence does not belong to this dataset and frozen experiment")
            for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("prediction.npy", "prediction_sha256"),
                              ("targets.npy", "targets_sha256")):
                if document.get(key) != digest(workspace_path(root, f"artifacts/{name}", "run artifact", must_exist=True)):
                    raise ValueError(f"ReZero artifact checksum mismatch: {name}")
        if training.get("source_files_sha256") != {name: digest(root / name) for name in
                                                  ("models/rezero_preact_resnet.py", "customonecycle.py")}:
            raise ValueError("Training source provenance differs from the pinned author files")
        reported = json.loads(execution["final"]["stdout"].strip().splitlines()[-1])
        if reported != metrics or any(metrics.get(key) is not True for key in
                                     ("pass", "protocol_pass", "independent_metrics_pass")):
            raise ValueError("Independent evaluation output and recorded verdict disagree")
        train_stdout = actual_steps[1].get("stdout", "")
        train_final = json.loads(train_stdout.strip().splitlines()[-1])
        for key in ("epochs_completed", "steps_completed", "best_accuracy_pct", "training_elapsed_s"):
            if train_final.get(key) != training.get(key):
                raise ValueError("Training log and stored completion evidence disagree")
        if (metrics.get("split") != "test" or metrics.get("samples") != 10000
                or metrics.get("epochs_completed") != 45 or metrics.get("steps_completed") != TOTAL_STEPS
                or metrics.get("best_epoch") != training["best_epoch"]
                or metrics.get("target_accuracy_pct") != TARGET_ACCURACY_PCT
                or set(metrics.get("metrics", {})) != {"top1_accuracy_pct", "cross_entropy"}
                or set(profile["paper"]["required_metrics"]) != {"top1_accuracy_pct", "cross_entropy"}):
            raise ValueError("ReZero independent metric scope is incomplete or changed")
        arrays = {}
        for name, key in (("prediction_recomputed.npy", "recomputed_prediction_sha256"),
                          ("targets_recomputed.npy", "recomputed_targets_sha256")):
            path = workspace_path(root, f"artifacts/{name}", "independent artifact", must_exist=True)
            if metrics.get(key) != digest(path):
                raise ValueError("Independent ReZero prediction checksum mismatch")
            arrays[name] = np.load(path, allow_pickle=False)
        saved_prediction = np.load(out / "prediction.npy", allow_pickle=False)
        saved_targets = np.load(out / "targets.npy", allow_pickle=False)
        recomputed, correct, predicted = metrics_from_logits(arrays["prediction_recomputed.npy"], arrays["targets_recomputed.npy"])
        saved_metrics, saved_correct, saved_classes = metrics_from_logits(saved_prediction, saved_targets)
        # Deserialization is restricted to the official batch already checked
        # against the frozen MD5 and SHA-256 above, never arbitrary uploaded data.
        with (root / DATASET_FOLDER / "test_batch").open("rb") as stream:
            canonical_labels = np.asarray(pickle.load(stream, encoding="latin1")["labels"])
        if (not np.array_equal(saved_targets, canonical_labels)
                or not np.array_equal(arrays["targets_recomputed.npy"], canonical_labels)
                or not np.array_equal(saved_classes, predicted)
                or not np.allclose(saved_prediction, arrays["prediction_recomputed.npy"], rtol=1e-5, atol=1e-5)
                or saved_correct != correct):
            raise ValueError("ReZero predictions do not match the independently inferred checkpoint and official labels")
        for key, value in recomputed.items():
            _close(metrics["metrics"].get(key), value, key)
        if (any(metrics.get(key) != correct for key in ("correct", "numpy_correct", "torch_correct"))
                or recomputed["top1_accuracy_pct"] != training["best_accuracy_pct"]
                or saved_metrics["top1_accuracy_pct"] != training["best_accuracy_pct"]
                or training.get("best_test_export", {}).get("correct") != correct):
            raise ValueError("Author-selected checkpoint accuracy differs from independent measured counts")
        quality = recomputed["top1_accuracy_pct"] >= TARGET_ACCURACY_PCT
        if metrics.get("quality_pass") is not quality:
            raise ValueError("ReZero quality verdict differs from the fixed 94 percent reference")
        return {"status": "reproduced" if quality else "reference_not_met",
                "result_level": "reproduced" if quality else "experiment_completed",
                "is_reproduced": quality, "optimization_eligible": False,
                "quality_pass": quality, "scope": "selected_paper_experiment",
                "protocol_pass": True, "independent_metrics_pass": True,
                "reason": ("ReZero §5 / Appendix E.2: full CIFAR-10, 45 epochs, author-selected checkpoint, independently evaluated top-1 accuracy ≥94%."
                           if quality else "Full ReZero paper protocol completed, but independently measured top-1 accuracy is below the fixed 94% reference."),
                "metrics_comparison": {"paper": {"top1_accuracy_pct": TARGET_ACCURACY_PCT}, "actual": recomputed},
                "metric_records": [{"name": name, "value": value, "unit": "percent" if name == "top1_accuracy_pct" else "scalar",
                                    "direction": "maximize" if name == "top1_accuracy_pct" else "minimize",
                                    "split": "test", "stage": "eval", "seed": profile["parameters"]["seed"],
                                    "source": "artifacts/metrics_test.json", "spec_sha256": spec_hash}
                                   for name, value in recomputed.items()],
                "training_summary": {"initial_loss": training["history"][0]["train"]["sample_cross_entropy"],
                                     "final_loss": training["history"][-1]["train"]["sample_cross_entropy"],
                                     "steps_completed": TOTAL_STEPS, "epochs_completed": 45,
                                     "training_elapsed_s": training["training_elapsed_s"],
                                     "best_epoch": training["best_epoch"], "best_accuracy_pct": training["best_accuracy_pct"],
                                     "train_samples": 50000, "test_samples": 10000}}
