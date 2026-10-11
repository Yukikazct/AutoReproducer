"""Pinned author's CIFAR-10 superconvergence loop, with an auditable main entry.

The author model and CustomOneCycleLR are imported from the repository without
rewriting them. Windows process startup, local dataset checks, and artifact
records surround the original 45-epoch, FP32, batch-512 training procedure.
"""
import hashlib
import importlib
import json
import math
import random
import time
from pathlib import Path


REFERENCE_PARAMETERS = {
    "arch": "rezero_preactresnet18", "seed": 6892, "batch_size": 512,
    "epochs": 45, "init_lr": .032, "point_1_step": .1, "point_1_lr": 1.2,
    "point_2_step": .9, "point_2_lr": .032, "end_lr": .001,
    "momentum_range": [.85, .95], "weight_decay": .0002,
    "resweight_lr": .1, "device": "cuda", "precision": "float32",
    "num_workers": 2, "protocol": "paper_superconvergence",
}
TRAIN_SAMPLES, TEST_SAMPLES, TARGET_ACCURACY_PCT = 50000, 10000, 94.0
AUTHOR_RESIDUAL_NAMES = [f"layer{layer}.{block}.resweight"
                         for layer in range(1, 5) for block in range(2)]
NUMERICS = {"cudnn_deterministic": True, "cudnn_benchmark": True,
            "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
            "autocast": False, "parameter_dtype": "torch.float32"}
CIFAR_FILE_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}
NORMALIZE_MEAN = [.4914, .48216, .44653]
NORMALIZE_STD = [.24703, .24349, .26159]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_protocol(parameters):
    if set(parameters) != set(REFERENCE_PARAMETERS):
        raise ValueError("Frozen ReZero parameter keys changed")
    for key, expected in REFERENCE_PARAMETERS.items():
        actual = parameters.get(key)
        if isinstance(actual, bool) or actual != expected:
            raise ValueError(f"Frozen ReZero parameter mismatch: {key}")


def reference_schedule(step, total_steps=4410):
    """Paper one-cycle learning rate and momentum at an optimizer update."""
    first, second = round(.1 * total_steps), round(.9 * total_steps)
    if not 0 <= step <= total_steps:
        raise ValueError("One-cycle step outside the frozen schedule")
    if step <= first:
        scale = step / first
        return .032 + scale * (1.2 - .032), .95 + scale * (.85 - .95)
    if step <= second:
        scale = (step - first) / (second - first)
        return 1.2 + scale * (.032 - 1.2), .85 - scale * (.85 - .95)
    scale = (step - second) / (total_steps - second)
    return .032 + scale * (.001 - .032), .95


def dataset_fingerprints(cfg, root="dataset"):
    """Check canonical CIFAR bytes before torchvision deserializes any batch."""
    folder = Path(root) / "cifar-10-batches-py"
    files = {}
    for name, expected_md5 in CIFAR_FILE_MD5.items():
        raw = (folder / name).read_bytes()
        if hashlib.md5(raw).hexdigest() != expected_md5:
            raise ValueError(f"Official CIFAR-10 checksum mismatch: {name}")
        files[name] = hashlib.sha256(raw).hexdigest()
    supplied = cfg.get("dataset_files_sha256")
    if not isinstance(supplied, dict) or set(supplied) != set(files):
        raise ValueError("Frozen CIFAR-10 file hashes are required")
    if supplied != files:
        raise ValueError("Frozen CIFAR-10 SHA-256 mismatch")
    manifest_sha = hashlib.sha256(json.dumps(files, sort_keys=True,
                                 separators=(",", ":")).encode()).hexdigest()
    if cfg.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Frozen CIFAR-10 manifest checksum mismatch")
    return {"archive_sha256": cfg["dataset_sha256"], "files_sha256": files,
            "manifest_sha256": manifest_sha, "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES, "split": "official_train_and_test"}


def build_transforms(transforms):
    normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(.25, .25, .25),
        transforms.RandomRotation(2),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    test = transforms.Compose([transforms.ToTensor(), normalize])
    return train, test


def build_datasets(datasets, transforms, root="dataset"):
    train_transform, test_transform = build_transforms(transforms)
    train = datasets.CIFAR10(root=root, train=True, download=False,
                             transform=train_transform)
    test = datasets.CIFAR10(root=root, train=False, download=False,
                            transform=test_transform)
    if len(train) != TRAIN_SAMPLES or len(test) != TEST_SAMPLES:
        raise ValueError("ReZero requires all 50000 training and 10000 test examples")
    return train, test


def load_author_components():
    module = importlib.import_module("models.rezero_preact_resnet")
    scheduler = importlib.import_module("customonecycle")
    return module.rezero_preactresnet18, scheduler.CustomOneCycleLR


def build_optimizers(torch, model, scheduler_type, parameters, batches):
    ordinary = [(name, p) for name, p in model.named_parameters()
                if "resweight" not in name]
    residual = [(name, p) for name, p in model.named_parameters()
                if "resweight" in name]
    if not ordinary or len(residual) != 8:
        raise ValueError("Unexpected author PreActResNet18 parameter groups")
    optimizer = torch.optim.SGD([
        {"params": [p for _, p in ordinary], "lr": .01}],
        weight_decay=parameters["weight_decay"])
    scheduler = scheduler_type(
        optimizer, num_steps=parameters["epochs"] * batches,
        init_lr=parameters["init_lr"],
        point_1=(parameters["point_1_step"], parameters["point_1_lr"]),
        point_2=(parameters["point_2_step"], parameters["point_2_lr"]),
        end_lr=parameters["end_lr"],
        momentum_range=tuple(parameters["momentum_range"]), exp_decay=False,
        param_group=0)
    residual_optimizer = torch.optim.Adagrad([
        {"params": [p for _, p in residual], "lr": parameters["resweight_lr"]}])
    groups = {"ordinary": {"optimizer": "SGD", "names": [n for n, _ in ordinary],
                           "weight_decay": parameters["weight_decay"],
                           "nesterov": False},
              "residual": {"optimizer": "Adagrad", "names": [n for n, _ in residual],
                           "lr": parameters["resweight_lr"], "weight_decay": 0.0,
                           "initial_accumulator_value": 0.0, "eps": 1e-10}}
    return optimizer, residual_optimizer, scheduler, groups


def optimizer_evidence(model, optimizer, residual_optimizer, scheduler):
    """Read the actual optimizer state after training, including update counts."""
    ordinary = optimizer.param_groups[0]["params"]
    residual = dict((name, parameter) for name, parameter in model.named_parameters()
                    if "resweight" in name)
    if list(residual) != AUTHOR_RESIDUAL_NAMES:
        raise ValueError("Author residual parameter names changed")
    return {
        "scheduler_last_step": scheduler.last_step,
        "sgd_parameter_tensors": len(ordinary),
        "sgd_momentum_buffers": sum("momentum_buffer" in optimizer.state[p] for p in ordinary),
        "adagrad_steps": {name: int(residual_optimizer.state[p]["step"].item())
                           for name, p in residual.items()},
    }


def configure_runtime(torch, torchvision):
    if (torch.__version__ != "2.5.1+cu121" or torchvision.__version__ != "0.20.1+cu121"
            or torch.version.cuda != "12.1"):
        raise RuntimeError("ReZero requires the frozen torch 2.5.1 / torchvision 0.20.1 CUDA 12.1 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU or reduced protocol fallback is permitted")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def execution_numerics(torch, model):
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    actual = {"cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "autocast": torch.is_autocast_enabled(),
              "parameter_dtype": next(iter(dtypes)) if len(dtypes) == 1 else "mixed"}
    if actual != NUMERICS:
        raise ValueError("ReZero requires author FP32 execution without mixed precision or TF32")
    return actual


def train_epoch(torch, loader, model, criterion, optimizer,
                residual_optimizer, scheduler, device, epoch):
    model.train()
    loss_sum = weighted_loss = 0.0
    correct = total = 0
    steps = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        lr, momentum = optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["momentum"]
        outputs = model(inputs)
        optimizer.zero_grad()
        residual_optimizer.zero_grad()
        loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite ReZero training loss")
        loss.backward()
        optimizer.step()
        residual_optimizer.step()
        scheduler.step()
        value = float(loss.item())
        batch_correct = int((outputs.detach().argmax(1) == targets).sum().item())
        samples = targets.size(0)
        loss_sum += value
        weighted_loss += value * samples
        correct += batch_correct
        total += samples
        steps.append({"epoch": epoch, "batch": batch_idx + 1, "samples": samples,
                      "loss": value, "correct": batch_correct, "lr": lr,
                      "momentum": momentum})
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch}/45 batch {batch_idx + 1}/{len(loader)} "
                  f"loss={value:.6f} train_accuracy={100.0 * correct / total:.2f}% "
                  f"lr={lr:.6f}", flush=True)
    if total != TRAIN_SAMPLES:
        raise ValueError("Incomplete CIFAR-10 training epoch")
    # Preserve the author's displayed loss, including its historical denominator;
    # sample_cross_entropy is the correctly weighted metric used for audit.
    return {"loss": loss_sum / (len(steps) - 1),
            "sample_cross_entropy": weighted_loss / total,
            "accuracy_pct": 100.0 * correct / total, "correct": correct,
            "samples": total, "batches": len(steps)}, steps


def infer_test(torch, loader, model, criterion, device, *, export=False):
    import numpy as np
    model.eval()
    loss_sum = weighted_loss = 0.0
    correct = total = batches = 0
    logits, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(outputs).all() or not torch.isfinite(loss):
                raise ValueError("Non-finite ReZero test output")
            value = float(loss.item())
            loss_sum += value
            weighted_loss += value * targets.size(0)
            correct += int((outputs.argmax(1) == targets).sum().item())
            total += targets.size(0)
            batches += 1
            if export:
                logits.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
    if total != TEST_SAMPLES:
        raise ValueError("ReZero test inference must cover all 10000 examples")
    result = {"loss": loss_sum / (batches - 1),
              "sample_cross_entropy": weighted_loss / total,
              "accuracy_pct": 100.0 * correct / total, "correct": correct,
              "samples": total, "batches": batches}
    return result, (np.concatenate(logits) if export else None), (
        np.concatenate(labels) if export else None)


def validate_training_record(record, cfg):
    """Reject unfinished/shortened runs before looking at their accuracy."""
    validate_protocol(cfg["parameters"])
    expected_steps = 45 * math.ceil(TRAIN_SAMPLES / 512)
    history = record.get("history", [])
    if (record.get("status") != "completed" or record.get("epochs_completed") != 45
            or record.get("steps_completed") != expected_steps
            or record.get("parameters") != cfg["parameters"]
            or record.get("spec_sha256") != cfg["spec_sha256"]
            or len(history) != 45
            or len(record.get("batch_history", [])) != expected_steps):
        raise ValueError("Incomplete or changed ReZero paper protocol")
    state = record.get("optimizer_state", {})
    ordinary_names = record.get("optimizer_groups", {}).get("ordinary", {}).get("names", [])
    if (state.get("scheduler_last_step") != expected_steps
            or not ordinary_names or len(set(ordinary_names)) != len(ordinary_names)
            or state.get("sgd_parameter_tensors") != len(ordinary_names)
            or state.get("sgd_momentum_buffers") != len(ordinary_names)
            or state.get("adagrad_steps") != {name: expected_steps for name in AUTHOR_RESIDUAL_NAMES}):
        raise ValueError("ReZero optimizer state does not prove all scheduled updates completed")
    if record.get("numerics") != NUMERICS or record.get("model_parameters") != 11171154:
        raise ValueError("ReZero author model or full precision execution changed")
    for epoch, item in enumerate(history, 1):
        if item.get("epoch") != epoch:
            raise ValueError("Missing ReZero epoch history")
        for split, expected, batches in (("train", TRAIN_SAMPLES, 98), ("test", TEST_SAMPLES, 20)):
            metrics = item.get(split, {})
            if metrics.get("samples") != expected or metrics.get("batches") != batches:
                raise ValueError("Incomplete ReZero dataset split history")
            for key in ("loss", "sample_cross_entropy", "accuracy_pct"):
                value = metrics.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Invalid measured ReZero history")
            if metrics["loss"] < 0 or metrics["sample_cross_entropy"] < 0:
                raise ValueError("Negative ReZero loss")
            correct = metrics.get("correct")
            if (isinstance(correct, bool) or not isinstance(correct, int)
                    or not 0 <= correct <= expected
                    or metrics["accuracy_pct"] != 100.0 * correct / expected):
                raise ValueError("ReZero accuracy is not backed by actual counts")
    for index, step in enumerate(record["batch_history"]):
        epoch, batch = divmod(index, 98)
        expected_lr, expected_momentum = reference_schedule(index, expected_steps)
        if (step.get("epoch") != epoch + 1 or step.get("batch") != batch + 1
                or step.get("samples") != (512 if batch < 97 else 336)
                or not isinstance(step.get("loss"), (int, float))
                or not math.isfinite(step["loss"]) or step["loss"] < 0
                or not isinstance(step.get("correct"), int)
                or not 0 <= step["correct"] <= step["samples"]
                or not isinstance(step.get("lr"), (int, float))
                or not math.isfinite(step["lr"])
                or not math.isclose(step["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-12)
                or not isinstance(step.get("momentum"), (int, float))
                or not math.isfinite(step["momentum"])
                or not math.isclose(step["momentum"], expected_momentum, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("ReZero batch history does not follow the frozen full-data schedule")
    for epoch, item in enumerate(history):
        batch_records = record["batch_history"][epoch * 98:(epoch + 1) * 98]
        train = item["train"]
        if (train["correct"] != sum(step["correct"] for step in batch_records)
                or not math.isclose(train["loss"], sum(step["loss"] for step in batch_records) / 97,
                                    rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(train["sample_cross_entropy"],
                                    sum(step["loss"] * step["samples"] for step in batch_records) / TRAIN_SAMPLES,
                                    rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("ReZero epoch metrics are not backed by recorded optimizer batches")
    best = max(history, key=lambda h: h["test"]["accuracy_pct"])
    if (record.get("best_epoch") != best["epoch"]
            or record.get("best_accuracy_pct") != best["test"]["accuracy_pct"]):
        raise ValueError("ReZero checkpoint selection differs from the author")
    return expected_steps


def main():
    import numpy as np
    import torch
    import torchvision
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
    p = cfg["parameters"]
    validate_protocol(p)
    configure_runtime(torch, torchvision)
    dataset = dataset_fingerprints(cfg)
    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    torch.cuda.manual_seed(p["seed"])
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(p["device"])
    trainset, testset = build_datasets(torchvision.datasets, torchvision.transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=p["batch_size"],
        shuffle=True, num_workers=p["num_workers"])
    testloader = torch.utils.data.DataLoader(testset, batch_size=p["batch_size"],
        shuffle=False, num_workers=p["num_workers"])
    factory, scheduler_type = load_author_components()
    model = factory().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer, residual_optimizer, scheduler, groups = build_optimizers(
        torch, model, scheduler_type, p, len(trainloader))
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    record = {"status": "running", "parameters": p, "spec_sha256": cfg["spec_sha256"],
              "dataset": dataset, "epochs_completed": 0, "steps_completed": 0,
              "history": [], "batch_history": [], "best_epoch": None,
              "best_accuracy_pct": 0.0, "optimizer_groups": groups,
              "model_parameters": sum(param.numel() for param in model.parameters()),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda": torch.version.cuda, "device": str(device), "seed": p["seed"],
              "precision": "float32", "historical_author_torch": "1.2.0",
              "numerics": execution_numerics(torch, model),
              "runtime_compatibility": "Author model, optimizer algorithm and schedule preserved; modern PyTorch/torchvision runtime, not bitwise equivalence to torch 1.2.0.",
              "source_files_sha256": {name: digest(name) for name in
                  ("models/rezero_preact_resnet.py", "customonecycle.py")}}
    write_json(out / "training.json", record)
    try:
        for epoch in range(1, p["epochs"] + 1):
            epoch_started = time.monotonic()
            train, steps = train_epoch(torch, trainloader, model, criterion,
                optimizer, residual_optimizer, scheduler, device, epoch)
            test, _, _ = infer_test(torch, testloader, model, criterion, device)
            if test["accuracy_pct"] > record["best_accuracy_pct"]:
                record["best_accuracy_pct"] = test["accuracy_pct"]
                record["best_epoch"] = epoch
                temporary = out / "checkpoint.pt.tmp"
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "accuracy_pct": test["accuracy_pct"],
                            "correct": test["correct"], "spec_sha256": cfg["spec_sha256"]}, temporary)
                temporary.replace(out / "checkpoint.pt")
            item = {"epoch": epoch, "train": train, "test": test,
                    "lr_after_epoch": optimizer.param_groups[0]["lr"],
                    "momentum_after_epoch": optimizer.param_groups[0]["momentum"],
                    "epoch_elapsed_s": time.monotonic() - epoch_started,
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated()}
            record["history"].append(item)
            record["batch_history"].extend(steps)
            record["epochs_completed"] = epoch
            record["steps_completed"] += len(steps)
            record["training_elapsed_s"] = time.monotonic() - started
            record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
            write_json(out / "training.json", record)
            print(f"Epoch {epoch}/45 completed train_loss={train['sample_cross_entropy']:.6f} "
                  f"train_accuracy={train['accuracy_pct']:.2f}% "
                  f"test_loss={test['sample_cross_entropy']:.6f} "
                  f"test_accuracy={test['accuracy_pct']:.2f}% "
                  f"best={record['best_accuracy_pct']:.2f}% "
                  f"max_gpu_bytes={record['gpu_peak_bytes']} "
                  f"elapsed_s={record['training_elapsed_s']:.1f}", flush=True)
        checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        best_test, prediction, targets = infer_test(torch, testloader, model, criterion, device, export=True)
        if best_test["correct"] != checkpoint["correct"]:
            raise ValueError("Saved author-selected checkpoint changed test accuracy")
        np.save(out / "prediction.npy", prediction, allow_pickle=False)
        np.save(out / "targets.npy", targets, allow_pickle=False)
        record.update(status="completed", checkpoint_sha256=digest(out / "checkpoint.pt"),
                      prediction_sha256=digest(out / "prediction.npy"),
                      targets_sha256=digest(out / "targets.npy"), best_test_export=best_test,
                      optimizer_state=optimizer_evidence(model, optimizer, residual_optimizer, scheduler),
                      training_elapsed_s=time.monotonic() - started)
        validate_training_record(record, cfg)
        write_json(out / "training.json", record)
        print(json.dumps({"epochs_completed": record["epochs_completed"],
                          "steps_completed": record["steps_completed"],
                          "best_accuracy_pct": record["best_accuracy_pct"],
                          "training_elapsed_s": record["training_elapsed_s"]}), flush=True)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      training_elapsed_s=time.monotonic() - started,
                      gpu_peak_bytes=torch.cuda.max_memory_allocated())
        write_json(out / "training.json", record)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise RuntimeError("FP32 batch-512 ReZero exceeds GPU memory; the frozen paper protocol was not reduced") from exc
        raise


if __name__ == "__main__":
    main()
