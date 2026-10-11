"""Independently infer the selected author checkpoint on all real CIFAR-10 test data."""
import json
import sys
from pathlib import Path

try:
    from run_experiment import (TEST_SAMPLES, TARGET_ACCURACY_PCT, build_transforms,
        dataset_fingerprints, digest, load_author_components, validate_training_record,
        write_json, configure_runtime, execution_numerics)
except ModuleNotFoundError:
    from src.experiments.rezero_runtime import (TEST_SAMPLES, TARGET_ACCURACY_PCT,
        build_transforms, dataset_fingerprints, digest, load_author_components,
        validate_training_record, write_json, configure_runtime, execution_numerics)


def metrics_from_logits(logits, targets, *, expected_samples=TEST_SAMPLES):
    """Compute accuracy and cross entropy solely from inferred values and labels."""
    import numpy as np
    logits, targets = np.asarray(logits), np.asarray(targets)
    if (logits.shape != (expected_samples, 10) or targets.shape != (expected_samples,)
            or not np.isfinite(logits).all() or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < 0) or np.any(targets >= 10)):
        raise ValueError("Invalid full CIFAR-10 test prediction or labels")
    predicted = np.argmax(logits, axis=1)
    correct = int(np.count_nonzero(predicted == targets))
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    losses = np.log(np.sum(np.exp(shifted), axis=1)) - shifted[np.arange(expected_samples), targets]
    return {"top1_accuracy_pct": 100.0 * correct / expected_samples,
            "cross_entropy": float(np.mean(losses))}, correct, predicted


def main():
    import numpy as np
    import torch
    import torchvision
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    if split != "test":
        raise ValueError("The paper protocol evaluates the official full CIFAR-10 test split")
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
    out = Path("artifacts")
    training = json.loads((out / "training.json").read_text(encoding="utf-8"))
    validate_training_record(training, cfg)
    dataset = dataset_fingerprints(cfg)
    if training["dataset"] != dataset:
        raise ValueError("Evaluation dataset differs from the frozen training data")
    for artifact, key in (("checkpoint.pt", "checkpoint_sha256"),
                          ("prediction.npy", "prediction_sha256"),
                          ("targets.npy", "targets_sha256")):
        if training.get(key) != digest(out / artifact):
            raise ValueError(f"ReZero artifact checksum mismatch: {artifact}")
    configure_runtime(torch, torchvision)
    device = torch.device(cfg["parameters"]["device"])
    _, transform = build_transforms(torchvision.transforms)
    testset = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                          download=False, transform=transform)
    if len(testset) != TEST_SAMPLES:
        raise ValueError("Independent evaluation requires the full real test dataset")
    loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    factory, _ = load_author_components()
    model = factory().to(device)
    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (checkpoint.get("spec_sha256") != cfg["spec_sha256"]
            or checkpoint.get("epoch") != training["best_epoch"]
            or checkpoint.get("accuracy_pct") != training["best_accuracy_pct"]):
        raise ValueError("Checkpoint does not belong to the author-selected frozen run")
    model.load_state_dict(checkpoint["state_dict"])
    numerics = execution_numerics(torch, model)
    model.eval()
    logits, targets = [], []
    torch_correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            output = model(inputs.to(device))
            torch_correct += int((output.argmax(1).cpu() == labels).sum().item())
            total += labels.size(0)
            logits.append(output.cpu().numpy())
            targets.append(labels.numpy())
    prediction, truth = np.concatenate(logits), np.concatenate(targets)
    if not np.array_equal(truth, np.asarray(testset.targets, dtype=truth.dtype)):
        raise ValueError("Independent labels are not in the official test order")
    metrics, numpy_correct, numpy_prediction = metrics_from_logits(prediction, truth)
    if total != TEST_SAMPLES or numpy_correct != torch_correct:
        raise ValueError("Independent NumPy and PyTorch accuracy counts disagree")
    saved_logits = np.load(out / "prediction.npy", allow_pickle=False)
    saved_targets = np.load(out / "targets.npy", allow_pickle=False)
    saved_metrics, saved_correct, saved_prediction = metrics_from_logits(saved_logits, saved_targets)
    if (not np.array_equal(saved_targets, truth)
            or not np.array_equal(saved_prediction, numpy_prediction)
            or not np.allclose(saved_logits, prediction, rtol=1e-5, atol=1e-5)
            or saved_correct != numpy_correct or checkpoint.get("correct") != numpy_correct
            or saved_metrics["top1_accuracy_pct"] != metrics["top1_accuracy_pct"]):
        raise ValueError("Re-inferred checkpoint predictions differ from the training export")
    np.save(out / "prediction_recomputed.npy", prediction, allow_pickle=False)
    np.save(out / "targets_recomputed.npy", truth, allow_pickle=False)
    result = {"pass": True, "protocol_pass": True, "independent_metrics_pass": True,
              "quality_pass": metrics["top1_accuracy_pct"] >= TARGET_ACCURACY_PCT,
              "target_accuracy_pct": TARGET_ACCURACY_PCT, "metrics": metrics,
              "split": "test", "samples": total, "correct": numpy_correct,
              "numpy_correct": numpy_correct, "torch_correct": torch_correct,
              "epochs_completed": training["epochs_completed"],
              "steps_completed": training["steps_completed"], "best_epoch": checkpoint["epoch"],
              "spec_sha256": cfg["spec_sha256"], "dataset": dataset,
              "numerics": numerics, "torch": torch.__version__,
              "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
              "checkpoint_sha256": digest(out / "checkpoint.pt"),
              "prediction_sha256": digest(out / "prediction.npy"),
              "targets_sha256": digest(out / "targets.npy"),
              "recomputed_prediction_sha256": digest(out / "prediction_recomputed.npy"),
              "recomputed_targets_sha256": digest(out / "targets_recomputed.npy")}
    write_json(out / "metrics_test.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
