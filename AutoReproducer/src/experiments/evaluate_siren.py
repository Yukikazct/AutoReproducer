"""Independent evaluator: rebuild labels from the frozen image, never train logs."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
    p = cfg["parameters"]
    raw = Path("camera.png").read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["dataset_sha256"]:
        raise ValueError("evaluation image checksum mismatch")
    truth = np.asarray(Image.open("camera.png").convert("L").resize(
        (p["sidelength"], p["sidelength"]), Image.Resampling.BILINEAR), dtype=np.float64) / 255.
    out = Path("artifacts")
    pred = np.load(out / "prediction.npy", allow_pickle=False).astype(np.float64)
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("invalid prediction shape or values")
    split = sys.argv[1] if len(sys.argv) > 1 else "fit"
    idx = np.arange(truth.size)
    if p["protocol"] == "pixel_holdout":
        order = np.random.default_rng(1729).permutation(truth.size)
        a, b = int(truth.size * .8), int(truth.size * .9)
        if split not in {"validation", "holdout"}:
            raise ValueError("pixel holdout requires an explicit evaluation split")
        idx = order[a:b] if split == "validation" else order[b:]
    elif split != "fit":
        raise ValueError("official full-image fit is not a holdout experiment")
    mse = float(np.mean((pred.ravel()[idx] - truth.ravel()[idx]) ** 2))
    if mse <= 0 or not np.isfinite(mse):
        raise ValueError("MSE must be finite and positive")
    metrics = {"mse": mse, "psnr": float(-10 * np.log10(mse))}
    result = {"pass": True, "metrics": metrics, "split": split, "samples": len(idx),
              "spec_sha256": cfg["spec_sha256"], "prediction_sha256": hashlib.sha256((out / "prediction.npy").read_bytes()).hexdigest()}
    (out / f"metrics_{split}.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    # No full-image error/holdout labels are exposed while selecting candidates.
    if split == "fit":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, im, title in zip(axs, [truth, pred, abs(pred-truth)], ["Original", "SIREN reconstruction", "Absolute error"]):
            ax.imshow(im, cmap="gray" if title != "Absolute error" else "magma"); ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(out / "reconstruction.png", dpi=130); plt.close(fig)
        training = json.loads((out / "training.json").read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(7, 4)); ax.semilogy(np.arange(len(training["losses"])) + 1, training["losses"])
        ax.set(xlabel="Training step", ylabel="Training MSE (normalized [-1,1])", title="Official SIREN image fit")
        fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=130); plt.close(fig)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
