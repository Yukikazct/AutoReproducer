"""Official notebook image fitting loop with auditable I/O and fixed seeds.

The model in author_model.py is extracted verbatim from cell 3 of the pinned
author notebook. Plotting during training is moved to the independent evaluator.
"""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from author_model import Siren


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
    p = cfg["parameters"]
    if digest("camera.png") != cfg["dataset_sha256"]:
        raise ValueError("camera source checksum mismatch")
    if p["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; prepare the CUDA environment before running")
    random.seed(p["seed"]); np.random.seed(p["seed"]); torch.manual_seed(p["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(p["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(p["device"])
    transform = Compose([Resize(p["sidelength"]), ToTensor(), Normalize([0.5], [0.5])])
    pixels = transform(Image.open("camera.png").convert("L")).permute(1, 2, 0).reshape(-1, 1)
    axis = torch.linspace(-1, 1, steps=p["sidelength"])
    coords = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    indices = np.arange(len(pixels))
    if p["protocol"] == "pixel_holdout":
        indices = np.random.default_rng(1729).permutation(len(pixels))[:int(len(pixels) * .8)]
    x, y = coords[indices].unsqueeze(0).to(device), pixels[indices].unsqueeze(0).to(device)
    model = Siren(in_features=2, out_features=1, hidden_features=p["hidden_features"],
                  hidden_layers=p["hidden_layers"], outermost_linear=True,
                  first_omega_0=p["first_omega_0"], hidden_omega_0=p["hidden_omega_0"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    losses = []
    started = time.monotonic()
    for step in range(p["steps"]):
        prediction, _ = model(x)
        loss = ((prediction - y) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        losses.append(float(loss.detach().cpu()))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 50 == 0:
            print(f"Step {step + 1}/{p['steps']} loss={losses[-1]:.8f}", flush=True)
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    with torch.no_grad():
        pred = model(coords.unsqueeze(0).to(device))[0].cpu().numpy().reshape(p["sidelength"], p["sidelength"])
    np.save(out / "prediction.npy", (pred + 1.) / 2., allow_pickle=False)
    torch.save(model.state_dict(), out / "checkpoint.pt")
    record = {"parameters": p, "steps_completed": len(losses), "losses": losses,
              "training_elapsed_s": time.monotonic() - started,
              "prediction_sha256": digest(out / "prediction.npy"),
              "checkpoint_sha256": digest(out / "checkpoint.pt"),
              "spec_sha256": cfg["spec_sha256"], "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": str(device), "seed": p["seed"]}
    (out / "training.json").write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    print(json.dumps({"steps_completed": len(losses), "training_elapsed_s": record["training_elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
