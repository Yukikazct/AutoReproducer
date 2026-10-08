"""Pinned author's spiral protocol; ODEFunc is extracted verbatim from the source."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from author_model import ODEFunc


class CountedODE(ODEFunc):
    def __init__(self):
        super().__init__()
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return super().forward(t, y)


def main():
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8")); p = cfg["parameters"]
    random.seed(p["seed"]); np.random.seed(p["seed"]); torch.manual_seed(p["seed"])
    torch.set_num_threads(2)
    y0 = torch.tensor([[2., 0.]])
    t = torch.linspace(0., 25., p["data_size"])
    matrix = torch.tensor([[-.1, 2.], [-2., -.1]])
    class Equation(nn.Module):
        def forward(self, t, y):
            return torch.mm(y**3, matrix)
    with torch.no_grad():
        truth = odeint(Equation(), y0, t, method="dopri5", rtol=1e-7, atol=1e-9)
    model = CountedODE()
    solve = lambda initial, times: odeint(model, initial, times, method=p["solver"], rtol=p["rtol"], atol=p["atol"])
    with torch.no_grad():
        initial_mae = float(torch.mean(torch.abs(solve(y0, t)-truth)))
    model.nfe = 0
    optimizer = torch.optim.RMSprop(model.parameters(), lr=p["learning_rate"])
    losses, fit_curve = [], []
    started = time.monotonic()
    eval_nfe = 0
    for step in range(1, p["steps"] + 1):
        s = torch.from_numpy(np.random.choice(np.arange(p["data_size"]-p["batch_time"], dtype=np.int64), p["batch_size"], replace=False))
        batch_y0 = truth[s]
        batch_t = t[:p["batch_time"]]
        batch_y = torch.stack([truth[s+i] for i in range(p["batch_time"])], dim=0)
        optimizer.zero_grad()
        loss = torch.mean(torch.abs(solve(batch_y0, batch_t)-batch_y))
        if not torch.isfinite(loss):
            raise ValueError("non-finite ODE training loss")
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if step % 20 == 0:
            before = model.nfe
            with torch.no_grad():
                fit = float(torch.mean(torch.abs(solve(y0, t)-truth)))
            eval_nfe += model.nfe-before
            fit_curve.append({"step": step, "mae": fit})
            print(f"Iter {step:04d} | Total Loss {fit:.6f}", flush=True)
    before = model.nfe
    with torch.no_grad():
        prediction = solve(y0, t).numpy().reshape(-1, 2)
    eval_nfe += model.nfe-before
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    np.save(out / "prediction.npy", prediction, allow_pickle=False)
    torch.save(model.state_dict(), out / "checkpoint.pt")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    training = {"parameters": p, "steps_completed": len(losses), "losses": losses, "fit_curve": fit_curve,
                "initial_fit_mae": initial_mae, "nfe_training": model.nfe-eval_nfe, "nfe_evaluation": eval_nfe,
                "training_elapsed_s": time.monotonic()-started,
                "prediction_sha256": digest(out / "prediction.npy"), "checkpoint_sha256": digest(out / "checkpoint.pt"),
                "spec_sha256": cfg["spec_sha256"], "torch": torch.__version__, "device": "cpu", "seed": p["seed"]}
    (out / "training.json").write_text(json.dumps(training, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
