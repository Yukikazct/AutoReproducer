"""Reconstruct trajectories independently with double-precision SciPy DOP853."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def reference(initial, times):
    matrix = np.array([[-.1, 2.], [-2., -.1]])
    solution = solve_ivp(lambda t, y: (y**3) @ matrix, (0., 25.), initial, t_eval=times,
                         method="DOP853", rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError("independent ODE reference failed")
    return solution.y.T


def main():
    cfg = json.loads(Path("experiment.json").read_text(encoding="utf-8")); p = cfg["parameters"]
    split = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if split not in {"fit", "validation", "holdout"}:
        raise ValueError("unknown evaluation split")
    out = Path("artifacts")
    t = np.linspace(0., 25., p["data_size"])
    import torch
    from torchdiffeq import odeint
    from author_model import ODEFunc
    torch.set_num_threads(2)
    model = ODEFunc()
    model.load_state_dict(torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True))
    model.eval()
    if split == "fit":
        predictions = np.load(out / "prediction.npy", allow_pickle=False)[None, ...]
        truths = reference([2., 0.], t)[None, ...]
    else:
        initials = [[1.5, 0.], [0., 1.5]] if split == "validation" else [[1.75, 0.], [0., 1.75]]
        predictions, truths = [], []
        for initial in initials:
            with torch.no_grad():
                trajectory = odeint(model, torch.tensor([initial]), torch.linspace(0.,25.,p["data_size"]),
                                     method=p["solver"], rtol=p["rtol"], atol=p["atol"])
            predictions.append(trajectory.numpy().reshape(-1,2))
            truths.append(reference(initial, t))
        predictions, truths = np.array(predictions), np.array(truths)
        np.save(out / f"prediction_{split}.npy", predictions, allow_pickle=False)
    if predictions.shape != truths.shape or not np.isfinite(predictions).all():
        raise ValueError("invalid ODE prediction shape or values")
    error = predictions.astype(np.float64)-truths
    metrics = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}
    result = {"pass": True, "metrics": metrics, "split": split, "samples": int(truths.size),
              "spec_sha256": cfg["spec_sha256"],
              "prediction_sha256": hashlib.sha256((out / "prediction.npy").read_bytes()).hexdigest()}
    if split == "fit":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,3,figsize=(14,4))
        for i in range(2):
            axs[0].plot(t, truths[0,:,i], label=f"True y{i}")
            axs[0].plot(t, predictions[0,:,i], '--', label=f"Pred y{i}")
        axs[0].set(title="Trajectories", xlabel="Time"); axs[0].legend()
        axs[1].plot(truths[0,:,0],truths[0,:,1],label="True")
        axs[1].plot(predictions[0,:,0],predictions[0,:,1],'--',label="Predicted")
        axs[1].set(title="Phase portrait", xlabel="y0", ylabel="y1"); axs[1].legend()
        x,y=np.meshgrid(np.linspace(-2,2,21),np.linspace(-2,2,21))
        with torch.no_grad():
            field=model(0,torch.tensor(np.stack([x,y],-1).reshape(-1,2),dtype=torch.float32)).numpy().reshape(21,21,2)
        axs[2].streamplot(x,y,field[:,:,0],field[:,:,1]); axs[2].set(title="Learned vector field")
        fig.tight_layout(); fig.savefig(out/'trajectories.png',dpi=130); plt.close(fig)
        training=json.loads((out/'training.json').read_text(encoding='utf-8'))
        fig,ax=plt.subplots(figsize=(7,4));ax.semilogy([v['step'] for v in training['fit_curve']],[v['mae'] for v in training['fit_curve']])
        ax.set(xlabel='Iteration',ylabel='Full trajectory MAE',title='Neural ODE: official fit')
        fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=130);plt.close(fig)
    (out / f"metrics_{split}.json").write_text(json.dumps(result,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()
