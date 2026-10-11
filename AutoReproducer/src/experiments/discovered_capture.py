"""Frozen project wrapper: observe author training, export, then fresh evaluation.

This file is copied verbatim to a fresh repository workspace. It never supplies a
model, dataset, optimizer or training loop; those must come from author source.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

import torch
import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check_sources(config):
    for name, expected in config["source_files"].items():
        path = (WORKSPACE / name).resolve(strict=True)
        if not path.is_relative_to(WORKSPACE) or sha(path) != expected:
            raise ValueError("Author source or bundled input changed: " + name)


def select(output, labels, indices, expected):
    if (not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] < 2
            or not torch.isfinite(output).all() or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1 or labels.dtype != torch.int64 or labels.shape[0] != output.shape[0]):
        raise ValueError("Capture requires finite class logits and corresponding integer class labels")
    if indices is not None:
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
            raise ValueError("Test selector must be a one-dimensional author tensor")
        if indices.dtype == torch.bool:
            if len(indices) != len(labels):
                raise ValueError("Boolean test selector has the wrong full-data length")
            indices = indices.nonzero().flatten()
        if (indices.dtype != torch.int64 or len(indices) != len(indices.unique())
                or not len(indices) or indices.min().item() < 0 or indices.max().item() >= len(labels)):
            raise ValueError("Test indices must be unique, complete, in-range integer indices")
        output, labels = output[indices], labels[indices]
    if len(labels) != expected or labels.min().item() < 0 or labels.max().item() >= output.shape[1]:
        raise ValueError("The captured test split does not match the prospective complete sample count")
    return output, labels


def cpu_tensor(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Named author capture must contain real CPU tensors")
    return value.detach().clone()


def capture(config):
    check_sources(config)
    if torch.cuda.is_available():
        raise ValueError("Frozen CPU execution unexpectedly exposes CUDA")
    observers = {}

    def before(optimizer, args, kwargs):
        key = id(optimizer)
        if key not in observers:
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            observers[key] = {"optimizer": optimizer, "parameters": parameters,
                              "initial": [p.detach().cpu().clone() for p in parameters], "steps": 0}

    def after(optimizer, args, kwargs):
        observers[id(optimizer)]["steps"] += 1

    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
    pre, post = register_optimizer_step_pre_hook(before), register_optimizer_step_post_hook(after)
    step = config["train"]
    argv = step["argv"]
    position = 2 if argv[1] == "-u" else 1
    cwd = (WORKSPACE / step["cwd"]).resolve(strict=True)
    script = (cwd / argv[position]).resolve(strict=True)
    if not cwd.is_relative_to(WORKSPACE) or not script.is_relative_to(WORKSPACE):
        raise ValueError("Author entrypoint escaped its frozen repository")
    os.chdir(cwd)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *argv[position + 1:]]
    try:
        namespace = runpy.run_path(str(script), run_name="__main__")
    finally:
        pre.remove()
        post.remove()
    contract = config["capture"]
    model = namespace[contract["model"]]
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Named author model is not a trained PyTorch module")
    implementation = getattr(getattr(type(model), "forward", None), "__code__", None)
    definition = Path(implementation.co_filename if implementation is not None
                      else inspect.getfile(type(model))).resolve(strict=True)
    built_in = type(model).__module__.startswith("torch.nn.")
    if built_in:
        # A native nn.Linear/Sequential may be constructed directly by the
        # author. Its construction stays in the unchanged entrypoint; its
        # parameters must still belong to the observed trained optimizer.
        definition = script
    if not definition.is_relative_to(WORKSPACE):
        raise ValueError("Model must be defined or constructed by pinned author source")
    name = definition.relative_to(WORKSPACE).as_posix()
    if name not in config["source_files"] or sha(definition) != config["source_files"][name]:
        raise ValueError("Model definition is not covered by the pinned author source manifest")
    parameters = {id(p) for p in model.parameters()}
    relevant = [item for item in observers.values() if parameters.intersection(id(p) for p in item["parameters"])]
    updates = sum(item["steps"] for item in relevant)
    changed = any(not torch.equal(p.detach().cpu(), initial)
                  for item in relevant for p, initial in zip(item["parameters"], item["initial"])
                  if id(p) in parameters)
    if updates != contract["expected_optimizer_steps"] or not changed:
        raise ValueError("Observed model optimizer steps or actual parameter changes do not prove full training")
    inputs = tuple(cpu_tensor(namespace[name]) for name in contract["forward_args"])
    labels = cpu_tensor(namespace[contract["labels"]])
    indices = None if contract["test_indices"] is None else cpu_tensor(namespace[contract["test_indices"]])
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)
        select(outputs, labels, indices, contract["expected_test_samples"])
        # Trace only the author's trained forward path; independent evaluation
        # also checks every output against the original author module's result.
        traced = torch.jit.trace(model, inputs, check_trace=True, strict=True)
        if not torch.allclose(traced(*inputs), outputs, atol=1e-6, rtol=1e-5):
            raise ValueError("Exported author forward path differs from the trained model")
        torch.jit.save(traced, str(ROOT / "model.pt"))
        torch.save({"inputs": inputs, "labels": labels, "test_indices": indices,
                    "outputs": outputs.detach()}, ROOT / "tensors.pt")
    check_sources(config)
    write("capture.json", {"version": 1, "pass": True, "spec_sha256": config["spec_sha256"],
          "optimizer_steps": updates, "parameters_changed": changed,
          "test_samples": contract["expected_test_samples"], "total_samples": len(labels),
          "classes": outputs.shape[1], "model_source": name,
          "model_source_sha256": config["source_files"][name],
          "model_type": type(model).__module__ + "." + type(model).__qualname__,
          "torch_version": torch.__version__, "native_torch_model": built_in,
          "files_sha256": {name: sha(ROOT / name) for name in ("model.pt", "tensors.pt")}})
    print("AUTHOR_CAPTURE_COMPLETE " + json.dumps({"optimizer_steps": updates,
          "test_samples": contract["expected_test_samples"]}), flush=True)


def evaluate(config):
    check_sources(config)
    receipt = read("capture.json")
    if receipt["spec_sha256"] != config["spec_sha256"] or receipt.get("pass") is not True:
        raise ValueError("Training capture is not from this frozen run")
    for name in ("model.pt", "tensors.pt"):
        if receipt["files_sha256"][name] != sha(ROOT / name):
            raise ValueError("Captured training artifact changed before fresh evaluation")
    tensors = torch.load(ROOT / "tensors.pt", map_location="cpu", weights_only=True)
    model = torch.jit.load(str(ROOT / "model.pt"), map_location="cpu").eval()
    with torch.no_grad():
        outputs = model(*tensors["inputs"])
        if not torch.allclose(outputs, tensors["outputs"], atol=1e-6, rtol=1e-5):
            raise ValueError("Freshly loaded author model disagrees with captured predictions")
        logits, labels = select(outputs, tensors["labels"], tensors["test_indices"],
                                config["capture"]["expected_test_samples"])
        values, targets = logits.double().numpy(), labels.numpy()
        accuracy = float(np.mean(np.argmax(values, axis=1) == targets))
        shifted = values - np.max(values, axis=1, keepdims=True)
        cross_entropy = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]))
        torch_accuracy = (logits.argmax(dim=1) == labels).double().mean().item()
        torch_loss = torch.nn.functional.cross_entropy(logits.double(), labels).item()
        if abs(accuracy - torch_accuracy) > 1e-12 or not np.isclose(cross_entropy, torch_loss, rtol=1e-10, atol=1e-12):
            raise ValueError("Independent NumPy metrics disagree with PyTorch cross-check")
    check_sources(config)
    result = {"version": 1, "pass": True, "spec_sha256": config["spec_sha256"],
              "capture_sha256": sha(ROOT / "capture.json"), "predictions_match": True,
              "test_samples": len(labels), "accuracy": accuracy, "cross_entropy": cross_entropy,
              "evaluation": "fresh_process_torchscript_forward_numpy_metrics", "device": "cpu",
              "numpy_torch_crosscheck": True}
    write("independent_metrics.json", result)
    print("INDEPENDENT_EVALUATION_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    configuration = read("config.json")
    {"capture": capture, "evaluate": evaluate}[sys.argv[1]](configuration)
