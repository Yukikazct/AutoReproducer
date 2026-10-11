"""Observe all native author repetitions without changing their training loop."""
import ast
from copy import deepcopy
import dis
import inspect
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

_support_path = Path(__file__).resolve().with_name("common.py")
_support_spec = importlib.util.spec_from_file_location("_autorepro_capture_support", _support_path)
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
ROOT, WORKSPACE = _support.ROOT, _support.WORKSPACE
check_sources, read, select, sha, write = (_support.check_sources, _support.read,
                                         _support.select, _support.sha, _support.write)


def artifact_names(index):
    return [f"run_{index:03d}_{suffix}" for suffix in ("model.pt", "tensors.pt")]


def flatten(value, tensors):
    if isinstance(value, torch.Tensor):
        if len(tensors) >= 64 or value.layout != torch.strided:
            raise ValueError("Capture forward tensors must be bounded dense author inputs")
        tensors.append(value.detach().cpu().clone())
        return {"kind": "tensor", "index": len(tensors) - 1}
    if type(value) in (bool, int, float) and (not isinstance(value, float) or np.isfinite(value)):
        return {"kind": "scalar", "value": value}
    if type(value) in (list, tuple) and 0 < len(value) <= 64:
        return {"kind": type(value).__name__, "items": [flatten(item, tensors) for item in value]}
    raise ValueError("Capture forward arguments must be real tensor trees or fixed author scalars")


def reconstruct(layout, tensors):
    if layout["kind"] == "tensor":
        return tensors[layout["index"]]
    if layout["kind"] == "scalar":
        return layout["value"]
    items = [reconstruct(item, tensors) for item in layout["items"]]
    return tuple(items) if layout["kind"] == "tuple" else items


class FlattenedAuthor(torch.nn.Module):
    def __init__(self, model, layouts):
        super().__init__()
        self.author_model, self.layouts = model, layouts

    def forward(self, *tensors):
        return self.author_model(*(reconstruct(layout, tensors) for layout in self.layouts))


def model_source(model, script, config):
    implementation = getattr(getattr(type(model), "forward", None), "__code__", None)
    built_in = type(model).__module__.startswith("torch.nn.")
    definition = script if built_in else Path(implementation.co_filename if implementation is not None
                                             else inspect.getfile(type(model))).resolve(strict=True)
    if not definition.is_relative_to(WORKSPACE):
        raise ValueError("Observed model must come from pinned author source")
    name = definition.relative_to(WORKSPACE).as_posix()
    if name not in config["source_files"] or sha(definition) != config["source_files"][name]:
        raise ValueError("Observed model definition changed or is absent from the pinned manifest")
    return name


def capture_return(namespace, index, observers, script, config):
    contract = config["capture"]
    model = namespace[contract["model"]]
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Natural author return does not expose its trained model")
    name = model_source(model, script, config)
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    relevant = [(key, item) for key, item in observers.items()
                if parameter_ids.intersection(id(parameter) for parameter in item["parameters"])]
    updates = sum(item["steps"] for _, item in relevant)
    changed = any(not torch.equal(parameter.detach().cpu(), initial)
                  for _, item in relevant for parameter, initial in zip(item["parameters"], item["initial"])
                  if id(parameter) in parameter_ids)
    if not changed or not 0 < updates <= contract["expected_optimizer_steps"]:
        raise ValueError("Observed optimizer steps and actual model changes do not prove training")
    rule, counter = contract["completion"], None
    if rule["kind"] == "fixed_steps":
        if updates != contract["expected_optimizer_steps"]:
            raise ValueError("Native author repetition did not finish all fixed optimizer steps")
    else:
        counter = namespace[rule["epoch_counter"]]
        if (type(counter) is not int or counter < 0
                or (counter + rule["epoch_offset"]) * rule["updates_per_epoch"] != updates):
            raise ValueError("Observed optimizer steps disagree with the native author's epoch counter")
    if any(parameter.device.type != contract["device"] for parameter in model.parameters()):
        raise ValueError("The actual author's trained model device differs from the frozen plan")
    inputs = tuple(namespace[key] for key in contract["forward_args"])
    labels = namespace[contract["labels"]]
    indices = None if contract["test_indices"] is None else namespace[contract["test_indices"]]
    if not isinstance(labels, torch.Tensor) or (indices is not None and not isinstance(indices, torch.Tensor)):
        raise ValueError("Natural return must expose the actual author labels and complete test selector")
    expected = contract["expected_test_samples"][index]
    # Snapshot the model: observing a natural return must not change the
    # author's module mode, device or subsequent native program behavior.
    model = deepcopy(model).eval()
    with torch.no_grad():
        original = model(*inputs)
        select(original, labels, indices, expected)
        originals = original.detach().cpu().clone()
        tensors = []
        layouts = [flatten(value, tensors) for value in inputs]
        if not tensors or sum(value.numel() * value.element_size() for value in tensors) > 2 * 1024**3:
            raise ValueError("Captured tensor inputs are absent or exceed the frozen memory bound")
        labels = labels.detach().cpu().clone()
        indices = None if indices is None else indices.detach().cpu().clone()
        wrapper = FlattenedAuthor(model.cpu(), layouts).eval()
        outputs = wrapper(*tensors)
        if (not torch.allclose(outputs, originals, atol=1e-6, rtol=1e-5)
                or not torch.equal(outputs.argmax(1), originals.argmax(1))):
            raise ValueError("CPU export changed the author's original device predictions")
        traced = torch.jit.trace(wrapper, tuple(tensors), check_trace=True, strict=True)
        if (not torch.allclose(traced(*tensors), outputs, atol=1e-6, rtol=1e-5)
                or not torch.equal(traced(*tensors).argmax(1), outputs.argmax(1))):
            raise ValueError("Export changed the captured author's forward path")
        model_file, tensor_file = artifact_names(index)
        torch.jit.save(traced, str(ROOT / model_file))
        torch.save({"inputs": tuple(tensors), "labels": labels, "test_indices": indices,
                    "outputs": outputs.detach()}, ROOT / tensor_file)
    for key, _ in relevant:
        del observers[key]
    record = {"index": index, "natural_return": True, "optimizer_steps": updates,
              "author_epoch_counter": counter, "parameters_changed": True,
              "test_samples": expected, "total_samples": len(labels), "classes": outputs.shape[1],
              "model_source": name, "model_source_sha256": config["source_files"][name],
              "files_sha256": {name: sha(ROOT / name) for name in artifact_names(index)}}
    print(f"AUTHOR_NATIVE_RUN_CAPTURED index={index} steps={updates} test_samples={expected}", flush=True)
    return record


def capture(config):
    check_sources(config)
    contract = config["capture"]
    if (contract["device"] == "cuda") != torch.cuda.is_available():
        raise ValueError("Native training device availability differs from the frozen plan")
    argv, step = config["train"]["argv"], config["train"]
    position = 2 if argv[1] == "-u" else 1
    cwd = (WORKSPACE / step["cwd"]).resolve(strict=True)
    script = (cwd / argv[position]).resolve(strict=True)
    scope_path = (WORKSPACE / contract["scope"]["path"]).resolve(strict=True)
    if not all(path.is_relative_to(WORKSPACE) for path in (cwd, script, scope_path)):
        raise ValueError("Author entrypoint or capture scope escaped the workspace")
    tree = ast.parse(scope_path.read_text(encoding="utf-8"))
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == contract["scope"]["function"])
    first_line = min([target.lineno] + [node.lineno for node in target.decorator_list])
    observers, runs, return_instructions = {}, [], {}

    def before(optimizer, args, kwargs):
        key = id(optimizer)
        if key not in observers:
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            observers[key] = {"optimizer": optimizer, "parameters": parameters,
                              "initial": [p.detach().cpu().clone() for p in parameters], "steps": 0}

    def after(optimizer, args, kwargs):
        observers[id(optimizer)]["steps"] += 1

    def profile(frame, event, result):
        code = frame.f_code
        if (event != "return" or code.co_name != target.name or code.co_firstlineno != first_line
                or Path(code.co_filename).resolve() != scope_path):
            return
        if code not in return_instructions:
            return_instructions[code] = {instruction.offset for instruction in dis.get_instructions(code)
                                         if instruction.opname in {"RETURN_VALUE", "RETURN_CONST"}}
        if frame.f_lasti not in return_instructions[code]:
            return  # Exception unwinding is not a natural completion.
        if len(runs) >= contract["expected_runs"]:
            raise ValueError("Author entrypoint executed more native repetitions than the frozen protocol")
        rule = contract["completion"]
        if rule["kind"] == "author_epoch_counter" and rule["epoch_counter"] not in frame.f_locals:
            raise ValueError("Author completion counter must come from the observed function's actual locals")
        namespace = {**frame.f_globals, **frame.f_locals}
        runs.append(capture_return(namespace, len(runs), observers, script, config))

    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
    pre, post = register_optimizer_step_pre_hook(before), register_optimizer_step_post_hook(after)
    old_profile = sys.getprofile()
    os.chdir(cwd)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *argv[position + 1:]]
    try:
        sys.setprofile(profile)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.setprofile(old_profile)
        pre.remove()
        post.remove()
    if len(runs) != contract["expected_runs"]:
        raise ValueError("Author entrypoint did not naturally finish every declared native repetition")
    if observers:
        raise ValueError("Training updates outside the captured complete native series remain unaccounted")
    check_sources(config)
    write("capture.json", {"version": 1, "kind": contract["kind"], "pass": True,
          "spec_sha256": config["spec_sha256"], "aggregation": "mean", "training_device": contract["device"],
          "torch_version": torch.__version__, "runs": runs, "parameters_changed": True,
          "optimizer_steps": sum(run["optimizer_steps"] for run in runs),
          "test_samples": [run["test_samples"] for run in runs],
          "files_sha256": {name: digest for run in runs for name, digest in run["files_sha256"].items()}})
    print("AUTHOR_CAPTURE_COMPLETE native_runs=" + str(len(runs)), flush=True)


def evaluate(config):
    check_sources(config)
    receipt = read("capture.json")
    contract, runs = config["capture"], []
    if (receipt.get("pass") is not True or receipt["spec_sha256"] != config["spec_sha256"]
            or len(receipt["runs"]) != contract["expected_runs"]):
        raise ValueError("Complete native series capture is missing or belongs to another run")
    for index, captured in enumerate(receipt["runs"]):
        model_file, tensor_file = artifact_names(index)
        for name in (model_file, tensor_file):
            if captured["files_sha256"][name] != sha(ROOT / name):
                raise ValueError("Native series artifact changed before fresh evaluation")
        tensors = torch.load(ROOT / tensor_file, map_location="cpu", weights_only=True)
        model = torch.jit.load(str(ROOT / model_file), map_location="cpu").eval()
        with torch.no_grad():
            output = model(*tensors["inputs"])
            if (not torch.allclose(output, tensors["outputs"], atol=1e-6, rtol=1e-5)
                    or not torch.equal(output.argmax(1), tensors["outputs"].argmax(1))):
                raise ValueError("Fresh series model changed the captured predictions")
            logits, labels = select(output, tensors["labels"], tensors["test_indices"],
                                    contract["expected_test_samples"][index])
            values, targets = logits.double().numpy(), labels.numpy()
            accuracy = float(np.mean(np.argmax(values, axis=1) == targets))
            shifted = values - np.max(values, axis=1, keepdims=True)
            loss = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]))
            torch_accuracy = (logits.argmax(1) == labels).double().mean().item()
            torch_loss = torch.nn.functional.cross_entropy(logits.double(), labels).item()
            if abs(accuracy - torch_accuracy) > 1e-12 or not np.isclose(loss, torch_loss, rtol=1e-10, atol=1e-12):
                raise ValueError("Independent native-run NumPy metrics disagree with PyTorch")
        runs.append({"index": index, "test_samples": len(labels), "accuracy": accuracy, "cross_entropy": loss,
                     "predictions_match": True, "numpy_torch_crosscheck": True})
        print(f"INDEPENDENT_NATIVE_RUN_EVALUATED index={index} accuracy={accuracy}", flush=True)
    check_sources(config)
    write("independent_metrics.json", {"version": 1, "pass": True, "spec_sha256": config["spec_sha256"],
          "capture_sha256": sha(ROOT / "capture.json"), "aggregation": "mean", "runs": runs,
          "accuracy": float(np.mean([run["accuracy"] for run in runs])),
          "cross_entropy": float(np.mean([run["cross_entropy"] for run in runs])),
          "test_samples": [run["test_samples"] for run in runs],
          "predictions_match": True, "numpy_torch_crosscheck": True, "device": "cpu",
          "evaluation": "fresh_process_all_native_models_numpy_mean"})
    print("INDEPENDENT_EVALUATION_COMPLETE native_runs=" + str(len(runs)), flush=True)


if __name__ == "__main__":
    {"capture": capture, "evaluate": evaluate}[sys.argv[1]](read("config.json"))
