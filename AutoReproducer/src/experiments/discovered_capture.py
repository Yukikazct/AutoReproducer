"""Frozen project wrapper: observe author training, export, then fresh evaluation.

This file is copied verbatim to a fresh repository workspace. It never supplies a
model, dataset, optimizer or training loop; those must come from author source.
"""
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

from author_values import geometric_data_fields, resolve_author_value
from split_counts import expected_from_census

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
    for name, record in config.get('data_files', {}).items():
        path = (WORKSPACE / name).resolve(strict=True)
        if (not path.is_relative_to(WORKSPACE) or sha(path) != record['sha256']
                or path.stat().st_size != record['bytes']):
            raise ValueError('Frozen original author dataset changed: ' + name)


def load_native_operators(config):
    """Register only frozen graph libraries before loading the exported graph."""
    modules = config.get('operator_imports', [])
    if (not isinstance(modules, list) or len(modules) != len(set(modules))
            or any(name not in {'torch_sparse', 'torch_scatter', 'dgl'} for name in modules)):
        raise ValueError('Independent evaluation operator imports are not from reviewed graph libraries')
    for name in modules:
        if name == 'dgl':
            os.environ['DGLBACKEND'] = 'pytorch'
        importlib.import_module(name)


def prepare_output_directories(config):
    """Prepare only parents compiled from source-bound author save statements."""
    for item in config['train'].get('output_directories', []):
        source = item['source_path']
        if config['source_files'].get(source) != item['source_sha256']:
            raise ValueError('Author output parent lacks pinned source evidence')
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Author output parent escaped the repository')
        target = (WORKSPACE / relative).resolve()
        if not target.is_relative_to(WORKSPACE):
            raise ValueError('Author output parent escaped the repository')
        for parent in [WORKSPACE / relative, *(WORKSPACE / relative).parents]:
            if parent == WORKSPACE:
                break
            if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400):
                raise ValueError('Author output parent contains a link or junction')
        target.mkdir(parents=True, exist_ok=True)


def author_integer_tensor(value):
    if isinstance(value, torch.Tensor):
        if value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
            return value.to(dtype=torch.int64)
        if value.dtype == torch.bool:
            return value
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        if value.dtype.kind == 'i' or value.dtype.kind == 'u' and (not value.size or value.max() <= np.iinfo(np.int64).max):
            return torch.from_numpy(np.array(value, dtype=np.int64, copy=True))
        if value.dtype.kind == 'b':
            return torch.from_numpy(value.copy())
    raise ValueError('Author labels/indices must be actual integer tensors or arrays without rounding')


def select(output, labels, indices, expected, output_alignment='full_dataset'):
    if output_alignment not in {'full_dataset', 'selected_test'}:
        raise ValueError('Unknown author output alignment')
    if (not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] < 2
            or not torch.isfinite(output).all() or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1 or labels.dtype != torch.int64
            or output_alignment == 'full_dataset' and labels.shape[0] != output.shape[0]):
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
        if output_alignment == 'selected_test':
            if output.shape[0] != len(indices):
                raise ValueError('Selected author logits do not contain exactly the complete test rows')
            labels = labels[indices.to(labels.device)]
        else:
            output, labels = output[indices.to(output.device)], labels[indices.to(labels.device)]
    elif output_alignment != 'full_dataset':
        raise ValueError('Selected-test logits require the original full-data test indices')
    if len(labels) != expected or labels.min().item() < 0 or labels.max().item() >= output.shape[1]:
        raise ValueError("The captured test split does not match the prospective complete sample count")
    return output, labels


def cpu_tensor(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Named author capture must contain real CPU tensors")
    return value.detach().clone()


def expected_test_count(contract, index, labels):
    specification = contract['expected_test_samples']
    if not isinstance(specification, dict):
        return specification[index], None
    classes, counts = torch.unique(labels.detach().cpu(), sorted=True, return_counts=True)
    if not torch.equal(classes, torch.arange(len(classes), dtype=classes.dtype)):
        raise ValueError('Source-derived class split requires complete labels numbered from zero')
    census = counts.tolist()
    return expected_from_census(specification, census), census


def flatten(value, tensors):
    """Trace sparse author inputs via exact index/value components, never dense copies."""
    if value is None:
        return {'kind': 'none'}
    fields = geometric_data_fields(value)
    if fields is not None:
        if (not 1 <= len(fields) <= 64 or any(not isinstance(key, str)
                or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,127}', key) for key in fields)):
            raise ValueError('Author PyG Data must have a bounded complete mapping of original fields')
        return {'kind': 'torch_geometric_data', 'fields': {key: flatten(item, tensors) for key, item in fields.items()}}
    if isinstance(value, torch.Tensor):
        def append(tensor):
            if len(tensors) >= 64:
                raise ValueError("Captured author tensor inputs exceed the component bound")
            tensors.append(tensor.detach().cpu().clone())
            return len(tensors) - 1

        if value.layout == torch.strided:
            return {"kind": "tensor", "index": append(value)}
        if value.layout == torch.sparse_coo:
            value = value.detach().cpu().coalesce()
            return {"kind": "sparse_coo", "indices": append(value.indices()),
                    "values": append(value.values()), "size": list(value.size())}
        if value.layout == torch.sparse_csr:
            return {"kind": "sparse_csr", "crow": append(value.crow_indices()),
                    "col": append(value.col_indices()), "values": append(value.values()),
                    "size": list(value.size())}
        raise ValueError("Unsupported author tensor layout")
    if type(value) in (bool, int, float) and (not isinstance(value, float) or np.isfinite(value)):
        return {"kind": "scalar", "value": value}
    if type(value) in (list, tuple) and 0 < len(value) <= 64:
        return {"kind": type(value).__name__, "items": [flatten(item, tensors) for item in value]}
    raise ValueError("Capture forward arguments must be real tensor trees or fixed author scalars")


def reconstruct(layout, tensors):
    if layout['kind'] == 'none':
        return None
    if layout["kind"] == "tensor":
        return tensors[layout["index"]]
    if layout["kind"] == "sparse_coo":
        return torch.sparse_coo_tensor(tensors[layout["indices"]], tensors[layout["values"]],
                                       size=layout["size"], is_coalesced=True)
    if layout["kind"] == "sparse_csr":
        return torch.sparse_csr_tensor(tensors[layout["crow"]], tensors[layout["col"]],
                                       tensors[layout["values"]], size=layout["size"])
    if layout["kind"] == "scalar":
        return layout["value"]
    if layout['kind'] == 'torch_geometric_data':
        from torch_geometric.data import Data
        return Data(**{key: reconstruct(value, tensors) for key, value in layout['fields'].items()})
    items = [reconstruct(item, tensors) for item in layout["items"]]
    return tuple(items) if layout["kind"] == "tuple" else items


class FlattenedAuthor(torch.nn.Module):
    def __init__(self, model, layouts):
        super().__init__()
        self.author_model, self.layouts = model, layouts

    def forward(self, *tensors):
        return self.author_model(*(reconstruct(layout, tensors) for layout in self.layouts))


def cpu_snapshot(model):
    """Move copied parameters, buffers AND plain cached tensor trees to CPU."""
    registered = [(module, dict(module._parameters), dict(module._buffers)) for module in model.modules()]
    model.cpu()
    memo, active = {}, set()
    for module, parameters, buffers in registered:
        for original, current in ((parameters, module._parameters), (buffers, module._buffers)):
            for name, tensor in original.items():
                if tensor is not None:
                    memo[id(tensor)] = current[name]

    def move(value, depth=0):
        identifier = id(value)
        if identifier in active:
            raise ValueError('Copied model cache tensor tree contains a cycle')
        if identifier in memo:
            return memo[identifier]
        if depth > 16:
            raise ValueError('Copied model cache tensor tree exceeds the supported nesting bound')
        if isinstance(value, torch.Tensor):
            result = value.detach().cpu().clone()
        elif type(value) in (list, tuple, dict):
            active.add(identifier)
            try:
                if type(value) is dict:
                    result = {key: move(item, depth + 1) for key, item in value.items()}
                else:
                    values = [move(item, depth + 1) for item in value]
                    result = tuple(values) if type(value) is tuple else values
            finally:
                active.remove(identifier)
        else:
            return value
        memo[identifier] = result
        return result
    for module in model.modules():
        # Operate on an already copied module; never alter the training instance.
        for name, value in list(vars(module).items()):
            if name not in {'_parameters', '_buffers', '_modules'}:
                vars(module)[name] = move(value)
    return model


def capture(config):
    check_sources(config)
    prepare_output_directories(config)
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
    model = resolve_author_value(namespace, contract['model'])
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
    inputs = tuple(resolve_author_value(namespace, name) for name in contract['forward_args'])
    raw_labels = resolve_author_value(namespace, contract['labels'])
    labels = cpu_tensor(author_integer_tensor(raw_labels))
    indices = None if contract['test_indices'] is None else cpu_tensor(
        author_integer_tensor(resolve_author_value(namespace, contract['test_indices'])))
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)
        select(outputs, labels, indices, contract['expected_test_samples'], contract.get('output_alignment', 'full_dataset'))
        # Trace only the author's trained forward path; independent evaluation
        # also checks every output against the original author module's result.
        tensors = []
        layouts = [flatten(value, tensors) for value in inputs]
        if not tensors or sum(value.numel() * value.element_size() for value in tensors) > 2 * 1024**3:
            raise ValueError("Captured author inputs exceed the frozen memory bound")
        wrapper = FlattenedAuthor(model, layouts).eval()
        traced = torch.jit.trace(wrapper, tuple(tensors), check_trace=True, strict=True)
        if not torch.allclose(traced(*tensors), outputs, atol=1e-6, rtol=1e-5):
            raise ValueError("Exported author forward path differs from the trained model")
        torch.jit.save(traced, str(ROOT / "model.pt"))
        torch.save({"inputs": tuple(tensors), "labels": labels, "test_indices": indices,
                    "outputs": outputs.detach()}, ROOT / "tensors.pt")
    check_sources(config)
    write("capture.json", {"version": 1, "pass": True, "spec_sha256": config["spec_sha256"],
          "optimizer_steps": updates, "parameters_changed": changed,
          "test_samples": contract["expected_test_samples"], "total_samples": len(labels),
          "classes": outputs.shape[1], "model_source": name,
          "author_label_dtype": str(raw_labels.dtype),
          "output_alignment": contract.get('output_alignment', 'full_dataset'),
          "model_source_sha256": config["source_files"][name],
          "model_type": type(model).__module__ + "." + type(model).__qualname__,
          "torch_version": torch.__version__, "native_torch_model": built_in,
          "files_sha256": {name: sha(ROOT / name) for name in ("model.pt", "tensors.pt")}})
    print("AUTHOR_CAPTURE_COMPLETE " + json.dumps({"optimizer_steps": updates,
          "test_samples": contract["expected_test_samples"]}), flush=True)


def evaluate(config):
    check_sources(config)
    load_native_operators(config)
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
                                config['capture']['expected_test_samples'],
                                config['capture'].get('output_alignment', 'full_dataset'))
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
