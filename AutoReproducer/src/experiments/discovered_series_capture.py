"""Observe all native author repetitions without changing their training loop."""
import ast
from copy import deepcopy
from contextlib import contextmanager
import dis
import inspect
import importlib.util
import math
import os
import random
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
from native_loop import resolve_module_loop, resolve_function_selection

_support_path = Path(__file__).resolve().with_name("common.py")
_support_spec = importlib.util.spec_from_file_location("_autorepro_capture_support", _support_path)
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
ROOT, WORKSPACE = _support.ROOT, _support.WORKSPACE
check_sources, read, select, sha, write = (_support.check_sources, _support.read,
                                         _support.select, _support.sha, _support.write)
flatten, FlattenedAuthor = _support.flatten, _support.FlattenedAuthor
resolve_author_value, author_integer_tensor = _support.resolve_author_value, _support.author_integer_tensor


@contextmanager
def preserve_rng():
    """Exporting a snapshot must not change later native random initializations."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=None if torch.cuda.is_available() else []):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def artifact_names(index):
    return [f"run_{index:03d}_{suffix}" for suffix in ("model.pt", "tensors.pt")]


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


@preserve_rng()
def capture_return(namespace, index, observers, script, config, retained=None):
    contract = config["capture"]
    model = retained['model'] if retained is not None else resolve_author_value(namespace, contract['model'])
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Natural author return does not expose its trained model")
    name = model_source(model, script, config)
    parameter_ids = retained['parameter_ids'] if retained is not None else {id(parameter) for parameter in model.parameters()}
    relevant = [(key, item) for key, item in observers.items()
                if parameter_ids.intersection(id(parameter) for parameter in item["parameters"])]
    updates = sum(item["steps"] for _, item in relevant)
    changed = any(not torch.equal(parameter.detach().cpu(), initial)
                  for _, item in relevant for parameter, initial in zip(item["parameters"], item["initial"])
                  if id(parameter) in parameter_ids)
    if not changed or not 0 < updates <= contract["expected_optimizer_steps"]:
        raise ValueError("Observed optimizer steps and actual model changes do not prove training")
    if retained is not None and (not retained['parameters_changed']
            or not 0 < retained['optimizer_steps'] <= updates):
        raise ValueError('Selected model must already have undergone real optimizer updates')
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
    inputs = retained['inputs'] if retained is not None else tuple(
        resolve_author_value(namespace, key) for key in contract['forward_args'])
    raw_labels = retained['labels'] if retained is not None else resolve_author_value(namespace, contract['labels'])
    labels = author_integer_tensor(raw_labels)
    raw_indices = retained['indices'] if retained is not None else (None if contract['test_indices'] is None
        else resolve_author_value(namespace, contract['test_indices']))
    indices = None if raw_indices is None else author_integer_tensor(raw_indices)
    expected, census = _support.expected_test_count(contract, index, labels)
    # Snapshot the model: observing a natural return must not change the
    # author's module mode, device or subsequent native program behavior.
    model = deepcopy(model).eval()
    with torch.no_grad():
        original = model(*inputs)
        selected, selected_labels = select(original, labels, indices, expected, contract.get('output_alignment', 'full_dataset'))
        if retained is not None:
            actual = (selected.argmax(1) == selected_labels.to(selected.device)).double().mean().item()
            if not math.isclose(actual, retained['accuracy'], rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError('Retained model does not reproduce the original validation-selected test result')
        originals = original.detach().cpu().clone()
        tensors = []
        layouts = [flatten(value, tensors) for value in inputs]
        if not tensors or sum(value.numel() * value.element_size() for value in tensors) > 2 * 1024**3:
            raise ValueError("Captured tensor inputs are absent or exceed the frozen memory bound")
        labels = labels.detach().cpu().clone()
        indices = None if indices is None else indices.detach().cpu().clone()
        wrapper = FlattenedAuthor(_support.cpu_snapshot(model), layouts).eval()
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
              "author_label_dtype": str(raw_labels.dtype),
              "output_alignment": contract.get('output_alignment', 'full_dataset'),
              "model_source": name, "model_source_sha256": config["source_files"][name],
              "files_sha256": {name: sha(ROOT / name) for name in artifact_names(index)}}
    if census is not None:
        record['split_census'] = census
    if retained is not None:
        record['validation_selection'] = {'source': contract['scope']['selection'],
                'accuracy': retained['accuracy'], 'optimizer_steps': retained['optimizer_steps']}
        if 'loop_counter' in contract['scope']:
            record.update(natural_return=False, native_loop_completed=True, author_loop_counter=index)
    print(f"AUTHOR_NATIVE_RUN_CAPTURED index={index} steps={updates} test_samples={expected}", flush=True)
    return record


def capture(config):
    check_sources(config)
    _support.prepare_output_directories(config)
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
    module_loop = 'loop_counter' in contract['scope']
    function_selection = not module_loop and 'selection' in contract['scope']
    if module_loop:
        loop = resolve_module_loop(scope_path.read_text('utf-8'), contract['scope'])
    else:
        target = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == contract["scope"]["function"])
        first_line = min([target.lineno] + [node.lineno for node in target.decorator_list])
        if function_selection:
            loop = resolve_function_selection(scope_path.read_text('utf-8'), contract['scope'])
    observers, runs, return_instructions = {}, [], {}
    loop_state = {'pending': False, 'retained': None, 'frame': None}

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
        if function_selection and loop_state['retained'] is None:
            raise ValueError('Complete native function returned without an original validation-selected model')
        runs.append(capture_return(namespace, len(runs), observers, script, config,
                                   retained=loop_state['retained'] if function_selection else None))
        if function_selection:
            loop_state.update(retained=None, pending=False, frame=None)

    @preserve_rng()
    def retain_selected(namespace):
        if module_loop:
            counter = namespace[contract['scope']['loop_counter']]
            if type(counter) is not int or counter != len(runs) or counter >= contract['expected_runs']:
                raise ValueError('Native module loop counter does not match the frozen complete repetition order')
        model = resolve_author_value(namespace, contract['model'])
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Validation selection does not expose the actual author model')
        model_source(model, script, config)
        ids = {id(parameter) for parameter in model.parameters()}
        relevant = [item for item in observers.values()
                    if ids.intersection(id(parameter) for parameter in item['parameters'])]
        changed = any(not torch.equal(parameter.detach().cpu(), initial)
            for item in relevant for parameter, initial in zip(item['parameters'], item['initial'])
            if id(parameter) in ids)
        value = namespace[contract['scope']['selection']['result']]
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().item()
        elif isinstance(value, np.floating):
            value = value.item()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Author validation-selected result must be a real finite scalar accuracy')
        value /= 100 if contract['scope']['selection']['result_unit'] == 'percent' else 1
        if not 0 <= value <= 1:
            raise ValueError('Author selected accuracy has inconsistent units')
        loop_state['retained'] = {'model': deepcopy(model), 'parameter_ids': ids,
            'parameters_changed': changed, 'optimizer_steps': sum(item['steps'] for item in relevant),
            'accuracy': value,
            'inputs': deepcopy(tuple(resolve_author_value(namespace, key) for key in contract['forward_args'])),
            'labels': deepcopy(resolve_author_value(namespace, contract['labels'])),
            'indices': None if contract['test_indices'] is None else deepcopy(
                resolve_author_value(namespace, contract['test_indices']))}

    def trace_module(frame, event, result):
        if event == 'exception':
            loop_state['pending'] = False
        elif event == 'line':
            line = frame.f_lineno
            namespace = {**frame.f_globals, **frame.f_locals}
            inside_assignment = loop['selection_start'] <= line <= loop['selection_end']
            if loop_state['pending'] and not inside_assignment:
                retain_selected(namespace)  # The original assignment has actually completed.
                loop_state['pending'] = False
            if inside_assignment:
                loop_state['pending'] = True
            if module_loop and line == loop['loop_line'] and loop_state['retained'] is not None:
                if namespace[contract['scope']['loop_counter']] != len(runs):
                    raise ValueError('Original module loop did not finish the selected repetition in order')
                runs.append(capture_return(namespace, len(runs), observers, script, config,
                                           retained=loop_state['retained']))
                loop_state['retained'] = None
        elif event == 'return' and function_selection and loop_state['pending']:
            if frame.f_lasti in {item.offset for item in dis.get_instructions(frame.f_code)
                                if item.opname in {'RETURN_VALUE', 'RETURN_CONST'}}:
                retain_selected({**frame.f_globals, **frame.f_locals})
            loop_state['pending'] = False
        return trace_module

    def trace(frame, event, result):
        if (event == 'call' and frame.f_code.co_name == loop['scope_name']
                and frame.f_code.co_filename == str(scope_path)
                and (loop['function_line'] is None or frame.f_code.co_firstlineno == loop['function_line'])):
            if function_selection:
                if loop_state['frame'] is not None:
                    raise ValueError('Native selected training function must not be recursive or overlapping')
                loop_state['frame'] = frame
            return trace_module
        return None

    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
    pre, post = register_optimizer_step_pre_hook(before), register_optimizer_step_post_hook(after)
    old_profile, old_trace = sys.getprofile(), sys.gettrace()
    os.chdir(cwd)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *argv[position + 1:]]
    try:
        if module_loop or function_selection:
            sys.settrace(trace)
        if not module_loop:
            sys.setprofile(profile)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.setprofile(old_profile)
        sys.settrace(old_trace)
        pre.remove()
        post.remove()
    if len(runs) != contract["expected_runs"]:
        raise ValueError("Author entrypoint did not naturally finish every declared native repetition")
    if observers:
        raise ValueError("Training updates outside the captured complete native series remain unaccounted")
    check_sources(config)
    write("capture.json", {"version": 1, "kind": contract["kind"], "pass": True,
          "spec_sha256": config["spec_sha256"], "aggregation": "mean", "training_device": contract["device"],
          "torch_version": torch.__version__, "process_pid": os.getpid(),
          "runs": runs, "parameters_changed": True,
          "optimizer_steps": sum(run["optimizer_steps"] for run in runs),
          "test_samples": [run["test_samples"] for run in runs],
          "files_sha256": {name: digest for run in runs for name, digest in run["files_sha256"].items()}})
    print("AUTHOR_CAPTURE_COMPLETE native_runs=" + str(len(runs)), flush=True)


def evaluate(config):
    check_sources(config)
    _support.load_native_operators(config)
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
        expected_samples, census = _support.expected_test_count(contract, index, tensors['labels'])
        if census is not None and census != captured.get('split_census'):
            raise ValueError('Independent complete-label census differs from training')
        model = torch.jit.load(str(ROOT / model_file), map_location="cpu").eval()
        with torch.no_grad():
            output = model(*tensors["inputs"])
            if (not torch.allclose(output, tensors["outputs"], atol=1e-6, rtol=1e-5)
                    or not torch.equal(output.argmax(1), tensors["outputs"].argmax(1))):
                raise ValueError("Fresh series model changed the captured predictions")
            logits, labels = select(output, tensors["labels"], tensors["test_indices"],
                                    expected_samples, contract.get('output_alignment', 'full_dataset'))
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
        if census is not None:
            runs[-1]['split_census'] = census
        print(f"INDEPENDENT_NATIVE_RUN_EVALUATED index={index} accuracy={accuracy}", flush=True)
    check_sources(config)
    write("independent_metrics.json", {"version": 1, "pass": True, "spec_sha256": config["spec_sha256"],
          "capture_sha256": sha(ROOT / "capture.json"), "aggregation": "mean", "runs": runs,
          "accuracy": float(np.mean([run["accuracy"] for run in runs])),
          "cross_entropy": float(np.mean([run["cross_entropy"] for run in runs])),
          "test_samples": [run["test_samples"] for run in runs],
          "predictions_match": True, "numpy_torch_crosscheck": True, "device": "cpu",
          "evaluation": "fresh_process_all_native_models_numpy_mean",
          "process_pid": os.getpid(), "capture_process_pid": receipt['process_pid']})
    print("INDEPENDENT_EVALUATION_COMPLETE native_runs=" + str(len(runs)), flush=True)


if __name__ == "__main__":
    {"capture": capture, "evaluate": evaluate}[sys.argv[1]](read("config.json"))
