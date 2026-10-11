"""Project-owned capture around unchanged, evidence-bound author entrypoints."""
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from src.method_adapters import digest, read_json, write_json
from src.safety.paths import workspace_path


SINGLE_CAPTURE_SCHEMA = {
    "kind": "torch_classification", "model": "author_identifier_or_fixed_key_selector",
    "forward_args": ["author_identifier_or_fixed_key_selector"], "labels": "author_integer_tensor_or_array_selector",
    "test_indices": "author_selector_or_null_for_complete_labels",
    "output_alignment": "full_dataset|selected_test (selected_test requires test_indices also in forward_args)",
    "expected_optimizer_steps": "positive_source_cited_integer",
    "expected_test_samples": "positive_source_cited_integer",
    "reported_metric": {"label": "literal_author_printed_accuracy_label", "unit": "fraction|percent"},
    "citations": ["repository_citation_ids"],
}
SERIES_CAPTURE_SCHEMA = {
    **SINGLE_CAPTURE_SCHEMA,
    "kind": "torch_classification_series", "device": "cpu|cuda",
    "scope": "{path, function} for a unique top-level function; OR {path, loop_counter, selection:{validation_metric,comparison:'lt|le|gt|ge',best_metric,result,result_unit:'fraction|percent'}} for an original module-level repetition loop",
    "forward_args": ["author_local_or_global_tensor_tree_or_scalar_selector"],
    "expected_runs": "positive_source_cited_integer",
    "expected_test_samples": ["one_source_cited_complete_test_count_per_native_run"],
    "expected_optimizer_steps": "source_cited_per_run_maximum_optimizer_steps",
    "completion": {"kind": "fixed_steps OR author_epoch_counter",
                   "epoch_counter": "local_identifier_only_for_author_epoch_counter",
                   "epoch_offset": "0_or_1_only_for_author_epoch_counter",
                   "updates_per_epoch": "positive_source_cited_integer_only_for_author_epoch_counter"},
    "aggregation": "mean",
}
CAPTURE_SCHEMA = {"single_run": SINGLE_CAPTURE_SCHEMA, "native_series": SERIES_CAPTURE_SCHEMA}
CAPTURE_DIR = "_autorepro_capture"
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")


def _selector_names(selector):
    if isinstance(selector, str):
        return [selector]
    if (not isinstance(selector, dict) or set(selector) != {'name', 'keys'}
            or not isinstance(selector['name'], str) or not isinstance(selector['keys'], list)
            or not 1 <= len(selector['keys']) <= 8
            or any(not (isinstance(key, str) and 1 <= len(key) <= 128
                        or type(key) is int and 0 <= key < 10000000) for key in selector['keys'])):
        raise ValueError('Author selectors must be identifiers or fixed dictionary/list key paths')
    return [selector['name']]


def validate_capture(capture, packet, workspace):
    """Only named author globals are selectable; no expressions or new programs."""
    if not isinstance(capture, dict) or capture.get("kind") not in {"torch_classification", "torch_classification_series"}:
        raise ValueError("Only full-test PyTorch classification capture is supported")
    required = {"kind", "model", "forward_args", "labels", "test_indices",
                "expected_optimizer_steps", "expected_test_samples", "reported_metric", "citations"}
    series = capture["kind"] == "torch_classification_series"
    if series:
        required |= {"device", "scope", "expected_runs", "completion", "aggregation"}
    if set(capture) - {'output_alignment'} != required:
        raise ValueError("Capture must use the complete fixed schema without executable extensions")
    alignment = capture.get('output_alignment', 'full_dataset')
    if not isinstance(alignment, str) or alignment not in {'full_dataset', 'selected_test'}:
        raise ValueError('Capture output alignment must be explicit full data or selected test rows')
    args = capture.get("forward_args")
    if not isinstance(args, list) or not 1 <= len(args) <= 16:
        raise ValueError("Capture needs a bounded ordered list of named forward tensors")
    names = [capture.get("model"), capture.get("labels"), *args]
    if capture.get("test_indices") is not None:
        names.append(capture["test_indices"])
    source = "\n".join(item["text"] for name, item in packet["repository"]["files"].items()
                       if name.endswith(".py"))
    for selector in names:
        for name in _selector_names(selector):
            if (not _IDENTIFIER.fullmatch(name)
                    or not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", source)):
                raise ValueError("Capture selectors must identify values present in author Python source")
        if isinstance(selector, dict):
            for key in selector['keys']:
                if isinstance(key, str) and not re.search(r"['\"]" + re.escape(key) + r"['\"]", source):
                    raise ValueError('Capture dictionary key is absent from the pinned author source')
    if alignment == 'selected_test' and (capture['test_indices'] is None or capture['test_indices'] not in args):
        raise ValueError('Selected-test output must use the same complete author test indices as a forward input')
    for key in ("expected_optimizer_steps",) if series else ("expected_optimizer_steps", "expected_test_samples"):
        if type(capture.get(key)) is not int or not 0 < capture[key] <= 10000000:
            raise ValueError("Capture requires positive, bounded prospective completion counts")
    if series:
        if capture["device"] not in {"cpu", "cuda"} or capture["aggregation"] != "mean":
            raise ValueError("Native series needs a fixed device and all-run mean aggregation")
        runs, samples = capture["expected_runs"], capture["expected_test_samples"]
        if (type(runs) is not int or not 1 <= runs <= 100 or not isinstance(samples, list)
                or len(samples) != runs or any(type(n) is not int or not 0 < n <= 10000000 for n in samples)):
            raise ValueError("Native series requires one complete prospective test count for every run")
        scope = capture["scope"]
        if (not isinstance(scope, dict) or not isinstance(scope.get('path'), str)
                or scope.get('path') not in packet["repository"]["files"]
                or not scope['path'].endswith('.py')):
            raise ValueError("Capture scope must name pinned author Python source")
        tree = ast.parse(packet["repository"]["files"][scope["path"]]["text"])
        if 'loop_counter' in scope:
            from src.native_loop import resolve_module_loop
            resolve_module_loop(packet['repository']['files'][scope['path']]['text'], scope)
        else:
            if (set(scope) != {'path','function'} or not isinstance(scope['function'], str)
                    or not _IDENTIFIER.fullmatch(scope['function'])):
                raise ValueError('Capture scope must name an original function or module loop')
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name == scope["function"]]
            if len(functions) != 1 or functions[0] not in tree.body or isinstance(functions[0], ast.AsyncFunctionDef):
                raise ValueError("Capture scope must resolve to one unique synchronous top-level author function")
        completion = capture["completion"]
        if not isinstance(completion, dict):
            raise ValueError("Series completion must have a fixed, non-executable rule")
        if completion.get("kind") == "fixed_steps":
            if set(completion) != {"kind"}:
                raise ValueError("Fixed-step completion cannot carry executable or extra fields")
        elif completion.get("kind") == "author_epoch_counter":
            if (set(completion) != {"kind", "epoch_counter", "epoch_offset", "updates_per_epoch"}
                    or not isinstance(completion["epoch_counter"], str)
                    or not _IDENTIFIER.fullmatch(completion["epoch_counter"])
                    or type(completion["epoch_offset"]) is not int or completion["epoch_offset"] not in {0, 1}
                    or type(completion["updates_per_epoch"]) is not int
                    or not 0 < completion["updates_per_epoch"] <= capture["expected_optimizer_steps"]):
                raise ValueError("Author epoch completion needs a local counter and fixed source-backed multiplier")
        else:
            raise ValueError("Unknown native series completion rule")
    reported = capture.get("reported_metric")
    if (not isinstance(reported, dict) or set(reported) != {"label", "unit"}
            or reported.get("unit") not in {"fraction", "percent"}
            or not isinstance(reported.get("label"), str)
            or not 1 <= len(reported["label"].strip()) <= 200
            or any(c in reported["label"] for c in "\r\n\x00")):
        raise ValueError("Capture requires the literal author accuracy label and unit")
    if not isinstance(capture.get("citations"), list) or not capture["citations"]:
        raise ValueError("Capture requires source citations")
    for name, item in packet["repository"]["files"].items():
        path = workspace_path(workspace, name, "capture source", must_exist=True, forbid_git=True)
        if digest(path) != item["sha256"]:
            raise ValueError("Capture source changed after planning")
    return deepcopy(capture)


def materialize_capture(workspace, plan, spec_hash, *, snapshot=None):
    root = Path(workspace).resolve()
    destination = workspace_path(root, CAPTURE_DIR, "capture directory", forbid_git=True)
    if destination.exists():
        raise ValueError("Project capture directory would overwrite author repository content")
    destination.mkdir()
    runtime = Path(__file__).parent / "experiments" / "discovered_capture.py"
    series = plan["capture"]["kind"] == "torch_classification_series"
    runtime_files = ["runtime.py", "config.json"]
    if series:
        (destination / "common.py").write_bytes(runtime.read_bytes())
        runtime = runtime.with_name("discovered_series_capture.py")
        runtime_files.append("common.py")
        (destination / 'native_loop.py').write_bytes((Path(__file__).parent / 'native_loop.py').read_bytes())
        runtime_files.append('native_loop.py')
    (destination / "runtime.py").write_bytes(runtime.read_bytes())
    train = next(step for step in plan["steps"] if step["kind"] == "train")
    config = {"version": 1, "spec_sha256": spec_hash, "capture": deepcopy(plan["capture"]),
              "train": deepcopy(train), "source_files": deepcopy((snapshot or {}).get("files", {}))}
    write_json(destination / "config.json", config)
    steps = []
    for original in plan["steps"]:
        step = deepcopy(original)
        step["env"] = {"PYTHONUNBUFFERED": "1"}
        if plan["capture"].get("device", "cpu") == "cpu":
            step["env"]["CUDA_VISIBLE_DEVICES"] = ""
        if step["kind"] == "train":
            step.update(argv=["python", "-u", f"{CAPTURE_DIR}/runtime.py", "capture"], cwd=".")
            step["artifacts"] = [f"{CAPTURE_DIR}/capture.json"] + [f"{CAPTURE_DIR}/{name}" for name in
                (series_artifact_names(plan["capture"]["expected_runs"]) if series else ["model.pt", "tensors.pt"])]
        steps.append(step)
    if any(step["id"] == "independent_evaluation" for step in steps):
        raise ValueError("Author step identifier collides with independent evaluation")
    steps.append({"id": "independent_evaluation", "kind": "eval", "required": True,
                  "depends_on": [step['id'] for step in steps],
                  "requires": [f"{CAPTURE_DIR}/capture.json"] + [f"{CAPTURE_DIR}/{name}" for name in
                      (series_artifact_names(plan['capture']['expected_runs']) if series else ['model.pt', 'tensors.pt'])],
                  "argv": ["python", "-u", f"{CAPTURE_DIR}/runtime.py", "evaluate"], "cwd": ".",
                  "timeout_s": 300,
                  "env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
                  "artifacts": [f"{CAPTURE_DIR}/independent_metrics.json"]})
    return {"files": {f"{CAPTURE_DIR}/{name}": digest(destination / name)
                       for name in runtime_files}, "steps": steps}


def series_artifact_names(runs):
    return [f"run_{index:03d}_{suffix}" for index in range(runs) for suffix in ("model.pt", "tensors.pt")]


def _reported_accuracy(plan, execution):
    identifier = next(step["id"] for step in plan["steps"] if step["kind"] == "train")
    step = next(item for item in execution["steps"] if item["id"] == identifier)
    log_root = Path(execution["run_dir"]).resolve(strict=True)
    stdout_path = Path(step["stdout_path"]).resolve(strict=True)
    if (stdout_path != log_root / f"{identifier}.stdout.log" or not stdout_path.is_file()
            or stdout_path.stat().st_size > 64 * 1024 * 1024):
        raise ValueError("Author metric log must be the bounded complete owned training stdout file")
    stdout = stdout_path.read_text(encoding="utf-8", errors="strict")
    reported = plan["capture"]["reported_metric"]
    pattern = re.escape(reported["label"]) + r"\s*[:=]?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    values = re.findall(pattern, stdout)
    if not values:
        raise ValueError("The completed author training log has no declared final accuracy")
    literal = values[-1]
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError("The author accuracy is non-finite")
    from decimal import Decimal
    # Half of the least printed decimal digit is the actual rounding allowance.
    rounding = 0.5 * 10.0 ** Decimal(literal).as_tuple().exponent
    divisor = 100 if reported["unit"] == "percent" else 1
    return {"value": value / divisor, "rounding_tolerance": rounding / divisor + 1e-8,
            "literal": literal, "unit": reported["unit"], "label": reported["label"]}


def verify_capture(plan, workspace, execution, spec_hash):
    """Accept only completed owned processes and their independently checked files."""
    expected_ids = [step["id"] for step in plan["steps"]] + ["independent_evaluation"]
    records = execution.get("steps", [])
    if (execution.get("success") is not True or execution.get("skipped")
            or [record.get("id") for record in records] != expected_ids
            or any(record.get("success") is not True or record.get("executed") is not True
                   or record.get("exit_code") != 0 or record.get("timed_out") or record.get("cancelled")
                   for record in records)):
        raise ValueError("Every required author and independent evaluation process must finish successfully")
    root = Path(workspace) / CAPTURE_DIR
    capture, metrics = read_json(root / "capture.json"), read_json(root / "independent_metrics.json")
    for record in (capture, metrics):
        if record.get("spec_sha256") != spec_hash or record.get("pass") is not True:
            raise ValueError("Independent runtime evidence does not match the frozen experiment")
    series = plan["capture"]["kind"] == "torch_classification_series"
    if series:
        verify_series_receipts(plan["capture"], root, capture, metrics)
    if ((not series and capture.get("optimizer_steps") != plan["capture"]["expected_optimizer_steps"])
            or capture.get("test_samples") != plan["capture"]["expected_test_samples"]
            or capture.get("parameters_changed") is not True
            or metrics.get("test_samples") != capture["test_samples"]
            or metrics.get("capture_sha256") != digest(root / "capture.json")
            or metrics.get("predictions_match") is not True
            or metrics.get("numpy_torch_crosscheck") is not True):
        raise ValueError("Training updates, complete test split or independent prediction identity failed")
    for name in series_artifact_names(plan["capture"]["expected_runs"]) if series else ("model.pt", "tensors.pt"):
        if capture.get("files_sha256", {}).get(name) != digest(root / name):
            raise ValueError("Training artifact changed after independent evaluation")
    accuracy, loss = metrics.get("accuracy"), metrics.get("cross_entropy")
    if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
            or isinstance(loss, bool) or not isinstance(loss, (int, float))
            or not math.isfinite(loss) or loss < 0):
        raise ValueError("Independent metrics are missing, invalid or non-finite")
    reported = _reported_accuracy(plan, execution)
    if abs(reported["value"] - accuracy) > reported["rounding_tolerance"]:
        raise ValueError("Author printed accuracy and fresh model evaluation disagree")
    comparisons = []
    for metric in plan["metrics"]:
        actual = accuracy if metric["name"] == "accuracy" else loss
        if metric["unit"] == "percent":
            actual *= 100
        tolerance = abs(metric["reference"]) * metric["tolerance_relative"]
        passed = (actual >= metric["reference"] - tolerance if metric["direction"] == "maximize"
                  else actual <= metric["reference"] + tolerance)
        comparisons.append({**metric, "actual": actual, "pass": passed,
                            "absolute_tolerance": tolerance})
    accepted = all(item["pass"] for item in comparisons)
    return {"status": "reproduced" if accepted else "reference_not_met", "is_reproduced": accepted,
            "result_level": "reproduced" if accepted else "experiment_completed",
            "reason": "完整作者训练与独立评估通过，所选论文实验指标达到预设容差" if accepted else
                      "完整作者训练与独立评估已完成，指标未达到论文参考值的预设容差",
            "scope": "selected_paper_experiment", "protocol_pass": True,
            "independent_metrics_pass": True, "comparisons": comparisons,
            "completion_checks": {"required_processes_exited": True,
                "full_training_and_test_capture": True, "all_native_runs_evaluated": True,
                "fresh_model_predictions_verified": True, "author_metric_reconciled": True,
                "paper_reference_met": accepted},
            "metrics_comparison": {"paper": {item["name"]: item["reference"] for item in comparisons},
                                   "actual": {item["name"]: item["actual"] for item in comparisons},
                                   "paper_units": {item["name"]: item["unit"] for item in comparisons},
                                   "actual_units": {item["name"]: item["unit"] for item in comparisons}},
            "metric_records": [{"name": item["name"], "value": item["actual"], "unit": item["unit"],
                                "spec_sha256": spec_hash, "source": "independent_evaluation",
                                "split": "test", "stage": "eval", "direction": item["direction"]}
                               for item in comparisons],
            "training_summary": {"optimizer_steps": capture["optimizer_steps"],
                                 "steps_completed": capture["optimizer_steps"],
                                 "runs_completed": len(capture['runs']) if series else 1,
                                 "runs_required": plan['capture'].get('expected_runs', 1),
                                 "test_samples": capture["test_samples"], "complete": True},
            "metrics": {item["name"]: item["actual"] for item in comparisons},
            "capture": capture, "independent_metrics": metrics, "author_reported_metric": reported}


def verify_series_receipts(contract, root, capture, metrics):
    """Reconcile every natural return, update count, full split and fresh metric."""
    runs, independent = capture.get("runs"), metrics.get("runs")
    expected = contract["expected_runs"]
    if (capture.get("kind") != "torch_classification_series" or capture.get("aggregation") != "mean"
            or metrics.get("aggregation") != "mean" or capture.get("training_device") != contract["device"]
            or metrics.get("device") != "cpu" or not isinstance(runs, list) or len(runs) != expected
            or not isinstance(independent, list) or len(independent) != expected):
        raise ValueError("Every declared native repetition must have training and fresh evaluation evidence")
    for index, (run, evaluated) in enumerate(zip(runs, independent)):
        steps = run.get("optimizer_steps")
        rule = contract["completion"]
        loop = 'loop_counter' in contract['scope']
        if (run.get("index") != index or evaluated.get("index") != index
                or (run.get('native_loop_completed') is not True if loop else run.get("natural_return") is not True)
                or run.get("parameters_changed") is not True
                or type(steps) is not int or not 0 < steps <= contract["expected_optimizer_steps"]
                or run.get("test_samples") != contract["expected_test_samples"][index]
                or evaluated.get("test_samples") != run["test_samples"]
                or evaluated.get("predictions_match") is not True
                or evaluated.get("numpy_torch_crosscheck") is not True):
            raise ValueError("A native repetition lacks complete training or independent full-test evidence")
        if loop:
            selection = run.get('validation_selection') or {}
            value, selected_steps = selection.get('accuracy'), selection.get('optimizer_steps')
            if (run.get('author_loop_counter') != index or selection.get('source') != contract['scope']['selection']
                    or type(selected_steps) is not int or not 0 < selected_steps <= steps
                    or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                    or not math.isclose(value, evaluated.get('accuracy', float('nan')), rel_tol=1e-10, abs_tol=1e-8)):
                raise ValueError('Validation-selected weights do not match the original selected result')
        if rule["kind"] == "fixed_steps":
            if steps != contract["expected_optimizer_steps"]:
                raise ValueError("Native repetition did not complete all fixed optimizer steps")
        elif (type(run.get("author_epoch_counter")) is not int
              or run["author_epoch_counter"] < 0
              or (run["author_epoch_counter"] + rule["epoch_offset"]) * rule["updates_per_epoch"] != steps):
            raise ValueError("Observed updates disagree with the author's actual completion counter")
        for field in ("accuracy", "cross_entropy"):
            value = evaluated.get(field)
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (field == "accuracy" and value > 1)):
                raise ValueError("Native repetition has an invalid independent metric")
        for name in series_artifact_names(expected)[2 * index:2 * index + 2]:
            if run.get("files_sha256", {}).get(name) != digest(root / name):
                raise ValueError("A native repetition artifact changed after evaluation")
    if (capture.get("optimizer_steps") != sum(run["optimizer_steps"] for run in runs)
            or capture.get("parameters_changed") is not True):
        raise ValueError("Aggregate training evidence does not reconcile with every repetition")
    for field in ("accuracy", "cross_entropy"):
        if not math.isclose(metrics.get(field, float("nan")), sum(row[field] for row in independent) / expected,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Reported aggregate is not the mean of every independent repetition")
