"""Project-owned capture around unchanged, evidence-bound author entrypoints."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from src.method_adapters import digest, read_json, write_json
from src.safety.paths import workspace_path


CAPTURE_SCHEMA = {
    "kind": "torch_classification", "model": "simple_global_identifier",
    "forward_args": ["simple_global_tensor_identifier"], "labels": "simple_global_identifier",
    "test_indices": "simple_global_identifier_or_null_for_complete_labels",
    "expected_optimizer_steps": "positive_source_cited_integer",
    "expected_test_samples": "positive_source_cited_integer",
    "reported_metric": {"label": "literal_author_printed_accuracy_label", "unit": "fraction|percent"},
    "citations": ["repository_citation_ids"],
}
CAPTURE_DIR = "_autorepro_capture"
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")


def validate_capture(capture, packet, workspace):
    """Only named author globals are selectable; no expressions or new programs."""
    if not isinstance(capture, dict) or capture.get("kind") != "torch_classification":
        raise ValueError("Only full-test PyTorch classification capture is supported")
    required = {"kind", "model", "forward_args", "labels", "test_indices",
                "expected_optimizer_steps", "expected_test_samples", "reported_metric", "citations"}
    if set(capture) != required:
        raise ValueError("Capture must use the complete fixed schema without executable extensions")
    args = capture.get("forward_args")
    if not isinstance(args, list) or not 1 <= len(args) <= 16:
        raise ValueError("Capture needs a bounded ordered list of named forward tensors")
    names = [capture.get("model"), capture.get("labels"), *args]
    if capture.get("test_indices") is not None:
        names.append(capture["test_indices"])
    source = "\n".join(item["text"] for name, item in packet["repository"]["files"].items()
                       if name.endswith(".py"))
    for name in names:
        if (not isinstance(name, str) or not _IDENTIFIER.fullmatch(name)
                or not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", source)):
            raise ValueError("Capture selectors must be simple identifiers present in author Python source")
    for key in ("expected_optimizer_steps", "expected_test_samples"):
        if type(capture.get(key)) is not int or not 0 < capture[key] <= 10000000:
            raise ValueError("Capture requires positive, bounded prospective completion counts")
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
    (destination / "runtime.py").write_bytes(runtime.read_bytes())
    train = next(step for step in plan["steps"] if step["kind"] == "train")
    config = {"version": 1, "spec_sha256": spec_hash, "capture": deepcopy(plan["capture"]),
              "train": deepcopy(train), "source_files": deepcopy((snapshot or {}).get("files", {}))}
    write_json(destination / "config.json", config)
    steps = []
    for original in plan["steps"]:
        step = deepcopy(original)
        step["env"] = {"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
        if step["kind"] == "train":
            step.update(argv=["python", "-u", f"{CAPTURE_DIR}/runtime.py", "capture"], cwd=".")
            step["artifacts"] = [f"{CAPTURE_DIR}/capture.json", f"{CAPTURE_DIR}/model.pt",
                                 f"{CAPTURE_DIR}/tensors.pt"]
        steps.append(step)
    if any(step["id"] == "independent_evaluation" for step in steps):
        raise ValueError("Author step identifier collides with independent evaluation")
    steps.append({"id": "independent_evaluation", "kind": "eval", "required": True,
                  "argv": ["python", "-u", f"{CAPTURE_DIR}/runtime.py", "evaluate"], "cwd": ".",
                  "timeout_s": 300, "depends_on": [steps[-1]["id"]],
                  "env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
                  "requires": [f"{CAPTURE_DIR}/capture.json", f"{CAPTURE_DIR}/model.pt",
                               f"{CAPTURE_DIR}/tensors.pt"],
                  "artifacts": [f"{CAPTURE_DIR}/independent_metrics.json"]})
    return {"files": {f"{CAPTURE_DIR}/{name}": digest(destination / name)
                       for name in ("runtime.py", "config.json")}, "steps": steps}


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
    if (capture.get("optimizer_steps") != plan["capture"]["expected_optimizer_steps"]
            or capture.get("test_samples") != plan["capture"]["expected_test_samples"]
            or capture.get("parameters_changed") is not True
            or metrics.get("test_samples") != capture["test_samples"]
            or metrics.get("capture_sha256") != digest(root / "capture.json")
            or metrics.get("predictions_match") is not True
            or metrics.get("numpy_torch_crosscheck") is not True):
        raise ValueError("Training updates, complete test split or independent prediction identity failed")
    for name in ("model.pt", "tensors.pt"):
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
                                 "test_samples": capture["test_samples"], "complete": True},
            "metrics": {item["name"]: item["actual"] for item in comparisons},
            "capture": capture, "independent_metrics": metrics, "author_reported_metric": reported}
