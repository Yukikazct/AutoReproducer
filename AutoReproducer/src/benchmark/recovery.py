"""Pre-registered, repository-separated recovery trials and paired summaries.

The runner and oracle are trusted application callbacks, never model-generated
commands. An aggregate checks receipt identities; it is not an oracle itself.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import time
from typing import Callable
import uuid

from src.holdout_registry import digest, product_hashes, repository_key
from src.benchmark.recovery_admission import (
    admission_receipt, prepare_admission, repository_identity, source_receipt, validate_admission,
    verify_admission, verify_author_source,
)

ARMS = ("A", "B", "C")
DEFAULT_BUDGET = {"max_model_requests": None, "max_tokens": None, "max_probes": 4,
                  "max_source_repair_rounds": 2, "max_repair_actions": 8}
DEFAULT_TOOLS = ("source_context", "runtime_metadata", "artifact_contract", "author_contract")
CATEGORIES = {"runtime_environment", "local_import", "api_compatibility", "result_evidence"}
ACCOUNTING_KEYS = {"model_requests": "max_model_requests", "tokens_charged": "max_tokens",
                   "probes": "max_probes", "source_repair_rounds": "max_source_repair_rounds",
                   "repair_actions": "max_repair_actions"}
BLOCKERS = {"external_condition", "needs_information", "unsupported", "execution_failed",
            "budget_exhausted", "cleanup_unconfirmed", "unclassified"}
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_TRIAL_SOURCE_BYTES = 512 * 1024 * 1024
RETRY_METRICS = ('invalid_retries', 'execution_retries', 'same_failure_retries')


def _hash(value, size=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _relative(name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and all(p not in {"", ".", ".."} for p in name.split("/"))


def _json_size(value):
    try:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Benchmark data must be finite JSON") from exc


def manifest_digest(manifest):
    return digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def validate_manifest(manifest, *, require_frozen=False):
    """Validate identities, preregistration and group-level split separation."""
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Recovery manifest version must be 1")
    if _json_size(manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("Recovery manifest exceeds the size limit")
    value = deepcopy(manifest)
    if type(value.get("repetitions")) is not int or value["repetitions"] != 3:
        raise ValueError("Recovery benchmark requires exactly three repetitions")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("Exactly the A, B and C arms are required")
    for arm in ARMS:
        record = arms[arm]
        if (not isinstance(record, dict) or not _hash(record.get("product_revision"), 40)
                or not _hash(record.get("product_source_sha256"))):
            raise ValueError("Each arm requires a pinned product revision and source digest")
    if arms["B"] != arms["C"]:
        raise ValueError("B and C must use the same frozen enhanced product")
    if not _text(value.get("model")):
        raise ValueError("A fixed model identifier is required")
    if value.get("adapter_driver_sha256") is not None and not _hash(value["adapter_driver_sha256"]):
        raise ValueError("Optional adapter driver identity must be a SHA-256 digest")
    tools = value.get("tools")
    if (not isinstance(tools, list) or not tools or any(not isinstance(tool, str) for tool in tools)
            or len(tools) != len(set(tools))
            or any(tool not in DEFAULT_TOOLS for tool in tools)):
        raise ValueError("Tools must be a nonempty unique subset of the fixed recovery tools")
    budget = value.get("budget")
    if not isinstance(budget, dict) or set(budget) != set(DEFAULT_BUDGET):
        raise ValueError("Budget must declare every shared limit")
    for key, number in budget.items():
        if key in {"max_model_requests", "max_tokens"} and number is None:
            continue
        if (not _integer(number, 1)
                or DEFAULT_BUDGET[key] is not None and number > DEFAULT_BUDGET[key]):
            raise ValueError("Explicit limits must be positive and preserve probe/repair caps")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one preregistered case is required")
    ids, repository_splits = set(), {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every recovery case must be an object")
        identifier = case.get("id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", identifier)
                or identifier in ids):
            raise ValueError("Duplicate or invalid recovery case id")
        ids.add(identifier)
        repo = repository_key(case.get("repository_url", ""))
        if not _hash(case.get("repository_revision"), 40):
            raise ValueError("Every case requires a pinned author revision")
        split = case.get("split")
        if split not in {"dev", "holdout"} or case.get("origin") not in {"natural", "injected"}:
            raise ValueError("Unknown case split or origin")
        if case.get("category") not in CATEGORIES:
            raise ValueError("Each case must preregister a known blocker category")
        if repo in repository_splits and repository_splits[repo] != split:
            raise ValueError("A repository cannot occur in both dev and holdout")
        repository_splits[repo] = split
        if split == "holdout" and case.get("unused_for_tuning") is not True:
            raise ValueError("Holdout cases must declare unused_for_tuning=true before trials")
        files = case.get("source_files")
        if (not isinstance(files, dict) or not files or len(files) > 20000
                or any(not _relative(name) or not _hash(sha) for name, sha in files.items())):
            raise ValueError("Each case requires safe relative source paths and SHA-256 digests")
        if len({name.casefold() for name in files}) != len(files):
            raise ValueError("Case source paths must not collide on Windows")
        if not _text(case.get("goal")):
            raise ValueError("Each case requires a preregistered goal")
        oracle = case.get("oracle")
        if (not isinstance(oracle, dict) or not _text(oracle.get("kind"))
                or not _text(oracle.get("reference")) or not _hash(oracle.get("sha256"))):
            raise ValueError("Each case requires a preregistered, hash-bound oracle reference")
        evidence = case.get("initial_evidence")
        if not isinstance(evidence, dict) or case.get("initial_evidence_sha256") != digest(evidence):
            raise ValueError("Initial evidence is missing or its digest changed")
    validate_admission(value)
    supplied = value.get("manifest_sha256")
    if supplied is not None and supplied != manifest_digest(value):
        raise ValueError("Frozen recovery manifest changed")
    if require_frozen and supplied is None:
        raise ValueError("Freeze the recovery manifest before execution or aggregation")
    return value


def freeze_manifest(manifest, *, admission_audit_path=None, registry_path=None):
    """Return detached preregistration, optionally reading a trusted exposure audit."""
    value = validate_manifest(manifest)
    now = datetime.now(timezone.utc).isoformat()
    if admission_audit_path is not None:
        if "manifest_sha256" in value or "holdout_admission" in value:
            raise ValueError("A frozen or admitted manifest cannot be re-admitted")
        value["frozen_at"] = now
        value["holdout_admission"] = prepare_admission(value, audit_path=admission_audit_path,
                                                      registry_path=registry_path)
    else:
        value.setdefault("frozen_at", now)
    value["manifest_sha256"] = manifest_digest(value)
    return validate_manifest(value, require_frozen=True)


def verify_product_identity(root, expected):
    """Re-read the actual checkout and product source, using the holdout freezer."""
    root = Path(root).resolve(strict=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("Product root must be a readable Git checkout") from exc
    actual = {"product_revision": revision, "product_source_sha256": digest(product_hashes(root))}
    if actual != expected:
        raise ValueError("Product checkout/source changed from its frozen arm identity")
    return actual


def _configuration(manifest, case, arm):
    return {"model": manifest["model"], "tools": [] if arm == "A" else deepcopy(manifest["tools"]),
            "budget": None if arm == "A" else deepcopy(manifest["budget"]),
            "budget_policy": "frozen_baseline" if arm == "A" else "shared_recovery",
            "initial_evidence_sha256": case["initial_evidence_sha256"],
            "product": deepcopy(manifest["arms"][arm]),
            "mode": {"A": "baseline", "B": "single", "C": "collaborative"}[arm],
            **({"adapter_driver_sha256": manifest["adapter_driver_sha256"]}
               if manifest.get("adapter_driver_sha256") else {})}


def _shared_configuration(manifest, case):
    return {key: value for key, value in _configuration(manifest, case, "B").items() if key != "mode"}


def _public_case(case):
    return deepcopy({key: case[key] for key in ("id", "repository_url", "repository_revision",
        "source_files", "goal", "initial_evidence", "initial_evidence_sha256", "origin", "category", "split")})


def _read_source(root, name, expected):
    root = Path(root).resolve(strict=True)
    path = root
    for part in PurePosixPath(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in frozen case source")
    if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError("Case source path escapes its root or is not a file")
    if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
        raise ValueError("A case source file exceeds the size limit")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Frozen author source changed: " + name)
    return content


def _prepare_workspace(case, source_root, destination):
    total = 0
    for name, expected in case["source_files"].items():
        content = _read_source(source_root, name, expected)
        total += len(content)
        if total > MAX_TRIAL_SOURCE_BYTES:
            raise ValueError("Frozen case source exceeds the per-trial size limit")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _usage(raw):
    value = raw.get("usage") if isinstance(raw, dict) else None
    value = value if isinstance(value, dict) else {}
    result = {}
    for key in (*ACCOUNTING_KEYS, "prompt_tokens", "completion_tokens", *RETRY_METRICS):
        number = value.get(key)
        result[key] = number if _integer(number) else None
    for key in ("cost", "duration_seconds"):
        number = value.get(key)
        result[key] = (number if type(number) in {int, float} and number >= 0 and math.isfinite(number)
                       else None)
    return result


def _outcome(raw, evaluated):
    if not isinstance(raw, dict) or not isinstance(evaluated, dict):
        raise ValueError("Runner and independent oracle must return objects")
    for field in ("oracle_passed", "protocol_preserved", "cleanup_confirmed"):
        if type(evaluated.get(field)) is not bool:
            raise ValueError("The independent evaluator must explicitly establish " + field)
    human = raw.get("human_interventions", 0)
    if not _integer(human):
        raise ValueError("human_interventions must be a nonnegative integer")
    interrupted = {}
    for field in ("cancelled", "timed_out"):
        interrupted[field] = raw.get(field, False)
        oracle_interrupted = evaluated.get(field, False)
        if type(interrupted[field]) is not bool or type(oracle_interrupted) is not bool:
            raise ValueError("Interruption flags must be explicit booleans")
        interrupted[field] = interrupted[field] or oracle_interrupted
    safe_completion = (evaluated["oracle_passed"] and evaluated["protocol_preserved"]
                       and evaluated["cleanup_confirmed"] and not any(interrupted.values()))
    completed = safe_completion and human == 0
    claimed = raw.get("claimed_success", False)
    if type(claimed) is not bool:
        raise ValueError("claimed_success must be boolean")
    blocker = evaluated.get("blocker") or raw.get("blocker") or "unclassified"
    return {"autonomous_completed": completed, "claimed_success": claimed,
            **interrupted,
            **{field: evaluated[field] for field in ("oracle_passed", "protocol_preserved", "cleanup_confirmed")},
            "needs_human": human > 0 or not completed, "human_interventions": human,
            "blocker": "" if completed else blocker if blocker in BLOCKERS else "unclassified",
            "false_success": claimed and not safe_completion}


def run_benchmark(manifest, source_roots, output_dir, *, trial_runner: Callable,
                  oracle_evaluator: Callable, product_roots=None, split="holdout",
                  admission_audit_path=None, registry_path=None):
    """Run trusted callbacks in fresh trial roots; never expose oracle to runner.

    trial_runner receives keyword arguments case, arm, repeat, workspace,
    environment, context and configuration. oracle_evaluator receives case,
    result, workspace and environment; it is responsible for authoritative
    execution/protocol/cleanup checks, not interpreting model success text.
    """
    frozen = validate_manifest(manifest, require_frozen=True)
    if split not in {"dev", "holdout"}:
        raise ValueError("Unknown benchmark split")
    selected = [case for case in frozen["cases"] if case["split"] == split]
    if not selected:
        raise ValueError("No cases registered for the selected split")
    verified_admission = verify_admission(frozen, audit_path=admission_audit_path,
                                          registry_path=registry_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Benchmark output must be a new directory")
    roots = {case["id"]: Path(source_roots[case["id"]]).resolve(strict=True) for case in selected}
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots.values()):
        raise ValueError("Benchmark output and author source roots must be separate")
    products = None
    if product_roots is not None:
        if not isinstance(product_roots, dict) or set(product_roots) != set(ARMS):
            raise ValueError("Product roots must identify all three arms")
        products = {arm: Path(product_roots[arm]).resolve(strict=True) for arm in ARMS}
        for arm in ARMS:
            if output.is_relative_to(products[arm]) or products[arm].is_relative_to(output):
                raise ValueError("Benchmark output must be outside the frozen product checkouts")
            verify_product_identity(products[arm], frozen["arms"][arm])
    for case in selected:
        for name, expected in case["source_files"].items():
            _read_source(roots[case["id"]], name, expected)
        verify_author_source(frozen, case, roots[case["id"]])
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    for case in selected:
        for repeat in range(1, 4):
            arms = list(ARMS)
            random.Random(digest([frozen["manifest_sha256"], case["id"], repeat])).shuffle(arms)
            for arm in arms:
                verified_admission = verify_admission(frozen, audit_path=admission_audit_path,
                                                      registry_path=registry_path)
                verified_source = verify_author_source(frozen, case, roots[case["id"]])
                verified_product = (verify_product_identity(products[arm], frozen["arms"][arm])
                                    if products is not None else None)
                trial = output / (case["id"] + "_" + arm + "_" + str(repeat))
                workspace, environment = trial / "workspace", trial / "environment"
                workspace.mkdir(parents=True)
                environment.mkdir()
                _prepare_workspace(case, roots[case["id"]], workspace)
                identity = {key: uuid.uuid4().hex for key in ("trial_id", "workspace_id", "environment_id", "context_id")}
                configuration = _configuration(frozen, case, arm)
                began = time.monotonic()
                raw = None
                try:
                    raw = trial_runner(case=_public_case(case), arm=arm, repeat=repeat,
                        workspace=workspace, environment=environment, context={},
                        configuration=deepcopy(configuration))
                    evaluated = oracle_evaluator(case=deepcopy(case), result=deepcopy(raw),
                                                workspace=workspace, environment=environment)
                    if _json_size(evaluated) > 1024 * 1024:
                        raise ValueError("Oracle evaluation receipt exceeds the size limit")
                    outcome = _outcome(raw, evaluated)
                except Exception as exc:
                    # Infrastructure exceptions are visible failures, never dropped trials.
                    raw = raw if isinstance(raw, dict) else {}
                    failed_raw = {**raw, "claimed_success": raw.get("claimed_success") is True,
                        "human_interventions": raw.get("human_interventions") if _integer(raw.get("human_interventions")) else 0,
                        "cancelled": raw.get("cancelled") is True, "timed_out": raw.get("timed_out") is True}
                    evaluated = {"oracle_passed": False, "protocol_preserved": False, "cleanup_confirmed": False}
                    evaluated["infrastructure_error"] = type(exc).__name__
                    outcome = {**_outcome(failed_raw, evaluated), "infrastructure_error": type(exc).__name__}
                for name, expected in case["source_files"].items():
                    _read_source(roots[case["id"]], name, expected)
                if products is not None:
                    verify_product_identity(products[arm], frozen["arms"][arm])
                verify_admission(frozen, audit_path=admission_audit_path, registry_path=registry_path)
                verify_author_source(frozen, case, roots[case["id"]])
                usage = _usage(raw)
                usage["duration_seconds"] = time.monotonic() - began
                record = {"version": 1, "case_id": case["id"], "arm": arm, "repeat": repeat,
                    "manifest_sha256": frozen["manifest_sha256"],
                    "initial_evidence_sha256": case["initial_evidence_sha256"],
                    "oracle_sha256": case["oracle"]["sha256"],
                    "configuration_sha256": digest(configuration),
                    "shared_configuration_sha256": digest(_shared_configuration(frozen, case)),
                    **identity, "outcome": outcome, "usage": usage,
                    "product_identity": {"verified": verified_product is not None,
                                         "actual": verified_product},
                    **({"holdout_admission": deepcopy(verified_admission)} if verified_admission else {}),
                    **({"author_source_identity": deepcopy(verified_source)} if verified_source else {}),
                    "execution_metadata": {key: raw.get(key) for key in
                        ("driver_sha256", "model_source", "execution_scope")} if isinstance(raw, dict) else {},
                    "evaluation": {"trial_id": identity["trial_id"], "oracle_sha256": case["oracle"]["sha256"],
                                   "result": evaluated, "result_sha256": digest(evaluated)}}
                records.append(record)
                # Incremental receipts survive interruption and keep completed failures.
                with (output / "trials.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if (not outcome["cleanup_confirmed"] or outcome["cancelled"] or outcome["timed_out"]
                        or evaluated.get("infrastructure_error")):
                    summary = aggregate_results(frozen, records)
                    summary["stopped_reason"] = ("cleanup_unconfirmed" if not outcome["cleanup_confirmed"]
                                                 else "cancelled" if outcome["cancelled"] else "timed_out"
                                                 if outcome["timed_out"] else "oracle_infrastructure_error")
                    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    return summary
    summary = aggregate_results(frozen, records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate_results(manifest, records):
    """Summarize every preregistered holdout case, retaining missing trials."""
    frozen = validate_manifest(manifest, require_frozen=True)
    if not isinstance(records, list):
        raise ValueError("Trial receipts must be a list")
    cases = {case["id"]: case for case in frozen["cases"]}
    indexed, unique_ids, budget_problems, unverified_products = {}, set(), [], []
    unverified_admissions = []
    unverified_sources = []
    expected_admission = admission_receipt(frozen)
    expected_sources = {case["id"]: source_receipt(frozen, case) for case in frozen["cases"]}
    for record in records:
        if not isinstance(record, dict) or record.get("version") != 1:
            raise ValueError("Unsupported trial receipt")
        case = cases.get(record.get("case_id"))
        arm, repeat = record.get("arm"), record.get("repeat")
        if case is None or arm not in ARMS or type(repeat) is not int or repeat not in {1, 2, 3}:
            raise ValueError("Unregistered case, arm or repetition")
        key = (case["id"], arm, repeat)
        if key in indexed:
            raise ValueError("Duplicate trial receipt")
        expected = {"manifest_sha256": frozen["manifest_sha256"],
            "initial_evidence_sha256": case["initial_evidence_sha256"],
            "oracle_sha256": case["oracle"]["sha256"],
            "configuration_sha256": digest(_configuration(frozen, case, arm)),
            "shared_configuration_sha256": digest(_shared_configuration(frozen, case))}
        if any(record.get(name) != value for name, value in expected.items()):
            raise ValueError("Trial receipt does not match frozen inputs/configuration/oracle")
        metadata = record.get("execution_metadata") or {}
        if frozen.get("adapter_driver_sha256") and metadata.get("driver_sha256") != frozen["adapter_driver_sha256"]:
            raise ValueError("Trial adapter driver differs from preregistration")
        if metadata.get("model_source") == "scripted_development" and (case["split"] != "dev" or case["origin"] != "injected"):
            raise ValueError("Scripted trials cannot become natural or holdout evidence")
        product = record.get("product_identity")
        if not isinstance(product, dict) or type(product.get("verified")) is not bool:
            raise ValueError("Trial receipt must state whether the actual product was checked")
        if product["verified"]:
            if product.get("actual") != frozen["arms"][arm]:
                raise ValueError("Actual product identity differs from the frozen arm")
        else:
            unverified_products.append({"case_id": case["id"], "arm": arm, "repeat": repeat})
        expected_source = expected_sources[case["id"]]
        for field, proof in (("holdout_admission", expected_admission),
                             ("author_source_identity", expected_source)):
            supplied_proof = record.get(field)
            if supplied_proof is not None and (not isinstance(supplied_proof, dict)
                    or type(supplied_proof.get("version")) is not int or supplied_proof["version"] != 1
                    or supplied_proof.get("verified") is not True or supplied_proof != proof):
                raise ValueError("Trial admission receipt differs from the frozen registry/exposure review/source identity")
        if expected_source is not None and record.get("author_source_identity") != expected_source:
            unverified_sources.append({"case_id": case["id"], "arm": arm, "repeat": repeat})
        if expected_admission is None or record.get("holdout_admission") != expected_admission:
            unverified_admissions.append({"case_id": case["id"], "arm": arm, "repeat": repeat})
        for name in ("trial_id", "workspace_id", "environment_id", "context_id"):
            identifier = record.get(name)
            if not _text(identifier) or identifier in unique_ids:
                raise ValueError("Each trial needs fresh workspace, environment and context identities")
            unique_ids.add(identifier)
        outcome = record.get("outcome")
        if not isinstance(outcome, dict):
            raise ValueError("Missing independent outcome")
        for field in ("autonomous_completed", "claimed_success", "oracle_passed", "protocol_preserved",
                      "cleanup_confirmed", "needs_human", "false_success", "cancelled", "timed_out"):
            if type(outcome.get(field)) is not bool:
                raise ValueError("Outcome booleans must be explicit")
        evaluation = record.get("evaluation")
        if (not isinstance(evaluation, dict) or evaluation.get("trial_id") != record["trial_id"]
                or evaluation.get("oracle_sha256") != case["oracle"]["sha256"]
                or not isinstance(evaluation.get("result"), dict)
                or evaluation.get("result_sha256") != digest(evaluation["result"])):
            raise ValueError("Independent evaluation receipt is not bound to this trial/oracle")
        recomputed = _outcome(outcome, evaluation["result"])
        if any(outcome[field] != recomputed[field] for field in recomputed):
            raise ValueError("Outcome contradicts authoritative oracle/protocol/cleanup fields")
        usage = record.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Invalid usage receipt")
        for metric in (*ACCOUNTING_KEYS, "prompt_tokens", "completion_tokens", *RETRY_METRICS):
            if usage.get(metric) is not None and not _integer(usage[metric]):
                raise ValueError("Known usage counts must be nonnegative integers")
        for subset, total in (("same_failure_retries", "execution_retries"),
                              ("invalid_retries", "model_requests")):
            if (usage.get(subset) is not None and usage.get(total) is not None
                    and usage[subset] > usage[total]):
                raise ValueError("Retry count exceeds its observed total: " + subset)
        for metric in ("cost", "duration_seconds"):
            number = usage.get(metric)
            if number is not None and (type(number) not in {int, float} or number < 0 or not math.isfinite(number)):
                raise ValueError("Known costs and durations must be finite and nonnegative")
        if arm in {"B", "C"}:
            for metric, limit in ACCOUNTING_KEYS.items():
                number = usage.get(metric)
                cap = frozen["budget"][limit]
                if cap is not None and (not _integer(number) or number > cap):
                    budget_problems.append({"case_id": case["id"], "arm": arm, "repeat": repeat,
                                            "field": metric, "reason": "unknown_or_over_budget"})
        indexed[key] = record
    main = [case for case in frozen["cases"] if case["split"] == "holdout" and case["origin"] == "natural"]
    missing = [{"case_id": case["id"], "arm": arm, "repeat": repeat} for case in main
               for arm in ARMS for repeat in range(1, 4) if (case["id"], arm, repeat) not in indexed]
    repositories = {repository_identity(frozen, case) for case in main}
    category_counts = {category: sum(case["category"] == category for case in main) for category in sorted(CATEGORIES)}
    case_rows, arm_rows = [], {}
    main_ids = {case["id"] for case in main}
    main_records = [record for record in records if record["case_id"] in main_ids]
    for case in main:
        row = {"case_id": case["id"], "repository": repository_key(case["repository_url"]), "arms": {}}
        for arm in ARMS:
            runs = [indexed[(case["id"], arm, repeat)] for repeat in range(1, 4)
                    if (case["id"], arm, repeat) in indexed]
            successes = sum(record["outcome"]["autonomous_completed"] for record in runs)
            human = sum(record["outcome"]["needs_human"] for record in runs)
            row["arms"][arm] = {"completed_repetitions": len(runs), "autonomous_repetitions": successes,
                "majority_autonomous": len(runs) == 3 and successes >= 2,
                "majority_needs_human": len(runs) != 3 or human >= 2}
        case_rows.append(row)
    for arm in ARMS:
        runs = [record for record in main_records if record["arm"] == arm]
        arm_rows[arm] = {"autonomous_cases": sum(row["arms"][arm]["majority_autonomous"] for row in case_rows),
            "needs_human_cases": sum(row["arms"][arm]["majority_needs_human"] for row in case_rows),
            "completed_trials": len(runs), "false_successes": sum(record["outcome"]["false_success"] for record in runs),
            "protocol_violations": sum(not record["outcome"]["protocol_preserved"] for record in runs),
            "human_interventions": sum(record["outcome"]["human_interventions"] for record in runs),
            "usage_totals": {metric: (sum(record.get("usage", {}).get(metric) or 0 for record in runs)
                if runs and all(record.get("usage", {}).get(metric) is not None for record in runs) else None)
                for metric in (*ACCOUNTING_KEYS, "prompt_tokens", "completion_tokens", *RETRY_METRICS, "cost", "duration_seconds")}}
    a, b, c = (arm_rows[arm] for arm in ARMS)
    net_gain = c["autonomous_cases"] - a["autonomous_cases"]
    collaboration_gain = c["autonomous_cases"] - b["autonomous_cases"]
    human_reduction = ((a["needs_human_cases"] - c["needs_human_cases"]) / a["needs_human_cases"]
                       if a["needs_human_cases"] else None)
    problems = [item for item in budget_problems if item["case_id"] in main_ids]
    unchecked = [item for item in unverified_products if item["case_id"] in main_ids]
    interrupted = [{"case_id": record["case_id"], "arm": record["arm"], "repeat": record["repeat"]}
                   for record in main_records if not record["outcome"]["cleanup_confirmed"]
                   or record["outcome"]["cancelled"] or record["outcome"]["timed_out"]
                   or record["evaluation"]["result"].get("infrastructure_error")]
    reasons = []
    unaudited = [item for item in unverified_admissions if item["case_id"] in main_ids]
    if expected_admission is None:
        reasons.append("Formal holdouts require a frozen shared registry and bounded exposure review")
    elif unaudited:
        reasons.append("Registry/exposure admission was not verified for some formal trials")
    if unverified_sources:
        reasons.append("Original author Git checkout identity was not verified for some formal trials")
    if len(main) < 12:
        reasons.append("At least 12 preregistered natural holdout cases are required")
    if len(repositories) < 6:
        reasons.append("At least six holdout repositories are required")
    if any(count == 0 for count in category_counts.values()):
        reasons.append("Natural holdout cases must cover all four preregistered blocker categories")
    if missing:
        reasons.append("Some preregistered trials are missing")
    if problems:
        reasons.append("B/C budget accounting is missing or exceeds shared limits")
    if unchecked:
        reasons.append("Actual frozen product checkouts were not verified for some trials")
    if interrupted:
        reasons.append("Cancelled, timed-out or unconfirmed-cleanup trials prevent formal completion")
    complete = not reasons
    false_successes = sum(value["false_successes"] for value in arm_rows.values())
    protocol_violations = sum(value["protocol_violations"] for value in arm_rows.values())
    criteria = {"net_autonomous_gain_at_least_two": net_gain >= 2,
                "human_need_reduced_at_least_30_percent": human_reduction is not None and human_reduction >= 0.3,
                "zero_false_successes": false_successes == 0,
                "zero_protocol_violations": protocol_violations == 0}
    return {"version": 1, "manifest_sha256": frozen["manifest_sha256"], "evaluation_complete": complete,
        "incomplete_reasons": reasons, "engineering_criteria_met": complete and all(criteria.values()),
        "criteria": criteria, "natural_holdout_cases": len(main), "holdout_repositories": len(repositories),
        "category_counts": category_counts,
        "missing_trials": missing, "budget_problems": problems, "unverified_product_trials": unchecked,
        "interrupted_trials": interrupted,
        "holdout_admission": {"audited": expected_admission is not None,
            "unverified_trials": unaudited,
            "registry_sha256": expected_admission["registry_sha256"] if expected_admission else None,
            "audit_sha256": expected_admission["audit_sha256"] if expected_admission else None},
        "unverified_author_source_trials": unverified_sources,
        "arms": arm_rows, "cases": case_rows,
        "comparisons": {"C_minus_A_autonomous_cases": net_gain, "C_minus_B_autonomous_cases": collaboration_gain,
                        "C_vs_A_human_need_relative_reduction": human_reduction},
        "collaboration_gain_observed_in_sample": complete and collaboration_gain > 0,
        "recommended_mode": "collaborative_candidate" if complete and all(criteria.values()) and collaboration_gain > 0
                             else "single_candidate" if complete else "not_evaluated",
        "other_cases": [{"case_id": case["id"], "split": case["split"], "origin": case["origin"]}
                        for case in frozen["cases"] if case["id"] not in main_ids],
        "limitations": ["Case-majority summaries; repetitions and same-repository cases are not independent samples.",
            "Results concern selected blocking tasks, not overall paper reproduction success.",
            "Receipt hashes bind identity; independent oracle execution is the evaluator's responsibility.",
            "Exposure eligibility is limited to the trusted operator's declared review scope and recorded aliases.",
            "Three new complete-paper experiments must be accepted separately."]}
