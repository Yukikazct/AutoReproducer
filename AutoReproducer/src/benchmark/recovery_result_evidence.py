"""Read-only result investigations for frozen-product recovery trials.

This trusted driver is also loaded with runpy beside an old product checkout.
Keep imports here in the standard library; production imports inside functions
must come from the product selected by the worker, never the driver's checkout.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


FIELDS = ("metric", "value", "unit", "method", "dataset", "split", "aggregation", "runs")
IDENTITY_FIELDS = tuple(name for name in FIELDS if name != "value")
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
STATUSES = {"invalid_evidence", "insufficient_reference", "insufficient_evidence", "reference_gap", "reference_met"}


def _unconfirmed(execution):
    pending, seen, diagnostics = [(execution, False)], set(), []
    while pending:
        record, receipt = pending.pop()
        if isinstance(record, (list, dict)):
            identity = (id(record), receipt)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(record, list):
            pending.extend((item, receipt) for item in record)
            continue
        if not isinstance(record, dict):
            continue
        if (record.get("cleanup_confirmed") is False or record.get("process_cleanup_confirmed") is False
                or any(record.get(key) for key in ("cleanup_failed", "cleanup_unconfirmed", "cleanup_error", "cleanup_errors"))
                or record.get("status") in {"cleanup_failed", "cleanup_unconfirmed"}
                or receipt and (record.get("confirmed") is False or record.get("status") in {"failed", "unconfirmed"})):
            return True
        diagnostics.extend(str(record.get(key) or "") for key in ("reason", "stderr", "stdout"))
        pending.extend((record.get(key), True) for key in ("cleanup", "process_cleanup", "cleanup_receipt"))
        pending.extend((record.get(key), False) for key in
                       ("final", "effective_env_config", "environment", "steps", "attempts", "stages"))
    return bool(re.search(r"process[- ](?:tree|group) cleanup (?:failed|was denied)|\bcleanup_unconfirmed\b",
                          "\n".join(diagnostics), re.I))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(workspace, name, *, exists=True):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or PurePosixPath(name).is_absolute()
            or any(part.casefold() in {".git", "private", "hidden", "hidden_labels"} for part in name.split("/"))):
        raise ValueError("Result artifact must have a safe repository-relative path")
    root = Path(workspace).resolve(strict=True)
    target = root.joinpath(*PurePosixPath(name).parts)
    for item in [target, *target.parents]:
        if item == root:
            break
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError("Result artifacts cannot use links or junctions")
    target = target.resolve(strict=exists)
    if not target.is_relative_to(root):
        raise ValueError("Result artifact escaped its original workspace")
    return target


def _protocol_reference(plan):
    """Read existing explicit protocol fields; never synthesize missing values."""
    protocol = plan.get("experiment_protocol")
    if not isinstance(protocol, dict):
        return {}
    method, dataset, repeats = (protocol.get(name) or {} for name in ("method", "dataset", "repetitions"))
    if any(not isinstance(node, dict) for node in (method, dataset, repeats)):
        raise ValueError("Invalid frozen experiment protocol fields")
    metrics = protocol.get("metrics") or []
    if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
        raise ValueError("Result adapter needs one explicitly selected protocol metric")
    metric = metrics[0]
    return {"metric": metric.get("name"), "value": metric.get("reference"), "unit": metric.get("unit"),
            "method": method.get("name"), "dataset": dataset.get("name"), "split": dataset.get("split"),
            "aggregation": repeats.get("aggregation"), "runs": repeats.get("count")}


def validate_result_evidence(case, workspace):
    """Validate the result-specific envelope after common frozen-plan gates."""
    evidence = deepcopy(case.get("initial_evidence"))
    if not isinstance(evidence, dict) or case.get("category") != "result_evidence":
        raise ValueError("Expected a preregistered result_evidence case")
    if evidence.get("editable", []) != []:
        raise ValueError("Result investigations cannot authorize source modifications")
    contract = evidence.get("result_contract")
    required = {"version", "artifact", "format", "reference", "relative_tolerance", "absolute_tolerance"}
    if (not isinstance(contract, dict) or set(contract) != required or contract["version"] != 1
            or contract["format"] != "metric_summary_json"):
        raise ValueError("Result contract must explicitly declare its artifact, reference and both tolerances")
    artifact = contract["artifact"]
    _path(workspace, artifact, exists=False)
    if not artifact.endswith(".json") or artifact in case.get("source_files", {}):
        raise ValueError("Result evidence must be a newly produced JSON artifact, not a frozen source")
    reference = contract["reference"]
    if not isinstance(reference, dict) or set(reference) - set(FIELDS):
        raise ValueError("Unknown result reference fields")
    for name, value in reference.items():
        if value is None:
            continue
        if name == "value":
            valid = _number(value)
        elif name == "runs":
            valid = type(value) is int and value > 0
        else:
            valid = isinstance(value, str) and bool(value.strip()) and len(value) <= 1000
        if not valid:
            raise ValueError("Invalid explicit reference field: " + name)
    for name in ("relative_tolerance", "absolute_tolerance"):
        value = contract[name]
        if value is not None and (not _number(value) or value < 0):
            raise ValueError("Result tolerance must be a finite nonnegative number or explicit null")
    for name, value in _protocol_reference(evidence.get("plan") or {}).items():
        if value is not None and reference.get(name) != value:
            raise ValueError("Result reference conflicts with the frozen experiment protocol: " + name)
    validation = evidence.get("initial_validation", {})
    if not isinstance(validation, dict) or len(json.dumps(validation, allow_nan=False).encode("utf-8")) > 256 * 1024:
        raise ValueError("Initial result validation must be a bounded original record")
    return evidence


def _snapshot(contract, workspace):
    """Identity only: no new artifact semantics are revealed to any arm."""
    try:
        path = _path(workspace, contract["artifact"])
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            return {}
        return {contract["artifact"]: hashlib.sha256(path.read_bytes()).hexdigest()}
    except (OSError, ValueError):
        return {}


def _assess_values(contract, actual, snapshot, issues=()):
    """Check only values actually supplied by the selected product's evidence."""
    issues, unknown = list(issues), []
    actual = {name: actual.get(name) for name in FIELDS}
    reference = contract["reference"]
    for field in FIELDS:
        left, right = reference.get(field), actual.get(field)
        if left is None:
            unknown.append("reference." + field)
        if right is None:
            unknown.append("actual." + field)
            continue
        if field == "value":
            if not _number(right):
                issues.append("invalid_actual_value")
            continue
        valid = type(right) is int and right > 0 if field == "runs" else isinstance(right, str) and bool(right.strip())
        if not valid:
            issues.append("invalid_actual_" + field)
        elif left is not None and right != left:
            issues.append(field + "_mismatch")
    for field in ("relative_tolerance", "absolute_tolerance"):
        if contract[field] is None:
            unknown.append(field)
    if issues:
        status = "invalid_evidence"
    elif any(name.startswith("reference.") or name.endswith("tolerance") for name in unknown):
        status = "insufficient_reference"
    elif unknown:
        status = "insufficient_evidence"
    else:
        with localcontext() as context:
            context.prec = 1000
            difference = abs(Decimal(str(actual["value"])) - Decimal(str(reference["value"])))
            threshold = max(Decimal(str(contract["absolute_tolerance"])),
                            abs(Decimal(str(reference["value"]))) * Decimal(str(contract["relative_tolerance"])))
            status = "reference_met" if difference <= threshold else "reference_gap"
    return {"status": status, "issue_codes": sorted(set(issues)), "unknown_fields": sorted(set(unknown)),
            "actual": actual, "reference": deepcopy(reference), "artifacts": deepcopy(snapshot),
            "numerical_equivalence": "unknown", "unit_conversion_applied": False}


def _observe(contract, workspace):
    """Private Oracle observation; never supplies new answers to the baseline."""
    name = contract["artifact"]
    issues = []
    snapshot, actual = {}, {}
    try:
        path = _path(workspace, name)
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Result artifact is not a bounded file")
        content = path.read_bytes()
        snapshot[name] = hashlib.sha256(content).hexdigest()
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate result field")
                result[key] = value
            return result
        def numeric(text):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("Nonfinite result")
            return value
        def integer(text):
            if len(text) > 128:
                raise ValueError("Result integer exceeds observation budget")
            return int(text)
        actual = json.loads(content, object_pairs_hook=unique, parse_float=numeric, parse_int=integer,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite result")))
        if not isinstance(actual, dict) or len(actual) > 100:
            raise ValueError("Expected bounded flat metric summary")
        actual = {name: actual.get(name) for name in FIELDS}
    except FileNotFoundError:
        issues.append("artifact_missing")
        actual = {}
    except (OSError, ValueError, UnicodeError, RecursionError):
        issues.append("artifact_invalid")
        actual = {}
    return _assess_values(contract, actual, snapshot, issues)


def _initial_validation(evidence):
    # Missing comparison is the old product's normal observational trigger.
    # Do not populate this with the new Oracle's artifact interpretation.
    return deepcopy(evidence.get("initial_validation", {"status": "completed",
        "is_reproduced": None, "paper_metrics_verified": False,
        "author_claim_comparison": {"comparisons": []}}))


def _unresolved(contract, artifacts):
    return {"status": "unresolved", "issue_codes": [], "unknown_fields": [], "actual": {},
            "reference": deepcopy(contract["reference"]), "artifacts": deepcopy(artifacts),
            "numerical_equivalence": "unknown", "unit_conversion_applied": False}


def _original_validation_claim(validation, contract, artifacts):
    """Retain only an existing validator's explicit verified comparison verdict.

    A diagnosis proposing checks or hypotheses is never a resolved result.
    The driver does not turn missing old evidence into a baseline capability.
    """
    comparison = validation.get("author_claim_comparison") or {}
    rows = comparison.get("comparisons") or []
    if comparison.get("status") != "compared" or len(rows) != 1 or comparison.get("validation_errors"):
        return None
    row = rows[0]
    if row.get("status") != "comparable" or type(row.get("within_tolerance")) is not bool:
        return None
    actual, reference = row.get("actual") or {}, row.get("reference") or {}
    if row.get("metric") != contract["reference"].get("metric"):
        return None
    for field in ("unit", "method", "dataset", "split", "aggregation", "runs"):
        if (actual.get(field) != reference.get(field) or reference.get(field) != contract["reference"].get(field)
                or field in {"unit", "split", "aggregation", "runs"} and any(
                    ((node.get("field_evidence") or {}).get(field) or {}).get("status") != "verified"
                    for node in (actual, reference))):
            return None
    if reference.get("value") != contract["reference"].get("value"):
        return None
    if (comparison.get("relative_tolerance") != contract["relative_tolerance"]
            or contract["absolute_tolerance"] != 0):
        return None  # Old comparisons have no independently declared absolute tolerance.
    claim = _assess_values(contract, {**actual, "metric": row["metric"]}, artifacts)
    # This contract measures closeness to a reference. The old validator's
    # meets_reference is directional and can be true far above a higher-is-
    # better reference, or unknown when within_tolerance is already verified.
    desired = "reference_met" if row["within_tolerance"] else "reference_gap"
    return claim if claim["status"] == desired else None


def run_result_evidence(*, evidence, workspace, environment, execution, llm, session,
                        run_dir, data, tools=None, verify_inputs=None, cancel_check=None):
    """Invoke the selected product's real result diagnosis and enhanced tools.

    An existing session is required for B/C; this function never creates a new
    ledger, patch, repair action, command, environment or author execution.
    """
    from src.repository_repairer import repair_is_stopped
    from src.result_diagnosis import diagnose_experiment_result
    contract = deepcopy(evidence["result_contract"])
    original_contract_sha = _digest(contract)
    original_plan_sha = _digest(evidence["plan"])
    original_execution_sha = _digest(execution)
    validation = _initial_validation(evidence)
    original_validation_sha = _digest(validation)
    result = {"version": 1, "initial_success_observed": False, "completed": False,
              "protocol_preserved": True, "cleanup_confirmed": True,
              "cancelled": False, "timed_out": False,
              "contract_sha256": original_contract_sha, "plan_sha256": original_plan_sha,
              "initial_validation_sha256": original_validation_sha,
              "execution_sha256": original_execution_sha, "assessment": {},
              "diagnosis": {}, "investigation_status": "not_requested",
              "observed_artifact_evidence_ids": [], "decision_evidence_ids": [], "numerical_equivalence": "unknown"}

    def persist():
        cleaned = session.safe(result) if session else deepcopy(result)
        data["result_evidence"] = cleaned
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "result_evidence.json").write_text(json.dumps(cleaned, ensure_ascii=False, allow_nan=False), "utf-8")

    def check_running():
        if cancel_check and cancel_check():
            result["cancelled"] = True
            if session:
                session.stop("result_investigation_cancelled")
            raise KeyboardInterrupt()
        if result["timed_out"] or not result["cleanup_confirmed"]:
            stopped = RuntimeError("Result investigation stopped before further model or tool activity")
            stopped.timed_out = result["timed_out"]
            stopped.process_cleanup_confirmed = result["cleanup_confirmed"]
            raise stopped
        if session:
            session.ensure_running()

    class GuardedLLM:
        # Older diagnosis preserves deterministic facts by catching ordinary
        # model errors. Preserve stop flags outside that catch boundary too.
        def __init__(self, target):
            self.target = target

        def __getattr__(self, name):
            return getattr(self.target, name)

        def chat(self, prompt, **kwargs):
            check_running()
            try:
                return self.target.chat(prompt, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                result["cancelled"] = True
                raise
            except Exception as exc:
                timed_out = (isinstance(exc, TimeoutError) or type(exc).__name__ == "RuntimePreparationTimeout"
                             or getattr(exc, "timed_out", False) is True)
                unclean = getattr(exc, "process_cleanup_confirmed", True) is False
                result["timed_out"] |= timed_out
                result["cleanup_confirmed"] &= not unclean
                if session and (timed_out or unclean):
                    session.stop("result_model_timeout" if timed_out else "result_model_cleanup_unconfirmed",
                                 cleanup_confirmed=False if unclean else None)
                    session.state["timed_out"] = session.state.get("timed_out", False) or timed_out
                raise

    if repair_is_stopped(execution) or _unconfirmed(execution):
        result.update(cleanup_confirmed=not _unconfirmed(execution),
                      cancelled=execution.get("cancelled") is True, timed_out=execution.get("timed_out") is True,
                      investigation_status="stopped")
        if session:
            session.record_execution(execution)
            result["cleanup_confirmed"] &= session.state.get("cleanup_confirmed", True) is True
            if not result["cleanup_confirmed"]:
                session.stop("result_execution_cleanup_unconfirmed", cleanup_confirmed=False)
        persist()
        return result
    steps = execution.get("steps") or []
    final = execution.get("final") or {}
    if (execution.get("success") is not True or execution.get("executed") is not True
            or final.get("exit_code") != 0 or not steps
            or any(step.get("exit_code") != 0 or step.get("executed") is not True for step in steps)):
        result["investigation_status"] = "author_execution_incomplete"
        persist()
        return result
    result["initial_success_observed"] = True
    artifacts = _snapshot(contract, workspace)
    result["assessment"] = _unresolved(contract, artifacts)

    def unchanged():
        if verify_inputs:
            verify_inputs()
        if (_digest(evidence["result_contract"]) != original_contract_sha
                or _digest(evidence["plan"]) != original_plan_sha
                or _digest(execution) != original_execution_sha
                or _digest(_initial_validation(evidence)) != original_validation_sha
                or _snapshot(contract, workspace) != artifacts):
            result["protocol_preserved"] = False
            raise ValueError("Result investigation changed the frozen protocol, execution or actual artifacts")

    original_session_llm = session.llm if session else None
    guarded_llm = GuardedLLM(llm)
    try:
        unchanged()
        check_running()
        if session:
            session.llm = guarded_llm if original_session_llm is llm else GuardedLLM(original_session_llm)
        diagnostic_llm = session.wrap_llm(guarded_llm) if session else guarded_llm
        diagnosis = diagnose_experiment_result(diagnostic_llm, evidence["plan"], evidence["packet"],
            workspace, execution, validation, run_dir, data)
        result["diagnosis"] = {"status": (diagnosis or {}).get("status", "not_needed"),
            "facts": deepcopy((diagnosis or {}).get("facts", [])),
            "model_status": ((diagnosis or {}).get("model_analysis") or {}).get("status", "not_requested")}
        unchanged()
        check_running()
        task = None
        if session:
            from src.investigation_tools import InvestigationTools
            packet = deepcopy(evidence["packet"])
            packet["investigation_artifacts"] = {name: {"sha256": sha, "public": True, "metric_summary": True}
                                                  for name, sha in artifacts.items()}
            toolset = tools or InvestigationTools(workspace, packet, environment, run_dir)
            execute = toolset.execute
            def checked_execute(name, arguments):
                check_running()
                unchanged()
                observed = execute(name, arguments)
                unchanged()
                return observed
            toolset.execute = checked_execute
            try:
                task = session.investigate(trigger="result_discrepancy", plan=evidence["plan"], packet=packet,
                    workspace=workspace, environment=environment, execution=execution,
                    validation=validation, tools=toolset)
            finally:
                toolset.execute = execute
            result["investigation_status"] = (task or {}).get("status", session.state.get("status"))
            result["observed_artifact_evidence_ids"] = sorted({item["id"] for probe in (task or {}).get("probes", [])
                for item in probe.get("evidence", []) if item.get("source") in artifacts and item.get("sha256") == artifacts[item["source"]]})
            result["decision_evidence_ids"] = sorted({ref for decision in (task or {}).get("decisions", [])
                if decision.get("action") == "finish" and decision.get("status") == "accepted_for_existing_validation"
                for ref in decision.get("evidence_ids", [])})
            selected = set(result["decision_evidence_ids"])
            if task and task.get("status") == "observed":
                for probe in task.get("probes", []):
                    if probe.get("status") != "completed" or probe.get("tool") != "artifact_contract":
                        continue
                    for item in probe.get("evidence", []):
                        values = (item.get("value") or {}).get("metric_summary")
                        if (item.get("id") in selected and item.get("source") in artifacts
                                and item.get("sha256") == artifacts[item["source"]]
                                and isinstance(values, dict) and set(values) <= set(FIELDS)
                                and isinstance(values.get("metric"), str) and _number(values.get("value"))):
                            result["assessment"] = _assess_values(contract, values, artifacts)
            check_running()
        else:
            # A's real old diagnosis remains an observation or hypothesis unless
            # its own original validator already established a full verdict.
            claim = _original_validation_claim(validation, contract, artifacts)
            if claim:
                result["assessment"] = claim
        unchanged()
        result["completed"] = (result["assessment"]["status"] in STATUSES
            and (session is None or bool(task and task.get("status") == "observed")))
    except (KeyboardInterrupt, SystemExit):
        result.update(cancelled=True, investigation_status="stopped")
        if session:
            session.stop("result_investigation_cancelled")
    except Exception as exc:
        result.update(investigation_status="stopped" if session and session.state.get("status") == "stopped" else "needs_human",
                      error_type=type(exc).__name__)
        if getattr(exc, "process_cleanup_confirmed", True) is False:
            result["cleanup_confirmed"] = False
        if session:
            result["cleanup_confirmed"] &= session.state.get("cleanup_confirmed", True) is True
            result["timed_out"] |= session.state.get("timed_out") is True
    finally:
        try:
            unchanged()
        except Exception:
            result.update(protocol_preserved=False, completed=False)
        if session:
            result["cleanup_confirmed"] &= session.state.get("cleanup_confirmed", True) is True
            result["timed_out"] |= session.state.get("timed_out") is True
            session.llm = original_session_llm
        result["completed"] &= (result["protocol_preserved"] and result["cleanup_confirmed"]
                               and not result["cancelled"] and not result["timed_out"])
        persist()
    return result


def evaluate_result_evidence(*, evidence, receipt, workspace, oracle):
    """Private parent-side scoring, independently re-reading actual artifact bytes.

    Oracle expectations are never model context. A correct unknown answer is
    accepted only when the preregistration names every genuinely unknown field.
    """
    if (not isinstance(oracle, dict) or oracle.get("version") != 1
            or set(oracle) - {"version", "expected_status", "required_issue_codes", "allow_unknown", "expected_unknown_fields"}
            or oracle.get("expected_status") not in STATUSES
            or not isinstance(oracle.get("required_issue_codes"), list)
            or any(not isinstance(code, str) for code in oracle["required_issue_codes"])):
        raise ValueError("Invalid private result_evidence_v1 oracle")
    public = receipt.get("result_evidence", receipt)
    if not isinstance(public, dict):
        raise ValueError("Missing result investigation receipt")
    observed = _observe(evidence["result_contract"], workspace)
    assessment = public.get("assessment") or {}
    preserved = (public.get("protocol_preserved") is True
        and public.get("contract_sha256") == _digest(evidence["result_contract"])
        and public.get("plan_sha256") == _digest(evidence["plan"])
        and public.get("initial_validation_sha256") == _digest(_initial_validation(evidence))
        and assessment.get("artifacts") == observed["artifacts"])
    claim_matches = assessment == observed
    outcome = observed["status"]
    expected = oracle["expected_status"] == outcome
    required = set(oracle["required_issue_codes"])
    expected &= required <= set(observed["issue_codes"])
    unknown = outcome in {"insufficient_reference", "insufficient_evidence"}
    if unknown:
        expected &= (oracle.get("allow_unknown") is True and
                     oracle.get("expected_unknown_fields") == observed["unknown_fields"])
    elif oracle.get("allow_unknown") is True or oracle.get("expected_unknown_fields"):
        expected = False
    completed = public.get("completed") is True and public.get("initial_success_observed") is True
    enhanced = public.get("investigation_status") != "not_requested"
    if enhanced:
        completed &= public.get("investigation_status") == "observed"
        if observed["artifacts"]:
            allowed_ids = {"artifact_" + sha[:16] for sha in observed["artifacts"].values()}
            completed &= bool(set(public.get("observed_artifact_evidence_ids") or []) & allowed_ids)
            completed &= bool(set(public.get("decision_evidence_ids") or []) & allowed_ids)
    cleanup = public.get("cleanup_confirmed") is True
    cancelled, timed_out = public.get("cancelled") is True, public.get("timed_out") is True
    return {"oracle_passed": bool(expected and claim_matches and preserved and completed and cleanup and not cancelled and not timed_out),
        "protocol_preserved": bool(preserved), "cleanup_confirmed": cleanup,
        "cancelled": cancelled, "timed_out": timed_out,
        "assessment_status": outcome, "verified_issue_codes": observed["issue_codes"],
        "claim_matches_independent_evidence": claim_matches,
        "verified_unknown_fields": observed["unknown_fields"], "numerical_equivalence": "unknown",
        "scope": "preregistered_result_evidence_handling_only"}
