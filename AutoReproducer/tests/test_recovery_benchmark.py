"""Task-level recovery trials cannot manufacture independent acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from scripts.evaluate_recovery import main
from src.benchmark.recovery import (
    ARMS, DEFAULT_BUDGET, DEFAULT_TOOLS, _configuration, _outcome,
    _shared_configuration, aggregate_results, freeze_manifest, run_benchmark,
    validate_manifest, verify_product_identity,
)
from src.holdout_registry import digest
from src.benchmark.recovery_admission import admission_receipt, source_receipt


def test_component_success_does_not_invent_full_pipeline_success_rates():
    frozen = freeze_manifest(manifest())
    summary = aggregate_results(frozen, receipts(frozen))
    for stages in (arm['stage_outcomes'] for arm in summary['arms'].values()):
        for observation in stages.values():
            assert observation == {'successful_trials': 0, 'observed_trials': 0,
                                   'unknown_trials': 36, 'rate': None}


def test_pipeline_stage_rates_require_independent_evaluator_facts():
    frozen = freeze_manifest(manifest())
    records = receipts(frozen)
    for record in records:
        evaluated = record['evaluation']['result']
        evaluated.update(planning_completed=True, author_execution_completed=True,
                         independent_metrics_verified=False)
        record['evaluation']['result_sha256'] = digest(evaluated)
        record['outcome'] = _outcome(record['outcome'], evaluated)
    summary = aggregate_results(frozen, records)
    assert summary['arms']['C']['stage_outcomes']['planning_completed']['rate'] == 1
    assert summary['arms']['C']['stage_outcomes']['independent_metrics_verified']['rate'] == 0
    records[0]['outcome']['independent_metrics_verified'] = True
    with pytest.raises(ValueError, match='contradicts'):
        aggregate_results(frozen, records)


def test_runner_cannot_claim_unobserved_pipeline_stages():
    frozen = freeze_manifest(manifest())
    records = receipts(frozen)
    records[0]['outcome']['planning_completed'] = True
    with pytest.raises(ValueError, match='independent evaluation'):
        aggregate_results(frozen, records)


def manifest(count=12):
    source = b"print('frozen original')\n"
    cases = []
    for index in range(count):
        case_source = source + (f"# repository {index // 2}\n".encode() if count >= 12 else b"")
        evidence = {"diagnostic": "A preregistered real blocking diagnostic", "case": index}
        cases.append({"id": f"case-{index}", "repository_url": f"https://github.com/example/repo-{index // 2}",
            "repository_revision": "1" * 40, "source_files": {"train.py": hashlib.sha256(case_source).hexdigest()},
            "split": "holdout", "origin": "natural", "unused_for_tuning": True,
            "category": ("runtime_environment", "local_import", "api_compatibility", "result_evidence")[index % 4],
            "goal": "Run the original command and check its unchanged protocol",
            "oracle": {"kind": "author_contract", "reference": "private expected-output receipt",
                       "sha256": digest({"expected": index})},
            "initial_evidence": evidence, "initial_evidence_sha256": digest(evidence)})
    enhanced = {"product_revision": "b" * 40, "product_source_sha256": "c" * 64}
    value = {"version": 1, "repetitions": 3,
        "arms": {"A": {"product_revision": "a" * 40, "product_source_sha256": "d" * 64},
                 "B": deepcopy(enhanced), "C": deepcopy(enhanced)},
        "model": "fixed-model-identifier", "tools": list(DEFAULT_TOOLS),
        "budget": deepcopy(DEFAULT_BUDGET), "cases": cases}
    # Synthetic trusted evaluator receipts below exercise aggregation semantics.
    # Real file/registry admission and mutation are tested separately.
    if count >= 12:
        registry = {"version": 1, "papers": []}
        audit = {"version": 1, "reviewed_at": "2026-01-01T00:00:00+00:00",
            "auditor": "synthetic test operator", "scope": ["synthetic fixture history"],
            "evidence": [{"path": "fixture-history.json", "sha256": "0" * 64}],
            "repository_groups": [{"id": f"repo-{index}",
                "repository_urls": [f"https://github.com/example/repo-{index}"],
                "aliases_reviewed": True, "finding": "no_tuning_found",
                "paper_identity": {"title": f"Fixture Paper {index}", "aliases": [],
                    "source_ids": [f"fixture:{index}"], "pdf_sha256": [digest({"pdf": index})]},
                "source_acquisitions": [{"version": 1, "acquired_at": "2025-12-31T00:00:00Z",
                    "repository_url": f"https://github.com/example/repo-{index}", "repository_revision": "1" * 40,
                    "git_tree": "2" * 40, "source_files": deepcopy(cases[index * 2]["source_files"]),
                    "git_blobs": {"train.py": {"mode": "100644", "sha1": "3" * 40}}, "evidence_index": 0}],
                "evidence_indices": [0], "events": []} for index in range((count + 1) // 2)]}
        value["frozen_at"] = "2026-01-02T00:00:00+00:00"
        value["holdout_admission"] = {"version": 1, "registry": registry, "registry_sha256": digest(registry),
            "audit": audit, "audit_sha256": digest(audit)}
    return value


def receipts(frozen, success=None):
    success = success or (lambda case, arm, repeat: True)
    rows = []
    for case in frozen["cases"]:
        for arm in ARMS:
            for repeat in range(1, 4):
                passed = success(case, arm, repeat)
                evaluation = {"oracle_passed": passed, "protocol_preserved": True, "cleanup_confirmed": True}
                raw = {"claimed_success": passed, "human_interventions": 0}
                trial_id = uuid.uuid4().hex
                rows.append({"version": 1, "case_id": case["id"], "arm": arm, "repeat": repeat,
                    "manifest_sha256": frozen["manifest_sha256"], "initial_evidence_sha256": case["initial_evidence_sha256"],
                    "oracle_sha256": case["oracle"]["sha256"],
                    "configuration_sha256": digest(_configuration(frozen, case, arm)),
                    "shared_configuration_sha256": digest(_shared_configuration(frozen, case)),
                    "trial_id": trial_id, "workspace_id": uuid.uuid4().hex,
                    "environment_id": uuid.uuid4().hex, "context_id": uuid.uuid4().hex,
                    "product_identity": {"verified": True, "actual": deepcopy(frozen["arms"][arm])},
                    **({"holdout_admission": admission_receipt(frozen)} if frozen.get("holdout_admission") else {}),
                    **({"author_source_identity": source_receipt(frozen, case)} if frozen.get("holdout_admission") else {}),
                    "outcome": _outcome(raw, evaluation),
                    "evaluation": {"trial_id": trial_id, "oracle_sha256": case["oracle"]["sha256"],
                                   "result": evaluation, "result_sha256": digest(evaluation)},
                    "usage": {"model_requests": 2, "tokens_charged": 2000, "probes": 1,
                        "source_repair_rounds": 1, "repair_actions": 1, "duration_seconds": 1,
                        "prompt_tokens": 100, "completion_tokens": 100, "cost": None}})
    return rows


def test_shared_defaults_do_not_limit_normal_model_calls_or_total_tokens():
    frozen = freeze_manifest(manifest())
    assert frozen["budget"]["max_model_requests"] is None
    assert frozen["budget"]["max_tokens"] is None
    rows = receipts(frozen)
    rows[3]["usage"].update(model_requests=100, tokens_charged=1000000)
    assert aggregate_results(frozen, rows)["evaluation_complete"]
    assert _configuration(frozen, frozen["cases"][0], "A")["budget"] is None
    assert _configuration(frozen, frozen["cases"][0], "A")["budget_policy"] == "frozen_baseline"


@pytest.mark.parametrize('subset,total', [('same_failure_retries', 'execution_retries'),
                                          ('invalid_retries', 'model_requests')])
def test_retry_receipts_cannot_exceed_their_actual_observed_total(subset, total):
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    rows[0]['usage'].update({subset: 3, total: 2})
    with pytest.raises(ValueError, match='Retry count exceeds'):
        aggregate_results(frozen, rows)


def test_author_retry_totals_are_distinct_from_format_retries_and_remain_unknown_for_old_results():
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    summary = aggregate_results(frozen, rows)
    assert summary['arms']['C']['usage_totals']['execution_retries'] is None
    for row in rows:
        row['usage'].update(execution_retries=2, same_failure_retries=1, invalid_retries=0)
    summary = aggregate_results(frozen, rows)
    assert summary['arms']['C']['usage_totals']['execution_retries'] == 72
    assert summary['arms']['C']['usage_totals']['same_failure_retries'] == 36
    assert summary['arms']['C']['usage_totals']['invalid_retries'] == 0


def test_freeze_is_detached_and_any_source_or_oracle_edit_invalidates_it():
    original = manifest()
    frozen = freeze_manifest(original)
    assert "manifest_sha256" not in original
    original["cases"][0]["goal"] = "changed"
    assert validate_manifest(frozen, require_frozen=True)
    for key in ("goal", "initial_evidence_sha256", "repository_revision"):
        changed = deepcopy(frozen)
        changed["cases"][0][key] = "x"
        with pytest.raises(ValueError):
            validate_manifest(changed)
    changed = deepcopy(frozen)
    changed["cases"][0]["oracle"]["reference"] = "new oracle"
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(changed)


@pytest.mark.parametrize("field,value", [("split", "other"), ("origin", "unknown"),
    ("repository_revision", "main"), ("unused_for_tuning", False),
    ("source_files", {"../escape.py": "0" * 64}), ("source_files", {"C:/escape.py": "0" * 64}),
    ("source_files", {"a\\b.py": "0" * 64}), ("goal", ""), ("initial_evidence_sha256", "0" * 64)])
def test_case_preregistration_rejects_ambiguous_identity(field, value):
    record = manifest()
    record["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(record)


def test_repository_split_cannot_be_bypassed_by_case_or_url_alias():
    value = manifest()
    value["cases"][1].update(split="dev", repository_url="https://github.com/EXAMPLE/repo-0.git")
    with pytest.raises(ValueError, match="both dev and holdout"):
        validate_manifest(value)
    value = manifest()
    value["arms"]["B"]["product_revision"] = "f" * 40
    with pytest.raises(ValueError, match="same frozen"):
        validate_manifest(value)


def test_smaller_or_injected_suite_cannot_claim_acceptance():
    frozen = freeze_manifest(manifest(2))
    summary = aggregate_results(frozen, receipts(frozen))
    assert not summary["evaluation_complete"]
    assert not summary["engineering_criteria_met"]
    assert summary["recommended_mode"] == "not_evaluated"
    assert summary["arms"]["C"]["usage_totals"]["cost"] is None
    injected = manifest()
    for case in injected["cases"]:
        case["origin"] = "injected"
    frozen = freeze_manifest(injected)
    summary = aggregate_results(frozen, receipts(frozen))
    assert summary["natural_holdout_cases"] == 0
    assert len(summary["other_cases"]) == 12
    assert not summary["evaluation_complete"]


def test_many_same_kind_cases_do_not_satisfy_preregistered_coverage():
    value = manifest()
    for case in value["cases"]:
        case["category"] = "api_compatibility"
    frozen = freeze_manifest(value)
    summary = aggregate_results(frozen, receipts(frozen))
    assert summary["natural_holdout_cases"] == 12
    assert not summary["evaluation_complete"]
    assert summary["category_counts"]["result_evidence"] == 0


def test_majorities_are_per_case_not_any_success_or_108_independent_samples():
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen, lambda case, arm, repeat: arm == "C" and repeat == 1)
    summary = aggregate_results(frozen, rows)
    assert summary["evaluation_complete"]
    assert summary["arms"]["C"]["autonomous_cases"] == 0
    assert summary["cases"][0]["arms"]["C"]["autonomous_repetitions"] == 1
    assert not summary["engineering_criteria_met"]
    assert summary["arms"]["C"]["needs_human_cases"] == 12


def test_product_gain_and_collaboration_gain_are_separate():
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen, lambda case, arm, repeat: arm in {"B", "C"} or int(case["id"].split("-")[1]) < 6)
    summary = aggregate_results(frozen, rows)
    assert summary["engineering_criteria_met"]
    assert summary["comparisons"]["C_minus_A_autonomous_cases"] == 6
    assert summary["comparisons"]["C_minus_B_autonomous_cases"] == 0
    assert not summary["collaboration_gain_observed_in_sample"]
    assert summary["recommended_mode"] == "single_candidate"


def test_missing_failed_trial_remains_in_denominator_and_blocks_completion():
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    del rows[0]
    summary = aggregate_results(frozen, rows)
    assert summary["natural_holdout_cases"] == 12
    assert not summary["evaluation_complete"]
    assert summary["missing_trials"] == [{"case_id": "case-0", "arm": "A", "repeat": 1}]


def test_manifest_strings_do_not_substitute_for_actual_product_freeze_checks():
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    rows[0]["product_identity"] = {"verified": False, "actual": None}
    summary = aggregate_results(frozen, rows)
    assert not summary["evaluation_complete"]
    assert summary["unverified_product_trials"] == [{"case_id": "case-0", "arm": "A", "repeat": 1}]


@pytest.mark.parametrize("interruption", ["cleanup_unconfirmed", "cancelled", "timed_out"])
def test_interrupted_trial_never_counts_as_complete_even_if_all_receipts_exist(interruption):
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    row = rows[0]
    evaluated = row["evaluation"]["result"]
    if interruption == "cleanup_unconfirmed":
        evaluated["cleanup_confirmed"] = False
    raw = {"claimed_success": False, "human_interventions": 0,
           "cancelled": interruption == "cancelled", "timed_out": interruption == "timed_out"}
    row["evaluation"]["result_sha256"] = digest(evaluated)
    row["outcome"] = _outcome(raw, evaluated)
    summary = aggregate_results(frozen, rows)
    assert not summary["evaluation_complete"]
    assert not summary["engineering_criteria_met"]
    assert len(summary["interrupted_trials"]) == 1


def test_actual_product_identity_reuses_git_and_holdout_file_hashes(tmp_path, monkeypatch):
    from src.benchmark import recovery
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    monkeypatch.setattr(recovery, "product_hashes", lambda root: {"src/code.py": "c" * 64})
    expected = {"product_revision": "a" * 40, "product_source_sha256": digest({"src/code.py": "c" * 64})}
    assert verify_product_identity(tmp_path, expected) == expected
    monkeypatch.setattr(recovery, "product_hashes", lambda root: {"src/code.py": "d" * 64})
    with pytest.raises(ValueError, match="changed"):
        verify_product_identity(tmp_path, expected)


@pytest.mark.parametrize("change", ["duplicate", "reused_context", "different_input", "different_configuration",
                                  "oracle_receipt", "dishonest_success", "silent_failure"])
def test_receipts_reject_leakage_or_self_reported_success(change):
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen, lambda *args: False)
    if change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "reused_context":
        rows[1]["context_id"] = rows[0]["context_id"]
    elif change == "different_input":
        rows[0]["initial_evidence_sha256"] = "0" * 64
    elif change == "different_configuration":
        rows[0]["configuration_sha256"] = "0" * 64
    elif change == "oracle_receipt":
        rows[0]["evaluation"]["trial_id"] = rows[1]["trial_id"]
    elif change == "dishonest_success":
        rows[0]["outcome"]["autonomous_completed"] = True
    else:
        rows[0]["outcome"]["needs_human"] = False
    with pytest.raises(ValueError):
        aggregate_results(frozen, rows)


def test_actual_false_success_is_counted_and_budget_exhaustion_cannot_be_hidden():
    value = manifest()
    value["budget"]["max_model_requests"] = 8
    frozen = freeze_manifest(value)
    rows = receipts(frozen, lambda case, arm, repeat: arm == "C")
    rows[0]["outcome"]["claimed_success"] = True
    rows[0]["outcome"]["false_success"] = True
    rows[3]["usage"]["model_requests"] = 9
    summary = aggregate_results(frozen, rows)
    assert summary["arms"]["A"]["false_successes"] == 1
    assert not summary["criteria"]["zero_false_successes"]
    assert not summary["evaluation_complete"]
    assert summary["budget_problems"][0]["field"] == "model_requests"


@pytest.mark.parametrize("field,number", [("cost", -1), ("duration_seconds", float("nan")),
                                          ("model_requests", True), ("invalid_retries", -1)])
def test_invalid_usage_cannot_make_cost_or_retry_summaries_look_better(field, number):
    frozen = freeze_manifest(manifest())
    rows = receipts(frozen)
    rows[0]["usage"][field] = number
    with pytest.raises(ValueError):
        aggregate_results(frozen, rows)


def test_runner_resets_workspace_environment_context_and_hides_oracle(tmp_path):
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    observed = []

    def run(**kwargs):
        assert "oracle" not in kwargs["case"]
        assert kwargs["context"] == {}
        assert list(kwargs["environment"].iterdir()) == []
        assert (kwargs["workspace"] / "train.py").read_bytes() == b"print('frozen original')\n"
        kwargs["context"]["previous_result"] = "must not leak"
        (kwargs["environment"] / "cache").write_text("do not share")
        (kwargs["workspace"] / "train.py").write_text("candidate")
        observed.append((kwargs["workspace"], kwargs["environment"], kwargs["configuration"]))
        return {"claimed_success": True, "usage": {"probes": 0, "source_repair_rounds": 1, "repair_actions": 1}}

    def oracle(**kwargs):
        assert kwargs["case"]["oracle"]["reference"] == "private expected-output receipt"
        return {"oracle_passed": True, "protocol_preserved": True, "cleanup_confirmed": True,
                "author_contract": "trusted check, without hidden labels"}

    summary = run_benchmark(frozen, {"case-0": source}, tmp_path / "output", trial_runner=run, oracle_evaluator=oracle)
    assert len(observed) == 9
    assert len({paths[0] for paths in observed}) == len({paths[1] for paths in observed}) == 9
    assert (source / "train.py").read_bytes() == b"print('frozen original')\n"
    assert not summary["evaluation_complete"]
    records = [json.loads(line) for line in (tmp_path / "output/trials.jsonl").read_text("utf-8").splitlines()]
    assert records[0]["evaluation"]["result"]["author_contract"].startswith("trusted check")
    assert records[0]["evaluation"]["trial_id"] == records[0]["trial_id"]


def test_runner_rechecks_product_before_and_after_every_trial(tmp_path, monkeypatch):
    from src.benchmark import recovery
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    products = {}
    for arm in ARMS:
        directory = tmp_path / ("product-" + arm)
        directory.mkdir()
        products[arm] = directory
    checks = []

    def verify(root, expected):
        checks.append(root)
        return deepcopy(expected)

    monkeypatch.setattr(recovery, "verify_product_identity", verify)
    run_benchmark(frozen, {"case-0": source}, tmp_path / "output", product_roots=products,
        trial_runner=lambda **kwargs: {"claimed_success": False, "usage": {"probes": 0,
            "source_repair_rounds": 0, "repair_actions": 0}},
        oracle_evaluator=lambda **kwargs: {"oracle_passed": False, "protocol_preserved": True,
                                         "cleanup_confirmed": True})
    assert len(checks) == 3 + 9 * 2
    records = [json.loads(line) for line in (tmp_path / "output/trials.jsonl").read_text("utf-8").splitlines()]
    assert len(records) == 9
    assert all(row["product_identity"]["verified"] for row in records)


def test_runner_failure_is_kept_and_source_drift_stops_execution(tmp_path):
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")

    def broken(**kwargs):
        raise RuntimeError("not a successful trial")

    summary = run_benchmark(frozen, {"case-0": source}, tmp_path / "output", trial_runner=broken,
                           oracle_evaluator=lambda **kwargs: pytest.fail("No raw result to evaluate"))
    assert sum(row["completed_trials"] for row in summary["arms"].values()) == 1
    assert summary["stopped_reason"] == "cleanup_unconfirmed"
    (source / "train.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        run_benchmark(frozen, {"case-0": source}, tmp_path / "new-output", trial_runner=broken,
                      oracle_evaluator=lambda **kwargs: {})
    assert not (tmp_path / "new-output").exists()


def test_runner_records_unconfirmed_cleanup_then_stops_all_later_trials(tmp_path):
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    calls = []

    def run(**kwargs):
        calls.append(kwargs["arm"])
        return {"claimed_success": False}

    summary = run_benchmark(frozen, {"case-0": source}, tmp_path / "output", trial_runner=run,
        oracle_evaluator=lambda **kwargs: {"oracle_passed": False, "protocol_preserved": True,
                                         "cleanup_confirmed": False})
    assert len(calls) == 1
    assert summary["stopped_reason"] == "cleanup_unconfirmed"
    assert not summary["evaluation_complete"]


@pytest.mark.parametrize("problem", ["cancelled", "timed_out", "infrastructure_error"])
def test_oracle_stage_stop_prevents_later_trials(tmp_path, problem):
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    calls = []
    def runner(**kwargs):
        calls.append(kwargs["arm"])
        return {"claimed_success": True}
    evaluation = {"oracle_passed": False, "protocol_preserved": True, "cleanup_confirmed": True,
                  problem: "WorkerFailed" if problem == "infrastructure_error" else True}
    summary = run_benchmark(frozen, {"case-0": source}, tmp_path / "output", trial_runner=runner,
                           oracle_evaluator=lambda **kwargs: evaluation)
    assert len(calls) == 1
    assert summary["stopped_reason"] == ("oracle_infrastructure_error" if problem == "infrastructure_error" else problem)
    assert not summary["evaluation_complete"]


def test_frozen_driver_metadata_cannot_be_mixed():
    value = manifest()
    value["adapter_driver_sha256"] = "e" * 64
    frozen = freeze_manifest(value)
    rows = receipts(frozen)
    for row in rows:
        row["execution_metadata"] = {"driver_sha256": "e" * 64, "model_source": "provider"}
    assert aggregate_results(frozen, rows)["evaluation_complete"]
    rows[0]["execution_metadata"]["driver_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="driver"):
        aggregate_results(frozen, rows)


def test_oracle_exception_keeps_existing_success_claim_and_usage(tmp_path):
    frozen = freeze_manifest(manifest(1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")

    def oracle(**kwargs):
        raise RuntimeError("independent evaluation failed")

    summary = run_benchmark(frozen, {"case-0": source}, tmp_path / "output",
        trial_runner=lambda **kwargs: {"claimed_success": True, "usage": {"model_requests": 7}},
        oracle_evaluator=oracle)
    rows = [json.loads(line) for line in (tmp_path / "output/trials.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"]["claimed_success"] is True
    assert rows[0]["outcome"]["false_success"] is True
    assert rows[0]["usage"]["model_requests"] == 7
    assert not summary["evaluation_complete"]


def test_cli_freeze_validate_and_aggregate_do_not_overwrite_or_claim_new_training(tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest(1)), encoding="utf-8")
    frozen_path, output = tmp_path / "frozen.json", tmp_path / "summary.json"
    assert main(["validate", "--manifest", str(path)]) == 0
    assert '"acceptance_executed": false' in capsys.readouterr().out
    assert main(["freeze", "--manifest", str(path), "--output", str(frozen_path)]) == 0
    frozen = json.loads(frozen_path.read_text("utf-8"))
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(row) for row in receipts(frozen)), encoding="utf-8")
    args = ["aggregate", "--manifest", str(frozen_path), "--results", str(results), "--output", str(output)]
    assert main(args) == 0
    assert not json.loads(output.read_text("utf-8"))["evaluation_complete"]
    before = output.read_bytes()
    assert main(args) == 1
    assert output.read_bytes() == before


def test_cli_run_loads_explicit_trusted_adapter_and_independent_oracle(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_recovery
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(freeze_manifest(manifest(1))), encoding="utf-8")
    source_roots, product_roots = tmp_path / "sources.json", tmp_path / "products.json"
    source_roots.write_text(json.dumps({"case-0": "prepared-source"}), encoding="utf-8")
    product_roots.write_text(json.dumps({arm: "frozen-product" for arm in ARMS}), encoding="utf-8")
    captured = {}

    def harness(*args, **kwargs):
        captured.update(kwargs)
        return {"evaluation_complete": False, "adapter_loaded": callable(kwargs["trial_runner"])}

    monkeypatch.setattr(evaluate_recovery, "run_benchmark", harness)
    assert main(["run", "--manifest", str(path), "--source-roots", str(source_roots),
        "--product-roots", str(product_roots), "--adapter", "src.holdout_registry:digest",
        "--oracle", "src.holdout_registry:digest", "--output-dir", str(tmp_path / "trials")]) == 0
    assert '"adapter_loaded": true' in capsys.readouterr().out
    assert captured["product_roots"]["A"] == "frozen-product"
