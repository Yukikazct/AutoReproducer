"""Frozen-product local-import recovery, with private independent assertions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_recovery_adapter import frozen_products, sha
from src.benchmark.recovery import DEFAULT_BUDGET, DEFAULT_TOOLS
from src.benchmark.recovery_adapter import RecoveryAdapter, _public_evidence


def import_case(workspace, *, modules=None, script=None, declarations=None):
    workspace.mkdir()
    sources = {"train.py": script or b"from helper import value\nprint(value())\n",
               "data/split.csv": b"frozen original split\n",
               **(modules or {"legacy/helper.py": b"def value():\n    return 7\n"})}
    for name, raw in sources.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    inventory = {name: {"sha256": sha(raw), "text_available": True} for name, raw in sources.items()}
    repository = {"url": "https://github.com/example/local-import-development", "revision": "c" * 40}
    return {"id": "local-import-development", "repository_url": repository["url"],
            "repository_revision": repository["revision"], "source_files": {name: record["sha256"] for name, record in inventory.items()},
            "category": "local_import", "split": "dev", "origin": "injected",
            "initial_evidence": {"adapter_version": 1, "plan": {"repository": repository,
                "steps": [{"id": "train", "kind": "train", "argv": ["python", "train.py"], "cwd": ".", "timeout_s": None,
                           "entrypoint": {"kind": "script", "source_path": "train.py"}}],
                "requirements": {"author": declarations or [], "compatibility": [], "repository_namespaces": []},
                "capture": {"device": "cpu"}}, "packet": {"file_inventory": inventory},
                "environment": {}, "editable": [], "initial_failure": {
                    "error_type": "ModuleNotFoundError", "diagnostic_contains": "No module named 'helper'"}}}


def setup_import(tmp_path, frozen_products, *, arm="C", responses=None, oracle_script=None, **case_options):
    workspace, environment = tmp_path / "workspace", tmp_path / "environment"
    environment.mkdir()
    case = import_case(workspace, **case_options)
    private = tmp_path / "private"
    private.mkdir()
    assertion = oracle_script or b"from helper import value\nassert value() == 7\n"
    (private / "original_assertions.py").write_bytes(assertion)
    oracle = private / "oracle.json"
    oracle.write_text(json.dumps({"version": 1, "scripts": [{
        "path": "original_assertions.py", "sha256": sha(assertion)}]}), "utf-8")
    case["oracle"] = {"kind": "original_assertions_v1", "reference": "private original assertions", "sha256": sha(oracle.read_bytes())}
    products, identities = frozen_products
    adapter = RecoveryAdapter(product_roots=products, oracle_paths={case["id"]: oracle},
        development_responses={} if responses is None else responses, worker_timeout_s=60)
    configuration = {"mode": {"A": "baseline", "B": "single", "C": "collaborative"}[arm],
        "model": "scripted-development-only", "tools": [] if arm == "A" else list(DEFAULT_TOOLS),
        "budget": None if arm == "A" else deepcopy(DEFAULT_BUDGET), "product": identities[arm]}
    return adapter, case, workspace, environment, configuration


def run(adapter, case, workspace, environment, configuration, arm="C"):
    return adapter.run_trial(case={key: deepcopy(value) for key, value in case.items() if key != "oracle"},
        arm=arm, repeat=1, workspace=workspace, environment=environment, context={}, configuration=configuration)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_unique_frozen_import_root_recovers_original_command_without_llm_or_source_edits(tmp_path, frozen_products, arm):
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, arm=arm)
    before = {name: (workspace / name).read_bytes() for name in case["source_files"]}
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is True, receipt
    assert result["cleanup_confirmed"] is True
    assert receipt["initial_failure_observed"] is True
    assert receipt["repository_pythonpath"] == ["legacy"]
    assert receipt["import_recovery_proofs"][0]["repair"]["source_path"] == "legacy/helper.py"
    assert result["usage"]["repair_actions"] == 1
    assert result["usage"]["model_requests"] == result["usage"]["probes"] == result["usage"]["source_repair_rounds"] == 0
    assert all((workspace / name).read_bytes() == raw for name, raw in before.items())
    assert ("src.intelligent_recovery" in receipt["loaded_product_modules"]) is (arm != "A")
    if arm != "A":
        assert result["usage"]["roles"] == {}
        assert receipt["collaboration"]["reviews"] == []
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is scored["protocol_preserved"] is scored["cleanup_confirmed"] is True, scored
    assert scored["numerical_equivalence"] == "unknown"
    serialized = json.dumps(receipt['import_recovery_proofs'])
    assert json.dumps(str((tmp_path / 'private').resolve()))[1:-1] not in serialized
    assert 'original_assertions.py' not in serialized


def test_two_actual_import_failures_share_repair_budget_and_restore_both_roots(tmp_path, frozen_products):
    modules = {"legacy/helper.py": b"def value():\n    return 7\n",
               "other/bonus.py": b"def extra():\n    return 2\n"}
    script = b"from helper import value\nfrom bonus import extra\nprint(value() + extra())\n"
    oracle_script = b"from helper import value\nfrom bonus import extra\nassert value() + extra() == 9\n"
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products,
        modules=modules, script=script, oracle_script=oracle_script)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is True, receipt
    assert receipt["repository_pythonpath"] == ["legacy", "other"]
    assert result["usage"]["repair_actions"] == 2
    assert len(receipt["import_recovery_proofs"]) == 2
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["oracle_passed"]


def test_new_missing_module_does_not_reset_shared_action_limit(tmp_path, frozen_products):
    modules = {"legacy/helper.py": b"def value():\n    return 7\n",
               "other/bonus.py": b"def extra():\n    return 2\n"}
    script = b"from helper import value\nfrom bonus import extra\nprint(value() + extra())\n"
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, modules=modules, script=script)
    config["budget"]["max_repair_actions"] = 1
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert result["blocker"] == "budget_exhausted"
    assert result["usage"]["repair_actions"] == 1
    assert receipt["repository_pythonpath"] == ["legacy"]
    assert len(receipt["import_recovery_proofs"]) == 1


@pytest.mark.parametrize("arm", ["A", "C"])
def test_ambiguous_namesake_is_investigated_without_guessing_root_or_installing_package(tmp_path, frozen_products, arm):
    modules = {"legacy/helper.py": b"def value():\n    return 7\n",
               "alternative/helper.py": b"def value():\n    return 8\n"}
    scripted = {"author_failure_analysis": [{"action": "blocked", "reason": "Two frozen modules remain ambiguous."}],
        "recovery_investigation": [{"action": "probe", "hypotheses": [], "reason": "Inspect both candidates.",
            "tool": "source_context", "arguments": {"paths": list(modules)}, "evidence_ids": ["failure"]},
            {"action": "needs_human", "hypotheses": [], "reason": "No evidence distinguishes the modules.",
             "evidence_ids": ["failure", "source_context"]}]}
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, arm=arm, modules=modules, responses=scripted)
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert receipt["repository_pythonpath"] == []
    assert receipt["import_recovery_proofs"] == []
    assert receipt["import_environment_preserved"] is True
    assert result["usage"]["repair_actions"] == 0
    assert result["usage"]["probes"] == (0 if arm == "A" else 1)
    if arm == "C":
        assert result["usage"]["roles"]["recovery_investigation"]["model_requests"] == 2
        assert sum(role["model_requests"] for role in result["usage"]["roles"].values()) == result["usage"]["model_requests"]
        assert receipt["collaboration"]["scope"] == "observed_recovery_activity_only"
    assert not list((environment / "dependencies").rglob("*.dist-info"))
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is False
    assert scored["protocol_preserved"] is scored["cleanup_confirmed"] is True


def test_oracle_rejects_unjustified_recorded_import_root(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products)
    result = run(adapter, case, workspace, environment, config)
    assert result["claimed_success"]
    receipt = adapter._receipts[str(workspace.resolve())]
    receipt["repository_pythonpath"] = ["data"]
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is scored["protocol_preserved"] is False


def test_external_cancel_retains_import_action_without_new_model_requests(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products)
    def after_import_action():
        try:
            return json.loads((environment / "recovery_meter.json").read_text("utf-8"))["usage"]["repair_actions"] >= 1
        except (OSError, ValueError):
            return False
    adapter.cancel_check = after_import_action
    result = run(adapter, case, workspace, environment, config)
    assert result["cancelled"] is True
    assert result["cleanup_confirmed"] is True
    assert result["usage"]["repair_actions"] == 1
    assert result["usage"]["model_requests"] == 0


def test_import_case_cannot_authorize_source_patch_or_nonterminal_signature(tmp_path):
    case = import_case(tmp_path / "workspace")
    case["initial_evidence"]["editable"] = ["train.py"]
    with pytest.raises(ValueError, match="source"):
        _public_evidence(case, tmp_path / "workspace")
    case["initial_evidence"]["editable"] = []
    case["initial_evidence"]["initial_failure"]["error_type"] = "RuntimeError"
    with pytest.raises(ValueError, match="import"):
        _public_evidence(case, tmp_path / "workspace")


def test_declared_external_namesake_never_installs_as_local_recovery(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, declarations=["helper==1.0"])
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert receipt["repository_pythonpath"] == []
    assert receipt["import_recovery_proofs"] == []
    assert receipt["runtime_preserved"] is True
    assert not list((environment / "dependencies").rglob("*.dist-info"))
    assert receipt["error_type"] == "ValueError"


def test_only_real_terminal_import_error_can_start_recovery(tmp_path, frozen_products):
    script = b"print(\"ModuleNotFoundError: No module named 'helper'\")\nraise RuntimeError('actual failure')\n"
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, script=script)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert receipt["initial_failure_observed"] is False
    assert result["claimed_success"] is False
    assert result["usage"]["repair_actions"] == result["usage"]["model_requests"] == 0


def test_author_side_effect_cannot_change_frozen_module_or_be_scored_success(tmp_path, frozen_products):
    script = (b"from helper import value\nfrom pathlib import Path\n"
              b"Path('legacy/helper.py').write_text('def value():\\n    return 8\\n')\nprint(value())\n")
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, script=script)
    result = run(adapter, case, workspace, environment, config)
    assert result["claimed_success"] is False
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is scored["protocol_preserved"] is False


def test_review_attribution_summary_redacts_details_and_deduplicates_candidates():
    from types import SimpleNamespace
    from src.benchmark.recovery_adapter import _collaboration_summary
    from src.intelligent_recovery import RecoverySession
    credential = "unit-test-review-credential"
    session = RecoverySession(SimpleNamespace(api_key=credential), mode="collaborative")
    session.state["usage"]["roles"] = {"recovery_counter_review": {
        "model_requests": 3, "known_tokens": 0, "total_tokens": None,
        "missing_usage_requests": 3, "duration_seconds": 0.1}}
    session.state["reviews"] = [
        {"stage": "patch", "reviewer": "independent", "candidate_sha256": "a" * 64,
         "action": action, "review_id": "review_" + str(index), "evidence_ids": ["failure"],
         "reason": credential + " observed", "prompt": "private raw prompt", "probe": {"arguments": "raw details"}}
        for index, action in enumerate(["probe", "reject", "reject"], 1)]
    summary = _collaboration_summary(session)
    assert summary["counts"] == {"independent_reviews": 3, "self_checks": 0,
        "review_probe_requests": 1, "distinct_patch_candidates_rejected": 1}
    assert summary["roles"]["recovery_counter_review"]["model_requests"] == 3
    serialized = json.dumps(summary)
    assert credential not in serialized and "private raw prompt" not in serialized and "raw details" not in serialized
    assert all("prompt" not in review and "probe" not in review for review in summary["reviews"])


@pytest.mark.parametrize("field,value", [("auto_prepare", True), ("dependency_health_check", True),
                                         ("unregistered_runtime_setting", "changed")])
def test_local_gate_rejects_unregistered_environment_controls_before_execution(tmp_path, monkeypatch, field, value):
    from src.benchmark.recovery_adapter import _run_local_import_recovery
    import src.author_runtime_dependencies as policy
    case = import_case(tmp_path / "workspace")
    workspace = tmp_path / "workspace"
    evidence = case["initial_evidence"]
    environment = {"python_executable": "fixed-python", "auto_prepare": False, "dependency_health_check": False}
    diagnostic = ('Traceback (most recent call last):\n'
        f'  File "{workspace / "train.py"}", line 1, in <module>\n'
        '    from helper import value\nModuleNotFoundError: No module named \'helper\'\n')
    initial = {"success": False, "executed": True, "final": {"exit_code": 1, "stderr": diagnostic},
               "effective_env_config": deepcopy(environment)}
    calls = []
    class Runner:
        def run(self, *args, **kwargs):
            calls.append(args)
            raise AssertionError("Unregistered controls must be rejected before a process starts")
    def selected_loop(runner, root, steps, effective, **kwargs):
        runner.run(root, steps, effective)
        changed = {**effective, "repository_pythonpath": ["legacy"], field: value}
        return runner.run(root, steps, changed)
    monkeypatch.setattr(policy, "run_with_dependency_repair", selected_loop)
    with pytest.raises(ValueError, match="environment"):
        _run_local_import_recovery(Runner(), initial, evidence["plan"]["steps"], evidence["plan"],
            evidence["packet"], workspace, environment, tmp_path, {}, lambda: None,
            lambda *args: None, lambda event: None)
    assert calls == []


def test_local_gate_checks_observed_environment_after_execution(tmp_path, monkeypatch):
    from src.benchmark.recovery_adapter import _run_local_import_recovery
    import src.author_runtime_dependencies as policy
    case = import_case(tmp_path / "workspace")
    workspace, evidence = tmp_path / "workspace", case["initial_evidence"]
    environment = {"python_executable": "fixed-python", "auto_prepare": False}
    diagnostic = ('Traceback (most recent call last):\n'
        f'  File "{workspace / "train.py"}", line 1, in <module>\n'
        '    from helper import value\nModuleNotFoundError: No module named \'helper\'\n')
    initial = {"success": False, "executed": True, "final": {"exit_code": 1, "stderr": diagnostic},
               "effective_env_config": deepcopy(environment)}
    class Runner:
        def run(self, root, steps, effective, **kwargs):
            return {"success": True, "executed": True, "final": {"exit_code": 0},
                    "effective_env_config": {**effective, "auto_prepare": True}}
    def selected_loop(runner, root, steps, effective, **kwargs):
        runner.run(root, steps, effective)
        return runner.run(root, steps, {**effective, "repository_pythonpath": ["legacy"]})
    monkeypatch.setattr(policy, "run_with_dependency_repair", selected_loop)
    data = {}
    with pytest.raises(ValueError, match="Actual import retry"):
        _run_local_import_recovery(Runner(), initial, evidence["plan"]["steps"], evidence["plan"],
            evidence["packet"], workspace, environment, tmp_path, data, lambda: None,
            lambda *args: None, lambda event: None)
    assert data["_import_environment_preserved"] is False


@pytest.mark.parametrize("stop", [{"cleanup_receipt": {"status": "unconfirmed"}},
                                 {"cleanup_confirmed": False}, {"final": {"timed_out": True}}])
def test_local_recovery_does_not_start_after_stop_or_unknown_cleanup(tmp_path, stop):
    from src.benchmark.recovery_adapter import _run_local_import_recovery
    class Runner:
        def run(self, *args, **kwargs):
            pytest.fail("Stopped original execution cannot start an import retry")
    case = import_case(tmp_path / "workspace")
    evidence = case["initial_evidence"]
    initial = {"success": False, "executed": True, **stop}
    data = {}
    observed = _run_local_import_recovery(Runner(), initial, evidence["plan"]["steps"], evidence["plan"],
        evidence["packet"], tmp_path / "workspace", {}, tmp_path, data, lambda: None,
        lambda *args: None, lambda event: None)
    assert observed is initial
    assert data["_import_recovery_proofs"] == []


def test_frozen_local_import_case_completes_all_nine_development_trials(tmp_path, frozen_products):
    from src.benchmark.recovery import freeze_manifest, run_benchmark
    from src.benchmark.recovery_adapter import _digest
    sources = tmp_path / "source"
    case = import_case(sources)
    case["goal"] = "Restore the unique original import root and satisfy unchanged author assertions"
    private = tmp_path / "oracle.json"
    assertion = b"from helper import value\nassert value() == 7\n"
    (tmp_path / "original_assertions.py").write_bytes(assertion)
    private.write_text(json.dumps({"version": 1, "scripts": [{
        "path": "original_assertions.py", "sha256": sha(assertion)}]}), "utf-8")
    case["oracle"] = {"kind": "original_assertions_v1", "reference": "private original assertions", "sha256": sha(private.read_bytes())}
    case["initial_evidence_sha256"] = _digest(case["initial_evidence"])
    products, identities = frozen_products
    adapter = RecoveryAdapter(product_roots=products, oracle_paths={case["id"]: private},
                              development_responses={}, worker_timeout_s=60)
    manifest = freeze_manifest({"version": 1, "repetitions": 3, "arms": identities,
        "model": "scripted-development-only", "tools": list(DEFAULT_TOOLS),
        "budget": deepcopy(DEFAULT_BUDGET), "cases": [case], "adapter_driver_sha256": adapter.driver_sha256})
    output = tmp_path / "trials"
    summary = run_benchmark(manifest, {case["id"]: sources}, output,
        trial_runner=adapter.run_trial, oracle_evaluator=adapter.evaluate_trial,
        product_roots=products, split="dev")
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 9
    assert all(row["outcome"]["autonomous_completed"] for row in rows), rows
    assert all(row["usage"]["repair_actions"] == 1 and row["usage"]["model_requests"] == 0 for row in rows)
    assert all(row["execution_metadata"]["execution_scope"] == "author_local_import_recovery" for row in rows)
    assert summary["evaluation_complete"] is False
    assert summary["natural_holdout_cases"] == 0
    assert sum(not row["outcome"]["protocol_preserved"] or row["outcome"]["false_success"] for row in rows) == 0


@pytest.mark.parametrize("arm,review_task,reviewer", [
    ("B", "recovery_self_check", "self_check"), ("C", "recovery_counter_review", "independent")])
def test_reviewer_requested_source_check_changes_next_action_and_is_attributed(tmp_path, frozen_products, arm, review_task, reviewer):
    modules = {"legacy/helper.py": b"def value():\n    return 7\n",
               "alternative/helper.py": b"def value():\n    return 8\n"}
    scripted = {"recovery_investigation": [
        {"action": "propose_repair", "hypotheses": [], "reason": "A compatible import root may be available.", "evidence_ids": ["failure"]},
        {"action": "needs_human", "hypotheses": [], "reason": "The requested inspection shows two incompatible namesakes.",
         "evidence_ids": ["failure", "source_context"]}], review_task: [
        {"action": "probe", "reason": "Inspect both original modules before choosing a root.", "evidence_ids": ["failure"],
         "probe": {"tool": "source_context", "arguments": {"paths": list(modules)}}}]}
    adapter, case, workspace, environment, config = setup_import(tmp_path, frozen_products, arm=arm, modules=modules, responses=scripted)
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert result["usage"]["probes"] == 1
    assert result["usage"]["model_requests"] == 3
    assert result["usage"]["roles"][review_task]["model_requests"] == 1
    assert result["usage"]["repair_actions"] == 0
    summary = receipt["collaboration"]
    assert summary["counts"]["review_probe_requests"] == 1
    assert summary["reviews"][0]["reviewer"] == reviewer
    assert summary["reviews"][0]["stage"] == "investigation"
    assert summary["reviews"][0]["action"] == "probe"
    assert receipt["repository_pythonpath"] == []
    assert receipt["investigation_status"] == "needs_human"
    persisted = json.loads((environment / "recovery_meter.json").read_text("utf-8"))
    assert persisted["collaboration"] == summary
    assert persisted["usage"]["roles"] == result["usage"]["roles"]


@pytest.mark.parametrize("arm,review_task", [("B", "recovery_self_check"), ("C", "recovery_counter_review")])
def test_api_recovery_receipt_reports_existing_role_activity_without_new_calls(tmp_path, frozen_products, arm, review_task):
    from test_recovery_adapter import setup_adapter
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, arm=arm)
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is True, receipt
    roles = result["usage"]["roles"]
    assert sum(role["model_requests"] for role in roles.values()) == result["usage"]["model_requests"] == 5
    assert roles[review_task]["model_requests"] == 2
    assert [row["stage"] for row in receipt["collaboration"]["reviews"]] == ["investigation", "patch"]
    assert len({row["candidate_sha256"] for row in receipt["collaboration"]["reviews"]}) == 2
    assert all(set(row) == {"stage", "reviewer", "candidate_sha256", "action", "review_id", "evidence_ids", "reason"}
               for row in receipt["collaboration"]["reviews"])
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["oracle_passed"]


def test_repeated_rejected_patch_counts_as_one_distinct_candidate(tmp_path, frozen_products):
    from test_recovery_adapter import setup_adapter
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, arm="C", reject=True)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    summary = receipt["collaboration"]
    assert summary["counts"]["distinct_patch_candidates_rejected"] == 1
    assert summary["counts"]["independent_reviews"] == 2
    assert result["usage"]["roles"]["recovery_counter_review"]["model_requests"] == 2
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["oracle_passed"] is False
