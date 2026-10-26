"""Actual frozen-product workers, compatibility repair and private scoring."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.benchmark.recovery import DEFAULT_BUDGET, DEFAULT_TOOLS, verify_product_identity
from src.benchmark.recovery_adapter import RecoveryAdapter, _digest, _environment, _public_evidence
from src.benchmark.recovery_adapter import RecoveryAdapterCleanup
from src.holdout_registry import product_hashes

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def frozen_products(tmp_path_factory):
    """Real old checkout and detached enhanced snapshot; no live-root drift."""
    directory = tmp_path_factory.mktemp("products")
    checkout = ROOT.parent if (ROOT.parent / ".git").exists() else ROOT
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline_revision = 'e0fb5a87714751f6f44009d4f0acc8cb3e986d0b'
    available = subprocess.run(['git', 'cat-file', '-e', baseline_revision + '^{commit}'],
                               cwd=ROOT, capture_output=True)
    if available.returncode:
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            pytest.fail('Recovery integration is missing its frozen baseline Git object; checkout with fetch-depth: 0')
        pytest.skip('Recovery A/B/C integration requires the real baseline Git object; fetch full history')
    roots = {}
    for name in ("baseline", "enhanced"):
        path = directory / name
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(checkout), str(path)], check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach",
                        baseline_revision if name == 'baseline' else revision], cwd=path, check=True)
        app = path / ROOT.name if (path / ROOT.name / "src").is_dir() else path
        roots[name] = app
    # Copy only current product files, preserving the baseline checkout as-is.
    shutil.copytree(ROOT / "src", roots["enhanced"] / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("app.py", "requirements.txt"):
        shutil.copy2(ROOT / name, roots["enhanced"] / name)
    products = {"A": roots["baseline"], "B": roots["enhanced"], "C": roots["enhanced"]}
    identities = {arm: {"product_revision": baseline_revision if arm == 'A' else revision,
        "product_source_sha256": _digest(product_hashes(path))} for arm, path in products.items()}
    for arm, path in products.items():
        verify_product_identity(path, identities[arm])
    return products, identities


def case_data(workspace):
    workspace.mkdir()
    sources = {"model.py": b"class Model:\n    def value(self):\n        return 7\n",
               "train.py": b"from model import Model\nprint(Model().get_value())\n",
               "data/split.csv": b"frozen author split\n"}
    for name, raw in sources.items():
        path = workspace / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    inventory = {name: {"sha256": sha(raw), "text_available": True} for name, raw in sources.items()}
    plan = {"repository": {"url": "https://github.com/example/adapter-development", "revision": "b" * 40},
            "steps": [{"id": "train", "kind": "train", "argv": ["python", "train.py"], "cwd": ".",
                       "timeout_s": None, "entrypoint": {"kind": "script", "source_path": "train.py"}}],
            "requirements": {"author": [], "compatibility": []}}
    return {"id": "api-development", "repository_url": plan["repository"]["url"],
        "repository_revision": "b" * 40, "source_files": {name: sha(raw) for name, raw in sources.items()},
        "category": "api_compatibility", "split": "dev", "origin": "injected",
        "initial_evidence": {"adapter_version": 1, "plan": plan,
           "packet": {"file_inventory": inventory}, "environment": {}, "editable": ["train.py"],
           "initial_failure": {"error_type": "AttributeError", "diagnostic_contains": "get_value"}}}


def responses(case, *, patch=None, reject=False):
    patch = patch or "from model import Model\nprint(Model().value())\n"
    return {"author_failure_analysis": [{"action": "blocked", "reason": "Use constrained source repair."}],
            "repository_repair": [{"reason": "Align the caller with the existing model method.",
                "patches": [{"path": "train.py", "base_sha256": case["source_files"]["train.py"], "content": patch}]}] * 2,
            "recovery_investigation": [{"hypotheses": [], "action": "propose_repair",
                "reason": "Observed caller uses a missing API.", "evidence_ids": ["failure", "source_context"]}],
            "recovery_counter_review": [
                {"action": "accept", "reason": "Evidence permits a bounded candidate.", "evidence_ids": ["failure"]},
                {"action": "reject" if reject else "accept", "reason": "Check original API contract.",
                 "evidence_ids": ["original_failure", "patch_sources"]}] * 2,
            "recovery_self_check": [
                {"action": "accept", "reason": "Evidence permits a bounded candidate.", "evidence_ids": ["failure"]},
                {"action": "accept", "reason": "Only the missing API is changed.",
                 "evidence_ids": ["original_failure", "patch_sources"]}] * 2}


def setup_adapter(tmp_path, frozen_products, *, arm="C", patch=None, reject=False, oracle_kind="trusted_stdout_v1"):
    workspace, environment = tmp_path / "workspace", tmp_path / "environment"
    environment.mkdir()
    case = case_data(workspace)
    private = tmp_path / "private"
    private.mkdir()
    oracle = {"version": 1, "stdout_sha256": sha(b"7\n")}
    if oracle_kind == "original_assertions_v1":
        assertion = b"from model import Model\nassert Model().value() == 7\n"
        (private / "author_contract.py").write_bytes(assertion)
        oracle = {"version": 1, "scripts": [{"path": "author_contract.py", "sha256": sha(assertion)}]}
    reference = private / "oracle.json"
    reference.write_text(json.dumps(oracle), "utf-8")
    case["oracle"] = {"kind": oracle_kind, "reference": "private/oracle.json", "sha256": sha(reference.read_bytes())}
    roots, identities = frozen_products
    adapter = RecoveryAdapter(product_roots=roots, oracle_paths={case["id"]: reference},
        development_responses=responses(case, patch=patch, reject=reject), worker_timeout_s=60)
    configuration = {"mode": {"A": "baseline", "B": "single", "C": "collaborative"}[arm],
        "model": "scripted-development-only", "tools": [] if arm == "A" else list(DEFAULT_TOOLS),
        "budget": None if arm == "A" else deepcopy(DEFAULT_BUDGET), "product": identities[arm]}
    return adapter, case, workspace, environment, configuration


def run(adapter, case, workspace, environment, config, arm="C"):
    public = {key: deepcopy(value) for key, value in case.items() if key != "oracle"}
    return adapter.run_trial(case=public, arm=arm, repeat=1, workspace=workspace,
                             environment=environment, context={}, configuration=config)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_actual_three_arms_use_frozen_code_and_original_commands(tmp_path, frozen_products, arm):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, arm=arm)
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"], receipt
    assert result["cleanup_confirmed"] is True
    assert receipt["initial_failure_observed"] is True
    assert receipt["plan_preserved"] is receipt["runtime_preserved"] is True
    assert receipt["loaded_product_modules"]["src.repository_runner"] == "src/repository_runner.py"
    assert ("src.intelligent_recovery" in receipt["loaded_product_modules"]) is (arm != "A")
    assert (workspace / "train.py").read_text() == "from model import Model\nprint(Model().get_value())\n"
    assert (Path(receipt["effective_workspace"]) / "train.py").read_text() == "from model import Model\nprint(Model().value())\n"
    assert Path(receipt["author_python"]).is_relative_to(environment)
    assert result["usage"]["model_requests"] >= (2 if arm == "A" else 5)
    assert result["usage"]["prompt_tokens"] is result["usage"]["cost"] is None
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is scored["protocol_preserved"] is scored["cleanup_confirmed"] is True, receipt
    assert scored["numerical_equivalence"] == "unknown"


@pytest.mark.parametrize("arm", ["A", "C"])
def test_literal_change_is_rejected_and_complete_candidate_is_rolled_back(tmp_path, frozen_products, arm):
    patch = "from model import Model\nprint(Model().value() + 1)\n"
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, arm=arm, patch=patch)
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert result["cleanup_confirmed"] is True
    candidate = Path(receipt["effective_workspace"])
    assert (candidate / "train.py").read_bytes() == (workspace / "train.py").read_bytes()
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is False
    assert scored["protocol_preserved"] is scored["cleanup_confirmed"] is True


def test_counter_review_really_vetoes_candidate(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, reject=True)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert receipt["repair_status"] != "succeeded"
    assert (Path(receipt["effective_workspace"]) / "train.py").read_bytes() == (workspace / "train.py").read_bytes()


def test_private_original_assertions_execute_after_worker_exit(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, oracle_kind="original_assertions_v1")
    result = run(adapter, case, workspace, environment, config)
    assert result["claimed_success"], adapter._receipts[str(workspace.resolve())]
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is scored["cleanup_confirmed"] is True
    assert "author_contract" not in (environment / "adapter_receipt.json").read_text("utf-8")
    assert not list(workspace.rglob("oracle*"))


def test_success_claim_cannot_replace_private_expected_output(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products)
    result = run(adapter, case, workspace, environment, config)
    reference = adapter.oracle_paths[case["id"]]
    reference.write_text(json.dumps({"version": 1, "stdout_sha256": sha(b"WRONG\n")}), "utf-8")
    case["oracle"]["sha256"] = sha(reference.read_bytes())
    assert result["claimed_success"] is True
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["oracle_passed"] is False


def test_reused_trial_and_scripted_holdout_are_rejected(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products)
    case["split"], case["origin"] = "holdout", "natural"
    with pytest.raises(ValueError, match="Scripted"):
        run(adapter, case, workspace, environment, config)
    case["split"], case["origin"] = "dev", "injected"
    case["category"] = "result_evidence"
    result = run(adapter, case, workspace, environment, config)
    assert result["blocker"] == "unsupported"
    assert result["driver_sha256"] == adapter.driver_sha256
    assert result["usage"]["tokens_charged"] == 0
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["unsupported"] is True
    with pytest.raises(ValueError, match="reused"):
        run(adapter, case, workspace, environment, config)


@pytest.mark.parametrize("argv", [["cmd", "/c", "echo 7"], ["python", "-c", "print(7)"],
                                 ["python", "-m", "module"], ["python", "../escape.py"]])
def test_case_cannot_select_shell_or_inline_code(tmp_path, argv):
    case = case_data(tmp_path / "workspace")
    case["initial_evidence"]["plan"]["steps"][0]["argv"] = argv
    with pytest.raises((ValueError, FileNotFoundError)):
        _public_evidence(case, tmp_path / "workspace")


def test_child_environment_removes_credentials_and_python_injection(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "another-test-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://credential@example.invalid")
    monkeypatch.setenv("PYTHONPATH", "host-injection")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    child = _environment()
    assert not {"LLM_API_KEY", "OPENAI_API_KEY", "PIP_INDEX_URL", "PYTHONPATH", "PYTHONOPTIMIZE"} & child.keys()


def test_worker_identity_change_is_rejected(tmp_path, frozen_products):
    adapter, *_ = setup_adapter(tmp_path, frozen_products)
    adapter.driver_files[str(adapter.worker)] = "0" * 64
    with pytest.raises(ValueError, match="worker changed"):
        adapter._verify_driver()


def test_budget_stop_retains_calls_and_cumulative_usage(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products)
    config["budget"]["max_model_requests"] = 1
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result["claimed_success"] is False, receipt
    assert result["blocker"] == "budget_exhausted"
    assert result["usage"]["model_requests"] == 1
    assert result["usage"]["tokens_charged"] > 0
    assert result["usage"]["source_repair_rounds"] == result["usage"]["repair_actions"] == 0
    assert result["cleanup_confirmed"] is True


def test_original_oracle_uses_actual_trial_dependency_paths(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, oracle_kind="original_assertions_v1")
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    dependency = environment / "dependencies" / "target"
    dependency.mkdir(parents=True)
    (dependency / "author_dependency.py").write_text("value = 7\n", "utf-8")
    receipt["dependency_paths"] = [str(dependency)]
    private = adapter.oracle_paths[case["id"]]
    assertion = b"import author_dependency\nassert author_dependency.value == 7\n"
    (private.parent / "author_contract.py").write_bytes(assertion)
    private.write_text(json.dumps({"version": 1, "scripts": [{"path": "author_contract.py", "sha256": sha(assertion)}]}), "utf-8")
    case["oracle"]["sha256"] = sha(private.read_bytes())
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)["oracle_passed"] is True


@pytest.mark.parametrize("flag", ["cancelled", "timed_out"])
def test_oracle_stop_reaches_harness_even_after_successful_recovery(tmp_path, frozen_products, monkeypatch, flag):
    from src.benchmark.recovery import _outcome
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, oracle_kind="original_assertions_v1")
    result = run(adapter, case, workspace, environment, config)
    assert result["claimed_success"]
    monkeypatch.setattr(adapter, "_invoke", lambda *args, **kwargs: {
        "oracle_passed": False, "cleanup_confirmed": True, flag: True})
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored[flag] is True
    outcome = _outcome(result, scored)
    assert outcome[flag] is True
    assert outcome["autonomous_completed"] is False


@pytest.mark.parametrize("record", [
    {"cleanup_errors": ["unconfirmed"]}, {"status": "cleanup_unconfirmed"},
    {"cleanup_receipt": {"status": "unconfirmed"}}, {"cleanup_receipt": {"confirmed": False}},
    {"final": {"stderr": "process-tree cleanup failed"}},
    {"steps": [{"cleanup_receipt": {"process_cleanup_confirmed": False}}]},
])
def test_all_unconfirmed_cleanup_shapes_block_further_trials(record):
    assert RecoveryAdapterCleanup.unconfirmed(record) is True


def test_default_cli_runs_all_nine_development_trials(tmp_path, frozen_products, monkeypatch, capsys):
    from scripts.evaluate_recovery import main
    from src.benchmark.recovery import freeze_manifest
    import src.benchmark.recovery_adapter as adapter_module
    sources = tmp_path / "author_source"
    case = case_data(sources)
    case["goal"] = "Recover the original API call with frozen settings"
    private = tmp_path / "oracle.json"
    private.write_text(json.dumps({"version": 1, "stdout_sha256": sha(b"7\n")}), "utf-8")
    case["oracle"] = {"kind": "trusted_stdout_v1", "reference": "private stdout reference", "sha256": sha(private.read_bytes())}
    case["initial_evidence_sha256"] = _digest(case["initial_evidence"])
    roots, identities = frozen_products
    frozen = freeze_manifest({"version": 1, "repetitions": 3, "arms": identities,
        "model": "scripted-development-only", "tools": list(DEFAULT_TOOLS),
        "budget": deepcopy(DEFAULT_BUDGET), "cases": [case]})
    config = {"product_roots": {key: str(value) for key, value in roots.items()},
        "oracle_paths": {case["id"]: str(private)}, "product_python": sys.executable,
        "development_responses": responses(case), "worker_timeout_s": 60}
    frozen.pop("manifest_sha256")
    frozen["adapter_driver_sha256"] = RecoveryAdapter(**config).driver_sha256
    frozen = freeze_manifest(frozen)
    files = {"manifest": frozen, "sources": {case["id"]: str(sources)},
             "products": config["product_roots"], "adapter": config}
    for name, value in files.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value), "utf-8")
    monkeypatch.setenv("AUTOREPRO_RECOVERY_ADAPTER_CONFIG", str(tmp_path / "adapter.json"))
    monkeypatch.setattr(adapter_module, "_DEFAULT_ADAPTER", None)
    output = tmp_path / "trials"
    code = main(["run", "--manifest", str(tmp_path / "manifest.json"),
        "--source-roots", str(tmp_path / "sources.json"), "--product-roots", str(tmp_path / "products.json"),
        "--adapter", "src.benchmark.recovery_adapter:run_trial", "--oracle", "src.benchmark.recovery_adapter:evaluate_trial",
        "--split", "dev", "--output-dir", str(output)])
    captured = capsys.readouterr()
    assert code == 0, captured
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 9
    assert all(row["outcome"]["autonomous_completed"] is True for row in rows), rows
    assert all(row["execution_metadata"]["driver_sha256"] == frozen["adapter_driver_sha256"] for row in rows)
    summary = json.loads((output / "summary.json").read_text("utf-8"))
    assert summary["evaluation_complete"] is False


def test_external_cancel_retains_started_model_calls(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products)
    def cancel_after_call():
        checkpoint = environment / "recovery_meter.json"
        try:
            return json.loads(checkpoint.read_text("utf-8"))["usage"]["model_requests"] >= 1
        except (OSError, ValueError):
            return False
    adapter.cancel_check = cancel_after_call
    result = run(adapter, case, workspace, environment, config)
    assert result["cancelled"] is True
    assert result["claimed_success"] is False
    assert result["cleanup_confirmed"] is True
    assert result["usage"]["model_requests"] >= 1


def test_oracle_infrastructure_error_is_distinct_from_assertion_failure(tmp_path, frozen_products, monkeypatch):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products, oracle_kind="original_assertions_v1")
    result = run(adapter, case, workspace, environment, config)
    monkeypatch.setattr(adapter, "_invoke", lambda *args, **kwargs: {
        "oracle_passed": False, "cleanup_confirmed": True, "error_type": "ModuleNotFoundError"})
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["infrastructure_error"] == "ModuleNotFoundError"
    monkeypatch.setattr(adapter, "_invoke", lambda *args, **kwargs: {
        "oracle_passed": False, "cleanup_confirmed": True, "error_type": "AssertionError"})
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert "infrastructure_error" not in scored


def test_private_oracle_corruption_is_reported_as_infrastructure_error(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_adapter(tmp_path, frozen_products)
    result = run(adapter, case, workspace, environment, config)
    adapter.oracle_paths[case["id"]].write_text("broken private reference", "utf-8")
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["infrastructure_error"] == "ValueError"
    assert scored["oracle_passed"] is False
