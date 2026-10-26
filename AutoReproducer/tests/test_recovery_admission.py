"""Formal results need actual registry/exposure checks, not an unused boolean."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from src.benchmark.recovery import aggregate_results, freeze_manifest, run_benchmark, validate_manifest
from src.benchmark.recovery_admission import admission_receipt, verify_admission, verify_author_source
from src.holdout_registry import digest
from scripts.evaluate_recovery import main
from test_recovery_benchmark import manifest, receipts


def store(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def inputs(tmp_path, count=12):
    value = manifest(count)
    value.pop("holdout_admission", None)
    value.pop("frozen_at", None)
    registry = {"version": 1, "papers": []}
    registry_path, audit_path = tmp_path / "registry.json", tmp_path / "audit.json"
    evidence = tmp_path / "review-history.json"
    evidence.write_text("fixture reviewed history; no product tuning", encoding="utf-8")
    audit = {"version": 1, "reviewed_at": "2026-01-01T00:00:00Z", "auditor": "trusted test operator",
        "scope": ["shared registry", "fixed development manifests and acquisition history"],
        "evidence": [{"path": evidence.name, "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
        "repository_groups": [{"id": f"repo-{index}",
            "repository_urls": [f"https://github.com/example/repo-{index}"],
            "aliases_reviewed": True, "finding": "no_tuning_found", "evidence_indices": [0],
            "paper_identity": {"title": f"Fixture Paper {index}", "aliases": [],
                "source_ids": [f"fixture:{index}"], "pdf_sha256": [digest({"pdf": index})]},
            "events": [{"kind": "discovered", "at": "2025-12-31T23:00:00Z", "evidence_index": 0}]}
            for index in range((count + 1) // 2)]}
    for index, group in enumerate(audit["repository_groups"]):
        acquisition = {"version": 1, "acquired_at": "2025-12-31T00:00:00Z",
            "repository_url": group["repository_urls"][0], "repository_revision": "1" * 40,
            "git_tree": "2" * 40, "source_files": deepcopy(value["cases"][index * 2]["source_files"]),
            "git_blobs": {"train.py": {"mode": "100644", "sha1": "3" * 40}}}
        acquisition_path = tmp_path / f"acquisition-{index}.json"
        store(acquisition_path, acquisition)
        audit["evidence"].append({"path": acquisition_path.name, "sha256": hashlib.sha256(acquisition_path.read_bytes()).hexdigest()})
        evidence_index = len(audit["evidence"]) - 1
        group["evidence_indices"].append(evidence_index)
        group["source_acquisitions"] = [{**acquisition, "evidence_index": evidence_index}]
    store(registry_path, registry)
    store(audit_path, audit)
    return value, registry, audit, registry_path, audit_path


def registered_regression(url):
    return {"id": "old-paper", "title": "Previously Debugged Paper", "aliases": ["Old title"],
        "source_ids": ["arxiv:2601.12345"], "pdf_sha256": ["a" * 64],
        "repository_url": url, "status": "regression", "retirement_reason": "product repair",
        "history": [{"outcome": "retired_after_repair", "evidence": "prior development trial"}]}


def freeze_files(tmp_path, count=12):
    value, registry, audit, registry_path, audit_path = inputs(tmp_path, count)
    frozen = freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)
    return frozen, registry, audit, registry_path, audit_path


def bind_checkout(value, audit, source, directory):
    def git(*args):
        return subprocess.check_output(["git", "-c", "core.autocrlf=false", *args], cwd=source,
                                       stderr=subprocess.PIPE).decode("utf-8").strip()
    git("init")
    git("remote", "add", "origin", value["cases"][0]["repository_url"])
    git("add", "--", "train.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-m", "fixture original")
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    acquisition = audit["repository_groups"][0]["source_acquisitions"][0]
    value["cases"][0]["repository_revision"] = acquisition["repository_revision"] = revision
    acquisition["git_tree"] = tree
    acquisition["git_blobs"]["train.py"]["sha1"] = git("rev-parse", "HEAD:train.py")
    index = acquisition["evidence_index"]
    path = directory / audit["evidence"][index]["path"]
    store(path, {key: item for key, item in acquisition.items() if key != "evidence_index"})
    audit["evidence"][index]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_108_receipts_remain_readable_but_cannot_prove_formal_holdout_completion():
    value = manifest()
    value.pop("holdout_admission")
    frozen = freeze_manifest(value)
    summary = aggregate_results(frozen, receipts(frozen))
    assert summary["arms"]["C"]["autonomous_cases"] == 12
    assert not summary["evaluation_complete"]
    assert not summary["engineering_criteria_met"]
    assert summary["recommended_mode"] == "not_evaluated"
    assert not summary["holdout_admission"]["audited"]


def test_actual_registry_and_hash_bound_exposure_review_are_frozen_and_rechecked(tmp_path):
    frozen, _, _, registry_path, audit_path = freeze_files(tmp_path)
    proof = verify_admission(frozen, audit_path=audit_path, registry_path=registry_path)
    assert proof == admission_receipt(frozen)
    assert aggregate_results(frozen, receipts(frozen))["evaluation_complete"]
    with pytest.raises(ValueError, match="original exposure"):
        verify_admission(frozen, registry_path=registry_path)
    frozen["holdout_admission"]["audit"]["scope"].append("invented later")
    with pytest.raises(ValueError, match="digest changed"):
        validate_manifest(frozen)


@pytest.mark.parametrize("as_fork", [False, True])
def test_renaming_case_or_using_a_reviewed_fork_cannot_admit_regression_repository(tmp_path, as_fork):
    value, registry, audit, registry_path, audit_path = inputs(tmp_path)
    original = "https://github.com/EXAMPLE/repo-0.git"
    registry["papers"].append(registered_regression(original))
    store(registry_path, registry)
    value["cases"][0]["id"] = "renamed-holdout-case"
    if as_fork:
        value["cases"][0]["repository_url"] = "https://github.com/fork/renamed"
        audit["repository_groups"][0]["repository_urls"].append("https://github.com/fork/renamed")
        store(audit_path, audit)
    with pytest.raises(ValueError, match="registered regression"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


@pytest.mark.parametrize("problem", ["unknown", "used_for_tuning", "missing_alias_review", "dev_event", "trial_event"])
def test_unknown_or_prior_development_exposure_never_becomes_an_unused_claim(tmp_path, problem):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    group = audit["repository_groups"][0]
    if problem in {"unknown", "used_for_tuning"}:
        group["finding"] = problem
    elif problem == "missing_alias_review":
        group["aliases_reviewed"] = False
    else:
        group["events"].append({"kind": "dev_case" if problem == "dev_event" else "holdout_trial",
            "at": "2025-12-31T23:30:00Z", "evidence_index": 0})
    store(audit_path, audit)
    with pytest.raises(ValueError, match="unknown|incomplete|prior tuning"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


def test_recorded_forks_cannot_count_as_six_repositories_or_cross_development_split(tmp_path):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    combined = audit["repository_groups"][0]
    combined["repository_urls"].extend(audit["repository_groups"][1]["repository_urls"])
    combined["source_acquisitions"].extend(audit["repository_groups"][1]["source_acquisitions"])
    combined["evidence_indices"].extend(audit["repository_groups"][1]["evidence_indices"])
    del audit["repository_groups"][1]
    store(audit_path, audit)
    frozen = freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)
    summary = aggregate_results(frozen, receipts(frozen))
    assert summary["holdout_repositories"] == 5
    assert not summary["evaluation_complete"]
    value["cases"][2]["split"] = value["cases"][3]["split"] = "dev"
    with pytest.raises(ValueError, match="aliases cannot cross"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


@pytest.mark.parametrize("problem", ["conflicting_alias", "future_review", "future_event", "naive_time", "missing_evidence", "escape"])
def test_incomplete_or_unverifiable_exposure_audits_are_rejected(tmp_path, problem):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    if problem == "conflicting_alias":
        audit["repository_groups"][1]["repository_urls"].append("https://github.com/EXAMPLE/repo-0.git")
    elif problem == "future_review":
        audit["reviewed_at"] = "9999-01-01T00:00:00Z"
    elif problem == "future_event":
        audit["repository_groups"][0]["events"][0]["at"] = "2026-01-02T00:00:00Z"
    elif problem == "naive_time":
        audit["reviewed_at"] = "2026-01-01T00:00:00"
    elif problem == "missing_evidence":
        audit["evidence"][0]["path"] = "missing.json"
    else:
        audit["evidence"][0]["path"] = "../outside.json"
    store(audit_path, audit)
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


def test_missing_or_mismatched_runtime_admission_proof_cannot_be_aggregated_as_complete(tmp_path):
    frozen, _, _, _, _ = freeze_files(tmp_path)
    rows = receipts(frozen)
    rows[0].pop("holdout_admission")
    summary = aggregate_results(frozen, rows)
    assert not summary["evaluation_complete"]
    assert len(summary["holdout_admission"]["unverified_trials"]) == 1
    rows[0]["holdout_admission"] = {"verified": True, "admission_sha256": "0" * 64}
    with pytest.raises(ValueError, match="admission receipt differs"):
        aggregate_results(frozen, rows)


@pytest.mark.parametrize("change", ["registry", "audit", "evidence"])
def test_registry_audit_and_evidence_drift_stop_before_author_or_model_execution(tmp_path, change):
    frozen, registry, audit, registry_path, audit_path = freeze_files(tmp_path, 1)
    if change == "registry":
        registry["papers"].append(registered_regression("https://github.com/another/repository"))
        store(registry_path, registry)
    elif change == "audit":
        audit["scope"].append("different searched material")
        store(audit_path, audit)
    else:
        (tmp_path / "review-history.json").write_text("changed exposure", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        run_benchmark(frozen, {}, tmp_path / "output", admission_audit_path=audit_path,
            registry_path=registry_path, trial_runner=lambda **kw: pytest.fail("must not execute"),
            oracle_evaluator=lambda **kw: pytest.fail("must not evaluate"))
    assert not (tmp_path / "output").exists()


def test_running_formal_trials_hide_audit_from_model_and_recheck_before_next_trial(tmp_path):
    value, _, audit, registry_path, audit_path = inputs(tmp_path, 1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    bind_checkout(value, audit, source, tmp_path)
    store(audit_path, audit)
    frozen = freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)
    calls = []

    def runner(**kwargs):
        assert "holdout_admission" not in kwargs["case"]
        assert "audit" not in kwargs["configuration"]
        calls.append(kwargs["arm"])
        (tmp_path / "review-history.json").write_text("mutated while trial ran", encoding="utf-8")
        return {"claimed_success": True}

    with pytest.raises(ValueError, match="evidence changed"):
        run_benchmark(frozen, {"case-0": source}, tmp_path / "output", admission_audit_path=audit_path,
            registry_path=registry_path, trial_runner=runner,
            oracle_evaluator=lambda **kw: {"oracle_passed": True, "protocol_preserved": True, "cleanup_confirmed": True})
    assert len(calls) == 1


@pytest.mark.parametrize("identity_field", ["title", "source_ids", "pdf_sha256"])
def test_regression_paper_without_repository_url_is_blocked_by_paper_identity(tmp_path, identity_field):
    value, registry, audit, registry_path, audit_path = inputs(tmp_path)
    historical = registered_regression("https://github.com/old/repository")
    historical.pop("repository_url")
    registry["papers"].append(historical)
    identity = audit["repository_groups"][0]["paper_identity"]
    identity[identity_field] = deepcopy(historical[identity_field])
    store(registry_path, registry)
    store(audit_path, audit)
    with pytest.raises(ValueError, match="paper identity is a registered regression"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


def test_one_full_source_snapshot_cannot_be_claimed_as_multiple_repositories(tmp_path):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    acquisition = audit["repository_groups"][1]["source_acquisitions"][0]
    acquisition["source_files"] = deepcopy(value["cases"][0]["source_files"])
    value["cases"][2]["source_files"] = value["cases"][3]["source_files"] = deepcopy(acquisition["source_files"])
    store(audit_path, audit)
    with pytest.raises(ValueError, match="same frozen source"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


@pytest.mark.parametrize("change", ["missing", "revision", "file_map"])
def test_source_acquisition_is_required_and_bound_to_actual_case_identity(tmp_path, change):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    group = audit["repository_groups"][0]
    if change == "missing":
        group.pop("source_acquisitions")
    elif change == "revision":
        value["cases"][0]["repository_revision"] = "9" * 40
    else:
        value["cases"][0]["source_files"]["train.py"] = "9" * 64
    store(audit_path, audit)
    with pytest.raises(ValueError, match="acquisition"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


@pytest.mark.parametrize("change", ["remote", "revision", "untracked_export", "source_bytes"])
def test_formal_author_checkout_rejects_a_relabelled_or_changed_snapshot(tmp_path, change):
    value, _, audit, registry_path, audit_path = inputs(tmp_path, 1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    bind_checkout(value, audit, source, tmp_path)
    store(audit_path, audit)
    frozen = freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)
    assert verify_author_source(frozen, frozen["cases"][0], source)["verified"] is True
    if change == "remote":
        subprocess.check_call(["git", "remote", "set-url", "origin", "https://github.com/fake/relabelled"], cwd=source)
    elif change == "revision":
        subprocess.check_call(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "different revision"], cwd=source,
            stdout=subprocess.DEVNULL)
    elif change == "untracked_export":
        export = tmp_path / "raw-export"
        export.mkdir()
        (export / "train.py").write_bytes((source / "train.py").read_bytes())
        source = export
    else:
        (source / "train.py").write_text("modified source", encoding="utf-8")
    with pytest.raises(ValueError, match="checkout|Author source bytes|source root"):
        verify_author_source(frozen, frozen["cases"][0], source)


def test_source_registration_rejects_resolved_escape_before_reading(tmp_path, monkeypatch):
    from src.benchmark import recovery_admission

    value, _, audit, _, _ = inputs(tmp_path, 1)
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "train.py"
    tracked.write_bytes(b"print('frozen original')\n")
    bind_checkout(value, audit, source, tmp_path)
    external = tmp_path / "external.py"
    external.write_bytes(tracked.read_bytes())
    resolve = Path.resolve
    read_bytes = recovery_admission._read_bytes
    reads = []

    # A junction/reparse-point resolution can escape without is_symlink().
    def resolved(path, *args, **kwargs):
        return external if path == tracked else resolve(path, *args, **kwargs)

    def observed_read(path, limit):
        reads.append(path)
        return read_bytes(path, limit)

    monkeypatch.setattr(Path, "resolve", resolved)
    monkeypatch.setattr(recovery_admission, "_read_bytes", observed_read)
    with pytest.raises(ValueError, match="resolves outside"):
        recovery_admission.collect_author_source(source)
    assert tracked not in reads


@pytest.mark.parametrize("location", ["admission", "audit", "registry", "proof"])
@pytest.mark.parametrize("wrong_type", [True, 1.0])
def test_new_admission_versions_require_exact_integers(tmp_path, location, wrong_type):
    frozen, _, _, _, _ = freeze_files(tmp_path)
    changed = deepcopy(frozen)
    changed.pop("manifest_sha256")
    admission = changed["holdout_admission"]
    if location == "proof":
        rows = receipts(frozen)
        rows[0]["holdout_admission"]["version"] = wrong_type
        with pytest.raises(ValueError, match="admission receipt differs"):
            aggregate_results(frozen, rows)
        return
    target = admission if location == "admission" else admission[location]
    target["version"] = wrong_type
    if location != "admission":
        admission[location + "_sha256"] = digest(target)
    with pytest.raises(ValueError, match="version|registry"):
        validate_manifest(changed)


@pytest.mark.parametrize("field", ["holdout_admission", "author_source_identity"])
@pytest.mark.parametrize("wrong_type", [1, 1.0, "true"])
def test_actual_admission_and_source_verification_flags_are_strict_booleans(tmp_path, field, wrong_type):
    frozen, _, _, _, _ = freeze_files(tmp_path)
    rows = receipts(frozen)
    rows[0][field]["verified"] = wrong_type
    with pytest.raises(ValueError, match="admission receipt differs"):
        aggregate_results(frozen, rows)


def test_future_manifest_freeze_cannot_make_a_later_exposure_review_eligible(tmp_path):
    frozen, _, _, _, _ = freeze_files(tmp_path)
    frozen.pop("manifest_sha256")
    frozen["frozen_at"] = "9999-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="freeze cannot be in the future"):
        validate_manifest(frozen)


@pytest.mark.parametrize("which", ["registry", "evidence"])
def test_actual_documents_and_evidence_are_read_with_byte_limits(tmp_path, monkeypatch, which):
    from src.benchmark import recovery_admission
    value, _, _, registry_path, audit_path = inputs(tmp_path)
    if which == "registry":
        monkeypatch.setattr(recovery_admission, "MAX_DOCUMENT_BYTES", 10)
    else:
        monkeypatch.setattr(recovery_admission, "MAX_EVIDENCE_BYTES", 10)
    with pytest.raises(ValueError, match="size bound"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


def test_acquisition_evidence_cannot_swap_json_integer_version_for_boolean(tmp_path):
    value, _, audit, registry_path, audit_path = inputs(tmp_path)
    acquisition_path = tmp_path / audit["evidence"][1]["path"]
    changed = json.loads(acquisition_path.read_text("utf-8"))
    changed["version"] = True
    store(acquisition_path, changed)
    audit["evidence"][1]["sha256"] = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
    store(audit_path, audit)
    with pytest.raises(ValueError, match="acquisition differs"):
        freeze_manifest(value, admission_audit_path=audit_path, registry_path=registry_path)


def test_cli_formal_freeze_reads_actual_operator_files_and_does_not_train(tmp_path, capsys):
    value, _, _, registry_path, audit_path = inputs(tmp_path)
    manifest_path, frozen_path = tmp_path / "manifest.json", tmp_path / "frozen.json"
    store(manifest_path, value)
    assert main(["freeze", "--manifest", str(manifest_path), "--admission-audit", str(audit_path),
        "--registry", str(registry_path), "--output", str(frozen_path)]) == 0
    capsys.readouterr()
    frozen = json.loads(frozen_path.read_text("utf-8"))
    assert frozen["holdout_admission"]["registry_sha256"] == digest(json.loads(registry_path.read_text("utf-8")))
    assert not (tmp_path / "trials").exists()


def test_source_registration_collects_complete_git_identity_without_running_authors_or_overwriting(tmp_path, capsys):
    value, _, audit, _, _ = inputs(tmp_path, 1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    bind_checkout(value, audit, source, tmp_path)
    (source / "poison.py").write_text("raise AssertionError('author code must never run during registration')\n", encoding="utf-8")
    subprocess.check_call(["git", "-c", "core.autocrlf=false", "add", "poison.py"], cwd=source)
    subprocess.check_call(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "full original source"], cwd=source, stdout=subprocess.DEVNULL)
    (source / "generated-untracked.log").write_text("not original author source", encoding="utf-8")
    output = tmp_path / "registered.json"
    arguments = ["register-source", "--source-root", str(source), "--output", str(output)]
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["file_count"] == 2
    assert summary["author_executed"] is False
    assert "git_blobs" not in summary and "source_files" not in summary
    registered = json.loads(output.read_text("utf-8"))
    assert registered["repository_url"] == value["cases"][0]["repository_url"]
    assert set(registered["source_files"]) == set(registered["git_blobs"]) == {"train.py", "poison.py"}
    original = output.read_bytes()
    assert main(arguments) == 1
    capsys.readouterr()
    assert output.read_bytes() == original


def test_source_registration_enforces_source_and_git_output_bounds(tmp_path, monkeypatch):
    from src.benchmark import recovery_admission
    value, _, audit, _, _ = inputs(tmp_path, 1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('frozen original')\n")
    bind_checkout(value, audit, source, tmp_path)
    monkeypatch.setattr(recovery_admission, "MAX_SOURCE_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="size bound"):
        recovery_admission.collect_author_source(source)
    monkeypatch.setattr(recovery_admission, "MAX_SOURCE_FILE_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(recovery_admission, "MAX_GIT_OUTPUT_BYTES", 4)
    with pytest.raises(ValueError, match="output size bound"):
        recovery_admission.collect_author_source(source)
