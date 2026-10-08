"""Authentic-source and verdict regressions without Git commits or training.

Git archives are injected bytes, and dataset fixtures are tiny checked CSVs.
No test installs packages, downloads data, trains a model, or calls an LLM.
"""
import hashlib
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import repository_reproduction as reproduction
from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.repository_profiles import get_profile
from src.resource_manager import ResourceManager


def completed_execution(stdout=None, *, success=True, exit_code=0):
    return {
        "success": success,
        "executed": True,
        "final": {
            "exit_code": exit_code,
            "stdout": stdout if stdout is not None else (
                "Epoch: 1 | Train Loss: 0.001 Vali Loss: 0.002\n"
                ">>>>>>>testing : current_run<<<<<<<<\n"
                "test 2785\nmse:3.75e-1, mae:3.99E-1\n"
            ),
            "stdout_path": "/current_run/train_and_eval.stdout.log",
            "stderr": "",
        },
    }


@pytest.mark.parametrize(
    "profile_id,seq_len,epochs,level",
    [("dlinear_etth1_smoke", 96, 1, "smoke"),
     ("dlinear_etth1_reference", 336, 10, "reference")],
)
def test_profiles_freeze_the_real_author_experiment(profile_id, seq_len, epochs, level):
    profile = get_profile(profile_id)
    assert profile["repository"] == {
        "url": "https://github.com/cure-lab/LTSF-Linear",
        "revision": "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
    }
    assert profile["parameters"]["seq_len"] == seq_len
    assert profile["parameters"]["pred_len"] == 96
    assert profile["parameters"]["train_epochs"] == epochs
    assert profile["parameters"]["patience"] == 3
    assert profile["parameters"]["seed"] == 2021
    assert profile["validation"]["level"] == level
    assert profile["validation"]["protocol_verified"] is False
    assert profile["paper"]["reference_source"].endswith("#S5.T2")
    assert profile["paper"]["required_metrics"] == ["mse", "mae"]
    assert profile["paper"]["metrics"] == ({} if level == "smoke" else {"mse": 0.375, "mae": 0.399})
    argv = next(step["argv"] for step in profile["steps"] if step["id"] == "train_and_eval")
    assert argv[argv.index("--seq_len") + 1] == str(seq_len)
    assert argv[argv.index("--pred_len") + 1] == "96"
    assert argv[argv.index("--train_epochs") + 1] == str(epochs)
    assert argv[argv.index("--model") + 1] == "DLinear"
    assert "--individual" not in argv
    assert profile["dataset"]["kind"] == "real"
    assert profile["dataset"]["revision"] in profile["dataset"]["url"]
    assert profile["dataset"]["sha256"] == "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"


def test_callers_cannot_mutate_the_next_frozen_profile():
    profile = get_profile("dlinear_etth1_reference")
    original_hash = reproduction.spec_digest(profile)
    profile["paper"]["metrics"]["mse"] = 99
    profile["steps"][1]["argv"].append("--invented-argument")
    assert reproduction.spec_digest(profile) != original_hash
    assert reproduction.spec_digest(get_profile("dlinear_etth1_reference")) == original_hash
    assert reproduction.spec_digest(get_profile("dlinear_etth1_smoke")) != original_hash


def test_unknown_profile_is_not_silently_replaced():
    with pytest.raises(ValueError, match="未适配"):
        get_profile("a_different_paper")


def test_metric_parser_uses_only_this_final_test_and_preserves_provenance():
    execution = completed_execution(
        "mse:9, mae:8\nEpoch: 1 | Train Loss: 0.00001\n"
        ">>>>>>>testing : current_run<<<<<<<<\ntest 2785\n"
        "mse:3.75e-1, mae:.399\n"
    )
    records = reproduction.dlinear_metric_records(execution, "frozen-spec-hash")
    assert {r["name"]: r["value"] for r in records} == {"mse": 0.375, "mae": 0.399}
    assert all(r["split"] == "test" and r["stage"] == "eval" for r in records)
    assert all(r["source"] == execution["final"]["stdout_path"] for r in records)
    assert all(r["spec_sha256"] == "frozen-spec-hash" and r["seed"] == 2021 for r in records)


@pytest.mark.parametrize(
    "stdout",
    [
        "Epoch: 1 | Train Loss: 0.375\nmse:0.375, mae:0.399\n",
        ">>>>>>>testing : current_run\nloss:0.375\n",
        ">>>>>>>testing : current_run\nmse:0.375\n",
        ">>>>>>>testing : current_run\nmse:nan, mae:0.399\n",
        ">>>>>>>testing : current_run\nmse:0.375, mae:+inf\n",
        ">>>>>>>testing : current_run\nmse:1e999, mae:0.399\n",
        ">>>>>>>testing : current_run\nmse:-0.375, mae:0.399\n",
        ">>>>>>>testing : previous_run\nmse:0.375, mae:0.399\n"
        ">>>>>>>testing : current_run\nmse:0.375, mae:0.399\n",
        ">>>>>>>testing : current_run\nmse:0.375, mae:0.399\n"
        "mse:0.374, mae:0.398\n",
    ],
    ids=["no-test-marker", "train-loss-only", "missing-mae", "nan", "inf",
         "overflow", "negative", "multiple-runs", "multiple-final-metrics"],
)
def test_invalid_metric_evidence_cannot_produce_a_success_verdict(stdout):
    execution = completed_execution(stdout)
    with pytest.raises(ValueError) as rejected:
        reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(
        get_profile("dlinear_etth1_smoke"), execution, [], str(rejected.value)
    )
    assert verdict["status"] == "execution_failed"
    assert verdict["is_reproduced"] is False
    assert verdict["result_level"] == "failed"


@pytest.mark.parametrize("success,exit_code", [(False, 0), (True, 7), (False, 7)])
def test_printed_matching_metrics_cannot_rescue_a_failed_process(success, exit_code):
    execution = completed_execution(success=success, exit_code=exit_code)
    with pytest.raises(ValueError, match="未成功") as rejected:
        reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(
        get_profile("dlinear_etth1_reference"), execution, [], str(rejected.value)
    )
    assert verdict["is_reproduced"] is False
    assert verdict["result_level"] == "failed"


def test_smoke_matching_paper_values_still_is_only_smoke():
    execution = completed_execution()
    records = reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(get_profile("dlinear_etth1_smoke"), execution, records)
    assert verdict["status"] == "smoke_passed"
    assert verdict["result_level"] == "smoke_passed"
    assert verdict["is_reproduced"] is None
    assert verdict["validation"]["match"] is None
    assert verdict["llm_calls"] == 0


@pytest.mark.parametrize(
    "records",
    [[], [{"name": "mse", "value": 0.375}],
     [{"name": "mse", "value": float("nan")}, {"name": "mae", "value": 0.399}],
     [{"name": "mse", "value": 0.375}, {"name": "mae", "value": float("inf")}],
     [{"name": "mse", "value": -0.375}, {"name": "mae", "value": 0.399}],
     [{"name": "mse", "value": 0.375}, {"name": "mae", "value": 0.399},
      {"name": "training_loss", "value": 0.001}]],
    ids=["no-records", "missing-mae", "nan", "inf", "negative", "wrong-metric-contract"],
)
def test_validator_independently_rejects_incomplete_or_invalid_metric_records(records):
    verdict = reproduction.validate_repository(
        get_profile("dlinear_etth1_smoke"), completed_execution(), records
    )
    assert verdict["status"] == "execution_failed"
    assert verdict["is_reproduced"] is False
    assert verdict["validation"]["metrics_within_tolerance"] is None


def test_validator_independently_rejects_nonzero_exit_even_with_valid_records():
    execution = completed_execution()
    records = reproduction.dlinear_metric_records(execution, "spec")
    execution["final"]["exit_code"] = 1
    verdict = reproduction.validate_repository(
        get_profile("dlinear_etth1_reference"), execution, records
    )
    assert verdict["status"] == "execution_failed"
    assert verdict["is_reproduced"] is False


def test_required_prior_failure_cannot_be_hidden_by_successful_final_test():
    execution = completed_execution()
    records = reproduction.dlinear_metric_records(execution, "spec")
    execution["steps"] = [{"id": "import_check", "success": False, "exit_code": 1}]
    with pytest.raises(ValueError, match="未成功"):
        reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(get_profile("dlinear_etth1_reference"), execution, records)
    assert verdict["result_level"] == "failed"
    assert verdict["is_reproduced"] is False


def test_failed_runner_marks_execution_card_error_and_still_writes_report(tmp_path, monkeypatch):
    def export(root, profile, destination, **kwargs):
        destination.mkdir()
        return {"path": str(destination), "resolved_sha": profile["repository"]["revision"],
                "url": profile["repository"]["url"]}

    monkeypatch.setattr(reproduction, "export_repository", export)
    monkeypatch.setattr(reproduction, "prepare_dataset", lambda *a, **kw: {
        "url": "checked-data", "verified": True})
    logger = AuditLogger(log_dir=str(tmp_path / "logs"), ledger_dir=str(tmp_path / "ledger"))
    runner = Mock()
    runner.run.return_value = completed_execution(success=False, exit_code=7)
    events = []
    result = reproduction.RepositoryReproduction(tmp_path / "data", logger, runner=runner).run(
        {"experiment_profile": "dlinear_etth1_smoke", "offline": True}, on_event=events.append)
    assert result["data"]["validation"]["result_level"] == "failed"
    assert any(e.get("state") == "EXECUTE_CODE" and e["status"] == "error" for e in events)
    assert "执行失败" in result["data"]["report"]
    assert (Path(result["data"]["run_dir"]) / "report.md").is_file()


@pytest.mark.parametrize("mse,mae,within_tolerance", [(0.375, 0.399, True), (0.50, 0.399, False)])
def test_reference_numerical_agreement_is_separate_from_unverified_protocol(mse, mae, within_tolerance):
    execution = completed_execution(f">>>>>>>testing : current_run\nmse:{mse}, mae:{mae}\n")
    records = reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(get_profile("dlinear_etth1_reference"), execution, records)
    assert verdict["status"] == "inconclusive"
    assert verdict["result_level"] == "experiment_completed"
    assert verdict["is_reproduced"] is None
    assert verdict["validation"]["match"] is None
    assert verdict["validation"]["metrics_within_tolerance"] is within_tolerance
    assert verdict["metrics_comparison"]["actual"] == {"mse": mse, "mae": mae}


@pytest.mark.parametrize("mse,mae", [(0.375, 0.399), (0.389, 0.407)])
def test_verified_reference_and_independent_metrics_within_tolerance_are_reproduced(mse, mae):
    execution = completed_execution(f">>>>>>>testing : current_run\nmse:{mse}, mae:{mae}\n")
    execution["protocol_verification"] = {"pass": True}
    execution["independent_metrics"] = {"pass": True}
    records = reproduction.dlinear_metric_records(execution, "verified-spec")
    verdict = reproduction.validate_repository(get_profile("dlinear_etth1_reference"), execution, records)
    assert verdict["status"] == "reproduced"
    assert verdict["result_level"] == "reproduced"
    assert verdict["is_reproduced"] is True
    assert verdict["validation"]["match"] is True
    assert verdict["validation"]["metrics_within_tolerance"] is True
    assert verdict["scope"] == "selected_paper_experiment"


@pytest.mark.parametrize("mse,mae", [(0.50, 0.399), (0.375, 0.422)])
def test_verified_reference_requires_every_metric_within_tolerance(mse, mae):
    execution = completed_execution(f">>>>>>>testing : current_run\nmse:{mse}, mae:{mae}\n")
    execution["protocol_verification"] = {"pass": True}
    execution["independent_metrics"] = {"pass": True}
    records = reproduction.dlinear_metric_records(execution, "verified-spec")
    verdict = reproduction.validate_repository(get_profile("dlinear_etth1_reference"), execution, records)
    assert verdict["status"] == "not_reproduced"
    assert verdict["result_level"] == "experiment_completed"
    assert verdict["is_reproduced"] is False
    assert verdict["validation"]["match"] is False
    assert verdict["validation"]["metrics_within_tolerance"] is False


@pytest.mark.parametrize("missing_check", ["protocol_verification", "independent_metrics"])
@pytest.mark.parametrize("evidence", [None, {}, {"pass": False}, {"pass": 1}, {"pass": "true"}],
                         ids=["absent", "missing-pass", "rejected", "integer-true", "string-true"])
def test_reference_matching_values_need_both_explicit_verifications(missing_check, evidence):
    profile = get_profile("dlinear_etth1_reference")
    # A legacy preset flag is not evidence that this particular run was checked.
    profile["validation"]["protocol_verified"] = True
    execution = completed_execution()
    execution["protocol_verification"] = {"pass": True}
    execution["independent_metrics"] = {"pass": True}
    if evidence is None:
        del execution[missing_check]
    else:
        execution[missing_check] = evidence
    records = reproduction.dlinear_metric_records(execution, "spec")
    verdict = reproduction.validate_repository(profile, execution, records)
    assert verdict["status"] == "inconclusive"
    assert verdict["is_reproduced"] is None
    assert verdict["result_level"] == "experiment_completed"
    assert verdict["validation"]["match"] is None
    assert verdict["validation"]["metrics_within_tolerance"] is True


def test_repository_api_transport_sends_only_public_protocol_and_keeps_local_evidence_local(
        tmp_path, monkeypatch):
    """Exercise the real service/client through an intercepted HTTP transport."""
    from src import execution_artifacts
    from src import repository_validation
    from src.llm import llm_client as client_module

    private_key = "TEST_FAKE_API_KEY_PROMPT_LEAK_SENTINEL"
    private_log = "LOCAL_LOG_CONTENTS_SENTINEL"
    private_path = str(tmp_path / "PRIVATE_WORKSPACE_PATH_SENTINEL")
    local_mse, local_mae = 0.3741234567, 0.3981234567
    execution = completed_execution(
        f"{private_log} {private_path} {private_key}\n"
        f">>>>>>>testing : current_run\nmse:{local_mse}, mae:{local_mae}\n")
    execution.update(mode="repository", environment={"dependencies_path": private_path + "/deps"})
    execution["final"].update(success=True, stdout_path=private_path + "/logs/train.stdout.log",
                              stderr="LOCAL_STDERR_SENTINEL")
    author_artifact = {"name": "author_figure.png", "path": private_path + "/author_figure.png",
                       "mime_type": "image/png", "sha256": "a" * 64, "bytes": 1}
    published_artifact = {"name": "pics/published.png", "path": private_path + "/pics/published.png",
                          "mime_type": "image/png", "sha256": "c" * 64, "bytes": 1}
    independent_artifact = {"name": "test_forecast.png", "path": private_path + "/test_forecast.png",
                            "mime_type": "image/png", "sha256": "b" * 64, "bytes": 1}
    execution["artifacts"] = [published_artifact, author_artifact]
    execution["final"]["artifacts"] = [published_artifact, author_artifact]
    execution["artifact_warnings"] = ["existing author figure warning"]
    runner = Mock()
    runner.run.return_value = execution

    def export(root, profile, destination, **kwargs):
        destination.mkdir()
        return {**profile["repository"], "resolved_sha": profile["repository"]["revision"],
                "path": str(destination), "cache_path": private_path,
                "files": {"pics/published.png": "c" * 64}}

    monkeypatch.setattr(reproduction, "export_repository", export)
    monkeypatch.setattr(reproduction, "prepare_dataset", lambda root, profile, workspace, **kwargs: {
        **profile["dataset"], "path": private_path + "/dataset/ETTh1.csv", "verified": True,
        "preview": "LOCAL_DATASET_CONTENTS_SENTINEL"})
    protocol = Mock(return_value={"pass": True, "reason": "LOCAL_PROTOCOL_DETAILS_SENTINEL"})
    monkeypatch.setattr(repository_validation, "verify_dlinear_protocol", protocol)
    recomputed = Mock(return_value={"pass": True, "metrics": {"mse": local_mse, "mae": local_mae},
                                   "artifact_dir": private_path + "/artifacts",
                                   "source": private_path + "/results/pred.npy"})
    monkeypatch.setattr(reproduction, "recompute_dlinear_metrics", recomputed)
    figures = Mock(return_value={"artifacts": [independent_artifact],
                                 "artifact_warnings": ["independent figure warning"]})
    monkeypatch.setattr(execution_artifacts, "collect_images", figures)

    review = {"summary": "公开协议：DLinear在ETTh1上使用336→96配置。",
              "reference_metrics": {"mse": 0.375, "mae": 0.399},
              "limitations": ["结论只覆盖该单项实验"]}
    response = {"choices": [{"message": {"content": json.dumps(review, ensure_ascii=False)},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 128, "completion_tokens": 48, "total_tokens": 176}}
    requests = []

    def intercepted_transport(request, **kwargs):
        requests.append(request)
        assert request.full_url == "https://model-provider.invalid/v1/chat/completions"
        return io.BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(client_module.urllib.request, "urlopen", intercepted_transport)
    logger = AuditLogger(log_dir=str(tmp_path / "logs"), ledger_dir=str(tmp_path / "ledger"))
    client = LLMClient(base_url="https://model-provider.invalid/v1", model="transport-test-model",
                       api_key=private_key, timeout=2, max_tokens=256, mock_mode=False,
                       usage_hook=logger.record_llm_usage)
    result = reproduction.RepositoryReproduction(tmp_path / "data", logger, runner=runner, llm=client).run(
        {"experiment_profile": "dlinear_etth1_reference", "offline": True, "use_llm_review": True,
         "analysis_mode": "public_protocol",
         "api_key": private_key, "local_notes": private_log})
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "reproduced"
    assert client.get_call_count() == 1 and len(requests) == 1
    body = requests[0].data.decode("utf-8")
    payload = json.loads(body)
    assert payload["model"] == "transport-test-model"
    assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
    prompt = payload["messages"][0]["content"]
    public_sources = json.loads(prompt.partition("\n")[2])
    assert set(public_sources) == {"paper", "published_table_row", "implementation_details",
                                   "author_seed_statement", "author_script"}
    assert "Table 2" in public_sources["published_table_row"]
    assert "MSE 0.375; MAE 0.399" in public_sources["published_table_row"]
    assert "seq_len=336, pred_len=96" in public_sources["author_script"]
    for forbidden in [private_key, private_log, private_path, str(local_mse), str(local_mae),
                      str(tmp_path), "LOCAL_STDERR_SENTINEL", "LOCAL_DATASET_CONTENTS_SENTINEL",
                      "LOCAL_PROTOCOL_DETAILS_SENTINEL", "stdout", "stderr", "stdout_path",
                      "run_dir", "dependencies_path", "pred.npy", "metrics.json"]:
        assert forbidden not in body
    # Authentication belongs in the transport header, never the model messages.
    assert requests[0].get_header("Authorization") == "Bearer " + private_key
    analysis = result["data"]["llm_analysis"]
    assert analysis["input_scope"] == "public_paper_only" and analysis["source"] == "real_api"
    assert analysis["usage"]["total_tokens"] == 176
    assert result["data"]["total_llm_calls"] == 1
    protocol.assert_called_once()
    recomputed.assert_called_once()
    figures.assert_called_once_with(private_path + "/artifacts", "full", logger.session_id)
    actual_execution = result["data"]["execution"]
    assert actual_execution["artifacts"] == [author_artifact, independent_artifact]
    assert actual_execution["final"]["artifacts"] == actual_execution["artifacts"]
    assert published_artifact not in actual_execution["artifacts"]
    assert "existing author figure warning" in actual_execution["artifact_warnings"]
    assert "independent figure warning" in actual_execution["artifact_warnings"]
    run_dir = Path(result["data"]["run_dir"])
    assert json.loads((run_dir / "llm_analysis.json").read_text())["input_scope"] == "public_paper_only"


def tar_bytes(files, *, extra_member=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if extra_member is not None:
            archive.addfile(extra_member)
    return stream.getvalue()


def required_source_files(profile):
    return {name: f"# original committed source: {name}\n".encode()
            for name in {"run_longExp.py", *profile["repository_map"].values()}}


def fake_git(monkeypatch, source, profile, archive, *, origin=None, resolved_sha=None):
    calls = []

    def git(repo, *args, timeout=30, binary=False):
        assert Path(repo) == source
        calls.append((args, binary))
        if args == ("remote", "get-url", "origin"):
            return origin if origin is not None else profile["repository"]["url"] + ".git"
        if args == ("rev-parse", "--verify", profile["repository"]["revision"] + "^{commit}"):
            return resolved_sha if resolved_sha is not None else profile["repository"]["revision"]
        if args == ("archive", "--format=tar", profile["repository"]["revision"]):
            assert binary is True
            return archive
        raise AssertionError(f"Unexpected Git command: {args}")

    monkeypatch.setattr(reproduction, "_git", git)
    return calls


def legacy_repository(data_root):
    source = data_root / "repos" / "dlinear_etth1_cpu_smoke" / "main"
    (source / ".git").mkdir(parents=True)
    return source


def test_export_uses_fixed_commit_archive_and_leaves_dirty_cache_untouched(tmp_path, monkeypatch):
    profile = get_profile("dlinear_etth1_smoke")
    root = tmp_path / "data"
    source = legacy_repository(root)
    (source / "run_longExp.py").write_text("# uncommitted local edit\n")
    (source / "results").mkdir()
    (source / "results" / "metrics.json").write_text('{"mse": 0.001}')
    (source / "checkpoint.pth").write_bytes(b"old checkpoint")
    files = required_source_files(profile)
    archive = tar_bytes(files)
    calls = fake_git(monkeypatch, source, profile, archive)
    destination = tmp_path / "run" / "repo"
    snapshot = reproduction.export_repository(root, profile, destination, offline=True)
    assert (destination / "run_longExp.py").read_bytes() == files["run_longExp.py"]
    assert not (destination / ".git").exists()
    assert not (destination / "results").exists()
    assert not (destination / "checkpoint.pth").exists()
    assert (source / "run_longExp.py").read_text() == "# uncommitted local edit\n"
    assert (source / "results" / "metrics.json").is_file()
    assert snapshot["resolved_sha"] == profile["repository"]["revision"]
    assert snapshot["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert snapshot["files"] == {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}
    assert calls[-1] == (("archive", "--format=tar", profile["repository"]["revision"]), True)


@pytest.mark.parametrize("mismatch", ["origin", "revision"])
def test_offline_repository_rejects_mismatched_source_or_commit(tmp_path, monkeypatch, mismatch):
    profile = get_profile("dlinear_etth1_smoke")
    root = tmp_path / "data"
    source = legacy_repository(root)
    calls = fake_git(
        monkeypatch, source, profile, b"",
        origin="https://github.com/unrelated/repo" if mismatch == "origin" else None,
        resolved_sha="a" * 40 if mismatch == "revision" else None,
    )
    monkeypatch.setattr(reproduction.subprocess, "run", Mock(side_effect=AssertionError("Offline must not clone")))
    destination = tmp_path / "repo"
    with pytest.raises(RuntimeError, match="离线模式"):
        reproduction.export_repository(root, profile, destination, offline=True)
    assert not destination.exists()
    assert all(call[0][0] != "archive" for call in calls)


@pytest.mark.parametrize("unsafe_name", [
    "../outside.py", "/absolute.py", r"\absolute.py", "C:relative.py",
    r"C:\absolute.py", r"\\server\share\file.py", r"nested\..\outside.py",
    ".git/config", ".GIT/config", "nested/file:stream",
])
def test_archive_paths_cannot_escape_or_copy_git_metadata(tmp_path, monkeypatch, unsafe_name):
    profile = get_profile("dlinear_etth1_smoke")
    root = tmp_path / "data"
    source = legacy_repository(root)
    files = {**required_source_files(profile), unsafe_name: b"forbidden"}
    fake_git(monkeypatch, source, profile, tar_bytes(files))
    write = Mock(side_effect=AssertionError("Invalid archive must not write any file"))
    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(ValueError, match="越界"):
        reproduction.export_repository(root, profile, tmp_path / "repo", offline=True)
    write.assert_not_called()
    assert not (tmp_path / "repo").exists()
    assert not (tmp_path / "outside.py").exists()
    assert not (tmp_path / "repo" / ".git").exists()


def test_archive_symlink_is_rejected(tmp_path, monkeypatch):
    profile = get_profile("dlinear_etth1_smoke")
    root = tmp_path / "data"
    source = legacy_repository(root)
    member = tarfile.TarInfo("link")
    member.type = tarfile.SYMTYPE
    member.linkname = "../outside"
    fake_git(monkeypatch, source, profile, tar_bytes(required_source_files(profile), extra_member=member))
    with pytest.raises(ValueError, match="symlink"):
        reproduction.export_repository(root, profile, tmp_path / "repo", offline=True)
    assert not (tmp_path / "repo" / "link").exists()


@pytest.fixture
def small_dataset(tmp_path):
    payload = ("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
               "2016-07-01 00:00:00,1,2,3,4,5,6,7\n").encode()
    profile = get_profile("dlinear_etth1_smoke")
    profile["dataset"].update(bytes=len(payload), rows=1, sha256=hashlib.sha256(payload).hexdigest())
    root = tmp_path / "data"
    cache = root / "dataset_cache" / "ETTh1" / profile["dataset"]["sha256"] / "ETTh1.csv"
    cache.parent.mkdir(parents=True)
    return profile, root, cache, payload


def test_dataset_copy_requires_frozen_hash_size_header_and_rows(small_dataset, tmp_path):
    profile, root, cache, payload = small_dataset
    cache.write_bytes(payload)
    workspace = tmp_path / "workspace"
    provenance = reproduction.prepare_dataset(root, profile, workspace, offline=True)
    target = workspace / "dataset" / "ETTh1.csv"
    assert target.read_bytes() == payload
    assert provenance["verified"] is True
    assert provenance["rows"] == 1
    assert provenance["kind"] == "real"
    assert provenance["sha256"] == hashlib.sha256(payload).hexdigest()
    target.write_bytes(b"local experiment edit")
    assert cache.read_bytes() == payload


def test_offline_bad_dataset_hash_cannot_download_or_synthesize(small_dataset, tmp_path, monkeypatch):
    profile, root, cache, payload = small_dataset
    corrupt = payload.replace(b",1,", b",9,")
    assert len(corrupt) == len(payload)
    cache.write_bytes(corrupt)
    download = Mock(side_effect=AssertionError("Offline must not use network"))
    monkeypatch.setattr(reproduction.urllib.request, "urlopen", download)
    workspace = tmp_path / "workspace"
    with pytest.raises(RuntimeError, match="校验和不符"):
        reproduction.prepare_dataset(root, profile, workspace, offline=True)
    download.assert_not_called()
    assert not workspace.exists()
    assert cache.read_bytes() == corrupt


@pytest.mark.parametrize("problem", ["rows", "header"])
def test_checksum_alone_does_not_accept_wrong_dataset_structure(small_dataset, tmp_path, problem):
    profile, root, cache, payload = small_dataset
    if problem == "rows":
        profile["dataset"]["rows"] = 2
    else:
        payload = payload.replace(b"date,", b"time,")
        profile["dataset"]["sha256"] = hashlib.sha256(payload).hexdigest()
        cache = root / "dataset_cache" / "ETTh1" / profile["dataset"]["sha256"] / "ETTh1.csv"
        cache.parent.mkdir(parents=True)
    cache.write_bytes(payload)
    with pytest.raises(RuntimeError, match="列名或行数"):
        reproduction.prepare_dataset(root, profile, tmp_path / "workspace", offline=True)


def test_corrupt_download_never_populates_real_cache_or_synthetic_fallback(small_dataset, tmp_path, monkeypatch):
    profile, root, cache, payload = small_dataset
    monkeypatch.setattr(reproduction.urllib.request, "urlopen", Mock(return_value=io.BytesIO(payload.replace(b",1,", b",9,"))))
    with pytest.raises(RuntimeError, match="不会替换为合成数据"):
        reproduction.prepare_dataset(root, profile, tmp_path / "workspace")
    assert not cache.exists()
    assert not (tmp_path / "workspace").exists()


def test_shared_orchestrator_prepare_only_never_installs_runs_or_calls_llm(tmp_path, monkeypatch):
    root = tmp_path / "data"
    logger = AuditLogger(log_dir=str(tmp_path / "audit"), ledger_dir=str(tmp_path / "ledger"))
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("Fixed profiles must not call an LLM"))
    runner = Mock(side_effect=AssertionError("Prepare only must not instantiate a runner"))
    monkeypatch.setattr("src.repository_runner.RepositoryRunner", runner)
    exports, datasets = [], []

    def export(data_root, profile, destination, *, offline=False):
        destination = Path(destination)
        destination.mkdir(parents=True)
        exports.append((Path(data_root), profile["id"], offline))
        return {**profile["repository"], "resolved_sha": profile["repository"]["revision"],
                "path": str(destination), "files": {}}

    def dataset(data_root, profile, workspace, *, offline=False):
        datasets.append((Path(data_root), Path(workspace), offline))
        return {**profile["dataset"], "verified": True, "path": str(Path(workspace) / "dataset/ETTh1.csv")}

    monkeypatch.setattr(reproduction, "export_repository", export)
    monkeypatch.setattr(reproduction, "prepare_dataset", dataset)
    orchestrator = Orchestrator(llm_client=llm, mock_mode=False, logger=logger,
                                resource_manager=ResourceManager(data_root=str(root)))
    events = []
    result = orchestrator.run({"experiment_profile": "dlinear_etth1_smoke",
                               "prepare_only": True, "offline": True}, on_event=events.append)
    assert result["state"] == "COMPLETED", result.get("error")
    data = result["data"]
    assert exports == [(root, "dlinear_etth1_smoke", True)]
    assert datasets[0][0] == root and datasets[0][2] is True
    assert data["validation"]["status"] == "prepared"
    assert data["validation"]["is_reproduced"] is None
    assert data["execution"]["executed"] is False
    assert data["total_llm_calls"] == 0
    llm.chat.assert_not_called()
    runner.assert_not_called()
    run_dir = Path(data["run_dir"])
    assert (run_dir / "repo").is_dir()
    for name in ["experiment_spec.json", "repository.json", "dataset.json", "execution_plan.json", "environment.json", "result.json", "report.md"]:
        assert (run_dir / name).is_file(), name
    stored = json.loads((run_dir / "experiment_spec.json").read_text())
    assert stored["sha256"] == reproduction.spec_digest(stored["spec"])
    assert not (run_dir / "metrics.json").exists()
    assert not any(event.get("state") == "EXECUTE_CODE" for event in events)


def test_mock_mode_does_not_fake_a_real_repository_run(tmp_path, monkeypatch):
    logger = AuditLogger(log_dir=str(tmp_path / "audit"), ledger_dir=str(tmp_path / "ledger"))
    export = Mock(side_effect=AssertionError("Mock profile must not prepare or execute a repository"))
    monkeypatch.setattr(reproduction, "export_repository", export)
    orchestrator = Orchestrator(mock_mode=True, logger=logger,
                                resource_manager=ResourceManager(data_root=str(tmp_path / "data")))
    result = orchestrator.run({"experiment_profile": "dlinear_etth1_smoke", "prepare_only": True})
    assert result["state"] == "ERROR"
    assert "Mock" in result["error"]
    export.assert_not_called()
    assert not (tmp_path / "data" / "runs").exists()
