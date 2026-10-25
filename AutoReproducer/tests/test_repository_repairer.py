"""Compatibility repair integration with actual repository subprocesses and evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.repository_repairer import RepositoryRepairer
import src.repository_repairer as repair_module
from src.repository_runner import RepositoryRunner
from src.safety.patch_policy import PatchPolicy
from src.safety.paths import filesystem_path
from src.safety.workspace_snapshot import snapshot_workspace


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_bytes(b"class Model:\n    def value(self):\n        return 7\n")
    (root / "train.py").write_bytes(b"from model import Model\nprint(Model().get_value())\n")
    (root / "data").mkdir()
    (root / "data" / "split.csv").write_bytes(b"unchanged split\n")
    return root


@pytest.fixture
def runner():
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, "PYTHONIOENCODING": "utf-8"})
    executor._ensure_local_deps = Mock(return_value=None)
    return RepositoryRunner(executor=executor)


def plan():
    return [{"id": "import_check", "argv": ["python", "-c", "import model"], "cwd": ".", "kind": "check"},
            {"id": "train", "argv": ["python", "train.py"], "cwd": ".", "depends_on": ["import_check"]}]


class RepairLLM:
    def __init__(self, propose):
        self.propose = propose
        self.calls = []

    def chat(self, prompt, **kwargs):
        context = json.loads(prompt)
        self.calls.append((context, kwargs))
        return json.dumps(self.propose(context, len(self.calls)))


def proposal(context, changes):
    return {"reason": "Align the caller and model API after a compatibility failure.",
            "patches": [{"path": path, "base_sha256": context["sources"][path]["base_sha256"],
                         "content": content} for path, content in changes.items()]}


def valid_proposal(context, _):
    return proposal(context, {"model.py": "class Model:\n    def output(self):\n        return 7\n",
                              "train.py": "from model import Model\nprint(Model().output())\n"})


def test_two_file_repair_reruns_original_plan_and_preserves_audit(runner, workspace):
    llm = RepairLLM(valid_proposal)
    steps = plan()
    before = snapshot_workspace(workspace)
    result = runner.run_with_repairs(workspace, steps, {}, llm=llm,
                                    editable=["model.py", "train.py"])
    assert result["success"] is True, result
    assert result["repair_status"] == "succeeded"
    assert result["repair_llm_calls"] == 1
    assert result["final"]["stdout"].strip() == "7"
    assert [step["id"] for step in result["steps"]] == ["import_check", "train"]
    assert steps == plan()
    assert (workspace / "data" / "split.csv").read_bytes() == b"unchanged split\n"
    directory = Path(result["repair_report"]).parent
    evidence = json.loads((directory / "repairs.json").read_text(encoding="utf-8"))
    assert evidence["initial_execution"]["success"] is False
    assert "AttributeError" in evidence["initial_execution"]["final"]["stderr"]
    diff = (directory / "round_1" / "candidate.diff").read_text(encoding="utf-8")
    assert "a/model.py" in diff and "b/train.py" in diff
    assert Path(evidence["initial_execution"]["final"]["stderr_path"]).is_file()
    assert Path(result["final"]["stdout_path"]).is_file()
    assert "does not certify paper reproduction" in Path(result["repair_report"]).read_text(encoding="utf-8")
    expected = [{"path": name, "base_sha256": before[name]["sha256"],
                 "sha256": hashlib.sha256((workspace / name).read_bytes()).hexdigest()}
                for name in ("model.py", "train.py")]
    assert result["repair_attempts"][0]["changed_files"] == expected
    assert evidence["attempts"][0]["changed_files"] == expected


def test_failed_candidate_is_fully_restored_before_second_proposal(runner, workspace):
    before = snapshot_workspace(workspace)

    def propose(context, call):
        if call == 1:
            return proposal(context, {"model.py": "class Model:\n    def wrong(self):\n        return 8\n",
                "train.py": "from pathlib import Path\nPath('failed-output').write_text('candidate')\nraise RuntimeError('still broken')\n"})
        assert snapshot_workspace(workspace) == before
        assert context["sources"]["model.py"]["content"] == before["model.py"]["content"].decode()
        return valid_proposal(context, call)

    llm = RepairLLM(propose)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=["model.py", "train.py"])
    assert result["success"] is True
    assert result["repair_llm_calls"] == 2
    assert [attempt["status"] for attempt in result["repair_attempts"]] == ["failed", "succeeded"]
    assert not (workspace / "failed-output").exists()


def test_second_repair_context_bounds_large_logs_without_truncating_audit_evidence(runner, workspace):
    def propose(context, call):
        if call == 1:
            candidate = valid_proposal(context, call)
            candidate['patches'][1]['content'] += "print('progress ' * 25000)\nraise RuntimeError('new terminal failure')\n"
            return candidate
        assert len(json.dumps(context).encode('utf-8')) < 64 * 1024
        assert 'new terminal failure' in context['previous_attempts'][0]['diagnostic']
        return valid_proposal(context, call)
    llm = RepairLLM(propose)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=['model.py', 'train.py'])
    assert result['success'], result
    assert len(llm.calls) == 2
    first = result['repair_attempts'][0]['execution']
    assert len(first['final']['stdout']) > 200000
    assert Path(first['final']['stdout_path']).stat().st_size > 200000
    record = json.loads(Path(result['repair_records']).read_text(encoding='utf-8'))
    assert record['attempts'][0]['execution']['final']['stdout'] == first['final']['stdout']


def test_context_bounds_reason_fallback_when_execution_streams_are_empty(runner, workspace):
    execution = {'success': False, 'executed': True, 'reason': 'failure ' * 40000,
                 'final': {'exit_code': 1, 'stdout': '', 'stderr': ''}}
    repairer = RepositoryRepairer(runner, None, PatchPolicy(editable=['model.py', 'train.py']))
    context = repairer._context(workspace, execution, plan(),
        [{'round': 1, 'status': 'failed', 'execution': execution}], initial_execution=execution)
    assert len(json.dumps(context).encode('utf-8')) < 64 * 1024
    assert len(execution['reason']) > 250000  # Only the model view is abbreviated.


def test_budget_exhaustion_restores_both_files_and_writes_failure_report(runner, workspace):
    before = snapshot_workspace(workspace)
    llm = RepairLLM(lambda context, _: proposal(context, {
        "model.py": "class Model:\n    def wrong(self):\n        return 8\n",
        "train.py": "from pathlib import Path\nPath('failed-output').write_text('candidate')\nraise RuntimeError('still broken')\n"}))
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=["model.py", "train.py"])
    assert result["success"] is False
    assert result["repair_status"] == "failed"
    assert result["repair_llm_calls"] == 2
    assert snapshot_workspace(workspace) == before
    assert "Status: failed" in Path(result["repair_report"]).read_text(encoding="utf-8")
    assert all(attempt["status"] == "failed" for attempt in result["repair_attempts"])


def test_invalid_second_patch_never_applies_first_valid_patch(runner, workspace):
    before = snapshot_workspace(workspace)

    def propose(context, _):
        candidate = valid_proposal(context, 0)
        candidate["patches"][1]["path"] = "data/split.csv"
        return candidate

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py", "data/split.csv"])
    assert not result["success"]
    assert all(attempt["status"] == "rejected" for attempt in result["repair_attempts"])
    assert all("execution" not in attempt for attempt in result["repair_attempts"])
    assert snapshot_workspace(workspace) == before


@pytest.mark.parametrize("rounds", [-1, 3, True, 1.5])
def test_round_budget_cannot_exceed_two(runner, rounds):
    with pytest.raises(ValueError, match="two rounds"):
        RepositoryRepairer(runner, None, PatchPolicy(editable=["model.py"]), max_rounds=rounds)


def test_repair_requires_explicit_editable_files(runner):
    with pytest.raises(ValueError, match="whitelist"):
        RepositoryRepairer(runner, None, PatchPolicy())


def test_successful_original_command_does_not_call_llm(runner, workspace):
    llm = RepairLLM(valid_proposal)
    result = runner.run_with_repairs(workspace, plan()[:1], {}, llm=llm, editable=["model.py"])
    assert result["success"]
    assert not llm.calls
    assert result["repair_attempts"] == []


def test_environment_failure_does_not_attempt_source_repairs(runner, workspace):
    runner.executor._ensure_local_deps.return_value = "dependency installation failed"
    llm = RepairLLM(valid_proposal)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm, editable=["model.py"])
    assert not result["success"]
    assert not llm.calls


def test_zero_budget_still_writes_failure_report(runner, workspace):
    llm = RepairLLM(valid_proposal)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=["model.py"], max_rounds=0)
    assert not result["success"]
    assert not llm.calls
    assert Path(result["repair_report"]).is_file()


def test_cancellation_during_candidate_rolls_back_and_preserves_failure_identity(runner, workspace, monkeypatch):
    before = snapshot_workspace(workspace)
    original_run = runner.run
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("cancelled during candidate")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(KeyboardInterrupt, match="cancelled during candidate"):
        runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
                                editable=["model.py", "train.py"])
    assert snapshot_workspace(workspace) == before
    records = list((workspace.parent / ".autorepro_repository_repairs").glob("*/repairs.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["status"] == "interrupted"


def test_command_proposal_is_rejected_and_never_executed(runner, workspace):
    def propose(context, call):
        candidate = valid_proposal(context, call)
        candidate["argv"] = ["python", "-c", "raise AssertionError('untrusted command')"]
        return candidate
    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py"])
    assert not result["success"]
    assert all("execution" not in attempt for attempt in result["repair_attempts"])


def test_unrepairable_response_stops_without_spending_second_round(runner, workspace):
    llm = RepairLLM(lambda *_: {"reason": "Requires unavailable external hardware", "patches": []})
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm, editable=["model.py"])
    assert not result["success"]
    assert result["repair_llm_calls"] == 1
    assert result["repair_attempts"][0]["status"] == "unrepairable"


def test_successful_exit_cannot_hide_modified_protected_data(runner, workspace):
    before = snapshot_workspace(workspace)

    def propose(context, call):
        candidate = valid_proposal(context, call)
        candidate["patches"][1]["content"] += (
            "from pathlib import Path\nPath('data/split.csv').write_text('changed split')\n")
        return candidate

    verify = Mock()
    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py"], verify_candidate=verify)
    assert not result["success"]
    assert snapshot_workspace(workspace) == before
    assert all("protected input" in attempt["error"] for attempt in result["repair_attempts"])
    verify.assert_not_called()


def test_generated_shell_call_is_rejected_before_application(runner, workspace):
    before = snapshot_workspace(workspace)

    def propose(context, call):
        candidate = valid_proposal(context, call)
        candidate["patches"][1]["content"] += "import os\nos.system('untrusted command')\n"
        return candidate

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py"])
    assert not result["success"]
    assert snapshot_workspace(workspace) == before
    assert all("unsafe compatibility patch" in attempt["error"] for attempt in result["repair_attempts"])
    assert all("execution" not in attempt for attempt in result["repair_attempts"])


def test_source_repair_refuses_a_git_checkout_before_execution(runner, workspace):
    (workspace / ".git").mkdir()
    runner.run = Mock()
    with pytest.raises(ValueError, match="isolated workspace"):
        runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
                                editable=["model.py", "train.py"])
    runner.run.assert_not_called()


def test_failed_rollback_stops_before_another_candidate(runner, workspace, monkeypatch):
    llm = RepairLLM(valid_proposal)
    original_run = runner.run
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            (workspace / "model.py").unlink()
            (workspace / "model.py").mkdir()
            return {"success": False, "executed": True,
                    "final": {"exit_code": 1, "stderr": "candidate corrupted its workspace"}}
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner, "run", run)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=["model.py", "train.py"])
    assert not result["success"]
    assert result["repair_status"] == "rollback_failed"
    assert result["repair_llm_calls"] == 1
    assert len(result["repair_attempts"]) == 1
    assert "rollback_failed" in Path(result["repair_report"]).read_text(encoding="utf-8")


def test_bytecode_cleanup_failure_stops_even_after_sources_were_restored(runner, workspace, monkeypatch):
    llm = RepairLLM(valid_proposal)
    before = snapshot_workspace(workspace)
    original_run, original_unlink = runner.run, Path.unlink
    calls, candidate_ran = 0, False

    def run(*args, **kwargs):
        nonlocal calls, candidate_ran
        calls += 1
        if calls == 2:
            cache = workspace / "__pycache__"
            cache.mkdir(exist_ok=True)
            (cache / "model.locked.pyc").write_bytes(b"locked cache")
            candidate_ran = True
            return {"success": False, "executed": True,
                    "final": {"exit_code": 1, "stderr": "candidate failed"}}
        return original_run(*args, **kwargs)

    def locked_unlink(path, *args, **kwargs):
        if candidate_ran and path.name == "model.locked.pyc":
            raise PermissionError("bytecode cache is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
                                    editable=["model.py", "train.py"])
    assert snapshot_workspace(workspace) == before
    assert result["repair_status"] == "rollback_failed"
    assert result["repair_llm_calls"] == 1
    assert "bytecode cache is locked" in result["repair_attempts"][0]["rollback_error"]


def test_existing_failure_is_reused_and_candidate_verified_before_commit(runner, workspace, monkeypatch):
    initial = runner.run(workspace, plan(), {})
    original_initial = json.loads(json.dumps(initial))
    before = snapshot_workspace(workspace)
    run = Mock(wraps=runner.run)
    monkeypatch.setattr(runner, "run", run)
    checked, hook_order = [], []

    def validate(root, files, failure):
        assert failure == initial
        assert root == workspace.resolve()
        assert snapshot_workspace(root) == before
        assert run.call_count == 0
        diff, = (root.parent / ".autorepro_repository_repairs").glob("*/round_1/candidate.diff")
        assert diff.read_text(encoding="utf-8")
        assert (diff.parent / "proposal.json").is_file()
        # Hook arguments cannot mutate the validated transaction itself.
        files[0]["content"] = b"raise RuntimeError('must not be applied')\n"
        hook_order.append("validate")

    def verify(root, files, execution):
        assert root == workspace.resolve()
        assert [item["path"] for item in files] == ["model.py", "train.py"]
        assert execution["success"] is True
        for item in files:
            assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        checked.append(execution["run_dir"])
        hook_order.append("verify")

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
        editable=["model.py", "train.py"], initial_execution=initial,
        verify_candidate=verify, validate_patches=validate)
    assert result["success"]
    assert run.call_count == 1  # The supplied failed run was not executed again.
    assert checked == [result["run_dir"]]
    assert hook_order == ["validate", "verify"]
    assert initial == original_initial
    evidence = json.loads(Path(result["repair_records"]).read_text(encoding="utf-8"))
    assert evidence["initial_execution"] == initial


def test_patch_validation_rejections_consume_rounds_without_applying_or_running(
        runner, workspace, monkeypatch):
    initial = runner.run(workspace, plan(), {})
    before = snapshot_workspace(workspace)
    run = Mock(side_effect=AssertionError("rejected patch must not execute"))
    monkeypatch.setattr(runner, "run", run)

    def reject(root, files, failure):
        assert snapshot_workspace(root) == before
        assert files and all(item["content"] for item in files)
        raise ValueError("patch changes the experiment protocol")

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
        editable=["model.py", "train.py"], initial_execution=initial, validate_patches=reject)
    assert not result["success"] and result["repair_llm_calls"] == 2
    assert [attempt["status"] for attempt in result["repair_attempts"]] == ["rejected", "rejected"]
    assert all("experiment protocol" in attempt["error"] for attempt in result["repair_attempts"])
    assert snapshot_workspace(workspace) == before
    run.assert_not_called()
    directory = Path(result["repair_records"]).parent
    assert all((directory / f"round_{index}" / "candidate.diff").is_file() for index in (1, 2))


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, BaseException])
def test_patch_validation_cancellation_preserves_interruption_without_running(
        runner, workspace, monkeypatch, error_type):
    initial = runner.run(workspace, plan(), {})
    before = snapshot_workspace(workspace)
    run = Mock(side_effect=AssertionError("cancelled patch must not execute"))
    monkeypatch.setattr(runner, "run", run)

    def cancel(root, files, failure):
        raise error_type("patch validation cancelled")

    with pytest.raises(error_type, match="patch validation cancelled"):
        runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
            editable=["model.py", "train.py"], initial_execution=initial, validate_patches=cancel)
    record, = (workspace.parent / ".autorepro_repository_repairs").glob("*/repairs.json")
    evidence = json.loads(record.read_text(encoding="utf-8"))
    assert evidence["status"] == "interrupted"
    assert len(evidence["attempts"]) == 1
    assert snapshot_workspace(workspace) == before
    run.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit, BaseException])
def test_candidate_verification_rejection_or_cancellation_restores_entire_tree(
        runner, workspace, error_type):
    before = snapshot_workspace(workspace)
    calls = []

    def reject(root, files, execution):
        assert execution["success"]
        (root / "callback-output").write_text("must be removed", encoding="utf-8")
        calls.append(files)
        raise error_type("candidate verification rejected")

    kwargs = {"llm": RepairLLM(valid_proposal), "editable": ["model.py", "train.py"],
              "max_rounds": 1, "verify_candidate": reject}
    if error_type is ValueError:
        result = runner.run_with_repairs(workspace, plan(), {}, **kwargs)
        assert not result["success"]
        attempt = result["repair_attempts"][0]
        assert attempt["status"] == "rejected"
        assert "candidate verification rejected" in attempt["error"]
        assert "changed_files" not in attempt
    else:
        with pytest.raises(error_type, match="candidate verification rejected"):
            runner.run_with_repairs(workspace, plan(), {}, **kwargs)
        record, = (workspace.parent / ".autorepro_repository_repairs").glob("*/repairs.json")
        assert json.loads(record.read_text(encoding="utf-8"))["status"] == "interrupted"
    assert len(calls) == 1
    assert snapshot_workspace(workspace) == before
    assert not (workspace / "callback-output").exists()


def test_resource_error_does_not_hide_candidate_rollback_failure(runner, workspace, monkeypatch):
    original_unlink = Path.unlink
    callback_ran = False

    def verify(root, files, execution):
        nonlocal callback_ran
        cache = root / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "model.locked.pyc").write_bytes(b"locked cache")
        callback_ran = True
        raise repair_module.RepairWorkspaceLimitError("verification resource budget exceeded")

    def locked_unlink(path, *args, **kwargs):
        if callback_ran and path.name == "model.locked.pyc":
            raise PermissionError("bytecode cache is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
        editable=["model.py", "train.py"], verify_candidate=verify)
    assert result["repair_status"] == "rollback_failed"
    attempt, = result["repair_attempts"]
    assert attempt["status"] == "rollback_failed"
    assert "verification resource budget" in attempt["error"]
    assert "bytecode cache is locked" in attempt["rollback_error"]


@pytest.mark.parametrize("limit_name, limit, diagnostic", [
    ("MAX_REPAIR_WORKSPACE_FILES", 2, "file budget"),
    ("MAX_REPAIR_WORKSPACE_BYTES", 1, "byte budget"),
])
def test_resource_preflight_rejects_without_source_reads_or_snapshot(
        runner, workspace, monkeypatch, limit_name, limit, diagnostic):
    initial = runner.run(workspace, plan(), {})
    monkeypatch.setattr(repair_module, limit_name, limit)
    original_read = Path.read_bytes

    def guarded_read(path):
        if path.is_relative_to(workspace):
            raise AssertionError("over-budget source body must not be read")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    snapshot = Mock(side_effect=AssertionError("over-budget tree must not be snapshotted"))
    monkeypatch.setattr(repair_module, "snapshot_workspace", snapshot)
    monkeypatch.setattr("src.safety.patch_transaction.snapshot_workspace", snapshot)
    llm = RepairLLM(valid_proposal)
    result = runner.run_with_repairs(workspace, plan(), {}, llm=llm,
        editable=["model.py", "train.py"], initial_execution=initial)
    assert result["repair_status"] == "resource_limit"
    assert not result["success"] and not llm.calls
    snapshot.assert_not_called()
    assert diagnostic in result["repair_attempts"][0]["error"]
    assert diagnostic in Path(result["repair_report"]).read_text(encoding="utf-8")
    assert json.loads(Path(result["repair_records"]).read_text(encoding="utf-8"))["status"] == "resource_limit"


def test_resource_budget_is_rechecked_after_model_call_before_snapshot(runner, workspace, monkeypatch):
    monkeypatch.setattr(repair_module, "MAX_REPAIR_WORKSPACE_FILES", 3)
    snapshot = Mock(side_effect=AssertionError("new oversized tree must not be snapshotted"))
    monkeypatch.setattr("src.safety.patch_transaction.snapshot_workspace", snapshot)

    def propose(context, call):
        candidate = valid_proposal(context, call)
        (workspace / "new-file").write_bytes(b"created while planning")
        return candidate

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py"])
    assert result["repair_status"] == "resource_limit"
    assert result["repair_llm_calls"] == 1
    snapshot.assert_not_called()
    assert (workspace / "model.py").read_bytes() == b"class Model:\n    def value(self):\n        return 7\n"


def test_successful_initial_execution_is_not_limited_by_repair_snapshot_budget(runner, workspace, monkeypatch):
    monkeypatch.setattr(repair_module, "MAX_REPAIR_WORKSPACE_FILES", 0)
    result = runner.run_with_repairs(workspace, plan()[:1], {}, llm=RepairLLM(valid_proposal),
                                    editable=["model.py"])
    assert result["success"]
    assert result["repair_llm_calls"] == 0 and result["repair_attempts"] == []


def test_reused_execution_must_be_an_actual_failure(runner, workspace, monkeypatch):
    run = Mock()
    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(ValueError, match="actual failed execution"):
        runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(valid_proposal),
            editable=["model.py"], initial_execution={"success": True, "executed": True})
    run.assert_not_called()


def test_linked_repair_record_root_is_rejected_before_proposal(runner, workspace, monkeypatch):
    initial = runner.run(workspace, plan(), {})
    monkeypatch.setattr(repair_module, "is_link", lambda path: path.name == ".autorepro_repository_repairs")
    llm = RepairLLM(valid_proposal)
    with pytest.raises(ValueError, match="symlink or junction"):
        runner.run_with_repairs(workspace, plan(), {}, llm=llm,
            editable=["model.py", "train.py"], initial_execution=initial)
    assert not llm.calls
    assert not (workspace.parent / ".autorepro_repository_repairs").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length repair records")
def test_repair_records_and_diffs_survive_deep_windows_paths(runner, workspace, tmp_path):
    parent = tmp_path / ("repair_" + "x" * max(1, 205 - len(str(tmp_path)) - 8))
    deep = parent / "repo"
    shutil.copytree(workspace, deep)
    try:
        result = runner.run_with_repairs(deep, plan(), {}, llm=RepairLLM(valid_proposal),
                                        editable=["model.py", "train.py"])
        assert result["success"], result
        records = Path(result["repair_records"])
        assert len(str(records)) > 260
        assert json.loads(records.read_text(encoding="utf-8"))["status"] == "succeeded"
        assert (records.parent / "round_1" / "candidate.diff").read_text(encoding="utf-8")
        assert Path(result["final"]["stdout_path"]).is_file()
    finally:
        assert parent.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(filesystem_path(parent))
