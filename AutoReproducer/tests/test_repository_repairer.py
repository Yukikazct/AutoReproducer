"""Compatibility repair integration with actual repository subprocesses and evidence."""
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.repository_repairer import RepositoryRepairer
from src.repository_runner import RepositoryRunner
from src.safety.patch_policy import PatchPolicy
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

    result = runner.run_with_repairs(workspace, plan(), {}, llm=RepairLLM(propose),
                                    editable=["model.py", "train.py"])
    assert not result["success"]
    assert snapshot_workspace(workspace) == before
    assert all("protected input" in attempt["error"] for attempt in result["repair_attempts"])


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
