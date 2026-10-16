"""Real filesystem checks for candidate validation, application and complete rollback."""
import hashlib
import os
import py_compile
import subprocess
import sys

import pytest

from src.safety.patch_policy import PatchPolicy
from src.safety.patch_transaction import PatchTransaction
import src.safety.patch_transaction as transaction_module
from src.safety.workspace_snapshot import snapshot_workspace


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_bytes(b"VALUE = 1\n")
    (root / "train.py").write_bytes(b"from model import VALUE\nprint(VALUE)\n")
    (root / "data").mkdir()
    (root / "data" / "split.csv").write_bytes(b"frozen data\n")
    return root


def patch(root, name, content):
    return {"path": name, "base_sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "content": content}


def test_valid_candidate_changes_both_files_and_commits(workspace):
    proposals = [patch(workspace, "model.py", "VALUE = 2\n"),
                 patch(workspace, "train.py", "from model import VALUE\nprint(VALUE + 1)\n")]
    transaction = PatchTransaction(workspace, proposals, PatchPolicy())
    with transaction:
        assert (workspace / "model.py").read_bytes() == b"VALUE = 2\n"
        assert b"VALUE + 1" in (workspace / "train.py").read_bytes()
        transaction.commit()
    assert (workspace / "model.py").read_bytes() == b"VALUE = 2\n"
    assert "--- a/model.py" in transaction.diff
    assert "+++ b/train.py" in transaction.diff
    assert (workspace / "data" / "split.csv").read_bytes() == b"frozen data\n"


@pytest.mark.parametrize("invalid", ["hash", "protected", "syntax", "duplicate", "too_many", "size"])
def test_one_invalid_patch_leaves_entire_workspace_untouched(workspace, invalid):
    proposals = [patch(workspace, "model.py", "VALUE = 2\n"),
                 patch(workspace, "train.py", "print('candidate')\n")]
    if invalid == "hash":
        proposals[1]["base_sha256"] = "0" * 64
    elif invalid == "protected":
        proposals[1] = patch(workspace, "data/split.csv", "changed data\n")
    elif invalid == "syntax":
        proposals[1]["content"] = "def broken(:\n"
    elif invalid == "duplicate":
        proposals[1] = {**proposals[0], "path": "./model.py"}
    elif invalid == "too_many":
        proposals *= 2
    else:
        proposals[1]["content"] = "#" * (96 * 1024 + 1)
    before = snapshot_workspace(workspace)
    with pytest.raises((ValueError, SyntaxError)):
        PatchTransaction(workspace, proposals, PatchPolicy())
    assert snapshot_workspace(workspace) == before


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_candidate_restores_every_file_and_removes_new_outputs(workspace, error):
    before = snapshot_workspace(workspace)
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n")], PatchPolicy())

    def run():
        with transaction:
            (workspace / "train.py").write_bytes(b"runtime change\n")
            (workspace / "data" / "split.csv").unlink()
            (workspace / "candidate_output.txt").write_bytes(b"failed output\n")
            if error:
                raise error("candidate failed")

    if error:
        with pytest.raises(error):
            run()
    else:
        run()
    assert snapshot_workspace(workspace) == before
    assert "candidate_output.txt" in transaction.removed


def test_failed_application_rolls_back_earlier_replacements(workspace, monkeypatch):
    before = snapshot_workspace(workspace)
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n"),
        patch(workspace, "train.py", "print('candidate')\n")], PatchPolicy())
    replace = os.replace
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-file failure")
        return replace(*args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second-file"):
        with transaction:
            pytest.fail("an incomplete candidate must never execute")
    assert snapshot_workspace(workspace) == before


def test_change_after_proposal_is_rejected_without_overwriting_it(workspace):
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n")], PatchPolicy())
    (workspace / "model.py").write_bytes(b"VALUE = 99\n")
    with pytest.raises(ValueError, match="base_sha256"):
        with transaction:
            pytest.fail("stale candidate must never execute")
    assert (workspace / "model.py").read_bytes() == b"VALUE = 99\n"


def test_patch_replacement_does_not_write_through_hardlink(workspace, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"VALUE = 1\n")
    target = workspace / "model.py"
    target.unlink()
    os.link(outside, target)
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n")], PatchPolicy())
    with transaction:
        assert target.read_bytes() == b"VALUE = 2\n"
        assert outside.read_bytes() == b"VALUE = 1\n"
    assert target.read_bytes() == b"VALUE = 1\n"
    assert outside.read_bytes() == b"VALUE = 1\n"


def test_same_size_same_timestamp_patch_cannot_reuse_stale_bytecode(workspace, monkeypatch):
    source = workspace / "model.py"
    timestamp = source.stat().st_mtime
    py_compile.compile(str(source), doraise=True)
    replace = os.replace

    def preserve_timestamp(staged, target):
        replace(staged, target)
        os.utime(target, (timestamp, timestamp))

    monkeypatch.setattr(os, "replace", preserve_timestamp)
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n")], PatchPolicy())
    with transaction:
        result = subprocess.run([sys.executable, "-c", "import model; print(model.VALUE)"],
                                cwd=workspace, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "2"
    result = subprocess.run([sys.executable, "-c", "import model; print(model.VALUE)"],
                            cwd=workspace, capture_output=True, text=True, timeout=10)
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_failed_rollback_preserves_original_exception(workspace, monkeypatch, error):
    transaction = PatchTransaction(workspace, [patch(workspace, "model.py", "VALUE = 2\n")], PatchPolicy())

    def cannot_restore(*args, **kwargs):
        raise PermissionError("snapshot restoration denied")

    monkeypatch.setattr(transaction_module, "restore_snapshot", cannot_restore)
    with pytest.raises(error, match="original failure") as caught:
        with transaction:
            raise error("original failure")
    assert "snapshot restoration denied" in transaction.rollback_error
    assert any("rollback" in note for note in caught.value.__notes__)
