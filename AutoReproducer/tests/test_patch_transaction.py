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


def test_exact_edits_preserve_other_bytes_and_rollback(workspace):
    proposal = patch(workspace, 'model.py', '')
    proposal.pop('content')
    proposal['edits'] = [{'before': 'VALUE = 1', 'after': 'VALUE = 2', 'count': 1}]
    with PatchTransaction(workspace, [proposal], PatchPolicy()):
        assert (workspace / 'model.py').read_bytes() == b'VALUE = 2\n'
    assert (workspace / 'model.py').read_bytes() == b'VALUE = 1\n'


@pytest.mark.parametrize('long_path', [False, True])
def test_failed_generated_module_does_not_leave_importable_bytecode(workspace, long_path):
    from pathlib import Path
    from src.safety.paths import filesystem_path
    directory = 'observer' if not long_path else 'observer_' + 'x' * max(20, 232 - len(str(workspace)))
    relative = directory + '/helper.py'
    transaction = PatchTransaction(workspace, [patch(workspace, 'model.py', 'VALUE = 2\n')], PatchPolicy())
    with transaction:
        generated = Path(filesystem_path(workspace / relative))
        generated.parent.mkdir()
        generated.write_text('VALUE = 7\n', encoding='utf-8')
        cached = py_compile.compile(str(generated), doraise=True)
        assert os.path.isfile(cached)
    assert not generated.exists()
    assert not os.path.exists(cached)
    assert relative in transaction.removed


@pytest.mark.parametrize('edits', [
    [{'before': '', 'after': 'anything'}],
    [{'before': 'VALUE = 1', 'after': 'VALUE = 2', 'count': 2}],
    [{'before': 'invented', 'after': 'replacement'}],
])
def test_invalid_exact_edits_never_mutate_source(workspace, edits):
    proposal = patch(workspace, 'model.py', '')
    proposal.pop('content')
    proposal['edits'] = edits
    with pytest.raises(ValueError):
        PatchTransaction(workspace, [proposal], PatchPolicy())
    assert (workspace / 'model.py').read_bytes() == b'VALUE = 1\n'


def test_rejected_edit_feedback_identifies_actual_fragment_and_remaining_counts(workspace):
    content = 'a = old_api()\nb = old_api()\nc = old_api()\n'
    (workspace / 'model.py').write_text(content, encoding='utf-8')
    proposal = patch(workspace, 'model.py', '')
    proposal.pop('content')
    proposal['edits'] = [
        {'before': 'a = old_api()', 'after': 'a = new_api()'},
        {'before': 'old_api()', 'after': 'new_api()'},
        {'before': 'old_api()', 'after': 'new_api()'},
    ]
    with pytest.raises(ValueError) as error:
        PatchTransaction(workspace, [proposal], PatchPolicy())
    feedback = str(error.value)
    assert "model.py, edit 2: requested 1, actual 2" in feedback
    assert "before='old_api()'" in feedback
    assert 'edit 3: requested 1, actual 2' in feedback
    assert 'do not duplicate or overlap' in feedback
    assert (workspace / 'model.py').read_text('utf-8') == content


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


@pytest.mark.parametrize('error_type', [None, RuntimeError, KeyboardInterrupt])
def test_deferred_rollback_preserves_uncommitted_tree_and_original_exception(workspace, monkeypatch, error_type):
    transaction = PatchTransaction(workspace, [patch(workspace, 'model.py', 'VALUE = 2\n')], PatchPolicy())

    def cannot_restore(*args, **kwargs):
        pytest.fail('unconfirmed process cleanup must not restore or delete files')

    monkeypatch.setattr(transaction_module, 'restore_snapshot', cannot_restore)
    error = error_type('original interruption') if error_type else None

    def run():
        with transaction:
            (workspace / 'candidate_output.txt').write_bytes(b'potentially live output')
            transaction.defer_rollback('process cleanup unconfirmed')
            if error is not None:
                raise error

    if error is not None:
        with pytest.raises(error_type) as caught:
            run()
        assert caught.value is error
    else:
        run()
    assert transaction.rollback_deferred and not transaction.committed
    assert transaction.rollback_deferred_reason == 'process cleanup unconfirmed'
    assert transaction.rollback_error is None
    assert (workspace / 'model.py').read_bytes() == b'VALUE = 2\n'
    assert (workspace / 'candidate_output.txt').read_bytes() == b'potentially live output'
    with pytest.raises(RuntimeError, match='cannot be committed'):
        transaction.commit()
