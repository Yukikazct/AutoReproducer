"""Read-only Windows files may be removed, while locks and links are preserved."""
import os
import stat
from pathlib import Path

import pytest

from src.safety.files import _remove_readonly_file, remove_tree


def _windows_error(code):
    error = PermissionError("Windows file operation failed")
    error.winerror = code
    return (PermissionError, error, None)


def test_readonly_callback_retries_regular_file(tmp_path):
    path = tmp_path / "git-object"
    path.write_bytes(b"object")
    path.chmod(stat.S_IREAD)

    def delete_unlocked(value):
        assert Path(value).stat().st_mode & stat.S_IWRITE
        os.unlink(value)

    _remove_readonly_file(delete_unlocked, path, _windows_error(5))
    assert not path.exists()


@pytest.mark.parametrize("error_code", [32, 33, 13])
def test_readonly_callback_does_not_bypass_other_failures(tmp_path, error_code):
    path = tmp_path / "locked"
    path.write_text("keep")
    path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            _remove_readonly_file(os.unlink, path, _windows_error(error_code))
        assert path.read_text() == "keep"
        assert not path.stat().st_mode & stat.S_IWRITE
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_readonly_callback_does_not_change_writable_files(tmp_path):
    path = tmp_path / "writable"
    path.write_text("keep")
    with pytest.raises(PermissionError):
        _remove_readonly_file(os.unlink, path, _windows_error(5))
    assert path.read_text() == "keep"


def test_readonly_callback_does_not_change_reparse_points(tmp_path, monkeypatch):
    path = tmp_path / "reparse-point"
    path.write_text("keep")
    path.chmod(stat.S_IREAD)
    monkeypatch.setattr("src.safety.files.is_link", lambda path: True)
    try:
        with pytest.raises(PermissionError):
            _remove_readonly_file(os.unlink, path, _windows_error(5))
        assert path.read_text() == "keep"
        assert not path.stat().st_mode & stat.S_IWRITE
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_readonly_callback_propagates_retry_failure(tmp_path):
    path = tmp_path / "still-locked"
    path.write_text("keep")
    path.chmod(stat.S_IREAD)

    def locked(value):
        raise PermissionError("still locked")

    with pytest.raises(PermissionError, match="still locked"):
        _remove_readonly_file(locked, path, _windows_error(5))
    assert path.read_text() == "keep"


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows read-only deletion semantics")
def test_windows_removes_readonly_git_objects(tmp_path):
    root = tmp_path / "repository"
    objects = root / ".git" / "objects" / "pack"
    objects.mkdir(parents=True)
    path = objects / "pack-example.pack"
    path.write_bytes(b"git object data")
    path.chmod(stat.S_IREAD)
    remove_tree(root)
    assert not root.exists()
