"""Recovery bookmarks select only marked, unlinked task data directories."""
import json
import os
from pathlib import Path

import pytest

from frontend.live_progress import (isolated_progress_root, progress_run_id,
                                    recent_progress_runs, resolve_progress_run)


SCOPE = "a1" * 16


def recovery_progress(data_root, identifier=SCOPE, name="progress_12345.jsonl"):
    root = data_root / "recovery" / identifier
    path = root / "runtime" / name
    path.parent.mkdir(parents=True)
    path.write_text('{"type":"state","state":"BUILD_ENV"}\n', encoding="utf-8")
    (root / ".restart.json").write_text(json.dumps({
        "version": 1, "original_cleanup": "unconfirmed",
        "progress_path": str(data_root / "runtime" / "progress_1.jsonl"),
    }), encoding="utf-8")
    return root, path


def test_ordinary_and_isolated_bookmarks_round_trip_without_changing_files(tmp_path):
    old = tmp_path / "runtime" / "progress_42.jsonl"
    old.parent.mkdir()
    old.write_text("ordinary progress", encoding="utf-8")
    root, path = recovery_progress(tmp_path)
    before = {item: item.read_bytes() for item in (old, path, root / ".restart.json")}
    assert progress_run_id(tmp_path, old) == old.name
    assert resolve_progress_run(tmp_path, old.name) == str(old)
    assert isolated_progress_root(tmp_path, old) is None
    identifier = SCOPE + "__" + path.name
    assert progress_run_id(tmp_path, path) == identifier
    assert resolve_progress_run(tmp_path, identifier) == str(path)
    assert isolated_progress_root(tmp_path, path) == root
    assert {item: item.read_bytes() for item in before} == before


@pytest.mark.parametrize("marker", [
    None, b"\x00" * 327, b"{truncated", b"[]",
    b'{"version":true,"original_cleanup":"unconfirmed"}',
    b'{"version":2,"original_cleanup":"unconfirmed"}',
    b'{"version":1,"original_cleanup":"confirmed"}',
    b'{"version":1}', b" " * 16385,
])
def test_recovery_bookmarks_require_an_intact_unconfirmed_marker(tmp_path, marker):
    root, path = recovery_progress(tmp_path)
    record = root / ".restart.json"
    if marker is None:
        record.unlink()
    else:
        record.write_bytes(marker)
    before = record.read_bytes() if record.exists() else None
    assert progress_run_id(tmp_path, path) is None
    assert resolve_progress_run(tmp_path, SCOPE + "__" + path.name) is None
    assert isolated_progress_root(tmp_path, path) is None
    assert (record.read_bytes() if record.exists() else None) == before


@pytest.mark.parametrize("identifier", [
    "../progress_12345.jsonl", "..\\progress_12345.jsonl", "C:/secret.jsonl",
    SCOPE + "/runtime/progress_12345.jsonl",
    SCOPE + "__../progress_12345.jsonl",
    SCOPE + "__progress_12345.jsonl/extra",
    SCOPE.upper() + "__progress_12345.jsonl",
    SCOPE[:-1] + "__progress_12345.jsonl",
    "f" * 32 + "__progress_12345.jsonl",
    SCOPE + "__progress_99999.jsonl", SCOPE + "__secret.jsonl", None,
])
def test_query_tokens_cannot_choose_an_arbitrary_or_missing_file(tmp_path, identifier):
    recovery_progress(tmp_path)
    assert resolve_progress_run(tmp_path, identifier) is None


def test_path_helpers_reject_files_outside_the_exact_scope_runtime(tmp_path):
    root, path = recovery_progress(tmp_path)
    wrong = [root / path.name, root / "runtime" / "nested" / path.name,
             tmp_path / "elsewhere" / path.name]
    for other in wrong:
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("private file", encoding="utf-8")
        assert progress_run_id(tmp_path, other) is None
        assert isolated_progress_root(tmp_path, other) is None
    assert progress_run_id(tmp_path, path.parent / "progress_99999.jsonl") is None
    assert isolated_progress_root(tmp_path, path.parent / "progress_99999.jsonl") is None


@pytest.mark.parametrize("linked_part", ["data", "recovery", "scope", "runtime", "marker", "progress"])
def test_bookmarks_never_follow_symlinks(tmp_path, require_symlinks, linked_part):
    base = tmp_path / "data"
    root, path = recovery_progress(base)
    original = {"data": base, "recovery": base / "recovery", "scope": root,
                "runtime": path.parent, "marker": root / ".restart.json", "progress": path}[linked_part]
    target = tmp_path / "linked-target"
    original.rename(target)
    original.symlink_to(target, target_is_directory=target.is_dir())
    assert resolve_progress_run(base, SCOPE + "__" + path.name) is None
    assert progress_run_id(base, path) is None
    assert isolated_progress_root(base, path) is None


def test_recent_progress_combines_verified_scopes_and_keeps_existing_tokens(tmp_path):
    ordinary = tmp_path / "runtime" / "progress_1.jsonl"
    ordinary.parent.mkdir()
    ordinary.write_text("old", encoding="utf-8")
    root, valid = recovery_progress(tmp_path)
    bad_root, bad = recovery_progress(tmp_path, "b2" * 16)
    (bad_root / ".restart.json").write_bytes(b"\x00" * 327)
    os.utime(ordinary, ns=(1000, 1000))
    os.utime(valid, ns=(2000, 2000))
    os.utime(bad, ns=(3000, 3000))
    assert recent_progress_runs(tmp_path) == [SCOPE + "__" + valid.name, ordinary.name]
    assert root.is_dir()


def test_bad_recovery_link_does_not_hide_safe_ordinary_progress(tmp_path, require_symlinks):
    ordinary = tmp_path / "runtime" / "progress_1.jsonl"
    ordinary.parent.mkdir()
    ordinary.write_text("ordinary", encoding="utf-8")
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "recovery").symlink_to(target, target_is_directory=True)
    assert recent_progress_runs(tmp_path) == [ordinary.name]
