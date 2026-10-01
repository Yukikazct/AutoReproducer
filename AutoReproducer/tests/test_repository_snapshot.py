"""Official ZIP fallback must pin its version and refuse altered cache files."""
import io
import json
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from src.repository_snapshot import MARKER, download_snapshot, verified_snapshot_commit
from src.resource_manager import ResourceManager

URL = "https://github.com/cure-lab/LTSF-Linear"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PREFIX = "cure-lab-LTSF-Linear-" + COMMIT[:7]


def payload(name="run_longExp.py", symlink=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        entry = zipfile.ZipInfo(PREFIX + "/" + name)
        if symlink:
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, "# official API source fixture\n")
    return stream.getvalue()


def network_boundary(monkeypatch, archive):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            return subprocess.CompletedProcess(cmd, 1, "", "HTTP2 connection reset")
        assert cmd[0] == "curl"
        target = Path(cmd[cmd.index("--output") + 1])
        if "/commits/" in cmd[-1]:
            target.write_text(json.dumps({"sha": COMMIT}))
        else:
            assert cmd[-1].endswith("/zipball/" + COMMIT)
            target.write_bytes(archive)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("src.repository_snapshot.subprocess.run", run)
    return calls


def test_official_snapshot_download_and_integrity_cache(tmp_path, monkeypatch):
    calls = network_boundary(monkeypatch, payload())
    root = tmp_path / "snapshot"
    root.mkdir()
    metadata = download_snapshot(root, URL, COMMIT)
    assert metadata["commit"] == COMMIT and metadata["transport"] == "github_api_zip"
    assert len(metadata["archive_sha256"]) == 64
    assert verified_snapshot_commit(root, URL, COMMIT) == COMMIT
    assert verified_snapshot_commit(root, URL, "f" * 40) == ""
    assert verified_snapshot_commit(root, "https://github.com/other/repo") == ""
    assert len(calls) == 2
    source = root / "run_longExp.py"
    original = source.read_bytes()
    source.write_text("changed source")
    assert verified_snapshot_commit(root, URL) == ""
    source.write_bytes(original)
    assert verified_snapshot_commit(root, URL) == COMMIT
    (root / "untracked.py").write_text("cannot join official source unnoticed")
    assert verified_snapshot_commit(root, URL) == ""


@pytest.mark.parametrize("name,symlink", [
    ("../../outside", False), ("/absolute", False), ("C:\\outside", False),
    ("link.py", True), (MARKER, False),
])
def test_snapshot_rejects_unsafe_archive(tmp_path, monkeypatch, name, symlink):
    network_boundary(monkeypatch, payload(name, symlink))
    root = tmp_path / "snapshot"
    root.mkdir()
    with pytest.raises(ValueError):
        download_snapshot(root, URL)
    assert not (tmp_path / "outside").exists()
    assert verified_snapshot_commit(root, URL) == ""


def test_git_failure_uses_official_zip_then_reuses_verified_cache(tmp_path, monkeypatch):
    calls = network_boundary(monkeypatch, payload())
    manager = ResourceManager(str(tmp_path / "data"))
    repo = tmp_path / "cache"
    repo.mkdir()
    (repo / "user_file").write_text("preserved")
    result = manager.fetch_code("p", URL, target=str(repo), revision=COMMIT, verify_repository=True)
    assert result["state"] == "cloned" and result["transport"] == "github_api_zip"
    assert result["commit"] == COMMIT
    assert (Path(result["previous_cache"]) / "user_file").read_text() == "preserved"
    assert len(calls) == 4  # two Git attempts + commit + ZIP
    repeat = manager.fetch_code("p", URL, target=str(repo), revision=COMMIT, verify_repository=True)
    assert repeat["state"] == "cached" and repeat["commit"] == COMMIT
    assert repeat["archive_sha256"] == result["archive_sha256"]
    assert len(calls) == 4
    assert not list(tmp_path.glob(".cache-api-*"))


def test_unavailable_zip_does_not_replace_invalid_cache(tmp_path, monkeypatch):
    network_boundary(monkeypatch, b"not a zip archive")
    repo = tmp_path / "cache"
    repo.mkdir()
    (repo / "user_file").write_text("preserved")
    result = ResourceManager(str(tmp_path / "data")).fetch_code(
        "p", URL, target=str(repo), verify_repository=True)
    assert result["state"] == "clone-failed" and not result["path"]
    assert (repo / "user_file").read_text() == "preserved"
    assert not list(tmp_path.glob(".cache-api-*"))
