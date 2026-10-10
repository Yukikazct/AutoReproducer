"""Frozen preset inputs recover transport/cache faults without weaker evidence."""
import hashlib
import io
from pathlib import Path
import subprocess
import stat
from types import SimpleNamespace
from unittest.mock import Mock, call
import urllib.error

import pytest

from src.method_adapters import download_dataset
from src.method_profiles import method_profile
from src.preset_downloads import download_bytes
from src.repository_profiles import get_profile
from src import repository_reproduction as reproduction
from tests.test_repository_reproduction import required_source_files, tar_bytes


@pytest.fixture(params=["dlinear", "siren"])
def dataset_case(tmp_path, request):
    root = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if request.param == "dlinear":
        profile = get_profile("dlinear_etth1_reference")
        payload = b"date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
        spec = profile["dataset"]
        spec.update(bytes=len(payload), rows=1, sha256=hashlib.sha256(payload).hexdigest())
        target = workspace / spec["target"]
        cache = root / "dataset_cache" / spec["name"] / spec["sha256"] / "ETTh1.csv"

        def load(offline=False):
            return reproduction.prepare_dataset(root, profile, workspace, offline=offline)
    else:
        spec = method_profile("siren_camera_quick")["dataset"]
        payload = b"Frozen image bytes used only by this transport fixture."
        spec.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        target = workspace / spec["target"]
        cache = root / "dataset_cache" / spec["name"] / spec["sha256"] / spec["target"]

        def load(offline=False):
            return download_dataset(root, spec, workspace, offline=offline)
    cache.parent.mkdir(parents=True)
    return SimpleNamespace(load=load, cache=cache, target=target, payload=payload, spec=spec)


def test_corrupt_dataset_cache_is_replaced_only_by_exact_verified_online_bytes(dataset_case, monkeypatch):
    case = dataset_case
    case.cache.write_bytes(b"corrupt previous cache")
    download = Mock(return_value=io.BytesIO(case.payload))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    result = case.load()
    assert result["verified"]
    assert case.cache.read_bytes() == case.target.read_bytes() == case.payload
    assert not list(case.cache.parent.glob("*.part"))
    download.assert_called_once()
    download.reset_mock()
    assert case.load(offline=True)["verified"]
    download.assert_not_called()


def test_offline_corrupt_dataset_never_downloads_or_changes_cache(dataset_case, monkeypatch):
    case = dataset_case
    case.cache.write_bytes(b"corrupt previous cache")
    download = Mock(side_effect=AssertionError("offline network"))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    with pytest.raises(RuntimeError, match="离线"):
        case.load(offline=True)
    download.assert_not_called()
    assert case.cache.read_bytes() == b"corrupt previous cache"
    assert not case.target.exists()


def test_transient_dataset_download_retries_then_recovers(dataset_case, monkeypatch):
    case = dataset_case
    download = Mock(side_effect=[
        urllib.error.URLError("PRIVATE_PROXY_TOKEN"),
        urllib.error.HTTPError(case.spec["url"], 503, "PRIVATE_REASON", {}, None),
        io.BytesIO(case.payload)])
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    sleep = Mock()
    monkeypatch.setattr("src.preset_downloads.time.sleep", sleep)
    assert case.load()["verified"]
    assert download.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    assert case.cache.read_bytes() == case.payload


def test_dataset_transport_retry_budget_is_bounded_and_error_is_safe(dataset_case, monkeypatch):
    case = dataset_case
    download = Mock(side_effect=urllib.error.URLError("PRIVATE_PROXY_TOKEN"))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    sleep = Mock()
    monkeypatch.setattr("src.preset_downloads.time.sleep", sleep)
    with pytest.raises(RuntimeError, match="已尝试 3 次") as failure:
        case.load()
    assert "PRIVATE" not in str(failure.value)
    assert download.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    assert not case.cache.exists()
    assert not case.target.exists()


def test_download_checksum_failure_preserves_existing_cache_without_retry(dataset_case, monkeypatch):
    case = dataset_case
    case.cache.write_bytes(b"corrupt previous cache")
    # Same-sized foreign content is an integrity failure, never accepted as a
    # fallback image/dataset and never published over the previous cache.
    download = Mock(return_value=io.BytesIO(b"x" * len(case.payload)))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    with pytest.raises((ValueError, RuntimeError), match="校验失败"):
        case.load()
    download.assert_called_once()
    assert case.cache.read_bytes() == b"corrupt previous cache"
    assert not case.target.exists()
    assert not list(case.cache.parent.glob("*.part"))


def test_atomic_dataset_publication_failure_cleans_temporary_and_preserves_cache(dataset_case, monkeypatch):
    case = dataset_case
    case.cache.write_bytes(b"corrupt previous cache")
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", Mock(return_value=io.BytesIO(case.payload)))
    original_replace = Path.replace

    def replace(path, target):
        if path.suffix == ".part":
            raise PermissionError("publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError):
        case.load()
    assert case.cache.read_bytes() == b"corrupt previous cache"
    assert not list(case.cache.parent.glob("*.part"))
    assert not case.target.exists()


@pytest.mark.parametrize("status,headers", [(403, {}), (404, {}), (429, {"Retry-After": "3600"})])
def test_permanent_http_failures_and_long_rate_limit_are_not_retried(monkeypatch, status, headers):
    download = Mock(side_effect=urllib.error.HTTPError("https://example.org/fixed", status, "PRIVATE", headers, None))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    sleep = Mock()
    monkeypatch.setattr("src.preset_downloads.time.sleep", sleep)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        download_bytes("https://example.org/fixed", max_bytes=64, timeout_s=30)
    download.assert_called_once()
    sleep.assert_not_called()


def git_case(tmp_path, monkeypatch, *, corrupt=False, clone_failures=0, clone_error=b"fatal: early EOF"):
    profile = get_profile("dlinear_etth1_reference")
    root = tmp_path / "data"
    revision = profile["repository"]["revision"]
    url = profile["repository"]["url"]
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache = root / "repository_cache" / key / revision
    if corrupt:
        (cache / ".git").mkdir(parents=True)
        (cache / "broken.marker").write_bytes(b"keep old cache for diagnosis")
    files = required_source_files(profile)
    archive = tar_bytes(files)
    clone_paths = []

    def run(argv, **kwargs):
        assert argv[:2] == ["git", "clone"]
        source = Path(argv[-1])
        clone_paths.append(source)
        (source / ".git").mkdir(parents=True)
        if len(clone_paths) <= clone_failures:
            return SimpleNamespace(returncode=128, stderr=clone_error, stdout=b"")
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    def git(repo, *args, **kwargs):
        repo = Path(repo)
        if args == ("remote", "get-url", "origin"):
            return url
        if args in [("rev-parse", "--verify", revision + "^{commit}"), ("rev-parse", "HEAD")]:
            return revision
        if args[0] in {"fetch", "checkout"}:
            return ""
        if args == ("archive", "--format=tar", revision):
            if (repo / "broken.marker").exists():
                raise RuntimeError("fatal: missing tree object")
            return archive
        raise AssertionError(args)

    monkeypatch.setattr(reproduction.subprocess, "run", Mock(side_effect=run))
    monkeypatch.setattr(reproduction, "_git", git)
    sleep = Mock()
    monkeypatch.setattr(reproduction.time, "sleep", sleep)
    return SimpleNamespace(profile=profile, root=root, cache=cache, clone_paths=clone_paths,
                           sleep=sleep, files=files)


def test_corrupt_git_object_cache_is_refetched_at_fixed_sha_and_reusable_offline(tmp_path, monkeypatch):
    case = git_case(tmp_path, monkeypatch, corrupt=True)
    destination = tmp_path / "run" / "repo"
    snapshot = reproduction.export_repository(case.root, case.profile, destination)
    assert snapshot["resolved_sha"] == case.profile["repository"]["revision"]
    assert Path(snapshot["cache_path"]) == case.cache
    assert len(case.clone_paths) == 1
    assert not (case.cache / "broken.marker").exists()
    quarantine = list(case.cache.parent.glob(".invalid-*"))
    assert len(quarantine) == 1
    assert (quarantine[0] / "broken.marker").read_bytes() == b"keep old cache for diagnosis"
    assert not list(case.cache.parent.glob(".fetch-*"))
    for relative, content in case.files.items():
        assert (destination / relative).read_bytes() == content
    reproduction.subprocess.run.reset_mock()
    reproduction.export_repository(case.root, case.profile, tmp_path / "second" / "repo", offline=True)
    reproduction.subprocess.run.assert_not_called()


def test_offline_corrupt_git_objects_do_not_refetch_or_modify_cache(tmp_path, monkeypatch):
    case = git_case(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(RuntimeError, match="离线模式"):
        reproduction.export_repository(case.root, case.profile, tmp_path / "repo", offline=True)
    reproduction.subprocess.run.assert_not_called()
    assert (case.cache / "broken.marker").exists()
    assert not (tmp_path / "repo").exists()


def test_transient_clone_failures_retry_with_new_clean_trees(tmp_path, monkeypatch):
    case = git_case(tmp_path, monkeypatch, clone_failures=2)
    reproduction.export_repository(case.root, case.profile, tmp_path / "repo")
    assert len(case.clone_paths) == 3 and len(set(case.clone_paths)) == 3
    assert case.sleep.call_args_list == [call(1), call(2)]
    assert not any(path.exists() for path in case.clone_paths)
    assert (case.cache / ".git").is_dir()


@pytest.mark.parametrize("failures,error,expected_attempts", [
    (3, b"fatal: early EOF PRIVATE_PROXY_KEY", 3),
    (1, b"fatal: repository not found PRIVATE_PROXY_KEY", 1),
])
def test_git_download_failure_is_bounded_safe_and_leaves_no_partial_cache(tmp_path, monkeypatch, failures, error, expected_attempts):
    case = git_case(tmp_path, monkeypatch, clone_failures=failures, clone_error=error)
    with pytest.raises(RuntimeError, match=f"已尝试 {expected_attempts} 次") as failure:
        reproduction.export_repository(case.root, case.profile, tmp_path / "repo")
    assert "PRIVATE" not in str(failure.value)
    assert len(case.clone_paths) == expected_attempts
    assert not case.cache.exists()
    assert not list(case.cache.parent.glob(".fetch-*"))
    assert not (tmp_path / "repo").exists()


def test_unexpected_fetched_revision_never_enters_frozen_cache(tmp_path, monkeypatch):
    case = git_case(tmp_path, monkeypatch)
    original_git = reproduction._git

    def wrong_head(repo, *args, **kwargs):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(reproduction, "_git", wrong_head)
    with pytest.raises(RuntimeError, match="固定版本"):
        reproduction.export_repository(case.root, case.profile, tmp_path / "repo")
    assert len(case.clone_paths) == 1
    assert not case.cache.exists()
    assert not list(case.cache.parent.glob(".fetch-*"))
    assert not (tmp_path / "repo").exists()


def test_simultaneous_preparation_reuses_already_verified_cache_without_replacing_it(tmp_path, monkeypatch):
    case = git_case(tmp_path, monkeypatch)
    reproduction.export_repository(case.root, case.profile, tmp_path / "repo")
    (case.cache / "untracked.marker").write_bytes(b"existing cache remains untouched")
    source, archive = reproduction._fetch_repository(
        case.cache, case.profile["repository"]["url"], case.profile["repository"]["revision"])
    assert source == case.cache
    assert (case.cache / "untracked.marker").read_bytes() == b"existing cache remains untouched"
    assert not list(case.cache.parent.glob(".invalid-*"))
    assert not list(case.cache.parent.glob(".fetch-*"))


def test_read_only_git_temporary_objects_are_cleaned_inside_checked_directory(tmp_path):
    parent = tmp_path / "repository-cache"
    source = parent / ".fetch-specific-preparation"
    obj = source / ".git" / "objects" / "pack" / "author.idx"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(b"read-only Git index")
    obj.chmod(stat.S_IREAD)
    reproduction._discard_temporary_repository(source, parent)
    assert not source.exists()


def test_git_cleanup_refuses_a_path_outside_the_intended_cache_directory(tmp_path):
    parent = tmp_path / "repository-cache"
    outside = tmp_path / ".fetch-outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_bytes(b"untouched")
    with pytest.raises(ValueError, match="超出固定缓存目录"):
        reproduction._discard_temporary_repository(outside, parent)
    assert marker.read_bytes() == b"untouched"


def test_http_date_retry_after_does_not_wait_for_long_publisher_rate_limit(monkeypatch):
    download = Mock(side_effect=urllib.error.HTTPError(
        "https://example.org/fixed", 429, "PRIVATE", {"Retry-After": "Thu, 01 Jan 1970 01:00:00 GMT"}, None))
    monkeypatch.setattr("src.preset_downloads.urllib.request.urlopen", download)
    monkeypatch.setattr("src.preset_downloads.time.time", lambda: 0)
    sleep = Mock()
    monkeypatch.setattr("src.preset_downloads.time.sleep", sleep)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        download_bytes("https://example.org/fixed", max_bytes=64, timeout_s=30)
    download.assert_called_once()
    sleep.assert_not_called()
