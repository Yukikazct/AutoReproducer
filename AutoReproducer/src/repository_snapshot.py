"""Verified GitHub source snapshots when Git's HTTPS transport is unavailable."""
import hashlib
import json
import re
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath
from urllib.parse import quote

MARKER = ".autorepro-repo-snapshot.json"
SOURCE_MARKER = ".autorepro-repo-source.json"
MAX_BYTES = 100 * 1024 * 1024


def normalize(url):
    return str(url).rstrip("/").removesuffix(".git")


def verified_snapshot_commit(root, url, revision=""):
    """A snapshot is reusable only while every downloaded file is unchanged."""
    try:
        marker = root / MARKER
        if marker.is_symlink():
            return ""
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        commit = metadata["commit"]
        if normalize(metadata["repo_url"]) != normalize(url) \
                or not re.fullmatch(r"[0-9a-f]{40}", commit) \
                or not re.fullmatch(r"[0-9a-f]{64}", metadata["archive_sha256"]) \
                or (revision and revision not in (commit, metadata.get("requested_revision"))):
            return ""
        files = metadata["files"]
        if not isinstance(files, dict) or not files:
            return ""
        actual = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                return ""
            if path.is_file() and path.name not in (MARKER, SOURCE_MARKER):
                actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return commit if actual == files else ""
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def download_snapshot(root, url, revision=""):
    """Resolve one official commit, download its ZIP and validate extraction."""
    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+)", normalize(url))
    if not match:
        raise ValueError("源码 ZIP 回退仅支持官方 GitHub HTTPS 地址")
    api = "https://api.github.com/repos/" + "/".join(match.groups())
    def download(source, target, budget=90):
        proc = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--connect-timeout", "10", "--max-time", str(budget), "--max-filesize", str(MAX_BYTES),
             "--output", str(target), source], capture_output=True, text=True, timeout=budget + 10)
        if proc.returncode:
            raise ValueError((proc.stderr or "官方源码下载失败")[-300:])
    with tempfile.TemporaryDirectory(prefix="autorepro_source_") as temporary:
        metadata_file = Path(temporary) / "commit.json"
        download(api + "/commits/" + quote(revision or "HEAD", safe=""), metadata_file)
        commit = json.loads(metadata_file.read_text(encoding="utf-8"))["sha"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("GitHub 未返回有效 commit")
        if re.fullmatch(r"[0-9a-f]{40}", revision) and revision != commit:
            raise ValueError("GitHub 返回的版本与请求版本不符")
        archive = Path(temporary) / "source.zip"
        source_url = api + "/zipball/" + commit
        download(source_url, archive, budget=240)
        files, prefix, total = {}, None, 0
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                name = entry.filename
                parts = name.split("/")
                if (name.startswith("/") or "\\" in name or PureWindowsPath(name).drive
                        or any(not p for p in parts[:-1])
                        or any(p in (".", "..") for p in parts)
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("官方源码压缩包含不安全路径")
                if prefix is None:
                    prefix = parts[0]
                if parts[0] != prefix or not prefix.endswith("-" + commit[:7]):
                    raise ValueError("源码压缩包的版本目录不符")
                if entry.is_dir():
                    continue
                relative = "/".join(parts[1:])
                if not relative or relative in files or relative in (MARKER, SOURCE_MARKER):
                    raise ValueError("源码压缩包含重复或保留路径")
                total += entry.file_size
                if total > MAX_BYTES or len(files) >= 10000:
                    raise ValueError("源码压缩包超过 CPU 冒烟体积预算")
                payload = bundle.read(entry)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                files[relative] = hashlib.sha256(payload).hexdigest()
        if not files:
            raise ValueError("官方源码压缩包为空")
        metadata = {"repo_url": normalize(url), "commit": commit, "requested_revision": revision,
                    "transport": "github_api_zip", "download_url": source_url,
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": files}
        (root / MARKER).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if verified_snapshot_commit(root, url, revision) != commit:
            raise ValueError("源码快照校验失败")
        return {key: value for key, value in metadata.items() if key != "files"}
