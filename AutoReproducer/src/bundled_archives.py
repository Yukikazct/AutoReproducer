"""Prepare unchanged dataset ZIPs from a pinned author repository."""
import hashlib
from pathlib import Path
import stat
import zipfile

from src.safety.paths import relative_path, workspace_path

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 5000


def inspect_bundled_zip(workspace, name, expected_sha256, target):
    archive = workspace_path(workspace, name, 'bundled ZIP', must_exist=True, forbid_git=True)
    if archive.suffix.lower() != '.zip' or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Bundled ZIP differs from the pinned author data')
    base = relative_path(target, 'ZIP extraction target', forbid_git=True)
    workspace_path(workspace, target, 'ZIP extraction target', forbid_git=True)
    files, seen, total = {}, set(), 0
    with zipfile.ZipFile(archive) as stream:
        if len(stream.infolist()) > MAX_FILES:
            raise ValueError('Bundled ZIP contains too many entries')
        for entry in stream.infolist():
            if '\\' in entry.orig_filename or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError('Bundled ZIP has an ambiguous, linked or encrypted member')
            relative = relative_path(entry.filename, 'ZIP member', forbid_git=True)
            name_in_workspace = (base / relative).as_posix()
            path = workspace_path(workspace, name_in_workspace, 'ZIP member', forbid_git=True)
            if any(part.casefold() == '_autorepro_capture' for part in (*base.parts, *relative.parts)):
                raise ValueError('Bundled ZIP cannot write execution metadata')
            if entry.is_dir():
                if path.exists() and not path.is_dir():
                    raise ValueError('Bundled ZIP directory collides with an author file')
                continue
            key = name_in_workspace.casefold()
            if key in seen or path.exists():
                raise ValueError('Bundled ZIP cannot duplicate or overwrite an author file')
            seen.add(key)
            total += entry.file_size
            if total > MAX_BYTES:
                raise ValueError('Bundled ZIP exceeds the full-data extraction limit')
            digest, size = hashlib.sha256(), 0
            with stream.open(entry) as member:
                while chunk := member.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry.file_size:
                        raise ValueError('Bundled ZIP member size changed')
                    digest.update(chunk)
            if size != entry.file_size:
                raise ValueError('Bundled ZIP member is incomplete')
            files[name_in_workspace] = {'member': entry.filename, 'bytes': size, 'sha256': digest.hexdigest()}
    if not files or not total:
        raise ValueError('Bundled ZIP has no real data bytes')
    for name_in_workspace in files:
        if any(parent.as_posix().casefold() in seen for parent in Path(name_in_workspace).parents if parent != Path('.')):
            raise ValueError('Bundled ZIP file collides with a parent directory')
    return {'path': name, 'archive_sha256': expected_sha256, 'target': target, 'files': files, 'bytes': total}


def extract_bundled_zip(workspace, manifest):
    actual = inspect_bundled_zip(workspace, manifest['path'], manifest['archive_sha256'], manifest['target'])
    if actual != manifest:
        raise ValueError('Bundled ZIP extraction differs from the frozen plan')
    archive = workspace_path(workspace, manifest['path'], 'bundled ZIP', must_exist=True, forbid_git=True)
    with zipfile.ZipFile(archive) as stream:
        for name, record in manifest['files'].items():
            path = workspace_path(workspace, name, 'prepared ZIP data', forbid_git=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with stream.open(record['member']) as source, path.open('xb') as destination:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > record['bytes']:
                        raise ValueError('Bundled ZIP extraction exceeded the frozen size')
                    digest.update(chunk)
                    destination.write(chunk)
            if size != record['bytes'] or digest.hexdigest() != record['sha256']:
                raise ValueError('Prepared ZIP data differs from the author archive')
    return manifest['files']
