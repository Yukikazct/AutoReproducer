"""Portable path boundaries for repository files and execution plans."""
import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath


def is_link(path):
    """Include Windows junctions, including on Python 3.11."""
    path = Path(path)
    if path.is_symlink():
        return True
    try:
        return bool(getattr(path.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False


def relative_path(value, kind="path", *, forbid_git=False):
    """Reject both POSIX and Windows escape syntax on every host OS."""
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{kind} must be a relative path inside the workspace")
    windows, posix = PureWindowsPath(raw), PurePosixPath(raw)
    if (windows.drive or windows.root or posix.root or ".." in windows.parts
            or any(":" in part for part in windows.parts)
            or (forbid_git and any(part.casefold() == ".git" for part in windows.parts))):
        raise ValueError(f"{kind} must be a relative path inside the workspace")
    return Path(*windows.parts)


def workspace_path(root, value, kind="path", *, must_exist=False, forbid_git=False):
    relative = relative_path(value, kind, forbid_git=forbid_git)
    root = Path(root).absolute()
    candidate = root / relative
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        if is_link(current):
            raise ValueError(f"{kind} contains a symlink or junction: {value}")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{kind} escapes the workspace: {value}")
    return resolved
