"""Directory removal that handles Windows read-only repository objects."""
import os
import shutil
import stat
from pathlib import Path

from src.safety.paths import filesystem_path, is_link


def directory_bytes(directory):
    """Measure an owned tree without following symlinks or Windows junctions."""
    total, pending = 0, [Path(filesystem_path(directory))]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                    raise ValueError("dependency environment contains a symlink or junction")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


def _remove_readonly_file(function, path, exc_info):
    """Windows rmtree callback; leave locks, links and other failures intact."""
    error = exc_info[1]
    if getattr(error, "winerror", None) != 5:
        raise error
    item = Path(path)
    metadata = item.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or is_link(item)
            or metadata.st_mode & stat.S_IWRITE):
        raise error
    item.chmod(metadata.st_mode | stat.S_IWRITE)
    function(path)


def remove_tree(path) -> None:
    """Delete a tree, retrying Windows read-only regular files once.

    Python 3.11's onerror API is used for compatibility. Directory permissions,
    sharing violations and non-read-only access failures continue to propagate.
    """
    if os.name == "nt":
        shutil.rmtree(filesystem_path(path), onerror=_remove_readonly_file)
    else:
        shutil.rmtree(path)
