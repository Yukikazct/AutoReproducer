"""Directory removal that handles Windows read-only repository objects."""
import os
import shutil
import stat
from pathlib import Path

from src.safety.paths import is_link


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
        shutil.rmtree(path, onerror=_remove_readonly_file)
    else:
        shutil.rmtree(path)
