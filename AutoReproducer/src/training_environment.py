"""Own disposable author interpreters without importing application dependencies."""
import json
import os
from pathlib import Path
import stat
import sys
import threading

from src.safety.files import remove_tree
from src.safety.paths import filesystem_path, is_link, workspace_path


JOURNAL = ".autorepro-training.json"
MARKER = ".autorepro-training-owner.json"
_CHILDREN = {"envs", "interpreters", "tools", "scratch", ".python.lock", MARKER}
_local = threading.local()


def _root_lock(root):
    from filelock import FileLock

    locks = getattr(_local, "locks", None)
    if locks is None:
        locks = _local.locks = {}
    key = os.path.normcase(str(root))
    if key not in locks:
        # Keep this inode outside the removable interpreter tree.
        locks[key] = FileLock(str(root.parent / ".autorepro-training-python.lock"))
    return locks[key]


def _checked_root(value, dependency_root):
    from src.project_paths import PROJECT_ROOT, get_data_root

    original = Path(value).absolute()
    if original.name != "training_python":
        raise ValueError("managed author Python must use a training_python directory")
    if any(is_link(parent) for parent in (original, *original.parents)):
        raise ValueError("training Python root contains a symlink or junction")
    root = workspace_path(original.parent, original.name, "training Python root")
    protected = (PROJECT_ROOT, Path(sys.prefix), get_data_root(), Path(dependency_root))
    if any(path.resolve().is_relative_to(root) for path in protected):
        raise ValueError("training Python root contains application or dependency files")
    # A package-tree replacement must never remove the selected interpreter.
    if root.is_relative_to(Path(dependency_root).resolve()):
        raise ValueError("training Python must be separate from the package cache")
    return root


def _roots(dependency_root):
    from src.project_paths import get_data_root

    journal = workspace_path(dependency_root, JOURNAL, "training Python journal")
    roots = [get_data_root() / "training_python"]
    if journal.is_file():
        state = json.loads(journal.read_text("utf-8"))
        if (not isinstance(state, dict) or state.get("version") != 1
                or not isinstance(state.get("roots"), list)
                or any(not isinstance(item, str) or not Path(item).is_absolute()
                       for item in state["roots"])):
            raise ValueError("invalid training Python ownership journal")
        roots.extend(Path(item) for item in state["roots"])
    return sorted({_checked_root(root, dependency_root) for root in roots})


def _owned_bytes(root):
    """Venv executable symlinks are references, never deletion/measurement targets."""
    total, pending = 0, [Path(filesystem_path(root))]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total


def release_training_environments(dependency_root):
    """Caller holds the dependency lease, including during crash recovery."""
    receipt = {"status": "released", "removed": [], "errors": [], "freed_bytes": 0}
    try:
        roots = _roots(dependency_root)
    except (OSError, ValueError, TypeError) as exc:
        receipt.update(status="error", errors=[{"path": JOURNAL, "reason": str(exc)}])
        return receipt
    for root in roots:
        try:
            if not root.exists():
                continue
            from filelock import Timeout
            try:
                lock = _root_lock(root)
                if lock.is_locked:
                    from src.environment_lifecycle import active_environment
                    owner = active_environment(dependency_root)
                    if owner is None or root not in owner.training_locks:
                        raise RuntimeError("training Python is owned by another active run")
                lock.acquire(timeout=0)
            except Timeout as exc:
                raise RuntimeError("training Python is owned by another active run") from exc
            try:
                if not root.is_dir() or any(child.name not in _CHILDREN for child in root.iterdir()):
                    raise ValueError("training Python root contains unrecognized files")
                size = _owned_bytes(root)
                # rmtree unlinks nested symlinks/junctions without following.
                remove_tree(root)
                receipt["removed"].append({"path": str(root), "bytes": size})
                receipt["freed_bytes"] += size
            finally:
                lock.release()
        except (OSError, ValueError, RuntimeError) as exc:
            receipt["errors"].append({"path": str(root), "reason": str(exc)})
    if receipt["errors"]:
        receipt["status"] = "error"
    else:
        try:
            workspace_path(dependency_root, JOURNAL, "training Python journal").unlink(missing_ok=True)
        except OSError as exc:
            receipt.update(status="error", errors=[{"path": JOURNAL, "reason": str(exc)}])
    return receipt


def prepare_training_environment(value):
    """Release the prior interpreter/packages before selecting another version."""
    from src.agents.code_executor import _INSTALLED_DEPS
    from src.environment_lifecycle import (active_environment, release_dependencies,
                                           EnvironmentCleanupError, _write_json)

    lease = active_environment()
    if lease is None:
        return Path(value).resolve()
    root = _checked_root(value, lease.root)
    if root not in lease.training_locks:
        root.parent.mkdir(parents=True, exist_ok=True)
        lock = _root_lock(root)
        try:
            if lock.is_locked:
                raise RuntimeError("training Python is owned by another active run")
            lock.acquire(timeout=0)
        except Exception as exc:
            raise EnvironmentCleanupError("训练解释器正在被其他运行使用", {
                "errors": [{"path": str(root), "reason": str(exc)}]}) from exc
        lease.training_locks[root] = lock
    # Include pre-journal legacy custom roots in this migration/replacement.
    known = _roots(lease.root)
    if root not in known:
        known.append(root)
    journal = workspace_path(lease.root, JOURNAL, "training Python journal")
    _write_json(journal, {"version": 1, "roots": [str(item) for item in known]})
    packages = release_dependencies(lease.root)
    training = release_training_environments(lease.root)
    receipt = {"status": "error" if packages["errors"] or training["errors"] else "released",
               "errors": [*packages["errors"], *training["errors"]],
               "dependencies": packages, "training_python": training}
    lease.replacements.append(receipt)
    if receipt["errors"]:
        raise EnvironmentCleanupError("旧训练环境未能释放，已停止创建新解释器", receipt)
    lease.selected = None
    _INSTALLED_DEPS.clear()
    # Publish ownership before creating the root, venv, uv or a download.
    _write_json(journal, {"version": 1, "roots": [str(root)]})
    root.mkdir(parents=True, exist_ok=True)
    # The journal is authoritative. An interrupted advisory marker write must
    # leave only a known filename so the next owner can still recover the tree.
    workspace_path(root, MARKER, "training Python owner").write_text('{"version": 1}', encoding="utf-8")
    return root


def training_process_environment(root):
    """Keep uv, pip, ensurepip and their downloaded files inside the owned root."""
    scratch = workspace_path(root, "scratch", "training Python scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("LLM_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "AUTOREPRO_GITHUB_TOKEN"):
        environment.pop(name, None)
    environment.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
                       PIP_USER="0", PIP_NO_CACHE_DIR="1", PIP_CACHE_DIR=str(scratch / "pip-cache"),
                       UV_CACHE_DIR=str(scratch / "uv-cache"))
    return environment
