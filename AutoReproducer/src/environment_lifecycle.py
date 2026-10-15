"""Own one disposable experiment environment through report persistence.

Application Python and its production dependencies are outside this scope.
Imports of the execution graph are deferred so runtime recovery stays usable.
"""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import sys
import tempfile
import threading


_local = threading.local()


class EnvironmentCleanupError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message + ": " + str(receipt["errors"]))
        self.receipt = receipt


def _key(root):
    return os.path.normcase(str(Path(root).resolve()))


def _write_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def active_environment(root=None):
    runs = getattr(_local, "runs", {})
    if root is None:
        return next(reversed(runs.values()), None)
    return runs.get(_key(root))


def release_dependencies(root, *, keep=()):
    """Caller holds the dependency lock; never follow links or remove the root."""
    from src.safety.files import directory_bytes, remove_tree
    from src.safety.paths import workspace_path

    root = Path(root).absolute()
    workspace_path(root, ".autorepro-cache.lock", "dependency cleanup root")
    resolved = root.resolve()
    # A misconfigured cache must not turn completion into deletion of the app.
    from src.project_paths import PROJECT_ROOT, get_data_root
    if any(path.resolve().is_relative_to(resolved)
           for path in (PROJECT_ROOT, Path(sys.prefix), get_data_root())):
        raise ValueError("dependency cleanup root contains application or data files")
    keep = {Path(path).resolve() for path in keep}
    removed, errors, remaining = [], [], []

    def visit(parent):
        for path in sorted(parent.iterdir()):
            if path.name in {".autorepro-cache.lock", ".autorepro-docker.json",
                             ".autorepro-training.json"} and parent == root:
                continue  # Keep the lock inode stable for queued processes.
            name = path.relative_to(root).as_posix()
            try:
                checked = workspace_path(root, name, "dependency release", must_exist=True)
                if checked in keep:
                    continue
                if any(item.is_relative_to(checked) for item in keep):
                    visit(checked)
                    continue
                if checked.is_dir():
                    size = directory_bytes(checked)
                    remove_tree(checked)
                else:
                    size = checked.stat().st_size
                    checked.unlink()
                removed.append({"path": name, "bytes": size})
            except (OSError, ValueError) as exc:
                errors.append({"path": name, "reason": str(exc)})
                remaining.append(name)

    if root.exists():
        visit(root)
    return {"status": "error" if errors else "released", "removed": removed,
            "freed_bytes": sum(item["bytes"] for item in removed),
            "remaining": remaining, "errors": errors}


def select_dependency_environment(root, directory):
    """Drop the previous requirements set before installing a different one."""
    from src.safety.paths import workspace_path
    root, directory = Path(root).absolute(), Path(directory).absolute()
    directory = workspace_path(root, directory.relative_to(root), "selected dependency environment")
    lease = active_environment(root)
    if lease is None or lease.selected == Path(directory):
        return False
    receipt = release_dependencies(root, keep=[directory])
    lease.replacements.append(receipt)
    if receipt["errors"]:
        raise EnvironmentCleanupError("旧运行依赖未能释放，已停止创建新环境", receipt)
    lease.selected = Path(directory)
    return True


def _release_environment(root):
    from src.runtime_preparation import release_runtime_downloads
    from src.docker_environment import release_owned_docker
    from src.training_environment import release_training_environments

    receipt = release_dependencies(root)
    training = release_training_environments(root)
    receipt["training_python"] = training
    receipt["freed_bytes"] += training["freed_bytes"]
    if training["errors"]:
        receipt["status"] = "error"
        receipt["errors"].extend(training["errors"])
        receipt["remaining"].append(".autorepro-training.json")
    docker = release_owned_docker(root)
    receipt["docker"] = docker
    if docker["errors"]:
        receipt["status"] = "error"
        receipt["errors"].extend(docker["errors"])
        receipt["remaining"].append(".autorepro-docker.json")
    try:
        downloads = release_runtime_downloads()
    except Exception as exc:
        downloads = {"status": "error", "errors": [{"reason": str(exc)}]}
    receipt["runtime_downloads"] = downloads
    if downloads["status"] == "error":
        receipt["status"] = "error"
        receipt["errors"].extend({"path": "runtime downloads", **item} for item in downloads["errors"])
    return receipt


def release_abandoned_environment(root=None):
    """An exited worker's supervisor may reclaim its cache without waiting.

    A different run can win the lock after the worker exits; in that case its
    environment belongs to the new owner and must be left alone.
    """
    from src.dependency_cache import cache_guard, DependencyCacheBusy
    from src.project_paths import get_deps_root

    root = Path(root if root is not None else get_deps_root())
    try:
        with cache_guard(root, cleanup=True):
            return _release_environment(root)
    except DependencyCacheBusy:
        return {"status": "in_use", "removed": [], "freed_bytes": 0,
                "remaining": [], "errors": []}


class _EnvironmentRun:
    def __init__(self, root):
        self.root, self.result, self.selected = root, None, None
        self.run_dir = None
        self.replacements = []
        self.training_locks = {}

    def record(self, receipt):
        result = self.result if isinstance(self.result, dict) else None
        data = result.setdefault("data", {}) if result is not None else {}
        if result is not None:
            data["environment_cleanup"] = receipt
            if receipt["status"] == "error":
                reason = "运行环境自动释放失败；详情见 environment_cleanup"
                result["state"] = "ERROR"
                result["error"] = "; ".join(filter(None, [result.get("error"), reason]))
        run_dir = data.get("run_dir") or self.run_dir
        if run_dir:
            from src.safety.paths import workspace_path
            directory = Path(run_dir)
            _write_json(workspace_path(directory, "environment_cleanup.json", "cleanup receipt"), receipt)
            path = workspace_path(directory, "result.json", "run result")
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved["data"]["environment_cleanup"] = receipt
                if result is not None:
                    saved["error"] = result.get("error")
                    if receipt["status"] == "error":
                        saved["state"] = "ERROR"
                elif receipt["status"] == "error":
                    saved["state"] = "ERROR"
                _write_json(path, saved)
            status_path = workspace_path(directory, "run_status.json", "run status")
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status["environment_cleanup"] = receipt["status"]
                if receipt["status"] == "error":
                    status["status"] = "failed"
                _write_json(status_path, status)
        report = result.get("report_path") if result is not None else None
        if report:
            archive = Path(report).with_suffix(".result.json")
            if archive.is_file():
                from src.report_revisions import sanitize_report_result
                _write_json(archive, sanitize_report_result(self.result))


def bind_environment_run(directory):
    """Keep a cleanup receipt even when the pipeline raises before its result."""
    from src.agents.code_executor import DEPS_CACHE_ROOT
    lease = active_environment(DEPS_CACHE_ROOT)
    if lease is not None:
        lease.run_dir = Path(directory)


def record_environment_cleanup(result, receipt):
    """Persist a supervisor's final receipt using the same result contract."""
    record = _EnvironmentRun(None)
    record.result = result
    record.record(receipt)


@contextmanager
def managed_environment(root=None):
    from src.agents.code_executor import DEPS_CACHE_ROOT, _INSTALLED_DEPS
    from src.dependency_cache import cache_guard

    root = Path(root if root is not None else DEPS_CACHE_ROOT)
    existing = active_environment(root)
    if existing is not None:
        yield existing
        return
    # The same reentrant root lock encloses install, execution, report and release.
    # Waiting has no training deadline; Ctrl-C still unwinds filelock acquisition.
    with cache_guard(root, timeout=-1):
        lease = _EnvironmentRun(root)
        initial = _release_environment(root)
        if initial["errors"]:
            raise EnvironmentCleanupError("上次运行环境仍有无法释放的文件，已停止创建新环境", initial)
        runs = getattr(_local, "runs", None)
        if runs is None:
            runs = _local.runs = {}
        runs[_key(root)] = lease
        failure = None
        try:
            yield lease
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                receipt = _release_environment(root)
                receipt["before_run"] = initial
                receipt["replacements"] = lease.replacements
                lease.record(receipt)
                if receipt["errors"] and failure is not None:
                    failure.add_note("运行环境自动释放失败：" + str(receipt["errors"]))
            except Exception as exc:
                if failure is None:
                    raise
                failure.add_note("运行环境自动释放失败：" + str(exc))
            finally:
                _INSTALLED_DEPS.clear()
                try:
                    for lock in lease.training_locks.values():
                        lock.release()
                finally:
                    runs.pop(_key(root), None)


def release_run_environment(function):
    """Also cover direct Python/CLI calls that do not use the web backend."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from src.agents.code_executor import DEPS_CACHE_ROOT
        if active_environment(DEPS_CACHE_ROOT) is not None:
            return function(*args, **kwargs)
        with managed_environment() as lease:
            lease.result = function(*args, **kwargs)
        return lease.result
    return wrapped
