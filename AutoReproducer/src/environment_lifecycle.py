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
import uuid


_local = threading.local()


class EnvironmentCleanupError(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message + ": " + str(receipt["errors"]))
        self.receipt = receipt
        if receipt.get('process_cleanup_confirmed') is False:
            self.process_cleanup_confirmed = False
            self.cleanup_receipt = receipt


def _key(root):
    return os.path.normcase(str(Path(root).resolve()))


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _write_text(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def active_environment(root=None):
    runs = getattr(_local, "runs", {})
    if root is None:
        return next(reversed(runs.values()), None)
    return runs.get(_key(root))


def retain_unconfirmed_processes(root, receipt):
    """Keep the current lease until an owner can prove its processes stopped."""
    receipt = {**receipt, 'status': 'error', 'process_cleanup_confirmed': False,
               'removed': [], 'freed_bytes': 0, 'remaining': [str(Path(root))]}
    receipt['errors'] = receipt.get('errors') or ['process cleanup remains unconfirmed']
    lease = active_environment(root)
    if lease is not None:
        lease.cleanup_receipt = receipt
    return receipt


def _preserve_task_gate(root, receipt):
    """Reuse the web owner gate for direct/CLI runs as well as background work."""
    from frontend.task_control import current_task, _root_paths, _paths, _write_record
    control = current_task()
    if control is not None and _key(control.deps_root) == _key(root):
        control.cleanup_receipt = receipt
        return
    directory = _root_paths(root)[0]
    directory.mkdir(parents=True, exist_ok=True)
    # A separate supervising process can hold the task owner lock. The current
    # dependency lease already serializes this publication; retain a unique
    # record in the same recovery protocol without acquiring that owner's lock.
    token = uuid.uuid4().hex
    progress = directory / ('environment-cleanup-' + token + '.progress')
    record = {'version': 1, 'pid': os.getpid(), 'token': token,
              'state': 'cleanup_unconfirmed', 'deps_root': str(Path(root).resolve()),
              'platform': os.name, 'progress_path': _key(progress)}
    _write_record(_paths(progress)[0], record)
    _write_record(_root_paths(root, progress)[2], record)


def _check_unconfirmed_task_gate(root):
    from frontend.task_control import _root_paths, _read_record
    directory = _root_paths(root)[0]
    if not directory.exists():
        return
    for path in directory.glob('*.json'):
        try:
            record = _read_record(path)
        except (OSError, ValueError) as exc:
            receipt = retain_unconfirmed_processes(root, {'errors': [str(exc)]})
            raise EnvironmentCleanupError('任务清理凭据无法确认，保留原环境', receipt) from exc
        if _key(record['deps_root']) == _key(root) and record['state'] == 'cleanup_unconfirmed':
            receipt = retain_unconfirmed_processes(root, {'errors': [
                'process cleanup remains unconfirmed: ' + str(path)]})
            raise EnvironmentCleanupError('任务进程尚未确认结束，保留原环境', receipt)


def release_dependencies(root, *, keep=()):
    """Caller holds the dependency lock; never follow links or remove the root."""
    from src.safety.files import directory_bytes, remove_tree
    from src.safety.paths import workspace_path

    root = Path(root).absolute()
    lease = active_environment(root)
    if lease is not None and lease.cleanup_receipt is not None:
        raise EnvironmentCleanupError('任务进程尚未确认结束，禁止释放或替换依赖', lease.cleanup_receipt)
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
        self.cleanup_receipt = None

    @staticmethod
    def _synchronize_recovery_cleanup(data, receipt, run_dir):
        """Late process uncertainty stops the saved ledger, without a new session.

        File/cache release failures are publication failures. Only explicit
        process cleanup evidence changes the task's recovery/execution boundary.
        Lifecycle receipts add wrappers beyond ordinary execution records.
        """
        from src.execution_safety import cleanup_is_unconfirmed
        from src.task_recovery import summarize_recovery_state
        pending, seen, unconfirmed = [receipt], set(), False
        while pending:
            record = pending.pop()
            if not isinstance(record, (dict, list, tuple)) or id(record) in seen:
                continue
            seen.add(id(record))
            if cleanup_is_unconfirmed(record):
                unconfirmed = True
                break
            pending.extend(record.values() if isinstance(record, dict) else record)
        recovery = data.get('intelligent_recovery')
        if unconfirmed:
            data['process_cleanup_confirmed'] = False
        if isinstance(recovery, dict):
            if unconfirmed:
                recovery.update(status='stopped', requires_human=True, cleanup_confirmed=False,
                    stop_reason='environment_process_cleanup_unconfirmed')
                recovery['cleanup_evidence'] = {'source': 'environment_cleanup.json',
                                                'process_cleanup_confirmed': False}
                for action in recovery.get('actions') or []:
                    if isinstance(action, dict) and action.get('status') == 'executing':
                        action.update(status='stopped', verification={'success': False,
                            'reason': 'Owned process cleanup was not confirmed at task finalization'})
            if run_dir:
                from src.safety.paths import workspace_path
                recovery['artifact_path'] = str(workspace_path(run_dir, 'intelligent_recovery.json',
                                                               'recovery artifact'))
            data['recovery_summary'] = summarize_recovery_state(recovery)
        return unconfirmed

    def record(self, receipt):
        result = self.result if isinstance(self.result, dict) else None
        data = result.setdefault("data", {}) if result is not None else {}
        run_dir = data.get("run_dir") or self.run_dir
        # Cancellation can save a failure result before raising to the lease.
        if result is None and run_dir:
            from src.safety.paths import workspace_path
            path = workspace_path(run_dir, "result.json", "run result")
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                data = result.setdefault("data", {})
        if result is not None:
            data["environment_cleanup"] = receipt
            process_unconfirmed = self._synchronize_recovery_cleanup(data, receipt, run_dir)
            if receipt["status"] == "error" or process_unconfirmed:
                reason = ("所属进程清理未确认；详情见 environment_cleanup" if process_unconfirmed else
                          "运行环境自动释放失败；详情见 environment_cleanup")
                result["state"] = "ERROR"
                if reason not in (result.get("error") or ""):
                    result["error"] = "; ".join(filter(None, [result.get("error"), reason]))
            self.result = result
            data["run_finalization"] = {"state": result.get("state"), "error": result.get("error")}
            from src.run_finalization import finalize_report_text
            if data.get("report"):
                data["report"] = finalize_report_text(data["report"], data)
                data["report_length"] = len(data["report"])
        try:
            self._persist(receipt, result, data, run_dir)
        except Exception as exc:
            if result is not None:
                reason = "运行收尾报告与归档保存失败: " + str(exc)
                result.update(state="ERROR", error="; ".join(filter(None, [result.get("error"), reason])))
                data["report_publication"] = {"status": "error", "reason": reason}
                data["run_finalization"] = {"state": "ERROR", "error": result["error"]}
                if data.get("report"):
                    data["report"] = finalize_report_text(data["report"], data)
                    data["report_length"] = len(data["report"])
                # Keep readable JSON copies honest even if a Markdown path is locked.
                try:
                    self._persist(receipt, result, data, run_dir, write_report=False)
                except Exception:
                    pass
            raise

    def _persist(self, receipt, result, data, run_dir, *, write_report=True):
        from src.safety.paths import workspace_path
        from src.report_revisions import sanitize_report_result

        report = (result.get("report_path") or data.get("report_path")) if result is not None else None
        if write_report and data.get("report"):
            paths = {Path(report)} if report else set()
            if run_dir:
                paths.add(workspace_path(run_dir, "report.md", "run report"))
            for path in paths:
                if path.is_file():
                    _write_text(path, data["report"])
        if run_dir:
            directory = Path(run_dir)
            recovery = data.get('intelligent_recovery')
            if isinstance(recovery, dict):
                _write_json(workspace_path(directory, 'intelligent_recovery.json', 'recovery artifact'),
                            sanitize_report_result({'data': {'intelligent_recovery': recovery}})
                                ['data']['intelligent_recovery'])
            _write_json(workspace_path(directory, "environment_cleanup.json", "cleanup receipt"), receipt)
            path = workspace_path(directory, "result.json", "run result")
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved["data"]["environment_cleanup"] = receipt
                if result is not None:
                    saved["data"].update(data)
                    saved["error"] = result.get("error")
                    if result.get("state") is not None:
                        saved["state"] = result["state"]
                    for key in ("session_id", "audit_logs", "audit_stats", "report_path"):
                        if key in result:
                            saved[key] = result[key]
                elif receipt["status"] == "error":
                    saved["state"] = "ERROR"
                _write_json(path, sanitize_report_result(saved))
            status_path = workspace_path(directory, "run_status.json", "run status")
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status["environment_cleanup"] = receipt["status"]
                if receipt["status"] == "error" or ((result or {}).get("state") == "ERROR"
                                                     and status.get("status") == "completed"):
                    status["status"] = "failed"
                _write_json(status_path, status)
        if report:
            archive = Path(report).with_suffix(".result.json")
            if archive.is_file():
                _write_json(archive, sanitize_report_result(result))


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
        _check_unconfirmed_task_gate(root)
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
                if getattr(failure, 'process_cleanup_confirmed', True) is False:
                    retain_unconfirmed_processes(root, getattr(failure, 'cleanup_receipt', {}) or {
                        'errors': [str(failure)]})
                result_data = (lease.result or {}).get('data', {}) if isinstance(lease.result, dict) else {}
                result_receipt = result_data.get('environment_cleanup') or {}
                if result_receipt.get('process_cleanup_confirmed') is False:
                    retain_unconfirmed_processes(root, result_receipt)
                if lease.cleanup_receipt is not None:
                    receipt = lease.cleanup_receipt
                    _preserve_task_gate(root, receipt)
                else:
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
