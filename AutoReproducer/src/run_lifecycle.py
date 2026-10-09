"""Cancellation and explicit recovery of abandoned method runs."""
import _thread
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from src.method_adapters import read_json, write_json
from src.sandbox_timeout import ParentWatch


@contextmanager
def cancellation_signals():
    """CLI signals/launcher exit unwind the same stack as Ctrl-C; threads keep their owner."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = {}
    stopped = threading.Event()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    def watch_launcher(watch):
        while not stopped.wait(.2):
            if not watch.alive():
                _thread.interrupt_main()
                return
    with ParentWatch(os.getppid()) as parent:
        try:
            for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    old[sig] = signal.signal(sig, interrupt)
            thread = threading.Thread(target=watch_launcher, args=(parent,), daemon=True)
            thread.start()
            yield
        finally:
            stopped.set()
            if "thread" in locals():
                thread.join(timeout=1)
            for sig, handler in old.items():
                signal.signal(sig, handler)


def recover_run(directory):
    """Mark a dead owner's incomplete records, without resuming or inventing results.

    An OS lock distinguishes a live owner from a stale 'running' file. Legacy runs
    without that lock contract are deliberately not rewritten automatically.
    """
    directory = Path(directory).resolve(strict=True)
    status_path = directory / "run_status.json"
    if not status_path.is_file():
        return {"status": "unknown_owner", "reason": "旧运行没有持有者锁；保留原记录，需人工核对"}
    try:
        with FileLock(str(directory / ".run.lock"), timeout=0):
            status = read_json(status_path)
            if status["status"] != "running":
                return status
            final_path = directory / "result.json"
            if final_path.is_file():
                final = read_json(final_path)
                if isinstance(final.get("data"), dict) and "error" in final:
                    terminal = ("interrupted" if final["data"].get("interrupted") else
                                "failed" if final["error"] else "completed")
                    status.update(status=terminal, recovered=True, recovered_at=time.time(),
                                  reason="已有最终结果，恢复运行状态；最终结果未改写")
                    write_json(directory / "recovery.json", status)
                    write_json(status_path, status)
                    return status
            # Preserve the pre-recovery documents so recovery is never mistaken for
            # the original run's final output or API response.
            backup = directory / "recovery_originals"
            backup.mkdir(exist_ok=True)
            recovered = []
            paths = [directory / "optimization.json", *directory.glob("trials/*/trial.json"),
                     *directory.glob("trials/*/holdout.json")]
            for path in paths:
                if not path.is_file():
                    continue
                record = read_json(path)
                if record.get("status") != "running":
                    continue
                relative = path.relative_to(directory)
                original = backup / relative
                original.parent.mkdir(parents=True, exist_ok=True)
                if not original.exists():
                    original.write_bytes(path.read_bytes())
                record.update(status="interrupted", reason="持有者已退出，恢复时识别到不完整运行", recovered=True)
                if path.name == "optimization.json":
                    record["optimized"] = False
                write_json(path, record)
                recovered.append(relative.as_posix())
            status.update(status="interrupted", recovered=True, recovered_at=time.time(),
                          recovered_records=recovered,
                          reason="持有者已退出；保留证据，不自动续跑、不据部分结果认定成功")
            write_json(directory / "recovery.json", status)
            write_json(status_path, status)
            return status
    except Timeout:
        return {"status": "running", "reason": "运行仍持有锁，未修改任何记录"}
