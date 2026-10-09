"""Run a fixed repository command plan without rewriting its source files.

This is a local, trusted-repository runner. Docker plans are rejected explicitly;
the existing single-script Docker executor is not a repository sandbox.
"""
import codecs
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.agents.code_executor import CodeExecutorAgent, DEPS_CACHE_ROOT, reqs_digest
from src.execution_artifacts import collect_images
from src.execution_plan import declared_paths, plan_steps
from src.safety.paths import is_link, relative_path, workspace_path
from src.dependency_cache import DependencyCacheBusy
from src.process_lifecycle import CREATE_SUSPENDED, ProcessJob, termination_signals, validate_windows_venv
from src.runtime_platform import runtime_fingerprint

# Evidence can be a model checkpoint; the report image budget is far too small.
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class RepositoryRunner:
    def __init__(self, logger=None, executor=None):
        # Dependency and environment helpers do not call the LLM.
        self.executor = executor or CodeExecutorAgent(None, logger=logger)
        self.logger = logger or self.executor.logger

    @staticmethod
    def cached_environment(env_config):
        runtime = runtime_fingerprint()
        return DEPS_CACHE_ROOT / "repository" / runtime / reqs_digest(env_config["requirements_txt"])

    @staticmethod
    def _contained(root, relative, kind):
        return workspace_path(root, relative, kind, must_exist=True)

    @staticmethod
    def _check_tree(root):
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if is_link(path):
                    raise ValueError(f"repository symlinks are unsupported: {path.relative_to(root)}")

    def _plan(self, root, steps):
        if not isinstance(steps, list) or not steps:
            raise ValueError("repository plan requires at least one step")
        planned, identifiers = [], set()
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("each repository step must be an object")
            identifier = step.get("id", "")
            if (not isinstance(identifier, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
                    or identifier in identifiers):
                raise ValueError("step ids must be unique safe file names")
            identifiers.add(identifier)
            argv = step.get("argv")
            if (not isinstance(argv, list) or not argv
                    or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)):
                raise ValueError(f"invalid argv for step {identifier}")
            argv = list(argv)
            python = argv[0] in {"python", "python3", sys.executable}
            if python:
                argv[0] = sys.executable
            cwd = self._contained(root, step.get("cwd", "."), "cwd")
            if not cwd.is_dir():
                raise ValueError(f"cwd is not a directory: {cwd}")
            if python:
                # -c/-m are supported for import checks. Ordinary script paths
                # must resolve inside the same repository, including nested cwd.
                for arg in argv[1:]:
                    if arg in {"-c", "-m"}:
                        break
                    if not arg.startswith("-"):
                        script = self._contained(root, cwd.relative_to(root) / relative_path(arg, "script"), "script")
                        if not script.is_file():
                            raise ValueError(f"script is not a file: {arg}")
                        break
            else:
                executable = relative_path(argv[0], "executable")
                if "/" in argv[0] or "\\" in argv[0]:
                    self._contained(root, cwd.relative_to(root) / executable, "executable")
            timeout = step.get("timeout_s", 600)
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError(f"invalid timeout_s for step {identifier}")
            extra_env = step.get("env", {})
            if (not isinstance(extra_env, dict)
                    or any(not isinstance(k, str) or not k or "=" in k or "\x00" in k
                           or not isinstance(v, str) or "\x00" in v
                           for k, v in extra_env.items())):
                raise ValueError(f"invalid environment for step {identifier}")
            dependencies = step.get("depends_on", [])
            if (not isinstance(dependencies, list)
                    or any(not isinstance(dep, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dep)
                           for dep in dependencies) or len(set(dependencies)) != len(dependencies)
                    or identifier in dependencies or not set(dependencies) <= identifiers):
                raise ValueError(f"invalid depends_on for step {identifier}")
            planned.append({"id": identifier, "argv": argv, "cwd": str(cwd),
                            "timeout_s": float(timeout), "env": dict(extra_env),
                            "kind": step.get("kind", "run"), "depends_on": list(dependencies),
                            "requires": declared_paths(step.get("requires"), identifier,
                                                       "requires", allow_glob=False),
                            "artifacts": declared_paths(step.get("artifacts"), identifier,
                                                        "artifacts", allow_glob=True),
                            "required": step.get("required", True) is not False})
        return planned

    def _capture(self, root, run_dir, step, emit):
        """Copy declared evidence into the run directory, hashing what was found.

        Collection failures never change a step's verdict: one mistyped glob must not
        turn an experiment that genuinely ran into a reported failure.
        """
        captured = []
        for declared in step["artifacts"]:
            entry = {"path": declared["path"], "matched": False}
            try:
                pattern = relative_path(declared["path"], "artifact")
                matches = sorted(path for path in root.glob(str(pattern)) if path.is_file())
                # A matched file is only evidence if every directory above it is real.
                matches = [path for path in matches
                           if not any(is_link(root / part) for part in
                                      path.parent.relative_to(root).parts)]
                if not matches:
                    entry["reason"] = "not found"
                elif sum(path.stat().st_size for path in matches) > MAX_ARTIFACT_BYTES:
                    entry["reason"] = "too large to copy"
                else:
                    entry["files"] = []
                    for path in matches:
                        relative = path.relative_to(root).as_posix()
                        raw = path.read_bytes()
                        destination = (run_dir / "artifacts" / step["id"] / relative)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(raw)
                        entry["files"].append({"path": relative, "bytes": len(raw),
                                               "sha256": hashlib.sha256(raw).hexdigest(),
                                               "captured_to": str(destination)})
                    entry.update(matched=True,
                                 bytes=sum(item["bytes"] for item in entry["files"]))
                    emit({"type": "repository_artifact", "step_id": step["id"],
                          "path": declared["path"], "files": entry["files"]})
            except (OSError, ValueError) as exc:
                entry["reason"] = str(exc)
            captured.append(entry)
        return captured

    def _unmet_requirement(self, root, step):
        """The first prerequisite path a step needs that no earlier step produced."""
        for declared in step["requires"]:
            try:
                path = workspace_path(root, declared["path"], "requirement")
            except ValueError as exc:
                return f"{declared['path']} ({exc})"
            if not path.is_file() or not path.stat().st_size:
                return declared["path"]
        return None

    @staticmethod
    def _kill_group(process):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS can reject a group whose leader has already exited.
                # Timeout termination happens while the supervisor is alive.
                if process.poll() is None:
                    process.kill()
                    return "process-group cleanup was denied; only the direct process was terminated"
        elif process.poll() is None:
            # Windows does not expose killpg. taskkill /T terminates the
            # supervisor's complete descendant tree, while it is still alive.
            try:
                killed = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, text=True, timeout=5, shell=False)
                if killed.returncode != 0 and process.poll() is None:
                    process.kill()
                    return "process-tree cleanup failed: " + (killed.stderr or killed.stdout).strip()
            except (OSError, subprocess.SubprocessError) as exc:
                if process.poll() is None:
                    process.kill()
                return f"process-tree cleanup failed: {exc}"
        return None

    def _execute(self, root, step, run_dir, emit):
        started = time.monotonic()
        stdout_path = run_dir / f"{step['id']}.stdout.log"
        stderr_path = run_dir / f"{step['id']}.stderr.log"
        record = {k: step[k] for k in ("id", "argv", "cwd", "timeout_s", "kind", "required")}
        identity = {"execution_id": run_dir.name,
                    "step_index": step["step_index"], "step_count": step["step_count"]}
        record.update(identity)
        record.update(stage="full", executed=False, stdout_path=str(stdout_path),
                      stderr_path=str(stderr_path))
        process, timed_out, group_killed = None, False, False
        interrupted, job = None, None
        exit_code, diagnostic, cleanup_errors = -2, "", []
        # Plot helpers live beside the logs, not inside the author repository.
        environment = self.executor._exec_env(str(run_dir))
        environment.update(step["env"])
        # Entry points under examples/ or scripts/ still need the repository's
        # top-level packages, rather than an unrelated installed namesake.
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(root), environment.get("PYTHONPATH", "")) if part)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["AUTOREPRO_PARENT_PID"] = str(os.getpid())
        supervisor = Path(__file__).with_name("sandbox_timeout.py")
        command = [sys.executable, "-S", str(supervisor), "execution",
                   str(step["timeout_s"] + 1.0), *step["argv"]]
        gate = run_dir / f"{step['id']}.start"
        if os.name == "nt":
            command = command[:5] + ["--start-gate", str(gate)] + command[5:]
        try:
            emit({"type": "repository_step", "step_id": step["id"],
                  "status": "running", **record})
            self._check_tree(root)
            # Recheck cwd/script after earlier steps may have created files. Only the
            # on-disk fields are revalidated: depends_on names ids from the full list,
            # and handed one step alone those ids are legitimately absent.
            recheck = {key: value for key, value in step.items() if key != "depends_on"}
            self._plan(root, [{**recheck, "cwd": str(Path(step["cwd"]).relative_to(root))}])
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                job = ProcessJob()
                process = subprocess.Popen(command, cwd=step["cwd"], env=environment,
                                           stdout=out, stderr=err, shell=False,
                                           start_new_session=(os.name == "posix"),
                                           # Suspend first so the job object claims the
                                           # supervisor before it can spawn descendants;
                                           # NEW_PROCESS_GROUP isolates console Ctrl+C so
                                           # only the driver cancels first, while the
                                           # supervisor stays alive for tree cleanup.
                                           creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED) if os.name == "nt" else 0)
                job.assign(process)
                job.resume(process)
                if os.name == "nt":
                    gate.touch()
                record["executed"] = True
                with stdout_path.open("rb") as out_tail, stderr_path.open("rb") as err_tail:
                    tails = {"stdout": out_tail, "stderr": err_tail}
                    decoders = {key: codecs.getincrementaldecoder("utf-8")("replace")
                                for key in tails}

                    def output(final=False):
                        for stream, reader in tails.items():
                            chunk = reader.read() if final else reader.read(65536)
                            text = decoders[stream].decode(chunk, final=final)
                            if text:
                                emit({"type": "repository_output", "step_id": step["id"],
                                      "stream": stream, "text": text, **identity})

                    # The host owns the process group and acts before the
                    # standalone supervisor's fallback deadline, while the
                    # group leader is still alive (required on macOS).
                    deadline = time.monotonic() + step["timeout_s"]
                    while process.poll() is None:
                        output()
                        if time.monotonic() >= deadline:
                            timed_out = True
                            job.close()
                            cleanup = self._kill_group(process)
                            if cleanup:
                                cleanup_errors.append(cleanup)
                            group_killed = True
                            break
                        time.sleep(0.1)
                    process.wait(timeout=5)
                    # The supervisor kills the immediate child on timeout;
                    # killing the process group also removes its descendants.
                    if not group_killed:
                        cleanup = self._kill_group(process)
                        if cleanup:
                            cleanup_errors.append(cleanup)
                    output(final=True)
                    exit_code = process.returncode
        except (KeyboardInterrupt, SystemExit) as exc:
            interrupted = exc
            diagnostic = f"repository step {step['id']} interrupted"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            diagnostic = str(exc)
        finally:
            if process is not None:
                if job is not None:
                    job.close()
                if not group_killed:
                    cleanup = self._kill_group(process)
                    if cleanup:
                        cleanup_errors.append(cleanup)
                process.wait(timeout=5)
            elif job is not None:
                job.close()
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        raw_stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        stderr, phase = CodeExecutorAgent._decode_docker_phase(raw_stderr)
        timed_out = timed_out or bool(phase.get("timeout"))
        if interrupted is not None:
            timed_out, exit_code = False, 130
        elif timed_out:
            exit_code = 124
            diagnostic = f"repository step {step['id']} timed out after {step['timeout_s']:g}s"
        if cleanup_errors:
            diagnostic += ("; " if diagnostic else "") + "; ".join(cleanup_errors)
            if exit_code == 0:
                exit_code = -2
        if diagnostic:
            stderr += ("\n" if stderr and not stderr.endswith("\n") else "") + diagnostic + "\n"
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(diagnostic + "\n")
        if not stdout_path.exists():
            stdout_path.write_text("", encoding="utf-8")
        record.update(success=exit_code == 0, exit_code=exit_code, stdout=stdout,
                      stderr=stderr, timed_out=timed_out, cancelled=interrupted is not None,
                      elapsed_s=round(time.monotonic() - started, 3))
        # Declared evidence is captured whatever the verdict was; a failed step's
        # partial output is often the thing a later diagnosis needs.
        record["artifacts"] = self._capture(root, run_dir, step, emit) if step["artifacts"] else []
        (run_dir / f"{step['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"type": "repository_step", "step_id": step["id"],
              "status": "interrupted" if interrupted is not None else "success" if record["success"] else "error", **record})
        if interrupted is not None:
            interrupted.repository_step_record = record
            raise interrupted
        return record

    @termination_signals()
    def run(self, workspace, steps, env_config, use_docker=False, on_event=None):
        """Execute the plan's argv steps in order; halt at the first required failure.

        Accepts either a validated ExecutionPlan or a bare reviewed step list.
        Each run retains full logs, metadata and declared evidence in a unique
        sibling directory, so a discarded workspace cannot erase what actually ran.
        Event callbacks observe progress; observer failures cannot interrupt a
        running process or overwrite its execution verdict.
        """
        attempts, warnings, skipped = [], [], []
        result = {"mode": "repository", "success": False, "executed": False,
                  "attempts": attempts, "stages": attempts, "steps": attempts,
                  "skipped": skipped, "step_artifacts": [],
                  "artifacts": [], "artifact_warnings": warnings, "llm_calls": 0}

        def emit(event):
            if on_event is not None:
                try:
                    on_event(event)
                except Exception as exc:
                    warnings.append(f"progress callback failed: {exc}")

        def reject(reason, code=-2):
            result.update(not_runnable=True, reason=reason)
            result["final"] = {"stage": "full", "success": False, "executed": False,
                               "exit_code": code, "stdout": "", "stderr": reason,
                               "timed_out": code == 124}
            return result

        if use_docker:
            return reject("repository Docker execution is unsupported; choose local mode", -3)
        try:
            validate_windows_venv()
            supplied = Path(workspace)
            if supplied.is_symlink():
                raise ValueError("workspace must not be a symlink")
            root = supplied.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("workspace must be an existing directory")
            self._check_tree(root)
            planned = self._plan(root, plan_steps(steps))
            records_root = root.parent / ".autorepro_repository_runs"
            if records_root.is_symlink():
                raise ValueError("execution records directory must not be a symlink")
            records_root.mkdir(exist_ok=True)
            run_dir = records_root / uuid.uuid4().hex
            run_dir.mkdir()
            result["run_dir"] = str(run_dir)
            self.executor.env_config = dict(env_config or {})
            # Native wheels cannot be shared across Python ABIs or machines.
            runtime = runtime_fingerprint()
            deps_root = DEPS_CACHE_ROOT / "repository" / runtime
            self.executor.deps_cache_root = deps_root
            self.executor.deps_lock_root = DEPS_CACHE_ROOT
            self.executor._deps_dir = None
            self.executor._heal_dirs.clear()
            prepare_dir = run_dir / "environment_prepare"
            prepare_dir.mkdir()
            with self.executor.dependency_scope(timeout=self.executor.env_config.get("cache_lock_timeout_s")):
                if self.executor.env_config.get("require_prepared"):
                    cached = self.cached_environment(self.executor.env_config)
                    if not (cached / ".ready").is_file():
                        return reject("实验环境尚未准备，请先准备实验环境；本次未安装或训练", -4)
                    self.executor._deps_dir = str(cached)
                    deps_error = None
                else:
                    deps_error = self.executor._ensure_local_deps(str(prepare_dir))
                # The legacy helper's process-cache fast path does not set the
                # dependency directory on a newly constructed executor.
                requirements = self.executor.env_config.get("requirements_txt", "")
                if not requirements:
                    requirements = "\n".join(self.executor.env_config.get("required_packages", []) or [])
                if not deps_error and requirements and not self.executor._deps_dir:
                    cached_deps = deps_root / reqs_digest(requirements)
                    if (cached_deps / ".ready").is_file():
                        self.executor._deps_dir = str(cached_deps)
                result["environment"] = {"python": sys.version, "executable": sys.executable,
                                         "platform": sys.platform,
                                         "dependencies_path": self.executor._deps_dir,
                                         "requirements_txt": self.executor.env_config.get("requirements_txt", ""),
                                         "preparation_path": str(prepare_dir)}
                (run_dir / "environment.json").write_text(
                    json.dumps(result["environment"], ensure_ascii=False, indent=2), encoding="utf-8")
                if deps_error:
                    return reject(deps_error, -4)
                # Linear halt: the first required failure stops everything after it.
                # A dependency graph is unnecessary because reviewed plans schedule each
                # step after its prerequisites; the authoring order is the running order.
                halted = None
                for step_index, step in enumerate(planned):
                    if halted is not None:
                        skipped.append({"id": step["id"], "not_run": "prerequisite_failed",
                                        "blocked_by": halted, "required": step["required"]})
                        emit({"type": "repository_step", "step_id": step["id"], "status": "skipped",
                              "not_run": "prerequisite_failed", "blocked_by": halted})
                        continue
                    # A prerequisite product can be absent even though every earlier step
                    # succeeded. Stopping here keeps a missing checkpoint from surfacing as
                    # a confusing failure deep inside the author's evaluator.
                    unmet = self._unmet_requirement(root, step)
                    if unmet is not None:
                        skipped.append({"id": step["id"], "not_run": "missing_requirement",
                                        "blocked_by": halted, "requirement": unmet,
                                        "required": step["required"]})
                        emit({"type": "repository_step", "step_id": step["id"], "status": "skipped",
                              "not_run": "missing_requirement", "requirement": unmet})
                        halted = step["id"]
                        continue
                    # Identity is 1-based: the position a viewer sees, not the list offset.
                    step = {**step, "step_index": step_index + 1, "step_count": len(planned)}
                    deadline = self.executor.env_config.get("deadline_monotonic")
                    if deadline is not None:
                        remaining = deadline - time.monotonic() - 2
                        if remaining <= 0:
                            return reject("实验总时间预算耗尽", 124)
                        step = {**step, "timeout_s": min(step["timeout_s"], remaining)}
                    try:
                        record = self._execute(root, step, run_dir, emit)
                    except KeyboardInterrupt as exc:
                        record = getattr(exc, "step_record", None)
                        if record is not None:
                            attempts.append(record)
                        result.update(cancelled=True, success=False,
                                      executed=any(item["executed"] for item in attempts),
                                      final=record or {"success": False, "executed": False,
                                                       "exit_code": 130, "cancelled": True,
                                                       "stderr": "实验已中断", "stdout": ""})
                        for later in planned[step_index+1:]:
                            skipped.append({"id": later["id"], "not_run": "interrupted",
                                            "blocked_by": step["id"], "required": later["required"]})
                        from src.method_adapters import write_json
                        write_json(run_dir / "execution.json", result)
                        exc.execution = result
                        raise
                    attempts.append(record)
                    result["executed"] = result["executed"] or record["executed"]
                    if not record["success"] and step["required"]:
                        halted = step["id"]
                # Every required step must have run and succeeded. A blocked step is a
                # failed verdict, never a "skipped/optional" one; an optional step may
                # fail on its own without dragging the experiment down.
                result["success"] = (not skipped
                                     and len(attempts) == len(planned)
                                     and all(s["success"] or not s["required"]
                                             for s in attempts))
                result["final"] = dict(attempts[-1]) if attempts else {
                    "stage": "full", "success": False, "executed": False, "exit_code": -2,
                    "stdout": "", "stderr": "所有步骤因前置条件未满足而未执行", "artifacts": []}
                result["effective_env_config"] = self.executor.env_config
                try:
                    collected = collect_images(str(root), "full", getattr(self.logger, "session_id", ""))
                    result["artifacts"] = collected["artifacts"]
                    warnings.extend(collected.get("artifact_warnings", []))
                except (ImportError, OSError, ValueError) as exc:
                    warnings.append(f"image collection failed: {exc}")
                result["final"]["artifacts"] = result["artifacts"]
                return result
        except (KeyboardInterrupt, SystemExit) as exc:
            record = getattr(exc, "repository_step_record", None)
            if record is not None:
                attempts.append(record)
            result.update(success=False, cancelled=True, status="interrupted",
                          executed=any(item.get("executed") for item in attempts))
            result["final"] = dict(record or {"stage": "full", "success": False, "executed": False,
                                             "exit_code": 130, "cancelled": True, "timed_out": False,
                                             "stdout": "", "stderr": "repository execution interrupted"})
            if "run_dir" in result:
                (Path(result["run_dir"]) / "execution.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            exc.execution = result
            raise
        except DependencyCacheBusy as exc:
            return reject(str(exc), -4)
        except (OSError, ValueError, TypeError) as exc:
            return reject(str(exc))
