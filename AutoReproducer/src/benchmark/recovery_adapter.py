"""Executable, trusted compatibility-recovery adapter for the A/B/C harness.

The driver is application code, never a case-selected plugin or shell command.
Each trial imports the selected frozen product in a fresh owned process. This
adapter covers registered author recovery and result investigations; it is not a replacement for
the complete PDF discovery/planning or paper-acceptance pipeline.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import secrets
import sys
import time


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8"))


_CATEGORY_FILES = {'runtime_environment': 'recovery_environment.py',
                   'result_evidence': 'recovery_result_evidence.py'}


def _category_driver(category):
    """Load operator-owned driver bytes without importing another product's src.

    A frozen baseline predates these harness helpers. Their code is part of the
    registered driver, while every production import inside them still resolves
    from the explicitly selected checkout.
    """
    name = _CATEGORY_FILES.get(category)
    return runpy.run_path(str(Path(__file__).with_name(name))) if name else {}


def _path(root, name, *, exists=True):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in {"", ".."} for part in name.split("/"))):
        raise ValueError("Expected a safe repository-relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute():
        raise ValueError("Absolute repository path is unsupported")
    root = Path(root).resolve(strict=True)
    target = root.joinpath(*relative.parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Links are unsupported in trial inputs")
    target = target.resolve(strict=exists)
    if not target.is_relative_to(root):
        raise ValueError("Trial path escaped its root")
    return target


def _environment():
    """Pass system runtime settings, never host provider/package credentials."""
    allowed = {"systemroot", "windir", "comspec", "path", "pathext", "temp", "tmp",
               "home", "userprofile", "localappdata", "appdata", "programfiles",
               "programfiles(x86)", "programdata", "lang", "lc_all", "number_of_processors",
               "processor_architecture", "ssl_cert_file", "ssl_cert_dir"}
    result = {key: value for key, value in os.environ.items() if key.casefold() in allowed}
    result.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                  PYTHONUNBUFFERED="1", PIP_CONFIG_FILE=os.devnull)
    return result


def _python_entrypoint(value):
    """Validate a trusted executable without dereferencing its venv symlink.

    POSIX virtual environments commonly link bin/python to the base runtime.
    Launching that resolved target loses pyvenv.cfg and the product's installed
    dependencies, even when the caller supplied the correct venv executable.
    """
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.is_file():
        raise FileNotFoundError("The configured Python executable is not a file")
    # Check broken links and missing ancestors, but preserve the launch path.
    path.resolve(strict=True)
    return str(path)


def _public_evidence(case, workspace):
    """Narrow, preregistered supported envelope; model text cannot add commands."""
    evidence = deepcopy(case["initial_evidence"])
    if evidence.get("adapter_version") != 1:
        raise ValueError("Executable cases require initial_evidence.adapter_version=1")
    plan = evidence.get("plan")
    if (not isinstance(plan, dict) or plan.get("repository") != {
            "url": case["repository_url"], "revision": case["repository_revision"]}):
        raise ValueError("Case plan must match the preregistered author repository")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Case needs an original Python command plan")
    for step in steps:
        argv = step.get("argv") if isinstance(step, dict) else None
        if (not isinstance(argv, list) or len(argv) < 2 or argv[0] not in {"python", "python3"}
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)
                or argv[1].startswith("-") or not argv[1].endswith(".py")
                or step.get("env") or step.get("required") is False):
            raise ValueError("Only fixed required Python script argv is supported; no shell/-c/-m/env")
        cwd = _path(workspace, step.get("cwd", "."))
        script = _path(cwd, argv[1])
        if script.relative_to(Path(workspace).resolve()).as_posix() not in case["source_files"]:
            raise ValueError("Command script must be in the frozen author inventory")
        timeout = step.get("timeout_s")
        if timeout is not None and (type(timeout) not in {int, float}
                                  or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid original command timeout")
    inventory = (evidence.get("packet") or {}).get("file_inventory")
    if (not isinstance(inventory, dict) or set(inventory) != set(case["source_files"])
            or any(not isinstance(record, dict) or record.get("sha256") != case["source_files"][name]
                   for name, record in inventory.items())):
        raise ValueError("Evidence packet must contain exactly the frozen source inventory")
    for name, expected in case["source_files"].items():
        if _sha(_path(workspace, name).read_bytes()) != expected:
            raise ValueError("Author source changed before the trial")
    editable = evidence.get("editable", [])
    if (not isinstance(editable, list) or len(editable) > 32
            or any(name not in case["source_files"] or not name.endswith(".py") for name in editable)):
        raise ValueError("Editable ceiling must consist of frozen Python sources")
    environment = evidence.get("environment", {})
    if not isinstance(environment, dict) or set(environment) - {"requirements_txt", "required_packages"}:
        raise ValueError("Case may declare dependencies, not executor/environment policy")
    # No indexes, URLs, local installs, options or unlocked dependency resolution.
    from packaging.requirements import Requirement
    requirements = environment.get("requirements_txt", "")
    packages = environment.get("required_packages", [])
    if not isinstance(requirements, str) or not isinstance(packages, list):
        raise ValueError("Invalid dependency declaration")
    lines = [line.strip() for line in requirements.splitlines() if line.strip()] + packages
    for line in lines:
        if not isinstance(line, str):
            raise ValueError("Invalid dependency declaration")
        requirement = Requirement(line)
        if (requirement.url or requirement.marker or len(requirement.specifier) != 1
                or any(spec.operator != "==" or "*" in spec.version for spec in requirement.specifier)):
            raise ValueError("Adapter dependencies must use exact public version pins")
    category = case.get('category')
    if category == 'runtime_environment':
        _category_driver(category)['validate_environment_evidence'](case, workspace, evidence)
        return evidence
    if category == 'result_evidence':
        return _category_driver(category)['validate_result_evidence'](case, workspace)
    failure = evidence.get("initial_failure")
    error_types = {"ModuleNotFoundError"} if case.get("category") == "local_import" else {"AttributeError", "TypeError"}
    if (not isinstance(failure, dict) or failure.get("error_type") not in error_types
            or not isinstance(failure.get("diagnostic_contains"), str) or not failure["diagnostic_contains"]):
        raise ValueError("Preregister an observed local import failure signature" if case.get("category") == "local_import"
                         else "Preregister an observed API failure signature")
    if case.get("category") == "local_import" and editable:
        raise ValueError("Local import recovery cannot authorize source patches")
    return evidence


def _import_descriptor(repair):
    return {key: repair.get(key) for key in ("action", "module", "root", "source_path",
                                            "source_sha256", "importing_sources")}


def _import_runtime_contract(environment):
    # These are observations written by the normal installer/cache, rather
    # than settings that authorize a different experiment or executable.
    telemetry = {"repository_pythonpath", "dependency_install_attempts", "dependency_health_attempts",
                 "dependency_cache_repairs", "dependency_cache_maintenance", "dependency_temporary_files",
                 "dependency_reuse"}
    return {key: deepcopy(value) for key, value in environment.items() if key not in telemetry}


def _validated_import_paths(receipt, case, workspace):
    """Re-derive each path from raw failure + original frozen sources, privately."""
    roots = receipt.get("repository_pythonpath") or []
    proofs = receipt.get("import_recovery_proofs") or []
    if case["category"] != "local_import":
        if roots or proofs:
            raise ValueError("Unregistered import-root recovery")
        return []
    if (not isinstance(roots, list) or not isinstance(proofs, list) or len(proofs) > 8
            or len(roots) != len(proofs) or receipt.get("import_environment_preserved") is not True):
        raise ValueError("Local import recovery is missing its frozen environment/proofs")
    from src.author_self_repair import repair_local_import
    evidence = case["initial_evidence"]
    expected = []
    for proof in proofs:
        if (not isinstance(proof, dict) or not isinstance(proof.get("failure"), dict)
                or not isinstance(proof.get("repair"), dict)):
            raise ValueError("Import proof must retain a real original failure")
        derived = repair_local_import(proof["failure"], workspace, evidence["packet"],
                                      evidence["plan"]["requirements"])
        if (not derived or _import_descriptor(derived) != _import_descriptor(proof.get("repair") or {})
                or derived["root"] in expected):
            raise ValueError("Recorded import root is not justified by its original traceback and source")
        expected.append(derived["root"])
    if roots != expected:
        raise ValueError("Final Python path does not match independently justified import roots")
    paths = [_path(workspace, name) for name in roots]
    if any(not path.is_dir() for path in paths):
        raise ValueError("A restored import root is not an original repository directory")
    return [str(path) for path in paths]


def _run_local_import_recovery(runner, execution, steps, plan, packet, workspace, environment,
                               run_dir, data, verify_inputs, analyze_failure, on_event):
    """Use the selected product's existing loop, restricted to local path repair.

    The first result is the actual author run already observed by the adapter.
    Subsequent runs retain the exact plan/runtime/dependencies and add only a
    unique root re-derived by that product's trusted repair_local_import. The
    generic loop may nominate other repairs, but this adapter never executes
    dependency additions, interpreter changes or model-selected paths.
    """
    from src.author_runtime_dependencies import run_with_dependency_repair
    from src.author_self_repair import failure_diagnostic, repair_local_import
    from src.repository_repairer import repair_is_stopped
    effective = deepcopy(execution.get("effective_env_config") or environment)
    if effective.get("repository_pythonpath"):
        raise ValueError("Initial import case cannot contain pre-restored roots")
    baseline = deepcopy(effective)
    frozen_runtime = _import_runtime_contract(baseline)
    frozen_steps = deepcopy(steps)
    proofs, roots = [], []
    data["_import_recovery_proofs"] = proofs
    data["_import_environment_preserved"] = True
    data["_local_import_last_execution"] = execution

    class CheckedRunner:
        def __init__(self):
            self.seed = True
            self.last = execution

        def run(self, selected_workspace, selected_steps, selected_env, **kwargs):
            verify_inputs()
            if Path(selected_workspace).resolve() != Path(workspace).resolve() or _digest(selected_steps) != _digest(frozen_steps):
                raise ValueError("Local import recovery changed the original command plan")
            if _import_runtime_contract(selected_env) != frozen_runtime:
                raise ValueError("Local import recovery cannot change the frozen runtime environment or install dependencies")
            if self.seed:
                self.seed = False
                if selected_env.get("repository_pythonpath"):
                    raise ValueError("Initial import path was not frozen")
                return deepcopy(execution)
            if repair_is_stopped(self.last) or RecoveryAdapterCleanup.unconfirmed(self.last):
                failure = RuntimeError("Original execution stopped; no further import recovery")
                if RecoveryAdapterCleanup.unconfirmed(self.last):
                    failure.process_cleanup_confirmed = False
                raise failure
            local = repair_local_import(self.last, workspace, packet, plan["requirements"])
            if not local or selected_env.get("repository_pythonpath") != [*roots, local["root"]]:
                raise ValueError("Only a uniquely justified original local import root may be restored")
            roots.append(local["root"])
            proofs.append({"failure": {"final": {"stderr": failure_diagnostic(self.last)}},
                           "repair": _import_descriptor(local)})
            verify_inputs()
            observed = runner.run(selected_workspace, selected_steps, selected_env, **kwargs)
            self.last = observed
            data["_local_import_last_execution"] = observed
            actual_runtime = _import_runtime_contract(observed.get("effective_env_config") or selected_env)
            data["_import_environment_preserved"] = actual_runtime == frozen_runtime
            if not data["_import_environment_preserved"]:
                raise ValueError("Actual import retry changed the frozen runtime environment")
            if RecoveryAdapterCleanup.unconfirmed(observed):
                failure = RuntimeError("Import retry process cleanup was not confirmed")
                failure.process_cleanup_confirmed = False
                raise failure
            return observed

    if repair_is_stopped(execution) or RecoveryAdapterCleanup.unconfirmed(execution):
        return execution
    return run_with_dependency_repair(CheckedRunner(), workspace, frozen_steps, effective, plan=plan,
        packet=packet, run_dir=run_dir, data=data, on_event=on_event, verify_inputs=verify_inputs,
        analyze_failure=analyze_failure)


class RecoveryAdapter:
    """Trusted factory; configuration and oracle paths never come from the model.

    product_roots are application directories containing src/. author_python is
    the base used to create a fresh venv for EVERY trial, not a reused author env.
    Credentials are read in memory and sent over the worker's stdin pipe only.
    development_responses is a scripted development fixture, forbidden on natural
    or holdout cases. Real provider usage is never fabricated for this fixture.
    """
    def __init__(self, *, product_roots, oracle_paths, product_python=None, author_python=None,
                 llm_config=None, development_responses=None, cancel_check=None,
                 worker_timeout_s=None):
        if set(product_roots) != {"A", "B", "C"}:
            raise ValueError("Specify actual product roots for A/B/C")
        self.products = {arm: Path(root).resolve(strict=True) for arm, root in product_roots.items()}
        if self.products["A"] == self.products["B"] or self.products["B"] != self.products["C"]:
            raise ValueError("A must have a separate old checkout; B/C share the enhanced checkout")
        self.python = _python_entrypoint(product_python or sys.executable)
        self.author_python = _python_entrypoint(author_python or self.python)
        self.oracle_paths = {key: Path(path).resolve(strict=True) for key, path in oracle_paths.items()}
        self.llm = dict(llm_config or {})
        if set(self.llm) - {"base_url", "api_key", "timeout", "max_tokens"}:
            raise ValueError("Unknown trusted provider setting")
        self.responses = deepcopy(development_responses)
        self.cancel_check, self.timeout = cancel_check, worker_timeout_s
        self.worker = Path(__file__).resolve().parents[2] / "scripts" / "recovery_trial_worker.py"
        self.driver_files = {str(Path(__file__).resolve()): _sha(Path(__file__).read_bytes()),
                             str(self.worker): _sha(self.worker.read_bytes())}
        category_hashes = {}
        for name in _CATEGORY_FILES.values():
            path = Path(__file__).with_name(name)
            category_hashes[name] = self.driver_files[str(path)] = _sha(path.read_bytes())
        self.driver_sha256 = _digest({"adapter": self.driver_files[str(Path(__file__).resolve())],
                                     "bootstrap": self.driver_files[str(self.worker)],
                                     "categories": category_hashes})
        self._receipts = {}

    def _verify_driver(self):
        if any(_sha(Path(path).read_bytes()) != sha for path, sha in self.driver_files.items()):
            raise ValueError("Recovery adapter/worker changed after registration")

    def _invoke(self, payload, *, cwd, environment, python=None, timeout=None):
        from src.runtime_preparation import run_owned_process
        self._verify_driver()
        nonce = secrets.token_hex(24)
        payload = {**payload, "nonce": nonce}
        cleaned = []
        env = _environment()
        env.update(AUTOREPRO_DATA_ROOT=str(Path(environment) / "data"),
                   AUTOREPRO_DEPS_ROOT=str(Path(environment) / "dependencies"),
                   PIP_CACHE_DIR=str(Path(environment) / "pip-cache"))
        result = None
        try:
            result = run_owned_process([python or self.python, "-I", "-X", "utf8", str(self.worker)],
                cwd=str(cwd), env=env, input=json.dumps(payload, ensure_ascii=False),
                timeout_s=timeout, cancel_check=self.cancel_check, on_cleanup=lambda: cleaned.append(True),
                max_output_bytes=4 * 1024 * 1024)
            lines = [line for line in result.stdout.splitlines() if line.startswith(nonce + " ")]
            if result.returncode or len(lines) != 1:
                return {"error_type": "WorkerFailed", "cleanup_confirmed": bool(cleaned)}
            value = json.loads(lines[0][len(nonce) + 1:])
            value["cleanup_confirmed"] = bool(cleaned) and value.get("cleanup_confirmed", True) is True
            return value
        except BaseException as exc:
            if not isinstance(exc, (Exception, KeyboardInterrupt)):
                raise
            unclean = getattr(exc, "process_cleanup_confirmed", None) is False
            saved = {}
            if payload.get("operation") == "trial" and cleaned and not unclean:
                try:
                    checkpoint = _path(environment, "recovery_meter.json")
                    value = json.loads(checkpoint.read_text("utf-8"))
                    if value.get("nonce") == nonce and value.get("configuration_sha256") == _digest(payload["configuration"]):
                        saved["usage"] = value["usage"]
                        saved["collaboration"] = value.get("collaboration", {})
                except (OSError, ValueError, KeyError):
                    pass
            return {**saved, "error_type": type(exc).__name__, "cancelled": isinstance(exc, KeyboardInterrupt),
                    "timed_out": type(exc).__name__ == "RuntimePreparationTimeout",
                    "cleanup_confirmed": bool(cleaned) and not unclean}
        finally:
            self._verify_driver()

    def run_trial(self, *, case, arm, repeat, workspace, environment, context, configuration):
        from src.benchmark.recovery import verify_product_identity
        self._verify_driver()
        workspace, environment = Path(workspace).resolve(), Path(environment).resolve()
        if context or any(environment.iterdir()) or workspace == environment:
            raise ValueError("Each trial needs a fresh context and environment")
        if workspace.is_relative_to(environment) or environment.is_relative_to(workspace):
            raise ValueError("Workspace and environment must be separate")
        key = str(workspace)
        if key in self._receipts:
            raise ValueError("A trial workspace cannot be reused")
        if configuration.get("mode") != {"A": "baseline", "B": "single", "C": "collaborative"}[arm]:
            raise ValueError("Arm mode does not match the frozen configuration")
        if self.responses is not None and (case.get("split") != "dev" or case.get("origin") != "injected"):
            raise ValueError("Scripted models are only allowed for injected dev demonstrations")
        root = self.products[arm]
        from src.holdout_registry import product_hashes
        if _digest(product_hashes(self.products["A"])) == _digest(product_hashes(self.products["B"])):
            raise ValueError("A must use a genuinely different frozen baseline product")
        expected_driver = configuration.get("adapter_driver_sha256")
        if expected_driver is not None and expected_driver != self.driver_sha256:
            raise ValueError("Adapter driver does not match preregistration")
        if case.get("split") == "holdout" and expected_driver is None:
            raise ValueError("Preregister adapter_driver_sha256 for default-adapter holdout execution")
        identity = verify_product_identity(root, configuration["product"])
        original = {name: _path(workspace, name).read_bytes() for name in case["source_files"]}
        receipt = {"original": original, "configuration": deepcopy(configuration), "arm": arm,
                   "product_root": str(root), "product": identity, "cleanup_confirmed": True,
                   "driver_sha256": self.driver_sha256}
        self._receipts[key] = receipt
        if case.get("category") not in {'api_compatibility', 'local_import', 'runtime_environment', 'result_evidence'}:
            raw = {"claimed_success": False, "human_interventions": 0, "blocker": "unsupported",
                   "cleanup_confirmed": True, "usage": {"model_requests": 0, "tokens_charged": 0,
                   "prompt_tokens": 0, "completion_tokens": 0, "invalid_retries": 0, "cost": None,
                   "probes": 0, "source_repair_rounds": 0, "repair_actions": 0},
                   "driver_sha256": self.driver_sha256, "execution_scope": "author_compatibility_recovery",
                   "model_source": "scripted_development" if self.responses is not None else "provider"}
            receipt["raw"] = deepcopy(raw)
            return raw
        evidence = _public_evidence(case, workspace)
        provider = {**self.llm}
        provider.setdefault("base_url", os.environ.get("LLM_BASE_URL", ""))
        provider.setdefault("api_key", os.environ.get("LLM_API_KEY", ""))
        if self.responses is None and (not provider["base_url"] or not configuration.get("model")):
            raise ValueError("Configure an actual provider for nonscripted trials")
        payload = {"operation": "trial", "product_root": str(root), "author_python": self.author_python,
                   "workspace": str(workspace), "environment": str(environment), "arm": arm,
                   "category": case["category"],
                   "origin": case.get('origin'), "split": case.get('split'),
                   "source_files": deepcopy(case["source_files"]), "evidence": evidence,
                   "configuration": deepcopy(configuration), "provider": provider,
                   "development_responses": self.responses}
        result = self._invoke(payload, cwd=environment, environment=environment, timeout=self.timeout)
        verify_product_identity(root, identity)
        receipt.update(result, driver_sha256=self.driver_sha256)
        raw = {"claimed_success": result.get("claimed_success") is True, "human_interventions": 0,
               "cancelled": result.get("cancelled") is True, "timed_out": result.get("timed_out") is True,
               "cleanup_confirmed": result.get("cleanup_confirmed") is True,
               "blocker": result.get("blocker", "execution_failed"), "usage": result.get("usage", {}),
               "execution_scope": {'local_import': 'author_local_import_recovery',
                    'runtime_environment': 'author_environment_recovery',
                    'result_evidence': 'author_result_investigation'}.get(case['category'], 'author_compatibility_recovery'), "model_source":
               "scripted_development" if self.responses is not None else "provider",
               "driver_sha256": receipt["driver_sha256"]}
        raw["collaboration"] = deepcopy(result.get("collaboration", {}))
        receipt["raw"] = deepcopy(raw)
        # No provider settings, prompts, hidden oracle data or credentials here.
        public = {k: deepcopy(v) for k, v in receipt.items() if k not in {"original", "configuration", "raw"}}
        (environment / "adapter_receipt.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), "utf-8")
        return raw

    def evaluate_trial(self, *, case, result, workspace, environment):
        """Independent private oracle. Candidate/model expected values are ignored."""
        from src.benchmark.recovery import verify_product_identity
        from src.source_repair import _compatibility_shape
        self._verify_driver()
        workspace, environment = Path(workspace).resolve(), Path(environment).resolve()
        receipt = self._receipts[str(workspace)]
        if result != receipt["raw"]:
            raise ValueError("Runner result was replaced before scoring")
        cleanup = receipt.get("cleanup_confirmed") is True
        if (result.get("blocker") == "unsupported" and case.get('category') not in
                {'api_compatibility', 'local_import', 'runtime_environment', 'result_evidence'}):
            return {"oracle_passed": False, "protocol_preserved": True, "cleanup_confirmed": cleanup,
                    "unsupported": True}
        verify_product_identity(receipt["product_root"], receipt["product"])
        passed, preserved = False, True
        interrupted = {"cancelled": result.get("cancelled") is True, "timed_out": result.get("timed_out") is True}
        infrastructure_error = None
        scoring_stage = "protocol"
        try:
            effective = Path(receipt.get("effective_workspace", workspace)).resolve(strict=True)
            if effective != workspace and not effective.is_relative_to(environment):
                raise ValueError("Effective source escaped the trial")
            editable = set(receipt.get("editable", []))
            for name, raw in receipt["original"].items():
                if _sha(raw) != case["source_files"][name] or _path(workspace, name).read_bytes() != raw:
                    raise ValueError("Original author export changed")
                candidate = _path(effective, name).read_bytes()
                if candidate != raw and (name not in editable or _compatibility_shape(raw) != _compatibility_shape(candidate)):
                    raise ValueError("Frozen protocol/input or noncompatibility source changed")
            if receipt.get("plan_preserved") is not True or receipt.get("runtime_preserved") is not True:
                raise ValueError("Original command/runtime contract was not preserved")
            category = case.get('category')
            if category == 'runtime_environment':
                import_paths = []
                _category_driver(category)['validate_environment_receipt'](receipt, case, workspace, environment)
            else:
                import_paths = _validated_import_paths(receipt, case, workspace)
            if result.get('blocker') == 'unsupported':
                self._verify_driver()
                return {'oracle_passed': False, 'protocol_preserved': True,
                        'cleanup_confirmed': cleanup, 'unsupported': True, **interrupted}
            scoring_stage = "oracle_setup"
            private = self.oracle_paths[case["id"]]
            oracle_spec = case["oracle"]
            if _sha(private.read_bytes()) != oracle_spec["sha256"]:
                raise ValueError("Private preregistered oracle changed")
            oracle = json.loads(private.read_text("utf-8"))
            if category == 'result_evidence':
                if oracle_spec['kind'] != 'result_evidence_v1':
                    raise ValueError('Result cases require a registered result_evidence_v1 oracle')
                evaluated = _category_driver(category)['evaluate_result_evidence'](
                    evidence=case['initial_evidence'], receipt=receipt, workspace=workspace, oracle=oracle)
                self._verify_driver()
                return {**evaluated, 'protocol_preserved': preserved and evaluated.get('protocol_preserved') is True,
                    'cleanup_confirmed': cleanup and evaluated.get('cleanup_confirmed') is True,
                    **{flag: interrupted[flag] or evaluated.get(flag) is True for flag in interrupted},
                    'oracle_passed': evaluated.get('oracle_passed') is True and preserved and cleanup
                        and not any(interrupted.values())}
            if oracle_spec["kind"] not in {"trusted_stdout_v1", "original_assertions_v1"} or oracle.get("version") != 1:
                raise ValueError("Unsupported independent oracle")
            if not receipt.get("execution_succeeded") or not receipt.get("initial_failure_observed"):
                return {"oracle_passed": False, "protocol_preserved": True, "cleanup_confirmed": cleanup,
                        "reason": "original_failure_or_recovery_not_observed"}
            if oracle_spec["kind"] == "trusted_stdout_v1":
                expected = oracle.get("stdout_sha256")
                if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
                    raise ValueError("Trusted stdout oracle needs its preregistered digest")
                passed = receipt.get("stdout_sha256") == expected
            else:
                scripts = oracle.get("scripts")
                if not isinstance(scripts, list) or not 1 <= len(scripts) <= 8:
                    raise ValueError("Original assertion oracle needs 1-8 fixed scripts")
                # Read private assertions only after the model worker has exited.
                contracts = []
                for script in scripts:
                    path = _path(private.parent, script["path"])
                    if _sha(path.read_bytes()) != script["sha256"]:
                        raise ValueError("Original oracle assertions changed")
                    if path.is_relative_to(workspace) or path.is_relative_to(environment):
                        raise ValueError("Oracle assertions cannot come from candidate files")
                    contracts.append({"path": str(path), "sha256": script["sha256"]})
                dependency_paths = []
                for entry in receipt.get("dependency_paths", []):
                    path = Path(entry).resolve(strict=True)
                    if not path.is_relative_to(environment / "dependencies"):
                        raise ValueError("Oracle dependency path escaped the private trial environment")
                    dependency_paths.append(str(path))
                observed = self._invoke({"operation": "oracle", "workspace": str(effective),
                    "scripts": contracts, "dependency_paths": [*import_paths, *dependency_paths]}, cwd=effective, environment=environment,
                    python=receipt["author_python"], timeout=60)
                cleanup = cleanup and observed.get("cleanup_confirmed") is True
                for flag in interrupted:
                    interrupted[flag] |= observed.get(flag) is True
                if observed.get("error_type") not in {None, "AssertionError"}:
                    infrastructure_error = observed["error_type"]
                passed = observed.get("oracle_passed") is True
        except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
            preserved = False
            if scoring_stage == "oracle_setup":
                infrastructure_error = type(exc).__name__
        self._verify_driver()
        return {"oracle_passed": passed and preserved and cleanup and not any(interrupted.values()),
                "protocol_preserved": preserved, "cleanup_confirmed": cleanup,
                **interrupted,
                **({"infrastructure_error": infrastructure_error} if infrastructure_error else {}),
                "numerical_equivalence": "unknown", "scope": "preregistered_recovery_contract_only"}


_DEFAULT_ADAPTER = None


def _configured_adapter():
    """CLI callbacks use one operator-selected local config, never a case path."""
    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        name = os.environ.get("AUTOREPRO_RECOVERY_ADAPTER_CONFIG", "")
        if not name:
            raise ValueError("Set AUTOREPRO_RECOVERY_ADAPTER_CONFIG to the trusted adapter JSON")
        path = Path(name).resolve(strict=True)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Adapter configuration is too large")
        config = json.loads(path.read_text("utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Adapter configuration must be an object")
        # Files are allowed to hold public endpoint/model settings only. Keys
        # supplied in memory through the Python constructor are never persisted.
        if "api_key" in config.get("llm_config", {}):
            raise ValueError("Keep provider credentials in LLM_API_KEY, not adapter files")
        _DEFAULT_ADAPTER = RecoveryAdapter(**config)
    return _DEFAULT_ADAPTER


def run_trial(**kwargs):
    """Default trusted CLI trial callback."""
    return _configured_adapter().run_trial(**kwargs)


def evaluate_trial(**kwargs):
    """Default trusted CLI private oracle callback, sharing the driver's receipts."""
    return _configured_adapter().evaluate_trial(**kwargs)


class _ScriptedLLM:
    mock_mode = False
    api_key = ""
    max_tokens = 8192

    def __init__(self, responses):
        self.responses = deepcopy(responses)
        self.last_usage = {}

    def chat(self, prompt, **kwargs):
        task = kwargs.get("task")
        rows = self.responses.get(task, [])
        if not rows:
            raise ValueError("Development response missing for " + str(task))
        value = rows.pop(0)
        return value if isinstance(value, str) else json.dumps(value)


class _MeteredLLM:
    def __init__(self, client, on_update=None):
        self.client = client
        self.calls, self.prompt, self.completion, self.known, self.estimated = 0, 0, 0, 0, 0
        self.missing, self.seconds, self.last_usage = 0, 0.0, {}
        self.invalid_retries = 0
        self.on_update, self.repair_calls = on_update, 0
        self.observed_repair_actions = 0
        self.in_flight = False
        self.author_runs = self.execution_retries = self.same_failure_retries = 0
        self.last_execution_failure = None
        self.author_execution_in_flight = False

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.repair_calls += int(kwargs.get("task") == "repository_repair")
        try:
            self.invalid_retries += int(bool(json.loads(prompt).get("format_feedback")))
        except (ValueError, TypeError, AttributeError):
            pass
        self.client.last_usage = {}
        self.in_flight = True
        if self.on_update:
            self.on_update()
        started, response = time.monotonic(), None
        try:
            response = self.client.chat(prompt, **kwargs)
            return response
        finally:
            self.seconds += time.monotonic() - started
            self.last_usage = getattr(self.client, "last_usage", {})
            usage = self.last_usage if isinstance(self.last_usage, dict) else {}
            if type(usage.get("total_tokens")) is int and usage["total_tokens"] > 0:
                self.known += usage["total_tokens"]
                for key, attribute in (("prompt_tokens", "prompt"), ("completion_tokens", "completion")):
                    count = usage.get(key)
                    if type(count) is int and count >= 0:
                        setattr(self, attribute, getattr(self, attribute) + count)
            else:
                self.missing += 1
                self.estimated += len((str(prompt) + str(kwargs.get("system_prompt", "")) + str(response or "")).encode())
            self.in_flight = False
            if self.on_update:
                self.on_update()


def _measurement(llm, session=None):
    usage = {"model_requests": llm.calls, "tokens_charged": None if llm.in_flight else llm.known + llm.estimated,
             "prompt_tokens": llm.prompt if not llm.missing and not llm.in_flight else None,
             "completion_tokens": llm.completion if not llm.missing and not llm.in_flight else None,
             "cost": None, "invalid_retries": llm.invalid_retries, "probes": 0,
             'execution_retries': None if llm.author_execution_in_flight else llm.execution_retries,
             'same_failure_retries': None if llm.author_execution_in_flight else llm.same_failure_retries,
             "source_repair_rounds": llm.repair_calls,
             "repair_actions": llm.repair_calls + llm.observed_repair_actions}
    if session:
        accounting = session.state["usage"]
        usage.update(probes=accounting["probes"],
                     source_repair_rounds=accounting.get("source_patch_rounds", 0),
                     repair_actions=accounting.get("repair_actions", 0),
                     roles=session.safe(deepcopy(accounting.get("roles", {}))))
    return usage


def _execution_failure_identity(execution):
    """Compare the final actual failure site; changed errors are separate work."""
    if execution.get('success') is True:
        return None
    from src.author_self_repair import failure_diagnostic
    diagnostic = failure_diagnostic(execution)
    lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
    if not lines:
        return None
    locations = re.findall(r'File "([^"]+)", line (\d+), in ([^\r\n]+)', diagnostic)
    location = locations[-1] if locations else None
    final = execution.get('final') or {}
    last = (execution.get('steps') or [{}])[-1]
    cwd = final.get('cwd') or last.get('cwd')
    terminal = lines[-1]
    if cwd:
        variants = {str(cwd), str(cwd).replace('\\', '/'), str(cwd).replace('/', '\\')}
        variants.update(value.replace('\\', '\\\\') for value in list(variants))
        for root in sorted(variants, key=len, reverse=True):
            root = root.rstrip('/\\')
            if root:
                terminal = re.sub(re.escape(root) + r'''(?=[/\\]|["']|$)''', '<author_workspace>', terminal)
    if location:
        path = location[0]
        if cwd:
            try:
                path = os.path.relpath(path, cwd)
            except ValueError:
                pass
        location = (os.path.normcase(path).replace('\\', '/'), location[1], location[2])
    return _digest([final.get('id') or last.get('id'), location, terminal])


def _record_author_execution(llm, execution):
    # Runtime preparation and policy rejection can return without starting an
    # author process. They neither add retries nor interrupt the sequence of
    # actual author failures. Interrupted invocations without a result stay
    # unknown in the checkpoint instead of inventing a completed observation.
    if execution.get('executed') is not True:
        return
    llm.execution_retries += int(llm.author_runs > 0)
    llm.author_runs += 1
    signature = _execution_failure_identity(execution)
    if signature is not None and signature == llm.last_execution_failure:
        llm.same_failure_retries += 1
    llm.last_execution_failure = signature


def _collaboration_summary(session=None):
    """Observed review activity, without prompts, patch bodies or verdict power."""
    if session is None:
        return {}
    reviews = [{key: deepcopy(row.get(key)) for key in (
        "stage", "reviewer", "candidate_sha256", "action", "review_id", "evidence_ids", "reason")}
        for row in session.state.get("reviews", [])]
    rejected = {row["candidate_sha256"] for row in reviews
                if row["stage"] == "patch" and row["action"] == "reject"
                and isinstance(row["candidate_sha256"], str)}
    return session.safe({"mode": session.mode, "roles": deepcopy(session.state["usage"].get("roles", {})),
        "reviews": reviews, "counts": {
            "independent_reviews": sum(row["reviewer"] == "independent" for row in reviews),
            "self_checks": sum(row["reviewer"] == "self_check" for row in reviews),
            "review_probe_requests": sum(row["action"] == "probe" for row in reviews),
            "distinct_patch_candidates_rejected": len(rejected)},
        "scope": "observed_recovery_activity_only"})


def _assert_product_modules(root):
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            file = getattr(module, "__file__", None)
            if file:
                path = Path(file).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Worker imported application code outside selected product")
                loaded[name] = path.relative_to(root).as_posix()
    return loaded


class _CaptureWorkerFailure:
    """Keep the concrete execution and shared meter when recovery stops."""
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, kind, failure, traceback):
        if failure is None:
            return False
        self.result.update(error_type=type(failure).__name__,
            cancelled=isinstance(failure, (KeyboardInterrupt, SystemExit)),
            timed_out=type(failure).__name__ == "RuntimePreparationTimeout",
            cleanup_confirmed=getattr(failure, "process_cleanup_confirmed", None) is not False)
        return True


def _trial_worker(payload):
    """Runs only under the fixed helper, with no application modules loaded yet."""
    from contextlib import ExitStack
    root = Path(payload["product_root"]).resolve(strict=True)
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise ValueError("Product modules were loaded before selecting the frozen checkout")
    sys.path.insert(0, str(root))
    from src.runtime_preparation import run_owned_process
    from src.repository_runner import RepositoryRunner
    from src.author_failure_analysis import analyze_author_failure
    from src.author_self_repair import failure_diagnostic
    from src.source_repair import repair_author_failure
    from src.execution_plan import build_plan
    from src.llm.llm_client import LLMClient
    from src.agent_collaboration import CollaborationSession
    workspace, environment = Path(payload["workspace"]), Path(payload["environment"])
    evidence, config = payload["evidence"], payload["configuration"]
    category = payload.get('category', 'api_compatibility')
    category_driver = _category_driver(category)
    case_identity = {key: payload.get(key) for key in ('category', 'origin', 'split', 'source_files')}
    environment_contract = (category_driver['validate_environment_evidence'](case_identity, workspace, evidence)
                            if category == 'runtime_environment' else None)
    venv = environment / "author-venv"
    setup = [payload["author_python"], "-I", "-m", "venv"]
    dependencies = evidence.get("environment", {})
    if category != 'runtime_environment' and not dependencies.get("requirements_txt") and not dependencies.get("required_packages"):
        setup.append("--without-pip")
    prepared = run_owned_process([*setup, str(venv)], timeout_s=60)
    if prepared.returncode:
        raise ValueError("Fresh isolated author venv could not be prepared")
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = {**dependencies, "python_executable": str(python), "auto_prepare": False,
           "dependency_health_check": False}
    if environment_contract is not None:
        env = category_driver['prepare_environment_runtime'](environment_contract, workspace, environment, env)
    original = evidence["plan"]
    spec = _digest(original)
    plan = build_plan(profile="recovery_benchmark", spec_sha256=spec, workspace=workspace,
                      repository=original["repository"], dataset={"sources": []}, steps=original["steps"])
    data = {"source_repair": {"enabled": True, "status": "pending"}, "env_config": env,
            "dataset_provenance": {"sources": []}, "execution_plan": plan, "spec_sha256": spec}
    request = {"source_repair": True, "source_repair_auto": True,
               "source_repair_editable": evidence.get("editable", [])}
    run_dir = environment / "recovery"
    run_dir.mkdir()
    source_files = payload["source_files"]

    def verify_inputs():
        for name, expected in source_files.items():
            if _sha(_path(workspace, name).read_bytes()) != expected:
                raise ValueError("Frozen author input changed during recovery")

    scripted = payload.get("development_responses")
    client = (_ScriptedLLM(scripted) if scripted is not None else
              LLMClient(model=config["model"], mock_mode=False, **payload["provider"]))
    llm = _MeteredLLM(client)
    runner = RepositoryRunner()
    session, task, failure = None, None, {}
    def checkpoint():
        value = {"nonce": payload["nonce"], "configuration_sha256": _digest(config),
                 "usage": _measurement(llm, session), "collaboration": _collaboration_summary(session)}
        temporary = environment / "recovery_meter.pending"
        temporary.write_text(json.dumps(value), "utf-8")
        temporary.replace(environment / "recovery_meter.json")
    llm.on_update = checkpoint
    author_run = runner.run
    def measured_author_run(*args, **kwargs):
        llm.author_execution_in_flight = True
        checkpoint()
        try:
            result = author_run(*args, **kwargs)
            llm.author_execution_in_flight = False
            _record_author_execution(llm, result)
            return result
        finally:
            checkpoint()
    runner.run = measured_author_run
    checkpoint()
    def repair_event(event):
        llm.observed_repair_actions = len((data.get("self_repair") or {}).get("attempts") or [])
        checkpoint()

    def investigate_failure(result, effective):
        nonlocal task
        if session is None:
            return None
        from src.investigation_tools import InvestigationTools
        probe_environment = {**effective, "investigation_dependency_paths":
            (result.get("environment") or {}).get("dependency_paths") or []}
        toolset = InvestigationTools(workspace, evidence["packet"], probe_environment, run_dir)
        execute = toolset.execute
        def allowed_tool(name, arguments):
            if name not in config["tools"]:
                return {"status": "rejected", "cleanup_confirmed": True, "evidence": [],
                        "reason": "Tool is outside the preregistered subset"}
            return execute(name, arguments)
        toolset.execute = allowed_tool
        task = session.investigate(trigger="execution_failure", plan=original,
            packet=evidence["packet"], workspace=workspace, environment=effective,
            execution=result, tools=toolset)
        session.ensure_running()
        return task

    def analyze_local_failure(result, effective):
        investigation_task = investigate_failure(result, effective)
        if session and investigation_task and investigation_task.get("status") != "repair_proposed":
            return None
        analyze_author_failure(session.wrap_llm() if session else llm, result, original,
            evidence["packet"], workspace, effective, run_dir, data)
        if session:
            session.ensure_running()
        # This adapter never applies dependency pins for a local-import task.
        return None

    def analyze_environment_failure(result, effective):
        investigation_task = investigate_failure(result, effective)
        if session and investigation_task and investigation_task.get('status') != 'repair_proposed':
            return None
        value = analyze_author_failure(session.wrap_llm() if session else llm, result, original,
            evidence['packet'], workspace, effective, run_dir, data)
        if session:
            session.ensure_running()
        return value

    execution, initial, observed = {}, {}, False
    with ExitStack() as stack:
        stack.enter_context(_CaptureWorkerFailure(failure))
        # Preserve the old pipeline's bounded handoff context in A. The enhanced
        # single mode deliberately uses one generalist, matching product policy.
        if payload["arm"] != "B":
            collaboration = CollaborationSession(credential=getattr(client, "api_key", ""))
            stack.enter_context(collaboration.scope())
        if payload["arm"] != "A":
            from src.intelligent_recovery import RecoverySession
            session = RecoverySession(llm, mode=config["mode"], budget=config["budget"])
            persist = session.persist
            def persist_and_measure():
                persist()
                checkpoint()
            session.persist = persist_and_measure
            session.attach(data, run_dir)
            stack.enter_context(session.scope())
        execution = runner.run(workspace, plan, deepcopy(env))
        if category == 'runtime_environment':
            data['_runtime_environment'] = category_driver['record_environment_execution'](
                execution, workspace, environment, env, contract=environment_contract)
            data['_runtime_environment_last_execution'] = execution
        initial = deepcopy(execution)
        diagnostic = failure_diagnostic(initial)
        signature = evidence.get('initial_failure') or {}
        observed = (category != 'result_evidence' and not initial.get("success") and initial.get("executed") is True
                    and bool(signature.get('error_type')) and signature['error_type'] in diagnostic
                    and bool(signature.get('diagnostic_contains')) and signature['diagnostic_contains'] in diagnostic)
        if payload.get("category") == "local_import":
            from src.author_self_repair import terminal_missing_import
            observed = observed and bool(terminal_missing_import(diagnostic))
        if session:
            session.record_execution(execution)
        if category == 'result_evidence':
            toolset = None
            if session:
                from src.investigation_tools import InvestigationTools
                packet = deepcopy(evidence['packet'])
                contract = evidence['result_contract']
                artifact = _path(workspace, contract['artifact'], exists=False)
                if artifact.is_file():
                    packet['investigation_artifacts'] = {contract['artifact']: {
                        'sha256': _sha(artifact.read_bytes()), 'public': True, 'metric_summary': True}}
                toolset = InvestigationTools(workspace, packet,
                    {**env, 'investigation_dependency_paths': (execution.get('environment') or {}).get('dependency_paths', [])}, run_dir)
                execute = toolset.execute
                def allowed_result_tool(name, arguments):
                    if name not in config['tools']:
                        return {'status': 'rejected', 'cleanup_confirmed': True, 'evidence': [],
                                'reason': 'Tool is outside the preregistered subset'}
                    return execute(name, arguments)
                toolset.execute = allowed_result_tool
            result_evidence = category_driver['run_result_evidence'](
                evidence=evidence, workspace=workspace, environment=env, execution=execution,
                llm=llm, session=session, run_dir=run_dir, data=data,
                tools=toolset, verify_inputs=verify_inputs)
        elif observed and category == 'runtime_environment':
            execution = category_driver['run_environment_recovery'](runner, execution, plan, original, evidence['packet'],
                workspace, env, run_dir, data, verify_inputs, analyze_environment_failure, repair_event,
                contract=environment_contract, trial_environment=environment)
            if session:
                session.record_execution(execution)
        elif observed and payload.get("category") == "local_import":
            execution = _run_local_import_recovery(runner, execution, plan, original, evidence["packet"],
                workspace, env, run_dir, data, verify_inputs, analyze_local_failure, repair_event)
            if session:
                session.record_execution(execution)
        elif observed and not execution.get("success"):
            if session:
                investigate_failure(execution, env)
            if session is None or task and task.get("status") == "repair_proposed":
                analyzer_llm = session.wrap_llm() if session else llm
                analyze_author_failure(analyzer_llm, execution, original, evidence["packet"],
                                       workspace, env, run_dir, data)
                if session:
                    session.ensure_running()
                execution = repair_author_failure(runner, llm, request, original,
                    {"files": source_files}, workspace, run_dir, data, execution,
                    verify_inputs, lambda event: None)
                if session:
                    session.record_execution(execution)
        verify_inputs()
    execution = data.pop("_local_import_last_execution", execution)
    execution = data.pop('_runtime_environment_last_execution', execution)
    effective = Path(data["source_repair"].get("workspace", workspace))
    prepared_runtime = initial.get("environment", {})
    final_runtime = execution.get("environment", {})
    runtime_keys = ("executable", "runtime", "requirements_txt")
    runtime_ok = all(final_runtime.get(key) == prepared_runtime.get(key) for key in runtime_keys)
    if category == 'runtime_environment':
        runtime_ok = (data.get('_runtime_environment') or {}).get('preserved') is True
    expected_steps = [(step["id"], step["argv"][1:]) for step in original["steps"]]
    actual_steps = [(step["id"], (step.get("argv") or [])[1:]) for step in execution.get("steps", [])]
    # A failed prefix still preserves the protocol; successful execution must run
    # every original required step in its registered order.
    plan_ok = actual_steps == expected_steps[:len(actual_steps)] and (
        not execution.get("success") or len(actual_steps) == len(expected_steps))
    usage = _measurement(llm, session)
    from src.repository_repairer import repair_is_stopped
    stopped = repair_is_stopped(execution) or bool(failure)
    reason = session.state.get("stop_reason", "") if session else ""
    repair_status = ((data.get('_runtime_environment') or {}).get('status') if category == 'runtime_environment' else
                    (data.get("self_repair") or {}).get("status") if payload.get("category") == "local_import"
                     else data["source_repair"].get("status"))
    if repair_status == "budget_exhausted":
        reason = "budget_exhausted"
    result_evidence = data.get('result_evidence', {})
    success = (result_evidence.get('completed') is True if category == 'result_evidence'
               else observed and execution.get('success') is True)
    return {"claimed_success": success and not stopped,
            "execution_succeeded": execution.get("success") is True and not stopped,
            "initial_failure_observed": observed, "plan_preserved": plan_ok,
            "runtime_preserved": runtime_ok, "effective_workspace": str(effective),
            "author_python": str(python), "editable": data["source_repair"].get("editable", []),
            "stdout_sha256": _sha((execution.get("final", {}).get("stdout") or "").encode("utf-8")),
            "blocker": 'unsupported' if repair_status == 'unsupported' else "budget_exhausted" if reason == "budget_exhausted" else
                       "execution_failed" if stopped or not execution.get("success") else "unclassified",
            "timed_out": failure.get("timed_out", False) or any(step.get("timed_out") for step in execution.get("steps", [])),
            "cancelled": failure.get("cancelled", False) or execution.get("cancelled") is True,
            "cleanup_confirmed": failure.get("cleanup_confirmed", True) and not RecoveryAdapterCleanup.unconfirmed(execution),
            "dependency_paths": final_runtime.get("dependency_paths") or [],
            "repository_pythonpath": (execution.get("effective_env_config") or {}).get("repository_pythonpath") or [],
            "import_recovery_proofs": data.get("_import_recovery_proofs", []),
            "import_environment_preserved": data.get("_import_environment_preserved", True),
            'runtime_environment': data.get('_runtime_environment'),
            'result_evidence': result_evidence,
            'initial_success_observed': result_evidence.get('initial_success_observed') is True,
            "error_type": failure.get("error_type"),
            "loaded_product_modules": _assert_product_modules(root), "usage": usage,
            "collaboration": _collaboration_summary(session),
            "repair_status": repair_status,
            "investigation_status": task.get("status") if task else None}


class RecoveryAdapterCleanup:
    @staticmethod
    def unconfirmed(execution):
        pending = [(execution, False)]
        seen, diagnostics = set(), []
        while pending:
            record, receipt = pending.pop()
            if not isinstance(record, (dict, list, tuple)):
                continue
            identity = (id(record), receipt)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(record, (list, tuple)):
                pending.extend((item, receipt) for item in record)
                continue
            if (record.get("cleanup_confirmed") is False or record.get("process_cleanup_confirmed") is False
                    or any(record.get(key) for key in ("cleanup_failed", "cleanup_unconfirmed", "cleanup_error", "cleanup_errors", "processes_alive"))
                    or record.get("state") == "cleanup_unconfirmed"
                    or record.get("status") in {"cleanup_failed", "cleanup_unconfirmed"}
                    or receipt and (record.get("confirmed") is False or record.get("status") in {"failed", "unconfirmed"})):
                return True
            diagnostics.extend(str(record.get(key) or "") for key in ("reason", "stderr", "stdout"))
            pending.extend((record.get(key), True) for key in ("cleanup", "process_cleanup", "cleanup_receipt", "environment_cleanup"))
            pending.extend((record.get(key), False) for key in
                           ('execution', 'final', 'effective_env_config', 'environment', 'steps', 'attempts', 'stages'))
        return bool(re.search(r"process[- ](?:tree|group) cleanup (?:failed|was denied)|\bcleanup_unconfirmed\b",
                              "\n".join(diagnostics), re.I))


def worker_main():
    """Fixed helper entry. It never takes an executable/plugin from a case."""
    from contextlib import redirect_stdout
    from io import StringIO
    import runpy
    payload = json.loads(sys.stdin.buffer.read(8 * 1024 * 1024 + 1))
    nonce = payload["nonce"]
    # Product logs cannot impersonate this structured control channel.
    with redirect_stdout(StringIO()):
        try:
            if payload["operation"] == "trial":
                result = _trial_worker(payload)
            elif payload["operation"] == "oracle":
                root = Path(payload["workspace"]).resolve(strict=True)
                sys.path[:0] = [str(root), *payload.get("dependency_paths", [])]
                for script in payload["scripts"]:
                    path = Path(script["path"])
                    if _sha(path.read_bytes()) != script["sha256"] or sys.flags.optimize:
                        raise ValueError("Oracle identity/assertions changed")
                    sys.argv = [str(path)]
                    try:
                        runpy.run_path(str(path), run_name="__main__")
                    except SystemExit as exit_status:
                        if exit_status.code not in (None, 0):
                            raise AssertionError("Original test exited unsuccessfully") from exit_status
                result = {"oracle_passed": True}
            else:
                raise ValueError("Unknown fixed worker operation")
        except BaseException as exc:
            result = {"error_type": type(exc).__name__, "oracle_passed": False,
                      "cancelled": isinstance(exc, KeyboardInterrupt),
                      "cleanup_confirmed": getattr(exc, "process_cleanup_confirmed", None) is not False}
    print(nonce + " " + json.dumps(result, ensure_ascii=False, allow_nan=False))
