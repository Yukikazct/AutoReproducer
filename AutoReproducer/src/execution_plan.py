"""Build and validate repository execution plans: argv, cwd, order, budget, evidence.

A plan is a frozen dict, never generated shell text. Reviewers author the steps;
this module only normalizes and checks them so a run can be replayed from disk.
"""
import math
import re

from src.safety.paths import relative_path

PLAN_VERSION = 1
MODES = {"repository"}
KINDS = {"check", "run", "train", "eval", "prepare"}
STEP_ID = re.compile(r"[A-Za-z0-9_-]+")
DEFAULT_TIMEOUT_S = 600.0


class RepositoryModeFallbackRejected(RuntimeError):
    """A repository profile must never be re-routed into single-file generation."""


def _step_identifier(value, position):
    if not isinstance(value, str) or not STEP_ID.fullmatch(value):
        raise ValueError(f"step {position} id must be a safe file name: {value!r}")
    return value


def declared_paths(value, identifier, field, *, allow_glob):
    """Normalize the relative paths a step declares as evidence or prerequisites.

    Shared with RepositoryRunner so an authored step list and a rebuilt plan cannot
    be validated by two different path rules.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"step {identifier} {field} must be a list")
    declared = []
    for entry in value:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"step {identifier} {field} entries need a path")
        path = relative_path(entry["path"], f"{field} path")
        if not allow_glob and "*" in path.as_posix():
            raise ValueError(f"step {identifier} {field} prerequisites cannot be globs: {entry['path']}")
        declared.append({"path": path.as_posix(),
                         "optional": entry.get("optional", False) is True,
                         "kind": entry.get("kind")})
    return declared


def _declared_paths(value, identifier, field, *, allow_glob):
    return declared_paths(value, identifier, field, allow_glob=allow_glob)


def _step(step, position):
    if not isinstance(step, dict):
        raise ValueError(f"step {position} must be an object")
    identifier = _step_identifier(step.get("id"), position)
    argv = step.get("argv")
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)):
        raise ValueError(f"step {identifier} argv must be a non-empty list of strings")
    kind = step.get("kind", "run")
    if kind not in KINDS:
        raise ValueError(f"step {identifier} kind must be one of {sorted(KINDS)}")
    timeout = step.get("timeout_s", DEFAULT_TIMEOUT_S)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError(f"step {identifier} timeout_s must be a positive number")
    env = step.get("env", {})
    if not isinstance(env, dict) or any(
            not isinstance(key, str) or not key or "=" in key or "\x00" in key
            or not isinstance(value, str) or "\x00" in value for key, value in env.items()):
        raise ValueError(f"step {identifier} env must map names to strings")
    # Omitted means required: forgetting the flag must never quietly downgrade a train
    # or eval step to "failure is acceptable" for a profile author who never opted in.
    # Declared explicitly here so it survives the write/read-back round trip.
    required = step.get("required", True)
    if not isinstance(required, bool):
        raise ValueError(f"step {identifier} required must be true or false")
    dependencies = step.get("depends_on", [])
    if not isinstance(dependencies, list):
        raise ValueError(f"step {identifier} depends_on must be a list")
    dependencies = [_step_identifier(dep, f"{identifier} dependency") for dep in dependencies]
    if len(set(dependencies)) != len(dependencies):
        raise ValueError(f"step {identifier} depends_on has duplicates")
    if identifier in dependencies:
        raise ValueError(f"step {identifier} cannot depend on itself")
    return {"id": identifier, "kind": kind, "argv": list(argv),
            "cwd": relative_path(step.get("cwd", "."), "cwd").as_posix() or ".",
            "env": dict(env), "timeout_s": float(timeout),
            "depends_on": dependencies, "required": required,
            "requires": _declared_paths(step.get("requires"), identifier, "requires", allow_glob=False),
            "artifacts": _declared_paths(step.get("artifacts"), identifier, "artifacts", allow_glob=True)}


def _ordered(steps):
    """Keep the reviewed list order and check every declared dependency against it.

    A dependency must name a step scheduled earlier, so edges only ever point
    backwards and a cycle is unrepresentable: an A->B->A loop would have to place A
    after itself. Rejecting forward references is therefore both the unknown-id and
    the cycle check, and a reviewer can write steps in reading order instead of
    maintaining a topological sort by hand.
    """
    seen, planned = set(), []
    for step in steps:
        # Evidence is archived under <run_dir>/artifacts/<id>/, so a repeated id
        # would let two steps write the same slot and make depends_on ambiguous.
        if step["id"] in seen:
            raise ValueError(f"step {step['id']} is already declared earlier in the plan")
        for dependency in step["depends_on"]:
            if dependency not in seen:
                raise ValueError(
                    f"step {step['id']} depends on {dependency}, which is not scheduled before it")
        seen.add(step["id"])
        planned.append(step)
    return planned


def build_plan(*, mode="repository", profile, spec_sha256, workspace, repo_root=None,
               repository=None, dataset=None, limits=None, steps):
    """Return a serializable plan; raises ValueError on anything a reviewer must fix."""
    if mode not in MODES:
        raise ValueError(f"unsupported plan mode: {mode}")
    if not isinstance(profile, str) or not profile:
        raise ValueError("plan profile must be a reviewed experiment id")
    if not isinstance(spec_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", spec_sha256):
        raise ValueError("plan requires the 64 hex digest of the frozen experiment spec")
    if not isinstance(steps, list) or not steps:
        raise ValueError("repository plan requires at least one step")
    planned = _ordered([_step(step, position) for position, step in enumerate(steps)])
    for name, value in (("repository", repository), ("dataset", dataset), ("limits", limits)):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"plan {name} must be an object")
    if limits:
        total = limits.get("total_seconds")
        if total is not None and (isinstance(total, bool) or not isinstance(total, (int, float))
                                  or not math.isfinite(total) or total <= 0):
            raise ValueError("plan limits.total_seconds must be a positive number")
    return {"version": PLAN_VERSION, "mode": mode, "profile": profile,
            "spec_sha256": spec_sha256, "workspace": str(workspace),
            "repo_root": str(repo_root if repo_root is not None else workspace),
            "repository": dict(repository or {}), "dataset": dict(dataset or {}),
            "limits": dict(limits or {}), "steps": planned}


def plan_step_ids(steps):
    """Return the ids a step list declares, tolerating entries read back from disk."""
    return [step.get("id") for step in steps if isinstance(step, dict) and "id" in step]


def plan_steps(plan):
    """Return the steps a plan describes, whether handed a plan or a bare step list.

    Extraction only: RepositoryRunner._plan stays the single authority that checks
    argv, cwd and script paths against the live workspace, including files created by
    earlier steps. Duplicating that here would let the two validations disagree.
    """
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("execution plan requires a non-empty steps list")
        if any(not isinstance(step, dict) for step in steps):
            raise ValueError("execution plan steps must be objects")
        return steps
    if not isinstance(plan, list) or not plan:
        raise ValueError("repository plan requires at least one step")
    return plan
