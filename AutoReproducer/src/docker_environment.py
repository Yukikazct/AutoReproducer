"""Docker resources owned by one disposable reproduction, with crash recovery."""
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from src.safety.paths import workspace_path
from src.safety.files import directory_bytes, remove_tree


JOURNAL = ".autorepro-docker.json"
LABEL = "org.autoreproducer.environment"
BUILDKIT_IMAGE = "moby/buildkit:buildx-stable-1"
LEGACY_IMAGES = ("autorepro-env:latest", "autorepro-base:latest")


def _command(argv, timeout=120):
    from src.runtime_preparation import run_owned_process
    environment = os.environ.copy()
    for name in ("LLM_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "AUTOREPRO_GITHUB_TOKEN"):
        environment.pop(name, None)
    return run_owned_process(argv, timeout_s=timeout, env=environment)


def _inspect(command, kind, name):
    result = _command([command, kind, "inspect", name])
    if result.returncode:
        if "no such" in (result.stderr or "").lower():
            return None
        raise RuntimeError(result.stderr or result.stdout or f"Cannot inspect {name}")
    value = json.loads(result.stdout)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Docker inspection did not identify one resource")
    return value[0]


def _write(root, state):
    from src.environment_lifecycle import _write_json
    _write_json(workspace_path(root, JOURNAL, "Docker ownership journal"), state)


def _check_state(state):
    if not isinstance(state, dict):
        raise ValueError("Invalid Docker ownership journal")
    owner = state.get("owner", "")
    if (state.get("version") != 1 or not re.fullmatch(r"[a-f0-9]{32}", owner)
            or Path(state.get("command", "")).name.lower() not in {"docker", "docker.exe"}
            or state.get("builder") not in {None, "autorepro-build-" + owner}
            or state.get("image") not in {None, "autorepro-env:" + owner}
            or not isinstance(state.get("containers"), list)
            or any(not re.fullmatch(r"autorepro-[a-f0-9]{32}", name) for name in state["containers"])
            or not isinstance(state.get("workspaces"), list)
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in state["workspaces"])
            or not isinstance(state.get("buildkit_owned"), bool)):
        raise ValueError("Invalid Docker ownership journal")


def _release_builder(root, state):
    command, builder = state["command"], state["builder"]
    if builder:
        found = _command([command, "buildx", "inspect", builder])
        if found.returncode == 0:
            removed = _command([command, "buildx", "rm", "--force", builder])
            if removed.returncode:
                raise RuntimeError(removed.stderr or f"Cannot remove builder {builder}")
        elif "no builder" not in (found.stderr or "").lower():
            raise RuntimeError(found.stderr or "Cannot inspect owned builder")
        state["builder"] = None
        _write(root, state)
    if state["buildkit_owned"]:
        if _inspect(command, "image", BUILDKIT_IMAGE) is not None:
            removed = _command([command, "image", "rm", BUILDKIT_IMAGE])
            if removed.returncode:
                raise RuntimeError(removed.stderr or "Cannot release downloaded BuildKit image")
        state["buildkit_owned"] = False
        _write(root, state)


def release_owned_docker(root):
    """Remove only journaled resources; the caller owns the dependency lock."""
    path = workspace_path(root, JOURNAL, "Docker ownership journal")
    receipt = {"status": "released", "removed": [], "errors": []}
    if not path.exists():
        return receipt
    try:
        state = json.loads(path.read_text("utf-8"))
        _check_state(state)
        command = state["command"]
        available = _command([command, "info", "--format", "{{.ServerVersion}}"])
        if available.returncode:
            raise RuntimeError(available.stderr or "Docker engine unavailable during cleanup")
        for name in list(state["containers"]):
            container = _inspect(command, "container", name)
            if container is not None:
                if (container.get("Config", {}).get("Labels") or {}).get(LABEL) != state["owner"]:
                    raise ValueError("Docker container ownership label changed; refusing deletion")
                removed = _command([command, "container", "rm", "--force", name])
                if removed.returncode:
                    raise RuntimeError(removed.stderr or f"Cannot remove container {name}")
            state["containers"].remove(name)
            _write(root, state)
            receipt["removed"].append({"container": name})
        for directory in list(state["workspaces"]):
            for name in (".autorepro_deps", ".autorepro_tmp"):
                target = workspace_path(directory, name, "Docker workspace dependencies")
                if target.exists():
                    directory_bytes(target)
                    remove_tree(target)
                    receipt["removed"].append({"workspace_dependencies": str(target)})
            state["workspaces"].remove(directory)
            _write(root, state)
        if state["builder"] or state["buildkit_owned"]:
            builder = state["builder"]
            _release_builder(root, state)
            receipt["removed"].append({"builder": builder})
        if state["image"]:
            image = _inspect(command, "image", state["image"])
            if image is not None:
                if (image.get("Config", {}).get("Labels") or {}).get(LABEL) != state["owner"]:
                    raise ValueError("Docker image ownership label changed; refusing deletion")
                removed = _command([command, "image", "rm", state["image"]])
                if removed.returncode:
                    raise RuntimeError(removed.stderr or "Cannot remove owned image")
            receipt["removed"].append({"image": state["image"]})
            state["image"] = None
            _write(root, state)
        path.unlink()
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        receipt.update(status="error", errors=[{"path": JOURNAL, "reason": str(exc)}])
    return receipt


class DockerEnvironment:
    def __init__(self, root, command):
        self.root = Path(root)
        self.state = {"version": 1, "owner": uuid.uuid4().hex, "command": command,
                      "builder": None, "image": None, "containers": [], "workspaces": [],
                      "buildkit_owned": False}

    def register_workspace(self, directory):
        directory = str(Path(directory).absolute())
        if directory not in self.state["workspaces"]:
            self.state["workspaces"].append(directory)
            _write(self.root, self.state)

    def register_container(self, name):
        if name not in self.state["containers"]:
            self.state["containers"].append(name)
            _write(self.root, self.state)

    def build(self, context):
        """Build in a private builder so cache removal never prunes user builds."""
        command, owner = self.state["command"], self.state["owner"]
        if self.state["image"]:
            receipt = release_owned_docker(self.root)
            if receipt["errors"]:
                raise RuntimeError(str(receipt["errors"]))
            self.state.update(image=None, builder=None, containers=[], workspaces=[], buildkit_owned=False)
        # These are the two fixed tags created by earlier app versions. Drop
        # them before creating a new experiment, without pruning shared caches.
        for legacy in LEGACY_IMAGES:
            if _inspect(command, "image", legacy) is not None:
                removed = _command([command, "image", "rm", legacy])
                if removed.returncode:
                    raise RuntimeError(removed.stderr or "Cannot release legacy experiment image " + legacy)
        self.state.update(builder="autorepro-build-" + owner, image="autorepro-env:" + owner,
                          buildkit_owned=_inspect(command, "image", BUILDKIT_IMAGE) is None)
        _write(self.root, self.state)  # Persist ownership before daemon mutations.
        failure = None
        try:
            created = _command([command, "buildx", "create", "--name", self.state["builder"],
                                "--driver", "docker-container", "--driver-opt", "image=" + BUILDKIT_IMAGE])
            if created.returncode:
                raise RuntimeError(created.stderr or created.stdout or "Cannot create private Docker builder")
            return _command([command, "buildx", "build", "--builder", self.state["builder"],
                             "--load", "--tag", self.state["image"], "--label", LABEL + "=" + owner,
                             str(context)], timeout=1800)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                _release_builder(self.root, self.state)
            except Exception as exc:
                if failure is None:
                    raise
                failure.add_note("Docker build cache cleanup failed: " + str(exc))

    @property
    def image(self):
        return self.state["image"]

    @property
    def container_label(self):
        return LABEL + "=" + self.state["owner"]

    def ensure_image(self, image):
        command = self.state["command"]
        if self.state["builder"] or self.state["buildkit_owned"]:
            raise RuntimeError("Previous Docker build cache was not released; refusing another environment")
        if self.state["image"] and self.state["image"] != image:
            receipt = release_owned_docker(self.root)
            if receipt["errors"]:
                raise RuntimeError(str(receipt["errors"]))
            self.state.update(image=None, containers=[], workspaces=[])
        if _inspect(command, "image", image) is not None:
            return image
        # Export a missing base via the private builder; Docker run never pulls
        # an unjournaled image into the user's shared daemon cache.
        with tempfile.TemporaryDirectory(prefix="autorepro_docker_base_") as directory:
            Path(directory, "Dockerfile").write_text("FROM " + image + "\n", encoding="utf-8")
            result = self.build(directory)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout or "Cannot prepare disposable Docker image")
        return self.image


def disposable_docker(root, command):
    from src.environment_lifecycle import active_environment
    if root is None:
        from src.agents.code_executor import DEPS_CACHE_ROOT
        root = DEPS_CACHE_ROOT
    lease = active_environment(root)
    if lease is None:
        return None
    owned = getattr(lease, "docker", None)
    if owned is None:
        owned = lease.docker = DockerEnvironment(root, command)
    return owned
