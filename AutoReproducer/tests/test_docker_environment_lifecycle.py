"""Owned Docker images and private build caches obey the run lease."""
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

import src.agents.code_executor as executor_module
from src import docker_environment as docker
from src.environment_lifecycle import managed_environment, EnvironmentCleanupError


class DockerDaemon:
    def __init__(self):
        self.images = {"user/image:latest": {"Config": {"Labels": {}}}}
        self.builders, self.containers, self.calls = set(), {}, []
        self.fail_build, self.fail_remove = False, False

    def __call__(self, argv, timeout=120):
        self.calls.append(argv)
        args = argv[1:]
        code, out, err = 0, "", ""
        if args[0] == "info":
            out = "29.0"
        elif args[:2] == ["image", "inspect"]:
            if args[2] in self.images:
                out = json.dumps([self.images[args[2]]])
            else:
                code, err = 1, "No such image"
        elif args[:2] == ["image", "rm"]:
            if self.fail_remove:
                code, err = 1, "image still in use"
            else:
                self.images.pop(args[2], None)
        elif args[:2] == ["container", "inspect"]:
            if args[2] in self.containers:
                out = json.dumps([{"Config": {"Labels": {docker.LABEL: self.containers[args[2]]}}}])
            else:
                code, err = 1, "No such container"
        elif args[:2] == ["container", "rm"]:
            self.containers.pop(args[-1])
        elif args[:2] == ["buildx", "create"]:
            self.builders.add(args[args.index("--name") + 1])
        elif args[:2] == ["buildx", "inspect"]:
            if args[2] not in self.builders:
                code, err = 1, "no builder found"
        elif args[:2] == ["buildx", "rm"]:
            self.builders.remove(args[-1])
        elif args[:2] == ["buildx", "build"]:
            self.images[docker.BUILDKIT_IMAGE] = {"Config": {"Labels": {}}}
            if self.fail_build:
                code, err = 1, "build failed after downloading dependencies"
            else:
                image = args[args.index("--tag") + 1]
                label = args[args.index("--label") + 1].split("=", 1)[1]
                self.images[image] = {"Config": {"Labels": {docker.LABEL: label}}}
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    root = tmp_path / "deps"
    monkeypatch.setattr(executor_module, "DEPS_CACHE_ROOT", root)
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(tmp_path / "data"))
    from src import runtime_preparation
    monkeypatch.setattr(runtime_preparation, "release_runtime_downloads", lambda: {
        "status": "released", "removed": [], "errors": []})
    instance = DockerDaemon()
    monkeypatch.setattr(docker, "_command", instance)
    return instance, root


def test_private_build_cache_is_removed_before_execution_and_image_after_report(daemon, tmp_path):
    server, root = daemon
    for _ in range(2):
        with managed_environment() as lease:
            owned = docker.disposable_docker(root, "docker")
            result = owned.build(tmp_path)
            assert result.returncode == 0
            assert not server.builders and docker.BUILDKIT_IMAGE not in server.images
            assert len(server.images) == 2 and owned.image in server.images
            container = "autorepro-" + "a" * 32
            owned.register_container(container)
            server.containers[container] = owned.state["owner"]
            report = tmp_path / "report.md"
            report.write_text("finished", encoding="utf-8")
            lease.result = {"state": "COMPLETED", "data": {"report": "finished"}}
        assert server.images.keys() == {"user/image:latest"}
        assert not server.containers and not (root / docker.JOURNAL).exists()
        assert report.read_text("utf-8") == "finished"
        assert lease.result["data"]["environment_cleanup"]["docker"]["status"] == "released"


def test_build_failure_releases_partial_downloads_and_private_builder(daemon, tmp_path):
    server, root = daemon
    server.fail_build = True
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        assert owned.build(tmp_path).returncode == 1
        assert not server.builders and docker.BUILDKIT_IMAGE not in server.images
    assert server.images.keys() == {"user/image:latest"}


def test_cleanup_failure_blocks_the_next_environment(daemon, tmp_path):
    server, root = daemon
    with managed_environment() as lease:
        owned = docker.disposable_docker(root, "docker")
        assert owned.build(tmp_path).returncode == 0
        server.fail_remove = True
        lease.result = {"state": "COMPLETED", "data": {}}
    assert lease.result["state"] == "ERROR" and (root / docker.JOURNAL).is_file()
    with pytest.raises(EnvironmentCleanupError):
        with managed_environment():
            pytest.fail("must release old Docker dependencies first")


def test_external_image_is_reused_and_preserved(daemon):
    server, root = daemon
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        assert owned.ensure_image("user/image:latest") == "user/image:latest"
    assert server.images.keys() == {"user/image:latest"}
    assert not any(args[1:3] == ["buildx", "create"] for args in server.calls)


def test_missing_base_is_exported_without_an_unowned_pull(daemon):
    server, root = daemon
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        image = owned.ensure_image("python:3.11-slim")
        assert image.startswith("autorepro-env:")
        assert "python:3.11-slim" not in server.images
    assert server.images.keys() == {"user/image:latest"}
    assert not any(args[1] == "pull" for args in server.calls)


def test_journal_recovers_resources_after_an_owner_disappears(daemon, tmp_path):
    server, root = daemon
    root.mkdir()
    owned = docker.DockerEnvironment(root, "docker")
    assert owned.build(tmp_path).returncode == 0
    name = "autorepro-" + "b" * 32
    owned.register_container(name)
    server.containers[name] = owned.state["owner"]
    with managed_environment():
        assert server.images.keys() == {"user/image:latest"}
        assert not server.containers


def test_changed_image_ownership_is_preserved_and_reported(daemon, tmp_path):
    server, root = daemon
    with managed_environment() as lease:
        owned = docker.disposable_docker(root, "docker")
        assert owned.build(tmp_path).returncode == 0
        server.images[owned.image]["Config"]["Labels"] = {}
        lease.result = {"state": "COMPLETED", "data": {}}
    assert lease.result["state"] == "ERROR" and owned.image in server.images


def test_env_builder_uses_a_single_disposable_image_without_a_shared_base(daemon, tmp_path, monkeypatch):
    from src.agents.env_builder import EnvBuilderAgent
    from src.base_agent import BaseAgent
    server, root = daemon
    monkeypatch.setattr(BaseAgent, "docker_engine_available", lambda *a, **kw: (True, "ready"))
    builder = EnvBuilderAgent(Mock(), logger=Mock())
    monkeypatch.setattr(builder, "_resolve_docker_cmd", lambda: "docker")
    builder.ensure_base_image = Mock(side_effect=AssertionError("shared experiment base must not be retained"))
    with managed_environment():
        result = builder.build_image({"dockerfile": "FROM python:3.11-slim\n"})
        assert result["success"] and result["tag"].startswith("autorepro-env:")
        assert not server.builders and len(server.images) == 2
    assert server.images.keys() == {"user/image:latest"}


def test_persistent_workspace_keeps_report_but_releases_mounted_packages(daemon, tmp_path):
    server, root = daemon
    work = tmp_path / "persistent-workspace"
    work.mkdir()
    report = work / "report.md"
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        owned.register_workspace(work)
        for name in (".autorepro_deps", ".autorepro_tmp"):
            (work / name).mkdir()
            (work / name / "package.py").write_text("VALUE = 42", encoding="utf-8")
        report.write_text("finished report", encoding="utf-8")
    assert report.read_text("utf-8") == "finished report"
    assert not (work / ".autorepro_deps").exists() and not (work / ".autorepro_tmp").exists()


def test_legacy_fixed_images_are_removed_before_a_new_environment_is_built(daemon, tmp_path):
    server, root = daemon
    for image in docker.LEGACY_IMAGES:
        server.images[image] = {"Config": {"Labels": {}}}
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        assert owned.build(tmp_path).returncode == 0
        assert not any(image in server.images for image in docker.LEGACY_IMAGES)
        assert server.images.keys() == {"user/image:latest", owned.image}
    assert server.images.keys() == {"user/image:latest"}


def test_switching_to_an_existing_image_releases_the_previous_owned_environment(daemon, tmp_path):
    server, root = daemon
    with managed_environment():
        owned = docker.disposable_docker(root, "docker")
        assert owned.build(tmp_path).returncode == 0
        previous = owned.image
        assert owned.ensure_image("user/image:latest") == "user/image:latest"
        assert previous not in server.images
    assert server.images.keys() == {"user/image:latest"}


def test_unreleased_build_cache_blocks_fallback_to_another_image(daemon, tmp_path, monkeypatch):
    server, root = daemon
    with managed_environment() as lease:
        owned = docker.disposable_docker(root, "docker")
        with monkeypatch.context() as patch:
            patch.setattr(docker, "_release_builder", Mock(side_effect=RuntimeError("cache still in use")))
            with pytest.raises(RuntimeError, match="cache still in use"):
                owned.build(tmp_path)
            with pytest.raises(RuntimeError, match="refusing another environment"):
                owned.ensure_image("user/image:latest")
        lease.result = {"state": "ERROR", "data": {}}
    assert server.images.keys() == {"user/image:latest"}


def test_a_replaced_container_is_preserved_and_blocks_cleanup(daemon):
    server, root = daemon
    with managed_environment() as lease:
        owned = docker.disposable_docker(root, "docker")
        name = "autorepro-" + "c" * 32
        owned.register_container(name)
        server.containers[name] = "different-owner"
        lease.result = {"state": "COMPLETED", "data": {}}
    assert lease.result["state"] == "ERROR" and name in server.containers
