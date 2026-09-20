"""EnvBuilder 共享底座镜像测试（P1-1）。

覆盖：
1. 底座 Dockerfile 内容（python:3.11-slim + CPU torch + numpy + tqdm +
   国内源注入 + build-essential + WORKDIR）；
2. build_base_image：无 Docker 时诚实报错（不伪造成功）；
3. ensure_base_image：镜像已存在直接复用（cached=True，不重复构建）；
   已存在检测失败/缺失时构建（mock docker build 成功）；
4. build_image(use_base_image=True)：底座就绪时论文 Dockerfile 的
   FROM python:* 替换为 autorepro-base；底座不可用时降级原 Dockerfile
   并带 degraded 标注，不阻断构建；
5. _swap_to_base_image 边界：已是底座不改、无 FROM 不改、非 python 基础
   镜像不改；
6. 既有空 Dockerfile 短路（Mock 模式不依赖 Docker）保持兼容。

运行: python -m pytest tests/test_env_builder_base.py -v
"""
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.env_builder import (  # noqa: E402
    BASE_IMAGE_FROM, BASE_IMAGE_TAG, EnvBuilderAgent, PIP_FIND_LINKS,
    PIP_INDEX_URL)
from src.llm.llm_client import LLMClient  # noqa: E402


def _agent():
    return EnvBuilderAgent(LLMClient(mock_mode=True))


def _proc(returncode=0, stdout="ok", stderr=""):
    return type("P", (), {"returncode": returncode,
                          "stdout": stdout, "stderr": stderr})()


class _NoDocker:
    """模拟无 Docker 环境：_resolve_docker_cmd 返回 None。"""

    def __init__(self, agent):
        self._agent = agent

    def _resolve_docker_cmd(self):
        return None


# ---------------- 1. 底座 Dockerfile ----------------

def test_base_dockerfile_content():
    dockerfile = EnvBuilderAgent._base_dockerfile()
    assert dockerfile.startswith(f"FROM {BASE_IMAGE_FROM}\n")
    assert PIP_INDEX_URL in dockerfile
    assert PIP_FIND_LINKS in dockerfile
    for pkg in ("torch", "torchvision", "numpy", "tqdm"):
        assert pkg in dockerfile
    assert "build-essential" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "--no-cache-dir" in dockerfile


def test_build_base_image_without_docker(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "_resolve_docker_cmd",
                        lambda: None)
    res = agent.build_base_image()
    assert res["success"] is False
    assert "Docker" in res["error"] or "docker" in res["error"]


def test_build_base_image_success(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: "docker")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _proc()

    monkeypatch.setattr("src.agents.env_builder.subprocess.run", fake_run)
    res = agent.build_base_image()
    assert res["success"] is True
    assert res["tag"] == BASE_IMAGE_TAG
    assert any("build" in str(c) for c in calls)
    assert any("-t" in str(c) and BASE_IMAGE_TAG in " ".join(map(str, c))
               for c in calls)


# ---------------- 2. ensure_base_image ----------------

def test_ensure_base_image_cached(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: "docker")
    build_calls = []

    def fake_run(cmd, **kw):
        if "images" in cmd:
            return _proc(stdout=BASE_IMAGE_TAG)     # 已存在
        build_calls.append(cmd)
        return _proc()

    monkeypatch.setattr("src.agents.env_builder.subprocess.run", fake_run)
    res = agent.ensure_base_image("docker")
    assert res["success"] is True
    assert res["cached"] is True
    assert not build_calls, "镜像已存在不应触发构建"


def test_ensure_base_image_builds_when_missing(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: "docker")

    def fake_run(cmd, **kw):
        if "images" in cmd:
            return _proc(stdout="")                  # 不存在
        return _proc()                               # docker build 成功

    monkeypatch.setattr("src.agents.env_builder.subprocess.run", fake_run)
    res = agent.ensure_base_image("docker")
    assert res["success"] is True
    assert res["cached"] is False


def test_ensure_base_image_build_failure_honest(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: "docker")

    def fake_run(cmd, **kw):
        if "images" in cmd:
            return _proc(stdout="")
        return _proc(returncode=1, stderr="network timeout")

    monkeypatch.setattr("src.agents.env_builder.subprocess.run", fake_run)
    res = agent.ensure_base_image("docker")
    assert res["success"] is False
    assert res["cached"] is False


# ---------------- 3. build_image 底座接入 ----------------

def _recorder(agent, monkeypatch, base_ok):
    """记录传入 _build_dockerfile 的 Dockerfile；底座可用性由 base_ok 控制。"""
    captured = {}

    def fake_ensure(docker_cmd):
        if base_ok:
            return {"success": True, "tag": BASE_IMAGE_TAG, "cached": True}
        return {"success": False, "tag": BASE_IMAGE_TAG, "cached": False,
                "error": "torch download failed (network)"}

    def fake_build(dockerfile, tag, reqs=""):
        captured["dockerfile"] = dockerfile
        captured["reqs"] = reqs
        return {"success": True, "tag": tag}

    monkeypatch.setattr(agent, "ensure_base_image", fake_ensure)
    monkeypatch.setattr(agent, "_build_dockerfile", fake_build)
    return captured


def test_build_image_uses_base_when_ready(monkeypatch):
    agent = _agent()
    captured = _recorder(agent, monkeypatch, base_ok=True)
    env_config = {
        "dockerfile": "FROM python:3.11-slim\nWORKDIR /app\n"
                      "COPY requirements.txt .\n"
                      "RUN pip install -r requirements.txt\n",
        "requirements_txt": "scikit-learn>=1.2\n",
    }
    res = agent.build_image(env_config, tag="autorepro-env")
    assert res["success"] is True
    assert res["degraded"] is None
    df = captured["dockerfile"]
    assert df.startswith(f"FROM {BASE_IMAGE_TAG}")
    assert "COPY requirements.txt ." in df
    assert "PIP_INDEX_URL" in df            # 国内源注入仍然保留


def test_build_image_degrades_when_base_unavailable(monkeypatch):
    agent = _agent()
    captured = _recorder(agent, monkeypatch, base_ok=False)
    dockerfile_in = ("FROM python:3.11-slim\nWORKDIR /app\n"
                     "RUN pip install -r requirements.txt\n")
    res = agent.build_image({"dockerfile": dockerfile_in,
                             "requirements_txt": "numpy"},
                            tag="autorepro-env")
    # 底座不可用 -> 降级为原 dockerfile 构建，成功但带 degraded 标注
    assert res["success"] is True
    assert res["degraded"] is not None
    assert "torch" in res["degraded"]
    df = captured["dockerfile"]
    assert df.startswith("FROM python:3.11-slim")    # 未替换
    assert "PIP_INDEX_URL" in df                     # 源注入仍在


# ---------------- 4. _swap_to_base_image 边界 ----------------

def test_swap_to_base_basic():
    out = EnvBuilderAgent._swap_to_base_image(
        "FROM python:3.11-slim\nWORKDIR /app\n")
    assert out.startswith(f"FROM {BASE_IMAGE_TAG}\n")
    assert "WORKDIR /app" in out


def test_swap_keeps_own_base():
    df = f"FROM {BASE_IMAGE_TAG}\nRUN echo ok\n"
    assert EnvBuilderAgent._swap_to_base_image(df) == df


def test_swap_non_python_base_untouched():
    df = "FROM ubuntu:22.04\nRUN apt-get update\n"
    assert EnvBuilderAgent._swap_to_base_image(df) == df


def test_swap_no_from_untouched():
    df = "WORKDIR /app\nCMD python run.py\n"
    assert EnvBuilderAgent._swap_to_base_image(df) == df


# ---------------- 5. 既有兼容 ----------------

def test_docker_not_required_for_mock():
    """Mock 模式不依赖 Docker；空 Dockerfile 仍短路报错。"""
    agent = _agent()
    assert agent.build_image({"dockerfile": ""})["success"] is False
    # doctype: 无 Docker 时同样诚实报错
    assert agent.build_image(
        {"dockerfile": "FROM python:3.11-slim\n"})["success"] is False