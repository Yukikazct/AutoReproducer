"""CodeExecutor 运行时缺模块 pip 自愈与隔离门测试（P1-⑧）。

覆盖：
1. real-mode code is blocked from local execution when Docker is not selected;
2. mock_mode local execution skips real installation;
3. Docker self-heal retains its dependency recovery behavior.

运行: python -m pytest tests/test_code_executor_self_heal.py -v
"""
import subprocess
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent,
    EXIT_ISOLATION_REQUIRED,
)
from src.base_agent import BaseAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """隔离 L0 依赖缓存：进程内缓存与磁盘 heal 目录不跨用例污染。"""
    ce_mod._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")
    # 引擎探测不依赖本机 Docker Desktop 状态（详见 test_sandbox_hardening）
    monkeypatch.setattr(BaseAgent, "docker_engine_available",
                        staticmethod(lambda *a, **k: (True, None)))
    # 镜像可用性探测/预拉同理：本文件的 fake_run 会把未知命令当成脚本执行
    # 来返回结果，一次 `docker images -q` 探测就能把自愈轮次的计数带偏。
    # 镜像可用性由 tests/test_docker_image_mirror.py 专门覆盖。
    monkeypatch.setattr(BaseAgent, "ensure_image_pulled",
                        staticmethod(lambda *a, **k: None))
    yield
    ce_mod._INSTALLED_DEPS.clear()
    executor = CodeExecutorAgent(LLMClient(mock_mode=False),
                                 mock_mode=False)
    executor._heal_dirs.clear()


def _executor(mock_mode: bool = False) -> CodeExecutorAgent:
    return CodeExecutorAgent(LLMClient(mock_mode=mock_mode),
                             mock_mode=mock_mode)


def _script_fail(returncode=1, stderr=""):
    return subprocess.CompletedProcess(
        ["python", "run.py"], returncode,
        stdout="", stderr=stderr or "ModuleNotFoundError: No module named 'cv2'\n")


def _ok(returncode=0, stdout="RUN_OK"):
    return subprocess.CompletedProcess(
        ["python", "run.py"], returncode, stdout=stdout, stderr="")


def _is_pip_cmd(cmd) -> bool:
    parts = [str(c) for c in cmd]
    return "-m" in parts and "pip" in parts and "install" in parts


def _is_script_cmd(cmd) -> bool:
    parts = [str(c) for c in cmd]
    return any(c == "run.py" or c.endswith("run.py") for c in parts)


# ---------------- 本地执行隔离门 ----------------

@pytest.mark.parametrize("code", [
    "print('would run on host')\n",
    "import importlib\n"
    "importlib.import_module('os').system('echo unsafe')\n",
])
def test_real_mode_local_execution_fails_closed(monkeypatch, code):
    """Neither normal dispatch nor direct local execution may reach subprocess."""
    executor = _executor(mock_mode=False)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("host execution must not be attempted")

    monkeypatch.setattr(ce_mod.subprocess, "run", fail_if_called)
    for execute in (
            lambda: executor._execute_code(code, stage="smoke"),
            lambda: executor._execute_code_local(code, stage="smoke")):
        result = execute()
        assert result["success"] is False
        assert result["isolation_required"] is True
        assert result["exit_code"] == EXIT_ISOLATION_REQUIRED


def test_local_self_heal_mock_skips_install(monkeypatch, tmp_path):
    """mock 模式：自愈短路成功（不真实安装），重跑脚本成功。"""
    counter = {"script": 0, "pip": 0}

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            counter["pip"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if _is_script_cmd(cmd):
            n = counter["script"]
            counter["script"] += 1
            if n == 0:
                return _script_fail(
                    stderr="ModuleNotFoundError: No module named 'cv2'")
            return _ok()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=True)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\nprint('ok')\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is True
    healed = result.get("healed")
    assert healed and healed[0]["ok"] is True
    # mock 模式不触网，不应出现真实 pip 调用
    assert counter["pip"] == 0


def test_local_self_heal_mock_repeat_no_install(monkeypatch, tmp_path):
    """mock 模式持续失败时也只短路，不无限安装（≤MAX 轮）。"""
    counter = {"pip": 0}

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            counter["pip"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if _is_script_cmd(cmd):
            return _script_fail(
                stderr="ModuleNotFoundError: No module named 'cv2'")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=True)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\nraise SystemExit(1)\n",
                                    stage="smoke", workdir=str(workdir))
    assert result["success"] is False
    assert counter["pip"] == 0            # mock 从不真实安装
    healed = result.get("healed") or []
    assert len(healed) <= ce_mod.MAX_PIP_SELF_HEAL


# ---------------- Docker 自愈 ----------------

def test_docker_self_heal_accumulates(monkeypatch, tmp_path):
    """基础镜像+空 requirements：首轮纯脚本；缺包累积进安装命令重跑。

    预算分离后每轮 = 独立安装 run + 独立脚本 run：缺包只可能出现在
    脚本阶段的 stderr 里（安装阶段成功，脚本阶段模拟依次缺 cv2/sklearn）。
    """
    calls: list = []
    script_calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            return _ok()  # 安装阶段成功
        script_calls.append(1)
        missing = {1: "cv2", 2: "sklearn"}.get(len(script_calls))
        if missing:
            return _script_fail(
                stderr=f"ModuleNotFoundError: No module named '{missing}'")
        return _ok()

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_docker)
    executor = _executor(mock_mode=False)
    executor.use_docker = True
    executor.env_config = {"image_tag": "python:3.11-slim",
                           "requirements_txt": ""}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\nimport sklearn\n",
                                    stage="smoke", workdir=str(workdir))
    assert result["success"] is True
    healed = result.get("healed")
    assert healed and len(healed) == 2
    packages = [h["package"] for h in healed]
    assert packages == ["opencv-python", "scikit-learn"]
    # 最后一次安装调用应同时累积两个包
    last_pip = next(" ".join(str(c) for c in cmd)
                    for cmd in reversed(calls)
                    if "pip install" in " ".join(str(c) for c in cmd))
    assert "opencv-python" in last_pip and "scikit-learn" in last_pip
    # 首轮是纯 python run.py（无 pip 前置）
    first = " ".join(str(c) for c in calls[0])
    assert "pip install" not in first


def test_docker_self_heal_with_reqs(monkeypatch, tmp_path):
    """基础镜像+requirements：安装独立预算；缺包再累积进安装命令。"""
    calls: list = []
    script_calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            return _ok()  # 安装阶段成功（含 -r requirements）
        script_calls.append(1)
        if len(script_calls) == 1:
            return _script_fail(
                stderr="ModuleNotFoundError: No module named 'cv2'")
        return _ok()

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_docker)
    executor = _executor(mock_mode=False)
    executor.use_docker = True
    executor.env_config = {"image_tag": "python:3.11-slim",
                           "requirements_txt": "numpy>=1.24"}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is True
    healed = result.get("healed")
    assert healed and healed[0]["package"] == "opencv-python"
    # 自愈轮安装命令同时携带 requirements 文件与自愈包
    heal_runner = next(c for c in calls
                       if "opencv-python" in " ".join(str(x) for x in c))
    joined = " ".join(str(x) for x in heal_runner)
    assert "-r /app/requirements.txt" in joined
    assert "opencv-python" in joined


def test_docker_custom_image_pip_heal_after_missing(monkeypatch, tmp_path):
    """自定义镜像：首轮无 pip 前置（镜像假定含依赖）；缺包后经 pip 自愈。"""
    calls: list = []
    script_calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            assert "opencv-python" in cmd_str
            return _ok()
        script_calls.append(1)
        if len(script_calls) == 1:
            return _script_fail(
                stderr="ModuleNotFoundError: No module named 'cv2'")
        return _ok()

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_docker)
    executor = _executor(mock_mode=False)
    executor.use_docker = True
    executor.env_config = {"image_tag": "autorepro:latest",
                           "requirements_txt": ""}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is True
    healed = result.get("healed")
    assert healed and healed[0]["package"] == "opencv-python"
    # 首轮是纯 python run.py
    first = " ".join(str(x) for x in calls[0])
    assert "pip install" not in first


def test_docker_self_heal_dedup(monkeypatch, tmp_path):
    """同一缺失模块去重：不重复累积，自愈风格为'pip 安装仍缺同模块'即停止。"""
    calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            # 模拟安装后运行脚本仍缺同一模块
            return _script_fail(
                stderr="ModuleNotFoundError: No module named 'cv2'")
        return _script_fail(
            stderr="ModuleNotFoundError: No module named 'cv2'")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_docker)
    executor = _executor(mock_mode=False)
    executor.use_docker = True
    executor.env_config = {"image_tag": "python:3.11-slim",
                           "requirements_txt": ""}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is False
    healed = result.get("healed")
    assert healed and len(healed) == 1
    assert healed[0]["module"] == "cv2"
