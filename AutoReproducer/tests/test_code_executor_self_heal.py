"""CodeExecutor 运行时缺模块 pip 自愈测试（P1-⑧）。

覆盖：
1. 本地自愈成功：脚本缺 cv2 -> 隔离安装 opencv-python -> 重跑成功，
   result[\"healed\"] 记录模块/包名，安装命令正确（--target heal 目录 + 镜像）；
2. 本地安装失败：自愈安装失败返回诊断，不继续重跑；
3. 本地自愈上限：多次缺不同模块，最多 MAX_PIP_SELF_HEAL 轮；
4. 非缺模块错误（如 ValueError）不触发自愈；
5. mock_mode 不真实安装（不触网），自愈逻辑仍短路成功；
6. 成功路径不产生 healed 记录；
7. Docker 自愈：基础镜像+空 requirements 首轮纯脚本，缺包累积进 runner 重跑；
8. Docker 自愈：基础镜像+requirements 首轮 pip(rqs)，缺包再累积；
9. Docker 自愈：自定义镜像首轮无 pip 前置，缺包后改走 pip 自愈；
10. Docker 自愈去重：同一缺失模块不重复累积。

运行: python -m pytest tests/test_code_executor_self_heal.py -v
"""
import subprocess
import sys
from pathlib import Path

import pytest

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import CodeExecutorAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """隔离 L0 依赖缓存：进程内缓存与磁盘 heal 目录不跨用例污染。"""
    ce_mod._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")
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


# ---------------- 本地自愈 ----------------

def test_local_self_heal_success(monkeypatch, tmp_path):
    """缺 cv2 -> 隔离安装 opencv-python -> 重跑成功，healed 记录 1 次。"""
    calls: list = []
    script_runs = {"n": 0}

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if _is_pip_cmd(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if _is_script_cmd(cmd):
            # 模拟真实脚本：第一次缺模块，重跑成功
            script_runs["n"] += 1
            if script_runs["n"] == 1:
                return _script_fail(
                    stderr="ModuleNotFoundError: No module named 'cv2'")
            return _ok()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=False)
    executor.env_config = {}  # 无预装依赖，直接触发运行时自愈
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import cv2\nprint('ok')",
                                    stage="smoke", workdir=str(workdir))

    assert result["success"] is True
    healed = result.get("healed")
    assert healed and len(healed) == 1
    assert healed[0]["module"] == "cv2"
    assert healed[0]["package"] == "opencv-python"
    assert healed[0]["ok"] is True

    # 安装命令：隔离目录 + 国内镜像 + --target
    pip_cmds = [c for c in calls if _is_pip_cmd(c)]
    joined = " ".join(str(x) for x in pip_cmds[0])
    assert "pip" in joined and "install" in joined
    assert "--target" in joined
    assert "opencv-python" in joined
    # 自愈目录落盘 + ready 标记
    heal_dir = ce_mod.DEPS_CACHE_ROOT / "heal-cv2"
    assert (heal_dir / ce_mod._DEPS_READY_MARK).is_file()


def test_local_self_heal_no_heal_on_success(monkeypatch, tmp_path):
    """脚本直接成功：不产生 healed 记录，不调用 pip install。"""
    calls: list = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if _is_script_cmd(cmd):
            return _ok()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=False)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("print('hello')", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is True
    assert "healed" not in result
    assert not [c for c in calls if _is_pip_cmd(c)]


def test_local_self_heal_install_failure(monkeypatch, tmp_path):
    """自愈安装失败：返回诊断，不重跑脚本。"""
    calls: list = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if _is_pip_cmd(cmd):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="ERROR: No matching distribution")
        if _is_script_cmd(cmd):
            return _script_fail()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=False)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import missing_pkg\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is False
    healed = result.get("healed")
    assert healed and healed[0]["ok"] is False
    assert "No matching distribution" in (healed[0]["error"] or "")
    # 脚本只跑了一轮（安装失败后不再重跑）
    script_cmds = [c for c in calls if _is_script_cmd(c)]
    assert len(script_cmds) == 1


def test_local_self_heal_max_rounds(monkeypatch, tmp_path):
    """连续缺不同模块：最多自愈 MAX_PIP_SELF_HEAL 轮后失败。"""
    missing = ["cv2", "sklearn", "pandas", "torch"]
    counter = {"script": 0, "pip": 0}

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            counter["pip"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if _is_script_cmd(cmd):
            n = counter["script"]
            counter["script"] += 1
            module = missing[min(n, len(missing) - 1)]
            return _script_fail(
                stderr=f"ModuleNotFoundError: No module named '{module}'")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=False)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("import torch\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is False
    healed = result.get("healed") or []
    # 安装次数 ≤ MAX_PIP_SELF_HEAL（即使脚本每次都缺新模块）
    assert len(healed) == ce_mod.MAX_PIP_SELF_HEAL
    assert counter["pip"] == ce_mod.MAX_PIP_SELF_HEAL


def test_local_no_self_heal_on_non_module_error(monkeypatch, tmp_path):
    """非缺模块错误（ValueError）不触发自愈安装。"""
    calls: list = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if _is_script_cmd(cmd):
            return _script_fail(stderr="ValueError: bad value")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor(mock_mode=False)
    executor.env_config = {}
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("x = int('abc')\n", stage="smoke",
                                    workdir=str(workdir))
    assert result["success"] is False
    assert "healed" not in result
    assert not [c for c in calls if _is_pip_cmd(c)]


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
    """基础镜像+空 requirements：首轮纯脚本；缺包累积进 runner 重跑。"""
    calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            has_cv2 = "opencv-python" in cmd_str
            has_sk = "scikit-learn" in cmd_str
            if has_cv2 and not has_sk:
                return _script_fail(
                    stderr="ModuleNotFoundError: No module named 'sklearn'")
            return _ok()
        # 首轮 python run.py（无 pip 前置）
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
    result = executor._execute_code("import cv2\nimport sklearn\n",
                                    stage="smoke", workdir=str(workdir))
    assert result["success"] is True
    healed = result.get("healed")
    assert healed and len(healed) == 2
    packages = [h["package"] for h in healed]
    assert packages == ["opencv-python", "scikit-learn"]
    # 最后一次 docker 调用应同时累积两个包
    last = " ".join(str(c) for c in calls[-1])
    assert "opencv-python" in last and "scikit-learn" in last
    # 首轮是纯 python run.py（无 pip 前置）
    first = " ".join(str(c) for c in calls[0])
    assert "pip install" not in first


def test_docker_self_heal_with_reqs(monkeypatch, tmp_path):
    """基础镜像+requirements：首轮 pip(rqs)；缺包再累积进同一命令。"""
    calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            if ("-r /app/requirements.txt" in cmd_str
                    and "opencv-python" not in cmd_str):
                # 首轮含 requirements，仍缺 cv2
                return _script_fail(
                    stderr="ModuleNotFoundError: No module named 'cv2'")
            return _ok()
        return _script_fail()

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
    # 自愈 runner 同时携带 requirements 文件与自愈包
    heal_runner = next(c for c in calls
                       if "opencv-python" in " ".join(str(x) for x in c))
    joined = " ".join(str(x) for x in heal_runner)
    assert "-r /app/requirements.txt" in joined
    assert "opencv-python" in joined


def test_docker_custom_image_pip_heal_after_missing(monkeypatch, tmp_path):
    """自定义镜像：首轮无 pip 前置（镜像假定含依赖）；缺包后经 pip 自愈。"""
    calls: list = []

    def fake_docker(cmd, **kw):
        calls.append(cmd)
        cmd_str = " ".join(str(c) for c in cmd)
        if "pip install" in cmd_str:
            assert "opencv-python" in cmd_str
            return _ok()
        return _script_fail(
            stderr="ModuleNotFoundError: No module named 'cv2'")

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
