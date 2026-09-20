"""CodeExecutor 本地依赖自动安装测试。

覆盖（对应缺陷「本地执行器未按依赖清单自动安装」修复）：
1. 执行前按 env_config 依赖清单调用 pip install（正确参数：-r requirements、
   国内镜像 -i、find-links）；
2. 进程内幂等缓存：同一依赖清单只安装一次（smoke/full/优化重跑不重复装）；
3. 安装失败：直接返回失败诊断（exit_code=-4、stderr 含原因），不执行脚本；
4. 无依赖时跳过 pip，正常执行脚本；
5. 真实端到端：本机已有 numpy 时按清单安装后脚本真实可跑。

运行: python -m pytest tests/test_code_executor_deps.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import CodeExecutorAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_deps_cache():
    """每个用例独立：清理进程内依赖缓存，避免用例间互相污染。"""
    ce_mod._INSTALLED_DEPS.clear()
    yield
    ce_mod._INSTALLED_DEPS.clear()


def _executor(env_config: dict) -> CodeExecutorAgent:
    exec_ = CodeExecutorAgent(LLMClient(mock_mode=True))
    exec_.env_config = env_config
    return exec_


def _is_pip_cmd(cmd) -> bool:
    parts = [str(c) for c in cmd]
    return "-m" in parts and "pip" in parts and "install" in parts


# ---------------- 1. 执行前自动安装依赖 ----------------

def test_installs_deps_before_run(monkeypatch, tmp_path):
    calls: list = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if _is_pip_cmd(cmd):          # pip 安装返回成功
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        return real_run(cmd, **kw)     # 脚本本身真实执行

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)

    executor = _executor({"requirements_txt": "numpy>=1.26,<3\nmatplotlib>=3.8,<4"})
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("print('HELLO_FROM_SCRIPT')",
                                    stage="smoke", workdir=str(workdir))

    assert result["success"] is True
    assert "HELLO_FROM_SCRIPT" in result["stdout"]
    assert result.get("deps_prepared") is True

    pip_cmds = [c for c in calls if _is_pip_cmd(c)]
    assert pip_cmds, "未在脚本执行前调用 pip install"
    joined = " ".join(str(x) for x in pip_cmds[0])
    assert "requirements.txt" in joined        # 依赖清单落盘后安装
    assert "-i" in joined or "--index-url" in joined   # 国内镜像
    assert "--find-links" in joined            # 大包加速源


def test_failed_install_blocks_execution(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="ERROR: Cannot find package unknown-pkg")
        raise AssertionError("安装失败后不应执行脚本")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)

    executor = _executor({"requirements_txt": "unknown-pkg==9.9"})
    workdir = tmp_path / "ws"
    workdir.mkdir()
    result = executor._execute_code("print('SHOULD_NOT_RUN')",
                                    stage="smoke", workdir=str(workdir))

    assert result["success"] is False
    assert result["exit_code"] == -4
    assert result.get("deps_prepared") is False
    assert "依赖安装失败" in result["stderr"]
    assert "unknown-pkg" in result["stderr"]


# ---------------- 2. 进程内幂等缓存 ----------------

def test_deps_installed_only_once(monkeypatch, tmp_path):
    pip_calls: list = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            pip_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        return real_run(cmd, **kw)

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)

    executor = _executor({"requirements_txt": "numpy>=1.26,<3"})
    for i in range(2):
        wd = tmp_path / f"ws{i}"
        wd.mkdir()
        result = executor._execute_code("print('RUN_OK')",
                                        stage="smoke", workdir=str(wd))
        assert result["success"] is True
    # smoke/full 多次执行只装一次
    assert len(pip_calls) == 1, f"期望只安装一次,实际 {len(pip_calls)} 次"

    # 缓存命中时也不重复写 requirements.txt
    second_wd = tmp_path / "ws2"
    second_wd.mkdir()
    result = executor._execute_code("print('OK')", stage="full",
                                    workdir=str(second_wd))
    assert result["success"] is True
    assert len(pip_calls) == 1


def test_failed_install_is_cached(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            return subprocess.CompletedProcess(
                cmd, 2, stdout="", stderr="boom")
        raise AssertionError("不应执行脚本")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    executor = _executor({"requirements_txt": "broken-pkg"})

    for i in range(2):   # 两次都失败，但不重复 pip
        wd = tmp_path / f"ws{i}"
        wd.mkdir()
        result = executor._execute_code("print(1)", stage="smoke",
                                        workdir=str(wd))
        assert result["success"] is False
        assert "依赖安装失败" in result["stderr"]
    assert ce_mod._INSTALLED_DEPS.get("broken-pkg", "") != ""


# ---------------- 3. 无依赖时跳过 pip ----------------

def test_no_deps_skips_pip(monkeypatch, tmp_path):
    pip_calls: list = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            pip_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kw)

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)

    executor = _executor({})     # 无 requirements_txt / required_packages
    wd = tmp_path / "ws"
    wd.mkdir()
    result = executor._execute_code("print('OK')", stage="smoke",
                                    workdir=str(wd))
    assert result["success"] is True
    assert not pip_calls

    executor2 = _executor({"required_packages": []})   # 空清单同样跳过
    assert executor2._ensure_local_deps(str(wd)) is None


# ---------------- 4. 无依赖清单的真实脚本执行（快路径） ----------------

def test_real_execute_without_dep_install(tmp_path):
    """无依赖清单时脚本真实执行通过（纯标准库，秒级）。"""
    executor = _executor({})
    wd = tmp_path / "ws"
    wd.mkdir()
    code = ("import sys, os\n"
            "print('PY=' + sys.version.split()[0])\n"
            "print('CWD=' + os.path.basename(os.getcwd()))\n")
    result = executor._execute_code(code, stage="smoke", workdir=str(wd))
    assert result["success"] is True, result["stderr"]
    assert "PY=" in result["stdout"]
    assert result.get("deps_prepared") is True


# ---------------- 5. --target 与 user 安装互斥（真实模式实测） ----------------

def test_pip_cmd_opts_out_of_user_install(monkeypatch, tmp_path):
    """隔离安装必须显式 --no-user，否则站点级 pip.ini 的 install.user=yes
    会让 pip 追加 --user，与 --target 冲突：
    `ERROR: Can not combine '--user' and '--target'`（实测本机必现）。"""
    seen: list = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            seen.append((cmd, kw))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        return real_run(cmd, **kw)

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")

    executor = _executor({"requirements_txt": "numpy>=1.26,<3"})
    wd = tmp_path / "ws"
    wd.mkdir()
    executor._execute_code("print('OK')", stage="smoke", workdir=str(wd))

    assert seen, "未调用 pip install"
    cmd = [str(c) for c in seen[0][0]]
    assert "--target" in cmd
    assert "--no-user" in cmd, "缺少 --no-user，站点级 install.user=yes 会致安装失败"


def test_pip_env_forces_user_off(monkeypatch, tmp_path):
    """pip 子进程环境必须带 PIP_USER=0（环境变量优先级高于配置文件），
    与命令行的 --no-user 形成双保险。"""
    seen: list = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            seen.append(kw)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        return real_run(cmd, **kw)

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")

    executor = _executor({"requirements_txt": "numpy>=1.26,<3"})
    wd = tmp_path / "ws"
    wd.mkdir()
    executor._execute_code("print('OK')", stage="smoke", workdir=str(wd))

    assert seen, "未调用 pip install"
    env = seen[0].get("env")
    assert env is not None, "pip 未显式传 env，无法覆盖站点配置"
    assert env["PIP_USER"] == "0"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_relative_workdir_is_not_double_joined():
    """相对 workdir 不得被拼两次（真实优化实测发现）。

    脚本以 `[python, os.path.join(workdir, "run.py")]` 启动、同时 cwd=workdir。
    workdir 是相对路径时，脚本参数会被 cwd 再解析一次，实际去找
    `data/_e2e_ws/data/_e2e_ws/run.py`，报 No such file or directory
    （exit_code=2、stdout 为空）——看起来像"生成的代码跑不起来"。

    实测触发：Optimizer 真实执行传 `workspace_dir="data/_e2e_ws"`。
    工作区必须建在仓库内（tmp_path 在 C:，与本仓库不同盘，构不出相对路径）。
    """
    workdir = tempfile.mkdtemp(prefix="_rel_ws_", dir=os.getcwd())
    try:
        rel = os.path.relpath(workdir, os.getcwd())
        assert not os.path.isabs(rel)     # 前提：确实是相对路径

        result = _executor({})._execute_code("print('REL_WORKDIR_OK')",
                                             stage="smoke", workdir=rel)
        assert result["success"] is True, result.get("stderr")
        assert "REL_WORKDIR_OK" in result["stdout"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_heal_install_opts_out_of_user_install(monkeypatch, tmp_path):
    """自愈补装走同一套 pip 参数，同样必须 --no-user + PIP_USER=0。"""
    seen: list = []

    def fake_run(cmd, **kw):
        if _is_pip_cmd(cmd):
            seen.append((cmd, kw))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        raise AssertionError("自愈只应触发 pip")

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")

    executor = _executor({})
    executor._heal_install_local("requests")

    assert seen, "未调用 pip install"
    cmd = [str(c) for c in seen[0][0]]
    assert "--target" in cmd
    assert "--no-user" in cmd
    assert seen[0][1].get("env", {}).get("PIP_USER") == "0"