"""Docker 沙箱加固参数测试（P1-⑪）。

覆盖：
1. 镜像白名单：允许官方/自建前缀；拒绝恶意/未知镜像（exit_code -5、
   sandbox.image_allowed=False，且不产生 subprocess 调用）；
2. 完整加固参数出现在 docker run 命令中：
   --cap-drop ALL / --security-opt no-new-privileges / --read-only /
   --tmpfs / --user（非 root）/ --cpus / --memory / --pids-limit；
3. 加固不兼容自动降级：unknown flag（如老 Docker 无 --pids-limit）->
   去掉资源限额降级；read-only file system -> 最小隔离（仅 cap-drop +
   no-new-privileges）重跑，记录 sandbox.degraded 级别；
4. 与加固无关的失败（ModuleNotFoundError）不降级，直接进入缺包自愈；
5. AUTOREPRO_DOCKER_HARDEN=0 时完全不加固（兼容极端环境）；
6. 加固开启时 pip 安装走 tmpfs（--target /tmp/site-packages +
   PYTHONPATH 注入），兼容只读 rootfs 与非 root 用户；
7. 白名单可通过 AUTOREPRO_DOCKER_IMAGE_ALLOWLIST 扩展。

运行: python -m pytest tests/test_sandbox_hardening.py -v
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
def _restore_hardening(monkeypatch):
    """每个用例显式恢复加固默认开启，避免用例间环境变量污染。"""
    monkeypatch.setattr(ce_mod, "DOCKER_HARDEN", True)
    monkeypatch.setattr(
        ce_mod, "DOCKER_IMAGE_ALLOWLIST",
        ["python:", "pytorch/", "autorepro", "nvidia/"])
    yield


def _executor() -> CodeExecutorAgent:
    executor = CodeExecutorAgent(LLMClient(mock_mode=True),
                                 mock_mode=True)
    executor.use_docker = True
    return executor


def _cmd_str(calls, index=-1) -> str:
    return " ".join(str(c) for c in calls[index])


# ---------------- 1. 镜像白名单 ----------------

class TestImageAllowlist:
    def test_allows_official_and_self_built(self, monkeypatch, tmp_path):
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim"}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert result["sandbox"]["image_allowed"] is True
        joined = _cmd_str(calls)
        assert "python:3.11-slim" in joined

    def test_allows_autorepro_custom_image(self, monkeypatch, tmp_path):
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "autorepro:latest",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert "autorepro:latest" in _cmd_str(calls)

    def test_rejects_unknown_image_without_running(self, monkeypatch,
                                                   tmp_path):
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            raise AssertionError("白名单拒绝后不应真的执行 docker run")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "evil/malicious:latest",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is False
        assert result["exit_code"] == -5
        assert result["sandbox"]["image_allowed"] is False
        assert "evil/malicious" in result["stderr"]
        assert calls == []  # 拒绝后绝不触网/绝不执行

    def test_allowlist_extendable_via_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ce_mod, "DOCKER_IMAGE_ALLOWLIST",
                            ["mycorp/"])
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "mycorp/ml:1.0"}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert "mycorp/ml:1.0" in _cmd_str(calls)


# ---------------- 2. 完整加固参数 ----------------

class TestHardeningArgs:
    def _capture(self, monkeypatch, tmp_path, image="python:3.11-slim"):
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": image, "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        return result, calls

    def test_full_hardening_present(self, monkeypatch, tmp_path):
        result, calls = self._capture(monkeypatch, tmp_path)
        assert result["success"] is True
        joined = _cmd_str(calls)
        assert "--cap-drop" in joined and "ALL" in joined
        assert "--security-opt" in joined and "no-new-privileges" in joined
        assert "--read-only" in joined
        assert "--tmpfs" in joined and "/tmp:rw" in joined
        assert "--user" in joined and "65534:65534" in joined
        assert "--cpus" in joined and "2.0" in joined
        assert "--memory" in joined and "2g" in joined
        assert "--pids-limit" in joined and "256" in joined
        # 顺序：run 参数位于 image 之前
        run_idx = joined.index("--rm")
        image_idx = joined.index("python:3.11-slim")
        assert run_idx < image_idx

    def test_sandbox_meta_reported(self, monkeypatch, tmp_path):
        result, _ = self._capture(monkeypatch, tmp_path)
        sandbox = result["sandbox"]
        assert sandbox == {"hardened": True, "level": 0,
                           "degraded": False,
                           "image_allowed": True}

    def test_pip_install_targets_tmpfs(self, monkeypatch, tmp_path):
        """加固开启：pip 安装走 tmpfs 目录 + PYTHONPATH 注入。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": "numpy>=1.24"}
        result = executor._execute_code_docker("import numpy\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        joined = _cmd_str(calls)
        assert "--target /tmp/site-packages" in joined
        assert "PYTHONPATH=/tmp/site-packages" in joined
        assert "-r /app/requirements.txt" in joined

    def test_hardening_disabled_no_extra_args(self, monkeypatch, tmp_path):
        """AUTOREPRO_DOCKER_HARDEN=0：完全不加固，pip 走系统路径。"""
        monkeypatch.setattr(ce_mod, "DOCKER_HARDEN", False)
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": "numpy>=1.24"}
        result = executor._execute_code_docker("import numpy\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert result["sandbox"]["hardened"] is False
        joined = _cmd_str(calls)
        assert "--cap-drop" not in joined
        assert "--read-only" not in joined
        assert "--user" not in joined
        assert "--pids-limit" not in joined
        assert "--target" not in joined
        assert "PYTHONPATH=/tmp/site-packages" not in joined


# ---------------- 3. 加固不兼容自动降级 ----------------

class TestHardeningDegrade:
    def test_degrade_on_unknown_flag_then_success(self, monkeypatch,
                                                  tmp_path):
        """老 Docker 不支持 --pids-limit：首轮 unknown flag -> 自动降级重跑。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            joined = " ".join(str(c) for c in cmd)
            if "--pids-limit" in joined:
                return subprocess.CompletedProcess(
                    cmd, 125,
                    stdout="", stderr="Error: unknown flag: --pids-limit")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert len(calls) == 2
        # 首轮含 pids-limit；重跑不带 pids-limit / cpus / memory
        assert "--pids-limit" in _cmd_str(calls, 0)
        second = _cmd_str(calls, 1)
        assert "--pids-limit" not in second
        assert "--cpus" not in second
        assert "--memory" not in second
        # 降级仍保留核心隔离：cap-drop + no-new-privileges + read-only
        assert "--cap-drop" in second and "ALL" in second
        assert "no-new-privileges" in second
        assert "--read-only" in second
        assert result["sandbox"] == {"hardened": True, "level": 1,
                                     "degraded": True,
                                     "image_allowed": True}

    def test_degrade_to_minimal_isolation(self, monkeypatch, tmp_path):
        """最坏场景：老 Docker + 只读 rootfs 不可用 -> 最小隔离（仅 cap-drop）。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            joined = " ".join(str(c) for c in cmd)
            if "--pids-limit" in joined:
                return subprocess.CompletedProcess(
                    cmd, 125, stdout="",
                    stderr="Error: unknown flag: --pids-limit")
            if "--read-only" in joined:
                return subprocess.CompletedProcess(
                    cmd, 126, stdout="",
                    stderr="docker: Error response from daemon: "
                           "read-only file system: unknown")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        assert len(calls) == 3
        third = _cmd_str(calls, 2)
        assert "--read-only" not in third
        assert "--user" not in third
        assert "--cpus" not in third
        # 最小隔离底线仍在
        assert "--cap-drop" in third and "ALL" in third
        assert "no-new-privileges" in third
        assert result["sandbox"] == {"hardened": True, "level": 2,
                                     "degraded": True,
                                     "image_allowed": True}

    def test_no_degrade_on_code_error(self, monkeypatch, tmp_path):
        """ModuleNotFoundError 与加固无关：不降级，进入缺包自愈。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            joined = " ".join(str(c) for c in cmd)
            if "pip install" in joined:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="ok", stderr="")
            return subprocess.CompletedProcess(
                cmd, 1, stdout="",
                stderr="ModuleNotFoundError: No module named 'cv2'")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("import cv2\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is True
        # 首轮缺模块（不降级，只降级加固不兼容）-> pip 自愈重跑
        assert len(calls) == 2
        first = _cmd_str(calls, 0)
        assert "--pids-limit" in first          # 首轮是完整加固
        assert result["sandbox"]["level"] == 0
        assert result["sandbox"]["degraded"] is False
        assert result["healed"][0]["package"] == "opencv-python"
        # 自愈轮 pip --target tmpfs（加固下 PYTHONPATH 注入）
        second = _cmd_str(calls, 1)
        assert "opencv-python" in second
        assert "PYTHONPATH=/tmp/site-packages" in second

    def test_unresolvable_hardening_failure_returns_last(self, monkeypatch,
                                                         tmp_path):
        """三级都失败且均为加固特征：返回最后一次结果，不无限重试。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 125, stdout="",
                stderr="Error: unknown flag: --bogus")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        assert result["success"] is False
        assert result["exit_code"] == 125
        assert len(calls) == 3                  # level0/1/2 各一次
        assert result["sandbox"]["level"] == 2


# ---------------- 4. 资源限额可配置 ----------------

class TestLimitsConfigurable:
    def test_custom_limits_from_module_constants(self, monkeypatch,
                                                 tmp_path):
        monkeypatch.setattr(ce_mod, "DOCKER_DEFAULT_CPUS", 4.0)
        monkeypatch.setattr(ce_mod, "DOCKER_DEFAULT_MEM", "4g")
        monkeypatch.setattr(ce_mod, "DOCKER_DEFAULT_PIDS", 512)
        monkeypatch.setattr(ce_mod, "DOCKER_DEFAULT_USER", "1000:1000")
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        executor._execute_code_docker("print(1)\n", "smoke",
                                      workdir=str(tmp_path))
        joined = _cmd_str(calls)
        assert "--cpus" in joined and "4.0" in joined
        assert "--memory" in joined and "4g" in joined
        assert "--pids-limit" in joined and "512" in joined
        assert "--user" in joined and "1000:1000" in joined