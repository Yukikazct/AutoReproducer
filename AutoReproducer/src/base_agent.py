"""Agent基类 - 所有 Agent 的抽象基类"""
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from src.audit.audit_logger import AuditLogger

# Docker Desktop / 常见安装路径（Windows 上 docker.exe 常不在系统 PATH 中）
_DOCKER_CANDIDATES = [
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\Program Files\Docker\Docker\resources\bin\docker-cli.exe",
    r"/usr/local/bin/docker",
    r"/usr/bin/docker",
]


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    每个 Agent 通过 `system_prompt` 声明自身的质量标准；
    Verifier 在执行 Prompt-Free 验证时直接复用该提示词，无需人工设计额外验证词。
    """

    # 本 Agent 的质量标准（系统提示词）
    system_prompt = ""

    def __init__(self, name: str, logger: Optional[AuditLogger] = None):
        self.name = name
        self.logger = logger or AuditLogger()

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """执行 Agent 的核心逻辑，返回结构化结果 dict。"""
        raise NotImplementedError

    def log(self, action: str, status: str, detail: str,
            data: Optional[dict] = None):
        """记录审计日志。"""
        if self.logger:
            self.logger.log(self.name, action, status, detail, data)

    def log_experiment(self, phase: str, decision: str,
                       inputs: Optional[dict] = None,
                       outputs: Optional[dict] = None,
                       result: Optional[dict] = None):
        """记录实验账本（Ledger），供时间轴回放与审计。"""
        if self.logger:
            self.logger.log_experiment(phase, decision, inputs, outputs, result)

    def __repr__(self) -> str:
        return f"Agent({self.name})"

    @staticmethod
    def _resolve_docker_cmd() -> Optional[str]:
        """解析可用的 docker CLI 路径。

        优先取系统 PATH，其次探测 DOCKER_PATH 环境变量与常见安装目录
        （Windows 上 Docker Desktop 的 docker.exe 常不在 PATH 中）。
        返回绝对路径或命令名，未找到返回 None。
        """
        env_path = os.environ.get("DOCKER_PATH", "").strip().strip('"')
        if env_path and os.path.isfile(env_path):
            return env_path
        found = shutil.which("docker")
        if found:
            return found
        for cand in _DOCKER_CANDIDATES:
            if os.path.isfile(cand):
                return cand
        return None

    @staticmethod
    def docker_engine_available(
            probe_cmd: Optional[List[str]] = None,
            timeout: float = 10.0) -> Tuple[bool, Optional[str]]:
        """探测 Docker **引擎（daemon）**是否真正可用，而非 CLI 二进制是否存在。

        为什么必须分开看：`shutil.which("docker")` 只证明 CLI 在 PATH 上——
        Docker Desktop 装了但**没启动**时它同样为真。于是侧边栏谎报「已就绪」，
        `docker run` 秒失败，用户拿到的是
        `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`。
        只有引擎在线，`docker version --format {{.Server.Version}}` 才有输出。

        为什么用 `version` 而不是 `info`：实测引擎未启动时 `docker version`
        **188ms** 即失败返回，而 `docker info` 要 **20.7s** 才返回——后者会让
        Streamlit 每轮重跑冻住 20 秒。探测命令选错本身就是另一起事故。

        返回 (True, None)；失败返回 (False, 原因文本)，原因直接可展示给用户。
        probe_cmd 供测试注入假 docker 命令；缺省自动解析 CLI 路径。
        """
        if probe_cmd is None:
            resolved = BaseAgent._resolve_docker_cmd()
            if resolved is None:
                return False, "本机未安装 Docker 或不在 PATH 中"
            probe_cmd = [resolved]
        try:
            res = subprocess.run(
                list(probe_cmd) + ["version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"Docker 探测超时({timeout:g}s)"
        except Exception as e:                  # CLI 不可执行 / 编码异常等
            return False, f"Docker 探测失败：{e}"
        if res.returncode == 0 and (res.stdout or "").strip():
            return True, None
        return False, "Docker 引擎未启动或不可用（daemon 连接失败）"