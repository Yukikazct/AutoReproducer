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

# ---- Docker 镜像源（docker.io 在受限网络下不可达时的一级回退） ----
# 逗号分隔的镜像源前缀，如 "docker.m.daocloud.io,dockerproxy.net"。
# 默认空 = 不启用（行为与改造前一致，只多一次本地存在性探测）。
#
# 为什么是「拉取后 tag 回原名」而不是直接把镜像名换成镜像源名：镜像白名单
# 按 canonical 名 startswith 判定（`docker.m.daocloud.io/library/python:...`
# 会被判成非白名单镜像直接拒绝），执行路径里还有 `image == "python:3.11-slim"`
# 的分支（决定是否注入 requirements 安装），报告展示的也是原名。tag 回原名
# 之后，镜像源对下游**完全透明**。
DOCKER_IMAGE_MIRRORS = [m.strip().rstrip("/") for m in os.environ.get(
    "AUTOREPRO_DOCKER_IMAGE_MIRROR", "").split(",") if m.strip()]
# 单个镜像的拉取预算（秒）：docker.io 不可达时 pull 会一直重试到
# "context deadline exceeded"，不给上限会让整条流水线卡死在这里。
DOCKER_PULL_TIMEOUT = float(os.environ.get(
    "AUTOREPRO_DOCKER_PULL_TIMEOUT", "600"))


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
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return False, f"Docker 探测超时({timeout:g}s)"
        except Exception as e:                  # CLI 不可执行 / 编码异常等
            return False, f"Docker 探测失败：{e}"
        if res.returncode == 0 and (res.stdout or "").strip():
            return True, None
        return False, "Docker 引擎未启动或不可用（daemon 连接失败）"

    # ---- 镜像可用性：本地存在性检查 + 镜像源回退拉取 ----

    @staticmethod
    def is_canonical_image(image: str) -> bool:
        """镜像名是否是不带 registry 前缀的 canonical 名。

        按 Docker 自己的解析规则：首个路径段既不含 `.` 也不含 `:` 时，
        该镜像属于 Docker Hub（`python:3.11-slim` 等价于
        `docker.io/library/python:3.11-slim`）；否则首段就是显式 registry
        （`gcr.io/proj/img`、`localhost:5000/foo`），**不能再往上拼镜像源**。
        """
        if not image:
            return False
        if "/" not in image:
            return True
        first = image.split("/", 1)[0]
        return "." not in first and ":" not in first

    @staticmethod
    def resolve_image_with_mirror(
            image: str,
            mirrors: Optional[List[str]] = None) -> List[str]:
        """给出该镜像的候选拉取名列表：原名在前，其后依次是各镜像源。

        第一个候选始终是原名——本地已有该标签时调用方根本不会走到拉取，
        而目录里配置了镜像源也不该让"先试官方名"这件事消失。
        """
        candidates = [image]
        if not image:
            return candidates
        effective = DOCKER_IMAGE_MIRRORS if mirrors is None else mirrors
        if not effective or not BaseAgent.is_canonical_image(image):
            return candidates
        return candidates + [f"{m}/{image}" for m in effective]

    @staticmethod
    def image_exists_locally(docker_cmd: str, image: str,
                             timeout: float = 60.0) -> bool:
        """探测本地是否已有该镜像标签（docker images -q 有输出即为有）。"""
        try:
            res = subprocess.run(
                [docker_cmd, "images", "-q", image],
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace")
        except Exception:
            return False
        return res.returncode == 0 and bool((res.stdout or "").strip())

    @staticmethod
    def ensure_image_pulled(docker_cmd: str, image: str,
                            mirrors: Optional[List[str]] = None,
                            pull_timeout: Optional[float] = None) -> Optional[str]:
        """确保镜像在本地可用；缺失时按镜像源候选顺序拉取并 tag 回原名。

        返回 None 表示可用；否则返回**可直接展示给人看**的原因文本。

        为什么必须显式预拉，而不是让 `docker run` 自己隐式拉：
        1. 隐式拉取失败时，原始报错会和容器启动日志混在一起，还会在加固
           降级链上白跑三级（每级一次 docker run，级别降完才轮到用户看到）；
        2. 缺失镜像时提前失败，诊断层才能把它归为"运行环境问题"而不是
           "代码跑挂了"——后者会让修复模型拿着一段 docker 报错去改论文代码，
           空烧修复轮次（实测 4 次 exit 125 全部是镜像拉取失败）。
        """
        if BaseAgent.image_exists_locally(docker_cmd, image):
            return None
        effective = DOCKER_IMAGE_MIRRORS if mirrors is None else mirrors
        if pull_timeout is None:
            pull_timeout = DOCKER_PULL_TIMEOUT
        tried: List[str] = []
        last_err = ""
        for candidate in BaseAgent.resolve_image_with_mirror(image, mirrors):
            tried.append(candidate)
            try:
                res = subprocess.run(
                    [docker_cmd, "pull", candidate],
                    capture_output=True, text=True, timeout=pull_timeout,
                    encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                last_err = f"拉取 {candidate} 超时（{pull_timeout:g}s）"
                continue
            except Exception as e:
                last_err = f"拉取 {candidate} 失败：{e}"
                continue
            if res.returncode != 0:
                last_err = ((res.stderr or res.stdout or "").strip()[-500:]
                            or f"docker pull {candidate} 退出码 {res.returncode}")
                continue
            if candidate != image:
                # 拉下来的名字必须 tag 回原名：白名单、`python:3.11-slim`
                # 特判、报告展示全部按原名判断。
                try:
                    tag_res = subprocess.run(
                        [docker_cmd, "tag", candidate, image],
                        capture_output=True, text=True, timeout=60.0,
                        encoding="utf-8", errors="replace")
                except Exception as e:
                    last_err = f"{candidate} 已拉取但打标签失败：{e}"
                    continue
                if tag_res.returncode != 0:
                    last_err = (f"{candidate} 已拉取但打标签失败："
                                f"{(tag_res.stderr or '').strip()[-300:]}")
                    continue
            return None
        if effective:
            hint = "；请检查网络，或更换 AUTOREPRO_DOCKER_IMAGE_MIRROR 镜像源"
        else:
            hint = ("；可设置 AUTOREPRO_DOCKER_IMAGE_MIRROR 指向可用镜像源"
                    "（如 docker.m.daocloud.io），或先手动 docker pull 该镜像")
        return (f"镜像 {image} 本地不存在且拉取失败"
                f"（已尝试：{'、'.join(tried)}）。{last_err}{hint}")