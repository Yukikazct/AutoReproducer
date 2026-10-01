"""CodeExecutorAgent - 代码执行 Agent，在沙箱中运行论文代码。

对齐方案「Phase 4: 代码执行」：
- 两阶段执行：smoke test（短时冒烟，快速暴露环境问题）-> full run（完整运行）；
- 捕获标准输出、错误日志、退出码；
- 支持本地子进程（隔离临时目录 + 超时）与 Docker 容器两种沙箱；
- 本地子进程只用于 Mock 演示；真实模式必须显式启用 Docker 隔离，否则拒绝执行；
- 本地模式执行前按 env_config 依赖清单自动 pip 安装（幂等缓存 +
  独立超时 + 失败诊断），修复"EnvBuilder 给出依赖但本地执行器直接运行
  导致 ModuleNotFoundError"缺陷——复现环境与执行环境现在保持一致；
- 存储优化（对齐方案「三层存储」L0 热缓存）：
  * mock_mode=True 时跳过真实 pip 安装（mock 演示不触网、不装大包）——
    修复"Mock 流水线 EnvBuilder 注入 torch 全家桶后本地执行器真实
    pip install torch(2GB+)"导致演示卡死/污染全局环境的缺陷；
  * 真实模式依赖隔离安装到 data/deps/<依赖清单哈希>/（pip --target），
    不再装进全局 site-packages——同一依赖清单全局只装一次、多论文
    天然共享去重；执行时经 PYTHONPATH 注入该隔离目录；
- 执行前语法门：清洗后的代码必须能 compile，不通过则针对"截断/语法
  错误"再生成（限次），仍不可编译则诚实短路为"未运行"，绝不把残码
  送进沙箱——避免把"代码被截断"掩盖成沙箱里的 IndentationError；
- 信息不足时**不再预先拒绝**：改成尽力而为生成一份最小可运行脚本并照常
  进沙箱（模型两次都不给代码时落本地兜底脚本 `_BEST_EFFORT_SCRIPT`），
  结果带 `best_effort` 标注一路传到报告，由 ResultValidator 判为"无法核对"
  而非"复现成功/失败"——既不交白卷，也不让占位结果冒充论文结论。
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.agents.env_builder import PIP_INDEX_URL, PIP_FIND_LINKS
from src.agents.dependency_resolver import (
    find_missing_module, python_package_for,
)
from src.code_units import sanitize_unit_id
from src.execution_plan import PlanStep
from src.resource_events import ResourceEventLogger

LOCAL_TIMEOUT_SMOKE = 10
LOCAL_TIMEOUT_FULL = 60
DOCKER_TIMEOUT_SMOKE = 30
DOCKER_TIMEOUT_FULL = 300
# Docker 容器内依赖安装的独立预算（秒）：修复「pip install torch 吃掉整个
# smoke 超时」的缺陷——安装阶段与脚本运行阶段分两次 docker run、分开计时，
# 脚本超时预算不再被依赖安装消耗。
DOCKER_INSTALL_TIMEOUT = 900
# 本地依赖安装超时（numpy/matplotlib/torch 等大包需要更长时间）
LOCAL_PIP_TIMEOUT = 300
# 运行时缺模块自我修复上限：缺包 -> 隔离安装 -> 重跑，最多 3 轮
# （对齐 ScholarAgent coder.py 的 MAX_SELF_CORRECTIONS=3）
MAX_PIP_SELF_HEAL = 3
# 执行计划单步修复上限：超时缩参 / 缺包补装最多重试 3 轮
MAX_PLAN_STEP_REPAIRS = 3
# GPU-only 脚本特征（stderr 命中则标记不可修复，回退生成脚本路径）
_GPU_ONLY_HINTS = ("cuda", "nccl", "no gpu", "out of memory on device",
                   "torch.cuda", "cudnn")
# 数据集缺失特征：官方脚本读 ./dataset/<name>/<name>.csv，而数据集在 README
# 的网盘链接里、不在仓库内（iTransformer 全系列仓库都是这个形态）。
# 这不是代码缺陷，重试无意义——单独归类，报告里才不会把它读成
# "官方代码跑不起来"。
_DATASET_MISSING_FILE_HINTS = ("filenotfounderror", "no such file or directory",
                               "errno 2", "cannot find the file")
_DATASET_MISSING_PATH_HINTS = (".csv", ".txt", "dataset", "/data/")
# 进程内依赖安装结果缓存：依赖清单文本 -> ""(已就绪) 或 失败诊断文本。
# smoke/full/多次优化重跑共用一个进程，只对同一清单安装一次；
# 失败也缓存，避免反复重装浪费时间。
_INSTALLED_DEPS: Dict[str, str] = {}

# ---- P1-⑪ Docker 沙箱加固参数（镜像白名单 + cap-drop + 只读 + 非 root + 限额） ----
# 镜像白名单前缀（逗号分隔，可用 AUTOREPRO_DOCKER_IMAGE_ALLOWLIST 覆盖）：
# 只允许官方/自建镜像前缀，拒绝任意第三方镜像拉取执行。
DOCKER_IMAGE_ALLOWLIST = [p.strip() for p in os.environ.get(
    "AUTOREPRO_DOCKER_IMAGE_ALLOWLIST",
    "python:,pytorch/,autorepro,nvidia/").split(",") if p.strip()]
# 加固总开关：AUTOREPRO_DOCKER_HARDEN=0 时完全不加防护参数（不推荐，仅兼容极端环境）
DOCKER_HARDEN = os.environ.get("AUTOREPRO_DOCKER_HARDEN", "1") != "0"
# 资源限额默认值（可覆盖 AUTOREPRO_DOCKER_CPUS/MEM/PIDS）
DOCKER_DEFAULT_CPUS = float(os.environ.get("AUTOREPRO_DOCKER_CPUS", "2.0"))
DOCKER_DEFAULT_MEM = os.environ.get("AUTOREPRO_DOCKER_MEM", "2g")
DOCKER_DEFAULT_PIDS = int(os.environ.get("AUTOREPRO_DOCKER_PIDS", "256"))
# 容器内非 root 用户（默认 nobody，可覆盖 AUTOREPRO_DOCKER_USER）
DOCKER_DEFAULT_USER = os.environ.get("AUTOREPRO_DOCKER_USER", "65534:65534")
# 容器内 pip 安装目标：**挂载卷内**的目录，不是 /tmp。
#
# 不能用 /tmp/site-packages：`--tmpfs /tmp` 是**每容器独立**的，容器一退出
# 就销毁。而安装步与脚本步是两次 `docker run`（预算分离的本意），于是第二次
# 容器里 PYTHONPATH 指向的是一个刚建出来的空目录——torch/numpy 全部
# ModuleNotFoundError，自愈循环再装一次也仍然丢掉，3 轮后判失败。
# 挂载卷（-v 宿主目录:/app）在两次容器之间保留，且 `--read-only` 只读的是
# 容器 rootfs，不影响卷，所以装到 /app 下才真的能带到脚本步。
DOCKER_PIP_SITE = "/app/.autorepro_site"
# 加固参数与容器环境不兼容的错误特征（命中则按可用性降级重跑）
_HARDEN_INCOMPATIBLE_HINTS = (
    "unknown flag", "unknown shorthand flag", "not supported",
    "operation not permitted", "permission denied",
    "read-only file system", "readonly file system",
    "cannot create directory", "mkdir", "no space left",
    # tmpfs 挂载不可执行时 C 扩展加载失败的特征（见 _sandbox_args 的 exec 说明）。
    # 成因已修，但另一台机器/另一版 Docker 可能仍有 noexec 的 tmpfs，降级到
    # level 2（无 tmpfs）能让 pip --target 落到可执行的可写层，跑得下去。
    "failed to map segment",
)

# ---- L0 依赖缓存（对齐方案「三层存储」：热缓存统一收敛到项目 data/ 下） ----
# src/agents/code_executor.py -> parents[2] 为仓库内 AutoReproducer 包根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 隔离依赖安装根目录：data/deps/<依赖清单 sha1 前 16 位>/
# 同一依赖清单跨论文跨会话只落一份，多论文共享去重；可配 AUTOREPRO_DEPS_ROOT 覆盖。
DEPS_CACHE_ROOT = Path(os.environ.get(
    "AUTOREPRO_DEPS_ROOT",
    str(_PROJECT_ROOT / "data" / "deps")))
# 安装完成标志文件：存在即视为该隔离目录已就绪
_DEPS_READY_MARK = ".ready"
# 隔离目录自带的元数据文件名（requirements / 安装与最后使用时间），
# 供「依赖缓存」管理界面识别与清理；不参与安装，写失败也不影响执行。
_DEPS_META_NAME = "meta.json"


def normalize_requirements(reqs: str) -> str:
    """依赖清单归一：只折叠**写法**差异，不改变 pip 的解析结果。

    隔离目录按清单哈希寻址，清单写法（行序、空行、缩进、重复行）一变就是
    另一个目录，同一份依赖会被整份重装。实测本机有两个内容等价、各
    53 MB / 1553 文件的 numpy 目录，只因清单文本不同就各存了一份。

    因此只做不改语义的折叠：去空行、去纯注释行、去首尾空白、保序去重、
    按包名排序（大小写不敏感）。

    例外：清单里若含 `-r` / `-c` / `--index-url` 这类**选项行**，顺序是有
    意义的（选项对其后的行生效），此时只去重去空白、**不排序**。
    """
    lines: List[str] = []
    for raw in (reqs or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    lines = list(dict.fromkeys(lines))          # 保序去重
    if any(l.startswith("-") for l in lines):
        return "\n".join(lines)
    return "\n".join(sorted(lines, key=str.lower))


def reqs_digest(reqs: str) -> str:
    """隔离目录名：归一化清单的 sha1 前 16 位。"""
    return hashlib.sha1(
        normalize_requirements(reqs).encode("utf-8")).hexdigest()[:16]


def _deps_meta_path(deps_dir: Path) -> Path:
    return deps_dir / _DEPS_META_NAME


def read_deps_meta(deps_dir: Path) -> Dict:
    """读隔离目录元数据；缺失/损坏时返回 {}（调用方回退到目录 mtime）。"""
    try:
        return json.loads(_deps_meta_path(deps_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_deps_meta(deps_dir: Path, kind: str, requirements: str = "",
                     module: str = "") -> None:
    """安装完成后写元数据：目录里到底装的是什么、什么时候装的。"""
    now = datetime.now().isoformat(timespec="seconds")
    meta: Dict = {"kind": kind, "installed_at": now, "last_used": now}
    if requirements:
        meta["requirements"] = requirements
        meta["normalized"] = normalize_requirements(requirements)
    if module:
        meta["module"] = module
    try:
        _deps_meta_path(deps_dir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if requirements:
            (deps_dir / "requirements.txt").write_text(
                requirements, encoding="utf-8")
    except OSError:
        pass        # 元数据只是可观测性，写不进去不能影响安装与执行


def touch_deps_meta(deps_dir: Path) -> None:
    """命中缓存时刷新 last_used（「保留最近 N 天」据此判断冷热）。"""
    meta = read_deps_meta(deps_dir)
    meta["last_used"] = datetime.now().isoformat(timespec="seconds")
    meta.setdefault("kind", "reqs")
    try:
        _deps_meta_path(deps_dir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

# 代码不可编译时的再生成次数上限（LLM 输出被截断是常见故障）
MAX_CODE_REGEN = 2
# 截断时"续写拼接"的次数上限。与再生成的区别：再生成是拿同一个 prompt
# 从头重写，若截断源于撞 max_tokens 上限，重写只会再撞一次；续写是把已写
# 部分的尾部交给模型接着写完，总长度 = 各段之和，才能真正突破单次上限。
MAX_CODE_CONTINUE = 3
# 模型"拒答"（只回占位标记/空输出）时的定向重试上限。比续写(3)/重生成(2)
# 更严：那两条针对"写得不够"，这条针对"没写"——同一条 prompt 重放只会再撞
# 一次，值得的只有换指令问一次。
MAX_MARK_RETRY = 1
# 执行阶段代码修复上限：每轮都必须重新通过语法/安全门并重跑。
MAX_EXECUTION_REPAIRS = 3
# 续写 prompt 里回灌的"已写内容"末尾行数（够模型接上下文即可，不必全给）
_CONTINUE_TAIL_LINES = 40
# 模型仍可能回的占位标记（整份"代码"只有这一行注释）。命中即走
# `_recover_from_placeholder`：定向重试 MAX_MARK_RETRY 次，仍拿不到代码就
# 落本地兜底脚本 `_BEST_EFFORT_SCRIPT`——绝不让"模型拒答"变成"没代码可跑"。
_INSUFFICIENT_INFO_MARK = "# INSUFFICIENT_INFO"
# 语法错误信息中提示"输出被截断"的特征词
_TRUNCATION_HINTS = (
    "unexpected eof", "eof in multi-line", "unterminated",
    "was never closed", "unexpected end of",
)
# 末尾行以这些字符结尾 -> 语句明显没写完（截断的典型特征）
_TRUNCATION_TAIL_CHARS = "=*+-([{,:\\"
# 拼接时判定"末行没写完、需要由续写重写"的悬挂尾字符。
# 与 _TRUNCATION_TAIL_CHARS 的区别：**不含 `:`**——`for i in range(3):`
# 是语法完整的行，正等着后续代码块，丢掉它反而会毁掉循环。
_DANGLING_TAIL_CHARS = "=*+-([{,|\\"
_TRACEBACK_RE = re.compile(
    r'File "([^"]+)", line (\d+)(?:, in ([^\n]+))?\s*\n\s*(.*)')
# 续写契约要求模型先重写 head 末行；该约定只在末行确实悬挂时生效
# 判定"未知/占位"论文信息用的空值模式（与 PaperReader._UNKNOWN_RE 判据一致）
_UNKNOWN_RE = re.compile(
    r"^\s*(|未知.*|未找到|无|n/?a|none|null)\s*$", re.IGNORECASE)
# Exit code：代码在进入沙箱前就被拦下（仅剩两道真门：语法错误 / 危险调用。
# 论文信息不足已不再是拦截理由——见模块 docstring 与 best_effort 标注）
EXIT_NOT_RUNNABLE = -5
# Exit code：本地执行被危险代码静态门拦下（命令执行/动态执行/网络/递归删除）
EXIT_DANGER_BLOCKED = -6
# Exit code：Docker 引擎（daemon）不可用，未进入沙箱
EXIT_DOCKER_DAEMON_DOWN = -4
EXIT_ISOLATION_REQUIRED = -7
# Exit code：Docker 镜像本地不存在且拉取失败（运行环境问题，与代码无关）
EXIT_DOCKER_IMAGE_UNAVAILABLE = -8
_NON_CODE_REPAIR_EXIT_CODES = {
    -1, -3, -4, EXIT_DANGER_BLOCKED, EXIT_ISOLATION_REQUIRED,
    EXIT_DOCKER_IMAGE_UNAVAILABLE,
}

# 镜像拉取失败的特征串（stderr/stdout 命中即归为运行环境问题，不修代码）。
# 不按退出码 125 判定：125 是 docker CLI 的通用"自身错误"码，加固降级链里
# unknown-flag 场景同样会拿到 125，把它整体划成不可修复会误伤真实的
# 「加固参数不兼容，降级后可跑」路径（那条路径靠 stderr 特征区分）。
_PULL_FAILED_HINTS = (
    "failed to resolve reference", "context deadline exceeded",
    "unable to find image", "pull access denied", "no such host",
    "connection refused", "i/o timeout", "manifest unknown",
    "dial tcp", "tls handshake timeout", "lookup ",
)

# 本地无沙箱执行前的危险代码静态门：命中即拒绝执行。高信号、对「复现
# 训练脚本」低误报；是正则兜底而非正式沙箱，生产复现不可信代码请用 Docker。
_DANGEROUS_PATTERNS = (
    (re.compile(r"\bsubprocess\b"), "subprocess 进程/命令执行"),
    (re.compile(r"\bos\.system\b"), "os.system 命令执行"),
    (re.compile(r"\bos\.popen\b"), "os.popen 命令执行"),
    (re.compile(r"\bos\.spawn\w*\b"), "os.spawn* 进程创建"),
    (re.compile(r"\bpty\b"), "pty 终端"),
    (re.compile(r"\beval\s*\("), "eval 动态执行"),
    (re.compile(r"\bexec\s*\("), "exec 动态执行"),
    (re.compile(r"\b__import__\s*\("), "__import__ 动态导入"),
    (re.compile(r"\bsocket\b"), "socket 网络"),
    (re.compile(r"\brequests\b"), "requests 网络外联"),
    (re.compile(r"\burllib\b"), "urllib 网络外联"),
    (re.compile(r"\bhttp\.client\b"), "http.client 网络"),
    (re.compile(r"\bftplib\b"), "ftplib 网络"),
    (re.compile(r"\bsmtplib\b"), "smtplib 邮件外发"),
    (re.compile(r"\bparamiko\b"), "paramiko SSH"),
    (re.compile(r"\bhttpx\b"), "httpx 网络外联"),
    (re.compile(r"\baiohttp\b"), "aiohttp 网络外联"),
    (re.compile(r"\bshutil\.rmtree\b"), "shutil.rmtree 递归删除"),
)

# 论文信息不足时的结果说明（报告与日志共用同一句话，避免两处措辞漂移）
_BEST_EFFORT_REASON = (
    "论文未提供可用的方法/数据集/声明指标；本轮代码为尽力而为的占位实现，"
    "其输出不是论文结论")

# 信息不足且模型两次都不给代码时使用的**本地兜底脚本**：保证"一定有代码、
# 一定能执行、一定有输出"。三条硬约束（每条都有测试兜）：
# 1. 纯标准库、确定性，且不含 `_DANGEROUS_PATTERNS` 的任何词——那是全文正则，
#    连注释里出现 subprocess/eval/socket 之类都会让本地执行被拒；
# 2. 打印内容避开 `_METRIC_PATTERNS` 的键名，且没有任何 `键 = 数值` 形式的行：
#    占位数字一旦被下游 `_extract_metrics` 抓成"实测指标"，就会喂出假的复现
#    结论（这也是本脚本刻意不打印 metrics 的原因）；
# 3. `_structurally_complete` 为真（末尾有顶层调用），否则会被当成"没写完"
#    再触发一轮续写。
_BEST_EFFORT_SCRIPT = '''# 占位复现脚本（论文信息不足时由系统自动生成）
# 说明: 论文未提供可用的方法/数据集/声明指标，本脚本只演示一条最小可运行
# 流程；其输出不是论文结论，不能用于评价论文的可复现性。
# 假设: 自变量在 [-1, 1] 上均匀取值
# 假设: 真实关系为 y = 2x + 1（合成数据，非论文数据）
# 假设: 以最小二乘解析解代替论文未给出的方法
import math


def build_data(n=101):
    xs = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    return xs, [2.0 * x + 1.0 for x in xs]


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-12
    return num / den, my - (num / den) * mx


def main():
    xs, ys = build_data()
    slope, intercept = fit(xs, ys)
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in resid) / len(resid))
    print("------------------------------------------")
    print("占位复现脚本: 论文信息不足, 以下输出不是论文结论")
    print("------------------------------------------")
    print("合成样本数:", len(xs))
    print("拟合斜率:", round(slope, 4), "拟合截距:", round(intercept, 4))
    print("拟合残差平方根均值:", round(rms, 4))
    print("论文未声明指标, 本脚本不产出可与论文比对的数值")


main()
'''

# markdown 代码块围栏（可能带 python 语言标注）
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# 行首行号（"1 def f(x):" 这类带行号转储）：数字前空白保留，数字后
# 最多吃掉一个分隔空白，剩下的空白是原本的缩进，必须留给代码。
_LINE_NO_RE = re.compile(r"^(\s*)\d+[ \t]?")
# 行首残留的围栏/引号残片
_FENCE_LEFT = re.compile(r"^\s*(```+|>>>|\.\.\.)\s*", re.MULTILINE)
# 判定"看起来像 Python 代码行"的行首（\w 会匹配中文,故全部用 ASCII 白名单）
# 覆盖：import/from/def/class/if/for/while/try/except/with/return/print/raise/
# pass/break/continue/del/assert/global/nonlocal/yield/match/case/lambda/
# 装饰器@/注释#/赋值= / 函数调用()/索引访问[]/属性访问. / 数字
_CODE_LINE_START = re.compile(
    r"^\s*(?:"
    r"import\s|from\s|def\s|class\s|if\s|elif\s|else\s*:|for\s|while\s|"
    r"try\s*:|except\s|finally\s*:|with\s|return\s|print\s*\(|raise\s|"
    r"pass\s*$|break\s*$|continue\s*$|del\s|assert\s|global\s|nonlocal\s|"
    r"yield\s|match\s|case\s|lambda\s|@|#|"
    r"[A-Za-z_][A-Za-z0-9_.]*\s*=|"                    # 赋值
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\(|"                   # 函数调用 super().__init__()
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\[|"                   # 索引 self.net[0]
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\."                    # 属性访问 self.net.forward
    r"|[A-Za-z_\[\(\"']|[\d+\-.]"                      # 兜底：字母/括号/引号/数字开头
    r")")
# 判定"中文叙述行"用的代码特征字符。含中文的行里只要有这些字符之一，
# 就更可能是**代码**（含中文字符串字面量，如 print(f"准确率: {acc}")），
# 而不是叙述段落——旧实现只按"含中文"就丢，会把合法的中文 print 一起删掉，
# 而删掉后代码往往仍能编译，于是"残缺"被静默放过。
_CODEISH_CHARS = set("=(){}[]\"'#:+-*/%<>@,.")
_CJK_RE = re.compile(r"[一-鿿]")
# 不产生任何行为的顶层节点：脚本若只有这些，编译通过但运行后什么也不做
_INERT_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.Import, ast.ImportFrom)


class CodeExecutorAgent(BaseAgent):
    """在 Docker 或本地沙箱中执行论文代码。"""

    system_prompt = "在沙箱中安全执行论文代码,输出运行日志、数值结果与退出码"

    def __init__(self, llm_client: LLMClient, logger=None,
                 use_docker: bool = False, mock_mode: bool = False):
        super().__init__("CodeExecutor", logger)
        self.llm = llm_client
        self.use_docker = use_docker
        self.mock_mode = mock_mode
        # 最近一次依赖就绪的隔离安装目录（供执行时注入 PYTHONPATH）
        self._deps_dir: Optional[str] = None
        # 运行时自愈补装的隔离目录集合（data/deps/heal-<module>/），
        # 全部注入 PYTHONPATH，与依赖清单目录不互相污染。
        self._heal_dirs: set = set()
        self.resource_events = ResourceEventLogger()

    @staticmethod
    def _diagnose_execution_error(result: Dict) -> Dict:
        """从执行结果提取可供修复模型使用的结构化诊断。"""
        stderr = str(result.get("stderr") or "")
        stdout = str(result.get("stdout") or "")
        combined = stderr or stdout
        match = None
        for match in _TRACEBACK_RE.finditer(combined):
            pass
        diagnosis = {
            "error_type": "execution_error",
            "message": combined[-2000:],
            "file": "",
            "line": None,
            "function": "",
            "source_line": "",
            "repairable": True,
        }
        if match:
            diagnosis.update({
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": (match.group(3) or "").strip(),
                "source_line": (match.group(4) or "").strip(),
            })
            tail = combined[match.start():]
            error_lines = [line.strip() for line in tail.splitlines()
                           if line.strip()]
            if error_lines:
                diagnosis["error_type"] = error_lines[-1].split(":", 1)[0]
                diagnosis["message"] = "\n".join(error_lines[-4:])[-2000:]
        lower = combined.lower()
        if "modulenotfounderror" in lower or "no module named" in lower:
            diagnosis["error_type"] = "missing_dependency"
            diagnosis["repairable"] = False
        if result.get("danger_blocked") or result.get("exit_code") in (
                _NON_CODE_REPAIR_EXIT_CODES):
            diagnosis["repairable"] = False
        if "timed out" in lower or "执行超时" in lower:
            # 超时可修复：此前硬编码 repairable=False 导致修复循环 1 轮即放弃
            # （iTransformer 实测：smoke 超时后直接停摆）。必须放在退出码门
            # **之后**判定——超时正是退出码 -1，会被 _NON_CODE_REPAIR_EXIT_CODES
            # 重新盖回不可修复。修复策略由 _repair_prompt 的超时分支给出
            # （缩参/缩短训练循环）。
            diagnosis["error_type"] = "timeout"
            diagnosis["repairable"] = True
        if (result.get("exit_code") == EXIT_DOCKER_IMAGE_UNAVAILABLE
                or any(hint in lower for hint in _PULL_FAILED_HINTS)):
            # 放最后判定：镜像拉不到是**运行环境**问题，改代码不可能修好。
            # 此前没有这个分类，exit 125 + 一段 docker 原始报错被判成
            # 可修复，3 轮 LLM 修复全烧在「改论文代码」上（实测 4 次 exit 125）。
            # 退出码与 stderr 特征两条路都认：预拉失败是我们自己返回的人话
            # 文本（对不上 docker 原始特征串），隐式拉取失败才是原始报错。
            diagnosis["error_type"] = "docker_pull_failed"
            diagnosis["repairable"] = False
        return diagnosis

    @staticmethod
    def _repair_prompt(paper_info: Dict, code: str, stage: str,
                       diagnosis: Dict) -> str:
        """生成定点修复请求，避免模型只解释错误而不返回代码。"""
        location = "未知位置"
        if diagnosis.get("line") is not None:
            location = f"{diagnosis.get('file', 'run.py')}:{diagnosis['line']}"
        extra_rule = ""
        if diagnosis.get("error_type") == "timeout":
            extra_rule = ("修复规则（超时）：大幅缩短运行时间以满足沙箱时限——"
                          "减小 seq_len/batch_size、把训练 epoch 减到 1~3、"
                          "缩小数据集规模；保持指标打印语句完整，"
                          "其余论文逻辑不变。\n")
        return f"""你正在修复一份论文复现 Python 脚本。脚本在 {stage} 阶段执行失败。
请根据 traceback 对代码做最小、定点的修复，保持论文方法、数据处理、指标打印
和已有工作区接口不变。只输出修复后的完整 Python 源码，不要 Markdown 围栏、
解释文字或占位标记。

论文方法: {paper_info.get('method', '未知')}
论文数据集: {paper_info.get('dataset', '未知')}
失败位置: {location}
异常类型: {diagnosis.get('error_type', 'execution_error')}
{extra_rule}错误上下文:
{diagnosis.get('message', '')}

当前代码:
```python
{code}
```
"""

    def _repair_after_execution(self, paper_info: Dict, code: str,
                                stage: str, result: Dict) -> tuple:
        """请求一次代码修复，并在返回后立即执行语法/安全门。"""
        diagnosis = self._diagnose_execution_error(result)
        if not diagnosis["repairable"]:
            return code, diagnosis, None
        raw = self.llm.chat(
            self._repair_prompt(paper_info, code, stage, diagnosis),
            task="code_executor_repair")
        candidate, stats = self._sanitize_code_ex(raw)
        diagnosis["sanitize_stats"] = stats
        if self._is_placeholder_code(candidate):
            diagnosis["repair_error"] = "修复结果为空或仅包含占位标记"
            return code, diagnosis, None
        syntax_error = self._syntax_error(candidate)
        if syntax_error:
            diagnosis["repair_error"] = f"修复结果语法错误: {syntax_error}"
            return code, diagnosis, None
        if not self.use_docker:
            danger = self._dangerous_constructs(candidate)
            if danger:
                diagnosis["repair_error"] = f"修复结果含危险调用: {danger}"
                return code, diagnosis, None
        return candidate, diagnosis, candidate

    def _execute_with_repair(self, code: str, paper_info: Dict) -> tuple:
        """执行 smoke/full；运行时错误按预算定点修复后从 smoke 重跑。"""
        stages = []
        attempts = []
        current = code
        for repair_round in range(MAX_EXECUTION_REPAIRS + 1):
            smoke = self._execute_code(current, stage="smoke")
            stages.append({"stage": "smoke", "repair_round": repair_round,
                           **smoke})
            if not smoke["success"]:
                failed_stage, failed = "smoke", smoke
            else:
                full = self._execute_code(current, stage="full")
                stages.append({"stage": "full", "repair_round": repair_round,
                               **full})
                if full["success"]:
                    return current, stages, attempts, full
                failed_stage, failed = "full", full
            if repair_round >= MAX_EXECUTION_REPAIRS:
                break
            fixed, diagnosis, candidate = self._repair_after_execution(
                paper_info, current, failed_stage, failed)
            attempt = {
                "round": repair_round + 1,
                "stage": failed_stage,
                "diagnosis": diagnosis,
                "code_before_sha1": hashlib.sha1(
                    current.encode("utf-8")).hexdigest(),
            }
            if candidate is None:
                attempt["status"] = "stopped"
                attempts.append(attempt)
                break
            attempt.update({
                "status": "retry",
                "code_after_sha1": hashlib.sha1(
                    fixed.encode("utf-8")).hexdigest(),
            })
            attempts.append(attempt)
            current = fixed
        return current, stages, attempts, stages[-1]

    # ---------------- 执行计划模式（多代码单元整体调用） ----------------

    def _execute_plan(self, plan: Dict, paper_info: Dict) -> Dict:
        """按 ExecutionPlan 逐步执行官方代码（安装 -> 下载 -> 运行 -> 解析）。

        安全姿态与生成脚本路径一致：真实模式强制 Docker 加固沙箱；
        mock 模式允许本地子进程（演示/测试夹具，仓库为可信 file:// 来源）。

        返回结果含 stages（逐步记录）/ final（拼接 run 输出，供
        ResultValidator 照常提取指标）/ actual_metrics /
        plan_failed_irreparably（官方路径彻底失败 -> run() 回退生成脚本）。
        """
        steps = plan.get("steps") or []
        units = plan.get("units") or []
        if not self.mock_mode and not self.use_docker:
            # 官方代码不可信：真实模式必须 Docker 隔离（复用既有门）
            return self._plan_irreparable(
                "真实模式执行官方代码必须启用 Docker 隔离",
                exit_code=EXIT_ISOLATION_REQUIRED)

        # 工作区：各单元 copytree 到临时目录（去 .git/__pycache__），
        # 整体挂载到容器 /app；chmod o+w 供容器内非 root 用户写入。
        workspace = tempfile.mkdtemp(prefix="autorepro_plan_")
        unit_dirs: Dict[str, str] = {}
        try:
            for unit in units or []:
                src = (unit or {}).get("local_path") or ""
                uid = sanitize_unit_id((unit or {}).get("unit_id") or "")
                if not src or not uid or not Path(src).is_dir():
                    continue
                dst = Path(workspace) / uid
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", ".ipynb_checkpoints"))
                unit_dirs[uid] = uid
            try:
                self._bind_plan_datasets(plan, Path(workspace))
            except (OSError, ValueError) as exc:
                return {**self._plan_irreparable(f"数据准备失败: {exc}"),
                        "execution_mode": "plan", "plan": plan}
            # 工作区根也要放开：mkdtemp 默认 0700，而挂载的是**根目录**，
            # 容器里的 nobody 要 traverse /app 才能进 /app/<unit>。
            self._chmod_tree_writable(Path(workspace))
            (Path(workspace) / ".plan.json").write_text(
                json.dumps(plan, ensure_ascii=False), encoding="utf-8")

            stage_records: List[Dict] = []
            step_outcomes: Dict[str, object] = {}
            run_outputs: List[str] = []
            for step_dict in steps or []:
                step = PlanStep.from_dict(step_dict)
                deps = step.depends_on or []
                if deps and not all(
                        step_outcomes.get(d) is True for d in deps):
                    record = {"step_id": step.step_id, "kind": step.kind,
                              "cmd": step.cmd, "unit_id": step.unit_id,
                              "success": False, "stdout": "", "stderr": "",
                              "exit_code": 0, "skipped_deps": True,
                              "repairs": []}
                    stage_records.append(record)
                    step_outcomes[step.step_id] = "skipped"
                    continue
                if step.kind == "parse":
                    record = self._parse_plan_step(step, run_outputs)
                else:
                    record = self._execute_plan_step_with_repair(
                        step, workspace, unit_dirs)
                stage_records.append(record)
                step_outcomes[step.step_id] = bool(record.get("success"))
                if record.get("success") and step.kind == "run":
                    run_outputs.append(record.get("stdout", "") or "")

            run_succeeded = any(r.get("success") and r.get("kind") == "run"
                                for r in stage_records)
            real_failures = [r for r in stage_records
                             if not r.get("success")
                             and not r.get("skipped_deps")
                             and r.get("kind") != "parse"]
            plan_failed = bool(real_failures) and not run_succeeded
            final_stdout = "\n".join(run_outputs)
            final_stderr = "\n".join(
                (r.get("stderr") or "") for r in stage_records
                if r.get("stderr"))
            actual_metrics = self._parse_plan_metrics(final_stdout, steps)
            final = {
                "success": not real_failures and bool(stage_records),
                "stdout": final_stdout,
                "stderr": final_stderr,
                "exit_code": 0 if not real_failures
                else (real_failures[0].get("exit_code") or 1),
            }
            result = {
                "success": final["success"],
                "stages": stage_records,
                "final": final,
                "code": self._entry_script_text(plan),
                "execution_mode": "plan",
                "plan": plan,
                "actual_metrics": actual_metrics,
                "experiment_profile": plan.get("experiment_profile", ""),
                "parameters": plan.get("parameters", {}),
                "datasets": plan.get("datasets", []),
                "best_effort": False,
                "fallback_used": False,
                "plan_failed_irreparably": plan_failed,
                # 键恒存在（失败时才有内容）：报告生成器读它渲染「回退原因」，
                # 先前这条正常失败路径漏了该键，报告里永远是空的。
                "plan_fail_reason": (self._plan_fail_reason(real_failures)
                                     if plan_failed else ""),
            }
            if plan_failed:
                self.log("execute_plan", "ERROR",
                         f"官方代码执行计划失败（{len(real_failures)} 步未通过），"
                         + ("保留失败记录，不替换官方实验"
                            if plan.get("experiment_profile") else "回退生成脚本路径"),
                         {"failures": [r.get("step_id") for r in real_failures]})
            else:
                self.log_experiment(
                    "EXECUTE_CODE", "官方代码执行计划完成",
                    inputs={"plan": plan.get("source")},
                    outputs={"stages": stage_records,
                             "actual_metrics": actual_metrics},
                    result={"success": final["success"]})
                self.log("execute_plan", "SUCCESS",
                         f"计划执行完成: {len(stage_records)} 步, "
                         f"指标 {list(actual_metrics.keys())}")
            return result
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _bind_plan_datasets(plan: Dict, workspace: Path) -> None:
        """Bind verified data into the copy, never modify the cached repository."""
        known_units = {u.get("unit_id") for u in plan.get("units", [])}
        for binding in plan.get("datasets") or []:
            uid = binding.get("unit_id", "")
            target = Path(binding.get("target") or "")
            if (uid not in known_units or not uid or sanitize_unit_id(uid) != uid
                    or target.is_absolute() or ".." in target.parts or not target.name):
                raise ValueError("数据集目标路径非法")
            dest = workspace / uid / target
            if not dest.resolve().is_relative_to((workspace / uid).resolve()):
                raise ValueError("数据集目标路径越界")
            source = Path(binding.get("path") or "")
            if source.is_symlink() or not source.is_file():
                raise ValueError("数据集文件不存在或为符号链接")
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != binding.get("sha256"):
                raise ValueError("数据集 SHA-256 校验失败")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

    @staticmethod
    def _prepare_pip_target(workdir: str) -> str:
        """在宿主工作区里建好容器内 pip --target 目录，返回容器内路径。

        目录建在挂载卷里（`/app/.autorepro_site`），才能从安装步活到脚本步；
        宿主侧预先建好并放开到 0777，免得容器里的 nobody（65534）写不进去。
        建不出来时返回 ""——调用方退化为不带 --target 的安装（容器层内安装，
        两次 run 之间不保留），而不是让 pip 在容器里报权限错误。
        """
        try:
            path = Path(workdir) / ".autorepro_site"
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o777)
        except OSError:
            return ""
        return DOCKER_PIP_SITE

    @staticmethod
    def _plan_irreparable(reason: str, exit_code: int = 1) -> Dict:
        """官方路径不可修复失败（安全门/GPU-only 等）统一返回。"""
        final = {"success": False, "stdout": "", "stderr": reason,
                 "exit_code": exit_code}
        return {"success": False, "stages": [], "final": final,
                "code": "", "execution_mode": "plan", "plan": {},
                "actual_metrics": {}, "best_effort": False,
                "plan_failed_irreparably": True,
                "plan_fail_reason": reason}

    @staticmethod
    def _plan_fail_reason(real_failures: List[Dict]) -> str:
        """计划失败（跑挂了/依赖装不上，但不属于"不可修复"）时的人话回退原因。

        此前只有 _plan_irreparable 路径写 plan_fail_reason，正常失败路径的结果
        字典**根本没有这个键**——报告生成器读到的恒为空串，用户只看到
        "已回退到生成脚本"，不知道为什么要回退。这里把最后一次修复诊断
        翻成人话补上。
        """
        first = real_failures[0] if real_failures else {}
        step_id = first.get("step_id") or "?"
        repairs = first.get("repairs") or []
        # 镜像拉取失败优先级最高：它是运行环境问题，正是用户最需要一眼看到
        # 的那类原因，不该被"最后一次尝试"的更琐碎诊断盖掉。
        diag = next((r for r in repairs
                     if (r or {}).get("error_type") == "docker_pull_failed"),
                    None) or (repairs[-1] if repairs else {})
        etype = (diag.get("error_type") or "").strip()
        detail = (diag.get("detail") or "").strip()
        lead = f"步骤 {step_id} 失败"
        if etype == "docker_pull_failed":
            return (f"{lead}：Docker 镜像本地不存在且拉取失败"
                    f"（运行环境问题，非论文代码问题）。"
                    f"{detail or '请检查网络或设置 AUTOREPRO_DOCKER_IMAGE_MIRROR'}")
        if etype == "gpu_only":
            return f"{lead}：{detail or '官方脚本需要 GPU，本环境不具备'}"
        if etype == "missing_dataset":
            return f"{lead}：{detail or '缺少数据集文件'}"
        if etype == "missing_dependency":
            return f"{lead}：依赖无法安装（{detail or '未知包'}）"
        if etype == "timeout":
            return f"{lead}：执行超时，缩参重试后仍未通过"
        stderr = (first.get("stderr") or "").strip()
        tail = stderr.splitlines()[-1][:200] if stderr else ""
        return f"{lead}：{tail or '执行未通过，且无可用的确定性修复策略'}"

    @staticmethod
    def _chmod_tree_writable(root: Path) -> None:
        """递归放开 other 权限：容器内非 root 用户（65534）需要 rwx。

        目录必须给 **o+x**，不能只给 o+w：`tempfile.mkdtemp` 建出来的是
        0700，只加 o+w 得到 0702——nobody 连 /app 都 traverse 不进去，
        脚本直接 "Permission denied"（真正的失败点比"写不了"更早，
        只在写权限上找原因会一直找不到）。
        文件不需要 o+x，给到 o+rw 即可（脚本还要能自写产出）。
        """
        try:
            paths = [root, *root.rglob("*")]
        except OSError:
            paths = [root]
        for path in paths:
            try:
                is_dir = path.is_dir()
                mode = path.stat().st_mode
                path.chmod(mode | (0o007 if is_dir else 0o006))
            except OSError:
                continue

    @staticmethod
    def _entry_script_text(plan: Dict) -> str:
        """读取入口脚本内容作为 execution.code（报告/优化工作区复用）。"""
        entry = plan.get("entry") or {}
        script = entry.get("script") or ""
        unit_id = entry.get("unit_id") or "main"
        units = plan.get("units") or []
        unit = next((u for u in units
                     if (u or {}).get("unit_id") == unit_id), None)
        local = (unit or {}).get("local_path") or ""
        if not script or not local:
            return ""
        path = Path(local) / script
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:20000]
        except OSError:
            return ""

    def _execute_plan_step_with_repair(self, step: PlanStep,
                                       workspace: str,
                                       unit_dirs: Dict[str, str]) -> Dict:
        """单步执行 + 修复循环（超时缩参 / 缺包补装，≤MAX_PLAN_STEP_REPAIRS 轮）。"""
        current = step
        repairs: List[Dict] = []
        record = self._execute_plan_step(current, workspace, unit_dirs)
        for _ in range(MAX_PLAN_STEP_REPAIRS):
            if record.get("success"):
                break
            fixed, diagnosis = self._repair_plan_step(current, record)
            repairs.append(diagnosis)
            if fixed is None:
                break
            current = fixed
            record = self._execute_plan_step(current, workspace, unit_dirs)
        record["repairs"] = repairs
        record["repair_attempts"] = len(repairs)
        record["final_cmd"] = current.cmd
        return record

    def _repair_plan_step(self, step: PlanStep,
                          record: Dict) -> tuple:
        """单步失败诊断与修复（确定性优先，无 LLM）。

        返回 (新步骤 | None, 诊断)。修复策略：
        - GPU-only 特征 -> None（不可修复，回退生成脚本）；
        - 超时且带 smoke_args -> 追加缩参重试一次；
        - 缺模块 -> 把对应包追加进 install_pkgs 重试；
        - 其余 -> None。
        """
        stderr = f"{(record.get('stderr') or '')} {(record.get('stdout') or '')}"
        lower = stderr.lower()
        diagnosis: Dict = {"error_type": "execution_error",
                           "strategy": "non_repairable", "detail": ""}
        if (record.get("exit_code") == EXIT_DOCKER_IMAGE_UNAVAILABLE
                or any(hint in lower for hint in _PULL_FAILED_HINTS)):
            # 基础设施故障，优先级高于所有代码类诊断：放在最前，避免被
            # GPU/超时/缺数据集任一条更"具体"的规则抢先归类。
            diagnosis.update(
                error_type="docker_pull_failed", strategy="non_repairable",
                detail="Docker 镜像本地不存在且拉取失败（运行环境问题，"
                       "非论文代码问题）；请检查网络或设置 "
                       "AUTOREPRO_DOCKER_IMAGE_MIRROR 指向可用镜像源")
            return None, diagnosis
        if any(hint in lower for hint in _GPU_ONLY_HINTS):
            diagnosis.update(error_type="gpu_only",
                             strategy="non_repairable",
                             detail="官方脚本需要 GPU，本环境不可修复")
            return None, diagnosis
        if record.get("timed_out"):
            # bash 入口优先用 smoke_cmd：官方 .sh 不转发 "$@"，把缩参追加到
            # `bash x.sh` 后面会被忽略，等于原样重跑一遍、再烧一个超时预算。
            if step.smoke_cmd:
                new = PlanStep.from_dict(step.to_dict())
                new.cmd = step.smoke_cmd
                new.smoke_cmd = ""       # 缩参只做一次
                new.smoke_args = {}
                new.retries += 1
                diagnosis.update(error_type="timeout", strategy="smoke_cmd",
                                 detail=f"缩参单次调用重试: {step.smoke_cmd[:200]}")
                return new, diagnosis
            if step.smoke_args:
                new = PlanStep.from_dict(step.to_dict())
                new.cmd = (new.cmd.strip() + " " + " ".join(
                    f"{k} {v}" for k, v in new.smoke_args.items())).strip()
                new.smoke_args = {}      # 缩参只做一次
                new.retries += 1
                diagnosis.update(error_type="timeout", strategy="smoke_args",
                                 detail=f"缩参重试: {dict(step.smoke_args)}")
                return new, diagnosis
            diagnosis.update(error_type="timeout", strategy="non_repairable",
                             detail="超时且无 smoke_args 可缩参")
            return None, diagnosis
        if (any(h in lower for h in _DATASET_MISSING_FILE_HINTS)
                and any(h in lower for h in _DATASET_MISSING_PATH_HINTS)):
            diagnosis.update(
                error_type="missing_dataset", strategy="non_repairable",
                detail="缺少数据集文件（官方脚本从 README 的网盘链接下载，"
                       "仓库内不含数据）；请先准备 dataset/ 目录再重试")
            return None, diagnosis
        module = find_missing_module(stderr)
        if module:
            pkg = python_package_for(module)
            if pkg and pkg not in step.install_pkgs:
                new = PlanStep.from_dict(step.to_dict())
                new.install_pkgs = list(step.install_pkgs) + [pkg]
                new.retries += 1
                diagnosis.update(error_type="missing_dependency",
                                 strategy="install_pkg", detail=pkg)
                return new, diagnosis
        diagnosis["detail"] = stderr[-300:]
        return None, diagnosis

    def _execute_plan_step(self, step: PlanStep, workspace: str,
                           unit_dirs: Dict[str, str]) -> Dict:
        """执行单条计划步骤（安装前缀与命令本体分开计时）。"""
        if self.use_docker:
            return self._execute_plan_step_docker(step, workspace)
        return self._execute_plan_step_local(step, workspace, unit_dirs)

    def _execute_plan_step_docker(self, step: PlanStep,
                                  workspace: str) -> Dict:
        """Docker 加固沙箱内执行单步：pip 前缀（独立预算）+ 命令本体。"""
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "",
                    "stderr": "本机未安装 Docker 或不在 PATH 中", "exit_code": -3}
        env_config = getattr(self, "env_config", None) or {}
        image = env_config.get("image_tag") or "python:3.11-slim"
        if not self._image_allowed(image):
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "",
                    "stderr": f"镜像 {image} 不在允许白名单，已拒绝执行",
                    "exit_code": -5}
        engine_ok, engine_reason = BaseAgent.docker_engine_available(
            [docker_cmd])
        if not engine_ok:
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "",
                    "stderr": f"Docker 引擎不可用：{engine_reason}",
                    "exit_code": EXIT_DOCKER_DAEMON_DOWN}

        # 3) 镜像可用性：本地缺失时按镜像源预拉。放在引擎守卫之后（daemon
        # 没起时拉取必然失败，先报 daemon 更准确）、构造 base_cmd 之前——
        # 拉不到就带着人话直接返回，绝不让 docker run 拿一段原始报错去喂
        # 修复循环（那条路会把"镜像拉取失败"当成"代码有 bug"来改）。
        pull_err = BaseAgent.ensure_image_pulled(docker_cmd, image)
        if pull_err:
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "", "stderr": pull_err,
                    "exit_code": EXIT_DOCKER_IMAGE_UNAVAILABLE,
                    "image_unavailable": True}

        mount = workspace.replace("\\", "/")
        cwd = step.cwd or "/app"
        base_cmd = [docker_cmd, "run", "--rm", "-v", f"{mount}:/app",
                    "-w", cwd]
        # 安装目标建在挂载卷里：这是安装步与脚本步（两次 docker run）之间
        # 唯一的共享存储，tmpfs 与容器层都带不过去。
        pip_target = self._prepare_pip_target(workspace)
        for key, value in step.env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
                raise ValueError(f"非法环境变量名: {key}")
            base_cmd += ["-e", f"{key}={value}"]
        if getattr(self, "experiment_profile", ""):
            for directory in (".tmp", ".home"):
                path = Path(workspace) / directory
                path.mkdir(exist_ok=True)
                path.chmod(0o777)
            base_cmd += ["-e", "TMPDIR=/app/.tmp", "-e", "HOME=/app/.home",
                         "-e", f"PIP_INDEX_URL={PIP_INDEX_URL}",
                         "-e", f"PIP_FIND_LINKS={PIP_FIND_LINKS}"]
            if step.kind != "install":
                base_cmd += ["--network", "none"]
            if not getattr(self, "_profile_image_id", ""):
                inspected = subprocess.run(
                    [docker_cmd, "image", "inspect", "--format", "{{.Id}}", image],
                    capture_output=True, text=True, timeout=15)
                self._profile_image_id = inspected.stdout.strip() if inspected.returncode == 0 else ""
        install_phase: Dict = {"run": False}

        # 1) 可选安装前缀：独立预算（安装大包不再吃掉脚本超时）
        if step.install_pkgs:
            pkgs = " ".join(step.install_pkgs)
            target = f"--target {pip_target} --no-cache-dir " if pip_target else ""
            install_cmd = (f"pip install -q -i {PIP_INDEX_URL} "
                           f"--find-links {PIP_FIND_LINKS} {target}{pkgs}")
            budget = step.install_budget_s or DOCKER_INSTALL_TIMEOUT
            try:
                res, _meta = self._run_docker_cmd_with_sandbox(
                    base_cmd, image, ["sh", "-c", install_cmd], budget)
            except subprocess.TimeoutExpired:
                return {"step_id": step.step_id, "kind": step.kind,
                        "cmd": step.cmd, "unit_id": step.unit_id,
                        "success": False, "stdout": "",
                        "stderr": f"依赖安装执行超时({budget}s)",
                        "exit_code": -1, "timed_out": True,
                        "install_phase": {"run": True, "timed_out": True}}
            install_phase = {"run": True, "success": res.returncode == 0,
                             "exit_code": res.returncode}
            if res.returncode != 0:
                return {"step_id": step.step_id, "kind": step.kind,
                        "cmd": step.cmd, "unit_id": step.unit_id,
                        "success": False, "stdout": res.stdout or "",
                        "stderr": res.stderr or "", "exit_code": res.returncode,
                        "install_phase": install_phase}

        # 2) 命令本体：只计脚本运行预算
        run_cmd = step.cmd
        if pip_target:
            # PATH 也要带上：`pip install --target` 把 console script 装到
            # <target>/bin（gdown 这类下载工具只有命令没有可 import 的入口），
            # 只注入 PYTHONPATH 的话 `gdown ...` 会 command not found。
            run_cmd = (f"PYTHONPATH={pip_target} "
                       f"PATH={pip_target}/bin:$PATH {run_cmd}")
        timeout = step.timeout_s or DOCKER_TIMEOUT_SMOKE
        try:
            res, sandbox_meta = self._run_docker_cmd_with_sandbox(
                base_cmd, image, ["sh", "-c", run_cmd], timeout)
        except subprocess.TimeoutExpired:
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s)", "exit_code": -1,
                    "timed_out": True, "install_phase": install_phase,
                    "sandbox": {"hardened": DOCKER_HARDEN}}
        return {"step_id": step.step_id, "kind": step.kind,
                "cmd": step.cmd, "unit_id": step.unit_id,
                "success": res.returncode == 0,
                "stdout": res.stdout or "", "stderr": res.stderr or "",
                "exit_code": res.returncode, "timed_out": False,
                "image": image, "image_id": getattr(self, "_profile_image_id", ""),
                "install_phase": install_phase, "sandbox": sandbox_meta}

    def _execute_plan_step_local(self, step: PlanStep, workspace: str,
                                 unit_dirs: Dict[str, str]) -> Dict:
        """本地子进程执行单步（仅 mock 模式：演示/测试夹具）。"""
        cwd = Path(workspace) / (unit_dirs.get(step.unit_id) or "")
        if not cwd.is_dir():
            cwd = Path(workspace)
        install_phase: Dict = {"run": False}
        if step.install_pkgs and not self.mock_mode:
            # 真实模式无 Docker 已在上层拒绝；此处仅 mock 兜底说明
            install_phase = {"run": False,
                             "detail": "本地模式不安装第三方包（mock）"}
        timeout = step.timeout_s or LOCAL_TIMEOUT_SMOKE
        try:
            proc = subprocess.run(
                step.cmd, shell=True, cwd=str(cwd), capture_output=True,
                text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
                env=self._exec_env())
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout or "", "stderr": proc.stderr or "",
                    "exit_code": proc.returncode, "timed_out": False,
                    "install_phase": install_phase,
                    "sandbox": {"hardened": False, "local": True}}
        except subprocess.TimeoutExpired:
            return {"step_id": step.step_id, "kind": step.kind,
                    "cmd": step.cmd, "unit_id": step.unit_id,
                    "success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s)", "exit_code": -1,
                    "timed_out": True, "install_phase": install_phase,
                    "sandbox": {"hardened": False, "local": True}}

    @staticmethod
    def _parse_plan_step(step: PlanStep, run_outputs: List[str]) -> Dict:
        """parse 步骤：从已收集的 run 输出中确定性提取指标（无命令执行）。"""
        stdout = "\n".join(run_outputs)
        metrics = {}
        if stdout:
            # _extract_metrics 为实例方法但不使用 self；借用其正则逻辑
            from src.agents.result_validator import ResultValidatorAgent
            metrics = ResultValidatorAgent._extract_metrics(None, stdout)
        return {"step_id": step.step_id, "kind": "parse", "cmd": "",
                "unit_id": step.unit_id, "success": bool(metrics),
                "stdout": stdout[-2000:], "stderr": "", "exit_code": 0,
                "metrics": metrics, "repairs": []}

    @staticmethod
    def _parse_plan_metrics(stdout: str, steps: List[Dict]) -> Dict:
        """从 run 输出中提取计划期望的指标（复用 ResultValidator 正则）。"""
        if not stdout:
            return {}
        from src.agents.result_validator import ResultValidatorAgent
        extracted = ResultValidatorAgent._extract_metrics(None, stdout)
        expected = set()
        for step in steps or []:
            expected.update((step or {}).get("expects", {}).get("metrics")
                            or [])
        return {k: v for k, v in extracted.items()}

    def run(self, input_data: dict) -> dict:
        """执行论文代码（smoke test + full run）。

        input_data: {"paper_info", "env_config", "resources", "code"(可选)}
        """
        self.log("execute_code", "START", "开始执行代码", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        env_config = input_data.get("env_config", {}) or {}
        code = input_data.get("code", "") or ""
        self.env_config = env_config  # 供执行阶段选择镜像/依赖
        self.experiment_profile = input_data.get("experiment_profile", "") if not self.mock_mode else ""
        # "尽力而为"标注位：外部代码路径没有生成这一步，保持默认值
        insufficient, fallback_used = False, False

        if not self.mock_mode and not self.use_docker:
            return self._not_runnable(
                "真实模式必须启用可用的 Docker 隔离后才能执行模型生成代码",
                code=code, exit_code=EXIT_ISOLATION_REQUIRED,
                extra={"isolation_required": True})

        # 执行计划模式（多代码单元整体调用）：官方代码优先执行。
        # 计划不可修复失败（GPU-only / 依赖不可装等）时回退生成脚本路径，
        # 计划执行记录合并进最终结果（execution_mode="generated_fallback"）。
        plan = input_data.get("execution_plan") or {}
        if self.experiment_profile and not plan.get("steps"):
            return self._not_runnable(
                "; ".join(plan.get("notes") or ["真实实验准备未完成"]), code="",
                extra={"experiment_profile": self.experiment_profile, "plan": plan,
                       "execution_mode": "plan", "fallback_used": False,
                       "parameters": plan.get("parameters", {}),
                       "datasets": plan.get("datasets", [])})
        plan_result = None
        if isinstance(plan, dict) and plan.get("steps"):
            plan_result = self._execute_plan(plan, paper_info)
            if self.experiment_profile:
                # The chosen experiment must never turn into generated fake
                # reproduction code if its environment or data is unavailable.
                return {**plan_result, "llm_calls": self._delta_llm_calls()}
            if not plan_result.get("plan_failed_irreparably"):
                return {**plan_result, "llm_calls": self._delta_llm_calls()}
            self.log("execute_plan", "WARNING",
                     "官方代码计划不可修复失败，回退生成脚本路径",
                     {"reason": plan_result.get("plan_fail_reason", "")})

        if code:
            # 外部提供的真实复现代码：只做清洗，不走生成/再生成
            code, sanitize_stats = self._sanitize_code_ex(code)
            self._record_sanitize("外部代码清洗", code, sanitize_stats)
        else:
            # 论文信息不足**不再预先拒绝**：降级为"尽力而为"标注，照样生成、
            # 照样进沙箱。用户实测过一次「代码长度 0 字符 + 未运行」的空报告，
            # 那条路是"论文信息不足就交白卷"；现在改成：跑出来的东西带
            # best_effort 标注一路传到报告，不参与复现判定——既不交白卷，
            # 也不让占位结果被读成论文结论。
            insufficient = self._info_insufficient(paper_info)
            if insufficient:
                self.log("generate_code", "WARNING",
                         "论文信息不足（缺少方法/数据集/声明指标）——改为尽力"
                         "而为生成最小可运行脚本，其结果不得当作论文结论")
            code, sanitize_stats, fallback_used = self._produce_code(
                paper_info, insufficient=insufficient)

        # 带进报告与验证层的"尽力而为"标注（外部代码路径全部为默认值）
        best_effort_fields = {
            "best_effort": insufficient,
            "best_effort_reason": _BEST_EFFORT_REASON if insufficient else "",
            "fallback_used": fallback_used,
            "assumptions": self._extract_assumptions(code),
        }

        # 语法门：不可编译的代码绝不进沙箱——残码在沙箱里会被报成
        # IndentationError 之类，掩盖"输出被截断"这个真实原因。
        syntax_error = self._syntax_error(code)
        if syntax_error:
            return self._not_runnable(
                f"代码存在语法错误，未执行: {syntax_error}", code=code,
                sanitize_stats=sanitize_stats, extra=best_effort_fields)

        # 危险代码静态门：本地执行无沙箱，拒绝明显危险的调用（命令执行/
        # 动态执行/网络外联/递归删除）。Docker 已是隔离沙箱，不必拦。
        if not self.use_docker:
            danger = self._dangerous_constructs(code)
            if danger:
                return self._not_runnable(
                    f"代码含危险调用，已拒绝执行: {danger}", code=code,
                    sanitize_stats=sanitize_stats, extra=best_effort_fields)

        code, stages, repair_attempts, final = self._execute_with_repair(
            code, paper_info)
        result = {"stages": stages, "success": final["success"],
                  "final": final, "code": code,
                  "repair_attempts": repair_attempts,
                  "sanitize_stats": sanitize_stats,
                  **best_effort_fields}
        # 官方路径失败后的生成脚本回退：合并计划执行记录与失败原因
        if plan_result:
            result["plan_execution"] = plan_result
            result["execution_mode"] = "generated_fallback"
            result["plan_fail_reason"] = plan_result.get(
                "plan_fail_reason", "")
        if not final["success"]:
            self.log_experiment(
                "EXECUTE_CODE", "执行失败，代码修复重试结束",
                inputs={"code": code}, outputs={"stages": stages,
                                                "repair_attempts": repair_attempts},
                result={"success": False})
            self.log("execute_code", "ERROR",
                     f"代码执行失败（已尝试 {len(repair_attempts)} 轮修复）",
                     {"exit_code": final.get("exit_code"),
                      "repair_attempts": len(repair_attempts)})
            return {**result, "llm_calls": self._delta_llm_calls()}

        self.log_experiment(
            "EXECUTE_CODE", "完成 smoke + full 两阶段执行",
            inputs={"code": code},
            outputs={"stages": stages, "repair_attempts": repair_attempts},
            result={"success": True},
        )
        self.log("execute_code", "SUCCESS",
                 f"代码执行成功（修复 {len(repair_attempts)} 轮）",
                 {"exit_code": final.get("exit_code"),
                  "repair_attempts": len(repair_attempts),
                  "stdout_tail": (final.get("stdout") or "")[-300:]})

        return {**result, "llm_calls": self._delta_llm_calls()}

    # ---------------- 代码生成 ----------------

    def _produce_code(self, paper_info: Dict,
                      insufficient: bool = False) -> tuple:
        """生成复现代码：截断则**续写拼接**，仍不完整才从头再生成。

        修复「代码生成不完整」的主路径：单次调用的输出上限（默认 8192）
        是硬天花板，用同一个 prompt 从头重写只会再撞一次；因此先按"断点
        续写"把总长度累加上去（最多 MAX_CODE_CONTINUE 轮），每轮都过语法门。
        续写仍不完整时，才回落到"从头再生成"作为最后手段。

        另一条路是模型"拒答"（只回占位标记或空输出）：定向重试
        MAX_MARK_RETRY 次，仍拿不到代码就落本地兜底脚本，保证"一定有代码可跑"
        ——见 `_recover_from_placeholder`。

        `insufficient=True` 时 prompt 会要求"信息不足也必须给出最小可运行
        脚本"（见 `_generate_code_prompt`）。

        返回 `(代码, 清洗统计, 是否用了本地兜底脚本)`；代码仍可能是不可编译
        的——由调用方 run() 的语法门统一判定并短路为"未运行"，此处不负责
        掩盖失败。
        """
        code, stats = self._sanitize_code_ex(
            self._generate_code(paper_info, insufficient))
        self._record_sanitize("初次生成", code, stats)

        fallback_used = False
        if self._is_placeholder_code(code):
            code, stats, fallback_used = self._recover_from_placeholder(
                paper_info, insufficient, stats)
            if fallback_used:
                return code, stats, True   # 兜底脚本本身完整，无需再续写

        # ---- 阶段 1：续写拼接（针对"输出被截断"/代码被洗残） ----
        prev_err = self._syntax_error(code)
        for round_no in range(1, MAX_CODE_CONTINUE + 1):
            if not self._needs_continuation(code, stats):
                return code, stats, fallback_used
            finish = getattr(self.llm, "last_finish_reason", "")
            self.log("generate_code", "WARNING",
                     f"生成代码疑似未写完，触发续写（第 {round_no} 轮）: "
                     f"{prev_err or self._incomplete_reason(code, stats)}"
                     + (f" [finish_reason={finish}]" if finish else ""))
            raw = self._continue_code(paper_info, code, insufficient)
            more, stats = self._sanitize_code_ex(raw)
            self._record_sanitize(f"续写第 {round_no} 轮", more, stats)
            if not more.strip() or self._is_placeholder_code(more):
                # 没给新内容（或只回了占位标记），别再空转（保留已有 code）
                break
            if more.strip() in code:
                # 续写返回的内容已原样存在于现有代码里 = 模型在复述而非续写。
                # 继续追问只会把重复内容越拼越长，还白烧 LLM 预算。
                self.log("generate_code", "WARNING",
                         "续写返回的是已有内容（模型复述），停止续写")
                break
            new_code = self._stitch(code, more)
            new_err = self._syntax_error(new_code)
            if new_err is not None and new_err == prev_err:
                # 症状一字未变 = 模型在复述而不是续写，继续问下去只是白烧预算
                self.log("generate_code", "WARNING",
                         f"续写未改善语法错误（仍是 {new_err}），停止续写")
                break
            code, prev_err = new_code, new_err
        if not self._needs_continuation(code, stats):
            return code, stats, fallback_used

        # ---- 阶段 2：续写仍不完整 -> 从头再生成（最后手段） ----
        for attempt in range(1, MAX_CODE_REGEN + 1):
            err = self._syntax_error(code)
            if err is None and not self._needs_continuation(code, stats):
                return code, stats, fallback_used
            reason = ("疑似输出被截断" if err and self._looks_truncated(code, err)
                      else (err or self._incomplete_reason(code, stats)))
            finish = getattr(self.llm, "last_finish_reason", "")
            self.log("generate_code", "WARNING",
                     f"续写后仍不完整（{reason}，重生成第 {attempt} 次）"
                     + (f" [finish_reason={finish}]" if finish else ""))
            new_code, stats = self._sanitize_code_ex(
                self._regenerate_code(paper_info, err or reason, attempt,
                                      insufficient))
            self._record_sanitize(f"重生成第 {attempt} 次", new_code, stats)
            if self._is_placeholder_code(new_code):
                # 重生成直接拒答：换指令定向重试，仍不成则落兜底脚本收工
                code, stats, fallback_used = self._recover_from_placeholder(
                    paper_info, insufficient, stats)
                if fallback_used:
                    return code, stats, True
                continue
            code = new_code
        return code, stats, fallback_used

    def _needs_continuation(self, code: str, stats: Optional[Dict] = None) -> bool:
        """判断代码是否"还没写完"，应该续写而不是从头重写。

        判据（任一命中即续写）：
        1. 清洗时丢掉了**疑似代码行**（`code_dropped > 0`）——说明代码可能被
           洗残了，必须让模型重写，不能带着缺损继续跑；
        2. API 明确说这次撞了 max_tokens（`finish_reason == "length"`）——
           比"末尾字符像断句"这类启发式可靠得多；
        3. 语法错误且形态像"话没说完"（尾部悬挂运算符/未闭合括号等）；
        4. 能编译但结构上不做任何事（只有 def/class/import，无顶层调用）——
           语法门查不出的"残缺"，同样属于没写完。

        注意：`prose_dropped`（剥叙述行）是预期行为，**不**触发续写。
        """
        if int((stats or {}).get("code_dropped", 0) or 0) > 0:
            return True
        is_truncated = getattr(self.llm, "is_truncated", None)
        if callable(is_truncated) and is_truncated():
            return True
        err = self._syntax_error(code)
        if err is not None:
            return self._looks_truncated(code, err)
        return not self._structurally_complete(code)

    @staticmethod
    def _incomplete_reason(code: str, stats: Optional[Dict] = None) -> str:
        """给"不完整"一个人话原因，用于日志与重生成提示。"""
        dropped = int((stats or {}).get("code_dropped", 0) or 0)
        if dropped:
            return f"清洗丢弃了 {dropped} 行疑似代码行"
        return "结构不完整（无顶层执行语句）"

    def _continue_code(self, paper_info: Dict, partial: str,
                       insufficient: bool = False) -> str:
        """续写：把已写部分的尾部交给模型，让它从断点接着写完。

        与 `_regenerate_code` 的关键区别：不重复整份需求、不从头重写，
        因此输出可以稳定接在前文之后，总长度突破单次调用上限。
        """
        tail = "\n".join(partial.rstrip("\n").splitlines()[-_CONTINUE_TAIL_LINES:])
        prompt = f"""你在帮我写一份论文复现的 Python 脚本，上一次输出因为长度
限制在中间被截断了。请**从断点继续往下写**，把剩余部分补完。

论文方法: {paper_info.get('method', '未知')}
数据集: {paper_info.get('dataset', '未知')}
指标: {paper_info.get('metrics', {})}

【已经写完的部分（结尾 {_CONTINUE_TAIL_LINES} 行）】
```python
{tail}
```

【续写要求 - 必须严格遵守】
1. **先把上面最后一行完整地重写一遍**（如果你认为它已经写完整了，就原样
   重复一次），然后再往下写——这样拼接时才不会出现半行相接；
2. 之后只输出**后续的新内容**：不要重写开头、不要重复更早的行、不要加
   任何解释说明、不要输出 markdown 围栏；
3. 每一行都必须是合法 Python 代码，缩进与前文保持一致；
4. 一直写到脚本真正结束为止：训练与评估完成，并打印出上面列出的指标。
"""
        if insufficient:
            prompt += """5. 论文信息不足也要给出可运行代码：不得回占位标记、不得
   留空；未给出的量用合成数据与默认值，并以 `# 假设: <内容>` 注释写明。
"""
        return self.llm.chat(prompt, task="code_executor")

    @staticmethod
    def _stitch(head: str, tail: str) -> str:
        """拼接续写片段，消除接缝处的半行与重复行。

        约定见 `_continue_code`：tail 的第一行是对 head 末行的重写，故先丢掉
        head 的末行再拼接；随后再做一次重复行去重，容忍模型多复述上文。
        """
        head_lines = head.rstrip("\n").splitlines()
        tail_lines = tail.strip("\n").splitlines()
        if not head_lines:
            return "\n".join(tail_lines)
        if not tail_lines:
            return "\n".join(head_lines)
        # 只在 head 末行**确实没写完**时才丢它，交给 tail 首行重写；末行本身
        # 完整时保留，重复交给下面的去重处理。旧实现无条件丢弃，模型一旦
        # 没按约定重写末行就会静默吞掉一行。
        if CodeExecutorAgent._line_incomplete(head_lines[-1]):
            head_lines = head_lines[:-1]

        # 去重：head 尾部与 tail 头部若有若干行相同（忽略缩进差异），去掉重复
        for k in range(min(len(head_lines), len(tail_lines)), 0, -1):
            if not any(ln.strip() for ln in tail_lines[:k]):
                continue
            if ([ln.strip() for ln in head_lines[-k:]]
                    == [ln.strip() for ln in tail_lines[:k]]):
                tail_lines = tail_lines[k:]
                break
        return "\n".join(head_lines + tail_lines)

    @staticmethod
    def _line_incomplete(line: str) -> bool:
        """这一行是否"没写完"：括号未闭合，或以悬挂运算符结尾。

        只看末字符是不够的——`print('%s' % best` 少了右括号，末字符却是
        `t`；因此先数括号。`:` 不算悬挂：`for i in range(3):` 是语法完整的
        行，正等着后续代码块，丢掉它反而会毁掉循环。

        （括号计数不含字符串字面量里的括号，属已知近似；判错时由语法门
        与续写流程兜住。）
        """
        stripped = line.strip()
        if not stripped:
            return False
        opens = (stripped.count("(") + stripped.count("[")
                 + stripped.count("{"))
        closes = (stripped.count(")") + stripped.count("]")
                  + stripped.count("}"))
        if opens > closes:
            return True
        return stripped[-1] in _DANGLING_TAIL_CHARS

    @staticmethod
    def _structurally_complete(code: str) -> bool:
        """编译之外的**结构**完整性：脚本必须真的会"做点什么"。

        全是 def/class/import、既无顶层调用也无 `__main__` 守卫的片段，
        能通过 `compile()` 却不会产出任何结果——这是纯语法门查不出的残缺。
        """
        if not code or not code.strip():
            return False
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return False
        for node in tree.body:
            if isinstance(node, _INERT_NODES):
                continue
            # 裸字面量/文档字符串不产生行为，不算"做了事"
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            return True
        return False

    def _record_sanitize(self, stage: str, code: str, stats: Dict) -> None:
        """记录清洗结果。

        `prose_dropped`（剥掉叙述行）是预期行为，记一条即可；
        `code_dropped`（丢了疑似代码行）则意味着**可能把代码洗残了**——
        记 WARNING，并由 `_needs_continuation` 据此触发补救。
        """
        prose = int((stats or {}).get("prose_dropped", 0) or 0)
        code_drop = int((stats or {}).get("code_dropped", 0) or 0)
        if code_drop:
            self.log("sanitize_code", "WARNING",
                     f"{stage}：清洗丢弃了 {code_drop} 行疑似代码行"
                     f"（剩余 {len(code.splitlines())} 行）——"
                     f"结果可能已被洗残，将触发补救重写",
                     {"code_dropped": code_drop, "stage": stage})
        elif prose:
            self.log("sanitize_code", "RUNNING",
                     f"{stage}：剥离了 {prose} 行叙述文字（预期行为）",
                     {"prose_dropped": prose, "stage": stage})

    def _generate_code_prompt(self, paper_info: Dict,
                              insufficient: bool = False) -> str:
        """生成 prompt；`insufficient=True` 时第 5 条换成"必须交出最小脚本"。

        信息充足（默认）时输出与历史版本逐字一致——`TestPrompts` 锁死了其中
        的「完整」「围栏」「字符串字面量」与四段流程等串。
        """
        return f"""根据论文信息生成一份**完整**的复现脚本。
论文方法: {paper_info.get('method', '未知')}
指标: {paper_info.get('metrics', {})}
数据集: {paper_info.get('dataset', '未知')}

【脚本必须覆盖的完整流程】
数据加载/构造 → 模型与方法定义 → 训练循环 → 评估 → **打印论文声明的各项指标**。
这是要能直接跑出复现结果的完整脚本，不是演示片段；不要为了简短而省略
训练循环、评估步骤或指标输出。

【输出格式 - 必须严格遵守】
1. 用**单个** ```python 围栏把整份脚本包起来，围栏内只有代码；
2. 围栏内每一行都必须是合法 Python 代码；不要写围栏外的解释文字、
   中文叙述段落或开场白；
3. 中文字符**允许**出现在字符串字面量与 # 注释里
   （例如 print(f"准确率: {{acc:.4f}}") 是合法的），只是不允许写成
   围栏外的叙述段落；
4. 若内容较长一次写不完，请在**一个完整语句的边界**停下（不要停在半个
   表达式中间），我会让你继续写完剩余部分；
5. {self._rule5(insufficient)}
"""

    @staticmethod
    def _rule5(insufficient: bool) -> str:
        """输出格式第 5 条，按信息是否充足分两种口径。

        信息不足时**不再提供"交白卷"的出口**：旧文案让模型"只输出一行占位
        标记并停止"，而那行标记当年没有任何代码识别它——结果是空烧预算、再把
        空脚本送进执行。禁令仍在，但宾语从"生成代码"改成"冒充结论"：可以写
        通用实现、可以用合成数据，但不许把无关数据集或编造的数值写成论文声明值。
        """
        if not insufficient:
            return ("若上面的论文方法/数据集确实是未知的占位值，无法据此写出"
                    "针对性代码，则只输出一行 "
                    f"`{_INSUFFICIENT_INFO_MARK}` 并停止，严禁用无关数据集"
                    "(如 CIFAR-10/IMDB)编造一个与本论文无关的模型来充数。")
        return """上面的论文方法/数据集/指标**缺失或为未知占位值**。即便如此也**必须**
   交出一份能直接运行的最小复现脚本——不允许以"信息不足"为由拒答、
   不允许只输出一行占位标记、不允许留空：
   - 用**纯标准库或最基础的依赖**实现一条最小但完整可跑的流程
     （数据构造 → 模型与方法定义 → 训练/拟合 → 评估 → 打印结果）；
   - 论文未给出的量（数据集、超参数、指标数值）一律用**合成数据与默认值**，
     并在代码里以 `# 假设: <内容>` 注释逐条写明来源；
   - 脚本开头必须打印一行明确声明：本脚本是信息不足下的占位实现，其输出
     **不是**论文结论、不能用于评价论文的可复现性；
   - 仍然**严禁**把 CIFAR-10/IMDB 这类与论文无关的数据集、或凭空编造的
     数值，写成"论文声明值"来冒充论文结论。"""

    def _generate_code(self, paper_info: Dict,
                       insufficient: bool = False) -> str:
        return self.llm.chat(
            self._generate_code_prompt(paper_info, insufficient),
            task="code_executor")

    def _regenerate_code(self, paper_info: Dict, err: str, attempt: int,
                         insufficient: bool = False) -> str:
        """再生成：把上一次的失败原因回灌给 LLM，要求输出完整脚本。"""
        prompt = self._generate_code_prompt(paper_info, insufficient) + f"""
【上一次输出不可用 - 第 {attempt} 次重试】
上一次生成的代码无法通过编译，原因: {err}
这通常意味着输出被截断了。请重新输出一份**完整**的 Python 脚本：
每个函数体/循环体都要有正确的缩进，最后一行必须是完整语句。
"""
        return self.llm.chat(prompt, task="code_executor")

    # ---------------- "拒答"的处置 ----------------

    @staticmethod
    def _is_placeholder_code(code: str) -> bool:
        """模型是否"根本没给代码"：去掉空行与 `#` 注释后没有任何内容。

        覆盖两种拒答：只回一行 `_INSUFFICIENT_INFO_MARK`，或返回空。
        刻意不看 `_structurally_complete`——那只说明"只有 def/import、还没
        写完"，属于要续写的场景，与"拒答"不是一回事。
        """
        body = [ln for ln in (code or "").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        return not body

    def _recover_from_placeholder(self, paper_info: Dict, insufficient: bool,
                                  stats: Dict) -> tuple:
        """模型"拒答"时的处置：定向重试 → 仍拒答则落本地兜底脚本。

        为什么不直接同 prompt 重放：`_regenerate_code` 那条路已经证明重复同
        一条指令只会再撞一次；这里换的是**指令本身**（"上一次只回复了占位
        标记，必须给出可运行脚本"），且只问 MAX_MARK_RETRY 次。兜底脚本保证
        "一定有代码、一定能执行、一定有输出"——这正是用户要的
        「一定要尝试生成代码并允许」的底线。

        返回 `(代码, 清洗统计, 是否用了兜底脚本)`。
        """
        self.log("generate_code", "WARNING",
                 "模型未给出代码（占位标记/空输出），发起定向重试"
                 f"（上限 {MAX_MARK_RETRY} 次）",
                 {"mark": _INSUFFICIENT_INFO_MARK})
        for _ in range(MAX_MARK_RETRY):
            raw = self.llm.chat(self._mark_retry_prompt(paper_info,
                                                        insufficient),
                                task="code_executor")
            code, stats = self._sanitize_code_ex(raw)
            self._record_sanitize("定向重试", code, stats)
            if not self._is_placeholder_code(code):
                return code, stats, False
        self.log("generate_code", "WARNING",
                 "定向重试仍未拿到代码，回落到本地兜底脚本"
                 f"（{len(_BEST_EFFORT_SCRIPT.splitlines())} 行）")
        return _BEST_EFFORT_SCRIPT, stats, True

    def _mark_retry_prompt(self, paper_info: Dict,
                           insufficient: bool) -> str:
        """定向重试 prompt：说明上一次只回了占位标记，明确要求可运行脚本。

        末尾那句是**覆盖式**的：即便上面第 5 条按信息充足的口径写了"无法写出
        针对性代码就回占位标记"，这里也要求必须给出脚本——否则两次拒答直接
        落到兜底脚本，用户就看不到任何"尝试生成"的结果。
        """
        return (self._generate_code_prompt(paper_info, insufficient) + f"""
【上一次只回复了占位标记】
上一次的输出是 `{_INSUFFICIENT_INFO_MARK}`，没有给出任何可运行的代码。
现在**必须**交出一份能直接运行的最小 Python 脚本（哪怕只是通用实现、用合成
数据、加 `# 假设:` 注释说明），不要再回占位标记、不要留空、不要解释。
""")

    @staticmethod
    def _extract_assumptions(code: str) -> List[str]:
        """收集脚本自述的假设（`# 假设: ...`），供报告如实展示生成依据。"""
        return [m.strip() for m in re.findall(
            r"^\s*#\s*假设\s*[:：]\s*(.+?)\s*$", code or "", re.MULTILINE)][:10]

    def _sanitize_code(self, raw: str) -> str:
        """将 LLM 原始输出清洗为可执行的纯净 Python 代码（兼容旧签名）。"""
        return self._sanitize_code_ex(raw)[0]

    def _sanitize_code_ex(self, raw: str) -> tuple:
        """清洗代码，返回 (代码, 统计)。

        统计: {"prose_dropped": n, "code_dropped": m}
        - `prose_dropped`：丢弃的**叙述行**（如"为了复现该论文…"），预期行为；
        - `code_dropped` ：丢弃的**疑似代码行**——可疑信号，意味着可能把代码
          洗残了，上层应据此触发补救，而不是带着缺损继续执行。

        三级策略（保真优先）：
        1. 有围栏且可编译 -> 整块原样返回，一个字符都不改；
        2. 否则只丢"确定是叙述"的行，**其余一律保留**；
        3. 保留版仍编译不过，才启用"像不像代码行"的白名单过滤兜底（有损）。
        """
        empty = {"prose_dropped": 0, "code_dropped": 0}
        if not raw or not raw.strip():
            return (raw or ""), dict(empty)
        # 只去首尾空行与行尾空白，**绝不 strip 首行缩进**——续写片段的第一行
        # 本来就可能是缩进行（如 `    print(...)`），一旦被 strip 掉就会变成
        # 顶格，拼起来直接 IndentationError。与 Batch 1 修的"缩进丢失"同源。
        text = raw.strip("\n").rstrip()

        # 1) 提取最长的 markdown 代码块（若被围栏包裹）——保真路径，优先走这条
        fenced = _CODE_FENCE.findall(text)
        if fenced:
            candidate = max(fenced, key=len).strip("\n").rstrip()
            if self._syntax_error(candidate) is None:
                return candidate, dict(empty)   # 原样返回，一个字符都不改
            text = candidate

        # 2) 只丢"确定是叙述"的行，其余一律保留（不靠"像不像代码"猜）
        code, prose = self._drop_prose_only(text)

        # 3) 仍不可编译 -> 才启用白名单过滤兜底，并单独计数被丢的疑似代码行
        code_dropped = 0
        if self._syntax_error(code) is not None:
            code, code_dropped = self._filter_code_lines(code)
        return code, {"prose_dropped": prose, "code_dropped": code_dropped}

    @staticmethod
    def _looks_like_prose(line: str) -> bool:
        """这一行是否像"中文叙述"（应丢弃），而非含中文的合法代码。

        判据：含中文、非 # 注释、且**没有任何代码特征字符**。这样
        print(f"准确率: {acc:.4f}")（有括号/引号/冒号）会被保留，而
        "为了复现该论文，我们使用以下配置。" 这种纯叙述会被丢弃。
        旧实现只按"含中文"就丢，会把合法的中文 print 一并删掉，且删完
        代码往往仍能编译，导致"残缺"被静默放过——这正是"代码不完整"
        却查不出来的路径之一。
        """
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return False
        if not _CJK_RE.search(stripped):
            return False
        return not (set(stripped) & _CODEISH_CHARS)

    @classmethod
    def _drop_prose_only(cls, text: str) -> tuple:
        """保真清洗：只丢弃"确定是叙述"的行，返回 (代码, 丢弃的叙述行数)。

        刻意**不**判断"这一行像不像代码"——那种白名单判据天然不完整，
        `)`、`*x, y = [1, 2, 3]`、docstring 里的 `* bullet` 都是合法代码却会
        被误杀；而误杀之后剩下的代码往往仍能编译，残缺就被静默放过了
        （实测：删掉 `*x, y = [1, 2, 3]` 后脚本照跑，只在运行期 NameError）。
        这里只做显式判据（`_looks_like_prose`），丢弃的行确定是叙述。
        """
        kept, prose = [], 0
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                if kept and kept[-1].strip():
                    kept.append(ln)
                continue
            if cls._looks_like_prose(stripped):
                prose += 1
                continue
            kept.append(ln)
        return "\n".join(kept).strip("\n"), prose

    @classmethod
    def _filter_code_lines(cls, text: str) -> tuple:
        """有损兜底：按"像不像代码行"丢弃，返回 (代码, 丢弃的非空行数)。

        仅在保真清洗后仍不可编译时调用。丢弃的行计入 `code_dropped`，
        因为其中可能混有被误杀的合法代码（`)`、`*x, y = ...` 等）——
        上层据此触发补救。
        """
        text = _FENCE_LEFT.sub("", text)
        kept, dropped = [], 0
        for ln in text.splitlines():
            # 只清掉行首行号与行尾空白，保留前导缩进——缩进一旦被抹掉，
            # 函数体/循环体会整体塌陷，把"输出被截断"这个真实原因
            # 伪装成一个更难定位的 IndentationError。
            fixed = _LINE_NO_RE.sub(r"\1", ln).rstrip()
            # 去行号后不像代码行（如续行 "  2)"）时保留原行,避免误删
            if not _CODE_LINE_START.match(fixed) \
                    and _CODE_LINE_START.match(ln.rstrip()):
                fixed = ln.rstrip()
            if not fixed.strip():
                continue
            if _CODE_LINE_START.match(fixed) and not cls._looks_like_prose(fixed):
                kept.append(fixed)
            else:
                dropped += 1
        return "\n".join(kept), dropped

    # ---------------- 执行前检查 ----------------

    @staticmethod
    def _dangerous_constructs(code: str) -> Optional[str]:
        """扫描代码中的危险调用，命中返回可读原因，否则 None。

        仅针对本地无沙箱执行（_execute_code_local）：本地模式以完整用户权限
        运行 LLM 代码，明显危险的调用（命令执行/动态执行/网络外联/递归删除）
        一律拒绝。正则兜底，非正式沙箱；生产复现不可信代码请用 Docker。
        """
        if not code:
            return None
        for pattern, label in _DANGEROUS_PATTERNS:
            if pattern.search(code):
                return label
        return None

    @staticmethod
    def _syntax_error(code: str) -> Optional[str]:
        """编译检查：语法错误返回可读信息，通过则返回 None。"""
        if not code or not code.strip():
            return "代码为空"
        try:
            compile(code, "<generated>", "exec")
            return None
        except SyntaxError as e:
            return f"{e.msg} (line {e.lineno})"
        except ValueError as e:      # 源码含空字节等
            return str(e)

    @staticmethod
    def _looks_truncated(code: str, err: str) -> bool:
        """判断语法错误是否更像"输出被截断"而非"模型写错了语法"。"""
        if any(h in err.lower() for h in _TRUNCATION_HINTS):
            return True
        tail = next((ln.strip() for ln in reversed(code.splitlines())
                     if ln.strip()), "")
        return bool(tail) and tail[-1] in _TRUNCATION_TAIL_CHARS

    @staticmethod
    def _info_insufficient(paper_info: Dict) -> bool:
        """论文结构化信息是否不足以生成针对性复现代码。

        判据：解析层显式标记 info_sufficient=False / insufficient_info，
        或方法与数据集双双缺失/为占位值。缺方法必不足以写代码；缺数据集
        但给出了声明指标时仍可尝试（例如纯数学/合成数据的方法）。
        """
        if paper_info.get("info_sufficient") is False:
            return True
        if paper_info.get("insufficient_info") is True:
            return True
        method = str(paper_info.get("method", "") or "")
        dataset = str(paper_info.get("dataset", "") or "")
        metrics = paper_info.get("metrics") or {}
        return bool(_UNKNOWN_RE.match(method) and _UNKNOWN_RE.match(dataset)
                    and not metrics)

    def _not_runnable(self, reason: str, code: str,
                      sanitize_stats: Optional[Dict] = None,
                      extra: Optional[Dict] = None,
                      exit_code: int = EXIT_NOT_RUNNABLE) -> dict:
        """代码未进入执行阶段（语法错误/危险调用）时的统一返回。

        与"跑了但失败"区分：exit_code=EXIT_NOT_RUNNABLE 且带 not_runnable
        标记，供 ResultValidator 判为"无法验证"而非"复现失败"。
        """
        stage = {"stage": "precheck", "success": False, "stdout": "",
                 "stderr": reason, "exit_code": exit_code,
                 "not_runnable": True}
        self.log_experiment(
            "EXECUTE_CODE", "代码未通过执行前检查,未进入沙箱",
            inputs={"code": code}, outputs=stage, result={"success": False})
        self.log("execute_code", "ERROR", f"代码未运行: {reason}",
                 {"exit_code": exit_code, "not_runnable": True})
        return {"stages": [stage], "success": False, "final": stage,
                "code": code, "not_runnable": True, "reason": reason,
                "sanitize_stats": sanitize_stats or
                {"prose_dropped": 0, "code_dropped": 0},
                **(extra or {}),
                "llm_calls": self._delta_llm_calls()}

    # ---------------- 执行 ----------------

    def _execute_code(self, code: str, stage: str,
                      workdir: Optional[str] = None) -> Dict:
        """执行代码：本地子进程或 Docker 容器，按阶段使用不同超时。

        workdir: 指定执行目录时在目标目录执行且不清理（生命周期由调用方
        管理，如优化器真实执行配合快照回滚）；缺省时使用临时目录（用完删除）。
        """
        if self.use_docker:
            return self._execute_code_docker(code, stage, workdir=workdir)
        return self._execute_code_local(code, stage, workdir=workdir)

    def execute_in_workspace(self, code: str, workdir: str,
                             stage: str = "full") -> Dict:
        """在指定工作区目录中执行代码（真实优化闭环用）。

        与 _execute_code 的区别：工作目录由调用方提供且执行后保留
        （不清理），配合 src.safety.workspace_snapshot 完成
        "补丁 -> 真实重跑 -> 快照回滚"的安全优化闭环。
        """
        return self._execute_code(code, stage, workdir=workdir)

    def _execute_code_local(self, code: str, stage: str,
                            workdir: Optional[str] = None) -> Dict:
        """在本地执行代码：临时目录（不指定 workdir）或目标目录执行。

        执行前按 env_config 依赖清单自动安装依赖（_ensure_local_deps），
        依赖安装失败时直接返回失败诊断，不浪费脚本执行预算。
        脚本运行失败且 stderr 命中缺失模块时，走运行时自愈：
        隔离安装 -> 重跑，最多 MAX_PIP_SELF_HEAL 轮（见 _self_heal_local）。
        """
        if not self.mock_mode:
            reason = ("真实模式必须启用可用的 Docker 隔离；"
                      "拒绝在宿主机执行模型生成代码")
            self.log("execute_local", "ERROR", reason)
            return {"success": False, "stdout": "", "stderr": reason,
                    "exit_code": EXIT_ISOLATION_REQUIRED,
                    "isolation_required": True}

        # 危险代码静态门（兜底）：Optimizer 真实执行 execute_in_workspace
        # 绕过 run() 直接进这里，仍需拦截危险调用。
        danger = self._dangerous_constructs(code)
        if danger:
            return {"success": False, "stdout": "",
                    "stderr": f"拒绝执行(危险代码): {danger}",
                    "exit_code": EXIT_DANGER_BLOCKED, "danger_blocked": True}

        # 本地无沙箱边界提示（一次性，避免 smoke/full/优化重跑反复刷屏）
        if not getattr(self, "_warned_unsandboxed", False):
            self._warned_unsandboxed = True
            self.log("execute_local", "WARNING",
                     "本地模式无沙箱隔离，仅用于演示/可信代码；"
                     "生产复现请 use_docker=True")

        cleanup = workdir is None
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="autorepro_exec_")
        else:
            os.makedirs(workdir, exist_ok=True)
            # 必须转绝对路径：脚本以 `[python, os.path.join(workdir, "run.py")]`
            # 启动、同时 cwd=workdir。workdir 若是相对路径（如 Optimizer 真实
            # 执行传入的 `data/_e2e_ws`），脚本参数会被 cwd 再解析一次，实际去找
            # `data/_e2e_ws/data/_e2e_ws/run.py` —— 报 "No such file or directory"，
            # exit_code=2、stdout 为空，看起来像"补丁跑不起来"。
            workdir = os.path.abspath(workdir)

        # 依赖预装：缺失依赖时运行必然失败，先安装再执行
        deps_err = self._ensure_local_deps(workdir)
        if deps_err:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)
            return {"success": False, "stdout": "",
                    "stderr": deps_err, "exit_code": -4,
                    "deps_prepared": False}

        script = os.path.join(workdir, "run.py")
        timeout = LOCAL_TIMEOUT_SMOKE if stage == "smoke" else LOCAL_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)

            result = self._run_local_script(script, workdir, timeout)
            # 运行时缺模块自愈：识别缺失模块 -> 隔离安装 -> 重跑（≤3 轮）
            healed = []
            for _ in range(MAX_PIP_SELF_HEAL):
                if result.get("success"):
                    break
                module = find_missing_module(result.get("stderr", "") or "")
                if not module:
                    break
                err = self._heal_install_local(module)
                heal = {"module": module,
                        "package": python_package_for(module),
                        "ok": err is None, "error": err}
                healed.append(heal)
                if err:
                    break
                result = self._run_local_script(script, workdir, timeout)
            if healed:
                result = {**result, "healed": healed}
            result["deps_prepared"] = True
            return result
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1,
                    "deps_prepared": True}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2, "deps_prepared": True}
        finally:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    def _run_local_script(self, script: str, workdir: str,
                          timeout: int) -> Dict:
        """执行 run.py 一次（子进程，注入隔离依赖 PYTHONPATH）。"""
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=timeout,
            # 与 _exec_env 的 PYTHONIOENCODING=utf-8 配套：显式指定 UTF-8 解码，
            # 不依赖系统 locale。缺了它，Windows 中文环境下捕获中文输出会抛
            # UnicodeDecodeError，stdout 变成 None。
            encoding="utf-8", errors="replace",
            cwd=workdir, env=self._exec_env())
        return {
            "success": result.returncode == 0,
            # `or ""` 兜底：解码失败等异常路径下 stdout/stderr 可能是 None，
            # 而 None 会让下游 `full.get("stdout", "")[-300:]` 直接崩
            # （key 存在但值为 None 时默认值不生效）。
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "exit_code": result.returncode,
            "deps_prepared": True,
        }

    def _pip_env(self) -> Dict:
        """本地 pip 子进程环境：强制关闭 user 安装。

        本机（真实模式实测）site 级 pip.ini 里写死了 `install.user = yes`，
        pip 于是总是追加 `--user`，与隔离安装的 `--target` 互斥，直接报
        `ERROR: Can not combine '--user' and '--target'` —— 结果**任何**
        真实模式依赖都装不上、代码永远跑不起来。

        PIP_USER=0 的优先级高于配置文件（环境变量 > 配置文件），与命令行
        的 `--no-user` 双保险。PYTHONIOENCODING 与脚本执行一致，保证 pip
        自己的输出也按 UTF-8 编码，父进程按 UTF-8 解码不会乱码。
        """
        env = os.environ.copy()
        env["PIP_USER"] = "0"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _heal_install_local(self, module: str) -> Optional[str]:
        """把缺失模块对应 PyPI 包隔离安装到 data/deps/heal-<module>/。

        None 表示成功（含 mock 模式短路与磁盘 ready 复用）；
        返回诊断文本表示安装失败。heal 目录独立于依赖清单目录，
        避免污染清单缓存的 .ready 语义；同一模块全局只装一次。
        """
        package = python_package_for(module)
        if self.mock_mode:
            # mock 演示：不触网、不装大包，视为就绪
            self._deps_dir == self._deps_dir  # noqa: B015 保持无副作用
            return None
        heal_dir = DEPS_CACHE_ROOT / f"heal-{module}"
        ready_mark = heal_dir / _DEPS_READY_MARK
        if ready_mark.is_file():
            self._heal_dirs.add(str(heal_dir))
            touch_deps_meta(heal_dir)       # 与清单目录同口径，便于统一清理
            self.log("self_heal", "RUNNING",
                     f"复用自愈目录 {heal_dir.name}（{package}）")
            return None
        try:
            heal_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "pip", "install",
                   "--disable-pip-version-check", "-q",
                   "--no-user",
                   "--target", str(heal_dir),
                   "-i", PIP_INDEX_URL]
            if PIP_FIND_LINKS:
                cmd += ["--find-links", PIP_FIND_LINKS]
            cmd += [package]
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 env=self._pip_env(),
                                 timeout=LOCAL_PIP_TIMEOUT)
            if res.returncode == 0:
                ready_mark.write_text("ok\n", encoding="utf-8")
                self._heal_dirs.add(str(heal_dir))
                _write_deps_meta(heal_dir, "heal", module=module)
                self.log("self_heal", "SUCCESS",
                         f"自愈安装完成 {module}->{package}: {heal_dir.name}")
                return None
            detail = (res.stderr or res.stdout or "").strip()[-400:]
            return f"自愈安装失败({package} exit={res.returncode}): {detail}"
        except subprocess.TimeoutExpired:
            return f"自愈安装超时({LOCAL_PIP_TIMEOUT}s): {package}"
        except Exception as e:
            return f"自愈安装异常: {e}"

    def _ensure_local_deps(self, workdir: str) -> Optional[str]:
        """确保本地执行环境已安装论文依赖；None 表示就绪，否则返回诊断文本。

        依赖来源与 Docker 路径一致：优先 env_config.requirements_txt，
        否则回退 required_packages。安装走 `pip install --target`
        （国内镜像 + find-links，与 EnvBuilder 同源），目标目录
        data/deps/<依赖清单 sha1[:16]>/；成功/失败均缓存到进程级
        _INSTALLED_DEPS + 磁盘 .ready 就绪标记，避免 smoke/full/优化
        重跑重复安装，同一依赖清单跨论文全局只装一次（L0 热缓存去重）。
        mock_mode=True 时跳过真实安装（mock 演示不触网、不装大包）。
        """
        env_config = getattr(self, "env_config", None) or {}
        reqs = (env_config.get("requirements_txt") or "").strip()
        if not reqs:
            pkgs = env_config.get("required_packages") or []
            if isinstance(pkgs, list):
                reqs = "\n".join(str(p) for p in pkgs if p).strip()
        if not reqs:
            return None

        key = reqs
        resource_id = reqs_digest(reqs)
        self.resource_events.emit(
            "dependency", resource_id, "install", "running",
            requirements=normalize_requirements(reqs))
        if key in _INSTALLED_DEPS:
            self.resource_events.emit(
                "dependency", resource_id, "install", "cached",
                detail="进程内依赖状态缓存命中")
            return _INSTALLED_DEPS[key] or None

        req_file = os.path.join(workdir, "requirements.txt")
        with open(req_file, "w", encoding="utf-8") as f:
            f.write(reqs)
        self.log("install_deps", "RUNNING",
                 f"按依赖清单安装环境依赖: {reqs[:120]}...")

        # ---- 隔离安装目录（对齐三层存储 L0 热缓存）----
        # 按**归一化**清单取哈希：清单写法差异（行序/空行/重复行）不再各存
        # 一份完整依赖（实测因此白占 53 MB）。
        deps_dir = DEPS_CACHE_ROOT / reqs_digest(reqs)
        ready_mark = deps_dir / _DEPS_READY_MARK

        if self.mock_mode:
            # mock 演示：不触网、不装大包，直接视为就绪
            self._deps_dir = None
            _INSTALLED_DEPS[key] = ""
            self.resource_events.emit(
                "dependency", resource_id, "install", "skipped",
                detail="mock_mode 跳过真实依赖安装")
            self.log("install_deps", "SUCCESS",
                     "mock_mode 跳过真实依赖安装")
            return None

        # 磁盘就绪检测：同一依赖清单已在隔离目录装过则直接复用
        if ready_mark.is_file():
            self._deps_dir = str(deps_dir)
            _INSTALLED_DEPS[key] = ""
            touch_deps_meta(deps_dir)       # 刷新 last_used，供冷热清理判断
            self.log("install_deps", "SUCCESS",
                     f"复用隔离依赖目录: {deps_dir.name}")
            self.resource_events.emit(
                "dependency", resource_id, "install", "cached",
                path=str(deps_dir), bytes=self._path_bytes(deps_dir))
            return None

        try:
            deps_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "pip", "install",
                   "--disable-pip-version-check", "-q",
                   "--no-user",
                   "--target", str(deps_dir),
                   "-i", PIP_INDEX_URL]
            if PIP_FIND_LINKS:
                cmd += ["--find-links", PIP_FIND_LINKS]
            cmd += ["-r", req_file]
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 env=self._pip_env(),
                                 timeout=LOCAL_PIP_TIMEOUT)
            if res.returncode == 0:
                ready_mark.write_text("ok\n", encoding="utf-8")
                self._deps_dir = str(deps_dir)
                _INSTALLED_DEPS[key] = ""
                _write_deps_meta(deps_dir, "reqs", requirements=reqs)
                self.log("install_deps", "SUCCESS",
                         f"隔离依赖安装完成: {deps_dir.name}")
                self.resource_events.emit(
                    "dependency", resource_id, "install", "succeeded",
                    path=str(deps_dir), bytes=self._path_bytes(deps_dir))
                return None
            detail = (res.stderr or res.stdout or "").strip()[-800:]
            _INSTALLED_DEPS[key] = (
                f"依赖安装失败(exit={res.returncode}), 无法在本地环境执行: "
                f"{detail}\n依赖清单: {reqs[:200]}...")
        except subprocess.TimeoutExpired:
            _INSTALLED_DEPS[key] = (
                f"依赖安装超时({LOCAL_PIP_TIMEOUT}s), 无法在本地环境执行: "
                f"{reqs[:200]}...")
        except Exception as e:      # 连失败原因都拿不到（如 pip 自身异常）
            _INSTALLED_DEPS[key] = f"依赖安装异常: {e}"
        self.log("install_deps", "ERROR", _INSTALLED_DEPS[key][:200])
        self.resource_events.emit(
            "dependency", resource_id, "install", "failed",
            detail=_INSTALLED_DEPS[key][:500])
        return _INSTALLED_DEPS[key]

    @staticmethod
    def _path_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*")
                   if item.is_file())

    def _exec_env(self) -> Dict:
        """构造子进程执行环境：依赖隔离目录存在时注入 PYTHONPATH。

        隔离安装的包（data/deps/<hash>/ + 自愈 heal-<module>/ 目录）经
        PYTHONPATH 前置，使子进程 import 优先命中隔离目录，不污染全局
        site-packages；无隔离目录时返回环境副本（行为与改造前一致）。
        """
        env = os.environ.copy()
        # 钉死子进程的标准流编码：父进程按 UTF-8 解码捕获到的输出，子进程
        # 就必须按 UTF-8 写出。否则在 Windows 中文环境下子进程默认用 GBK 写、
        # 父进程按 locale 解码，一旦生成代码打印中文/非 GBK 字节，reader 线程
        # 抛 UnicodeDecodeError，`stdout` 直接变成 None（后续切片即崩）。
        env["PYTHONIOENCODING"] = "utf-8"
        paths = []
        if self._deps_dir:
            paths.append(self._deps_dir)
        # 自愈目录排序注入，保证多模块顺序确定
        paths.extend(sorted(self._heal_dirs))
        if paths:
            existing = env.get("PYTHONPATH", "")
            joined = os.pathsep.join(paths)
            if existing:
                env["PYTHONPATH"] = joined + os.pathsep + existing
            else:
                env["PYTHONPATH"] = joined
        return env

    # ---------------- P1-⑪ Docker 沙箱加固 ----------------

    def _image_allowed(self, image: str) -> bool:
        """镜像白名单：只允许官方/自建镜像前缀，拒绝任意第三方镜像拉取执行。

        白名单可经 AUTOREPRO_DOCKER_IMAGE_ALLOWLIST 扩展（逗号分隔前缀）。
        """
        return any(image.startswith(prefix) for prefix in DOCKER_IMAGE_ALLOWLIST)

    def _sandbox_args(self, level: int = 0) -> List[str]:
        """按加固级别构造 docker run 参数。

        level 0（完整加固）：cap-drop ALL + no-new-privileges + 只读 rootfs
                             + tmpfs + 非 root + CPU/mem/pids 限额；
        level 1：去掉资源限额（老版本 Docker 不支持 --cpus/--pids-limit 时）；
        level 2（最小隔离）：仅 cap-drop + no-new-privileges（极端环境兜底）。
        """
        args = ["--cap-drop", "ALL",
                "--security-opt", "no-new-privileges"]
        if level >= 2:
            return args
        # tmpfs 的 exec 必须显式给：Docker `--tmpfs` 默认挂载选项是
        # rw,nosuid,nodev,**noexec**（本机实测 mount 输出），而加固模式下
        # pip --target 把依赖装进 /tmp/site-packages —— C 扩展的 .so 需要
        # mmap(PROT_EXEC)，noexec 下 numpy 直接
        # "failed to map segment from shared object"（Permission denied，126）。
        # 保留 nosuid/nodev：要挡的是 setuid 与设备节点，不是「执行刚装进来的
        # 库」——容器里跑的本来就是不可信代码，它本来就要被执行。
        args += ["--read-only",
                 "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=256m",
                 "--user", DOCKER_DEFAULT_USER]
        if level <= 0:
            args += ["--cpus", f"{DOCKER_DEFAULT_CPUS}",
                     "--memory", DOCKER_DEFAULT_MEM,
                     "--pids-limit", f"{DOCKER_DEFAULT_PIDS}"]
        return args

    def _run_docker_cmd_with_sandbox(
            self, base_cmd: List[str], image: str, runner: List[str],
            timeout: int) -> tuple:
        """带加固参数执行 docker run（含容器耗时计量，P1-⑫）。

        把实际执行委托给 _run_docker_cmd_with_sandbox_impl，
        无论成功/降级/非加固/超时路径，都在 finally 中以真实墙钟
        时长归入当前 plan 的 exec_calls/exec_seconds（无 plan 上
        下文时归入 unattributed 桶），支撑容器耗时的可审计核算。
        """
        t0 = time.monotonic()
        container_name = ""
        if getattr(self, "experiment_profile", ""):
            container_name = "autorepro-demo-" + uuid.uuid4().hex
            base_cmd = base_cmd + ["--name", container_name]
        try:
            return self._run_docker_cmd_with_sandbox_impl(
                base_cmd, image, runner, timeout)
        except subprocess.TimeoutExpired:
            # Killing the Docker CLI alone leaves training running in the
            # daemon. Remove only the container created for this bounded step.
            if container_name:
                try:
                    subprocess.run([base_cmd[0], "rm", "--force", container_name],
                                   capture_output=True, timeout=15)
                except (OSError, subprocess.SubprocessError):
                    pass
            raise
        finally:
            self.logger.record_sandbox_exec(round(time.monotonic() - t0, 3))

    def _run_docker_cmd_with_sandbox_impl(
            self, base_cmd: List[str], image: str, runner: List[str],
            timeout: int) -> tuple:
        """带加固参数执行 docker run；加固参数与 Docker/环境不兼容时自动降级。

        实际执行体（含降级链）。

        降级链（level 0 -> 1 -> 2）：失败 stderr 命中不兼容特征（unknown flag
        / permission denied / read-only file system 等）才降级；与加固无关的
        失败（缺模块、代码错误）不降级，直接返回以便上层自愈。超时直接抛出，
        不在加固级别间重试（避免重复等待）。返回 (subprocess.CompletedProcess,
        sandbox 元信息 dict)。
        """
        # encoding/errors 必须显式给：docker 的构建与运行输出是 UTF-8，父进程
        # 不指定就按系统 locale（Windows 中文 = GBK）解码，遇到非 GBK 字节
        # reader 线程抛 UnicodeDecodeError，`stdout` 变成 None —— 下游把它塞进
        # 报告时 `"\n".join(lines)` 直接崩（与 _run_local_script 同一根因）。
        if not DOCKER_HARDEN:
            result = subprocess.run(
                base_cmd + [image] + runner,
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace")
            return result, {"hardened": False, "level": None, "degraded": False}
        result = None  # 循环内必赋值；None 仅用于静态类型安抚
        last_meta: Dict = {"hardened": True, "level": 0, "degraded": False}
        for level in range(3):
            args = self._sandbox_args(level)
            cmd = base_cmd + args + [image] + runner
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                    encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                raise
            meta = {"hardened": True, "level": level,
                    "degraded": level > 0}
            last_meta = meta
            if result.returncode == 0:
                return result, meta
            stderr = (result.stderr or "").lower()
            if not any(hint in stderr
                       for hint in _HARDEN_INCOMPATIBLE_HINTS):
                # 失败与加固无关（缺模块 / 代码运行错误）——不降级
                return result, meta
        # level 2（最小隔离）仍失败或加固参数不兼容 → 用最后一级元信息返回
        return result, last_meta

    def _execute_code_docker(self, code: str, stage: str,
                             workdir: Optional[str] = None) -> Dict:
        """在 Docker 容器中执行代码（挂载临时目录或指定目录，隔离运行）。

        镜像选择：优先使用 env_config.image_tag（如流水线 EnvBuilder 已构建的
        autorepro-env 镜像，内含 requirements 依赖）；否则退回 python:3.11-slim，
        并把 env_config 中的 requirements 注入容器临时安装后执行。

        P1-⑪ 沙箱加固：
        - 引擎守卫：进入沙箱前探测 daemon 存活，不可用直接返回可操作提示
          （exit_code EXIT_DOCKER_DAEMON_DOWN），不产生无意义的 docker run；
        - 镜像白名单：非官方/自建镜像前缀直接拒绝执行（exit_code -5）；
        - 加固参数：cap-drop ALL / no-new-privileges / 只读 rootfs + tmpfs /
          非 root（nobody）/ CPU·mem·pids 限额，随 Docker 可用性自动降级；
          加固开启时 pip 安装到 tmpfs（/tmp/site-packages）并注入 PYTHONPATH，
          兼容只读 rootfs 与非 root 用户。
        """
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "stdout": "",
                    "stderr": "本机未安装 Docker 或不在 PATH 中", "exit_code": -3}

        env_config = getattr(self, "env_config", None) or {}
        image = env_config.get("image_tag") or "python:3.11-slim"
        # 镜像白名单：拒绝非白名单前缀镜像，防止恶意/未知镜像进入沙箱
        if not self._image_allowed(image):
            return {
                "success": False, "stdout": "",
                "stderr": (f"镜像 {image} 不在允许白名单 "
                           f"({'、'.join(DOCKER_IMAGE_ALLOWLIST)})，"
                           "已拒绝执行；可用 AUTOREPRO_DOCKER_IMAGE_ALLOWLIST "
                           "扩展白名单（逗号分隔前缀）"),
                "exit_code": -5,
                "sandbox": {"image_allowed": False, "image": image},
            }

        # 引擎存活守卫：CLI 存在 != daemon 在跑。Docker Desktop 装了没启动时
        # 直接 `docker run` 只会拿到 npipe 原始报错（还会在加固降级链里白跑
        # 三级）。先探测一次（实测失败仅 188ms），把原始报错换成人话——
        # 报告里的「错误输出」段是用户真正会读的地方。
        # 位置在白名单之后：镜像否决是关于镜像本身的安全判定，不该被
        # 「引擎没起」掩盖（daemon 不在时把非白名单镜像放行更不行）。
        engine_ok, engine_reason = BaseAgent.docker_engine_available([docker_cmd])
        if not engine_ok:
            return {
                "success": False, "stdout": "",
                "stderr": (f"Docker 引擎不可用：{engine_reason}。"
                           "请启动 Docker Desktop 后重试；或在侧边栏关闭"
                           "「Docker 沙箱执行」改用本地隔离执行。"),
                "exit_code": EXIT_DOCKER_DAEMON_DOWN,
                "sandbox": {"engine_available": False},
            }

        # 镜像可用性：本地缺失时按镜像源预拉（与计划路径同一道加固）。
        # 返回专属退出码，诊断层据此判定为运行环境问题，不再触发 LLM 改代码。
        pull_err = BaseAgent.ensure_image_pulled(docker_cmd, image)
        if pull_err:
            return {"success": False, "stdout": "", "stderr": pull_err,
                    "exit_code": EXIT_DOCKER_IMAGE_UNAVAILABLE,
                    "image_unavailable": True,
                    "sandbox": {"image_allowed": True, "image": image,
                                "image_unavailable": True}}

        reqs = (env_config.get("requirements_txt") or "").strip()
        if not reqs:
            pkgs = env_config.get("required_packages") or []
            if isinstance(pkgs, list):
                reqs = "\n".join(str(p) for p in pkgs if p).strip()

        cleanup = workdir is None
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="autorepro_docker_")
        else:
            os.makedirs(workdir, exist_ok=True)
            # 同本地路径：相对 workdir 会被 docker 的 -v 以错误形式解析
            workdir = os.path.abspath(workdir)
        script = os.path.join(workdir, "run.py")
        # mkdtemp 是 0700：容器里的 nobody 需要 o+rx 才能 traverse /app 并读
        # run.py，只给 o+w 会得到 0702，等于进不去。
        self._chmod_tree_writable(Path(workdir))
        timeout = DOCKER_TIMEOUT_SMOKE if stage == "smoke" else DOCKER_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            mount = workdir.replace("\\", "/")
            base_cmd = [docker_cmd, "run", "--rm",
                        "-v", f"{mount}:/app", "-w", "/app"]
            healed: list = []
            # runner 命令构造：python:3.11-slim 基础镜像场景把 requirements
            # 与自愈补装包都前置到 pip 安装（容器每次 --rm 不保留现场，
            # 缺包必须累积进命令重跑）；自定义 image_tag 镜像假定已含依赖，
            # 仅做脚本运行（缺包时同样改走 pip 前置自愈）。
            # 依赖装到**挂载卷内**的目录（默认 /app/.autorepro_site），并以
            # PYTHONPATH 注入，使 run.py 能导入新增依赖。装到 /tmp 的 tmpfs
            # 或容器层都不行：脚本步是另一个容器，那些内容已经没了。
            pip_target = self._prepare_pip_target(workdir)

            def _make_install_runner(heal_pkgs: list) -> list:
                """只安装依赖、不跑脚本（独立预算，见 DOCKER_INSTALL_TIMEOUT）。"""
                install_parts = [f"pip install -i {PIP_INDEX_URL} ",
                                 f"--find-links {PIP_FIND_LINKS} "]
                if pip_target:
                    install_parts.append(f"--target {pip_target} "
                                         "--no-cache-dir ")
                if reqs:
                    install_parts.append("-r /app/requirements.txt ")
                if heal_pkgs:
                    install_parts.append(" ".join(heal_pkgs) + " ")
                install_parts.append("-q")
                return ["sh", "-c", "".join(install_parts)]

            def _make_script_runner() -> list:
                if pip_target:
                    # PATH 带上 <target>/bin：console script（gdown 等）
                    # 只装命令不装可 import 入口，缺了它就会 command not found。
                    return ["sh", "-c",
                            f"PYTHONPATH={pip_target} "
                            f"PATH={pip_target}/bin:$PATH python run.py"]
                return ["python", "run.py"]

            reqs_file = None
            if image == "python:3.11-slim" and reqs:
                reqs_file = os.path.join(workdir, "requirements.txt")
                with open(reqs_file, "w", encoding="utf-8") as f:
                    f.write(reqs)

            # 运行时缺模块自愈（≤MAX_PIP_SELF_HEAL 轮）：识别缺失模块 ->
            # 累积进 pip 前置命令 -> 重跑；容器现场不保留，所以每轮都
            # 携带全部已识别缺包。
            result = None
            sandbox_meta: Dict = {"hardened": DOCKER_HARDEN,
                                  "image_allowed": True}
            seen = set()
            for _ in range(MAX_PIP_SELF_HEAL + 1):
                # 预算分离：依赖安装与脚本运行分两次 docker run、分开计时。
                # 此前合并为一条 `pip install ... && python run.py`，安装
                # torch 等大包直接吃掉整个 smoke 超时（iTransformer 实测）。
                needs_install = bool(
                    (image == "python:3.11-slim" and reqs) or healed)
                if needs_install:
                    heal_pkgs = [h["package"] for h in healed]
                    try:
                        install_result, _sandbox_install = \
                            self._run_docker_cmd_with_sandbox(
                                base_cmd, image,
                                _make_install_runner(heal_pkgs),
                                DOCKER_INSTALL_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        # 措辞含「执行超时」以命中 _diagnose_execution_error
                        # 的 timeout 分支（repairable=True，触发缩参修复）。
                        return {"success": False, "stdout": "",
                                "stderr": (f"依赖安装执行超时("
                                           f"{DOCKER_INSTALL_TIMEOUT}s)"),
                                "exit_code": -1,
                                "sandbox": {"image_allowed": True,
                                            "install_phase": True}}
                    if install_result.returncode != 0:
                        # 依赖装不上（找不到发行版/网络失败）：以安装结果
                        # 直接返回，避免把 pip 报错误判为脚本运行时错误。
                        result = install_result
                        sandbox_meta = {"image_allowed": True,
                                        "install_phase": True,
                                        **_sandbox_install}
                        break
                result, _sandbox_run = self._run_docker_cmd_with_sandbox(
                    base_cmd, image, _make_script_runner(), timeout)
                sandbox_meta = {"image_allowed": True, **_sandbox_run}
                if result.returncode == 0:
                    break
                module = find_missing_module(result.stderr or "")
                if not module or module in seen \
                        or len(healed) >= MAX_PIP_SELF_HEAL:
                    break
                seen.add(module)
                healed.append({"module": module,
                               "package": python_package_for(module),
                               "ok": True, "error": None})
                self.log("self_heal", "RUNNING",
                         f"Docker 缺模块 {module},累积重跑")
            # 循环至少执行一次（MAX_PIP_SELF_HEAL >= 0），result 必已赋值
            assert result is not None
            result = {
                "success": result.returncode == 0,
                # `or ""` 兜底：解码失败等异常路径下 stdout/stderr 可能是 None，
                # 而 None 会让下游 `full.get("stdout", "无输出")` 拿到 None、
                # 报告 `"\n".join(lines)` 崩（dict.get 的默认值对 None 不生效）。
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "exit_code": result.returncode,
                "sandbox": sandbox_meta,
            }
            if healed:
                result["healed"] = healed
            return result
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2}
        finally:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    # ---------------- 内部工具 ----------------

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    def extract_result_files(self, result: Dict) -> List[str]:
        """从执行产物中收集数值结果/文件（对齐方案的输出采集）。"""
        files = []
        for artifact in ("stdout", "stderr"):
            text = result.get(artifact, "") or ""
            for line in text.splitlines():
                if "=" in line and any(ch.isdigit() for ch in line):
                    files.append(line.strip())
        return files
