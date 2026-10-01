"""执行计划（ExecutionPlan）——多代码单元论文「整体调用」的计划层数据模型。

把「一篇论文由多个代码块组成、最后按顺序整体调用」显式建模为有序步骤
（安装 -> 数据下载/准备 -> 训练运行 -> 解析指标），每条步骤：
- 归属某个 CodeUnit（cwd = 容器内 /app/<unit_id>）；
- 只允许引用真实存在于仓库快照中的文件（validate_plan 强制，防 LLM 幻觉路径）；
- 脚本运行预算（timeout_s）与依赖安装预算（install_budget_s）分离；
- 携带 smoke_args（缩参版），供超时修复时自动降参重试；
- depends_on 表达步骤间依赖，前置失败则跳过（skipped_deps）。

计划由 ExecutionPlannerAgent 生成（LLM 优先、启发式兜底、无单元则空计划
source="none"，CodeExecutor 走回退路径），落盘 data/plans/<paper_id>.json。
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.code_units import CodeUnit

# 步骤种类
STEP_KINDS = ("install", "download", "prepare", "run", "parse")

# 计划来源：llm（LLM 生成）/ heuristic（确定性启发式兜底）/ none（无可用单元）
PLAN_SOURCES = ("llm", "heuristic", "none", "profile")

# 数据集线索 URL 特征（README 中的网盘/下载链接）。
# 前置字符类用 *（可空）：否则它会贪婪吞掉域名部分，导致永远无法匹配。
_DATASET_LINK_RE = re.compile(
    r"https?://[^\s\"'<>()]*(?:drive\.google\.com|cloud\.tsinghua|"
    r"pan\.baidu|docs\.google\.com|figshare|zenodo|1drv\.ms)[^\s\"'<>()]*",
    re.IGNORECASE)

# 快照边界：防大仓库扫爆（深度 4 / 最多 500 文件 / 跳过二进制与 .git）
_SNAPSHOT_MAX_DEPTH = 4
_SNAPSHOT_MAX_FILES = 500
_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".a", ".o",
                  ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz",
                  ".tgz", ".7z", ".rar", ".pdf", ".csv", ".parquet", ".bin",
                  ".pt", ".pth", ".ckpt", ".onnx", ".wav", ".mp4", ".npy"}

# 脚本运行预算默认值（秒）：run=180（smoke 风格，含独立安装前缀不计入）、
# 安装=600、下载=1200；可用 AUTOREPRO_STEP_BUDGET_RUN/INSTALL/DOWNLOAD 覆盖。
_STEP_BUDGETS = {
    "run": int(os.environ.get("AUTOREPRO_STEP_BUDGET_RUN", "180")),
    "install": int(os.environ.get("AUTOREPRO_STEP_BUDGET_INSTALL", "600")),
    "download": int(os.environ.get("AUTOREPRO_STEP_BUDGET_DOWNLOAD", "1200")),
    "prepare": int(os.environ.get("AUTOREPRO_STEP_BUDGET_PREPARE", "300")),
    "parse": 30,
}


@dataclass
class PlanStep:
    step_id: str = ""          # 唯一 ID："install_main" / "run_0" / "parse"
    kind: str = "run"          # STEP_KINDS 之一
    cmd: str = ""              # shell 命令（相对 cwd）
    cwd: str = ""              # 容器内目录："/app/<unit_id>"（"" = /app）
    unit_id: str = ""          # 归属单元（"" = 合成步骤）
    args: Dict = field(default_factory=dict)      # 论文超参（--seq_len 等）
    timeout_s: int = 0         # 脚本运行预算（不含安装）
    install_budget_s: int = 0  # pip/gdown 前缀预算（0 = 无安装前缀）
    install_pkgs: List[str] = field(default_factory=list)  # 追加安装的包
    depends_on: List[str] = field(default_factory=list)    # 前置 step_id
    expects: Dict = field(default_factory=dict)   # {"metrics": ["mse","mae"]}
    smoke_args: Dict = field(default_factory=dict)  # 超时缩参重试用
    smoke_cmd: str = ""        # 缩参后的**完整命令**（bash 入口用，见下）
    env: Dict = field(default_factory=dict)      # 额外环境变量
    retries: int = 0           # 已修复重试次数（执行时填充）

    # 为什么需要 smoke_cmd（与 smoke_args 并存）：
    # 官方入口常见形态是 `bash scripts/<task>/<ds>/<model>.sh`，而 .sh 内部是
    # 「一次导出 CUDA_VISIBLE_DEVICES + 4 段 python -u run.py（pred_len
    # 96/192/336/720）」。这类脚本**不转发 "$@"**，把 smoke_args 追加到
    # `bash x.sh` 后面会被 shell 直接忽略——缩参重试变成同一件事再跑一遍，
    # 白烧一整个超时预算。所以 bash 入口用 derive_smoke_cmd 从脚本里取出
    # 第一段 python 调用并注入缩参，作为可真正生效的 smoke_cmd。

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id, "kind": self.kind, "cmd": self.cmd,
            "cwd": self.cwd, "unit_id": self.unit_id,
            "args": dict(self.args), "timeout_s": self.timeout_s,
            "install_budget_s": self.install_budget_s,
            "install_pkgs": list(self.install_pkgs),
            "depends_on": list(self.depends_on),
            "expects": dict(self.expects),
            "smoke_args": dict(self.smoke_args),
            "smoke_cmd": self.smoke_cmd,
            "env": dict(self.env), "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlanStep":
        step = cls()
        step.step_id = str(data.get("step_id") or "")
        step.kind = str(data.get("kind") or "run")
        step.cmd = str(data.get("cmd") or "")
        step.smoke_cmd = str(data.get("smoke_cmd") or "")
        step.cwd = str(data.get("cwd") or "")
        step.unit_id = str(data.get("unit_id") or "")
        for key in ("args", "expects", "smoke_args", "env"):
            value = data.get(key)
            setattr(step, key, dict(value) if isinstance(value, dict) else {})
        for key in ("install_pkgs", "depends_on"):
            value = data.get(key)
            setattr(step, key, list(value) if isinstance(value, list) else [])
        for key in ("timeout_s", "install_budget_s", "retries"):
            try:
                setattr(step, key, int(data.get(key) or 0))
            except (TypeError, ValueError):
                setattr(step, key, 0)
        return step


@dataclass
class ExecutionPlan:
    paper_id: str = ""
    plan_version: int = 1
    source: str = "none"       # llm / heuristic / none
    units: List[CodeUnit] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    entry: Dict = field(default_factory=dict)  # {script, unit_id, interp}
    notes: List[str] = field(default_factory=list)
    datasets: List[Dict] = field(default_factory=list)
    experiment_profile: str = ""
    parameters: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "plan_version": self.plan_version,
            "source": self.source,
            "units": [u.to_dict() for u in self.units],
            "steps": [s.to_dict() for s in self.steps],
            "entry": dict(self.entry),
            "notes": list(self.notes),
            "datasets": list(self.datasets),
            "experiment_profile": self.experiment_profile,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionPlan":
        plan = cls()
        plan.paper_id = str(data.get("paper_id") or "")
        plan.plan_version = int(data.get("plan_version") or 1)
        plan.source = str(data.get("source") or "none")
        plan.units = [CodeUnit.from_dict(u) for u in (data.get("units") or [])
                      if isinstance(u, dict)]
        plan.steps = [PlanStep.from_dict(s) for s in (data.get("steps") or [])
                      if isinstance(s, dict)]
        plan.entry = dict(data.get("entry") or {})
        plan.notes = [str(n) for n in (data.get("notes") or [])]
        plan.datasets = [dict(d) for d in (data.get("datasets") or [])]
        plan.experiment_profile = str(data.get("experiment_profile") or "")
        plan.parameters = dict(data.get("parameters") or {})
        return plan

    def has_steps(self) -> bool:
        return bool(self.steps)


# ---------------- 确定性预分析（无 LLM） ----------------

def repo_snapshot(repo_dir: str) -> Dict:
    """有界目录扫描：文件清单 + scripts/*.sh + run.py + README + requirements。

    返回 {"files": [相对路径...], "scripts": [...], "run_pys": [...],
    "readmes": [...], "requirements": [...], "dataset_links": [URL...]}。
    超大仓库按边界截断并在 truncated 字段标注。
    """
    root = Path(repo_dir)
    files: List[str] = []
    scripts: List[str] = []
    run_pys: List[str] = []
    readmes: List[str] = []
    requirements: List[str] = []
    links: List[str] = []
    truncated = False

    if not root.is_dir():
        return {"files": [], "scripts": [], "run_pys": [], "readmes": [],
                "requirements": [], "dataset_links": [], "truncated": True}

    for current, dirs, names in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        if depth >= _SNAPSHOT_MAX_DEPTH:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in sorted(names):
            if len(files) >= _SNAPSHOT_MAX_FILES:
                truncated = True
                continue
            path = Path(current) / name
            rel = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            if suffix in _SKIP_SUFFIXES and not name.endswith(".sh"):
                continue
            files.append(rel)
            if name.endswith(".sh"):
                scripts.append(rel)
            if name in ("run.py", "main.py", "train.py", "trainer.py",
                        "evaluate.py", "test.py"):
                run_pys.append(rel)
            if name.lower().startswith("readme"):
                readmes.append(rel)
            if name in ("requirements.txt", "requirements.in",
                        "pyproject.toml", "setup.py", "environment.yml"):
                requirements.append(rel)
    # 数据集线索链接：从 README/文档类文本中提取
    for rel in readmes:
        try:
            text = (root / rel).read_text(encoding="utf-8",
                                          errors="replace")
        except OSError:
            continue
        for match in _DATASET_LINK_RE.finditer(text):
            link = match.group(0).rstrip(".,;")
            if link not in links:
                links.append(link)
    return {"files": files, "scripts": scripts, "run_pys": run_pys,
            "readmes": readmes, "requirements": requirements,
            "dataset_links": links[:20], "truncated": truncated}


def _match_hint(rel_path: str, hints: List[str]) -> bool:
    """路径是否命中任一提示词（数据集名/模型名/仓库名，小写匹配）。"""
    lowered = rel_path.lower()
    return any(hint and hint.lower() in lowered for hint in hints)


def detect_entry(snapshot: Dict, paper_info: Dict,
                 unit_name: str = "") -> Dict:
    """确定性入口检测：根 run.py -> scripts/**/<模型|数据集>.sh -> 兜底。

    返回 {"script": 相对路径, "interp": "python"|"bash", "reason": ...}；
    找不到返回 {}。优先级与 iTransformer（scripts/<task>/<ds>/<model>.sh）
    及 Time-Series-Library（根 run.py）两种结构对齐。
    """
    dataset = (paper_info.get("dataset") or "").strip()
    method = (paper_info.get("method") or "").strip()
    title = (paper_info.get("title") or "").strip()

    scripts = snapshot.get("scripts", []) or []
    # 非根级脚本（scripts/<task>/<dataset>/<model>.sh 这类官方入口）
    nested = [s for s in scripts if "/" in s]
    dataset_ok = bool(dataset) and not (title and dataset == title)

    # 0) 命中数据集的官方脚本**优先于根 run.py**。
    #
    # 这条顺序是被真实仓库推翻后改的：iTransformer 根目录确实有 run.py，
    # 按「根脚本优先」会选中它，而它的 argparse 默认值是
    # `--root_path ./data/electricity/ --data_path electricity.csv`——
    # 仓库里根本没有 data/ 目录，于是 `python run.py` 必然
    # FileNotFoundError。真正的官方调用在
    # scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh 里，
    # 只有它带着正确的 `--root_path ./dataset/ETT-small/`。
    # 数据集命中是强信号（脚本名/路径写死了数据集与数据路径），
    # 因此只在**数据集命中**时才越过根 run.py；论文没声明数据集时
    # 仍按原顺序走根 run.py，避免误选无关的 utilities 脚本。
    if dataset_ok:
        both = [s for s in nested if _match_hint(s, [dataset])
                and (not method or _match_hint(s, [method]))]
        if both:
            best = sorted(both, key=lambda p: (p.count("/"), p))[0]
            reason = f"官方脚本命中数据集: {dataset}"
            if method:
                reason += f" + 方法: {method}"
            return {"script": best, "interp": "bash", "reason": reason}
        only_ds = sorted((s for s in nested if _match_hint(s, [dataset])),
                         key=lambda p: (p.count("/"), p))
        if only_ds:
            return {"script": only_ds[0], "interp": "bash",
                    "reason": f"脚本路径命中数据集: {dataset}"}

    root_pys = [p for p in snapshot.get("run_pys", [])
                if "/" not in p and p in ("run.py", "main.py", "train.py")]
    if root_pys:
        py = next((p for p in ("run.py", "main.py") if p in root_pys),
                  root_pys[0])
        return {"script": py, "interp": "python",
                "reason": "仓库根入口脚本"}

    # 模型名命中（iTransformer: scripts/.../iTransformer.sh）
    if method:
        for rel in scripts:
            if _match_hint(rel, [method]):
                return {"script": rel, "interp": "bash",
                        "reason": f"脚本名命中论文方法: {method}"}
    # 兜底：任意根级或 scripts/ 下的第一个 .sh
    shallow = [s for s in scripts if s.count("/") <= 2]
    if shallow:
        return {"script": shallow[0], "interp": "bash",
                "reason": "兜底选择首个浅层脚本"}
    if scripts:
        return {"script": scripts[0], "interp": "bash",
                "reason": "兜底选择首个脚本"}
    return {}


def read_excerpts(repo_dir: str, snapshot: Dict,
                  readme_chars: int = 3000,
                  script_chars: int = 4000) -> Dict:
    """读取规划器需要的关键文件摘录（README/requirements/入口脚本/run.py 参数）。"""
    root = Path(repo_dir)
    excerpts: Dict = {"readme": "", "requirements": "", "entry": "",
                      "argparse_defaults": {}}
    readmes = snapshot.get("readmes") or []
    if readmes:
        try:
            text = (root / readmes[0]).read_text(encoding="utf-8",
                                                 errors="replace")
            excerpts["readme"] = text[:readme_chars]
        except OSError:
            pass
    reqs = snapshot.get("requirements") or []
    for rel in reqs:
        if rel.endswith("requirements.txt"):
            try:
                excerpts["requirements"] = (
                    root / rel).read_text(encoding="utf-8",
                                          errors="replace")[:2000]
                break
            except OSError:
                continue
    return excerpts


def extract_argparse_defaults(script_text: str) -> Dict:
    """从 run.py 文本中提取 argparse 默认值（--seq_len 96 -> 96）。

    只提取 add_argument('--x', default=V) 与 add_argument('--x', ...,
    type=int, default=V) 的确定性形态，供 smoke_args 缩参与命令构造。
    """
    defaults: Dict = {}
    pattern = re.compile(
        r"add_argument\(\s*['\"]--([\w\-]+)['\"][^)]*?"
        r"default\s*=\s*([^,)\s]+)", re.DOTALL)
    for name, value in pattern.findall(script_text):
        value = value.strip("'\"")
        if re.match(r"^-?\d+$", value):
            defaults[name] = int(value)
        elif value.lower() in ("true", "false"):
            defaults[name] = value.lower() == "true"
        elif value not in ("None",):
            defaults[name] = value
    return defaults


def extract_cli_args(script_text: str) -> Dict:
    """从入口脚本/run.py 文本中提取 --key value 超参（确定性）。

    覆盖 iTransformer 风格 .sh（--seq_len 96 --pred_len 96 ...）与
    run.py argparse 默认值两种来源；只保留 value 可解析的行。

    **首次出现优先**（`setdefault`）：官方 .sh 常把同一个超参在多个调用块里
    写成不同值（iTransformer 的 --pred_len 依次是 96/192/336/720），
    后出现的值只是同一脚本后面的 horizon，不代表论文主设置。此前的
    「后写覆盖」会让记录下来的 --pred_len 变成 720。
    """
    args: Dict = {}
    pattern = re.compile(r"--([\w\-]+)\s+([^\s\\\"]+)")
    for name, value in pattern.findall(script_text or ""):
        if name in args:
            continue
        value = value.strip("'\"")
        if re.match(r"^-?\d+(?:\.\d+)?$", value):
            try:
                args[name] = int(value) if value.isdigit() else float(value)
            except ValueError:
                continue
        elif value.lower() in ("true", "false"):
            args[name] = value.lower() == "true"
        else:
            args[name] = value
    return args


# shell 变量引用（$var / ${var}）：无法替换时整条命令作废
_SHELL_VAR_RE = re.compile(r"\$\{?(\w+)\}?")
# 变量赋值行（含 export）：model_name=iTransformer / export FOO=bar
_SH_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=(\S+)\s*$")


def _sh_variables(script_text: str) -> Dict:
    """收集 .sh 里的简单变量赋值（只认整行赋值，避免误吞命令行内容）。"""
    variables: Dict = {}
    for line in (script_text or "").splitlines():
        match = _SH_ASSIGN_RE.match(line)
        if match:
            variables[match.group(1)] = match.group(2).strip("'\"")
    return variables


def derive_smoke_cmd(script_text: str, smoke_args: Dict) -> str:
    """从官方 .sh 派生**可真正生效**的缩参命令（确定性，失败返回 ""）。

    动机：`bash scripts/<task>/<ds>/<model>.sh` 这类入口不转发 "$@"，
    把缩参追加到 bash 命令后面会被直接忽略——超时修复等于原地重跑。
    这里把脚本里的第一段 `python ... run.py ...`（带行尾反斜杠续行）
    取出来、替换掉 shell 变量、再追加缩参覆盖，得到一条单次调用的命令。

    只在脚本能被安全解析时返回命令；出现未定义变量/取不到 python 调用
    则返回 ""，调用方回退到原有行为。
    """
    text = (script_text or "").replace("\\\n", " ").replace("\\\r\n", " ")
    if not text.strip():
        return ""
    variables = _sh_variables(script_text)
    command = ""
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:python|python3|python[\d.]*)\s", stripped):
            command = stripped
            break
    if not command:
        return ""
    unresolved = False

    def _sub(match: "re.Match") -> str:
        nonlocal unresolved
        name = match.group(1)
        if name in variables:
            return variables[name]
        unresolved = True
        return match.group(0)

    command = _SHELL_VAR_RE.sub(_sub, command)
    if unresolved:
        return ""
    # 必须真的调用到某个 .py（否则可能取到无关的 python -c / 工具行）
    if ".py" not in command:
        return ""
    overrides = " ".join(f"{k} {v}" for k, v in (smoke_args or {}).items())
    # 续行拼接会留下连续空白：命令会进报告与审计日志，规整成单空格
    return re.sub(r"\s+", " ", f"{command} {overrides}").strip()


def suggest_smoke_args(args: Dict) -> Dict:
    """确定性缩参：只缩小已有键（epochs->1、seq_len/pred_len 减到 ≤32、
    batch_size ≤16），用于超时修复时自动降参重试。

    只允许改**成本类**参数。`enc_in/dec_in/c_out` 曾经在表里（缩到 4），
    但它们是**数据形状**参数——必须等于数据集变量数（ETTh1=7、Weather=21）。
    缩成 4 会让模型与真实数据的通道数对不上，把一次"超时"变成一次
    "shape 不匹配崩溃"，缩参重试反而更容易失败。故已移除。
    """
    shrunk: Dict = {}
    shrink_to = {"seq_len": 32, "pred_len": 32, "label_len": 16,
                 "input_len": 32, "d_model": 32, "d_ff": 64,
                 "batch_size": 16, "train_epochs": 1, "epochs": 1,
                 "itr": 1, "num_workers": 0}
    for key, value in (args or {}).items():
        limit = shrink_to.get(key)
        if limit is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if key in ("train_epochs", "epochs", "itr"):
            shrunk[key] = max(limit, 1)
        elif key == "num_workers":
            shrunk[key] = 0
        elif numeric > limit:
            shrunk[key] = limit
    return shrunk


def default_step_budgets(kind: str) -> int:
    """按步骤种类返回默认预算（可经 AUTOREPRO_STEP_BUDGET_* 覆盖）。"""
    return int(_STEP_BUDGETS.get(kind, 180))


# ---------------- 计划校验（防幻觉路径，失败关闭） ----------------

def validate_plan(plan: Dict, snapshots: Optional[Dict[str, Dict]] = None,
                  paper_id: str = "") -> List[str]:
    """确定性 schema 门：返回错误列表（空 = 通过）。

    硬约束：
    1. 步骤字段齐备（step_id/kind/cmd 非空、kind ∈ STEP_KINDS）；
    2. run/prepare/parse 步骤的命令必须引用真实存在于对应单元快照中的
       文件（snapshots 提供时）——LLM 幻觉路径在此失败关闭；
    3. cwd 要么为空要么映射到已知 unit_id；
    4. depends_on 只能指向先出现的 step_id；
    5. step_id 全局唯一且不含路径分隔符。
    """
    errors: List[str] = []
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return ["计划缺少 steps 列表"]
    seen_ids: set = set()
    known_units = {u.get("unit_id") for u in (plan.get("units") or [])
                   if isinstance(u, dict)}
    if snapshots:
        known_units.update(snapshots.keys())
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"steps[{index}] 非对象")
            continue
        step_id = str(step.get("step_id") or "").strip()
        kind = str(step.get("kind") or "").strip()
        cmd = str(step.get("cmd") or "").strip()
        if not step_id or "/" in step_id or "\\" in step_id:
            errors.append(f"steps[{index}] step_id 非法: {step_id!r}")
            continue
        if step_id in seen_ids:
            errors.append(f"steps[{index}] step_id 重复: {step_id}")
        seen_ids.add(step_id)
        if kind not in STEP_KINDS:
            errors.append(f"{step_id}: 未知 kind {kind!r}")
            continue
        if not cmd and kind != "parse":
            errors.append(f"{step_id}: 命令为空")
        cwd = str(step.get("cwd") or "")
        unit_id = str(step.get("unit_id") or "")
        if unit_id and unit_id not in known_units:
            errors.append(f"{step_id}: unit_id {unit_id!r} 不在计划单元中")
        if cwd and not cwd.lstrip("/").startswith("app/"):
            errors.append(f"{step_id}: cwd 必须在容器 /app 下: {cwd!r}")
        # 依赖只允许指向前置步骤
        for dep in (step.get("depends_on") or []):
            if dep not in seen_ids - {step_id}:
                errors.append(f"{step_id}: depends_on 指向未知/后置步骤 {dep}")
        # 命令必须引用真实文件（parse 与纯内联命令除外）
        if kind in ("run", "prepare") and cmd and snapshots:
            snapshot = (snapshots.get(unit_id)
                        or snapshots.get("main")
                        or {"files": [], "scripts": []})
            referenced = _cmd_references(cmd)
            files = snapshot.get("files") or []
            if referenced:
                missing = [ref for ref in referenced
                           if ref not in files]
                if missing:
                    errors.append(f"{step_id}: 命令引用不存在的文件 "
                                  f"{missing[:3]}")
    return errors


def _cmd_references(cmd: str) -> List[str]:
    """提取命令中引用的文件路径（相对路径形态，含 scripts/**/*.sh 与 *.py）。

    跳过：选项（- 开头）、环境变量赋值（首段含 =）、URL（含 ://）、
    绝对路径（/ 开头）。
    """
    refs: List[str] = []
    for token in re.split(r"\s+", cmd):
        token = token.strip("'\"")
        if not token or token.startswith("-") or "://" in token:
            continue
        head = token.split("/")[0]
        if "=" in head:
            continue
        if re.search(r"\.(sh|py)$", token) or "/" in token:
            refs.append(token.split(";")[0].split("&")[0].split("|")[0])
    return [r for r in refs if not r.startswith("/")]
