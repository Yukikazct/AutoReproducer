"""Execution policy for the first API-driven official CPU experiment.

The LLM reads repository evidence and proposes the training command. This
module validates that proposal and supplies data/environment/budget bindings;
it never replaces a missing LLM proposal with a preset or generated program.
"""
import ast
import platform
import re
import shlex
from pathlib import Path

from src.code_units import CodeUnit
from src.execution_plan import ExecutionPlan, PlanStep
from src.experiment_profiles import CPU_REQUIREMENTS, PARAMETERS

INTENT = "official_smoke"
REPO_URL = "https://github.com/cure-lab/LTSF-Linear"
ENTRY = "scripts/EXP-LongForecasting/Linear/etth1.sh"
RUNNER = "run_longExp.py"
DATA_TARGET = "dataset/ETT-small/ETTh1.csv"


def runner_arguments(source):
    """Read argparse declarations without importing third-party code."""
    arguments = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) \
                or node.func.attr != "add_argument":
            continue
        for value in node.args:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str) \
                    or not value.value.startswith("--"):
                continue
            settings = {}
            for keyword in node.keywords:
                if keyword.arg in ("default", "action"):
                    try:
                        settings[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass
            arguments[value.value] = settings
    return arguments


def is_real_smoke(data):
    return not data.get("mock_mode", False) and bool(
        data.get("experiment_profile") or data.get("execution_intent") == INTENT)


def repository_context(code):
    url = str(code.get("url", "")).rstrip("/").removesuffix(".git")
    if url.lower() != REPO_URL.lower():
        raise ValueError("本轮 API CPU 执行适配支持官方 LTSF-Linear / DLinear / ETTh1；检索到的仓库尚未适配")
    root = Path(code["path"]).resolve()
    documents = {}
    for name, budget in (("README.md", 10000), ("requirements.txt", 2000),
                         (ENTRY, 5000), (RUNNER, 12000), ("models/DLinear.py", 6000)):
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"官方仓库证据文件缺失或越界: {name}")
        documents[name] = path.read_text(encoding="utf-8")[:budget]
    return {"repo_url": url, "commit": code["commit"], "adapter": "ltsf_linear_cpu",
            "documents": documents}


def check_selection(selection, context):
    if not isinstance(selection, dict) or selection.get("model") != "DLinear" \
            or selection.get("dataset") != "ETTh1" \
            or selection.get("entry_script") != ENTRY or selection.get("runner") != RUNNER:
        raise ValueError("LLM 未从官方仓库识别出受支持的 DLinear / ETTh1 入口")
    evidence = selection.get("evidence_files")
    if not isinstance(evidence, list) or ENTRY not in evidence or any(
            not isinstance(name, str) or name not in context["documents"] for name in evidence):
        raise ValueError("LLM 的执行选择缺少真实入口文件证据")
    return {key: selection[key] for key in
            ("model", "dataset", "entry_script", "runner", "evidence_files")}


def cpu_environment(context):
    if context.get("adapter") != "ltsf_linear_cpu":
        raise ValueError("缺少已校验的官方仓库上下文")
    architecture = platform.machine().lower()
    if architecture not in ("arm64", "aarch64", "x86_64", "amd64"):
        raise ValueError(f"尚未适配的 CPU 架构: {architecture}")
    requirements = list(CPU_REQUIREMENTS) + ["scipy==1.15.3"]
    if architecture in ("x86_64", "amd64"):
        requirements[requirements.index("torch==2.0.0")] = "torch==2.0.0+cpu"
    return {"python_version": "3.11", "image_tag": "python:3.11-slim",
            "requirements_txt": "\n".join(requirements), "build_required": False,
            "upstream_requirements_txt": context["documents"]["requirements.txt"],
            "dependency_source": "validated_ltsf_cpu_compatibility", "architecture": architecture,
            "compatibility_notes": ["Python 3.11 使用 torch 2.0.0 CPU 兼容环境，代替官方 torch 1.9.0；算法源码不变。"]}


def build_llm_plan(data, proposal):
    context = data["repository_context"]
    selection = check_selection(proposal, context)
    raw_command = proposal.get("command")
    if not isinstance(raw_command, str):
        raise ValueError("LLM 未提供实际训练命令")
    tokens = shlex.split(raw_command)
    if tokens and tokens[0] in ("python", "python3"):
        tokens.pop(0)
    else:
        raise ValueError("仅支持直接调用官方 Python 入口")
    if tokens and tokens[0] == "-u":
        tokens.pop(0)
    if not tokens or tokens.pop(0) not in (RUNNER, "./" + RUNNER):
        raise ValueError("LLM 命令未引用真实官方入口")
    arguments = runner_arguments(context["documents"][RUNNER])
    flags = {}
    while tokens:
        key = tokens.pop(0)
        if key not in arguments or key in flags or not tokens:
            raise ValueError(f"LLM 命令包含未知、重复或无值参数: {key}")
        value = tokens.pop(0)
        if not re.fullmatch(r"[A-Za-z0-9_./+\-]+", value) or value.startswith("--"):
            raise ValueError(f"LLM 命令参数无效: {key}")
        if key in ("--train_only", "--do_predict", "--use_multi_gpu", "--use_amp", "--test_flop"):
            raise ValueError(f"CPU 冒烟不支持参数: {key}")
        if arguments[key].get("action", "store") != "store":
            raise ValueError(f"CPU 冒烟命令仅支持带值参数: {key}")
        flags[key] = value
    if flags.get("--model") != selection["model"] or flags.get("--data") != selection["dataset"]:
        raise ValueError("LLM 命令的模型/数据集与仓库证据不一致")
    requested_flags = dict(flags)
    # Bounds apply before the first run. Keep other valid upstream arguments
    # (e.g. learning rate) and record every override for review.
    flags.update({"--" + k: str(v) for k, v in PARAMETERS.items()})
    flags.update({"--is_training": "1", "--features": "M", "--data_path": "ETTh1.csv",
                  "--root_path": "/app/main/dataset/ETT-small/",
                  "--model_id": "DLinear_ETTh1_96_96_cpu_smoke"})
    fetched = data["storage"]["fetched"]
    dataset = fetched["dataset"]
    if dataset.get("data_kind") != "real" or dataset.get("state") not in ("downloaded", "cached"):
        raise ValueError("真实 ETTh1 尚未下载并校验")
    code = fetched["code"]
    if not code.get("commit") or code.get("state") not in ("cloned", "cached"):
        raise ValueError("官方仓库未通过校验")
    command = shlex.join(["python", "-u", RUNNER] + [item for kv in flags.items() for item in kv])
    effective_parameters = {key.removeprefix("--"): settings["default"]
                            for key, settings in arguments.items() if "default" in settings}
    effective_parameters.update({k.removeprefix("--"): v for k, v in flags.items()})
    # The upstream runner disables GPU when CUDA is unavailable. The environment
    # step verifies the CPU torch build before training.
    effective_parameters.update(seed=2021, device="cpu", use_gpu=False)
    plan = ExecutionPlan(paper_id=data["paper_id"], source="llm", execution_intent=INTENT,
                         units=[CodeUnit(unit_id="main", role="main", url=code["url"],
                                         local_path=code["path"], revision=code["commit"],
                                         fetch_state=code["state"])],
                         entry={"script": ENTRY, "interp": "bash", "unit_id": "main"},
                         datasets=[{**dataset, "unit_id": "main", "target": DATA_TARGET}],
                         parameters=effective_parameters)
    requirements = data["env_config"]["requirements_txt"].splitlines()
    plan.steps = [
        PlanStep(step_id="install_main", kind="install", unit_id="main", cwd="/app/main",
                 timeout_s=1200, install_budget_s=1200,
                 cmd="python -m pip install --no-cache-dir --target /app/.autorepro_site " + shlex.join(requirements)),
        PlanStep(step_id="environment", kind="prepare", unit_id="main", cwd="/app/main",
                 timeout_s=30, depends_on=["install_main"],
                 cmd="python -c 'import platform, subprocess, sys, torch; "
                     "print(\"Python\", platform.python_version(), \"Torch\", torch.__version__, \"CUDA\", torch.version.cuda); "
                     "assert torch.version.cuda is None, \"CPU build required\"; "
                     "subprocess.run([sys.executable, \"-m\", \"pip\", \"freeze\", \"--path\", \"/app/.autorepro_site\"], check=True)'"),
        PlanStep(step_id="run_etth1", kind="run", unit_id="main", cwd="/app/main",
                 timeout_s=600, depends_on=["environment"], cmd=command,
                 expects={"metrics": ["mse", "mae"]},
                 env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                      "PYTHONHASHSEED": "2021"}),
        PlanStep(step_id="parse", kind="parse", depends_on=["run_etth1"], expects={"metrics": ["mse", "mae"]}),
    ]
    plan.notes = ["LLM 基于实际仓库文件提出命令；执行器绑定真实 ETTh1 并强制 CPU 冒烟预算。",
                  *data["env_config"].get("compatibility_notes", []),
                  "沿用官方时间切分；流程验证，不声明达到论文数值，不生成替代脚本。"]
    result = plan.to_dict()
    result["llm_proposal"] = proposal
    result["policy_overrides"] = {k: {"requested": requested_flags.get(k), "effective": v}
                                  for k, v in flags.items() if requested_flags.get(k) != v}
    return result
