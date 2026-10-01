"""Small, deterministic experiments using official code and real data.

Profiles provide inputs and commands, never results. They use the normal
Orchestrator and Docker executor, and do not require an LLM API key.
"""
import shlex
import re
from pathlib import Path

from src.code_units import CodeUnit
from src.execution_plan import ExecutionPlan, PlanStep, derive_smoke_cmd

PROFILE_ID = "itransformer_etth1_cpu_smoke"
PROFILE_LABEL = "iTransformer / ETTh1（CPU 真实冒烟）"
PAPER_TITLE = "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
REPO_URL = "https://github.com/thuml/iTransformer"
REPO_REVISION = "c2426e68ca13f74aaec08045c5c724d8ad328124"
ENTRY_SCRIPT = "scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh"
DATA_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
DATA_URL = (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
            f"{DATA_REVISION}/ETT-small/ETTh1.csv")
DATA_RELATIVE_PATH = "dataset/ETT-small/ETTh1.csv"
PARAMETERS = {
    "seq_len": 96, "pred_len": 96, "enc_in": 7, "dec_in": 7, "c_out": 7,
    "train_epochs": 1, "batch_size": 32, "num_workers": 0, "e_layers": 1,
    "d_model": 64, "d_ff": 64, "itr": 1,
}

# These are execution adapters for discovered official repositories, not stored
# answers. DLinear/NLinear are two models from the same paper (two papers total).
PROFILES = {
    PROFILE_ID: {"label": PROFILE_LABEL, "title": PAPER_TITLE, "model": "iTransformer",
                 "repo": REPO_URL, "revision": REPO_REVISION,
                 "entry": ENTRY_SCRIPT, "runner": "run.py", "seed": 2023},
    "dlinear_etth1_cpu_smoke": {
        "label": "DLinear / ETTh1（轻量 CPU 演示）",
        "title": "Are Transformers Effective for Time Series Forecasting?",
        "model": "DLinear", "repo": "https://github.com/cure-lab/LTSF-Linear",
        "revision": "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
        "entry": "scripts/EXP-LongForecasting/Linear/etth1.sh",
        "runner": "run_longExp.py", "seed": 2021,
    },
}
PROFILES["nlinear_etth1_cpu_smoke"] = {
    **PROFILES["dlinear_etth1_cpu_smoke"], "model": "NLinear",
    "label": "NLinear / ETTh1（同论文另一模型）",
}
CPU_REQUIREMENTS = ("numpy==1.23.5", "pandas==1.5.3", "scikit-learn==1.2.2",
                    "matplotlib==3.7.0", "torch==2.0.0")


def check_profile(value: str) -> str:
    if value and value not in PROFILES:
        raise ValueError(f"未知实验预设: {value}")
    return value


def resolve_profile(title: str = "", explicit: str = "") -> str:
    """A paper title is the primary input; presets just select an adapter."""
    if explicit:
        return check_profile(explicit)
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    for key, value in PROFILES.items():
        if re.search(r"\b" + value["model"].lower() + r"\b", normalized):
            return key
    if normalized == "are transformers effective for time series forecasting":
        return "dlinear_etth1_cpu_smoke"
    return ""


def stage_result(state: str, data: dict) -> dict | None:
    """Deterministic preparation stages; execution/validation stay in agents."""
    profile_id = data.get("experiment_profile", "")
    if profile_id not in PROFILES or data.get("mock_mode"):
        return None
    profile = PROFILES[profile_id]
    if state == "READ_PAPER":
        return {"paper_info": {
            "title": profile["title"], "method": profile["model"], "dataset": "ETTh1",
            "metrics": {}, "dependencies": [], "info_sufficient": True,
            "source": "官方仓库固定版本的 CPU 冒烟预设，未提取论文数值声明",
        }, "raw_text": "", "llm_calls": 0}
    if state == "FIND_RESOURCES":
        from src.agents.repo_discovery import discover_repositories
        discovery = discover_repositories(
            profile["title"], preferred_url=data.get("preferred_repo_url", ""),
            github_first=True, timeout=5)
        matching = next((c for c in discovery.get("candidates", [])
                         if profile["repo"] in c.get("repo_urls", [])), None)
        if not matching and not data.get("preferred_repo_url"):
            discovery["discovery_chain"].append("demo_registry_fallback")
            discovery["fallback_used"] = True
            discovery["candidates"].append({"repo_urls": [profile["repo"]],
                                           "source": "demo_registry_fallback"})
        selected = (data.get("preferred_repo_url") or profile["repo"])
        discovery.update(selected_repo=selected, pinned_revision=profile["revision"])
        unit = CodeUnit(unit_id="main", role="main", url=selected,
                        revision=profile["revision"]).to_dict()
        return {"resources": {
            "code_repo_url": selected, "dataset_url": "ETTh1", "weights_url": "",
            "confidence": 1.0, "code_units": [unit],
            "repo_discovery": discovery,
            "repro_mode": {"effective_mode": "smoke"},
        }, "llm_calls": 0}
    if state == "BUILD_ENV":
        repo = Path(data.get("repo_path") or "")
        requirements = repo / "requirements.txt"
        original = requirements.read_text() if data.get("repo_path") and requirements.is_file() else ""
        return {"env_config": {
            "python_version": "3.11", "image_tag": "python:3.11-slim",
            "requirements_txt": (original if profile_id == PROFILE_ID else
                                 "\n".join(CPU_REQUIREMENTS) if original else ""),
            "upstream_requirements_txt": original,
            "build_required": False,
        }, "llm_calls": 0}
    if state == "PLAN_EXECUTION":
        return {"execution_plan": build_plan(data), "llm_calls": 0}
    return None


def build_plan(data: dict) -> dict:
    profile_id = data["experiment_profile"]
    profile = PROFILES[profile_id]
    fetched = (data.get("storage") or {}).get("fetched") or {}
    code = fetched.get("code") or {}
    dataset = fetched.get("dataset") or {}
    plan = ExecutionPlan(paper_id=data.get("paper_id", ""), source="profile",
                         experiment_profile=profile_id,
                         parameters={**PARAMETERS, "model": profile["model"],
                                     "seed": profile["seed"], "device": "cpu"})
    if code.get("state") not in ("cloned", "cached") or not code.get("path"):
        plan.notes = ["官方仓库不可用: " + code.get("detail", "尚未下载")]
        return plan.to_dict()
    if code.get("commit") != profile["revision"]:
        plan.notes = ["官方仓库版本不匹配，请移走该预设的仓库缓存后重试"]
        return plan.to_dict()
    root = Path(code["path"])
    if not all((root / p).is_file() for p in (profile["entry"], profile["runner"], "requirements.txt")):
        plan.notes = ["官方入口或 requirements.txt 缺失"]
        return plan.to_dict()
    if dataset.get("data_kind") != "real" or dataset.get("state") not in ("downloaded", "cached"):
        plan.notes = ["真实 ETTh1 不可用: " + dataset.get("detail", "尚未下载")]
        return plan.to_dict()

    # Keep the official first command, replace flags instead of executing all
    # four prediction horizons in the shell script. No upstream source edits.
    command = derive_smoke_cmd((root / profile["entry"]).read_text(), {})
    if not command:
        plan.notes = ["无法解析官方 ETTh1 入口"]
        return plan.to_dict()
    overrides = {"--" + k: str(v) for k, v in PARAMETERS.items()}
    overrides["--model"] = profile["model"]
    overrides["--root_path"] = "/app/main/dataset/ETT-small/"
    tokens = shlex.split(command)
    kept, i = [], 0
    while i < len(tokens):
        if tokens[i].startswith(">"):
            break  # Official LTSF script redirects logs; capture them in the executor.
        if tokens[i] in overrides:
            i += 2
        else:
            kept.append(tokens[i])
            i += 1
    for key, value in overrides.items():
        kept.extend([key, value])

    plan.units = [CodeUnit(unit_id="main", role="main", url=code.get("url") or profile["repo"],
                           local_path=str(root), revision=code["commit"],
                           fetch_state=code["state"])]
    plan.entry = {"script": profile["entry"], "interp": "bash", "unit_id": "main"}
    plan.datasets = [{**dataset, "unit_id": "main", "target": DATA_RELATIVE_PATH}]
    plan.steps = [
        PlanStep(step_id="install_main", kind="install", unit_id="main",
                 cwd="/app/main", timeout_s=1200, install_budget_s=1200,
                 cmd=("python -m pip install --no-cache-dir --target /app/.autorepro_site "
                      + ("-r requirements.txt" if profile_id == PROFILE_ID else
                         shlex.join(CPU_REQUIREMENTS)))),
        PlanStep(step_id="environment", kind="prepare", unit_id="main",
                 cwd="/app/main", timeout_s=30, depends_on=["install_main"],
                 cmd="python --version && python -m pip freeze --path /app/.autorepro_site"),
        PlanStep(step_id="run_etth1", kind="run", unit_id="main",
                 cwd="/app/main", timeout_s=600,
                 depends_on=["environment"], cmd=shlex.join(kept),
                 args={**PARAMETERS}, expects={"metrics": ["mse", "mae"]},
                 env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                      "MKL_NUM_THREADS": "2", "PYTHONHASHSEED": str(profile["seed"])}),
        PlanStep(step_id="parse", kind="parse", depends_on=["run_etth1"],
                 expects={"metrics": ["mse", "mae"]}),
    ]
    plan.notes = ["真实 ETTh1 完整时间序列，沿用官方时间切分；单次 CPU 缩参训练。",
                  "流程验证，不代表论文指标复现；不执行优化或生成脚本回退。"]
    if profile_id != PROFILE_ID:
        plan.notes.append("CPU/Python 3.11 兼容环境使用 torch 2.0.0，代替原仓库的 torch 1.9.0；算法源码不修改。")
    return plan.to_dict()
