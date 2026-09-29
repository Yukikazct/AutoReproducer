"""执行计划数据模型测试（ExecutionPlan + 确定性预分析助手）。

覆盖：
1. validate_plan：合法计划通过；幻觉文件路径 / 后置依赖 / 重复 step_id /
   未知 kind / cwd 越界 / 未知 unit_id 全部拒绝；
2. repo_snapshot：有界扫描、scripts 识别、README 数据集链接提取；
3. detect_entry：数据集命中官方脚本时越过根 run.py、命中方法/数据集、
   浅层兜底；
4. suggest_smoke_args：只缩小已有键、epochs->1、**不动数据形状参数**；
5. derive_smoke_cmd：从官方 .sh 派生可生效的缩参命令（变量替换、
   未定义变量放弃、无 python 调用放弃、空白规整）；
6. PlanStep/ExecutionPlan dict 往返。

运行: python -m pytest tests/test_execution_plan.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution_plan import (  # noqa: E402
    ExecutionPlan,
    PlanStep,
    default_step_budgets,
    derive_smoke_cmd,
    detect_entry,
    extract_cli_args,
    repo_snapshot,
    suggest_smoke_args,
    validate_plan,
)


# ---------------- 夹具：合成仓库树 ----------------

@pytest.fixture()
def itransformer_tree(tmp_path):
    """模拟 thuml/iTransformer 结构：根 run.py + scripts/<task>/<ds>/<model>.sh
    + README（含数据集网盘链接）+ requirements。"""
    root = tmp_path / "itransformer"
    root.mkdir()
    (root / "run.py").write_text(
        "import argparse\n"
        "parser.add_argument('--seq_len', type=int, default=96)\n"
        "parser.add_argument('--pred_len', type=int, default=96)\n"
        "parser.add_argument('--train_epochs', type=int, default=10)\n",
        encoding="utf-8")
    (root / "models").mkdir()
    (root / "models" / "iTransformer.py").write_text("pass\n")
    scripts = root / "scripts" / "multivariate_forecasting" / "ETTh1"
    scripts.mkdir(parents=True)
    (scripts / "iTransformer.sh").write_text(
        "python -u run.py --data ETTh1 --model iTransformer\n")
    (root / "requirements.txt").write_text("torch>=2.0.0\nnumpy\n")
    (root / "README.md").write_text(
        "# iTransformer\n下载数据集:\n"
        "https://drive.google.com/file/d/abc123/view\n"
        "或清华云盘: https://cloud.tsinghua.edu.cn/d/xyz789\n",
        encoding="utf-8")
    return root


# ---------------- 1. validate_plan ----------------

def _valid_plan(unit_id="main"):
    return {
        "paper_id": "p1",
        "units": [{"unit_id": "main", "role": "main", "url": "x"}],
        "steps": [
            {"step_id": "install", "kind": "install",
             "cmd": "pip install -q -r requirements.txt",
             "cwd": "/app/main", "unit_id": "main"},
            {"step_id": "run_0", "kind": "run",
             "cmd": "bash scripts/multivariate_forecasting/ETTh1/iTransformer.sh",
             "cwd": "/app/main", "unit_id": "main",
             "depends_on": ["install"], "expects": {"metrics": ["mse"]}},
        ],
    }


def test_validate_plan_accepts_valid():
    snapshots = {"main": {
        "files": ["requirements.txt",
                  "scripts/multivariate_forecasting/ETTh1/iTransformer.sh"],
        "scripts": ["scripts/multivariate_forecasting/ETTh1/iTransformer.sh"],
    }}
    assert validate_plan(_valid_plan(), snapshots) == []


def test_validate_plan_rejects_hallucinated_path():
    """LLM 幻觉路径必须失败关闭：命令引用快照中不存在的文件。"""
    plan = _valid_plan()
    plan["steps"][1]["cmd"] = "bash scripts/nonexistent/x.sh"
    snapshots = {"main": {
        "files": ["requirements.txt",
                  "scripts/multivariate_forecasting/ETTh1/iTransformer.sh"],
        "scripts": [],
    }}
    errors = validate_plan(plan, snapshots)
    assert errors and any("不存在的文件" in e for e in errors)


def test_validate_plan_rejects_forward_dependency():
    plan = _valid_plan()
    plan["steps"][0]["depends_on"] = ["run_0"]  # 指向后置步骤
    assert validate_plan(plan, None)


def test_validate_plan_rejects_duplicate_and_bad_kind():
    plan = _valid_plan()
    plan["steps"].append({"step_id": "install", "kind": "run",
                          "cmd": "echo hi"})
    errors = validate_plan(plan, None)
    assert any("重复" in e for e in errors)
    plan = _valid_plan()
    plan["steps"][0]["kind"] = "explode"
    assert any("未知 kind" in e for e in validate_plan(plan, None))


def test_validate_plan_rejects_unknown_unit_and_bad_cwd():
    plan = _valid_plan()
    plan["steps"][1]["unit_id"] = "ghost"
    assert any("不在计划单元中" in e for e in validate_plan(plan, None))
    plan = _valid_plan()
    plan["steps"][1]["cwd"] = "/etc/passwd"
    assert any("cwd" in e for e in validate_plan(plan, None))


def test_validate_plan_requires_steps_list():
    assert validate_plan({}, None)
    assert validate_plan("not a plan", None)


# ---------------- 2. repo_snapshot ----------------

def test_repo_snapshot_finds_scripts_and_links(itransformer_tree):
    snap = repo_snapshot(str(itransformer_tree))
    assert "run.py" in snap["files"]
    assert "scripts/multivariate_forecasting/ETTh1/iTransformer.sh" \
        in snap["scripts"]
    assert "requirements.txt" in snap["requirements"]
    assert snap["readmes"] == ["README.md"]
    assert any("drive.google.com" in link for link in snap["dataset_links"])
    assert any("cloud.tsinghua.edu.cn" in link
               for link in snap["dataset_links"])


def test_repo_snapshot_missing_dir_is_truncated(tmp_path):
    snap = repo_snapshot(str(tmp_path / "nope"))
    assert snap["files"] == []
    assert snap["truncated"] is True


# ---------------- 3. detect_entry ----------------

def test_detect_entry_prefers_dataset_script_over_root_run_py(
        itransformer_tree):
    """数据集命中官方脚本时**越过根 run.py**（真实仓库推翻的优先级）。

    iTransformer 根 run.py 的 argparse 默认值是
    `--root_path ./data/electricity/ --data_path electricity.csv`，仓库里
    并没有 data/ 目录，选中它必然 FileNotFoundError；真正的官方调用在
    scripts/multivariate_forecasting/ETTh1/iTransformer.sh 里，只有它带着
    正确的 `--root_path ./dataset/ETT-small/`。
    """
    snap = repo_snapshot(str(itransformer_tree))
    entry = detect_entry(snap, {"method": "iTransformer", "dataset": "ETTh1"})
    assert entry["script"] == \
        "scripts/multivariate_forecasting/ETTh1/iTransformer.sh"
    assert entry["interp"] == "bash"


def test_detect_entry_keeps_root_run_py_without_dataset_match(
        itransformer_tree):
    """数据集对不上任何脚本时仍按根 run.py，简单仓库行为不变。"""
    snap = repo_snapshot(str(itransformer_tree))
    entry = detect_entry(snap, {"method": "iTransformer",
                                "dataset": "Electricity"})
    assert entry["script"] == "run.py"
    assert entry["interp"] == "python"


def test_detect_entry_matches_method_in_scripts(tmp_path):
    root = tmp_path / "nopy"
    (root / "scripts" / "multivariate_forecasting" / "ETTh1").mkdir(
        parents=True)
    (root / "scripts" / "multivariate_forecasting" / "ETTh1"
     / "iTransformer.sh").write_text("echo hi\n")
    snap = repo_snapshot(str(root))
    entry = detect_entry(snap, {"method": "iTransformer", "dataset": "ETTh1"})
    assert entry["script"].endswith("iTransformer.sh")
    assert entry["interp"] == "bash"


def test_detect_entry_fallback_shallow_script(tmp_path):
    root = tmp_path / "onlysh"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_all.sh").write_text("echo hi\n")
    snap = repo_snapshot(str(root))
    entry = detect_entry(snap, {})
    assert entry["script"] == "scripts/run_all.sh"


def test_detect_entry_empty_when_nothing(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    snap = repo_snapshot(str(root))
    assert detect_entry(snap, {}) == {}


# ---------------- 4. suggest_smoke_args ----------------

def test_suggest_smoke_args_shrinks_only_existing_keys():
    args = {"seq_len": 96, "pred_len": 720, "train_epochs": 10,
            "batch_size": 64, "num_workers": 4}
    shrunk = suggest_smoke_args(args)
    assert shrunk["seq_len"] == 32
    assert shrunk["pred_len"] == 32
    assert shrunk["train_epochs"] == 1
    assert shrunk["batch_size"] == 16
    assert shrunk["num_workers"] == 0


def test_suggest_smoke_args_ignores_small_and_unknown():
    shrunk = suggest_smoke_args({"seq_len": 16, "learning_rate": 0.001})
    assert shrunk == {}


def test_default_step_budgets_kinds():
    for kind in ("install", "download", "prepare", "run", "parse"):
        assert default_step_budgets(kind) > 0


# ---------------- 5. dict 往返 ----------------

def test_plan_roundtrip():
    plan = ExecutionPlan(
        paper_id="p9", source="llm",
        steps=[PlanStep(step_id="run_0", kind="run",
                        cmd="python run.py", cwd="/app/main",
                        unit_id="main", args={"--seq_len": "96"},
                        smoke_args={"--seq_len": "32"},
                        smoke_cmd="python -u run.py --seq_len 32",
                        depends_on=["install"],
                        expects={"metrics": ["mse", "mae"]},
                        install_pkgs=["gdown"])],
        entry={"script": "run.py", "unit_id": "main", "interp": "python"},
        notes=["note1"],
    )
    restored = ExecutionPlan.from_dict(plan.to_dict())
    assert restored.paper_id == "p9"
    assert restored.source == "llm"
    assert restored.has_steps()
    step = restored.steps[0]
    assert step.step_id == "run_0"
    assert step.args == {"--seq_len": "96"}
    assert step.smoke_args == {"--seq_len": "32"}
    assert step.smoke_cmd == "python -u run.py --seq_len 32"
    assert step.depends_on == ["install"]
    assert step.expects["metrics"] == ["mse", "mae"]
    assert step.install_pkgs == ["gdown"]
    assert restored.entry["interp"] == "python"
    assert restored.notes == ["note1"]


# ---------------- 5. 缩参命令派生（官方 .sh 入口） ----------------

# 真实 thuml/iTransformer 的 scripts/multivariate_forecasting/ETT/
# iTransformer_ETTh1.sh 形态：变量赋值 + 4 段 python 调用（pred_len
# 96/192/336/720），行尾反斜杠续行原样保留。
_ITRANSFORMER_SH = """export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u run.py \\
  --is_training 1 \\
  --root_path ./dataset/ETT-small/ \\
  --data_path ETTh1.csv \\
  --model_id ETTh1_96_96 \\
  --model $model_name \\
  --data ETTh1 \\
  --features M \\
  --seq_len 96 \\
  --pred_len 96 \\
  --enc_in 7 \\
  --dec_in 7 \\
  --c_out 7 \\
  --des 'Exp' \\
  --train_epochs 10 \\
  --itr 1

python -u run.py \\
  --pred_len 720 \\
  --itr 1
"""


def test_derive_smoke_cmd_from_official_sh():
    args = extract_cli_args(_ITRANSFORMER_SH)
    assert args["pred_len"] == 96           # 首次出现优先（不是 720）
    smoke = {f"--{k}": str(v) for k, v in suggest_smoke_args(args).items()}
    cmd = derive_smoke_cmd(_ITRANSFORMER_SH, smoke)
    assert cmd.startswith("python -u run.py --is_training 1")
    assert "--model iTransformer" in cmd     # $model_name 已按赋值替换
    assert "$" not in cmd
    assert "--root_path ./dataset/ETT-small/" in cmd
    assert "--itr 1" in cmd
    # argparse 后写覆盖：追加的缩参必须真的出现在命令里
    assert "--seq_len 32" in cmd and "--train_epochs 1" in cmd
    # 数据形状参数保持原值（缩了会导致模型与数据通道数不匹配）
    assert "--enc_in 7" in cmd and "--c_out 7" in cmd
    assert "  " not in cmd                   # 续行拼接的空白已规整


def test_derive_smoke_cmd_gives_up_on_undefined_var():
    """未定义变量 -> 放弃（宁可退回原行为，不拼出跑不通的命令）。"""
    assert derive_smoke_cmd("python -u run.py --model $foo\n", {}) == ""


def test_derive_smoke_cmd_gives_up_without_python_call():
    assert derive_smoke_cmd("export A=1\necho hi\n", {"--x": "2"}) == ""


def test_suggest_smoke_args_keeps_data_shape_knobs():
    """enc_in/dec_in/c_out 是数据形状参数，不能被缩参改动。"""
    shrunk = suggest_smoke_args({"enc_in": 7, "dec_in": 7, "c_out": 7,
                                 "seq_len": 96, "d_model": 512, "epochs": 10})
    assert "enc_in" not in shrunk and "dec_in" not in shrunk
    assert "c_out" not in shrunk
    assert shrunk["seq_len"] == 32 and shrunk["d_model"] == 32
    assert shrunk["epochs"] == 1
