"""ExecutionPlannerAgent 测试（多代码单元整体调用规划器）。

覆盖：
1. 无可用单元 -> source="none"（CodeExecutor 走生成脚本回退）；
2. LLM 给出有效计划（引用真实文件）-> source="llm"；
3. LLM 幻觉路径 -> 步骤被 validate_plan 丢弃；
4. LLM 失败/空响应 -> 启发式兜底（install + run + parse）；
5. 未克隆成功的单元不进规划（clone-failed 被排除）。

运行: python -m pytest tests/test_execution_planner.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.execution_planner import ExecutionPlannerAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402
from src.resource_manager import ResourceManager  # noqa: E402


class _ScriptedLLM:
    """按序返回预置响应的假 LLM。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]

    def get_call_count(self) -> int:
        return self.call_count


@pytest.fixture()
def paper_input(tmp_path):
    """含本地 file:// 克隆单元的完整输入（iTransformer 风格结构）。"""
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--seq_len', type=int, default=96)\n"
        "p.add_argument('--train_epochs', type=int, default=10)\n"
        "print('mse: 0.01')\n", encoding="utf-8")
    scripts = repo / "scripts" / "multivariate_forecasting" / "ETTh1"
    scripts.mkdir(parents=True)
    (scripts / "iTransformer.sh").write_text(
        "python -u run.py --seq_len 96 --train_epochs 10\n")
    (repo / "requirements.txt").write_text("torch>=2.0.0\nnumpy\n")
    (repo / "README.md").write_text("# repo\n数据集: "
                                    "https://drive.google.com/x\n")
    for cmd in (["git", "init", "-q", str(repo)],
                ["git", "-C", str(repo), "config", "user.email", "t@e.c"],
                ["git", "-C", str(repo), "config", "user.name", "t"],
                ["git", "-C", str(repo), "add", "-A"],
                ["git", "-C", str(repo), "commit", "-q", "-m", "init"]):
        subprocess.run(cmd, check=True)

    rm = ResourceManager(data_root=str(tmp_path / "data"))
    infos = rm.fetch_units("pid", [
        {"unit_id": "main", "role": "main", "url": f"file://{repo}"},
        {"unit_id": "lib_0", "role": "library",
         "url": "https://github.com/example/lib"},  # 占位 -> 未克隆
    ])
    resources = {
        "code_units": [
            {"unit_id": "main", "role": "main", "url": f"file://{repo}"},
            {"unit_id": "lib_0", "role": "library",
             "url": "https://github.com/example/lib"},
        ],
    }
    storage = {"fetched": {"units": infos}}
    return {
        "paper_info": {"method": "iTransformer", "dataset": "ETTh1",
                       "metrics": {"mse": 0.01}},
        "resources": resources,
        "storage": storage,
        "paper_id": "pid",
    }


def _planner(llm) -> ExecutionPlannerAgent:
    return ExecutionPlannerAgent(llm)


def test_no_available_units_source_none(tmp_path):
    planner = _planner(_ScriptedLLM(["{}"]))
    result = planner.run({
        "paper_info": {}, "resources": {"code_units": []},
        "storage": {"fetched": {}}, "paper_id": "p0",
    })
    plan = result["execution_plan"]
    assert plan["source"] == "none"
    assert plan["steps"] == []
    assert any("无可用代码单元" in n for n in plan["notes"])


def test_llm_valid_plan_accepted(paper_input):
    valid = {
        "steps": [
            {"step_id": "install_main", "kind": "install",
             "cmd": "pip install -q -r requirements.txt",
             "cwd": "/app/main", "unit_id": "main"},
            {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
             "cwd": "/app/main", "unit_id": "main",
             "depends_on": ["install_main"],
             "expects": {"metrics": ["mse"]}},
        ],
        "entry": {"script": "run.py", "unit_id": "main", "interp": "python"},
    }
    import json
    planner = _planner(_ScriptedLLM([json.dumps(valid)]))
    result = planner.run(paper_input)
    plan = result["execution_plan"]
    assert plan["source"] == "llm"
    assert [s["step_id"] for s in plan["steps"]] == ["install_main", "run_0"]


def test_llm_hallucinated_path_dropped(paper_input):
    """命令引用不存在的文件 -> 该步骤被 validate_plan 丢弃。"""
    import json
    hallucinated = {
        "steps": [
            {"step_id": "run_0", "kind": "run",
             "cmd": "bash scripts/does_not_exist.sh",
             "cwd": "/app/main", "unit_id": "main"},
            {"step_id": "run_1", "kind": "run", "cmd": "python run.py",
             "cwd": "/app/main", "unit_id": "main"},
        ],
        "entry": {},
    }
    planner = _planner(_ScriptedLLM([json.dumps(hallucinated)]))
    result = planner.run(paper_input)
    plan = result["execution_plan"]
    # run_0 被丢弃；run_1 保留
    assert [s["step_id"] for s in plan["steps"]] == ["run_1"]


def test_llm_garbage_falls_back_to_heuristic(paper_input):
    planner = _planner(_ScriptedLLM(["不是 JSON"]))
    result = planner.run(paper_input)
    plan = result["execution_plan"]
    assert plan["source"] == "heuristic"
    kinds = [s["kind"] for s in plan["steps"]]
    assert kinds[0] == "install"
    assert "run" in kinds
    run_step = next(s for s in plan["steps"] if s["kind"] == "run")
    # 数据集 ETTh1 命中官方脚本 -> 优先脚本而非根 run.py
    assert run_step["cmd"] == \
        "bash scripts/multivariate_forecasting/ETTh1/iTransformer.sh"
    assert run_step["cwd"] == "/app/main"
    # 入口脚本含 --seq_len 96/--train_epochs 10 -> 提取 smoke 缩参
    assert run_step["smoke_args"].get("--seq_len") == "32"
    assert run_step["smoke_args"].get("--train_epochs") == "1"
    # bash 入口不转发 "$@"：缩参必须走 smoke_cmd（单次 python 调用），
    # 追加到 `bash x.sh` 后面是无效的。
    assert run_step["smoke_cmd"].startswith("python -u run.py")
    assert "--seq_len 32" in run_step["smoke_cmd"]
    assert "$" not in run_step["smoke_cmd"]


def test_uncloned_unit_excluded(paper_input):
    """clone-failed 单元不参与规划：可用单元只有 main。"""
    import json
    planner = _planner(_ScriptedLLM([json.dumps({"steps": []})]))
    result = planner.run(paper_input)
    plan = result["execution_plan"]
    assert [u["unit_id"] for u in plan["units"]] == ["main"]


def test_llm_exception_falls_back(paper_input, monkeypatch):
    """LLM 抛异常 -> 启发式兜底，流水线不阻断。"""
    class _BoomLLM(_ScriptedLLM):
        def chat(self, *a, **kw):
            raise RuntimeError("llm down")

    planner = _planner(_BoomLLM(["{}"]))
    result = planner.run(paper_input)
    plan = result["execution_plan"]
    assert plan["source"] == "heuristic"
    assert any("LLM 规划失败" in n for n in plan["notes"])
