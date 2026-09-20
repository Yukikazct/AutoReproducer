"""P1 真实优化闭环测试：RealSimulator + 安全网 + Orchestrator 接入。

核心断言：
1. Orchestrator 配置 workspace_dir 后注入真实执行器（替代哈希模拟）；
2. 补丁白名单拦截（policy_rejected）且不触碰工作区；
3. 真实执行 Keep：奖励 > 3% 且工作区快照回滚（原代码不被破坏）+ 补丁落盘；
4. 执行失败 Reject：奖励为 0 且工作区回滚；
5. Mock LLM 全流水线端到端：复现 -> 验证 -> 真实优化 -> 报告（COMPLETED）。

运行: python -m pytest tests/test_optimizer_real.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient
from src.agents.code_executor import CodeExecutorAgent
from src.agents.optimizer import OptimizerAgent
from src.optimizer.real_simulator import RealSimulator
from src.safety.patch_policy import PatchPolicy
from src.orchestrator import Orchestrator

_BASELINE_CODE = (
    "def train(epochs=10):\n"
    "    correct = 0\n"
    "    total = 200\n"
    "    for e in range(epochs):\n"
    "        correct = int(total * 0.852)\n"
    "    print('Training complete. Test accuracy: 85.2%')\n"
    "    print('Final loss: 0.3120')\n"
    "    return correct / total\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    train()\n"
)


class _BadPatchLLM:
    """返回不可编译补丁的假 LLM，用于失败回滚路径。"""

    def __init__(self, code: str = "def broken(: 语法错误\n"):
        self._code = code

    def chat(self, prompt: str, task: str = "", **kwargs) -> str:
        if task == "optimizer_patch":
            return self._code
        return ""

    def get_call_count(self) -> int:
        return 0

    def reset_call_count(self) -> None:
        pass


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.py").write_text(_BASELINE_CODE, encoding="utf-8")
    return ws


@pytest.fixture
def mock_llm() -> LLMClient:
    return LLMClient(mock_mode=True)


# ---------------- 1. Orchestrator 注入 ----------------

def test_orchestrator_injects_real_simulator_only_with_workspace(tmp_path):
    orch = Orchestrator(mock_mode=True, workspace_dir=str(tmp_path / "ws"))
    assert isinstance(orch.agents["optimizer"].simulator, RealSimulator)

    plain = Orchestrator(mock_mode=True)
    assert plain.agents["optimizer"].simulator is OptimizerAgent._simulate_trial


# ---------------- 2. 补丁白名单拦截 ----------------

def test_real_simulator_rejects_policy_violation(workspace, mock_llm):
    sim = RealSimulator(llm=mock_llm, executor=CodeExecutorAgent(mock_llm),
                        workspace_dir=str(workspace),
                        policy=PatchPolicy(editable=["other.py"]))
    reward, detail = sim("超参数调优(学习率)", 0.852)
    assert reward == 0.0
    assert detail["status"] == "policy_rejected"
    # 工作区未被触碰
    assert (workspace / "run.py").read_text(encoding="utf-8") == _BASELINE_CODE
    # 未生成补丁目录
    assert not (workspace.parent / "optimized_patches").exists()


# ---------------- 3. 真实执行 Keep + 回滚 + 落盘 ----------------

def test_real_simulator_keeps_improvement_and_rolls_back(workspace, mock_llm):
    sim = RealSimulator(llm=mock_llm, executor=CodeExecutorAgent(mock_llm),
                        workspace_dir=str(workspace))
    sim.bind_paper({"metrics": {"accuracy": 0.852}})
    reward, detail = sim("超参数调优(学习率)", 0.852)

    assert reward > 0.03, f"期望提升>=3%,实际 {reward:.2%}"  # 90.0% vs 85.2%
    assert detail["status"] == "kept"
    assert detail["metric"]["accuracy"] == pytest.approx(0.9, abs=1e-6)
    assert detail["restored"]  # 快照回滚真实发生
    # 工作区原貌未被破坏
    assert (workspace / "run.py").read_text(encoding="utf-8") == _BASELINE_CODE
    # Keep 补丁落盘
    kept = Path(detail["kept_patch"])
    assert kept.exists() and "90.0%" in kept.read_text(encoding="utf-8")


def test_real_simulator_rolls_back_after_execution_failure(workspace):
    llm = _BadPatchLLM("raise RuntimeError('boom')\n")
    sim = RealSimulator(llm=llm, executor=CodeExecutorAgent(llm),
                        workspace_dir=str(workspace))
    sim.bind_paper({"metrics": {"accuracy": 0.852}})
    reward, detail = sim("超参数调优(学习率)", 0.852)

    assert reward == 0.0
    assert detail["status"] == "rejected"
    assert "run.py" in detail.get("restored", [])
    assert (workspace / "run.py").read_text(encoding="utf-8") == _BASELINE_CODE


def test_patch_syntax_gate_rejects_before_touching_workspace(workspace):
    """不可编译的补丁必须被语法门拦下：不写盘、不执行。"""
    llm = _BadPatchLLM("def broken(: 语法错误\n")
    sim = RealSimulator(llm=llm, executor=CodeExecutorAgent(llm),
                        workspace_dir=str(workspace))
    sim.bind_paper({"metrics": {"accuracy": 0.852}})
    reward, detail = sim("超参数调优(学习率)", 0.852)

    assert reward == 0.0
    assert detail["status"] == "rejected"
    assert "不可编译" in detail.get("reason", "")
    # 关键：工作区一个字都没动过（连快照回滚都用不上）
    assert (workspace / "run.py").read_text(encoding="utf-8") == _BASELINE_CODE
    assert "restored" not in detail


# ---------------- 4. execute_in_workspace 指定目录执行 ----------------

def test_execute_in_workspace_runs_in_given_dir(tmp_path, mock_llm):
    workdir = tmp_path / "ws"
    workdir.mkdir()
    executor = CodeExecutorAgent(mock_llm)
    code = "import os\nprint('PWD=' + os.getcwd())\nprint('OK')\n"
    result = executor.execute_in_workspace(code, str(workdir), stage="smoke")
    assert result["success"] is True
    assert "OK" in result["stdout"]
    assert str(workdir).replace("\\", "/") in result["stdout"].replace("\\", "/")
    # 指定目录执行后保留（不清理），run.py 已写入
    assert (workdir / "run.py").exists()


# ---------------- 5. Mock LLM 全流水线端到端 ----------------

def test_full_pipeline_with_real_optimization(tmp_path, mock_llm):
    ws = tmp_path / "ws"
    orch = Orchestrator(llm_client=mock_llm, mock_mode=True,
                        workspace_dir=str(ws), max_trials=2)
    result = orch.run({"paper_title": "Dummy Paper"})
    data = result["data"]

    assert result["state"] == "COMPLETED"
    assert data["validation"]["is_reproduced"] is True
    opt = data.get("optimization", {})
    assert opt.get("optimized") is True
    # 真实优化:单案例提升（90.0% vs 85.2%）>= 3%
    assert opt.get("improvement", 0.0) > 0.03
    assert opt.get("best_arm")
    # 工作区已被物化且优化后未被补丁破坏（快照回滚生效，保持物化原样）
    materialized = (ws / "run.py").read_text(encoding="utf-8")
    assert "85.2%" in materialized
    # 审计账本记录真实执行
    records = opt.get("optimization_report", [])
    assert records and any(r.get("detail", {}).get("type") == "real_exec"
                           for r in records)
    assert (tmp_path / "optimized_patches").exists()


# ---------------- 6. 无工作区时行为不变(兼容) ----------------

def test_orchestrator_without_workspace_keeps_mock_simulation(mock_llm):
    orch = Orchestrator(llm_client=mock_llm, mock_mode=True, max_trials=1)
    result = orch.run({"paper_title": "Dummy Paper"})
    assert result["state"] == "COMPLETED"
    records = result["data"].get("optimization", {}).get(
        "optimization_report", [])
    assert records and all(
        r.get("detail", {}).get("type") == "ucb_mock" for r in records)