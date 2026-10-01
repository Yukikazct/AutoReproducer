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


# ---------------- 奖励指标的键匹配与方向 ----------------

class TestRewardMetricResolution:
    """论文声明 `MSE`、运行输出 `mse` 是同一个指标（与判定层同源）。

    键不归一的话 `self._metric_key in metrics` 恒为假，奖励方向会悄悄退化
    成"取与基线绝对值最接近的指标"这条启发式——方向可能判反。
    """

    @staticmethod
    def _sim(tmp_path: Path) -> RealSimulator:
        llm = LLMClient(mock_mode=True)
        sim = RealSimulator(llm=llm, executor=CodeExecutorAgent(llm),
                            workspace_dir=str(tmp_path / "ws"))
        sim.bind_paper({"metrics": {"MSE": 0.0892}})
        return sim

    def test_declared_metric_is_matched_despite_case(self, tmp_path):
        sim = self._sim(tmp_path)
        reward, metric, basis = sim._compute_reward(
            0.0892, "MSE: 0.0869\n", success=True)
        assert basis["metric_key"] == "mse"
        assert basis["metric_source"] == "论文声明指标"
        assert basis["direction"] == "越小越好"
        # MSE 变小 -> 正改进
        assert reward > 0

    def test_direction_is_negative_when_mse_grows(self, tmp_path):
        """MSE 变大必须得负奖励（原先键对不上会退回启发式，方向可能反）。"""
        sim = self._sim(tmp_path)
        reward, _, basis = sim._compute_reward(
            0.0892, "MSE: 0.096764\n", success=True)
        assert reward < 0
        assert basis["direction"] == "越小越好"

    def test_heuristic_fallback_is_disclosed(self, tmp_path):
        """声明指标在输出里没有时，如实说明是启发式选的键。"""
        sim = self._sim(tmp_path)
        # 输出里只有 loss，没有论文声明的 MSE -> 只能启发式选键
        reward, metric, basis = sim._compute_reward(
            0.0892, "loss: 0.0840\n", success=True)
        assert metric and "loss" in metric
        assert basis["metric_source"] == "启发式(与基线绝对值最接近)"

    def test_higher_is_better_metric_keeps_positive_direction(self, tmp_path):
        llm = LLMClient(mock_mode=True)
        sim = RealSimulator(llm=llm, executor=CodeExecutorAgent(llm),
                            workspace_dir=str(tmp_path / "ws"))
        sim.bind_paper({"metrics": {"Accuracy": 0.85}})
        reward, _, basis = sim._compute_reward(
            0.85, "accuracy: 0.90\n", success=True)
        assert reward > 0
        assert basis["direction"] == "越大越好"

    def test_basis_is_recorded_in_trial_detail(self, workspace, mock_llm):
        """trial 结果里要带上依据，供报告展示"按哪个指标、哪个方向"。"""
        sim = RealSimulator(llm=mock_llm, executor=CodeExecutorAgent(mock_llm),
                            workspace_dir=str(workspace))
        sim.bind_paper({"metrics": {"MSE": 0.852}})
        _, detail = sim("超参数调优(学习率)", 0.852)
        basis = detail.get("reward_basis") or {}
        assert basis.get("metric_key")
        assert basis.get("metric_source")
        assert basis.get("direction")


# ---------------- 模拟优化必须在报告里如实标注 ----------------

class TestReportDisclosesSimulatedOptimization:
    """`_simulate_trial` 拿方向名的哈希当"改进潜力"，与代码和指标都无关。

    不标注的话，"改进幅度 1.80% / 最优结果 0.0908056" 会被当成实测值读——
    真实模式下同样如此（app 从不传 workspace_dir，所以永远走模拟）。
    """

    @staticmethod
    def _report(optimization: dict) -> str:
        from src.agents.report_generator import ReportGeneratorAgent
        return ReportGeneratorAgent().run(
            {"optimization": optimization, "execution": {},
             "paper_info": {}})["report"]

    @staticmethod
    def _optimization(detail_type: str) -> dict:
        return {
            "optimized": True, "baseline": 0.0892, "best_arm": "改用 QR 分解",
            "improvement": 0.018, "best_result": 0.0908056,
            "budget_used": 2, "budget": 2,
            "optimization_report": [
                {"arm": "改用 QR 分解", "improvement": 0.018, "kept": True,
                 "detail": {"type": detail_type}},
            ],
        }

    def test_mock_run_is_flagged_as_simulated(self):
        report = self._report(self._optimization("ucb_mock"))
        assert "⚠️ 已优化（模拟）" in report
        assert "不是**跑出来的实测值" in report
        assert "⚠️ 哈希模拟" in report

    def test_mock_numbers_are_marked_inline(self):
        """幅度与结果两处都要带"（模拟）"，避免只看到数字那一行。"""
        report = self._report(self._optimization("ucb_mock"))
        assert "**改进幅度**: 1.80%（模拟）" in report
        assert "**最优结果**: 0.0908056（模拟）" in report

    def test_real_run_is_not_flagged(self):
        report = self._report(self._optimization("real_exec"))
        assert "模拟" not in report
        assert "✅ 已优化" in report
        assert "真实执行" in report

    def test_real_run_shows_metric_and_direction(self):
        """真实执行要写出按哪个指标、哪个方向判的。"""
        opt = self._optimization("real_exec")
        opt["optimization_report"][0]["detail"]["reward_basis"] = {
            "metric_key": "mse", "metric_source": "论文声明指标",
            "direction": "越小越好"}
        report = self._report(opt)
        assert "真实执行 · mse↓" in report

    def test_heuristic_pick_is_marked_in_report(self):
        opt = self._optimization("real_exec")
        opt["optimization_report"][0]["detail"]["reward_basis"] = {
            "metric_key": "loss", "metric_source": "启发式(与基线绝对值最接近)",
            "direction": "越小越好"}
        report = self._report(opt)
        assert "启发式选键" in report