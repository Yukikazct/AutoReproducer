"""OptimizerAgent - 智能优化 Agent,基于 UCB 预算调度"""
import hashlib
import json
from src.base_agent import BaseAgent
from src.optimizer.ucb_scheduler import UCBScheduler


class OptimizerAgent(BaseAgent):
    """复现成功后,提出优化方向并用 UCB 在预算内智能分配尝试资源(Keep/Reject)。"""

    system_prompt = "基于复现结果提出可证伪的优化假设,并在预算内用 UCB 智能调度尝试"

    DEFAULT_ARMS = [
        "超参数调优(学习率)",
        "超参数调优(batch size)",
        "模块替换(优化器)",
        "模块替换(激活函数)",
        "架构调整(层数/宽度)",
        "架构调整(正则化)",
    ]

    def __init__(self, llm_client, logger=None, max_trials=10, c=1.0):
        super().__init__("Optimizer", logger)
        self.llm = llm_client
        self.max_trials = max_trials
        self.c = c

    def run(self, input_data: dict) -> dict:
        """input_data 需含 paper_info、validation(含 is_reproduced)。"""
        paper_info = input_data.get("paper_info", {})
        validation = input_data.get("validation", {})

        if not validation.get("is_reproduced", False):
            self.log("optimize", "SKIP", "复现未成功,跳过优化")
            return {"optimized": False, "reason": "复现未成功,跳过优化"}

        baseline = self._get_baseline(paper_info, validation)
        arms = self._propose_arms(paper_info)

        self.log("optimize", "START",
                 f"开始优化: 方向数={len(arms)}, 预算={self.max_trials}",
                 {"baseline": baseline})

        scheduler = UCBScheduler(arms, budget=self.max_trials, c=self.c)
        best = {"arm": None, "improvement": 0.0, "result": baseline}
        records = []

        while not scheduler.is_exhausted():
            arm = scheduler.select_arm()
            reward = self._simulate_trial(arm, baseline)
            scheduler.update(arm, reward)

            kept = reward > 0
            records.append({"arm": arm, "improvement": round(reward, 4), "kept": kept})
            if kept and reward > best["improvement"]:
                best = {"arm": arm, "improvement": reward,
                        "result": baseline * (1 + reward)}

        self.log("optimize", "SUCCESS",
                 f"优化完成,最优方向={best['arm']},改进={best['improvement']:.2%}",
                 {"best": best, "trace": scheduler.trace()})

        return {
            "optimized": True,
            "optimization_report": records,
            "ucb_trace": scheduler.trace(),
            "best_result": best["result"],
            "best_arm": best["arm"],
            "improvement": best["improvement"],
            "baseline": baseline,
        }

    def _get_baseline(self, paper_info, validation) -> float:
        """取复现基线指标:优先论文声明值,其次验证结果里的实际值。"""
        metrics = paper_info.get("metrics", {})
        if metrics:
            return float(list(metrics.values())[0])

        comp = validation.get("metrics_comparison", {})
        actual = comp.get("actual", {})
        if actual:
            return float(list(actual.values())[0])
        return 0.5

    def _propose_arms(self, paper_info) -> list:
        """LLM 建议方向 + 固定方向合并;方向数不超过预算。"""
        prompt = f"""针对论文方法提出 2-3 条具体优化建议。
方法: {paper_info.get('method', '未知')}
返回 JSON: {{"suggestions": ["建议1", "建议2"]}}
"""
        llm_result = self.llm.chat(prompt)
        suggestions = []
        try:
            parsed = json.loads(llm_result)
            suggestions = parsed.get("suggestions", [])
        except (json.JSONDecodeError, AttributeError):
            pass

        arms = list(suggestions) + self.DEFAULT_ARMS
        return arms[: self.max_trials]

    def _simulate_trial(self, arm: str, baseline: float) -> float:
        """Mock 模式:确定性模拟某方向的改进幅度(可复现)。

        真实模式下此处应为实际重跑代码得到的指标增量。
        用 arm 名称哈希得到固定"潜力",保证同一方向每次结果一致;
        约 1/3 方向为负,用于演示 Keep/Reject。
        """
        h = int(hashlib.md5(arm.encode("utf-8")).hexdigest()[:8], 16)
        base = (h % 10000) / 10000.0  # 0.0 ~ 1.0
        return round((base - 0.35) * 0.15, 4)  # -0.0525 ~ +0.0975
