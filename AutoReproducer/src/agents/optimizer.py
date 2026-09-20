"""OptimizerAgent - 智能优化 Agent，基于 UCB 预算调度（方案创新点二/六）。

对齐方案「Phase 6: 智能优化」：
- 仅在复现成功后触发；分析论文方法提出可证伪的改进假设；
- 尝试方向：超参数调优、模块替换、架构调整；
- UCB 算法在预算内分配尝试资源，真实执行后 Keep/Reject；
- 输出优化报告：尝试记录、最佳结果、改进幅度。

融合增强（P1-⑦）：
- ResearchSpec 冻结契约：可传入 spec（dict 或 ResearchSpec），运行前后复算
  sha256 校验契约未被改动，Keep/Reject 与胜负判定按 spec.direction/min_delta
  执行，避免"优化过程中偷改验收标准"的泄漏；
- 隐藏 Holdout：对最终 best 候选用注入的 holdout_runner 做 holdout_repeats
  轮独立评估（目标/均值/稳定性/通过判定），防单次评估噪音与过拟合。
"""
import hashlib
import json
import math
from typing import Any, Dict, List, Optional

from src.base_agent import BaseAgent
from src.optimizer.research_spec import (
    ResearchSpec,
    run_hidden_holdout,
    verify_frozen_spec,
)
from src.optimizer.ucb_scheduler import UCBScheduler


class OptimizerAgent(BaseAgent):
    """复现成功后，提出优化方向并用 UCB 在预算内智能分配尝试资源（Keep/Reject）。"""

    system_prompt = "基于复现结果提出可证伪的优化假设,并在预算内用 UCB 智能调度尝试"

    DEFAULT_ARMS = [
        "超参数调优(学习率)",
        "超参数调优(batch size)",
        "模块替换(优化器)",
        "模块替换(激活函数)",
        "架构调整(层数/宽度)",
        "架构调整(正则化)",
    ]

    def __init__(self, llm_client, logger=None, max_trials=10, c=1.0,
                 simulator=None, spec: Optional[ResearchSpec | Dict] = None,
                 holdout_runner=None):
        super().__init__("Optimizer", logger)
        self.llm = llm_client
        self.max_trials = max_trials
        self.c = c
        # simulator: 可注入的真实执行器（如根据方向修改代码并重跑 CodeExecutor），
        # 默认使用确定性 Mock 模拟（演示 Keep/Reject 机制）。
        self.simulator = simulator or self._simulate_trial
        # spec: 冻结契约对象；run() 时可用 input_data["spec"] 覆盖。
        self.spec = self._coerce_spec(spec)
        # _spec_input: 原始输入（dict 携带 spec_sha256 时运行期复算校验）。
        self._spec_input = spec
        # holdout_runner: (arm, run_index) -> {"exit_code", "metrics"}。
        self.holdout_runner = holdout_runner

    # ---------------- 结果判定 ----------------

    @staticmethod
    def _coerce_spec(spec: Any) -> Optional[ResearchSpec]:
        """把构造参数（ResearchSpec / dict / None）归一为 ResearchSpec。"""
        if spec is None:
            return None
        if isinstance(spec, ResearchSpec):
            return spec
        if isinstance(spec, dict):
            return ResearchSpec.from_dict(spec)
        raise TypeError(
            f"spec must be ResearchSpec or dict, got {type(spec).__name__}")

    def _reject_reason(self, spec: Optional[ResearchSpec],
                       input_spec: Any) -> str:
        """给出 spec 冻结校验的拒绝原因；无问题返回空串。"""
        if input_spec is None or spec is None:
            return ""
        declared = ""
        if isinstance(input_spec, dict):
            declared = str(input_spec.get("spec_sha256", ""))
        ok, message = verify_frozen_spec(spec, declared)
        return "" if ok else message

    def _keep_reward(self, spec: Optional[ResearchSpec], reward: float,
                     baseline: float) -> float:
        """把方向相对改进率换算成契约内的得分口径。

        无 spec 时保持兼容：直接返回 reward（正=Keep）。
        """
        if spec is None:
            return float(reward)
        result = baseline * (1 + float(reward))
        return result

    def run(self, input_data: dict) -> dict:
        """input_data 需含 paper_info、validation(含 is_reproduced)。
        可选: spec（dict/ResearchSpec 冻结契约）与 holdout_runner。
        """
        paper_info = input_data.get("paper_info", {}) or {}
        validation = input_data.get("validation", {}) or {}

        if not validation.get("is_reproduced", False):
            self.log("optimize", "SKIP", "复现未成功,跳过优化")
            return {"optimized": False, "reason": "复现未成功,跳过优化"}

        # 冻结契约：构造参数优先，input_data 可覆盖；运行前后校验指纹。
        spec = self.spec
        input_spec = None
        if "spec" in input_data and input_data.get("spec") is not None:
            input_spec = input_data["spec"]
            spec = self._coerce_spec(input_spec)
        elif self._spec_input is not None:
            input_spec = self._spec_input
        reject = self._reject_reason(spec, input_spec)
        if spec is not None and reject:
            self.log("optimize", "ABORT",
                     f"Spec 冻结校验失败,中止优化: {reject}")
            return {"optimized": False, "reason": f"spec 冻结校验失败: {reject}",
                    "spec_sha256": spec.sha256(), "spec_frozen": False}
        spec_sha256 = spec.sha256() if spec else None

        baseline = self._get_baseline(paper_info, validation)
        arms = self._propose_arms(paper_info)
        if not arms:
            arms = self.DEFAULT_ARMS
        arms = arms[: max(1, self.max_trials)]

        self.log("optimize", "START",
                 f"开始优化: 方向数={len(arms)}, 预算={self.max_trials}"
                 + (f", spec_sha256={spec_sha256[:16]}" if spec_sha256 else ""),
                 {"baseline": baseline})

        scheduler = UCBScheduler(arms, budget=self.max_trials, c=self.c)
        best = {"arm": None, "improvement": 0.0, "result": baseline}
        records: List[Dict] = []

        while not scheduler.is_exhausted():
            arm = scheduler.select_arm()
            if arm is None:  # 防御：预算耗尽（与循环条件保持一致）
                break
            reward, trial_detail = self._run_trial(arm, baseline)
            scheduler.update(arm, reward)

            # 按契约判定 Keep：maximize 方向 result 增高 / minimize 方向降低
            result = baseline * (1 + float(reward))
            kept = False
            if spec is not None:
                kept = spec.improves(result, baseline)
            else:
                kept = abs(reward) > 1e-9 and reward > 0
            records.append({
                "arm": arm,
                "improvement": round(float(reward), 4),
                "result": round(float(result), 6),
                "kept": kept,
                "detail": trial_detail,
            })
            if kept:
                margin = reward if spec is None else (
                    result - baseline if spec.direction == "maximize"
                    else baseline - result)
                if best["arm"] is None or margin > best["improvement"]:
                    best = {"arm": arm, "improvement": float(margin),
                            "result": float(result)}

        # 隐藏 Holdout：只对最终 best 状态多轮独立评估，防单次噪音。
        holdout = run_hidden_holdout(spec, best["arm"], self.holdout_runner) \
            if spec is not None else {
                "mode": "disabled", "reason": "no frozen spec",
                "passed": False}

        self.log_experiment(
            "OPTIMIZE", "UCB 预算调度下完成优化尝试",
            inputs={"arms": arms, "budget": self.max_trials, "baseline": baseline},
            outputs={"records": records, "best": best,
                     "ucb_trace": scheduler.trace()},
            result={"best_arm": best["arm"], "improvement": best["improvement"]},
        )
        self.log("optimize", "SUCCESS",
                 f"优化完成,最优方向={best['arm']},改进={best['improvement']:.2%}"
                 + (f", holdout={'PASS' if holdout.get('passed') else 'FAIL'}"
                    if spec is not None else ""),
                 {"best": best, "trace": scheduler.trace()})

        return {
            "optimized": True,
            "optimization_report": records,
            "ucb_trace": scheduler.trace(),
            "best_result": best["result"],
            "best_arm": best["arm"],
            "improvement": best["improvement"],
            "baseline": baseline,
            "budget_used": scheduler.total_pulls,
            "budget": self.max_trials,
            # ---- P1-⑦ 冻结契约（无 spec 时为 None，向后兼容） ----
            "spec_sha256": spec_sha256,
            "spec_frozen": True if spec is not None else False,
            "holdout": holdout,
            "spec_pass": (holdout.get("passed") if spec is not None else None),
        }

    # ---------------- 内部工具 ----------------

    def _get_baseline(self, paper_info: Dict, validation: Dict) -> float:
        """取复现基线指标：优先论文声明值，其次验证结果里的实际值。"""
        metrics = paper_info.get("metrics", {}) or {}
        if metrics:
            try:
                return float(list(metrics.values())[0])
            except (TypeError, ValueError):
                pass
        comp = validation.get("metrics_comparison", {}) or {}
        actual = comp.get("actual", {}) or {}
        if actual:
            try:
                return float(list(actual.values())[0])
            except (TypeError, ValueError):
                pass
        return 0.5

    def _propose_arms(self, paper_info: Dict) -> List[str]:
        """LLM 建议方向 + 固定方向合并；方向数不超过预算。"""
        prompt = f"""针对论文方法提出 2-3 条具体优化建议。
方法: {paper_info.get('method', '未知')}
返回 JSON: {{"suggestions": ["建议1", "建议2"]}}
"""
        llm_result = self.llm.chat(prompt, task="optimizer_arms")
        suggestions: List[str] = []
        try:
            parsed = json.loads(llm_result)
            if isinstance(parsed, dict):
                suggestions = [str(s) for s in parsed.get("suggestions", [])]
        except (json.JSONDecodeError, AttributeError):
            match = None
            try:
                import re
                match = re.search(r"\{.*\}", llm_result, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    suggestions = [str(s) for s in parsed.get("suggestions", [])]
            except (json.JSONDecodeError, AttributeError):
                pass

        arms = list(suggestions) + self.DEFAULT_ARMS
        return arms[: max(1, self.max_trials)]

    def _run_trial(self, arm: str, baseline: float):
        """执行一次优化尝试，返回 (reward, detail)。

        reward > 0 表示相对基线的改进（Keep），否则 Reject。
        """
        if self.simulator is not self._simulate_trial:
            result = self.simulator(arm, baseline)
            return result if isinstance(result, tuple) else (result, {})

        reward = self._simulate_trial(arm, baseline)
        detail = {"type": "ucb_mock", "note":
                  "Mock 模式:确定性模拟该方向的改进潜力（可复现）"}
        return reward, detail

    @staticmethod
    def _simulate_trial(arm: str, baseline: float) -> float:
        """Mock 模式：确定性模拟某方向的改进幅度（可复现）。

        用 arm 名称哈希得到固定"潜力"，保证同一方向每次结果一致；
        约 1/3 方向为负，用于演示 Keep/Reject。
        """
        h = int(hashlib.md5(arm.encode("utf-8")).hexdigest()[:8], 16)
        base = (h % 10000) / 10000.0  # 0.0 ~ 1.0
        return round((base - 0.35) * 0.15, 4)  # -0.0525 ~ +0.0975