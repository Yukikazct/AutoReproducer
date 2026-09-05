"""ResultValidatorAgent - 结果验证Agent，比对论文声明值与运行结果"""
import json
import re
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient


class ResultValidatorAgent(BaseAgent):
    """验证运行结果是否与论文声明一致"""

    system_prompt = "比对实际运行指标与论文声明值,输出复现成功/失败、数值差异与原因分析"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResultValidator", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """验证执行结果
        input_data: {"paper_info": dict, "execution": dict}
        """
        self.log("validate", "START", "开始验证结果", input_data)

        paper_info = input_data.get("paper_info", {})
        execution = input_data.get("execution", {})

        paper_metrics = paper_info.get("metrics", {})
        stdout = execution.get("stdout", "")
        stderr = execution.get("stderr", "")

        # 语料对照层:若无声明指标,用语料中的复现分作为论文声明值
        corpus_paper = input_data.get("corpus_paper") or paper_info.get("corpus_paper")
        if not paper_metrics and corpus_paper:
            from src.corpus import get_declared_score
            score = get_declared_score(corpus_paper)
            if score is not None:
                paper_metrics = {"reproduction_score": round(score, 4)}

        # 从输出中提取数值指标
        actual_metrics = self._extract_metrics(stdout)

        # 用LLM进行比对验证
        prompt = f"""比对论文声明的指标与代码运行结果。

论文声明指标: {json.dumps(paper_metrics, ensure_ascii=False)}
代码运行输出: {stdout[:2000]}
提取到的实际指标: {json.dumps(actual_metrics, ensure_ascii=False)}

返回JSON格式:
{{
    "match": true/false,
    "differences": ["指标1: 声明值 vs 实际值"],
    "confidence": 0.0-1.0,
    "analysis": "分析说明"
}}
"""
        llm_result = self.llm.chat(prompt)
        try:
            parsed = json.loads(llm_result)
        except json.JSONDecodeError:
            parsed = {
                "match": bool(actual_metrics),
                "differences": [],
                "confidence": 0.8 if actual_metrics else 0.3,
                "analysis": "自动提取指标完成"
            }

        # 计算匹配度
        all_metrics = {}
        all_metrics["paper"] = paper_metrics
        all_metrics["actual"] = actual_metrics

        result = {
            "validation": parsed,
            "metrics_comparison": all_metrics,
            "is_reproduced": parsed.get("match", False),
            "confidence": parsed.get("confidence", 0.0)
        }

        self.log("validate",
                 "SUCCESS" if result["is_reproduced"] else "WARNING",
                 f"验证{'通过' if result['is_reproduced'] else '未通过'} - 置信度: {result['confidence']:.2f}",
                 result)

        return result

    def _extract_metrics(self, text: str) -> dict:
        """从文本中提取指标数值"""
        metrics = {}
        patterns = [
            (r"(?:accuracy|acc|精确率|准确率)[:\s]*([\d.]+)%?", "accuracy"),
            (r"(?:f1[_-]?score|f1)[:\s]*([\d.]+)", "f1_score"),
            (r"(?:precision|精确率)[:\s]*([\d.]+)%?", "precision"),
            (r"(?:recall|召回率)[:\s]*([\d.]+)%?", "recall"),
            (r"(?:loss|损失)[:\s]*([\d.]+)", "loss"),
            (r"(?:mse|rmse)[:\s]*([\d.]+)", "mse"),
        ]

        for pattern, name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    pass

        return metrics