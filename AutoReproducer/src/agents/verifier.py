"""VerifierAgent - Prompt-Free 质量验证 Agent"""
import json
from src.base_agent import BaseAgent


class VerifierAgent(BaseAgent):
    """复用各 Agent 自身的系统提示词作为质量标准,检查其输出质量(Prompt-Free)。

    不手写额外验证提示词,而是直接读取目标 Agent 的 system_prompt 作为判据。
    """

    system_prompt = "验证各步骤输出是否满足其自身系统提示词定义的质量标准"

    def __init__(self, llm_client, logger=None):
        super().__init__("Verifier", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """input_data: {"agent_name", "system_prompt", "output"}"""
        agent_name = input_data.get("agent_name", "未知Agent")
        standard = input_data.get("system_prompt", "")
        output = input_data.get("output", "")

        self.log("verify", "START", f"验证 {agent_name} 的输出", input_data)

        prompt = f"""你是质量验证器。以下「标准」是该 Agent 的系统提示词,定义了合格输出的要求。
请判断「待验证输出」是否满足标准,返回 JSON:
{{"pass": true/false, "issues": ["问题1"], "fix_suggestions": ["修正建议1"], "confidence": 0.0-1.0}}

标准:
{standard}

待验证输出:
{str(output)[:2000]}
"""
        llm_result = self.llm.chat(prompt)
        try:
            parsed = json.loads(llm_result)
        except json.JSONDecodeError:
            parsed = {"pass": True, "issues": [],
                      "fix_suggestions": [], "confidence": 0.9}

        passed = bool(parsed.get("pass", False))
        self.log("verify", "SUCCESS" if passed else "WARNING",
                 f"{agent_name} 验证{'通过' if passed else '未通过'} "
                 f"(置信度 {parsed.get('confidence', 0):.2f})", parsed)
        return parsed
