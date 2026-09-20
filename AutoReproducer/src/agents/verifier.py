"""VerifierAgent - Prompt-Free 质量验证 Agent（方案创新点三）。

复用各 Agent 自身的系统提示词作为质量标准，检查输出质量；
验证不通过时给出修正建议（fix_suggestions），供优化闭环触发修正。
"""
import json
import re
from typing import Dict
from src.base_agent import BaseAgent

# 待验证输出送进 LLM 的字符上限。旧值 2000 对代码类输出太小：
# 一份 3600 字符的复现脚本会被从中截断，验证器据此误报"代码被截断"。
VERIFY_INPUT_LIMIT = 8000


class VerifierAgent(BaseAgent):
    """复用目标 Agent 的 system_prompt 作为判据进行质量验证（Prompt-Free）。"""

    system_prompt = "验证各步骤输出是否满足其自身系统提示词定义的质量标准"

    def __init__(self, llm_client, logger=None):
        super().__init__("Verifier", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """input_data: {"agent_name", "system_prompt", "output"}"""
        agent_name = input_data.get("agent_name", "未知Agent")
        standard = input_data.get("system_prompt", "") or ""
        output = input_data.get("output", "")

        self.log("verify", "START", f"验证 {agent_name} 的输出", input_data)

        prompt = f"""你是质量验证器。以下「标准」是该 Agent 的系统提示词,定义了合格输出的要求。
请判断「待验证输出」是否满足标准,返回 JSON:
{{"pass": true/false, "issues": ["问题1"], "fix_suggestions": ["修正建议1"], "confidence": 0.0-1.0}}

标准:
{standard}

待验证输出:
{self._render_output(output)}
"""
        llm_result = self.llm.chat(prompt, task="verifier")
        parsed = self._parse_json(llm_result)
        if not parsed or "pass" not in parsed:
            parsed = self._local_check(agent_name, output, standard)

        passed = bool(parsed.get("pass", False))
        self.log_experiment(
            "VERIFY", f"Prompt-Free 验证 {agent_name} 输出",
            inputs={"standard": standard[:200]},
            outputs=parsed,
            result={"pass": passed},
        )
        self.log("verify", "SUCCESS" if passed else "WARNING",
                 f"{agent_name} 验证{'通过' if passed else '未通过'} "
                 f"(置信度 {parsed.get('confidence', 0):.2f})", parsed)
        return {**parsed, "llm_calls": self._delta_llm_calls()}

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _render_output(output) -> str:
        """渲染待验证输出；超长时**标注**截断，避免验证器误判。

        旧实现直接 `str(output)[:2000]`：验证器只能看到前 2000 字符，
        既看不到后面的真实缺陷，又会把自己看到的那半截**当成被测方的
        缺陷**报上来（例如对 3616 字符的代码报「code 字段被截断」——
        截断其实发生在验证器自己的入参上）。这里放大窗口，并在确实截断
        时明确告知"是入参被截断，不代表内容缺失"。
        """
        text = str(output)
        if len(text) <= VERIFY_INPUT_LIMIT:
            return text
        return (f"{text[:VERIFY_INPUT_LIMIT]}\n"
                f"[⚠️ 以上为待验证输出的前 {VERIFY_INPUT_LIMIT} / {len(text)} "
                f"字符，因长度限制被截断——未展示的部分**不代表缺失**，"
                f"请勿仅因内容在此处结束就判定不完整]")

    def _delta_llm_calls(self) -> int:
        """本 Agent 本次 run() 期间新增的 LLM 调用次数（用于预算统计）。"""
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    # ---------------- 本地规则校验（LLM 不可用时兜底） ----------------

    @staticmethod
    def _local_check(agent_name: str, output, standard: str) -> Dict:
        """基于结构完整性的确定性检查：关键字段存在且 non-trivial。"""
        problems = []
        suggestions = []
        if output is None:
            problems.append("输出为空")
            suggestions.append("Agent 应返回结构化结果 dict")
        else:
            text = str(output)
            if len(text.strip()) < 5:
                problems.append("输出内容过短,缺少实质信息")
                suggestions.append("补充关键字段(如 title/paper_info/results)")
            # 结果类输出必须包含关键字段
            for field in ("paper_info", "resources", "env_config",
                          "execution", "validation", "report"):
                if field in text and ("{}" in text or "None" in text):
                    problems.append(f"字段 {field} 为空值")
                    suggestions.append(f"确保 {field} 包含实际内容")
        return {"pass": len(problems) == 0, "issues": problems,
                "fix_suggestions": suggestions,
                "confidence": 0.8 if not problems else 0.5}

    @staticmethod
    def _parse_json(text: str) -> Dict:
        for candidate in (text, re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)):
            if not candidate or not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        return {}