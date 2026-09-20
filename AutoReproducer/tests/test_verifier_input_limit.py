"""Verifier 入参截断回归测试。

缺陷背景：旧实现把待验证输出硬截到 2000 字符（`str(output)[:2000]`），
导致 3616 字符的复现代码被从中间切断，验证器看到半截代码后**误报**
「code 字段被截断」——截断其实发生在验证器自己的入参上；同时 2000 字符
之后的真实缺陷验证器完全看不到。

运行: python -m pytest tests/test_verifier_input_limit.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.verifier import VerifierAgent, VERIFY_INPUT_LIMIT  # noqa: E402


class _CapturingLLM:
    """记录 prompt 的假 LLM，返回"通过"。"""

    def __init__(self):
        self.prompts = []
        self.call_count = 0

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        self.prompts.append(prompt)
        self.call_count += 1
        return ('{"pass": true, "issues": [], "fix_suggestions": [], '
                '"confidence": 0.9}')

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        self.call_count = 0


def _run_with(output):
    llm = _CapturingLLM()
    VerifierAgent(llm).run({"agent_name": "CodeExecutor",
                            "system_prompt": "在沙箱中安全执行论文代码",
                            "output": output})
    return llm.prompts[0]


def test_short_output_is_passed_through_verbatim():
    output = {"code": "print('accuracy=0.852')\n", "success": True}
    prompt = _run_with(output)
    assert "print('accuracy=0.852')" in prompt
    assert "不代表缺失" not in prompt       # 没截断就不该出现截断提示


def test_long_code_is_not_cut_at_2000_chars():
    """3600 字符量级的代码必须完整送进验证器（旧实现砍在 2000）。"""
    body = "\n".join(f"x_{i} = {i}" for i in range(300))
    output = {"code": body, "success": True}
    prompt = _run_with(output)
    assert f"x_{299} = 299" in prompt, "代码尾部应可见"
    assert "不代表缺失" not in prompt


def test_oversized_output_is_annotated_not_silently_cut():
    """真超过上限时必须标注"是入参被截断、不代表内容缺失"。"""
    huge = "a" * (VERIFY_INPUT_LIMIT + 5000)
    prompt = _run_with({"code": huge})
    assert "不代表缺失" in prompt
    assert "请勿仅因内容在此处结束就判定不完整" in prompt
    assert str(len(str({"code": huge}))) in prompt      # 标注里含真实长度


@pytest.mark.parametrize("limit_output", [
    {"code": "x" * 100},
    {"stdout": "y" * 50, "exit_code": 0},
])
def test_render_output_identity_below_limit(limit_output):
    assert VerifierAgent._render_output(limit_output) == str(limit_output)
