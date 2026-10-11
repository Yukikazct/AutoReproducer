"""Executor retries must receive bounded feedback and the actual mode limits."""
from copy import deepcopy

import pytest

from src.agents.code_executor import CodeExecutorAgent, _INSUFFICIENT_INFO_MARK
from src.llm.llm_client import LLMClient


FEEDBACK = {
    "agent": "CodeExecutor",
    "state": "EXECUTE_CODE",
    "attempt": 2,
    "issues": ["执行前被拒绝，尚无实测数值"],
    "fix_suggestions": ["使用已有本地数据并关闭数据集下载"],
    "previous_failure": "拒绝执行: urllib 网络外联",
}
PAPER_INFO = {"title": "ReZero", "method": "Transformer", "dataset": "CIFAR-10"}


def capture_generation(monkeypatch, agent):
    prompts = []

    def chat(prompt, **kwargs):
        prompts.append(prompt)
        return "print('test-only-execution')\n"

    def execute(code, paper_info):
        final = {"success": True, "stdout": "test-only-execution", "stderr": "", "exit_code": 0}
        return code, [{"stage": "full", **final}], [], final

    monkeypatch.setattr(agent.llm, "chat", chat)
    monkeypatch.setattr(agent, "_execute_with_repair", execute)
    return prompts


def test_retry_feedback_reaches_generated_prompt_without_mutating_input(monkeypatch):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    prompts = capture_generation(monkeypatch, agent)
    feedback = deepcopy(FEEDBACK)

    result = agent.run({"paper_info": PAPER_INFO, "retry_feedback": feedback})

    assert result["success"] is True
    assert feedback == FEEDBACK
    assert len(prompts) == 1
    prompt = prompts[0]
    for expected in (FEEDBACK["previous_failure"], FEEDBACK["issues"][0],
                     FEEDBACK["fix_suggestions"][0], "针对这些问题修复并输出完整脚本",
                     "保留正常的 model.eval()", "不得为获得通过"):
        assert expected in prompt


@pytest.mark.parametrize("next_feedback", [
    None,
    {**FEEDBACK, "agent": "EnvBuilder"},
    {**FEEDBACK, "state": "BUILD_ENV"},
    "invalid-feedback",
])
def test_feedback_is_reset_and_cannot_leak_between_invocations(monkeypatch, next_feedback):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    prompts = capture_generation(monkeypatch, agent)
    agent.run({"paper_info": PAPER_INFO, "retry_feedback": FEEDBACK})
    agent.run({"paper_info": PAPER_INFO, "retry_feedback": next_feedback})

    assert FEEDBACK["previous_failure"] in prompts[0]
    assert "【本次重试反馈" not in prompts[1]
    assert FEEDBACK["previous_failure"] not in prompts[1]
    assert agent._retry_feedback is None


def test_feedback_prompt_is_bounded_and_excludes_previous_code(monkeypatch):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    prompts = capture_generation(monkeypatch, agent)
    feedback = {**FEEDBACK, "previous_failure": {"reason": "R" * 10000, "code": "EXCLUDED_CODE"},
                "issues": ["I" * 10000] * 100, "fix_suggestions": ["F" * 10000] * 100}

    agent.run({"paper_info": PAPER_INFO, "retry_feedback": feedback})

    feedback_prompt = prompts[0].split("【本次重试反馈", 1)[1]
    assert len(feedback_prompt) < 6400
    assert "EXCLUDED_CODE" not in feedback_prompt
    assert "R" * 1300 not in feedback_prompt
    assert "I" * 500 not in feedback_prompt


def test_local_generation_prompt_sets_execution_and_data_limits():
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    prompt = agent._generate_code_prompt(PAPER_INFO)

    for expected in ("本地执行", "内置 eval/exec", "命令/子进程执行或网络下载",
                     "下载选项必须关闭", "model.eval()", "缺少真实本地数据",
                     _INSUFFICIENT_INFO_MARK, "不得拿合成数据替代论文数据"):
        assert expected in prompt


def test_docker_prompt_uses_container_policy_without_local_download_ban():
    agent = CodeExecutorAgent(LLMClient(mock_mode=True), use_docker=True)
    prompt = agent._generate_code_prompt(PAPER_INFO)

    assert "Docker 容器执行" in prompt
    assert "容器权限与网络策略" in prompt
    assert "本地执行" not in prompt
    assert "下载选项必须关闭" not in prompt
    assert "命令/子进程执行或网络下载" not in prompt
    assert "不得拿合成数据替代论文数据" in prompt


def test_real_continuation_preserves_mode_and_evidence_constraints(monkeypatch):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    prompts = capture_generation(monkeypatch, agent)
    agent._continue_code(PAPER_INFO, "print('partial')", insufficient=True)

    assert _INSUFFICIENT_INFO_MARK in prompts[0]
    assert "下载选项必须关闭" in prompts[0]
    assert "未给出的量用合成数据与默认值" not in prompts[0]


def test_mock_continuation_retains_explicit_demo_behavior(monkeypatch):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
    prompts = capture_generation(monkeypatch, agent)
    agent._continue_code({}, "print('partial')", insufficient=True)

    assert "未给出的量用合成数据与默认值" in prompts[0]
    assert _INSUFFICIENT_INFO_MARK not in prompts[0]
