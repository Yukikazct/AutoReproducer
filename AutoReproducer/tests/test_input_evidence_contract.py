"""Paper inputs must distinguish actual implementations from discovered hints."""
import json
from unittest.mock import Mock

import pytest

from src.agents import repo_discovery
from src.agents.code_executor import CodeExecutorAgent, _INSUFFICIENT_INFO_MARK
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.resource_finder import ResourceFinderAgent
from src.agents.result_validator import ResultValidatorAgent


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = responses
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response if isinstance(response, str) else json.dumps(response)

    def get_call_count(self):
        return self.calls


@pytest.mark.parametrize("response", ["", _INSUFFICIENT_INFO_MARK])
@pytest.mark.parametrize("paper", [{}, {"method": "known method", "dataset": "known dataset"}])
def test_real_generation_refusal_never_runs_unrelated_fallback(response, paper):
    llm = ScriptedLLM(response)
    events = []
    agent = CodeExecutorAgent(llm, logger=Mock(), on_event=events.append)
    agent._execute_code = Mock(side_effect=AssertionError("must not execute fallback"))
    result = agent.run({"paper_info": paper})
    assert result["not_runnable"] is True
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["fallback_used"] is False
    assert result["reproduction_scope"] == "generated_reconstruction"
    assert result["repository_executed"] is False
    assert llm.calls == 2  # Initial attempt and one evidence-based retry.
    assert events == []  # Preflight rejection starts no execution invocation.
    validation = ResultValidatorAgent(llm, logger=Mock()).run({"execution": result})
    assert validation["status"] == "insufficient_evidence"
    assert validation["result_level"] == "inconclusive"
    assert validation["is_reproduced"] is None


def test_real_incomplete_metadata_still_attempts_evidence_based_generation():
    llm = ScriptedLLM("def main():\n    print('real implementation')\nmain()")
    agent = CodeExecutorAgent(llm, logger=Mock())
    final = {"stage": "full", "exit_code": 0, "success": True, "stdout": "mse=0.1"}
    agent._execute_with_repair = Mock(return_value=("code", [final], [], final))
    result = agent.run({"paper_info": {"title": "Unknown paper", "insufficient_info": True},
                        "raw_text": "Known paper evidence: method uses cubic dynamics."})
    assert llm.calls == 1
    assert "cubic dynamics" in llm.prompts[0]
    assert "不得为了可运行" not in llm.prompts[0]  # Only the retry has this instruction.
    assert "严禁用无关模型" in llm.prompts[0]
    assert "即便如此也**必须**" not in llm.prompts[0]
    assert result["best_effort"] is True
    assert result["repository_executed"] is False
    validation = ResultValidatorAgent(llm, logger=Mock()).run(
        {"execution": result, "paper_info": {"metrics": {"mse": 0.1}}})
    assert validation["is_reproduced"] is None
    assert validation["status"] == "best_effort"


def test_mock_fallback_is_best_effort_even_with_sufficient_metadata():
    llm = ScriptedLLM(_INSUFFICIENT_INFO_MARK)
    agent = CodeExecutorAgent(llm, logger=Mock(), mock_mode=True)
    final = {"stage": "full", "exit_code": 0, "success": True, "stdout": "mse=0.1"}
    agent._execute_with_repair = Mock(return_value=("demo", [final], [], final))
    result = agent.run({"paper_info": {"method": "known", "dataset": "known"}})
    assert result["fallback_used"] is True
    assert result["best_effort"] is True
    assert result["reproduction_scope"] == "mock_demo"


def test_validator_cannot_accept_fallback_metrics_with_missing_best_effort_flag():
    result = ResultValidatorAgent(ScriptedLLM({"match": True}), logger=Mock()).run(
        {"paper_info": {"metrics": {"mse": 0.1}},
         "execution": {"fallback_used": True, "best_effort": False,
                       "final": {"stage": "full", "exit_code": 0, "success": True,
                                 "stdout": "mse=0.1"}}})
    assert result["status"] == "best_effort"
    assert result["is_reproduced"] is None


def test_paper_code_url_precedes_lexical_repository_search(monkeypatch):
    monkeypatch.setattr(repo_discovery, "_search_pwc", Mock(side_effect=AssertionError("unneeded search")))
    monkeypatch.setattr(repo_discovery, "_search_github", Mock(side_effect=AssertionError("unneeded search")))
    result = ResourceFinderAgent(ScriptedLLM({}), logger=Mock()).run(
        {"paper_info": {"title": "Unknown paper", "code_url": "https://github.com/author/project"},
         "raw_text": "Code: https://github.com/author/project"})["resources"]
    assert result["selected_repo"] == "https://github.com/author/project"
    assert result["repository_identity"]["status"] == "paper_linked"
    assert result["repository_identity"]["source"] == "paper_code_url"
    assert result["repository_identity"]["is_official"] is None
    assert result["repository_identity"]["repository_executed"] is False


def test_search_match_is_candidate_without_authorship_evidence(monkeypatch):
    monkeypatch.setattr(repo_discovery, "_search_pwc", lambda *a, **k: ([], "timeout"))
    candidate = repo_discovery.RepoCandidate(
        repo_name="thirdparty/Quantum-Entropy", description="Quantum entropy paper implementation",
        repo_urls=["https://github.com/thirdparty/Quantum-Entropy"], source="github_search", score_hint=10)
    monkeypatch.setattr(repo_discovery, "_search_github", lambda *a, **k: ([candidate], ""))
    result = ResourceFinderAgent(ScriptedLLM({"code_repo_url": "guess", "confidence": 1}), logger=Mock()).run(
        {"paper_info": {"title": "Quantum Entropy"}})["resources"]
    assert result["selected_repo"] == candidate.repo_urls[0]
    assert result["repository_identity"]["status"] == "candidate_unverified"
    assert result["repository_identity"]["source"] == "github_search"
    assert result["confidence"] <= 0.5


def test_model_metadata_repo_has_no_paper_link_evidence():
    result = ResourceFinderAgent(ScriptedLLM({}), logger=Mock(), offline=True).run(
        {"paper_info": {"title": "Unknown paper", "code_url": "https://github.com/guess/project"},
         "raw_text": "This paper text contains no code URL."})["resources"]
    assert result["repository_identity"]["status"] == "candidate_unverified"
    assert result["repository_identity"]["source"] == "paper_code_url"


def test_extracted_later_page_code_link_retains_identity_evidence():
    result = ResourceFinderAgent(ScriptedLLM({}), logger=Mock(), offline=True).run(
        {"paper_info": {"title": "Unknown paper", "code_url": "https://github.com/author/project"},
         "raw_text": "First page without the link.",
         "extracted_code_urls": ["https://github.com/author/project"]})["resources"]
    assert result["repository_identity"]["status"] == "paper_linked"


def test_unrelated_paper_commit_does_not_pin_selected_repository():
    result = ResourceFinderAgent(ScriptedLLM({}), logger=Mock(), offline=True).run(
        {"paper_info": {"title": "Unknown paper"},
         "preferred_repo_url": "https://github.com/requested/repo",
         "raw_text": "Citation https://github.com/unrelated/repo/commit/abcd1234"})["resources"]
    assert "pinned_revision" not in result["repo_discovery"]


def test_report_states_evidence_gap_and_generated_execution_scope():
    llm = ScriptedLLM(_INSUFFICIENT_INFO_MARK)
    execution = CodeExecutorAgent(llm, logger=Mock()).run({"paper_info": {}})
    validation = ResultValidatorAgent(llm, logger=Mock()).run({"execution": execution})
    report = ReportGeneratorAgent(logger=Mock()).run(
        {"paper_info": {"title": "Unknown paper"}, "execution": execution, "validation": validation,
         "resources": {"code_repo_url": "https://github.com/guess/repo", "repository_identity": {
             "source": "github_search", "reason": "候选，尚未核验与论文作者关联。"}}})["report"]
    assert "未运行已发现仓库" in report
    assert "候选，尚未核验与论文作者关联" in report
    assert "无法核验（论文证据不足" in report
    assert "系统本地兜底脚本" not in report
