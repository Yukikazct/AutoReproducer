"""Online rejections retain reasons without bypassing source validation."""
from types import SimpleNamespace
import json

import pytest

from src import method_advice
from src.method_profiles import method_profile
from src.agents.report_generator import ReportGeneratorAgent


LLM = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test",
                      api_key="private-test-key")
SOURCES = [{"source_id": "demo", "url": "https://example.org/pinned/demo.py",
            "locator": "demo.py", "content": "optimizer = RMSprop(parameters, lr=1e-3)"}]


def accepted(**extra):
    return {"status": "accepted", "summary": "作者示例使用 RMSprop。",
            "evidence": [{"source_id": "demo", "quote": "RMSprop(parameters, lr=1e-3)"}], **extra}


def test_finder_rejection_preserves_reader_and_exact_response(monkeypatch):
    rejected = {"status": "insufficient_evidence", "summary": "缺少库入口与示例调用的映射说明。",
                "evidence": []}
    raw = json.dumps(rejected, ensure_ascii=False)
    responses = iter([json.dumps(accepted()), raw])
    monkeypatch.setattr(method_advice, "request_text", lambda *args: {"response": next(responses)})
    stages = []
    with pytest.raises(method_advice.SourceReviewError, match="缺少库入口") as raised:
        method_advice.review_sources(LLM, method_profile("neural_ode_spiral"), SOURCES,
                                     lambda *args: stages.append(args))
    analysis = raised.value.analysis
    assert analysis["failed_role"] == "finder"
    assert analysis["status"] == "rejected"
    assert [r["role"] for r in analysis["reviews"]] == ["reader"]
    assert analysis["attempts"][-1]["raw_response"] == raw
    assert analysis["attempts"][-1]["parsed_response"] == rejected
    assert stages == [("reader", "running"), ("reader", "success"), ("finder", "running")]
    assert LLM.api_key not in json.dumps(analysis)


@pytest.mark.parametrize("payload,reason", [
    ("not-json", "有效 JSON"),
    ("[]", "JSON 对象"),
    (json.dumps(accepted(status="maybe")), "未知审核状态"),
    (json.dumps(accepted(evidence=["bad"])), "引用必须是 JSON 对象"),
    (json.dumps(accepted(evidence=[{"source_id": "demo", "quote": "invented result"}])), "不符合公开原文"),
])
def test_malformed_or_invented_evidence_remains_rejected(monkeypatch, payload, reason):
    monkeypatch.setattr(method_advice, "request_text", lambda *args: {"response": payload})
    with pytest.raises(method_advice.SourceReviewError, match=reason) as raised:
        method_advice.review_sources(LLM, method_profile("neural_ode_spiral"), SOURCES, lambda *args: None)
    assert raised.value.analysis["reviews"] == []
    assert raised.value.analysis["attempts"][0]["raw_response"] == payload


def test_review_scope_separates_author_evidence_from_project_contract(monkeypatch):
    contexts = []
    def request(llm, prompt, timeout):
        contexts.append(json.loads(prompt.split("\n", 1)[1]))
        return {"response": json.dumps(accepted(role="forged-role"))}
    monkeypatch.setattr(method_advice, "request_text", request)
    profile = method_profile("neural_ode_spiral")
    analysis = method_advice.review_sources(LLM, profile, SOURCES, lambda *args: None)
    assert analysis["status"] == "accepted"
    assert [r["role"] for r in analysis["reviews"]] == ["reader", "finder", "builder", "verifier"]
    assert contexts[0]["experiment_scope"] == {
        "kind": "official_method_experiment", "paper_table_reproduction": False,
        "paper_full_text_provided": False, "execution_completed": False}
    assert contexts[0]["experiment_contract"]["parameters"] == profile["parameters"]
    assert len(contexts[-1]["previous_reviews"]) == 3
    assert contexts[0]["sources"] == SOURCES


def test_report_explains_failed_review_and_preserves_accepted_review():
    report = ReportGeneratorAgent()._build_report({
        "experiment_spec": method_profile("neural_ode_spiral"),
        "method_analysis": {"status": "rejected", "failed_role": "finder",
                            "reason": "缺少入口映射", "reviews": [{"role": "reader", **accepted()}]},
        "execution": {"mode": "repository", "executed": False, "success": False},
        "validation": {"status": "analysis_failed", "result_level": "failed"},
        "analysis_status": "public_readiness_rejected",
    })
    assert "**失败阶段**: 资源核对" in report
    assert "缺少入口映射" in report
    assert "训练尚未开始" in report
    assert "**执行状态**: ⏳ 未运行" in report
    assert "❌ 失败" not in report
    assert "在线来源分析：reader" in report
    assert "method_analysis.json" in report
