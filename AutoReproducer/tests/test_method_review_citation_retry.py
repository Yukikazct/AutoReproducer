"""Only malformed citations get one explicit, recorded correction per role."""
from copy import deepcopy
import json
import subprocess
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.method_advice as advice
from src.method_profiles import method_profile


ROLES = ("reader", "finder", "builder", "verifier")
MODEL_QUOTE = "class ODEFunc(nn.Module):"
LOSS_QUOTE = (
    "        pred_y = odeint(func, batch_y0, batch_t).to(device)\n"
    "        loss = torch.mean(torch.abs(pred_y - batch_y))"
)
OFFICIAL_SOURCE = {
    "source_id": "author_ode_demo",
    "origin": "official_repository",
    "url": "https://github.com/rtqichen/torchdiffeq/blob/fixed/examples/ode_demo.py",
    "locator": "examples/ode_demo.py",
    "content": (
        "class ODEFunc(nn.Module):\n"
        "    def forward(self, t, y):\n        return self.net(y**3)\n\n"
        "if __name__ == '__main__':\n"
        "    for itr in range(1, args.niters + 1):\n" + LOSS_QUOTE + "\n"
    ),
}


def accepted_payload(role):
    official = {"source_id": "author_ode_demo", "quote": LOSS_QUOTE}
    evidence = [official, {"source_id": "author_ode_demo", "quote": MODEL_QUOTE}]
    if role in {"builder", "verifier"}:
        evidence = [official, {"source_id": "project_frozen_contract", "quote": '"seed": 2021'}]
    return {"status": "accepted", "summary": f"Supported scope for {role}", "evidence": evidence}


def response(payload):
    return {"response": json.dumps(payload, ensure_ascii=False), "usage": {"total_tokens": 7}}


def accepted_response(role):
    return response(accepted_payload(role))


def bad_citation_response(role, kind="indentation"):
    payload = accepted_payload(role)
    if kind == "indentation":
        payload["evidence"][0]["quote"] = dedent(LOSS_QUOTE)
        assert payload["evidence"][0]["quote"] not in OFFICIAL_SOURCE["content"]
    elif kind == "unknown_source":
        payload["evidence"][0]["source_id"] = "not_a_supplied_source"
    elif kind == "format":
        payload["evidence"][0] = "author_ode_demo: the author uses an absolute loss"
    else:
        raise AssertionError(f"Unexpected citation test case: {kind}")
    return response(payload)


@pytest.fixture
def review(monkeypatch):
    request = Mock()
    monkeypatch.setattr(advice, "request_text", request)
    llm = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test-model")
    stages = []
    sources = [deepcopy(OFFICIAL_SOURCE)]

    def run():
        return advice.review_sources(
            llm, method_profile("neural_ode_spiral"), sources,
            lambda role, status: stages.append((role, status)))

    return SimpleNamespace(run=run, request=request, stages=stages, sources=sources)


def completed_stage_events(roles=ROLES):
    return [(role, status) for role in roles for status in ("running", "success")]


def assert_attempt(attempt, role, number, status, raw_response):
    assert attempt["role"] == role and attempt["attempt"] == number
    assert attempt["status"] == status
    assert attempt["raw_response"] == raw_response
    if status == "rejected":
        assert isinstance(attempt["reason"], str) and attempt["reason"]


@pytest.mark.parametrize("kind", ["indentation", "unknown_source", "format"])
def test_one_bad_citation_is_corrected_by_a_second_api_response(review, kind):
    bad = bad_citation_response("finder", kind)
    corrected = accepted_response("finder")
    responses = [accepted_response("reader"), bad, corrected,
                 accepted_response("builder"), accepted_response("verifier")]
    review.request.side_effect = responses

    analysis = review.run()

    assert analysis["status"] == "accepted"
    assert review.request.call_count == 5
    assert review.stages == completed_stage_events()
    attempts = analysis["attempts"]
    assert [(attempt["role"], attempt["attempt"], attempt["status"]) for attempt in attempts] == [
        ("reader", 1, "accepted"), ("finder", 1, "rejected"), ("finder", 2, "accepted"),
        ("builder", 1, "accepted"), ("verifier", 1, "accepted")]
    assert_attempt(attempts[1], "finder", 1, "rejected", bad["response"])
    assert_attempt(attempts[2], "finder", 2, "accepted", corrected["response"])
    assert attempts[1]["parsed_response"] == json.loads(bad["response"])
    assert analysis["reviews"][1]["evidence"] == json.loads(corrected["response"])["evidence"]
    assert review.request.call_args_list[1].args[1] != review.request.call_args_list[2].args[1]
    assert review.sources == [OFFICIAL_SOURCE]


def test_each_role_has_an_independent_two_call_limit_and_preserves_original_quotes(review):
    responses = []
    for role in ROLES:
        responses.extend([bad_citation_response(role), accepted_response(role)])
    review.request.side_effect = responses

    analysis = review.run()

    assert analysis["status"] == "accepted" and review.request.call_count == 8
    assert review.stages == completed_stage_events()
    assert [item["role"] for item in analysis["reviews"]] == list(ROLES)
    assert len(analysis["attempts"]) == 8
    for index, role in enumerate(ROLES):
        rejected, accepted = analysis["attempts"][2 * index:2 * index + 2]
        assert_attempt(rejected, role, 1, "rejected", responses[2 * index]["response"])
        assert_attempt(accepted, role, 2, "accepted", responses[2 * index + 1]["response"])
        assert rejected["parsed_response"]["evidence"][0]["quote"] == dedent(LOSS_QUOTE)
        assert analysis["reviews"][index]["evidence"] == accepted_payload(role)["evidence"]
    assert review.sources == [OFFICIAL_SOURCE]


@pytest.mark.parametrize("role", ROLES)
def test_correction_feedback_identifies_original_citation_and_allowed_sources(review, role):
    index = ROLES.index(role)
    payload = accepted_payload(role)
    payload["evidence"][1]["quote"] += " # this text is absent from the source"
    original_citation = deepcopy(payload["evidence"][1])
    bad = response(payload)
    review.request.side_effect = [accepted_response(previous) for previous in ROLES[:index]] + [
        bad, accepted_response(role), *[accepted_response(later) for later in ROLES[index + 1:]]]
    original_sources = deepcopy(review.sources)

    analysis = review.run()

    initial_context = json.loads(review.request.call_args_list[index].args[1].split("\n", 1)[1])
    corrected_context = json.loads(review.request.call_args_list[index + 1].args[1].split("\n", 1)[1])
    assert "citation_correction" not in initial_context
    correction = corrected_context["citation_correction"]
    assert correction["previous_attempt"] == 1
    assert len(correction["invalid_evidence"]) == 1
    invalid = correction["invalid_evidence"][0]
    assert invalid["evidence_index"] == 2
    assert invalid["source_id"] == original_citation["source_id"]
    assert invalid["quote"] == original_citation["quote"]
    expected_ids = ["author_ode_demo"]
    if role in {"builder", "verifier"}:
        expected_ids.append("project_frozen_contract")
    assert correction["allowed_source_ids"] == expected_ids
    rejected_attempt = analysis["attempts"][index]
    feedback = rejected_attempt["citation_feedback"]
    assert feedback == {key: value for key, value in correction.items() if key != "previous_attempt"}
    assert rejected_attempt["parsed_response"]["evidence"][1] == original_citation
    assert rejected_attempt["raw_response"] == bad["response"]
    assert initial_context["sources"] == corrected_context["sources"]
    assert review.sources == original_sources
    assert analysis["status"] == "accepted" and review.request.call_count == 5
    assert review.stages == completed_stage_events()


def test_long_summary_does_not_reject_otherwise_valid_source_review(review):
    summary = "说明限定在已提供的作者源码与项目适配，未宣称未执行的训练或论文表格已经通过。" * 30
    assert len(summary) > 400
    responses = []
    for role in ROLES:
        payload = accepted_payload(role)
        payload["summary"] = summary
        responses.append(response(payload))
    review.request.side_effect = responses

    analysis = review.run()

    assert analysis["status"] == "accepted" and review.request.call_count == 4
    assert review.stages == completed_stage_events()
    assert all(item["summary"] == summary for item in analysis["reviews"])
    assert all(item["attempt"] == 1 and item["status"] == "accepted" for item in analysis["attempts"])


@pytest.mark.parametrize("failed_role", ROLES)
def test_two_invalid_citations_stop_the_role_and_keep_both_failures(review, failed_role):
    index = ROLES.index(failed_role)
    first = bad_citation_response(failed_role, "unknown_source")
    second = bad_citation_response(failed_role, "indentation")
    review.request.side_effect = [accepted_response(role) for role in ROLES[:index]] + [
        first, second, AssertionError("A role must not make a third citation attempt")]

    with pytest.raises(advice.SourceReviewError) as caught:
        review.run()

    analysis = caught.value.analysis
    assert analysis["status"] == "rejected" and analysis["failed_role"] == failed_role
    assert review.request.call_count == index + 2
    assert review.stages == completed_stage_events(ROLES[:index]) + [(failed_role, "running")]
    assert [item["role"] for item in analysis["reviews"]] == list(ROLES[:index])
    assert_attempt(analysis["attempts"][-2], failed_role, 1, "rejected", first["response"])
    assert_attempt(analysis["attempts"][-1], failed_role, 2, "rejected", second["response"])
    assert analysis["attempts"][-2]["parsed_response"] == json.loads(first["response"])
    assert analysis["attempts"][-1]["parsed_response"] == json.loads(second["response"])


@pytest.mark.parametrize("failed_role", ROLES)
def test_real_insufficient_evidence_is_not_retried_or_rewritten(review, failed_role):
    index = ROLES.index(failed_role)
    summary = f"The required official fact for {failed_role} is unsupported."
    rejected = response({"status": "insufficient_evidence", "summary": summary, "evidence": []})
    review.request.side_effect = [accepted_response(role) for role in ROLES[:index]] + [
        rejected, AssertionError("Insufficient evidence must not receive a citation retry")]

    with pytest.raises(advice.SourceReviewError) as caught:
        review.run()

    analysis = caught.value.analysis
    assert analysis["failed_role"] == failed_role and analysis["status"] == "rejected"
    assert review.request.call_count == index + 1
    assert summary in analysis["reason"]
    assert review.stages == completed_stage_events(ROLES[:index]) + [(failed_role, "running")]
    assert_attempt(analysis["attempts"][-1], failed_role, 1, "rejected", rejected["response"])
    assert analysis["attempts"][-1]["parsed_response"] == json.loads(rejected["response"])


@pytest.mark.parametrize("failure", ["api_error", "invalid_json", "timeout", "transport_error"])
def test_api_and_json_failures_do_not_consume_a_citation_retry(review, failure):
    if failure == "api_error":
        rejected = {"error": "API failed", "response": "", "usage": {}}
    elif failure == "invalid_json":
        rejected = {"response": "{this is not valid JSON"}
    elif failure == "timeout":
        rejected = subprocess.TimeoutExpired(["fake-api-worker"], 45)
    else:
        rejected = OSError("API transport failed")
    review.request.side_effect = [accepted_response("reader"), accepted_response("finder"),
                                 rejected, AssertionError("This is not a citation failure")]

    with pytest.raises(advice.SourceReviewError) as caught:
        review.run()

    analysis = caught.value.analysis
    assert analysis["failed_role"] == "builder" and analysis["status"] == "rejected"
    assert review.request.call_count == 3
    assert review.stages == completed_stage_events(ROLES[:2]) + [("builder", "running")]
    raw = rejected.get("response", "") if isinstance(rejected, dict) else ""
    assert_attempt(analysis["attempts"][-1], "builder", 1, "rejected", raw)


@pytest.mark.parametrize("kind", ["missing_summary", "unknown_status", "not_an_object"])
def test_non_citation_response_contract_errors_are_not_retried(review, kind):
    payload = accepted_payload("reader")
    if kind == "missing_summary":
        payload.pop("summary")
    elif kind == "unknown_status":
        payload["status"] = "looks_good"
    else:
        payload = [payload]
    rejected = response(payload)
    review.request.side_effect = [rejected, AssertionError("Only citation errors may retry")]

    with pytest.raises(advice.SourceReviewError) as caught:
        review.run()

    analysis = caught.value.analysis
    assert analysis["failed_role"] == "reader" and analysis["reviews"] == []
    assert review.request.call_count == 1 and review.stages == [("reader", "running")]
    assert_attempt(analysis["attempts"][0], "reader", 1, "rejected", rejected["response"])
