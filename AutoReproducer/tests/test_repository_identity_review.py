"""Evidence mechanics for LLM review of fetched paper/author repositories."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from discovered_fixtures import identity_review, make_experiment, write_pdf
from src.agents.repository_verifier import (RepositoryVerifierAgent, identity_view,
                                           validate_identity_review)
from src.discovered_repository_plan import build_evidence_packet, propose_plan


@pytest.fixture
def packet(tmp_path):
    fixture = make_experiment(tmp_path)
    return build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"],
                                 for_identity_review=True)


def test_plain_link_can_be_reviewed_but_cannot_start_planning(tmp_path):
    fixture = make_experiment(tmp_path)
    write_pdf(fixture["pdf"], [["Separating Signed Signals", "Abstract",
              "The implementation accompanying our experiments:", fixture["snapshot"]["url"]]])
    packet = build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"],
                                   for_identity_review=True)
    assert all(not link["is_author_code"] for link in packet["pdf"]["repository_links"])
    llm = Mock()
    with pytest.raises(ValueError, match="not been verified"):
        propose_plan(llm, packet, fixture["workspace"])
    llm.chat.assert_not_called()
    review = validate_identity_review(packet, identity_review(identity_view(packet)))
    assert review["accepted"] and review["repository_revision"] == "a" * 40


def test_candidate_location_beyond_page_viewport_remains_exact_source_evidence(packet):
    import hashlib
    page = packet['pdf']['pages'][0]
    url = packet['repository']['url']
    page['text'] = 'Title\n' + 'Earlier prose. ' * 1500 + '\nOur code is available at ' + url + '\nAUTHOR_CONTEXT'
    page['sha256'] = hashlib.sha256(page['text'].encode()).hexdigest()
    for link in packet['pdf']['repository_links']:
        link['raw_url'] = url
    view = identity_view(packet)
    assert url not in view['pdf']['pages'][0]['text']
    location = view['evidence_excerpts']['pdf_location_0']
    assert 'AUTHOR_CONTEXT' in location['quote'] and location['quote'] in page['text']
    assert view['code_locations'][0]['page'] == 1


def test_actual_pdf_and_git_excerpts_reach_llm_and_quotes_are_verified(packet):
    view = identity_view(packet)
    llm = Mock(mock_mode=False, model="fixture-model")
    llm.chat.return_value = json.dumps(identity_review(view))
    review = RepositoryVerifierAgent(llm).run(packet)
    prompt = llm.chat.call_args.args[0]
    assert view["evidence_excerpts"]["repo_0_0"]["quote"] in prompt.replace("\\n", "\n")
    assert view["repository"]["url"] in prompt and "a" * 40 in prompt
    assert "dataset.json" in prompt
    assert llm.chat.call_args.kwargs["task"] == "repository_identity_review"
    assert validate_identity_review(packet, review)["accepted"]
    assert review["model"] == "fixture-model"


@pytest.mark.parametrize("mutation", ["fake_quote", "fake_page", "fake_file", "missing_pdf",
                                      "failed_check", "unrelated", "fake_digest", "title_as_authorship", "fake_excerpt"])
def test_unverifiable_or_rejected_verdict_cannot_confirm_repository(packet, mutation):
    review = identity_review(identity_view(packet))
    check = review["checks"]["author_implementation"]
    if mutation == "fake_quote":
        check["citations"][1]["quote"] = "This official code belongs to a fictional author."
    elif mutation == "fake_page":
        check["citations"][0]["page"] = 199
    elif mutation == "fake_file":
        check["citations"][1]["path"] = "invented.md"
    elif mutation == "missing_pdf":
        check["citations"] = [check["citations"][1]] * 2
    elif mutation == "failed_check":
        check["passed"] = False
    elif mutation == "unrelated":
        review["verdict"] = "third_party"
    elif mutation == "fake_digest":
        review["evidence_sha256"] = "0" * 64
    elif mutation == "fake_excerpt":
        check["citations"][0] = "invented_pdf_excerpt"
    else:
        check["citations"][0].update(page=3, quote="Evaluation protocol and results")
    with pytest.raises(ValueError):
        validate_identity_review(packet, review)


@pytest.mark.parametrize("change", ["pdf", "commit", "readme", "candidate"])
def test_accepted_review_cannot_transfer_to_different_paper_or_repository(packet, change):
    review = validate_identity_review(packet, identity_review(identity_view(packet)))
    altered = deepcopy(packet)
    if change == "pdf":
        altered["pdf"]["sha256"] = "0" * 64
    elif change == "commit":
        altered["repository"]["revision"] = "b" * 40
    elif change == "readme":
        altered["repository"]["files"]["README.md"]["sha256"] = "0" * 64
    else:
        altered["repository"]["url"] = "https://github.com/unrelated/other"
    with pytest.raises(ValueError):
        validate_identity_review(altered, review)


def test_reference_only_link_and_missing_readme_are_not_author_evidence(packet):
    for link in packet["pdf"]["repository_links"]:
        link["evidence_type"] = "reference"
    with pytest.raises(ValueError, match="候选链接"):
        identity_view(packet)
    for link in packet["pdf"]["repository_links"]:
        link["evidence_type"] = "repository_link"
    packet["repository"]["files"].pop("README.md")
    with pytest.raises(ValueError, match="README"):
        identity_view(packet)


def test_mock_llm_and_invalid_response_cannot_confirm_identity(packet):
    with pytest.raises(ValueError, match="真实 LLM"):
        RepositoryVerifierAgent(Mock(mock_mode=True)).run(packet)
    llm = Mock(mock_mode=False)
    llm.chat.return_value = "[LLM API Error: timeout]"
    with pytest.raises(ValueError, match="有效 JSON"):
        RepositoryVerifierAgent(llm).run(packet)


def test_semantic_rejection_is_not_retried_as_quote_repair(packet):
    llm = Mock(mock_mode=False)
    llm.chat.return_value = json.dumps({"verdict": "uncertain", "reason": "Authorship is not established."})
    with pytest.raises(ValueError, match="未确认"):
        RepositoryVerifierAgent(llm).run(packet)
    assert llm.chat.call_count == 1


def test_bad_citation_gets_one_bounded_repair_then_still_requires_source_evidence(packet):
    llm = Mock(mock_mode=False, model="fixture-model")
    bad = identity_review(identity_view(packet))
    bad["checks"]["paper_identity"]["citations"][0]["quote"] = "An invented quotation cannot establish identity."
    valid = identity_review(identity_view(packet))
    llm.chat.side_effect = [json.dumps(bad), json.dumps(valid)]
    assert RepositoryVerifierAgent(llm).run(packet)["accepted"]
    assert llm.chat.call_count == 2
    llm.chat.side_effect = [json.dumps(bad), json.dumps(bad)]
    with pytest.raises(ValueError, match="引用不属于"):
        RepositoryVerifierAgent(llm).run(packet)
    assert llm.chat.call_count == 4


def test_service_rejects_first_candidate_and_verifies_actual_alternative(tmp_path, monkeypatch):
    import shutil
    from src import discovered_repository_reproduction as service
    from src import discovered_repository_plan as planner
    from src.audit.audit_logger import AuditLogger
    from discovered_fixtures import PlannedLLM
    author = make_experiment(tmp_path, "signals")
    other = make_experiment(tmp_path, "pairs")
    write_pdf(author["pdf"], [["Separating Signed Signals", "Abstract",
              "Our experiments are implemented here: " + author["snapshot"]["url"],
              "We compare with baseline source code at " + other["snapshot"]["url"]]])

    def export(root, profile, workspace, **kwargs):
        fixture = author if profile["repository"]["url"] == author["snapshot"]["url"] else other
        shutil.copytree(fixture["workspace"], workspace)
        return {**fixture["snapshot"], "resolved_sha": "a" * 40}

    llm = PlannedLLM(author["proposal"])
    chat = llm.chat

    def review(prompt, **kwargs):
        if kwargs["task"] == "repository_identity_review" and other["snapshot"]["url"] in json.loads(
                prompt.split("Fetched evidence:\n", 1)[1])["repository"]["url"]:
            llm.calls.append((prompt, kwargs))
            return json.dumps({"verdict": "third_party", "reason": "Different paper and baseline."})
        return chat(prompt, **kwargs)

    llm.chat = review
    monkeypatch.setattr(service, "export_repository", export)
    monkeypatch.setattr(service, "resolve_repository_revision", lambda *a, **kw: ("a" * 40, {}))
    propose = Mock(side_effect=ValueError("Fixture stops after verified selection"))
    monkeypatch.setattr(planner, "propose_plan", propose)
    from src.agents.experiment_protocol import ExperimentProtocolAgent
    monkeypatch.setattr(ExperimentProtocolAgent, 'run', Mock(return_value={'scope':'identity_fixture'}))
    runner = service.DiscoveredRepositoryReproduction(tmp_path / "service", AuditLogger(log_dir=str(tmp_path / "logs")), llm=llm)
    runner.runner.run = Mock(side_effect=AssertionError("Selection test cannot execute training"))
    result = runner.run({"pdf_path": str(author["pdf"]), "resources": {
        "code_repo_url": other["snapshot"]["url"],
        "repository_identity": {"status": "author_verified", "review": {"accepted": True}}}})
    data = result["data"]
    assert data["resources"]["code_repo_url"] == author["snapshot"]["url"]
    assert data["resources"]["repository_identity"]["status"] == "author_verified"
    assert [attempt["accepted"] for attempt in data["repository_identity_attempts"]] == [False, True]
    assert propose.call_args.args[1]["repository_identity_pending"] is False
    assert propose.call_args.args[1]["repository_identity_review"]["accepted"]
    runner.runner.run.assert_not_called()
