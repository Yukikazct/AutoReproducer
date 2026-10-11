"""Planning gates run before any execution and use complete original evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from discovered_fixtures import PlannedLLM, accepted_review, make_experiment
from src import discovered_repository_plan as planner


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # Only this test module isolates the runtime validator; service/runtime
    # integration tests use the real capture implementation and real scripts.
    monkeypatch.setattr(planner, "_validate_capture", lambda value, *_: deepcopy(value))
    monkeypatch.setattr(planner, "_capture_schema", lambda: "fixture capture schema")
    fixture = make_experiment(tmp_path)
    fixture["packet"] = planner.build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    return fixture


def validate(fixture, proposal=None):
    return planner.validate_plan(proposal or fixture["proposal"], fixture["packet"], fixture["workspace"])


@pytest.mark.parametrize("family", ["signals", "pairs"])
def test_two_unregistered_papers_keep_all_pages_and_independent_verifier(tmp_path, monkeypatch, family):
    monkeypatch.setattr(planner, "_validate_capture", lambda value, *_: deepcopy(value))
    monkeypatch.setattr(planner, "_capture_schema", lambda: "fixture capture schema")
    fixture = make_experiment(tmp_path, family)
    packet = planner.build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    assert [page["page"] for page in packet["pdf"]["pages"]] == [1, 2, 3]
    assert "100.0" in packet["pdf"]["pages"][-1]["text"]
    assert str(tmp_path) not in json.dumps(packet)
    assert packet["pdf"]["sha256"] == hashlib.sha256(fixture["pdf"].read_bytes()).hexdigest()
    llm = PlannedLLM(fixture["proposal"])
    result = planner.propose_plan(llm, packet, fixture["workspace"])
    assert [call[1]["task"] for call in llm.calls] == [
        "discovered_repository_plan", "discovered_repository_plan_review"]
    assert all(call[1]["temperature"] == 0 for call in llm.calls)
    assert planner._canonical(packet) in llm.calls[1][0]
    assert result["semantic_review"]["accepted"] is True
    assert result["semantic_review"]["reviewed_plan_sha256"] != result["plan_sha256"]
    frozen = deepcopy(result)
    expected = frozen.pop("plan_sha256")
    assert hashlib.sha256(planner._canonical(frozen).encode()).hexdigest() == expected
    assert result["requirements"]["author"] == ["torch==1.13.1"]
    assert result["requirements"]["runtime_required"] == ["torch==2.5.1+cpu", "numpy==1.26.4"]
    assert result["datasets"][0]["files_sha256"] == {
        "dataset.json": fixture["snapshot"]["files"]["dataset.json"]}


@pytest.mark.parametrize("change,match", [
    (lambda p: p["citations"][1].update(quote="A fabricated reference value is 100.0"), "verbatim"),
    (lambda p: p["citations"][1].update(page=1), "verbatim"),
    (lambda p: p["citations"][1].update(page=4), "missing PDF page"),
    (lambda p: p["metrics"][0].update(reference=97.123), "exact PDF citation"),
    (lambda p: p["metrics"][0].update(tolerance_relative=0.5), "tolerance"),
    (lambda p: p["metrics"][0].update(unit="fraction"), "unit"),
    (lambda p: p.update(training_code="print(1)"), "Generated implementation"),
    (lambda p: p["repository"].update(revision="b" * 40), "another repository"),
    (lambda p: p["requirements"].update(author=["numpy==1.26.4"]), "Original dependencies"),
    (lambda p: p["requirements"].update(author=["torch==1.13"]), "Original dependencies"),
    (lambda p: p["requirements"].update(compatibility=["torch==1.0", "numpy==1.26.4"]), "version map"),
    (lambda p: p["requirements"].update(compatibility=["torch==2.5.1+cpu"]), "torch and numpy"),
    (lambda p: p["requirements"]["compatibility"].append("scipy==1.14.1"), "lacks source/import"),
    (lambda p: p["steps"][0].update(argv=["python", "-c", "print(1)"]), "Generated code"),
    (lambda p: p["steps"][0].update(argv=["python", "-m", "train"]), "Generated code"),
    (lambda p: p["steps"][0].update(argv=["bash", "train.py"]), "structured Python"),
    (lambda p: p["steps"][0].update(env={"EPOCHS": "1"}), "environment"),
    (lambda p: p["steps"][0].update(required=False), "optionalize"),
    (lambda p: p["steps"][0].update(timeout_s=7201), "timeouts"),
    (lambda p: p["steps"][0]["argv"].extend(["--subset", "1"]), "shortened"),
    (lambda p: p["datasets"][0].update(paths=["*.json"]), "explicit"),
    (lambda p: p["datasets"][0].update(paths=["../dataset.json"]), "snapshot"),
    (lambda p: p["capture"].update(expected_test_samples=12345), "source-backed"),
])
def test_pretraining_deterministic_rejections(experiment, change, match):
    proposal = deepcopy(experiment["proposal"])
    change(proposal)
    with pytest.raises((planner.PlanEvidenceError, ValueError), match=match):
        validate(experiment, proposal)


def test_parameter_number_elsewhere_does_not_authorize_different_cli_value(experiment):
    proposal = deepcopy(experiment["proposal"])
    # 7 is in the seed documentation, but it is not the author's epoch count.
    proposal["steps"][0]["argv"][3] = "7"
    with pytest.raises(planner.PlanEvidenceError, match="CLI defaults"):
        validate(experiment, proposal)


@pytest.mark.parametrize("target", ["train.py", "dataset.json"])
def test_source_or_dataset_mutation_fails_closed(experiment, target):
    path = experiment["workspace"] / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(planner.PlanEvidenceError, match="changed"):
        validate(experiment)


def test_pdf_page_mutation_fails_even_if_quote_still_present(experiment):
    experiment["packet"]["pdf"]["pages"][2]["text"] += " edited"
    with pytest.raises(planner.PlanEvidenceError, match="PDF page evidence changed"):
        validate(experiment)


@pytest.mark.parametrize("failure", ["wrong_row", "train_split", "shortened_epochs", "wrong_flag_value"])
def test_independent_semantic_rejection_is_final_before_execution(experiment, failure):
    proposal = deepcopy(experiment["proposal"])
    if failure == "wrong_row":
        # 98.0 exists, but belongs to a different method and split.
        proposal["metrics"][0]["reference"] = 98.0
        gate = "same_experiment"
    elif failure == "train_split":
        proposal["capture"]["test_indices"] = None
        gate = "capture_uses_test_data"
    elif failure == "shortened_epochs":
        proposal["parameters"]["epochs"] = 7
        gate = "complete_author_protocol"
    else:
        proposal["parameters"]["seed"] = 40
        gate = "complete_author_protocol"
    # Lexically all these values occur in real source. Only a contextual
    # protocol review can establish the cross-source relationships.
    validate(experiment, proposal)
    review = accepted_review()
    review["accepted"] = False
    review["checks"][gate] = {"passed": False, "reason": failure, "citations": ["script", "paper_results"]}
    llm = PlannedLLM(proposal, review)
    with pytest.raises(planner.PlanEvidenceError, match="semantic review"):
        planner.propose_plan(llm, experiment["packet"], experiment["workspace"])
    assert len(llm.calls) == 2


@pytest.mark.parametrize("change", [
    lambda v: v.update(accepted="true"),
    lambda v: v["checks"].pop("same_experiment"),
    lambda v: v["checks"]["same_experiment"].update(passed=False),
    lambda v: v["checks"]["same_experiment"].update(reason=""),
    lambda v: v["checks"]["same_experiment"].update(citations=["script"]),
    lambda v: v["checks"]["same_experiment"].update(citations=["invented"]),
])
def test_incomplete_or_uncertain_semantic_review_cannot_accept(experiment, change):
    verdict = accepted_review()
    change(verdict)
    llm = PlannedLLM(experiment["proposal"], verdict)
    with pytest.raises(planner.PlanEvidenceError):
        planner.propose_plan(llm, experiment["packet"], experiment["workspace"])
    assert len(llm.calls) == 2


def test_quote_repairs_are_bounded_and_review_happens_only_after_validation(experiment):
    invalid = deepcopy(experiment["proposal"])
    invalid["citations"][1]["quote"] = "This quote was hallucinated."
    llm = PlannedLLM(experiment["proposal"], prefixes=[invalid])
    result = planner.propose_plan(llm, experiment["packet"], experiment["workspace"])
    assert len(llm.calls) == 3 and result["semantic_review"]["accepted"]
    assert "not verbatim" in llm.calls[1][0]
    invalid_llm = PlannedLLM(invalid)
    with pytest.raises(planner.PlanEvidenceError, match="No validated"):
        planner.propose_plan(invalid_llm, experiment["packet"], experiment["workspace"])
    assert len(invalid_llm.calls) == 2
    assert all(call[1]["task"] == "discovered_repository_plan" for call in invalid_llm.calls)


def test_quote_whitespace_is_reconciled_to_exact_original_without_editing_proposal(experiment):
    proposal = deepcopy(experiment["proposal"])
    citation = proposal["citations"][1]
    original = citation["quote"]
    citation["quote"] = "\n\t".join(original.split())
    accepted = validate(experiment, proposal)
    evidence = next(item for item in accepted["citations"] if item["id"] == citation["id"])
    assert evidence["quote"] == original
    assert evidence["proposed_quote"] == citation["quote"]
    assert citation["quote"] != original


def test_quote_whitespace_reconciliation_cannot_change_reference_number(experiment):
    proposal = deepcopy(experiment["proposal"])
    proposal["citations"][1]["quote"] = "\n".join(
        proposal["citations"][1]["quote"].replace("100.0", "100.1").split())
    with pytest.raises(planner.PlanEvidenceError, match="not verbatim"):
        validate(experiment, proposal)


def test_quote_feedback_identifies_all_invalid_citations_in_one_attempt(experiment):
    proposal = deepcopy(experiment["proposal"])
    invalid = proposal["citations"][:2]
    for item in invalid:
        item["quote"] = "Fabricated quote with different numbers 123.456."
    with pytest.raises(planner.PlanEvidenceError, match="not verbatim") as failure:
        validate(experiment, proposal)
    assert all(item["id"] in str(failure.value) for item in invalid)


def test_source_request_is_bounded_and_added_before_evidence_is_frozen(experiment):
    packet = experiment["packet"]
    del packet["repository"]["files"]["train.py"]
    llm = PlannedLLM(experiment["proposal"], prefixes=[{"request_files": ["train.py"]}])
    accepted = planner.propose_plan(llm, packet, experiment["workspace"])
    assert "train.py" in packet["repository"]["files"]
    assert accepted["evidence_sha256"] == hashlib.sha256(planner._canonical(packet).encode()).hexdigest()
    assert len(llm.calls) == 3
    excessive = PlannedLLM(experiment["proposal"], prefixes=[{"request_files": ["train.py"]}] * 4)
    with pytest.raises(planner.PlanEvidenceError, match="discovery rounds are exhausted"):
        planner.propose_plan(excessive, packet, experiment["workspace"])
    assert len(excessive.calls) == 4


def test_unidentified_external_dataset_is_not_downloaded(experiment):
    proposal = deepcopy(experiment["proposal"])
    proposal["datasets"] = [{"kind": "https", "url": "https://example.com/dataset.bin",
                             "target": "download.bin", "sha256": "a" * 64, "bytes": 100,
                             "citations": ["readme"]}]
    with pytest.raises(planner.PlanEvidenceError, match="cited canonical HTTPS"):
        validate(experiment, proposal)


def test_binary_text_suffix_is_skipped_without_hiding_real_source_mutation(experiment):
    fixture = experiment
    binary = b"\xff\x00pickle-like data"
    (fixture["workspace"] / "data.txt").write_bytes(binary)
    fixture["snapshot"]["files"]["data.txt"] = hashlib.sha256(binary).hexdigest()
    packet = planner.build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    assert not packet["file_inventory"]["data.txt"]["text_available"]
    assert "data.txt" not in packet["repository"]["files"]
    (fixture["workspace"] / "train.py").write_text("changed", encoding="utf-8")
    with pytest.raises(planner.PlanEvidenceError, match="changed after snapshot"):
        planner.build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
