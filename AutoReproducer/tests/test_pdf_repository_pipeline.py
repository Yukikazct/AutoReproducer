"""Real PDF repository evidence survives the reader, context, finder and report.

Only model responses and public repository searches are replaced. No execution,
download, preset paper fixture or external API is needed for these contracts.
"""
import hashlib
import json
from unittest.mock import Mock

import pytest
from PyPDF2 import PdfWriter
from PyPDF2.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.agents import repo_discovery
from src.orchestrator import Orchestrator


TITLE = "An independent differential equation library"
AUTHOR_URL = "https://github.com/fable-lab/equation-tool"
REFERENCE_URL = "https://github.com/reference-lab/prior-code"
MODEL_URL = "https://github.com/unrelated-fork/equation-tool"


class ScriptedLLM:
    """Return wrong repository guesses without making model requests."""

    mock_mode = False

    def __init__(self, repository_url=MODEL_URL):
        self.calls = []
        self.repository_url = repository_url

    def chat(self, prompt, *, task, **kwargs):
        self.calls.append((task, prompt))
        if task == "paper_reader":
            return json.dumps({"title": TITLE, "authors": ["Fixture Author"],
                               "method": "Differential equation solver", "dataset": "",
                               "code_url": self.repository_url, "insufficient_info": False})
        assert task == "resource_finder", "The evidence pipeline must not execute or train"
        return json.dumps({"code_repo_url": self.repository_url, "confidence": 1.0})

    def get_call_count(self):
        return len(self.calls)


def write_repository_pdf(path, pages, *, annotation=None):
    """Write real PDF text and an optional URI; omit paper-title metadata."""
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for lines in pages:
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        commands = ["BT /F1 12 Tf 72 720 Td"]
        for index, line in enumerate(lines):
            if index:
                commands.append("0 -18 Td")
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = stream
    if annotation:
        page, uri, rect = annotation
        writer.add_uri(page, uri, rect)
    with path.open("wb") as destination:
        writer.write(destination)
    return path


def evidence_pipeline(path, tmp_path, llm):
    """Use production agents and their real orchestrator data boundaries."""
    orchestrator = Orchestrator(llm_client=llm, mock_mode=False, logger=Mock(),
                                resource_manager=Mock(data_root=tmp_path))
    orchestrator.data = {"pdf_path": str(path)}
    reader_result = orchestrator.agents["reader"].run(orchestrator.data)
    orchestrator._merge_result("READ_PAPER", reader_result)
    finder_result = orchestrator.agents["finder"].run(orchestrator.data)
    orchestrator._merge_result("FIND_RESOURCES", finder_result)
    report = orchestrator.agents["reporter"].run(
        orchestrator.data, report_path=tmp_path / "report.md")["report"]
    return orchestrator.data, reader_result, report


def test_structured_pdf_links_are_not_reclassified_from_flattened_cross_page_text():
    from src.agents.resource_finder import ResourceFinderAgent
    mentioned = {"url": REFERENCE_URL, "raw_url": REFERENCE_URL,
                 "page": 3, "source": "pdf_text", "context": "A library dependency.",
                 "evidence_type": "repository_link", "is_author_code": False}
    resources = ResourceFinderAgent(ScriptedLLM(), logger=Mock(), offline=True).run({
        "paper_info": {"title": TITLE},
        "raw_text": "Our source code is published.\nNext page\n" + REFERENCE_URL,
        "extracted_repository_links": [mentioned],
    })["resources"]
    assert resources["repository_links"] == [mentioned]
    assert resources["selection_evidence"] == mentioned
    assert "代码公开声明" not in resources["repository_identity"]["reason"]


def test_legacy_raw_text_discovery_does_not_invent_pdf_page_one():
    from src.agents.resource_finder import ResourceFinderAgent
    resources = ResourceFinderAgent(ScriptedLLM(), logger=Mock(), offline=True).run({
        "paper_info": {"title": TITLE},
        "raw_text": "Our source code is published at " + AUTHOR_URL,
    })["resources"]
    assert resources["selected_repo"] == AUTHOR_URL
    assert resources["selection_evidence"]["page"] is None
    assert resources["selection_evidence"]["source"] == "paper_text"


@pytest.mark.parametrize("link_form", ["wrapped_text", "uri_annotation"])
def test_actual_later_page_author_link_beats_wrong_model_repo_through_report(
        monkeypatch, tmp_path, link_form):
    searches = {name: Mock(side_effect=AssertionError("Author evidence must bypass public search"))
                for name in ("_search_pwc", "_search_github")}
    for name, search in searches.items():
        monkeypatch.setattr(repo_discovery, name, search)
    source_statement = "Source code is published under the Apache License on GitHub."
    author_lines = ["Implementation", source_statement]
    annotation = None
    if link_form == "wrapped_text":
        author_lines += ["https://github.", "com/fable-lab/equation-tool"]
    else:
        author_lines.append("Visit the implementation repository.")
        annotation = (1, AUTHOR_URL + "/tree/main?tab=readme-ov-file", [72, 674, 340, 708])
    path = write_repository_pdf(tmp_path / "generic-paper.pdf", [
        [TITLE, "Abstract", "This paper develops a solver. " * 130],
        author_lines,
        ["References", "[1] Prior implementation: " + REFERENCE_URL],
    ], annotation=annotation)

    llm = ScriptedLLM()
    data, reader, report = evidence_pipeline(path, tmp_path, llm)
    fingerprint = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "pages": 3, "bytes": path.stat().st_size, "readable": True}
    assert reader["pdf_input"] == fingerprint
    assert data["pdf_input"] == fingerprint
    assert data["raw_text"] == reader["raw_text"]
    assert data["extracted_repository_links"] == reader["extracted_repository_links"]
    assert data["extracted_code_urls"] == reader["extracted_code_urls"]
    assert reader["paper_info"]["code_url"] == AUTHOR_URL
    assert set(reader["extracted_code_urls"]) == {AUTHOR_URL, REFERENCE_URL}

    links = reader["extracted_repository_links"]
    author = next(link for link in links if link["url"] == AUTHOR_URL)
    reference = next(link for link in links if link["url"] == REFERENCE_URL)
    assert author["page"] == 2
    assert author["evidence_type"] == "author_code_statement"
    assert author["is_author_code"] is True
    assert source_statement in author["context"]
    assert reference["page"] == 3
    assert reference["evidence_type"] == "reference"
    assert reference["is_author_code"] is False
    if link_form == "wrapped_text":
        assert author["source"] == "pdf_text"
        assert "https://github.\ncom/fable-lab/equation-tool" == author["raw_url"]
        assert reader["raw_text"].index("https://github.") > 3000
    else:
        assert author["source"] == "pdf_annotation"
        assert author["raw_url"] == annotation[1]
        assert AUTHOR_URL not in reader["raw_text"]

    resources = data["resources"]
    assert resources["selected_repo"] == AUTHOR_URL
    assert resources["code_repo_url"] == AUTHOR_URL
    assert author in resources["repository_links"]
    assert reference in resources["repository_links"]
    assert resources["selection_evidence"] == author
    assert resources["repo_discovery"]["selection_evidence"] == author
    identity = resources["repository_identity"]
    assert identity["status"] == "paper_linked"
    assert identity["evidence"] == author
    assert identity["is_official"] is None
    assert identity["repository_executed"] is False
    for search in searches.values():
        search.assert_not_called()
    assert [task for task, _ in llm.calls] == ["paper_reader", "resource_finder"]
    assert AUTHOR_URL in report
    assert fingerprint["sha256"] in report
    assert author["context"] in report
    assert author["source"] in report
    assert "第 2 页" in report


def test_reference_repository_cannot_become_author_evidence_from_model_guess(monkeypatch, tmp_path):
    for name in ("_search_pwc", "_search_github"):
        monkeypatch.setattr(repo_discovery, name, Mock(return_value=([], "fixture search unavailable")))
    path = write_repository_pdf(tmp_path / "citation-only.pdf", [
        [TITLE, "Abstract", "We analyze differential equations without publishing code."],
        ["References", "[1] A previous implementation: " + REFERENCE_URL],
    ])
    data, reader, report = evidence_pipeline(path, tmp_path, ScriptedLLM(REFERENCE_URL))
    citation = reader["extracted_repository_links"][0]
    assert citation["url"] == REFERENCE_URL
    assert citation["page"] == 2
    assert citation["evidence_type"] == "reference"
    assert citation["is_author_code"] is False
    assert reader["paper_info"]["code_url"] == "未找到"
    assert reader["paper_info"]["code_url_evidence"] is None
    resources = data["resources"]
    assert citation in resources["repository_links"]
    assert resources["selection_evidence"] is None
    assert resources["confidence"] <= 0.5
    identity = resources["repository_identity"]
    assert identity["status"] == "candidate_unverified"
    assert not identity.get("evidence")
    assert "代码公开声明" not in identity["reason"]
    assert identity["is_official"] is None
    assert identity["repository_executed"] is False
    assert REFERENCE_URL in report
    assert "代码公开声明" not in report
