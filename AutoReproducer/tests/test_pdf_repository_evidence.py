"""PDF repository claims must be grounded in complete, readable document bytes."""
import hashlib
import json
import re
from unittest.mock import Mock

import pytest
from PyPDF2 import PdfReader, PdfWriter

from src.agents.paper_reader import PaperReaderAgent
from src.llm.llm_client import LLMClient
from src.pdf_input import PDFInputError, extract_pdf_input
from src.repository_evidence import extract_repository_links, normalize_repository_url, is_author_code_statement
from test_pdf_input_routing import write_pdf


@pytest.mark.parametrize("visible_url", [
    "github.com/researchers/method.",
    "https://github.\ncom/researchers/method",
    "https://github.\ncom/researchers/\nmethod.git/tree/main/examples).",
    "https://github.com/researchers/long-\nmethod/tree/main",
])
def test_bare_wrapped_and_subpath_urls_retain_document_provenance(visible_url):
    records = extract_repository_links(["Title\nAbstract", "Our source code is published at " + visible_url])
    assert len(records) == 1
    evidence = records[0]
    expected_repo = "long-method" if "long-" in visible_url else "method"
    assert evidence["url"] == "https://github.com/researchers/" + expected_repo
    assert evidence["raw_url"] == visible_url.split(")", 1)[0]
    assert evidence["page"] == 2 and evidence["source"] == "pdf_text"
    assert evidence["is_author_code"] and evidence["evidence_type"] == "author_code_statement"
    assert "Our source code is published" in evidence["context"]


def test_author_statement_outranks_earlier_links_and_reference_claims_remain_references():
    records = extract_repository_links([
        "Other packages include github.com/dependency/tool.\n"
        "Our implementation is available at github.com/authors/method.\n"
        "References\n[1] Source code is available at github.com/cited/work.",
        "[2] Software is available at github.com/cited/another.",
    ])
    assert records[0]["url"] == "https://github.com/authors/method"
    assert records[0]["is_author_code"] is True
    by_url = {record["url"]: record for record in records}
    assert by_url["https://github.com/dependency/tool"]["evidence_type"] == "repository_link"
    for repository in ("work", "another"):
        evidence = by_url["https://github.com/cited/" + repository]
        assert evidence["evidence_type"] == "reference" and not evidence["is_author_code"]


def test_url_at_line_end_does_not_absorb_the_next_sentence():
    records = extract_repository_links(["github.com/team/method\nNext sentence discusses results."])
    assert records[0]["url"] == "https://github.com/team/method"
    assert records[0]["raw_url"] == "github.com/team/method"


@pytest.mark.parametrize("declaration", [
    "Code is avail-\nable at:", "Code is avail- able at:",
    "Our implemen-\ntation is publicly avail-\nable at:",
    "Our soft-\nware is re-\nleased at:", "Code is avail\u00adable at:",
])
def test_hyphenated_author_declarations_keep_original_context_and_repository_hyphen(declaration):
    raw_url = "https://github.com/cure-lab/LTSF-\nLinear"
    page = declaration + raw_url
    evidence = extract_repository_links([page])[0]
    assert evidence["url"] == "https://github.com/cure-lab/LTSF-Linear"
    assert evidence["raw_url"] == raw_url
    assert evidence["context"] == re.sub(r"\s+", " ", page)
    assert evidence["is_author_code"] and evidence["evidence_type"] == "author_code_statement"
    # Downstream routing receives the preserved, whitespace-collapsed snippet.
    assert is_author_code_statement(evidence["context"])


@pytest.mark.parametrize("prefix", ["References\n[1] ", "Our base-\nline ", "Third-\nparty "])
def test_hyphenated_reference_and_external_declarations_remain_non_author(prefix):
    evidence = extract_repository_links([
        prefix + "code is avail-\nable at https://github.com/cited/LTSF-\nLinear"])[0]
    assert evidence["is_author_code"] is False
    assert evidence["evidence_type"] == ("reference" if prefix.startswith("References") else "repository_link")


def test_dlinear_style_binary_pdf_hyphenation_retains_bytes_page_and_raw_link(tmp_path):
    path = write_pdf(tmp_path / "hyphenated.pdf", [
        "Are Transformers Effective for Time Series Forecasting?", "Abstract",
        "Code is avail-", "able at:https://github.com/cure-lab/LTSF-", "Linear"])
    original = path.read_bytes()
    document = extract_pdf_input(path)
    evidence = next(record for record in document.repository_links if record["source"] == "pdf_text")
    assert evidence["url"] == "https://github.com/cure-lab/LTSF-Linear"
    assert evidence["raw_url"] == "https://github.com/cure-lab/LTSF-\nLinear"
    assert "Code is avail- able at:" in evidence["context"]
    assert evidence["is_author_code"] and evidence["page"] == 1
    assert document.sha256 == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


@pytest.mark.parametrize("prefix,visible_url", [
    ("A soft\u00adhyphen before the statement. ", "https://github.com/authors/method"),
    ("A zero\u200bwidth marker before the statement. ", "https://github.com/authors/method"),
    ("", "https://git\u00adhub.com/authors/method"),
    ("", "https://github.com/auth\u200bors/meth\u00adod"),
    ("Hidden\u00ad characters\u200b before the statement. ",
     "https://github.\ncom/auth\u00adors/meth\u200bod"),
])
def test_hidden_characters_preserve_original_url_offsets_context_and_page(prefix, visible_url):
    page = prefix + "Our source code is published at " + visible_url
    evidence = extract_repository_links(["Title\nAbstract", page])[0]
    assert evidence["url"] == "https://github.com/authors/method"
    assert evidence["raw_url"] == visible_url
    assert evidence["context"] == re.sub(r"\s+", " ", page)
    assert evidence["page"] == 2 and evidence["source"] == "pdf_text"
    assert evidence["evidence_type"] == "author_code_statement" and evidence["is_author_code"]


@pytest.mark.parametrize("value", [
    "https://example.org/github.com/team/repo", "https://github.com/topics/science",
    "https://github.com/team", "https://name:password@github.com/team/repo",
])
def test_non_repository_and_credential_urls_cannot_become_repository_roots(value):
    assert normalize_repository_url(value) == ""


def add_uri(path, page_number, url, rect=(72, 680, 400, 705)):
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_uri(page_number, url, rect)
    writer.add_uri(page_number, url, rect)  # A wrapped visible link can have two rectangles.
    with path.open("wb") as destination:
        writer.write(destination)


def test_annotation_only_url_uses_nearby_author_statement_and_preserves_original_hash(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", ["Unknown paper", "Our source code is available here."])
    add_uri(path, 0, "https://github.com/authors/annotation-only/tree/main", rect=(72, 695, 400, 708))
    original = path.read_bytes()
    document = extract_pdf_input(path)
    assert len(document.repository_links) == 1
    evidence = document.repository_links[0]
    assert evidence["url"] == "https://github.com/authors/annotation-only"
    assert evidence["source"] == "pdf_annotation" and evidence["page"] == 1
    assert evidence["is_author_code"] is True
    assert "Our source code is available here." in evidence["context"]
    assert document.sha256 == hashlib.sha256(original).hexdigest()
    assert document.byte_size == len(original) and path.read_bytes() == original


def test_annotation_on_unreadable_pdf_does_not_replace_required_visible_text(tmp_path):
    path = write_pdf(tmp_path / "blank.pdf", [])
    add_uri(path, 0, "https://github.com/authors/code")
    with pytest.raises(PDFInputError):
        extract_pdf_input(path)


@pytest.mark.parametrize("details", [[], ["[1] Cited work.", "Citation details.", "Further citation details."]])
def test_annotation_only_reference_cannot_inherit_author_code_claim_from_cited_work(tmp_path, details):
    lines = ["Unknown paper", "References", *details, "Cited source code is published here."]
    path = write_pdf(tmp_path / "references.pdf", lines)
    baseline = 720 - 18 * (len(lines) - 1)
    add_uri(path, 0, "https://github.com/cited/annotation-only", rect=(72, baseline - 10, 400, baseline + 3))
    evidence = extract_pdf_input(path).repository_links[0]
    assert evidence["source"] == "pdf_annotation"
    assert evidence["evidence_type"] == "reference" and not evidence["is_author_code"]


def test_reader_grounds_code_in_later_page_author_evidence_even_when_model_invents_url(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", ["Unknown paper", "Abstract", "x" * 3100],
                     later_pages=(["Our source code is published at https://github.",
                                   "com/authors/method.git/tree/main."],))
    original = path.read_bytes()
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(return_value=json.dumps({"title": "Unknown paper", "code_url": "https://github.com/guess/wrong"}))
    result = PaperReaderAgent(llm, logger=Mock()).run({"pdf_path": str(path)})
    assert result["paper_info"]["code_url"] == "https://github.com/authors/method"
    evidence = result["paper_info"]["code_url_evidence"]
    assert evidence["page"] == 2 and evidence["is_author_code"]
    assert evidence in result["extracted_repository_links"]
    assert result["extracted_code_urls"] == ["https://github.com/authors/method"]
    assert "Our source code is published" in llm.chat.call_args.args[0]
    assert result["pdf_input"]["sha256"] == hashlib.sha256(original).hexdigest()


def test_reader_keeps_reference_links_as_candidates_and_discards_unsupported_model_code_url(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", ["Unknown paper", "Abstract", "Readable body.",
                                              "References", "[1] Source code is available at github.com/cited/work."])
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(return_value=json.dumps({"title": "Unknown paper", "code_url": "https://github.com/guess/wrong"}))
    result = PaperReaderAgent(llm, logger=Mock()).run({"pdf_path": str(path)})
    assert result["paper_info"]["code_url"] == "未找到"
    assert result["paper_info"]["code_url_evidence"] is None
    assert result["extracted_code_urls"] == ["https://github.com/cited/work"]
    assert result["extracted_repository_links"][0]["evidence_type"] == "reference"


def test_numbered_code_footnote_uses_canonical_annotation_despite_glued_column_text():
    url = "https://github.com/authors/new-method"
    page = "The code is available on Github1.\n" + "Body text. " * 100 + "\n1" + url + "viandvj."
    records = extract_repository_links(["Title", page], [{"page": 2, "raw_url": url,
                                                         "context": "Unrelated figure. 1" + url}])
    evidence = next(item for item in records if item["source"] == "pdf_annotation")
    assert evidence["url"] == evidence["raw_url"] == url
    assert evidence["page"] == 2 and evidence["is_author_code"]
    assert "The code is available on Github1" in evidence["context"]
    assert "1" + url in evidence["context"]


@pytest.mark.parametrize("declaration,footnotes", [
    ("The code is available on Github2.", "1https://github.com/authors/new-method"),
    ("Third-party code is available on Github1.", "1https://github.com/authors/new-method"),
    ("Baseline code is available on Github1.", "1https://github.com/authors/new-method"),
    ("References\nCode is available on Github1.", "1https://github.com/authors/new-method"),
    ("Code is available on Github1.", "1https://github.com/authors/new-method\n1https://github.com/other/tool"),
    ("Code is available on Github1. Our code is available on Github1.", "1https://github.com/authors/new-method"),
])
def test_numbered_annotation_needs_unambiguous_non_reference_author_declaration(declaration, footnotes):
    annotations = [{"page": 1, "raw_url": url, "context": "Unrelated figure."} for url in
                   ("https://github.com/authors/new-method", "https://github.com/other/tool")]
    page = declaration + "\n" + "Body text. " * 100 + "\n" + footnotes
    evidence = [item for item in extract_repository_links([page], annotations)
                if item["source"] == "pdf_annotation"]
    assert all(not item["is_author_code"] for item in evidence)


def test_numbered_annotation_cannot_link_a_declaration_on_another_page():
    url = "https://github.com/authors/new-method"
    records = extract_repository_links(["Code is available on Github1.", "1" + url],
                                       [{"page": 2, "raw_url": url}])
    assert all(not record["is_author_code"] for record in records)
