"""Real PDF bytes must select only their visible paper identity, without a model."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PyPDF2 import PdfWriter
from PyPDF2.generic import DecodedStreamObject, DictionaryObject, NameObject

import src.pdf_input as pdf_input
import src.repository_reproduction as repository
from src.agents.paper_reader import PaperReaderAgent
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator


TITLE = "Neural Ordinary Differential Equations"


def write_pdf(path, lines=(), *, metadata_title=None, encrypted=False, later_pages=()):
    """Tiny real PDF fixture with independent header, metadata and body text."""
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for content in (lines, *later_pages):
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        commands = ["BT /F1 12 Tf 72 720 Td"]
        for index, line in enumerate(content):
            if index:
                commands.append("0 -18 Td")
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if metadata_title is not None:
        writer.add_metadata({"/Title": metadata_title})
    if encrypted:
        writer.encrypt("fixture-password")
    with path.open("wb") as destination:
        writer.write(destination)
    return path


@pytest.mark.parametrize("lines,profile,scope", [
    ([TITLE, "Ricky T. Q. Chen", "Abstract", "We introduce neural ODEs."],
     "neural_ode_spiral", "official_method_experiment"),
    (["arXiv:1806.07366", "Neural Ordinary", "Differential Equations", "Authors", "Abstract"],
     "neural_ode_spiral", "official_method_experiment"),
    (["Are Transformers Effective for Time Series Forecasting?", "Authors", "Abstract"],
     "dlinear_etth1_reference", "selected_paper_experiment"),
    (["Implicit Neural Representations with", "Periodic Activation Functions", "Authors", "Abstract"],
     "siren_camera_quick", "official_method_experiment"),
])
def test_visible_canonical_pdf_title_selects_fixed_profile_with_actual_byte_provenance(tmp_path, lines, profile, scope):
    path = write_pdf(tmp_path / "unrelated-filename.pdf", lines)
    request = {"pdf_path": str(path), "paper_title": "unrelated title textbox state",
               "use_llm_review": False, "allow_result_summary_review": False,
               "optimization_mode": "off"}
    result = pdf_input.resolve_pdf_request(request)
    assert result["experiment_profile"] == profile
    resolution = result["pdf_resolution"]
    assert resolution["source"] == "exact_pdf_header"
    assert resolution["profile"] == profile and resolution["scope"] == scope
    assert resolution["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolution["bytes"] == path.stat().st_size and resolution["pages"] == 1
    assert all(result[key] == value for key, value in request.items())
    assert "experiment_profile" not in request and "pdf_resolution" not in request


@pytest.mark.parametrize("lines,metadata,later", [
    (["An unrelated research paper", "Authors", "Abstract", TITLE], TITLE, ()),
    (["An unrelated research paper", "Authors", "References", TITLE], None, ()),
    (["An unrelated research paper", TITLE, "Authors", "Abstract"], TITLE, ()),
    (["Abstract", TITLE], TITLE, ()),
    (["An unrelated research paper", "Authors", "Abstract"], None, ([TITLE, "Abstract"],)),
    (["Neural Ordinary Differential Equations for Images", "Abstract"], None, ()),
    ([TITLE, "Authors", "Abstract"], "Are Transformers Effective for Time Series Forecasting?", ()),
])
def test_metadata_mentions_references_later_pages_and_similar_titles_cannot_route(tmp_path, lines, metadata, later):
    path = write_pdf(tmp_path / "mention.pdf", lines, metadata_title=metadata, later_pages=later)
    request = {"pdf_path": str(path)}
    result = pdf_input.resolve_pdf_request(request)
    assert result == request and result is not request


@pytest.mark.parametrize("explicit", [
    {"experiment_profile": "dlinear_etth1_smoke"}, {"mock_mode": True},
    {"code": "print('chosen code')"}, {"preferred_repo_url": "https://github.com/example/chosen"},
    {"code_repo_url": "https://github.com/example/chosen"}, {"corpus_paper": "chosen-paper"},
])
def test_valid_pdf_respects_explicit_inputs_and_mock_stays_offline(tmp_path, explicit):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract"])
    request = {"pdf_path": str(path), **explicit}
    result = pdf_input.resolve_pdf_request(request)
    assert result == request and result is not request


@pytest.mark.parametrize("kind", ["missing", "not_pdf", "corrupt", "blank", "encrypted"])
@pytest.mark.parametrize("mock_mode", [False, True])
def test_invalid_or_unreadable_pdf_rejected_before_any_llm_or_code(tmp_path, kind, mock_mode):
    path = tmp_path / "input.pdf"
    if kind == "not_pdf":
        path.write_text(TITLE, encoding="utf-8")
    elif kind == "corrupt":
        path.write_bytes(b"%PDF-1.7\nnot a document")
    elif kind == "blank":
        write_pdf(path)
    elif kind == "encrypted":
        write_pdf(path, [TITLE], encrypted=True)
    with pytest.raises(pdf_input.PDFInputError):
        pdf_input.resolve_pdf_request({"pdf_path": str(path), "mock_mode": mock_mode,
                                       "paper_title": TITLE})
    llm = LLMClient(mock_mode=mock_mode)
    llm.chat = Mock(side_effect=AssertionError("Unreadable PDF must not reach a model"))
    reader = PaperReaderAgent(llm, logger=Mock())
    with pytest.raises(pdf_input.PDFInputError):
        reader.run({"pdf_path": str(path), "paper_title": TITLE})
    llm.chat.assert_not_called()


def test_empty_primary_parser_uses_fallback_instead_of_returning_blank_text(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE])
    monkeypatch.setattr(pdf_input, "_extract_pypdf", Mock(return_value=([""], TITLE)))
    fallback = Mock(return_value=([TITLE + "\nAbstract\nReadable content."], TITLE))
    monkeypatch.setattr(pdf_input, "_extract_pdfplumber", fallback)
    document = pdf_input.extract_pdf_input(path)
    assert document.text.startswith(TITLE) and document.page_count == 1
    fallback.assert_called_once_with(path.read_bytes())


def test_undecoded_math_font_uses_complete_secondary_text_and_annotations(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'paper.pdf', [TITLE])
    broken = ['Sample paper', 'Table 1\nModel 73:25\x062:50', 'Final appendix']
    clean = ['Sample paper', 'Table 1\nModel 73.25±2.50', 'Final appendix']
    link = {'page': 1, 'raw_url': 'https://github.com/example/paper',
            'context': 'Our code is available at this link.'}
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(broken, '', [])))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(return_value=(clean, '', [link])))
    document = pdf_input.extract_pdf_input(path)
    assert document.text == '\n'.join(clean) and document.page_count == 3
    assert document.page_texts == tuple(clean)
    assert document.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert document.repository_links[0]['url'] == link['raw_url']


def test_secondary_text_keeps_primary_annotation_context_from_same_bytes(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'paper.pdf', [TITLE])
    link = {'page': 1, 'raw_url': 'https://github.com/example/paper',
            'context': 'Our code is available at this link.'}
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(['Title\x06'], '', [link])))
    interleaved = {**link, 'context': 'Our code is other column text available at this link.'}
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(return_value=(['Title'], '', [interleaved])))
    document = pdf_input.extract_pdf_input(path)
    assert document.text == 'Title'
    assert any(item['is_author_code'] and item['context'] == link['context']
               for item in document.repository_links)


@pytest.mark.parametrize('secondary', [
    ['Sample paper', 'Table 1 Model 73.25±2.50'],
    ['Sample paper', 'Table 1 Model 73.25±2.50', ''],
    ['Sample paper', 'Table 1 Model 73:25\x062:50\x07', 'Final appendix'],
])
def test_glyph_recovery_cannot_drop_readable_pages_or_select_more_damage(monkeypatch, tmp_path, secondary):
    path = write_pdf(tmp_path / 'paper.pdf', [TITLE])
    primary = ['Sample paper', 'Table 1 Model 73:25\x062:50', 'Final appendix']
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(primary, '')))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(return_value=(secondary, '')))
    assert pdf_input.extract_pdf_input(path).text == '\n'.join(primary)


def test_missing_secondary_parser_retains_readable_primary_with_glyph_damage(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'paper.pdf', [TITLE])
    primary = ['Sample paper\nModel 73:25\x062:50']
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(primary, '')))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(side_effect=ImportError('missing')))
    assert pdf_input.extract_pdf_input(path).text == primary[0]


def test_clean_ratios_and_times_are_never_rewritten_as_decimals(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'paper.pdf', [TITLE])
    primary = ['Sample paper\nRatio 3:2; time 14:30; accuracy 73.25']
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(primary, '')))
    fallback = Mock(side_effect=AssertionError('Clean text needs no recovery'))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', fallback)
    assert pdf_input.extract_pdf_input(path).text == primary[0]
    fallback.assert_not_called()


@pytest.mark.parametrize('broken,clean', [
    ('Method 61.27+/-0.1472.31+/-0.36', 'Method 61.27+/-0.14 72.31+/-0.36'),
    ('Estimator 0 .731 0.820.64', 'Estimator 0.731 0.82 0.64'),
])
def test_fragmented_numeric_columns_retry_real_geometry_extraction(monkeypatch, tmp_path, broken, clean):
    path = write_pdf(tmp_path / 'table.pdf', ['Unrelated table study'],
                     later_pages=([clean], ['Complete appendix']))
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(
        ['Unrelated table study', broken, 'Complete appendix'], '')))
    document = pdf_input.extract_pdf_input(path)
    assert document.page_texts == ('Unrelated table study', clean, 'Complete appendix')
    assert document.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_two_column_fallback_keeps_sentences_and_table_rows(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'two-columns.pdf', ['Two column paper'])
    from PyPDF2 import PdfReader
    writer = PdfWriter()
    writer.add_page(PdfReader(path).pages[0])
    stream = DecodedStreamObject()
    stream.set_data(b'\n'.join([
        b'BT /F1 10 Tf 45 720 Td (Independent left column.) Tj',
        b'0 -14 Td (This is unrelated training prose.) Tj',
        b'0 -14 Td (It must remain in its own paragraph.) Tj ET',
        b'BT /F1 10 Tf 320 720 Td (We report mean test accuracy) Tj',
        b'0 -14 Td (with standard deviation over 7 runs.) Tj',
        b'0 -28 Td (Method Dataset-A Dataset-B) Tj',
        b'0 -14 Td (Ours 85.5 73.4) Tj ET',
    ]))
    writer.pages[0][NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as output:
        writer.write(output)
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(
        ['Unreadable primary font\x06'], '')))
    document = pdf_input.extract_pdf_input(path)
    assert 'We report mean test accuracy\nwith standard deviation over 7 runs.' in document.text
    assert 'Method Dataset-A Dataset-B\nOurs 85.5 73.4' in document.text
    assert 'Independent left column.\nThis is unrelated training prose.' in document.text
    assert document.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(document.page_texts) == 1


def test_rotated_table_fallback_retains_geometric_character_order(tmp_path):
    import pdfplumber
    from PyPDF2 import PdfReader
    path = write_pdf(tmp_path / 'rotated-table.pdf', ['Rotated table'])
    writer = PdfWriter()
    writer.add_page(PdfReader(path).pages[0])
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 10 Tf 0 1 -1 0 400 100 Tm '
                    b'(Method A 85.5 73.4) Tj 0 -14 Td (Method B 79.0 81.3) Tj ET')
    writer.pages[0][NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as output:
        writer.write(output)
    with pdfplumber.open(path) as document:
        page = document.pages[0]
        expected = page.extract_text(x_tolerance=1)
        assert expected != page.extract_text(x_tolerance=1, use_text_flow=True)
    pages, _, _ = pdf_input._extract_pdfplumber(path.read_bytes())
    assert pages == [expected]


def test_numeric_spacing_recovery_cannot_introduce_undecoded_glyphs(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'table.pdf', ['Unrelated table study'])
    primary = ['Method 61.27+/-0.1472.31+/-0.36']
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(primary, '')))
    alternate = Mock(return_value=(['Method 61.27\x060.14 72.31\x060.36'], ''))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', alternate)
    assert pdf_input.extract_pdf_input(path).text == primary[0]
    alternate.assert_called_once()


def test_table_recovery_keeps_math_pages_from_the_better_original_parser(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / 'table.pdf', ['Unrelated table study'])
    primary = ['Method 61.27+/-0.1472.31+/-0.36', 'The equation is x squared.']
    alternate = ['Method 61.27+/-0.14 72.31+/-0.36', 'The equation is x\x07.']
    monkeypatch.setattr(pdf_input, '_extract_pypdf', Mock(return_value=(primary, '')))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(return_value=(alternate, '')))
    document = pdf_input.extract_pdf_input(path)
    assert document.page_texts == (alternate[0], primary[1])


def test_missing_parsers_distinguished_from_damaged_pdf(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE])
    for name in ("_extract_pypdf", "_extract_pdfplumber"):
        monkeypatch.setattr(pdf_input, name, Mock(side_effect=ImportError("missing parser")))
    with pytest.raises(pdf_input.PDFParserUnavailable):
        pdf_input.extract_pdf_input(path)


@pytest.mark.parametrize("primary", [Mock(return_value=([""], "")),
                                     Mock(side_effect=ValueError("primary could not parse"))])
def test_missing_fallback_parser_allows_environment_repair_before_unreadable_verdict(monkeypatch, tmp_path, primary):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE])
    monkeypatch.setattr(pdf_input, "_extract_pypdf", primary)
    monkeypatch.setattr(pdf_input, "_extract_pdfplumber", Mock(side_effect=ImportError("fallback missing")))
    with pytest.raises(pdf_input.PDFParserUnavailable):
        pdf_input.extract_pdf_input(path)


def test_managed_request_with_matching_explicit_profile_rebuilds_pdf_provenance(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract"])
    request = {"pdf_path": str(path), "experiment_profile": "neural_ode_spiral",
               "pdf_resolution": {"sha256": "caller supplied wrong hash"}}
    result = pdf_input.resolve_pdf_request(request)
    assert result["experiment_profile"] == request["experiment_profile"]
    assert result["pdf_resolution"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert request["pdf_resolution"] == {"sha256": "caller supplied wrong hash"}


def test_paper_reader_uses_visible_pdf_title_over_model_or_stale_title_and_keeps_fingerprint(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", "Readable method content."])
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(return_value=json.dumps({"title": "Hallucinated title", "method": "ODE",
                                            "dataset": "spiral", "insufficient_info": False}))
    result = PaperReaderAgent(llm, logger=Mock()).run({"pdf_path": str(path), "paper_title": "stale title"})
    assert result["paper_info"]["title"] == TITLE
    assert "Readable method content." in llm.chat.call_args.args[0]
    assert result["pdf_input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["pdf_input"]["readable"] is True


def test_real_title_only_model_cannot_upgrade_unverified_inference_to_complete_paper_info():
    llm = LLMClient(mock_mode=False)
    llm.chat = Mock(return_value=json.dumps({"title": "An unknown paper",
                                            "method": "invented method", "dataset": "invented dataset",
                                            "insufficient_info": False}))
    result = PaperReaderAgent(llm, logger=Mock()).run({"paper_title": "An unknown paper"})
    assert result["paper_info"]["insufficient_info"] is True
    assert result["paper_info"]["info_sufficient"] is False


def test_pdf_later_page_resource_evidence_is_not_lost_by_reader_viewport_limits(tmp_path):
    author_url = "https://github.com/example/paper-author-code"
    path = write_pdf(tmp_path / "paper.pdf", ["An unknown paper", "Abstract", "x" * 3100],
                     later_pages=(["Implementation", author_url],))
    llm = LLMClient(mock_mode=True)
    result = PaperReaderAgent(llm, logger=Mock()).run({"pdf_path": str(path)})
    assert author_url in result["raw_text"]
    assert author_url in result["extracted_code_urls"]


def test_direct_orchestrator_routes_visible_pdf_to_author_profile_without_generic_agents(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", "Readable method content."])
    llm = LLMClient(mock_mode=False)
    llm.chat = Mock(side_effect=AssertionError("No generic model calls"))
    logger = Mock()
    logger.get_summary.return_value = []
    logger.get_stats.return_value = {}
    orchestrator = Orchestrator(llm_client=llm, mock_mode=False, logger=logger,
                                resource_manager=Mock(data_root=tmp_path))
    for agent in orchestrator.agents.values():
        agent.run = Mock(side_effect=AssertionError("No generic agents"))
    reviewed = Mock()
    reviewed.run.return_value = {"state": "COMPLETED", "error": None, "data": {}}
    monkeypatch.setattr(repository, "RepositoryReproduction", Mock(return_value=reviewed))
    result = orchestrator.run({"pdf_path": str(path), "use_llm_review": False})
    request = reviewed.run.call_args.args[0]
    assert request["experiment_profile"] == "neural_ode_spiral"
    assert result["data"]["pdf_resolution"] == request["pdf_resolution"]
    assert request["pdf_resolution"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    llm.chat.assert_not_called()
    for agent in orchestrator.agents.values():
        agent.run.assert_not_called()


@pytest.mark.parametrize("mock_mode", [False, True])
def test_direct_orchestrator_rejects_missing_pdf_before_generic_or_reviewed_routes(tmp_path, monkeypatch, mock_mode):
    llm = LLMClient(mock_mode=mock_mode)
    llm.chat = Mock(side_effect=AssertionError("No model calls on rejected upload"))
    logger = Mock()
    logger.get_summary.return_value = []
    logger.get_stats.return_value = {}
    orchestrator = Orchestrator(llm_client=llm, mock_mode=mock_mode, logger=logger,
                                resource_manager=Mock(data_root=tmp_path))
    for agent in orchestrator.agents.values():
        agent.run = Mock(side_effect=AssertionError("No agents on rejected upload"))
    reviewed = Mock(side_effect=AssertionError("No reviewed route on rejected upload"))
    monkeypatch.setattr(repository, "RepositoryReproduction", reviewed)
    events = []
    result = orchestrator.run({"pdf_path": str(tmp_path / "missing.pdf"), "paper_title": TITLE}, on_event=events.append)
    assert result["state"] == "ERROR" and result["data"]["pdf_input"]["readable"] is False
    assert any(event.get("outcome") == "invalid_pdf" for event in events)
    assert "execution" not in result["data"]
    llm.chat.assert_not_called()
    reviewed.assert_not_called()
    for agent in orchestrator.agents.values():
        agent.run.assert_not_called()
