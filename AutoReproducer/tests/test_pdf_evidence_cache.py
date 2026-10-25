"""Run-local PDF reuse preserves current source identity and exact canonical text."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import sleep
from unittest.mock import Mock

import pytest

from src import agent_collaboration, pdf_input
from src.agent_collaboration import CollaborationSession, collaborative_run, current_collaboration


@pytest.fixture
def parser(monkeypatch, tmp_path):
    path = tmp_path / 'paper.pdf'
    path.write_bytes(b'%PDF-current source')
    primary = Mock(return_value=(['Paper title\nSignal 81.5'], 'Paper title', []))
    monkeypatch.setattr(pdf_input, '_extract_pypdf', primary)
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(side_effect=AssertionError('unexpected parser')))

    def convert(payload, pages):
        return tuple(pages), {'status': 'converted', 'pages': [{'page': 1, 'reason': '',
            'tables': [{'columns': ['Alpha', 'Beta']}]}], 'pdf_sha256': hashlib.sha256(payload).hexdigest()}
    conversion = Mock(side_effect=convert)
    monkeypatch.setattr(pdf_input, 'convert_pdf_pages', conversion)
    return path, primary, conversion


def test_same_bytes_across_roles_and_paths_are_parsed_once_with_detached_records(parser, tmp_path):
    path, primary, conversion = parser
    same = tmp_path / 'another.pdf'
    same.write_bytes(path.read_bytes())
    session = CollaborationSession()
    with session.scope():
        first = pdf_input.extract_pdf_input(path)
        first.extraction['pages'][0]['tables'][0]['columns'].append('injected')
        second = pdf_input.extract_pdf_input(same)
        assert second.extraction['pages'][0]['tables'][0]['columns'] == ['Alpha', 'Beta']
        second.extraction['pages'][0]['tables'][0]['columns'].clear()
        third = pdf_input.extract_pdf_input(path)
    assert third.extraction['pages'][0]['tables'][0]['columns'] == ['Alpha', 'Beta']
    assert second.sha256 == third.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert second.text == third.text and second.page_texts == third.page_texts
    assert type(third.page_texts) is tuple and type(third.repository_links) is tuple
    assert primary.call_count == conversion.call_count == 1
    stats = session.snapshot()['evidence_cache']
    assert stats['hits'] == 2 and stats['misses'] == 1 and stats['stores'] == 1
    assert 'Paper title' not in json.dumps(session.snapshot())


def test_changed_pdf_bytes_are_always_read_and_reparsed(parser):
    path, primary, conversion = parser
    with CollaborationSession().scope():
        original = pdf_input.extract_pdf_input(path)
        path.write_bytes(b'%PDF-replaced current source')
        changed = pdf_input.extract_pdf_input(path)
        path.unlink()
        with pytest.raises(pdf_input.PDFInputError, match='不存在'):
            pdf_input.extract_pdf_input(path)
    assert original.sha256 != changed.sha256
    assert primary.call_count == conversion.call_count == 2


def test_runs_and_unscoped_reads_never_share_the_cache(parser):
    path, primary, conversion = parser
    first, second = CollaborationSession(), CollaborationSession()
    with first.scope():
        pdf_input.extract_pdf_input(path)
        with second.scope():
            pdf_input.extract_pdf_input(path)
        pdf_input.extract_pdf_input(path)
    pdf_input.extract_pdf_input(path)
    pdf_input.extract_pdf_input(path)
    assert primary.call_count == conversion.call_count == 4
    assert first.snapshot()['evidence_cache']['hits'] == 1
    assert second.snapshot()['evidence_cache']['hits'] == 0


@pytest.mark.parametrize('change', ['parser', 'converter', 'dependency', 'source_generation'])
def test_parser_converter_dependency_or_source_identity_changes_cannot_reuse_old_evidence(
        parser, monkeypatch, change):
    path, primary, conversion = parser
    session = CollaborationSession()
    with session.scope():
        pdf_input.extract_pdf_input(path)
        if change == 'parser':
            replacement = Mock(return_value=(['Updated original page 82.5'], '', []))
            monkeypatch.setattr(pdf_input, '_extract_pypdf', replacement)
        elif change == 'converter':
            replacement = Mock(wraps=conversion)
            monkeypatch.setattr(pdf_input, 'convert_pdf_pages', replacement)
        elif change == 'dependency':
            import importlib.metadata
            original = importlib.metadata.version
            monkeypatch.setattr(importlib.metadata, 'version', lambda name:
                'changed-version' if name == 'markitdown' else original(name))
        else:
            monkeypatch.setattr(pdf_input, '_source_fingerprint', lambda: 'changed-source-generation')
        result = pdf_input.extract_pdf_input(path)
    if change == 'parser':
        assert '82.5' in result.text and replacement.call_count == 1
    else:
        assert primary.call_count == 2
    assert conversion.call_count == 2
    assert session.snapshot()['evidence_cache']['hits'] == 0


@pytest.mark.parametrize('change', ['converter', 'dependency', 'source_generation'])
def test_identity_changed_during_parsing_never_populates_the_cache(parser, monkeypatch, change):
    path, primary, conversion = parser
    original = conversion.side_effect
    changed = [False]
    if change == 'dependency':
        import importlib.metadata
        original_version = importlib.metadata.version
        monkeypatch.setattr(importlib.metadata, 'version', lambda name:
            'changed-version' if changed[0] and name == 'markitdown' else original_version(name))
    def convert(payload, pages):
        result = original(payload, pages)
        if change == 'converter':
            monkeypatch.setattr(pdf_input, 'convert_pdf_pages', Mock(side_effect=original))
        elif change == 'source_generation':
            monkeypatch.setattr(pdf_input, 'PDF_SOURCE_FINGERPRINT', 'changed-implementation')
        changed[0] = True
        return result
    conversion.side_effect = convert
    session = CollaborationSession()
    with session.scope():
        pdf_input.extract_pdf_input(path)
    assert primary.call_count == conversion.call_count == 1
    assert session.snapshot()['evidence_cache']['entries'] == 0
    assert session.snapshot()['evidence_cache']['rejected'] == 1


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(130)])
def test_cancelled_parse_is_not_cached_and_retry_can_complete(parser, error):
    path, primary, conversion = parser
    primary.side_effect = [error, (['Paper title\nSignal 81.5'], '', [])]
    session = CollaborationSession()
    with session.scope():
        with pytest.raises(type(error)):
            pdf_input.extract_pdf_input(path)
        assert session.snapshot()['evidence_cache']['entries'] == 0
        pdf_input.extract_pdf_input(path)
    assert primary.call_count == 2 and conversion.call_count == 1


def test_parse_errors_are_not_cached(parser, monkeypatch):
    path, primary, conversion = parser
    primary.side_effect = ValueError('invalid PDF')
    fallback = Mock(side_effect=ValueError('also invalid'))
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', fallback)
    with CollaborationSession().scope():
        for _ in range(2):
            with pytest.raises(pdf_input.PDFInputError):
                pdf_input.extract_pdf_input(path)
    assert primary.call_count == fallback.call_count == 2
    conversion.assert_not_called()


def test_temporary_parser_failure_followed_by_complete_fallback_can_retry_primary(parser, monkeypatch):
    path, primary, conversion = parser
    primary.side_effect = [ValueError('temporary'), (['Recovered original 82.5'], '', [])]
    monkeypatch.setattr(pdf_input, '_extract_pdfplumber', Mock(return_value=(['Fallback original 81.5'], '', [])))
    session = CollaborationSession()
    with session.scope():
        assert '81.5' in pdf_input.extract_pdf_input(path).text
        assert '82.5' in pdf_input.extract_pdf_input(path).text
    assert primary.call_count == conversion.call_count == 2
    assert session.snapshot()['evidence_cache']['rejected'] == 1


@pytest.mark.parametrize('reason', ['markitdown_dependency_unavailable', 'conversion_failed:ValueError',
                                  'page_serialization_failed', 'page_count_mismatch'])
def test_conversion_failures_are_not_cached(parser, reason):
    path, primary, conversion = parser
    conversion.side_effect = lambda payload, pages: (tuple(pages), {'status': 'fallback',
        'pages': [{'page': 1, 'reason': reason}]})
    session = CollaborationSession()
    with session.scope():
        pdf_input.extract_pdf_input(path)
        pdf_input.extract_pdf_input(path)
    assert primary.call_count == conversion.call_count == 2
    assert session.snapshot()['evidence_cache']['entries'] == 0


def test_cache_limits_evict_oldest_and_skip_oversized_records(parser, monkeypatch):
    path, primary, conversion = parser
    monkeypatch.setattr(agent_collaboration, 'MAX_EVIDENCE_CACHE_ENTRIES', 2)
    session = CollaborationSession()
    with session.scope():
        for content in (b'%PDF-one', b'%PDF-two', b'%PDF-three', b'%PDF-one'):
            path.write_bytes(content)
            pdf_input.extract_pdf_input(path)
        assert primary.call_count == 4
        stats = session.snapshot()['evidence_cache']
        assert stats['entries'] == 2 and stats['evictions'] == 2
        monkeypatch.setattr(agent_collaboration, 'MAX_EVIDENCE_CACHE_CHARS', 10)
        path.write_bytes(b'%PDF-too-big-to-cache')
        pdf_input.extract_pdf_input(path)
        assert session.snapshot()['evidence_cache']['rejected'] == 1


def test_cache_serialized_character_budget_remains_bounded_without_truncating_evidence(monkeypatch):
    monkeypatch.setattr(agent_collaboration, 'MAX_EVIDENCE_CACHE_ENTRIES', 3)
    monkeypatch.setattr(agent_collaboration, 'MAX_EVIDENCE_CACHE_CHARS', 35)
    session = CollaborationSession()
    for index in range(5):
        assert session.cache_evidence(hashlib.sha256(str(index).encode()).hexdigest(), {'text': str(index) * 10})
    stats = session.snapshot()['evidence_cache']
    assert stats['entries'] <= stats['max_entries'] and stats['serialized_chars'] <= stats['max_serialized_chars']
    assert session.cached_evidence(hashlib.sha256(b'4').hexdigest()) == {'text': '4' * 10}
    assert stats['evictions'] == 4


def test_cache_stats_persist_without_any_document_payload_or_identity(parser, tmp_path):
    path, _, _ = parser
    session = CollaborationSession()
    with session.scope():
        pdf_input.extract_pdf_input(path)
        pdf_input.extract_pdf_input(path)
        session.persist(tmp_path)
    record = json.loads((tmp_path / 'agent_collaboration.json').read_text('utf-8'))
    assert record['evidence_cache']['hits'] == 1
    encoded = json.dumps(record)
    assert 'Paper title' not in encoded and hashlib.sha256(path.read_bytes()).hexdigest() not in encoded


@pytest.mark.parametrize('text', ['sk-1234567890abcdefgh', 'api_key=private-value', 'Bearer private-value'])
def test_credential_bearing_document_is_not_retained_in_session_cache(parser, text):
    path, primary, conversion = parser
    primary.return_value = ([text], '', [])
    session = CollaborationSession(credential='private-value')
    with session.scope():
        assert text in pdf_input.extract_pdf_input(path).text
        pdf_input.extract_pdf_input(path)
    assert primary.call_count == conversion.call_count == 2
    assert session.snapshot()['evidence_cache']['entries'] == 0
    assert session.snapshot()['evidence_cache']['rejected'] == 2


def test_evidence_cache_never_evaluates_unknown_or_custom_container_objects():
    class UnsafeDictionary(dict):
        def items(self):
            raise AssertionError('Custom mapping evaluated')
    class UnsafeObject:
        def __repr__(self):
            raise AssertionError('Unknown object rendered')
    session = CollaborationSession()
    assert not session.cache_evidence('a' * 64, UnsafeDictionary(text='private'))
    assert not session.cache_evidence('b' * 64, {'object': UnsafeObject()})
    assert session.snapshot()['evidence_cache']['entries'] == 0


def test_concurrent_roles_parse_identical_evidence_once(parser):
    path, primary, conversion = parser
    original = conversion.side_effect
    def slow(payload, pages):
        sleep(.02)
        return original(payload, pages)
    conversion.side_effect = slow
    session = CollaborationSession()
    def read(_):
        with session.scope():
            return pdf_input.extract_pdf_input(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = list(pool.map(read, range(4)))
    assert all(document == documents[0] for document in documents)
    assert primary.call_count == conversion.call_count == 1
    assert session.snapshot()['evidence_cache']['hits'] == 3


def test_decorated_sequential_service_runs_have_distinct_cache_lifetimes(parser):
    path, primary, conversion = parser
    class Service:
        @collaborative_run
        def run(self):
            pdf_input.extract_pdf_input(path)
            pdf_input.extract_pdf_input(path)
            return {'data': {}}
    service = Service()
    first, second = service.run()['data'], service.run()['data']
    assert first['agent_collaboration']['run_id'] != second['agent_collaboration']['run_id']
    assert first['agent_collaboration']['evidence_cache']['hits'] == 1
    assert second['agent_collaboration']['evidence_cache']['hits'] == 1
    assert primary.call_count == conversion.call_count == 2


def test_older_live_host_mailbox_without_cache_api_keeps_original_parsing(parser, monkeypatch):
    path, primary, conversion = parser
    from types import SimpleNamespace
    monkeypatch.setattr(agent_collaboration, 'current_collaboration', lambda: SimpleNamespace(run_id='legacy'))
    pdf_input.extract_pdf_input(path)
    pdf_input.extract_pdf_input(path)
    assert primary.call_count == conversion.call_count == 2


def test_real_sgc_repeated_read_reuses_exact_page_evidence(monkeypatch):
    pytest.importorskip('markitdown')
    path = Path(__file__).resolve().parents[1] / 'examples/papers/pdf/sgc_1902.07153_holdout_01.pdf'
    conversion = Mock(wraps=pdf_input.convert_pdf_pages)
    monkeypatch.setattr(pdf_input, 'convert_pdf_pages', conversion)
    session = CollaborationSession()
    with session.scope():
        first = pdf_input.extract_pdf_input(path)
        second = pdf_input.extract_pdf_input(path)
    assert first == second and first is not second
    assert first.extraction is not second.extraction
    assert '| SGC | 81.0 ± 0.0 | 71.9 ± 0.1 | 78.9 ± 0.0 |' in second.page_texts[5]
    assert conversion.call_count == 1 and session.snapshot()['evidence_cache']['hits'] == 1
