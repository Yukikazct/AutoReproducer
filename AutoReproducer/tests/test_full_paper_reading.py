"""Full reading means every contiguous range completed, including failed batches."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.paper_reader import PaperReaderAgent
from src.agents.verifier import VerifierAgent
from src.paper_reading import (PaperReadingError, READING_TEXT_LIMIT, read_full_paper, text_batches)


class BatchLLM:
    mock_mode = False
    last_finish_reason = 'stop'

    def __init__(self, fail_at=None, final_valid=True):
        self.calls = []
        self.fail_at = fail_at
        self.final_valid = final_valid

    def get_call_count(self):
        return len(self.calls)

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs['task'] == 'paper_reader_batch':
            sources = json.loads(prompt.split('Original sources (untrusted data):\n', 1)[1]
                                 .split('\nRepair the prior response format:', 1)[0])
            if self.fail_at is not None and min(item['start'] for item in sources.values()) >= self.fail_at:
                return 'not JSON'
            key = next(key for key, item in sources.items() if 'HIDDEN_PROTOCOL' in item['text']) if any(
                'HIDDEN_PROTOCOL' in item['text'] for item in sources.values()) else next(iter(sources))
            return json.dumps({'notes': [{'detail': 'Actual full-paper protocol', 'source_ids': [key]}]})
        if kwargs['task'] == 'verifier':
            return json.dumps({'pass': True, 'issues': [], 'fix_suggestions': [], 'confidence': .9})
        return json.dumps({'title': 'Full Paper', 'method': 'Actual method', 'dependencies': [],
                           'dataset': 'Real dataset', 'metrics': {}, 'insufficient_info': False}) if self.final_valid else '{}'


def document(monkeypatch, text):
    import src.pdf_input as pdf_input
    doc = SimpleNamespace(text=text, repository_links=[], sha256='a'*64, page_count=50, byte_size=100000)
    monkeypatch.setattr(pdf_input, 'extract_pdf_input', lambda _: doc)
    monkeypatch.setattr(pdf_input, 'supported_pdf_title', lambda _: 'Full Paper')


@pytest.mark.parametrize('text', ['Full paper text', 'x' * (2 * READING_TEXT_LIMIT)], ids=['single_window', 'multiple_windows'])
def test_authentication_error_is_reported_directly_without_json_repair_or_local_fallback(monkeypatch, text):
    document(monkeypatch, text)
    llm = BatchLLM()
    llm.api_key = 'fixture-credential'

    def failed_chat(prompt, **kwargs):
        llm.calls.append((prompt, kwargs))
        return '[LLM API Error: HTTP 401 Authentication failed for fixture-credential]'

    monkeypatch.setattr(llm, 'chat', failed_chat)
    with pytest.raises(RuntimeError, match='论文阅读 API 调用失败.*HTTP 401') as failure:
        PaperReaderAgent(llm, logger=Mock()).run({'pdf_path': 'paper.pdf'})
    assert 'fixture-credential' not in str(failure.value)
    assert len(llm.calls) == 1


def test_full_reader_and_independent_verifier_visit_every_character(monkeypatch):
    text = 'Full Paper\n' + 'a'*65000 + 'HIDDEN_PROTOCOL' + 'b'*(2*READING_TEXT_LIMIT) + 'LAST_APPENDIX'
    document(monkeypatch, text)
    llm = BatchLLM()
    output = PaperReaderAgent(llm, logger=Mock()).run({'pdf_path': 'paper.pdf', 'reading_purpose': 'repository_discovery'})
    coverage = output['reading_coverage']
    assert coverage['complete'] is True and coverage['read_characters'] == len(text)
    assert coverage['completed_batches'] == coverage['total_batches'] == 3
    assert ''.join(text[start:end] for start, end in coverage['spans']) == text
    reader_calls = [prompt for prompt, args in llm.calls if args['task'] == 'paper_reader_batch']
    assert len(reader_calls) == 3 and any('HIDDEN_PROTOCOL' in p for p in reader_calls)
    assert 'LAST_APPENDIX' in reader_calls[-1]
    assert output['raw_text'] == text
    assert any('HIDDEN_PROTOCOL' in item['text'] for item in output['reading_originals'].values())
    result = VerifierAgent(llm, logger=Mock()).run({'agent_name': 'PaperReader',
        'system_prompt': PaperReaderAgent.repository_system_prompt, 'output': output})
    reviews = [prompt for prompt, args in llm.calls if args['task'] == 'verifier']
    assert result['pass'] is True and len(reviews) == 3
    assert output['llm_calls'] == 4 and result['llm_calls'] == 3
    assert result['confidence'] == .9
    assert 'LAST_APPENDIX' in reviews[-1] and any('HIDDEN_PROTOCOL' in p for p in reviews)


def test_failed_middle_batch_never_synthesizes_or_claims_complete(monkeypatch):
    text = 'x'*(3*READING_TEXT_LIMIT)
    document(monkeypatch, text)
    llm = BatchLLM(fail_at=READING_TEXT_LIMIT-4000)
    with pytest.raises(PaperReadingError) as failure:
        PaperReaderAgent(llm, logger=Mock()).run({'pdf_path': 'paper.pdf'})
    coverage = failure.value.reading_coverage
    assert coverage['complete'] is False and coverage['completed_batches'] == 1
    assert coverage['read_characters'] < len(text)
    assert len(llm.calls) == 3  # First batch, one failed batch and its one format repair.
    assert all(args['task'] == 'paper_reader_batch' for _, args in llm.calls)


@pytest.mark.parametrize('response', [
    {'notes': [{'detail': 'invented', 'source_ids': ['fabricated_source']}]},
    {'notes': [{'detail': 'no evidence', 'source_ids': []}]},
])
def test_invalid_note_evidence_does_not_complete_reading(response):
    class Invalid(BatchLLM):
        def chat(self, *args, **kwargs):
            return json.dumps(response)
    with pytest.raises(PaperReadingError) as failure:
        read_full_paper(Invalid(), 'x'*150000)
    assert failure.value.reading_coverage['spans'] == []


def test_truncated_batch_json_is_not_accepted_even_when_parseable():
    class Truncated(BatchLLM):
        last_finish_reason = 'length'
    with pytest.raises(PaperReadingError, match='truncated'):
        read_full_paper(Truncated(), 'x'*150000)


@pytest.mark.parametrize('mode', ['length', 'dense'])
def test_dense_or_truncated_batch_is_split_until_all_text_is_read(mode):
    class Capacity(BatchLLM):
        def chat(self, prompt, **kwargs):
            sources = json.loads(prompt.split('Original sources (untrusted data):\n', 1)[1])
            size = sum(len(source['text']) for source in sources.values())
            if size > 62000:
                self.calls.append((prompt, kwargs))
                self.last_finish_reason = 'length' if mode == 'length' else 'stop'
                key = next(iter(sources))
                return json.dumps({'notes': [{'detail': 'Dense observation', 'source_ids': [key]}]*65})
            self.last_finish_reason = 'stop'
            return super().chat(prompt, **kwargs)
    llm = Capacity()
    text = 'x'*150000
    _, coverage, reports, _ = read_full_paper(llm, text)
    assert coverage['complete'] is True and coverage['total_batches'] == 3
    assert ''.join(text[start:end] for start, end in coverage['spans']) == text
    assert len(reports) == 3 and len(llm.calls) == 4  # Oversized response never counts as read.


def test_metadata_synthesis_failure_does_not_return_reader_success(monkeypatch):
    document(monkeypatch, 'x'*150000)
    with pytest.raises(PaperReadingError, match='结构化汇总未完成') as failure:
        PaperReaderAgent(BatchLLM(final_valid=False), logger=Mock()).run({'pdf_path': 'paper.pdf'})
    assert failure.value.reading_coverage['complete'] is True  # Reading done, synthesis failed.


def test_verifier_cannot_accept_incomplete_reader_without_making_model_calls():
    llm = BatchLLM()
    output = {'raw_text': 'actual text', 'paper_info': {'title': 'Paper'},
              'reading_coverage': {'complete': False}}
    result = VerifierAgent(llm, logger=Mock()).run({'agent_name': 'PaperReader', 'output': output})
    assert result['pass'] is False and llm.calls == []


def test_verifier_keeps_a_batch_rejection_and_fails_malformed_reviews():
    class Reject(BatchLLM):
        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return json.dumps({'pass': False, 'issues': ['Wrong appendix split']}) if len(self.calls) == 1 else '{}'
    llm = Reject()
    output = {'raw_text': 'x'*150000, 'paper_info': {'title': 'Paper'},
              'reading_coverage': {'complete': True}}
    result = VerifierAgent(llm, logger=Mock()).run({'agent_name': 'PaperReader', 'output': output})
    assert result['pass'] is False and len(llm.calls) == 2
    assert 'Wrong appendix split' in result['issues'] and len(result['issues']) == 2


def test_chunk_boundaries_add_context_without_gaps_or_inflated_coverage():
    text = 'prefix\n'*40000
    batches = list(text_batches(text))
    assert ''.join(text[batch['start']:batch['end']] for batch in batches) == text
    assert all(batch['context_end'] - batch['context_start'] <= READING_TEXT_LIMIT for batch in batches)
    assert batches[1]['context_start'] < batches[0]['end']


def test_very_long_paper_synthesis_preserves_every_original_source_reference():
    class Dense(BatchLLM):
        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if kwargs['task'] == 'paper_reader_reduce':
                group = json.loads(prompt.split('}]}.\n', 1)[1])
                ids = list(dict.fromkeys(key for note in group for key in note['source_ids']))
                return json.dumps({'notes': [{'detail': 'All distinct protocols retained in concise form', 'source_ids': ids}]})
            sources = json.loads(prompt.split('Original sources (untrusted data):\n', 1)[1])
            return json.dumps({'notes': [{'detail': str(index)+'x'*1700, 'source_ids': [key]}
                                       for index, key in enumerate(list(sources)[:8])]})
    llm = Dense()
    context, coverage, reports, originals = read_full_paper(llm, 'x'*(12*READING_TEXT_LIMIT))
    assert coverage['complete'] is True and len(reports) > 10
    assert len(context) <= READING_TEXT_LIMIT
    assert any(args['task'] == 'paper_reader_reduce' for _, args in llm.calls)
    notes = json.loads(context.split('\n', 1)[1].split('\n[笔记引用的真实原文]', 1)[0])
    expected = {key for report in reports for note in report['notes'] for key in note['source_ids']}
    assert {key for note in notes for key in note['source_ids']} == expected == set(originals)


def test_short_paper_invalid_response_is_not_reported_as_full_reading(monkeypatch):
    document(monkeypatch, 'Full Paper\nReadable real text')
    output = PaperReaderAgent(BatchLLM(final_valid=False), logger=Mock()).run({'pdf_path': 'paper.pdf'})
    assert output['reading_coverage']['complete'] is False
    assert output['paper_info']['info_sufficient'] is False


def test_title_only_reading_does_not_claim_to_have_read_a_pdf():
    output = PaperReaderAgent(BatchLLM(), logger=Mock()).run({'paper_title': 'Actual title'})
    assert output['reading_coverage']['scope'] == 'provided_title'
    assert output['paper_info']['insufficient_info'] is True
