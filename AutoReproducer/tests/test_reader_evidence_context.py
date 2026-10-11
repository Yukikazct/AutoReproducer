"""Reader and review must see experimental evidence outside the first page."""
import json
from types import SimpleNamespace

from src.agents.paper_reader import (PaperReaderAgent, READING_TEXT_LIMIT, reading_context,
                                    repository_reading_context)
from src.agents.verifier import VerifierAgent


class RecordingLLM:
    mock_mode = False

    def __init__(self):
        self.calls = []

    def get_call_count(self):
        return len(self.calls)

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps({"pass": True} if kwargs['task'] == 'verifier' else
                          {"title": "Evidence Paper", "method": "Actual method", "dependencies": [],
                           "metrics": {"accuracy": 89.4}, "dataset": "LateDataset", "insufficient_info": False})


def test_reader_and_verifier_receive_late_experiments_and_preserve_library_semantics(monkeypatch):
    import src.pdf_input as pdf_input
    text = "Evidence Paper\n" + "Introduction. " * 400 + "\nTest accuracy 89.4% on LateDataset."
    doc = SimpleNamespace(text=text, repository_links=[], sha256='a' * 64, page_count=5, byte_size=9000)
    monkeypatch.setattr(pdf_input, 'extract_pdf_input', lambda _: doc)
    monkeypatch.setattr(pdf_input, 'supported_pdf_title', lambda _: 'Evidence Paper')
    llm = RecordingLLM()
    output = PaperReaderAgent(llm).run({'pdf_path': 'actual.pdf'})
    assert 'Test accuracy 89.4% on LateDataset.' in llm.calls[0][0]
    assert output['reading_coverage']['complete'] is True
    result = VerifierAgent(llm).run({'agent_name': 'PaperReader',
        'system_prompt': PaperReaderAgent.system_prompt, 'output': output})
    assert result['pass'] is True
    assert 'Test accuracy 89.4% on LateDataset.' in llm.calls[1][0]
    assert '不包括基线模型' in llm.calls[1][0]
    assert '不要因显示形式为 Python 字典' in llm.calls[1][0]
    assert '空列表是正确结果' in llm.calls[1][0]


def test_long_paper_excerpts_include_head_middle_tail_with_explicit_coverage():
    text = 'TITLE ' + 'x' * READING_TEXT_LIMIT + 'RESULT_TABLE'
    context, coverage = reading_context(text)
    assert coverage['complete'] is False
    assert coverage['total_characters'] == len(text)
    assert context.startswith('[原文字符 0:')
    assert 'TITLE ' in context and context.endswith('RESULT_TABLE')
    assert sum(end-start for start, end in coverage['spans']) == READING_TEXT_LIMIT
    assert len(context) < READING_TEXT_LIMIT + 2000


def test_retry_feedback_is_used_without_replacing_original_evidence():
    llm = RecordingLLM()
    PaperReaderAgent(llm).run({'paper_title': 'Actual title', 'retry_feedback': {
        'agent': 'PaperReader', 'issues': ['Keep unknown software libraries empty']}})
    assert 'Keep unknown software libraries empty' in llm.calls[0][0]
    assert '无证据的建议不能执行' in llm.calls[0][0]


def test_repository_discovery_targets_late_code_paragraph_and_keeps_page_evidence():
    url = 'https://github.com/authors/example'
    text = 'TITLE\n' + 'Early prose. ' * 1800 + '\nOur source code is available at ' + url + '\nFOOTNOTE_CONTEXT'
    links = [{'url': url, 'page': 7, 'context': 'Author code declaration', 'source': 'pdf_text'}]
    context, coverage = repository_reading_context(text, links)
    assert url in context and 'FOOTNOTE_CONTEXT' in context
    assert 'Author code declaration' in context and '"page": 7' in context
    assert coverage['purpose'] == 'repository_discovery' and coverage['repository_locations'] == 1


def test_repository_reader_does_not_require_experimental_metrics(monkeypatch):
    import src.pdf_input as pdf_input
    text = 'Paper Title\nOur source code is available at https://github.com/authors/example'
    doc = SimpleNamespace(text=text, repository_links=[], sha256='a' * 64, page_count=1, byte_size=9000)
    monkeypatch.setattr(pdf_input, 'extract_pdf_input', lambda _: doc)
    monkeypatch.setattr(pdf_input, 'supported_pdf_title', lambda _: 'Paper Title')
    llm = RecordingLLM()
    output = PaperReaderAgent(llm).run({'pdf_path': 'actual.pdf', 'reading_purpose': 'repository_discovery'})
    assert output['reading_purpose'] == 'repository_discovery'
    assert '当前阶段仅定位论文仓库' in llm.calls[0][0]
    VerifierAgent(llm).run({'agent_name': 'PaperReader', 'output': output,
                          'system_prompt': PaperReaderAgent.repository_system_prompt})
    assert '本阶段不要求数据集' in llm.calls[1][0]
