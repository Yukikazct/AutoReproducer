"""Early failures use the same bounded ledger as nested author execution."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.intelligent_recovery import RecoverySession, current_investigation, investigating_run
from src.stage_recovery import run_stage, StageEvidenceError
from test_pdf_input_routing import write_pdf


@pytest.fixture
def paper(tmp_path):
    path = write_pdf(tmp_path / '中文论文.pdf', ['Unregistered Recovery Paper', 'Abstract',
        'Our implementation is at https://github.com/example/author'])
    return {'pdf_path': str(path), 'paper_title': 'Unregistered Recovery Paper'}


def reading(complete=True):
    return {'paper_info': {'title': 'Unregistered Recovery Paper'}, 'raw_text': 'original text',
            'reading_coverage': {'complete': complete, 'spans': [], 'total_batches': 1,
                                 'completed_batches': int(complete)}, 'llm_calls': 1}


def test_incomplete_reading_rechecks_original_pages_and_verifies_actual_output(paper):
    agent = SimpleNamespace(name='PaperReader', run=Mock(side_effect=[reading(False), reading()]))
    with RecoverySession(None).scope() as session:
        result = run_stage(agent, 'READ_PAPER', paper)
        assert result['reading_coverage']['complete'] is True
        assert result['llm_calls'] == 2
        assert session.state['usage']['repair_actions'] == 1
        feedback = agent.run.call_args.args[0]['retry_feedback']
        assert feedback['evidence']['pdf_sha256']
        assert feedback['evidence']['pages'][0]['covered'] is False
        assert session.state['actions'][-1]['verification']['success'] is True


def test_unsuccessful_reread_stops_and_retains_failed_verification(paper):
    agent = SimpleNamespace(name='PaperReader', run=Mock(return_value=reading(False)))
    with RecoverySession(None).scope() as session:
        with pytest.raises(StageEvidenceError):
            run_stage(agent, 'READ_PAPER', paper)
        assert agent.run.call_count == 2
        assert session.state['stop_reason'] == 'no_progress'
        assert session.state['actions'][-1]['verification']['success'] is False


def test_discovery_cannot_replace_explicit_user_repository(paper):
    paper['preferred_repo_url'] = 'https://github.com/example/original'
    output = {'resources': {'code_repo_url': 'https://github.com/example/different'}}
    agent = SimpleNamespace(name='ResourceFinder', run=Mock(return_value=output))
    with RecoverySession(None).scope() as session:
        with pytest.raises(StageEvidenceError):
            run_stage(agent, 'FIND_RESOURCES', paper)
        assert session.state['actions'][-1]['verification']['success'] is False
        assert agent.run.call_args.args[0]['preferred_repo_url'] == paper['preferred_repo_url']


def test_provider_failure_is_not_retried_as_document_repair(paper):
    agent = SimpleNamespace(name='PaperReader', run=Mock(side_effect=RuntimeError('论文阅读 API 调用失败：auth')))
    with RecoverySession(None).scope() as session:
        with pytest.raises(RuntimeError, match='auth'):
            run_stage(agent, 'READ_PAPER', paper)
        assert agent.run.call_count == 1
        assert not session.state['actions']


def test_cleanup_failure_blocks_any_new_stage_action(paper):
    error = RuntimeError('worker cleanup failed')
    error.process_cleanup_confirmed = False
    agent = SimpleNamespace(name='PaperReader', run=Mock(side_effect=error))
    with RecoverySession(None).scope() as session:
        with pytest.raises(RuntimeError, match='cleanup'):
            run_stage(agent, 'READ_PAPER', paper)
        assert agent.run.call_count == 1
        assert session.state['cleanup_confirmed'] is False


def test_disabled_recovery_preserves_api_behavior(paper):
    agent = SimpleNamespace(name='PaperReader', run=Mock(return_value=reading(False)))
    assert run_stage(agent, 'READ_PAPER', paper)['reading_coverage']['complete'] is False
    agent.run.assert_called_once()


def test_reading_exception_retains_completed_span_evidence(paper):
    from src.paper_reading import PaperReadingError
    coverage = {'complete': False, 'spans': [[0, 30]], 'completed_batches': 1, 'total_batches': 2}
    agent = SimpleNamespace(name='PaperReader', run=Mock(side_effect=[
        PaperReadingError('second batch incomplete', coverage), reading()]))
    with RecoverySession(None).scope():
        run_stage(agent, 'READ_PAPER', paper)
    assert agent.run.call_args.args[0]['retry_feedback']['evidence']['reading_coverage'] == coverage


def _orchestrator(tmp_path):
    from src.orchestrator import Orchestrator
    from src.llm.llm_client import LLMClient
    from src.resource_manager import ResourceManager
    from src.audit.audit_logger import AuditLogger
    return Orchestrator(llm_client=LLMClient(mock_mode=True), mock_mode=False,
        logger=AuditLogger(log_dir=str(tmp_path / 'logs')),
        resource_manager=ResourceManager(data_root=str(tmp_path / 'data')))


def test_orchestrator_and_nested_repository_share_preparation_budget(paper, tmp_path, monkeypatch):
    import src.discovered_repository_reproduction as service
    orch = _orchestrator(tmp_path)
    orch.agents['reader'].run = Mock(side_effect=[reading(False), reading()])
    orch.agents['finder'].run = Mock(return_value={
        'resources': {'code_repo_url': 'https://github.com/example/author'}})
    observed = []

    @investigating_run
    def nested(self, request, **kwargs):
        session = current_investigation()
        observed.append(session)
        assert session.state['usage']['repair_actions'] == 1
        result_data = {}
        session.attach(result_data, tmp_path / 'nested')
        session.run_action('replan', lambda: True, problem='entrypoint missing',
                           evidence={'file': 'train.py'}, verify=lambda value: {'success': value})
        return {'state': 'COMPLETED', 'data': result_data, 'error': None}

    monkeypatch.setattr(service.DiscoveredRepositoryReproduction, 'run', nested)
    result = orch.run({**paper, 'intelligent_recovery': True})
    assert result['state'] == 'COMPLETED', result['error']
    assert result['data']['intelligent_recovery'] is observed[0].state
    assert result['data']['intelligent_recovery']['usage']['repair_actions'] == 2
    assert current_investigation() is None


def test_early_failure_has_persisted_report_and_does_not_start_discovery(paper, tmp_path):
    orch = _orchestrator(tmp_path)
    orch.agents['reader'].run = Mock(return_value=reading(False))
    orch.agents['finder'].run = Mock(side_effect=AssertionError('cannot discover after failed reading'))
    result = orch.run({**paper, 'intelligent_recovery': True})
    assert result['state'] == 'ERROR'
    orch.agents['finder'].run.assert_not_called()
    from pathlib import Path
    root = Path(result['data']['run_dir'])
    assert (root / 'intelligent_recovery.json').is_file()
    assert (root / 'report.md').is_file()
    saved = json.loads((root / 'result.json').read_text('utf-8'))
    assert saved['data']['intelligent_recovery']['usage']['repair_actions'] == 1
    assert result['data']['report']


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit])
def test_cancelled_reader_persists_terminal_report_before_propagating(paper, tmp_path, interruption):
    orch = _orchestrator(tmp_path)
    orch.agents['reader'].run = Mock(side_effect=interruption('cancelled'))
    with pytest.raises(interruption):
        orch.run({**paper, 'intelligent_recovery': True})
    from pathlib import Path
    root = Path(orch.data['run_dir'])
    saved = json.loads((root / 'result.json').read_text('utf-8'))
    assert saved['state'] == 'ERROR'
    assert saved['data']['interrupted'] is True
    assert saved['data']['intelligent_recovery']['stop_reason'] == 'run_interrupted'
    assert saved['data']['environment_cleanup']['status'] == 'released'
    assert (root / 'report.md').is_file()


def test_original_pdf_change_after_initial_resolution_cannot_be_recovered(paper, tmp_path):
    from pathlib import Path
    orch = _orchestrator(tmp_path)
    def replace_pdf(data):
        assert data['pdf_input']['sha256']
        write_pdf(Path(paper['pdf_path']), ['A Different Paper', 'Different experiment'])
        raise ValueError('reader failure')
    orch.agents['reader'].run = Mock(side_effect=replace_pdf)
    result = orch.run({**paper, 'intelligent_recovery': True})
    assert result['state'] == 'ERROR'
    assert 'PDF changed' in result['error']
    orch.agents['reader'].run.assert_called_once()
