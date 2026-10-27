"""Production PDF route, including a real native CPU author command when configured."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.intelligent_recovery import current_investigation
from test_discovered_repository_service import (setup_service, cpu_dependency_cache,
                                                use_preinstalled_cache)


def enable_model(llm):
    original = llm.chat
    def chat(prompt, **kwargs):
        if kwargs['task'].startswith('recovery_'):
            llm.calls.append((prompt, kwargs))
            context = json.loads(prompt)
            if kwargs['task'] in {'recovery_counter_review', 'recovery_self_check'}:
                return json.dumps({'action': 'accept', 'reason': 'Only observations; no changed protocol.',
                                   'evidence_ids': [context['evidence'][0]['id']]})
            if not context['probe_results']:
                return json.dumps({'action': 'probe', 'tool': 'source_context',
                    'arguments': {'paths': ['train.py']}, 'reason': 'Check the frozen author entrypoint.',
                    'evidence_ids': ['frozen_plan']})
            return json.dumps({'action': 'finish', 'reason': 'No additional verified action; retain the original result.',
                               'evidence_ids': [context['probe_results'][0]['evidence'][0]['id']]})
        return original(prompt, **kwargs)
    llm.chat = chat


def test_prepare_only_has_no_investigation_or_new_execution_gate(tmp_path, monkeypatch):
    _, service, request, llm = setup_service(tmp_path, monkeypatch)
    enable_model(llm)
    service.runner.run = Mock(side_effect=AssertionError('prepare-only cannot execute'))
    result = service.run({**request, 'prepare_only': True, 'intelligent_recovery': True})
    assert result['state'] == 'COMPLETED', result.get('error')
    state = result['data']['intelligent_recovery']
    assert state['tasks'] == []
    assert state['usage']['model_requests'] == len(llm.calls) > 0
    assert not any(role.startswith('recovery_') for role in state['usage']['roles'])
    assert state['budget']['max_tokens'] is None
    assert Path(state['artifact_path']).is_file()
    assert current_investigation() is None
    assert '## 智能调查与处理结果' in result['data']['report']


def test_execution_failure_on_pdf_route_consumes_probe_and_keeps_failure(tmp_path, monkeypatch):
    _, service, request, llm = setup_service(tmp_path, monkeypatch)
    enable_model(llm)
    service.runner.run = Mock(return_value={'success': False, 'reason': 'Actual native operation failed',
        'final': {'exit_code': 1, 'stderr': 'RuntimeError: unsupported native operation'}})
    result = service.run({**request, 'intelligent_recovery': True})
    assert result['state'] == 'ERROR'
    state = result['data']['intelligent_recovery']
    assert state['usage']['probes'] == 1
    assert state['tasks'][0]['status'] == 'needs_human'
    assert state['requires_human'] is True
    assert result['data']['validation']['is_reproduced'] is None
    assert service.runner.run.call_count == 1
    assert state['tasks'][0]['evidence'][0]['id'] == 'failure'


@pytest.mark.parametrize('stop', ['cleanup', 'timeout'])
def test_result_investigation_process_stop_is_not_swallowed_as_optional_analysis(tmp_path, monkeypatch, stop):
    from src.discovered_repository_reproduction import _diagnose_outcome
    from src.intelligent_recovery import RecoverySession, InvestigationStopped
    from src.result_conclusion import build_reproduction_conclusion
    monkeypatch.setattr('src.result_diagnosis.diagnose_experiment_result', Mock(return_value={'status': 'diagnosed'}))
    data = {'execution': {'executed': True, 'success': True,
        'final': {'stage': 'eval', 'exit_code': 0, 'success': True}},
        'validation': {'status': 'reproduced', 'is_reproduced': True,
            'scope': 'selected_paper_experiment', 'protocol_pass': True,
            'independent_metrics_pass': True, 'paper_metrics_verified': True}}
    original = deepcopy(data)
    session = RecoverySession(None)
    def stopped(**kwargs):
        if stop == 'timeout':
            session.state['timed_out'] = True
        session.stop('probe_stopped', cleanup_confirmed=False if stop == 'cleanup' else True)
    session.investigate = stopped
    with session.scope():
        session.attach(data, tmp_path)
        with pytest.raises(InvestigationStopped) as error:
            _diagnose_outcome(None, {}, {}, tmp_path, data['execution'], data['validation'], tmp_path, data, Mock())
    if stop == 'cleanup':
        assert error.value.process_cleanup_confirmed is False
    assert data['validation'] == original['validation']
    assert data['execution'] == original['execution']
    conclusion = build_reproduction_conclusion(data)
    assert conclusion['status'] == ('cleanup_unconfirmed' if stop == 'cleanup' else 'timed_out')
    assert conclusion['is_reproduced'] is None


def test_completed_pdf_author_command_runs_post_result_investigation_without_promoting_success(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, service, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, service, cpu_dependency_cache)
    enable_model(llm)
    result = service.run({**request, 'intelligent_recovery': True})
    assert result['state'] == 'COMPLETED', result.get('error')
    data = result['data']
    assert data['validation']['status'] == 'author_code_completed'
    assert data['validation']['is_reproduced'] is None
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    state = data['intelligent_recovery']
    assert state['tasks'][0]['trigger'] == 'result_discrepancy'
    assert state['usage']['probes'] == 1
    assert state['tasks'][0]['experiment_understanding']['steps']
    assert '## 智能调查与处理结果' in data['report']
    assert 'paper_reproduced' not in state['tasks'][0]['status']
