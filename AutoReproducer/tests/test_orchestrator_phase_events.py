"""Removed review stages cannot block actual execution or fake completion."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager
from src.execution_completion import completed_execution


@pytest.fixture
def pipeline(tmp_path):
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError('Reviewer must not call a model'))
    orch = Orchestrator(llm_client=llm, mock_mode=True,
        logger=AuditLogger(log_dir=str(tmp_path / 'logs')),
        resource_manager=ResourceManager(data_root=str(tmp_path / 'data')))
    outputs = {
        'reader': {'paper_info': {'title': 'Public paper'}},
        'finder': {'resources': {'code_repo_url': 'https://github.com/example/repo'}},
        'builder': {'env_config': {}},
        'executor': {'success': True, 'executed': True, 'final': {'success': True, 'exit_code': 0}},
        'reporter': {'report': 'Actual command output'},
    }
    for name, output in outputs.items():
        orch.agents[name].run = Mock(return_value=deepcopy(output))
    orch._fetch_resources = Mock()
    orch._finalize_storage = Mock()
    return orch


@pytest.mark.parametrize('mock_mode', [True, False])
def test_all_generic_review_agents_and_stages_are_removed(pipeline, mock_mode):
    pipeline.mock_mode = mock_mode
    events = []
    result = pipeline.run({'paper_title': 'Public paper', 'use_llm_review': True,
                           'code': "print('explicit test program')"}, on_event=events.append)
    assert result['state'] == 'COMPLETED', result['error']
    assert 'verifier' not in pipeline.agents and 'validator' not in pipeline.agents
    assert not hasattr(pipeline, '_verify_step')
    stages = next(e['stages'] for e in events if e['type'] == 'pipeline_plan')
    assert not any(s['id'].startswith(('verify_', 'review_', 'validate_')) for s in stages)
    assert result['data']['verifications'] == []
    assert result['data']['validation']['is_reproduced'] is None
    pipeline.llm.chat.assert_not_called()


@pytest.mark.parametrize('exit_code', [1, 2, 130])
def test_failed_command_keeps_error_state_and_generates_failure_report(pipeline, exit_code):
    pipeline.agents['executor'].run.return_value = {
        'success': False, 'executed': True,
        'final': {'success': False, 'exit_code': exit_code, 'stderr': 'Actual author error'}}
    events = []
    result = pipeline.run({'paper_title': 'Public paper'}, on_event=events.append)
    assert result['state'] == 'ERROR' and 'Actual author error' in result['error']
    pipeline.agents['reporter'].run.assert_called_once()
    assert result['data']['validation']['status'] == 'execution_failed'
    report = result['data']['report']
    original, marker, summary = report.partition('\n\n<!-- autoreproducer-run-finalization:start -->\n')
    assert original == 'Actual command output'
    assert marker and report.count('<!-- autoreproducer-run-finalization:start -->') == 1
    assert report.count('<!-- autoreproducer-run-finalization:end -->') == 1
    assert '**总任务状态**: 异常结束' in summary
    assert '**环境释放**: 已释放' in summary
    assert 'Actual author error' in summary
    assert result['data']['run_finalization'] == {'state': result['state'], 'error': result['error']}
    assert result['data']['environment_cleanup']['status'] == 'released'
    assert result['data']['report_length'] == len(report)
    assert any(e.get('phase_id') == 'execute_code' and e.get('status') == 'error' for e in events)


@pytest.mark.parametrize('change', [
    {'executed': False}, {'success': False}, {'cancelled': True}, {'interrupted': True},
    {'final': {'success': True, 'exit_code': None}},
    {'steps': [{'exit_code': 0, 'timed_out': True}]},
    {'steps': [{'exit_code': 0, 'cancelled': True}]},
])
def test_preparation_cancellation_and_timeout_cannot_be_reported_as_completion(change):
    execution = {'executed': True, 'success': True, 'final': {'success': True, 'exit_code': 0}}
    execution.update(change)
    with pytest.raises(RuntimeError):
        completed_execution(execution)


def test_code_completion_does_not_claim_training_or_paper_acceptance():
    result = completed_execution({'success': True, 'final': {'success': True, 'exit_code': 0}})
    assert result['status'] == 'author_code_completed'
    assert result['paper_metrics_verified'] is False and result['is_reproduced'] is None
    assert 'training_observed' not in result
