"""Final resource release cannot leave a detached recovery receipt optimistic."""
from copy import deepcopy
import json

import pytest

from src.agents.report_generator import ReportGeneratorAgent
from src.environment_lifecycle import _EnvironmentRun, record_environment_cleanup
from src.intelligent_recovery import RecoverySession, RecoveryBudgetExceeded, current_investigation
from src.task_recovery import summarize_recovery_state


def _finished_task(tmp_path):
    run_dir = tmp_path / '中文运行'
    run_dir.mkdir()
    data = {'run_dir': str(run_dir), 'report_path': str(run_dir / 'report.md'),
        'paper_info': {'title': 'Original experiment'},
        'execution': {'executed': True, 'mode': 'repository',
                      'final': {'success': True, 'exit_code': 0, 'stdout': 'accuracy=0.8'}},
        'validation': {'status': 'author_code_completed', 'paper_metrics_verified': False}}
    with RecoverySession(None).scope() as session:
        session.attach(data, run_dir)
        session.run_action('replan', lambda: {'valid': True}, problem='Missing entrypoint',
            verify=lambda value: {'success': value['valid']})
    assert current_investigation() is None
    data['report'] = ReportGeneratorAgent.build_report(data, report_path=run_dir / 'report.md')
    (run_dir / 'report.md').write_text(data['report'], encoding='utf-8')
    result = {'state': 'COMPLETED', 'error': None, 'data': data}
    (run_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    return result, run_dir


@pytest.mark.parametrize('nested', [
    {'process_cleanup_confirmed': False},
    {'supervisor_cleanup': {'cleanup_receipt': {'process_cleanup_confirmed': False}}},
    {'training_python': {'attempts': [{'cleanup': {'status': 'unconfirmed'}}]}},
])
def test_late_process_cleanup_stops_detached_recovery_and_every_saved_view(tmp_path, nested):
    result, run_dir = _finished_task(tmp_path)
    original = deepcopy(result['data']['execution'])
    receipt = {'status': 'error', 'errors': [{'reason': 'Process ownership cleanup unconfirmed'}],
               'removed': [], 'remaining': [], **nested}
    record_environment_cleanup(result, receipt)
    recovery = result['data']['intelligent_recovery']
    assert current_investigation() is None
    assert recovery['status'] == 'stopped'
    assert recovery['cleanup_confirmed'] is False
    assert recovery['requires_human'] is True
    assert 'cleanup' in recovery['stop_reason']
    assert result['data']['recovery_summary']['status'] == 'stopped'
    assert result['data']['execution'] == original
    assert result['data']['report_view_model']['conclusion']['execution']['status'] == 'cleanup_unconfirmed'
    assert '进程清理未确认' in result['data']['report']
    for path in (run_dir / 'result.json',):
        saved = json.loads(path.read_text('utf-8'))
        assert saved['data']['intelligent_recovery'] == recovery
        assert saved['data']['recovery_summary'] == result['data']['recovery_summary']
    assert json.loads((run_dir / 'intelligent_recovery.json').read_text('utf-8')) == recovery
    assert (run_dir / 'report.md').read_text('utf-8') == result['data']['report']


def test_file_release_failure_preserves_author_execution_and_recovery_evidence(tmp_path):
    result, run_dir = _finished_task(tmp_path)
    before = deepcopy(result['data']['intelligent_recovery'])
    record_environment_cleanup(result, {'status': 'error', 'removed': [], 'remaining': ['wheel.whl'],
        'errors': [{'path': 'wheel.whl', 'reason': 'Permission denied'}],
        'runtime_downloads': {'status': 'error', 'errors': [{'reason': 'file is locked'}]}})
    assert result['state'] == 'ERROR'  # Publication/resource release failed.
    assert result['data']['intelligent_recovery'] == before
    assert result['data']['recovery_summary']['actions_verified'] == 1
    assert result['data']['report_view_model']['conclusion']['execution']['status'] == 'completed'
    assert json.loads((run_dir / 'intelligent_recovery.json').read_text('utf-8')) == before


def test_cancellation_reloads_saved_result_before_late_cleanup(tmp_path):
    result, run_dir = _finished_task(tmp_path)
    lease = _EnvironmentRun(None)
    lease.run_dir = run_dir
    lease.record({'status': 'released', 'errors': [],
                  'supervisor_cleanup': {'process_cleanup': {'status': 'unconfirmed'}}})
    assert lease.result['state'] == 'ERROR'
    assert lease.result['data']['recovery_summary']['requires_human'] is True
    assert current_investigation() is None
    saved = json.loads((run_dir / 'result.json').read_text('utf-8'))
    assert saved['data']['intelligent_recovery']['status'] == 'stopped'


def test_lifecycle_receipt_also_stops_a_still_bound_session_without_recreating_it(tmp_path):
    result, run_dir = _finished_task(tmp_path)
    with RecoverySession(None).scope() as session:
        session.attach(result['data'], run_dir)
        record_environment_cleanup(result, {'status': 'error', 'errors': [],
                                           'process_cleanup_confirmed': False})
        assert current_investigation() is session
        with pytest.raises(RecoveryBudgetExceeded):
            session.chat(None, 'must not call a model after cleanup uncertainty')


@pytest.mark.parametrize('status,execution', [
    ('stopped', {'status': 'stopped'}),
    ('verified', {'status': 'not_executed'}),
    ('verified', {'status': 'failed'}),
    ('no_progress', {'status': 'completed'}),
    ('verified', {'status': 'completed', 'result': {'cleanup_confirmed': False}}),
])
def test_stale_verification_true_cannot_inflate_detached_summary(status, execution):
    state = {'actions': [{'status': status, 'execution': execution, 'verification': {'success': True}}]}
    assert summarize_recovery_state(state)['actions_verified'] == 0
