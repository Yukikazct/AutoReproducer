"""Keep interrupted evidence across forced owner shutdown without early success."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontend.backend_pipeline import ProgressStore, run_pipeline_core
from src import runtime_preparation as runtime
from test_author_claim_comparison_service import comparison_service
from test_preset_runtime_delegation import recover


@pytest.mark.parametrize('phase', ['training', 'comparison', 'report'])
def test_repository_emits_saved_interruption_before_environment_cleanup(tmp_path, monkeypatch, phase):
    from src.agents.report_generator import ReportGeneratorAgent
    import src.environment_lifecycle as lifecycle

    options = {'runtime_outcome': 'interrupted'} if phase == 'training' else {}
    if phase == 'comparison':
        options['outcome'] = KeyboardInterrupt('comparison cancelled')
    _, service, request = comparison_service(tmp_path, monkeypatch, **options)
    if phase == 'report':
        monkeypatch.setattr(ReportGeneratorAgent, 'run', Mock(side_effect=KeyboardInterrupt('report cancelled')))
    events, saved = [], []
    original = lifecycle._release_environment
    def release(root):
        checkpoints = [e['result'] for e in events if e.get('type') == 'result_checkpoint']
        if checkpoints:
            checkpoint = checkpoints[-1]
            disk = json.loads((Path(checkpoint['data']['run_dir']) / 'result.json').read_text('utf-8'))
            assert checkpoint == disk
            saved.append(checkpoint)
        return original(root)
    monkeypatch.setattr(lifecycle, '_release_environment', release)
    with pytest.raises(KeyboardInterrupt):
        service.run(request, on_event=events.append)
    assert saved, 'interrupted evidence must be emitted before slow cleanup can be force-stopped'
    checkpoint = saved[-1]
    assert checkpoint['state'] == 'ERROR' and checkpoint['data']['interrupted'] is True
    validation = checkpoint['data']['validation']
    assert validation['status'] == ('interrupted' if phase == 'training' else 'author_code_completed')


@pytest.mark.parametrize('confirmed', [True, False])
def test_supervisor_recovers_checkpoint_only_after_confirmed_process_cleanup(
        recover, tmp_path, monkeypatch, confirmed):
    import frontend.backend_pipeline as backend
    path, run = tmp_path / 'progress.jsonl', tmp_path / 'retained-run'
    run.mkdir()
    result = {'state': 'ERROR', 'error': 'interrupted', 'data': {
        'run_dir': str(run), 'interrupted': True, 'validation': {'status': 'author_code_completed'},
        'execution': {'success': True, 'stdout': 'actual training evidence'}}}
    (run / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    release = Mock(return_value={'status': 'released', 'errors': [], 'remaining': []})
    monkeypatch.setattr(runtime, 'release_worker_environment', release)
    def execute(argv, **kwargs):
        child = ProgressStore(json.loads(kwargs['input'])['progress_path'], reset=False)
        child.emit({'type': 'result_checkpoint', 'result': result})
        view = child.read_snapshot(str(path))
        assert view['running'] and view['result'] is None and not view['done']
        assert 'result_checkpoint' not in view
        assert child.read_snapshot(str(path), include_pending_result=True)['result_checkpoint'] == result
        if confirmed:
            kwargs['on_cleanup']()
            raise KeyboardInterrupt('forced shutdown during slow cleanup')
        error = runtime.ProcessCleanupError('descendants unconfirmed')
        error.process_cleanup_confirmed = False
        error.cleanup_receipt = {'status': 'error', 'process_cleanup_confirmed': False, 'errors': ['unconfirmed']}
        raise error
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline_core(str(path), experiment_profile='neural_ode_spiral')
    view = ProgressStore.read_snapshot(str(path))
    assert view['running'] is not confirmed and view['done'] is confirmed
    if confirmed:
        release.assert_called_once()
        final = view['result']
        assert final['data']['run_dir'] == str(run)
        assert final['data']['validation'] == result['data']['validation']
        assert final['data']['execution'] == result['data']['execution']
        assert final['data']['environment_cleanup']['status'] == 'released'
        assert json.loads((run / 'result.json').read_text('utf-8'))['data'] == final['data']
    else:
        release.assert_not_called()
        assert view['result'] is None and view['active_workers']
        assert not (run / 'environment_cleanup.json').exists()


@pytest.mark.parametrize('value', [None, {}, {'state': 'COMPLETED', 'data': {'interrupted': True}},
    {'state': 'ERROR', 'data': {}}, {'state': 'ERROR', 'data': {'interrupted': True, 'run_dir': ''}}])
def test_invalid_or_successful_checkpoint_cannot_become_a_result(tmp_path, value):
    store = ProgressStore(str(tmp_path / 'progress.jsonl'))
    store.emit({'type': 'result_checkpoint', 'result': value})
    view = store.read_snapshot(str(store.path), include_pending_result=True)
    assert view.get('result_checkpoint') is None
    assert view['running'] and view['result'] is None and not view['done']


def test_backend_forwards_interruption_checkpoint_before_lease_cleanup(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend
    import src.environment_lifecycle as lifecycle
    _, service, request = comparison_service(tmp_path, monkeypatch, runtime_outcome='interrupted')
    progress = tmp_path / 'forwarding.jsonl'
    def orchestrator(**kwargs):
        service.logger = kwargs['logger']
        return SimpleNamespace(run=lambda _request, on_event: service.run(request, on_event=on_event))
    monkeypatch.setattr(backend, '_create_orchestrator', orchestrator)
    original, observed = lifecycle._release_environment, []
    def release(root):
        view = ProgressStore.read_snapshot(str(progress), include_pending_result=True)
        if view.get('result_checkpoint'):
            assert view['result'] is None and not view['done']
            observed.append(view['result_checkpoint'])
        return original(root)
    monkeypatch.setattr(lifecycle, '_release_environment', release)
    with pytest.raises(KeyboardInterrupt, match='author command cancelled'):
        run_pipeline_core(str(progress), paper_title='Interrupted fixture', mock_mode=True)
    assert observed and observed[0]['data']['interrupted'] is True
    final = ProgressStore.read_snapshot(str(progress))['result']
    assert final['session_id'] and final['audit_logs']
    assert final['data']['run_dir'] == observed[0]['data']['run_dir']
    assert final['data']['environment_cleanup']['status'] == 'released'
    saved = json.loads((Path(final['data']['run_dir']) / 'result.json').read_text('utf-8'))
    for key in ('session_id', 'audit_logs', 'audit_stats'):
        assert saved[key] == final[key]


def test_background_keeps_finalized_interruption_instead_of_bootstrap_diagnostic(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend
    from frontend.task_control import task_status
    progress = tmp_path / 'background.jsonl'
    result = {'state': 'ERROR', 'error': 'interrupted; 后台执行已取消', 'session_id': 'fixture',
              'data': {'run_dir': str(tmp_path / 'run'), 'interrupted': True,
                       'environment_cleanup': {'status': 'released', 'errors': []}}}
    def core(**kwargs):
        ProgressStore(kwargs['progress_path'], reset=False).emit({'type': 'done', 'result': result})
        raise KeyboardInterrupt('运行环境准备已取消')
    monkeypatch.setattr(backend, 'run_pipeline_core', core)
    thread = backend.run_pipeline_background(str(progress), paper_title='interrupted')
    thread.join(timeout=10)
    assert not thread.is_alive()
    view = ProgressStore.read_snapshot(str(progress))
    assert view['done'] and not view['running']
    assert view['result'] == result
    assert view['error'] == result['error']
    assert task_status(str(progress))['active'] is False
