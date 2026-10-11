"""Host recovery continues the same preset without exposing request credentials."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from frontend.backend_pipeline import ProgressStore, run_pipeline_core
from src import runtime_preparation as runtime


@pytest.fixture
def recover(monkeypatch, tmp_path):
    prepared = runtime.RuntimePreparation(str(tmp_path / 'safe' / 'python.exe'),
                                          True, False, 'Store host', 'cpython-312')
    monkeypatch.setattr(runtime, 'runtime_requirement', lambda: 'Switch Store host')
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(runtime, 'prepare_runtime', prepare)
    return prepared, prepare


def test_owned_worker_preserves_bootstrap_order_and_private_request(recover, tmp_path, monkeypatch):
    path = tmp_path / 'progress.jsonl'
    secret = 'sk-private-fixture-123456789'
    monkeypatch.setenv('LLM_API_KEY', 'sk-host-fixture-987654321')
    captured = {}
    def execute(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        payload = json.loads(kwargs['input'])
        assert payload['api_key'] == secret
        assert 'LLM_API_KEY' not in kwargs['env']
        child = ProgressStore(payload['progress_path'], reset=False)
        child.emit({'type':'pipeline_plan', 'stages':[{'id':'execute_repository', 'status':'waiting'}]})
        result = {'state':'COMPLETED', 'error':None, 'data':{'runtime_preparation':payload['_runtime_preparation']}}
        child.emit({'type':'done', 'result':result})
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    result = run_pipeline_core(str(path), experiment_profile='dlinear_etth1_reference',
                               api_key=secret, offline=True)
    assert result['state'] == 'COMPLETED'
    assert result['data']['runtime_preparation']['reused'] is True
    assert secret not in ' '.join(captured['argv'])
    assert secret not in path.read_text(encoding='utf-8')
    assert 'sk-host-fixture-987654321' not in json.dumps(captured['env'])
    assert recover[1].call_args.kwargs['offline'] is True
    stages = ProgressStore.read_snapshot(str(path))['pipeline_stages']
    assert [s['id'] for s in stages] == ['prepare_runtime', 'execute_repository']
    assert stages[0]['status'] == 'success'


def test_environment_key_is_forwarded_only_in_worker_pipe(recover, tmp_path, monkeypatch):
    secret = 'environment-credential-without-prefix'
    monkeypatch.setenv('LLM_API_KEY', secret)
    def execute(argv, **kwargs):
        payload = json.loads(kwargs['input'])
        assert payload['api_key'] == secret
        assert 'LLM_API_KEY' not in kwargs['env']
        return subprocess.CompletedProcess(argv, 1, '', f'worker failed: {secret}')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    path = tmp_path / 'progress.jsonl'
    result = run_pipeline_core(str(path), experiment_profile='dlinear_etth1_reference')
    assert result['state'] == 'ERROR'
    assert secret not in result['error']
    assert secret not in path.read_text(encoding='utf-8')


def test_worker_failure_keeps_successful_runtime_and_redacts_credential(recover, tmp_path, monkeypatch):
    secret = 'sk-private-fixture-123456789'
    monkeypatch.setattr(runtime, 'run_owned_process', Mock(return_value=
        subprocess.CompletedProcess([], 1, '', f'Authorization: Bearer {secret}')))
    path = tmp_path / 'progress.jsonl'
    result = run_pipeline_core(str(path), experiment_profile='dlinear_etth1_reference', api_key=secret)
    assert result['state'] == 'ERROR'
    view = ProgressStore.read_snapshot(str(path))
    assert view['pipeline_stages'][0]['status'] == 'success'
    assert not view['running']
    assert secret not in path.read_text(encoding='utf-8')


def test_runtime_recovery_failure_does_not_construct_orchestrator(recover, tmp_path, monkeypatch):
    import frontend.backend_pipeline as pipeline
    monkeypatch.setattr(runtime, 'prepare_runtime', Mock(side_effect=runtime.RuntimePreparationError('offline miss')))
    factory = Mock(side_effect=AssertionError('unsafe host must not execute'))
    monkeypatch.setattr(pipeline, 'Orchestrator', factory)
    path = tmp_path / 'progress.jsonl'
    result = run_pipeline_core(str(path), experiment_profile='dlinear_etth1_reference')
    assert result['state'] == 'ERROR'
    assert ProgressStore.read_snapshot(str(path))['pipeline_stages'][0]['status'] == 'error'
    factory.assert_not_called()


def test_local_settings_respect_explicit_environment_and_whitelist(tmp_path, monkeypatch):
    from src.local_llm_settings import load_local_llm_settings
    directory = tmp_path / '.streamlit'
    directory.mkdir()
    (directory / 'secrets.toml').write_text('LLM_API_KEY="local-secret"\nLLM_MODEL="deepseek-chat"\nUNRELATED="value"\n', encoding='utf-8')
    monkeypatch.setenv('LLM_API_KEY', 'explicit-secret')
    monkeypatch.delenv('LLM_MODEL', raising=False)
    monkeypatch.delenv('UNRELATED', raising=False)
    load_local_llm_settings(tmp_path)
    import os
    assert os.environ['LLM_API_KEY'] == 'explicit-secret'
    assert os.environ['LLM_MODEL'] == 'deepseek-chat'
    assert 'UNRELATED' not in os.environ


@pytest.mark.parametrize('exit_code', [0, 1])
def test_dead_owned_worker_closes_abandoned_invocation_as_interrupted(recover, tmp_path, monkeypatch, exit_code):
    path = tmp_path / 'progress.jsonl'
    def execute(argv, **kwargs):
        child = ProgressStore(json.loads(kwargs['input'])['progress_path'], reset=False)
        child.emit({'type':'state', 'state':'EXECUTE_CODE', 'agent':'CodeExecutor',
                    'phase_id':'execute_repository', 'status':'running'})
        child.emit({'type':'execution_run', 'execution_id':'abandoned',
                    'phase_id':'execute_repository', 'status':'running'})
        child.emit({'type':'execution_step', 'execution_id':'abandoned', 'step_index':1,
                    'phase_id':'execute_repository', 'status':'running'})
        return subprocess.CompletedProcess(argv, exit_code, '', 'worker exited')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    result = run_pipeline_core(str(path), experiment_profile='neural_ode_spiral')
    view = ProgressStore.read_snapshot(str(path))
    assert result['state'] == 'ERROR'
    assert view['done'] and not view['running']
    assert not view['active_runs'] and not view['active_executions'] and not view['active_workers']
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    terminal = next(event for event in events if event.get('execution_id') == 'abandoned'
                    and event.get('type') == 'execution_run' and event.get('status') == 'interrupted')
    assert terminal['cleanup_confirmed'] is True


def test_owned_worker_early_done_waits_for_owner_cleanup(recover, tmp_path, monkeypatch):
    path = tmp_path / 'progress.jsonl'
    def execute(argv, **kwargs):
        child = ProgressStore(json.loads(kwargs['input'])['progress_path'], reset=False)
        child.emit({'type':'done', 'result':{'state':'COMPLETED', 'data':{}, 'error':None}})
        live = ProgressStore.read_snapshot(str(path))
        assert live['running'] and not live['done'] and live['finalization_pending']
        assert len(live['active_workers']) == 1
        kwargs['on_cleanup']()
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    result = run_pipeline_core(str(path), experiment_profile='neural_ode_spiral')
    assert result['state'] == 'COMPLETED'
    assert not ProgressStore.read_snapshot(str(path))['active_workers']


@pytest.mark.parametrize('confirmed', [True, False])
def test_worker_timeout_only_finalizes_after_confirmed_cleanup(recover, tmp_path, monkeypatch, confirmed):
    path = tmp_path / 'progress.jsonl'
    def execute(argv, **kwargs):
        child = ProgressStore(json.loads(kwargs['input'])['progress_path'], reset=False)
        child.emit({'type':'execution_run', 'execution_id':'timeout-run',
                    'phase_id':'execute_repository', 'status':'running'})
        if confirmed:
            kwargs['on_cleanup']()
        raise runtime.RuntimePreparationTimeout('owned timeout')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    result = run_pipeline_core(str(path), experiment_profile='neural_ode_spiral')
    view = ProgressStore.read_snapshot(str(path))
    assert result['state'] == 'ERROR'
    assert view['running'] is not confirmed
    assert view['done'] is confirmed
    assert bool(view['active_runs']) is not confirmed
    assert bool(view['active_workers']) is not confirmed


def test_worker_failed_exit_rejects_early_success_result(recover, tmp_path, monkeypatch):
    path = tmp_path / 'progress.jsonl'
    def execute(argv, **kwargs):
        child = ProgressStore(json.loads(kwargs['input'])['progress_path'], reset=False)
        child.emit({'type':'done', 'result':{'state':'COMPLETED', 'data':{}, 'error':None}})
        return subprocess.CompletedProcess(argv, 1, '', 'failed after writing done')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    result = run_pipeline_core(str(path), experiment_profile='neural_ode_spiral')
    assert result['state'] == 'ERROR' and result['error']
    assert ProgressStore.read_snapshot(str(path))['state'] == 'ERROR'


def test_unknown_title_also_recovers_store_host_before_generic_agents(recover, tmp_path, monkeypatch):
    import frontend.backend_pipeline as pipeline
    factory = Mock(side_effect=AssertionError('unsafe generic host must not execute'))
    monkeypatch.setattr(pipeline, 'Orchestrator', factory)
    captured = {}
    def execute(argv, **kwargs):
        payload = json.loads(kwargs['input'])
        captured.update(payload)
        child = ProgressStore(payload['progress_path'], reset=False)
        child.emit({'type':'done', 'result':{'state':'COMPLETED', 'error':None, 'data':{
            'validation':{'status':'inconclusive', 'is_reproduced':None}}}})
        return subprocess.CompletedProcess(argv, 0, '', '')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    path = tmp_path / 'generic.jsonl'
    result = run_pipeline_core(str(path), paper_title='Unadapted Actual Research Paper')
    assert result['state'] == 'COMPLETED'
    assert captured['experiment_profile'] is None
    assert captured['paper_title'] == 'Unadapted Actual Research Paper'
    assert captured['_managed_runtime'] is True
    factory.assert_not_called()
    assert not ProgressStore.read_snapshot(str(path))['active_workers']
