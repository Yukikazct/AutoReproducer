"""Preset entrypoints ignore stale review flags and keep genuine exit failures."""
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.audit.audit_logger import AuditLogger
from src.repository_reproduction import RepositoryReproduction
from src.method_reproduction import MethodReproduction


@pytest.mark.parametrize('profile_id,service_module,service_type', [
    ('dlinear_etth1_reference', 'src.repository_reproduction', RepositoryReproduction),
    ('neural_ode_spiral', 'src.method_reproduction', MethodReproduction),
    ('siren_camera_quick', 'src.method_reproduction', MethodReproduction),
])
@pytest.mark.parametrize('exit_code', [0, 1])
def test_review_flags_cannot_invoke_preset_reviewers(tmp_path, monkeypatch, profile_id, service_module, service_type, exit_code):
    import importlib
    module = importlib.import_module(service_module)
    workspace = tmp_path / 'author'
    workspace.mkdir()
    snapshot = {'path': str(workspace), 'url': 'https://github.com/example/author',
                'resolved_sha': 'a'*40, 'files': {}}
    import src.repository_reproduction as repository_module
    monkeypatch.setattr(repository_module, 'export_repository', lambda *a, **k: snapshot)
    adapter = Mock()
    adapter.prepare_dataset.return_value = {'url': 'https://example.org/real-data', 'verified': False}
    adapter.materialize.return_value = {'files': {}}
    adapter.public_sources.return_value = []
    adapter.steps.return_value = [
        {'id': 'import_check', 'kind': 'check', 'argv': ['python', '-c', 'raise AssertionError()'], 'timeout_s': 60},
        {'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'], 'timeout_s': 60, 'depends_on': ['import_check']},
        {'id': 'evaluate', 'kind': 'eval', 'argv': ['python', 'evaluate.py'], 'timeout_s': 60},
    ]
    adapter.metric_records.side_effect = ValueError('No structured metrics; execution must still complete')
    monkeypatch.setattr(module, 'get_adapter', lambda profile: adapter)
    calls = []
    def execute(root, steps, env, **kwargs):
        calls.append((steps, env))
        preparation = steps[0]['id'] == 'prepare_dependencies'
        code = 0 if preparation else exit_code
        final = {'success': code == 0, 'exit_code': code, 'stderr': '' if code == 0 else 'Actual author error', 'artifacts': []}
        return {'success': code == 0, 'executed': True, 'repository_executed': True,
                'final': final, 'steps': [{**step, **final, 'executed': True} for step in steps],
                'environment': {}, 'artifacts': [], 'effective_env_config': env}
    runner = Mock(); runner.run.side_effect = execute
    llm = Mock(); llm.get_call_count.return_value = 0
    llm.chat.side_effect = AssertionError('Deleted reviewer must not call a model')
    logger = AuditLogger(log_dir=str(tmp_path / 'logs'))
    service = service_type(tmp_path / 'service', logger, runner=runner, llm=llm)
    events = []
    result = service.run({'experiment_profile': profile_id, 'use_llm_review': True,
                         'allow_result_summary_review': True, 'optimization_mode': 'validate'}, on_event=events.append)
    assert result['state'] == ('COMPLETED' if exit_code == 0 else 'ERROR'), result['error']
    if exit_code == 0:
        assert result['data']['validation']['status'] == 'author_code_completed'
        assert result['data']['validation']['is_reproduced'] is None
    else:
        assert 'Actual author error' in result['error']
    assert calls and all(env['dependency_health_check'] is False for _, env in calls)
    assert all(step['timeout_s'] is None for steps, _ in calls for step in steps
               if step['kind'] != 'check')
    assert all(env.get('deadline_monotonic') is None for _, env in calls)
    assert all(step['id'] not in ('import_check', 'evaluate') for steps, _ in calls for step in steps)
    stages = next(e['stages'] for e in events if e['type'] == 'pipeline_plan')
    assert not any(s['id'].startswith(('review_', 'verify_', 'validate_', 'analyze_')) for s in stages)
    llm.chat.assert_not_called()
    adapter.analysis.assert_not_called()
    adapter.verify.assert_not_called()
    adapter.verify_environment.assert_not_called()
    adapter.verify_protocol.assert_not_called()
    adapter.recompute_metrics.assert_not_called()
    adapter.validate.assert_not_called()
