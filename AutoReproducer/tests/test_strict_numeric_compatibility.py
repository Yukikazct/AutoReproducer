"""Native strict retries preserve complete training and frozen author sources.

These invented CPU experiments exercise runtime mechanics, not real paper
acceptance. Only the model response and an already prepared cache are injected.
"""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from discovered_fixtures import PlannedLLM, configured_native_cache, make_experiment
from src import discovered_repository_reproduction as service
from src.audit.audit_logger import AuditLogger
from src.dependency_versions import compatibility_pins
from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.discovered_repository_runtime import CAPTURE_DIR, materialize_capture, verify_capture
from src.method_adapters import digest, read_json, write_json
from src.repository_reproduction import spec_digest
from test_discovered_repository_runtime import invoke, native_environment
from test_discovered_repository_series import series_fixture
from test_discovered_repository_service import use_preinstalled_cache


RAGGED_SUMMARY = '''
summary = [[accuracy, .5, np.array([1., 2.], dtype=np.float32)]
           for accuracy, _ in results]
test_mean, val_mean, weights_mean = np.mean(summary, axis=0)
test_var, val_var, weights_var = np.var(summary, axis=0)
assert np.array_equal(weights_mean, np.array([1., 2.]))
assert np.array_equal(weights_var, np.zeros(2))
print('AUTHOR_SUMMARY_FINISHED', flush=True)
'''


def _experiment(root, *, series=True, suffix=''):
    fixture = series_fixture(root) if series else make_experiment(root)
    script = fixture['workspace'] / 'train.py'
    source = script.read_text('utf-8') + '\n' + suffix
    script.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][script.name] = digest(script)
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    fixture['packet'] = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    return fixture


def _service_for(fixture, root, monkeypatch, cache):
    snapshot = {**fixture['snapshot'], 'resolved_sha': fixture['snapshot']['revision']}

    def export(_root, profile, workspace, **_kwargs):
        assert profile['repository'] == fixture['proposal']['repository']
        shutil.copytree(fixture['workspace'], workspace)
        return deepcopy(snapshot)

    monkeypatch.setattr(service, 'export_repository', export)
    llm = PlannedLLM(fixture['proposal'])
    logger = AuditLogger(log_dir=str(root / 'logs'), ledger_dir=str(root / 'ledger'))
    runner = service.DiscoveredRepositoryReproduction(root / 'service', logger, llm=llm)
    use_preinstalled_cache(monkeypatch, runner, cache)
    request = {'pdf_path': str(fixture['pdf']), 'offline': True, 'strict_validation': True,
               'paper_info': {'title': 'Invented numeric compatibility integration fixture'},
               'resources': {'code_repo_url': snapshot['url'],
                             'repo_discovery': {'selected_repo': snapshot['url'],
                                                'pinned_revision': snapshot['revision']}}}
    return runner, request


def _cache():
    cache = configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', compatibility_pins()['torch'])
    if cache is None:
        pytest.skip('Requires a dedicated prepared CPU cache; no installation is performed')
    return cache


@pytest.fixture(scope='module')
def completed_ragged_service(tmp_path_factory):
    root = tmp_path_factory.mktemp('strict_ragged_service')
    fixture = _experiment(root, suffix=RAGGED_SUMMARY)
    with pytest.MonkeyPatch.context() as monkeypatch:
        runner, request = _service_for(fixture, root, monkeypatch, _cache())
        result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    return fixture, result['data']


def test_real_service_retries_post_training_ragged_summary_and_evaluates_every_run(completed_ragged_service):
    fixture, data = completed_ragged_service
    validation = data['validation']
    assert data['strict_validation']['status'] == 'verified'
    assert validation['paper_metrics_verified'] is True
    assert validation['protocol_pass'] is True
    assert validation['independent_metrics_pass'] is True
    assert validation['is_reproduced'] is False  # Keep the frozen 100% reference.
    repairs = data['numeric_repair_attempts']
    assert len(repairs) == 1
    repair = repairs[0]
    assert repair['kind'] == 'numpy_ragged_summary'
    assert [call['operation'] for call in repair['calls']] == ['mean', 'var']
    failed = Path(repair['failed_execution_dir'])
    assert failed != Path(data['execution']['run_dir'])
    assert 'inhomogeneous' in (failed / 'train.stderr.log').read_text('utf-8')
    assert (failed / 'train.stdout.log').read_text('utf-8').count('AUTHOR_NATIVE_RUN_CAPTURED') == 3
    capture, metrics = validation['capture'], validation['independent_metrics']
    assert [run['optimizer_steps'] for run in capture['runs']] == [8, 10, 12]
    assert capture['optimizer_steps'] == 30
    assert [run['accuracy'] for run in metrics['runs']] == [1.0, .75, 1.0]
    assert metrics['accuracy'] == (1.0 + .75 + 1.0) / 3
    assert capture['process_pid'] != metrics['process_pid']
    assert metrics['capture_process_pid'] == capture['process_pid']
    assert all(run['predictions_match'] and run['numpy_torch_crosscheck'] for run in metrics['runs'])
    runtime = Path(data['run_dir']) / 'repo' / CAPTURE_DIR
    compatibility = read_json(runtime / 'config.json')['numeric_compatibility']
    assert len(compatibility) == 1
    assert capture['numeric_compatibility'] == metrics['numeric_compatibility'] == compatibility
    assert validation['numeric_compatibility'] == compatibility
    assert not (runtime / 'numeric_failure_train.json').exists()
    assert all(digest(runtime.parent / name) == expected for name, expected in fixture['snapshot']['files'].items())
    assert 'AUTHOR_SUMMARY_FINISHED' in (Path(data['execution']['run_dir']) / 'train.stdout.log').read_text('utf-8')


@pytest.mark.parametrize('target', ['config.json', 'capture.json', 'independent_metrics.json'])
def test_strict_host_rejects_tampered_compatibility_receipts(completed_ragged_service, target):
    _, data = completed_ragged_service
    workspace = Path(data['run_dir']) / 'repo'
    runtime = workspace / CAPTURE_DIR
    path = runtime / target
    before = path.read_bytes()
    value = read_json(path)
    value['numeric_compatibility'] = []
    write_json(path, value)
    try:
        with pytest.raises(ValueError, match='(?i)compatibility|repair'):
            verify_capture(data['grounded_repository_plan'], workspace, data['execution'], data['spec_sha256'])
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize('series,updates', [(False, 40), (True, 30)])
def test_post_training_alias_failure_cannot_be_repaired_after_completed_captures(
        tmp_path, monkeypatch, series, updates):
    fixture = _experiment(tmp_path, series=series,
                          suffix='import numpy as np\nlegacy_integer = np.int(1)\n')
    runner, request = _service_for(fixture, tmp_path, monkeypatch, _cache())
    result = runner.run(request)
    assert result['state'] == 'ERROR'
    data = result['data']
    assert data['reproduction_scope'] == 'selected_paper_experiment'
    assert not data.get('numeric_repair_attempts')
    runtime = Path(data['run_dir']) / 'repo' / CAPTURE_DIR
    failure = read_json(runtime / 'numeric_failure_train.json')
    assert failure['repair']['kind'] == 'numpy_builtin_alias'
    assert failure['optimizer_steps'] == updates
    assert not read_json(runtime / 'config.json')['numeric_compatibility']
    assert not (runtime / 'capture.json').exists()
    assert not (runtime / 'independent_metrics.json').exists()
    assert all(digest(runtime.parent / name) == expected for name, expected in fixture['snapshot']['files'].items())
    assert data['validation']['is_reproduced'] is None


@pytest.mark.parametrize('series', [False, True])
def test_unrelated_value_error_stays_failure_and_invalidates_old_success_receipts(tmp_path, series):
    fixture = _experiment(tmp_path, series=series,
                          suffix="raise ValueError('intentional unrelated author failure')\n")
    plan = propose_plan(PlannedLLM(fixture['proposal']), fixture['packet'], fixture['workspace'])
    spec_hash = spec_digest(plan)
    materialize_capture(fixture['workspace'], plan, spec_hash, snapshot=fixture['snapshot'])
    runtime = fixture['workspace'] / CAPTURE_DIR
    stale = ['capture.json', 'independent_metrics.json', 'numeric_failure_train.json']
    for name in stale:
        write_json(runtime / name, {'pass': True, 'spec_sha256': spec_hash, 'stale': True})
    completed = invoke(fixture['workspace'], 'capture', native_environment())
    assert completed.returncode != 0
    assert 'intentional unrelated author failure' in completed.stderr
    assert not any((runtime / name).exists() for name in stale)
    assert all(digest(fixture['workspace'] / name) == expected
               for name, expected in fixture['snapshot']['files'].items())


def test_failed_fresh_evaluation_clears_previous_success_receipt(completed_ragged_service):
    _, data = completed_ragged_service
    workspace = Path(data['run_dir']) / 'repo'
    runtime = workspace / CAPTURE_DIR
    capture_path, metrics_path = runtime / 'capture.json', runtime / 'independent_metrics.json'
    capture_before, metrics_before = capture_path.read_bytes(), metrics_path.read_bytes()
    capture = read_json(capture_path)
    capture['numeric_compatibility'] = []
    write_json(capture_path, capture)
    try:
        completed = invoke(workspace, 'evaluate', native_environment())
        assert completed.returncode != 0
        assert not metrics_path.exists()
    finally:
        capture_path.write_bytes(capture_before)
        metrics_path.write_bytes(metrics_before)
