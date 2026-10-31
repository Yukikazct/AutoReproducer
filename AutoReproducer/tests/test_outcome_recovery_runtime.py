"""A corrected author invocation owns fresh artifacts and never selects a best run."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.method_adapters import read_json, write_json
from src.repository_reproduction import spec_digest
from src.outcome_recovery_runtime import recover_outcome


def runtime_case(tmp_path, monkeypatch, *, verdict='reproduced', interrupted=False):
    from src import outcome_recovery, outcome_recovery_runtime as runtime
    from src import discovered_repository_reproduction as reproduction
    from src import intelligent_recovery, planning_recovery
    run_dir = tmp_path / 'initial'
    workspace = run_dir / 'repo'
    workspace.mkdir(parents=True)
    raw = b"print('invented author execution mechanics only')\n"
    (workspace / 'train.py').write_bytes(raw)
    repository = {'url': 'https://github.com/invented-fixtures/outcome', 'revision': 'a' * 40}
    snapshot = {**repository, 'resolved_sha': 'a' * 40, 'path': str(workspace),
                'files': {'train.py': hashlib.sha256(raw).hexdigest()}}
    plan = {'repository': repository, 'steps': [{'id': 'train', 'argv': ['python', 'train.py']}],
            'parameters': {}, 'scope': 'author_code_execution'}
    corrected = deepcopy(plan)
    corrected['steps'][0]['argv'].append('--author-reproduction')
    packet = {'pdf': {'sha256': 'b' * 64}}
    data = {'spec_sha256': spec_digest(plan), 'run_dir': str(run_dir),
        'grounded_repository_plan': plan, 'execution': {'success': True},
        'validation': {'status': 'ai_not_reproduced'},
        'outcome_diagnosis': {'reason': 'Original observed discrepancy'}}
    for name, value in [('experiment_spec.json', {'spec': plan, 'sha256': data['spec_sha256']}),
                        ('execution.json', data['execution']), ('source_packet.json', packet)]:
        write_json(run_dir / name, value)
    original_bytes = {p.name: p.read_bytes() for p in run_dir.glob('*.json')}
    proposed = {'status': 'proposed', 'reason': 'Invented fixed author command evidence',
                'plan': corrected, 'packet': packet, 'evidence': []}
    monkeypatch.setattr(outcome_recovery, 'prepare_outcome_recovery', lambda *args: deepcopy(proposed))
    reservations = []
    monkeypatch.setattr(intelligent_recovery, 'current_investigation', lambda: SimpleNamespace(
        consume_repair=lambda: reservations.append('reserved'), ensure_running=lambda: None))
    def export(root, profile, target, **kwargs):
        target.mkdir(parents=True)
        (target / 'train.py').write_bytes(raw)
        return {**snapshot, 'path': str(target)}
    monkeypatch.setattr(runtime, 'export_repository', export)
    monkeypatch.setattr(reproduction, 'environment_for_plan', lambda value: {'device': 'cpu'})
    monkeypatch.setattr(reproduction, 'prepare_plan_datasets', lambda *args, **kwargs: {'verified': False, 'sources': []})
    monkeypatch.setattr(planning_recovery, 'validate_plan_consistency', lambda *args: {'status': 'verified'})
    calls = []
    def execute(request, child, candidate, exported, child_workspace, child_dir, environment, emit, event):
        assert child_dir != run_dir and child_workspace != workspace
        assert read_json(child_dir / 'experiment_spec.json') == {'spec': candidate, 'sha256': spec_digest(candidate)}
        assert all((run_dir / name).read_bytes() == value for name, value in original_bytes.items())
        assert 'outcome_diagnosis' not in child
        calls.append(child_dir)
        child['execution'] = {'success': not interrupted, 'cancelled': interrupted}
        if interrupted:
            raise KeyboardInterrupt('controlled test cancellation')
        child['validation'] = {'status': 'ai_' + verdict,
            'reproduction_assessment': {'status': 'completed', 'verdict': verdict},
            'author_claim_comparison': {'status': 'partial'}}
    service = SimpleNamespace(root=tmp_path, _execute_frozen=execute)
    return locals()


@pytest.mark.parametrize('verdict,expected', [('reproduced', 'corrected'), ('not_reproduced', 'unresolved')])
def test_correction_freezes_new_attempt_and_uses_its_actual_result(tmp_path, monkeypatch, verdict, expected):
    case = runtime_case(tmp_path, monkeypatch, verdict=verdict)
    recover_outcome(case['service'], {}, case['data'], case['plan'], case['packet'], case['snapshot'],
                    case['workspace'], case['run_dir'], lambda event: None, lambda *args: None)
    data = case['data']
    assert case['reservations'] == ['reserved'] and len(case['calls']) == 1
    assert data['outcome_recovery']['status'] == expected
    assert data['validation']['status'] == 'ai_' + verdict
    assert data['grounded_repository_plan'] == case['corrected']
    assert data['active_attempt_dir'] == str(case['calls'][0])
    assert data['run_dir'] == str(case['run_dir'])
    assert all((case['run_dir'] / name).read_bytes() == value for name, value in case['original_bytes'].items())
    assert data['outcome_recovery']['numerical_equivalence_verified'] is False
    recover_outcome(case['service'], {}, data, case['plan'], case['packet'], case['snapshot'],
                    case['workspace'], case['run_dir'], lambda event: None, lambda *args: None)
    assert len(case['calls']) == 1


def test_cancelled_correction_keeps_cancellation_and_both_attempts(tmp_path, monkeypatch):
    case = runtime_case(tmp_path, monkeypatch, interrupted=True)
    with pytest.raises(KeyboardInterrupt):
        recover_outcome(case['service'], {}, case['data'], case['plan'], case['packet'], case['snapshot'],
                        case['workspace'], case['run_dir'], lambda event: None, lambda *args: None)
    data = case['data']
    assert data['execution']['cancelled'] is True
    assert data['outcome_recovery']['status'] == 'failed'
    assert read_json(case['run_dir'] / 'outcome_recovery.json')['attempts'][0]['status'] == 'failed'
    assert all((case['run_dir'] / name).read_bytes() == value for name, value in case['original_bytes'].items())


def test_optional_persistence_error_does_not_mask_cancellation(tmp_path, monkeypatch):
    case = runtime_case(tmp_path, monkeypatch, interrupted=True)
    from src import outcome_recovery_runtime as runtime
    real_write = runtime.write_json
    def fail_after_execution(path, value):
        if case['calls'] and Path(path).name == 'outcome_recovery.json':
            raise OSError('controlled disk write failure')
        real_write(path, value)
    monkeypatch.setattr(runtime, 'write_json', fail_after_execution)
    with pytest.raises(KeyboardInterrupt):
        recover_outcome(case['service'], {}, case['data'], case['plan'], case['packet'], case['snapshot'],
                        case['workspace'], case['run_dir'], lambda event: None, lambda *args: None)
    assert case['data']['execution']['cancelled'] is True
    assert case['data']['outcome_recovery']['persistence_errors']


def test_source_preparation_failure_keeps_completed_baseline_execution(tmp_path, monkeypatch):
    case = runtime_case(tmp_path, monkeypatch)
    from src import outcome_recovery_runtime as runtime
    def fail_export(*args, **kwargs):
        raise OSError('controlled source export failure')
    monkeypatch.setattr(runtime, 'export_repository', fail_export)
    with pytest.raises(OSError):
        recover_outcome(case['service'], {}, case['data'], case['plan'], case['packet'], case['snapshot'],
                        case['workspace'], case['run_dir'], lambda event: None, lambda *args: None)
    assert case['data']['execution']['success'] is True
    assert case['data']['validation']['status'] == 'ai_not_reproduced'
    assert 'active_attempt_dir' not in case['data']
    assert case['data']['outcome_recovery']['original_execution_preserved'] is True


def test_cancelled_event_before_export_still_finalizes_attempt_record(tmp_path, monkeypatch):
    case = runtime_case(tmp_path, monkeypatch)
    def cancel(*args):
        raise KeyboardInterrupt('cancel at progress event')
    with pytest.raises(KeyboardInterrupt):
        recover_outcome(case['service'], {}, case['data'], case['plan'], case['packet'], case['snapshot'],
                        case['workspace'], case['run_dir'], lambda event: None, cancel)
    saved = read_json(case['run_dir'] / 'outcome_recovery.json')
    assert saved['status'] == 'failed' and saved['attempts'][0]['interrupted'] is True
    assert not case['calls']


@pytest.mark.parametrize('cleanup_failed', [False, True])
@pytest.mark.parametrize('cancelled', [False, True])
def test_final_cleanup_synchronizes_canonical_and_attempt_results(tmp_path, cleanup_failed, cancelled):
    from src.environment_lifecycle import record_environment_cleanup
    root = tmp_path / 'root'
    child = root / 'outcome_recovery/attempt_1'
    child.mkdir(parents=True)
    data = {'run_dir': str(root), 'active_attempt_dir': str(child),
        'outcome_recovery': {'attempts': [{'id': 1, 'run_dir': str(child)}]},
        'execution': {'success': not cancelled, 'cancelled': cancelled}}
    result = {'state': 'ERROR' if cancelled else 'COMPLETED', 'error': 'interrupted' if cancelled else None, 'data': data}
    write_json(root / 'result.json', result)
    write_json(root / 'run_status.json', {'status': 'interrupted' if cancelled else 'completed'})
    write_json(child / 'run_status.json', {'status': 'interrupted' if cancelled else 'completed'})
    receipt = {'status': 'error' if cleanup_failed else 'released',
        'errors': [{'reason': 'controlled release failure'}] if cleanup_failed else []}
    record_environment_cleanup(result, receipt)
    assert read_json(root / 'result.json') == read_json(child / 'result.json')
    assert read_json(root / 'environment_cleanup.json') == read_json(child / 'environment_cleanup.json')
    assert read_json(root / 'run_status.json')['status'] == read_json(child / 'run_status.json')['status']
    assert (read_json(child / 'result.json')['state'] == 'ERROR') == (cancelled or cleanup_failed)
