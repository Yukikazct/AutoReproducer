"""Cleanup evidence must stop recovery and invalidate stale report success."""
from copy import deepcopy

import pytest

from src.benchmark.recovery_adapter import RecoveryAdapterCleanup
from src.intelligent_recovery import RecoverySession
from src.repository_repairer import cleanup_is_unconfirmed, repair_is_stopped
from src.result_conclusion import build_reproduction_conclusion
from test_intelligent_recovery import ScriptedModel, experiment
from test_result_conclusion import accepted


def nested_record(wrapper, value):
    record = {'cleanup_receipt': value}
    return {wrapper: [record] if wrapper in {'steps', 'attempts', 'stages'} else record}


@pytest.mark.parametrize('wrapper', ['environment', 'effective_env_config', 'final',
                                    'steps', 'attempts', 'stages'])
@pytest.mark.parametrize('receipt', [{'confirmed': False}, {'status': 'unconfirmed'}])
def test_nested_cleanup_evidence_blocks_every_recovery_and_the_report(wrapper, receipt, tmp_path, experiment):
    data = accepted()
    data['execution'].update(nested_record(wrapper, receipt))
    original = deepcopy(data)
    assert cleanup_is_unconfirmed(data['execution']) is True
    assert repair_is_stopped(data['execution']) is True
    assert RecoveryAdapterCleanup.unconfirmed(data['execution']) is True
    workspace, packet, plan, _ = experiment
    model = ScriptedModel(lambda *_: pytest.fail('Unconfirmed process cleanup must prevent model calls'))
    session = RecoverySession(model)
    session.attach({}, tmp_path)
    task = session.investigate(trigger='result_discrepancy', plan=plan, packet=packet,
        workspace=workspace, environment={}, execution=data['execution'])
    assert task is None
    assert session.state['cleanup_confirmed'] is False
    assert not model.calls
    conclusion = build_reproduction_conclusion(data)
    assert conclusion['status'] == 'cleanup_unconfirmed'
    assert conclusion['is_reproduced'] is None
    assert data == original


def test_same_dictionary_in_execution_and_receipt_keeps_its_receipt_meaning():
    shared = {'confirmed': False}
    execution = {'final': shared, 'cleanup_receipt': shared}
    assert cleanup_is_unconfirmed(execution) is True
    assert RecoveryAdapterCleanup.unconfirmed(execution) is True


def test_successful_cleanup_and_failed_author_assertions_are_distinct():
    execution = {'success': False, 'final': {'exit_code': 1, 'status': 'failed'},
        'environment': {'cleanup_receipt': {'confirmed': True}},
        'attempts': [{'cleanup_receipt': {'status': 'confirmed'}}]}
    assert cleanup_is_unconfirmed(execution) is False
    assert repair_is_stopped(execution) is False
    assert RecoveryAdapterCleanup.unconfirmed(execution) is False


def test_cyclic_execution_wrappers_do_not_hide_cleanup_or_loop_forever():
    execution = {'success': True}
    execution['environment'] = execution
    execution['attempts'] = [execution]
    assert cleanup_is_unconfirmed(execution) is False
    assert RecoveryAdapterCleanup.unconfirmed(execution) is False
    execution['cleanup_receipt'] = {'confirmed': False}
    assert cleanup_is_unconfirmed(execution) is True
    assert RecoveryAdapterCleanup.unconfirmed(execution) is True


@pytest.mark.parametrize('wrapper', ['environment', 'attempts', 'stages'])
@pytest.mark.parametrize('flag', ['timed_out', 'cancelled'])
def test_nested_process_stop_prevents_further_recovery(wrapper, flag):
    data = accepted()
    stopped = {flag: True}
    data['execution'][wrapper] = [stopped] if wrapper in {'attempts', 'stages'} else stopped
    assert repair_is_stopped(data['execution']) is True
    assert cleanup_is_unconfirmed(data['execution']) is False


def test_confirmed_optional_step_timeout_does_not_replace_required_completion():
    data = accepted()
    optional = {'id': 'optional', 'required': False, 'executed': True, 'success': False,
                'exit_code': 124, 'timed_out': True, 'cleanup_confirmed': True}
    required = {'id': 'evaluate', 'required': True, 'executed': True, 'success': True,
                'exit_code': 0}
    data['execution'].update(steps=[optional, required], attempts=[optional, required], final=required)
    conclusion = build_reproduction_conclusion(data)
    assert conclusion['execution']['status'] == 'completed'
    assert conclusion['is_reproduced'] is True
    optional['cleanup_confirmed'] = False
    conclusion = build_reproduction_conclusion(data)
    assert conclusion['status'] == 'cleanup_unconfirmed'
    assert conclusion['is_reproduced'] is None
