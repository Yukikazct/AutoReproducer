"""Actual author retry failures are distinct from model formatting retries."""
from copy import deepcopy

from src.benchmark.recovery_adapter import (_MeteredLLM, _measurement,
    _execution_failure_identity, _record_author_execution)
from test_source_repair_service import repair_case


def test_repeated_actual_author_failure_is_counted_and_success_resets_identity(repair_case):
    case = repair_case
    meter = _MeteredLLM(object())
    _record_author_execution(meter, case['initial'])
    repeat = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    _record_author_execution(meter, repeat)
    assert meter.same_failure_retries == 1
    restored = case['run']()
    assert restored['success'] is True
    _record_author_execution(meter, restored)
    _record_author_execution(meter, case['initial'])
    assert meter.same_failure_retries == 1
    usage = _measurement(meter)
    assert usage['same_failure_retries'] == 1
    assert usage['invalid_retries'] == 0
    assert usage['execution_retries'] == 3
    assert meter.author_runs == 4


def test_environment_rejection_does_not_count_as_actual_author_retry():
    meter = _MeteredLLM(object())
    actual = {'executed': True, 'success': False, 'final': {
        'id': 'train', 'stderr': 'ModuleNotFoundError: missing_package'}}
    rejected = {**deepcopy(actual), 'executed': False}
    _record_author_execution(meter, rejected)
    assert meter.author_runs == meter.execution_retries == meter.same_failure_retries == 0
    _record_author_execution(meter, actual)
    _record_author_execution(meter, rejected)
    _record_author_execution(meter, actual)
    assert meter.author_runs == 2
    assert meter.execution_retries == meter.same_failure_retries == 1


def test_abruptly_interrupted_author_invocation_keeps_retry_counts_unknown():
    meter = _MeteredLLM(object())
    meter.author_execution_in_flight = True
    observed = _measurement(meter)
    assert observed['execution_retries'] is observed['same_failure_retries'] is None
    assert observed['model_requests'] == 0


def test_changed_failure_site_or_original_step_is_not_the_same_failed_retry():
    first = {'success': False, 'final': {'id': 'train', 'cwd': '/author',
        'stderr': 'Traceback (most recent call last):\n  File "/author/models/model.py", line 7, in run\nValueError: invalid input'}}
    second = deepcopy(first)
    second['final']['stderr'] = second['final']['stderr'].replace('/models/', '/datasets/')
    assert _execution_failure_identity(first) != _execution_failure_identity(second)
    second = deepcopy(first)
    second['final']['id'] = 'evaluate'
    assert _execution_failure_identity(first) != _execution_failure_identity(second)


def test_different_repair_copy_root_is_not_counted_as_a_new_error():
    first = {'success': False, 'final': {'id': 'train', 'cwd': '/author',
        'stderr': 'Traceback (most recent call last):\n  File "/author/models/model.py", line 7, in run\nValueError: invalid input'}}
    second = deepcopy(first)
    second['final']['cwd'] = '/candidate'
    second['final']['stderr'] = second['final']['stderr'].replace('/author/', '/candidate/')
    assert _execution_failure_identity(first) == _execution_failure_identity(second)


def test_workspace_path_in_terminal_exception_does_not_disguise_same_failure():
    first = {'success': False, 'final': {'id': 'train', 'cwd': '/author',
        'stderr': 'Traceback (most recent call last):\n  File "/author/train.py", line 7, in run\n'
                  "FileNotFoundError: [Errno 2] No such file: '/author/data/input.json'"}}
    second = deepcopy(first)
    second['final']['cwd'] = '/candidate'
    second['final']['stderr'] = second['final']['stderr'].replace('/author/', '/candidate/')
    assert _execution_failure_identity(first) == _execution_failure_identity(second)
