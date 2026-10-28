"""Automatic source scopes are derived from real failures, never entire repos."""
from copy import deepcopy
from pathlib import Path

import pytest

from src import source_repair as repair
from src.method_adapters import digest, read_json
from test_source_repair_service import CompatibilityLLM, MODEL_SOURCE, TRAIN_SOURCE, repair_case


def enable_auto(case, editable=()):
    case['request'].clear()
    case['request'].update(repair.normalize_source_repair_request({
        'pdf_path': 'fixture.pdf', 'source_repair': True, 'source_repair_auto': True,
        'source_repair_editable': list(editable)}))
    case['data']['source_repair']['editable'] = list(editable)


def test_normalization_accepts_explicit_auto_without_file_list_and_drops_disabled_scope():
    request = repair.normalize_source_repair_request({'pdf_path': 'fixture.pdf',
                'source_repair': True, 'source_repair_auto': True})
    assert request['source_repair_auto'] and request['source_repair_editable'] == []
    disabled = repair.normalize_source_repair_request({**request, 'source_repair': False})
    assert not disabled['source_repair_auto'] and disabled['source_repair_editable'] == []
    with pytest.raises(ValueError):
        repair.normalize_source_repair_request({**request, 'source_repair_auto': False})


@pytest.mark.parametrize('automatic', ['yes', 1, None, []])
def test_automatic_scope_requires_boolean_opt_in(automatic):
    with pytest.raises(ValueError, match='source_repair_auto'):
        repair.normalize_source_repair_request({'pdf_path': 'fixture.pdf', 'source_repair': True,
                                                'source_repair_auto': automatic})


@pytest.mark.parametrize('changes', [{'strict_validation': True}, {'mock_mode': True}, {'pdf_path': ''},
                                   {'code': 'print(1)'}, {'experiment_profile': 'preset'}])
def test_automatic_scope_retains_route_and_strict_mode_boundaries(changes):
    with pytest.raises(ValueError):
        repair.normalize_source_repair_request({'pdf_path': 'fixture.pdf', 'source_repair': True,
                                                'source_repair_auto': True, **changes})


@pytest.mark.parametrize('stream', ['stderr', 'stdout'])
def test_real_auto_repair_derives_two_file_scope_and_preserves_frozen_experiment(repair_case, stream):
    case = repair_case
    enable_auto(case)
    if stream == 'stdout':
        case['initial']['final']['stdout'] = case['initial']['final']['stderr']
        case['initial']['final']['stderr'] = ''
    original = deepcopy(case['data']['execution_plan'])
    request = deepcopy(case['request'])
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result['success'], result
    assert result['final']['stdout'].strip() == '7\n7'
    assert len(llm.calls) == 1
    assert 'AttributeError' in llm.calls[0]['diagnostic']
    assert set(llm.calls[0]['sources']) == {'model.py', 'train.py'}
    state = case['data']['source_repair']
    assert state['auto'] is True
    assert state['auto_selection']['missing_names'] == ['iteritems']
    assert set(state['editable']) == {'model.py', 'train.py'}
    assert state['auto_selection']['selected'] == state['editable']
    assert read_json(case['run_dir'] / 'source_repair.json')['auto_selection'] == state['auto_selection']
    assert case['request'] == request  # no new persistent permission added to the request
    assert case['data']['original_execution_plan'] == original
    assert case['data']['execution_plan']['steps'][0]['argv'] == original['steps'][0]['argv']
    case['verify']()
    validation = repair.verify_repaired_execution(case['plan'], case['data'], result)
    assert validation['source_modified'] and validation['paper_metrics_verified'] is False
    assert validation['is_reproduced'] is None


def test_auto_scope_does_not_expand_an_explicit_allowlist(repair_case):
    case = repair_case
    enable_auto(case, ['model.py'])
    llm = CompatibilityLLM(lambda *_: {'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()')})
    result = case['run'](llm)
    assert not result['success']
    state = case['data']['source_repair']
    assert state['editable'] == ['model.py']
    assert any(item['path'] == 'train.py' and item['reason'] == 'outside_explicit_scope'
               for item in state['auto_selection']['rejected'])
    assert all(set(context['sources']) == {'model.py'} for context in llm.calls)
    assert all(digest(Path(state['workspace']) / name) == sha for name, sha in case['snapshot']['files'].items())
    case['verify']()


def test_compatibility_fix_keeps_original_argument_taking_evaluator_call(repair_case):
    case = repair_case
    source = ('class Evaluator:\n    def eval(self, values):\n        return values\n'
              'class Model:\n    def value(self):\n'
              '        return Evaluator().eval(dict(answer=7)).iteritems()\n')
    (case['workspace'] / 'model.py').write_text(source, encoding='utf-8')
    case['snapshot']['files']['model.py'] = digest(case['workspace'] / 'model.py')
    latest = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    case['initial'].clear()
    case['initial'].update(latest)
    enable_auto(case)
    llm = CompatibilityLLM(lambda context, _: {name: value['content'].replace('.iteritems()', '.items()')
                                              for name, value in context['sources'].items()})
    result = case['run'](llm)
    assert result['success'], result
    assert result['final']['stdout'].strip() == '7\n7'
    case['verify']()


def staged_failure(case, next_path='adapter.py'):
    """A second source/API is reached only after the first real failure is fixed."""
    caller = ("from model import Model\nprint(list(Model().value())[0][1])\n"
              f"from {Path(next_path).stem} import values\nprint(list(values())[0][1])\n")
    following = "def values():\n    return dict(answer=7).viewitems()\n"
    (case['workspace'] / 'train.py').write_text(caller, encoding='utf-8')
    (case['workspace'] / next_path).write_text(following, encoding='utf-8')
    for name in ('train.py', next_path):
        case['snapshot']['files'][name] = digest(case['workspace'] / name)
    initial = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    assert not initial['success'] and 'iteritems' in initial['final']['stderr']
    case['initial'].clear()
    case['initial'].update(initial)
    return following


@pytest.mark.parametrize('following_name', ['adapter.py', 'worker_eval.py', 'test_runtime_helper.py'])
def test_auto_second_round_selects_new_traceback_source_after_real_compatibility_progress(repair_case, following_name):
    case = repair_case
    following = staged_failure(case, next_path=following_name)
    enable_auto(case)
    def propose(context, number):
        changes = {'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()')}
        if number == 1:
            assert set(context['sources']) == {'model.py'}
        else:
            assert 'viewitems' in context['diagnostic']
            assert 'iteritems' in context['initial_diagnostic']
            assert set(context['sources']) == {'model.py', following_name}
            assert context['sources']['model.py']['content'] == MODEL_SOURCE
            changes[following_name] = following.replace('.viewitems()', '.items()')
        return changes
    llm = CompatibilityLLM(propose)
    result = case['run'](llm)
    assert result['success'], result
    assert result['final']['stdout'].strip() == '7\n7'
    assert len(llm.calls) == 2
    state = case['data']['source_repair']
    assert state['editable'] == ['model.py', following_name]
    assert [item['selected'] for item in state['auto_selections']] == [['model.py'], [following_name]]
    assert {item['path'] for item in state['changed_files']} == {'model.py', following_name}
    assert read_json(case['run_dir'] / 'source_repair.json')['auto_selections'] == state['auto_selections']
    case['verify']()
    repair.verify_repaired_execution(case['plan'], case['data'], result)


def test_auto_second_round_keeps_manual_file_boundary_and_skips_unjustified_model_call(repair_case):
    case = repair_case
    staged_failure(case)
    enable_auto(case, ['model.py'])
    llm = CompatibilityLLM(lambda *_: {'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()')})
    result = case['run'](llm)
    assert not result['success'] and len(llm.calls) == 1
    state = case['data']['source_repair']
    assert state['editable'] == ['model.py']
    assert any(item['path'] == 'adapter.py' and item['reason'] == 'outside_explicit_scope'
               for item in state['auto_selections'][1]['rejected'])
    assert all(digest(Path(state['workspace']) / name) == sha for name, sha in case['snapshot']['files'].items())
    case['verify']()


def test_auto_cumulative_scope_cannot_exceed_three_files_across_rounds(repair_case):
    case = repair_case
    staged_failure(case)
    bridge = ("from model import Model\ndef invoke():\n"
              "    if False: dict(answer=7).iteritems()\n    return Model().value()\n")
    caller = ("from bridge import invoke\nif False: dict(answer=7).iteritems()\n"
              "print(list(invoke())[0][1])\nfrom adapter import values\nprint(list(values())[0][1])\n")
    for name, content in [('bridge.py', bridge), ('train.py', caller)]:
        (case['workspace'] / name).write_text(content, encoding='utf-8')
        case['snapshot']['files'][name] = digest(case['workspace'] / name)
    latest = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    case['initial'].clear()
    case['initial'].update(latest)
    enable_auto(case)
    llm = CompatibilityLLM(lambda *_: {'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()')})
    result = case['run'](llm)
    assert not result['success'] and len(llm.calls) == 1
    state = case['data']['source_repair']
    assert set(state['editable']) == {'model.py', 'bridge.py', 'train.py'}
    assert any(item['path'] == 'adapter.py' and item['reason'] == 'session_file_budget'
               for item in state['auto_selections'][1]['rejected'])
    assert all(digest(Path(state['workspace']) / name) == sha for name, sha in case['snapshot']['files'].items())
    case['verify']()


@pytest.mark.parametrize('flag,where', [('cancelled', 'top'), ('cancelled', 'final'),
                                     ('timed_out', 'top'), ('timed_out', 'final'),
                                     ('interrupted', 'top'), ('interrupted', 'final')])
def test_cancelled_or_timed_out_execution_never_derives_scope_or_calls_model(repair_case, flag, where):
    case = repair_case
    enable_auto(case)
    (case['initial'] if where == 'top' else case['initial']['final'])[flag] = True
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result is case['initial'] and not llm.calls
    assert case['data']['source_repair']['status'] == 'not_applicable'
    assert 'auto_selection' not in case['data']['source_repair']
    assert not (case['run_dir'] / 'repair_workspace').exists()


@pytest.mark.parametrize('flag', ['cancelled', 'timed_out', 'cleanup_unconfirmed', 'processes_alive'])
def test_stale_success_with_lifecycle_failure_cannot_skip_repair_gate(repair_case, flag):
    case = repair_case
    enable_auto(case)
    case['initial'].update(success=True)
    case['initial']['final'][flag] = True
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result['success'] is False and not llm.calls
    assert case['data']['source_repair']['status'] == 'not_applicable'
    assert not (case['run_dir'] / 'repair_workspace').exists()


@pytest.mark.parametrize('diagnostic', [
    "AttributeError: 'dict' object has no attribute 'iteritems'",  # missing traceback
    "Traceback (most recent call last):\nValueError: numerical value below reference",
    "Traceback (most recent call last):\nAttributeError: model has no attribute 'loss'",
    "Traceback (most recent call last):\nTypeError: model got an unexpected keyword argument 'seed'",
    "Traceback (most recent call last):\nAttributeError: obj has no attribute 'iteritems'\n"
    "Recovered optional API\nTraceback (most recent call last):\nRuntimeError: unrelated final failure",
])
def test_unsupported_or_quoted_failures_do_not_copy_sources_or_request_patches(repair_case, diagnostic):
    case = repair_case
    enable_auto(case)
    case['initial']['final']['stderr'] = diagnostic
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result is case['initial'] and not llm.calls
    assert case['data']['source_repair']['status'] == 'not_applicable'
    assert not (case['run_dir'] / 'repair_workspace').exists()


def selection_fixture(tmp_path, names, *, symbol='legacy_api', kind='attribute'):
    files, frames = {}, []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f'obj.{symbol}()\n' if kind == 'attribute' else f'call({symbol}=1)\n'
        path.write_text(content, encoding='utf-8')
        files[name] = digest(path)
        frames.append(f'  File "{path}", line 1, in run')
    exception = (f"AttributeError: 'object' object has no attribute '{symbol}'" if kind == 'attribute' else
                 f"TypeError: call() got an unexpected keyword argument '{symbol}'")
    execution = {'executed': True, 'success': False, 'final': {'exit_code': 1,
        'stderr': 'Traceback (most recent call last):\n' + '\n'.join(frames) + '\n' + exception}}
    return execution, {'files': files}


def test_auto_scope_selects_only_three_nearest_traceback_files(tmp_path):
    execution, snapshot = selection_fixture(tmp_path, [f'source_{index}.py' for index in range(5)])
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert selected['selected'] == ['source_4.py', 'source_3.py', 'source_2.py']
    assert [item['reason'] for item in selected['rejected']] == ['file_budget', 'file_budget']


@pytest.mark.parametrize('mutation,reason', [('hash', 'source_hash_changed'),
                                         ('symbol', 'missing_api_not_in_source'),
                                         ('protected', 'protected_or_oversized_source'),
                                         ('size', 'protected_or_oversized_source')])
def test_auto_scope_requires_frozen_allowed_python_source_containing_real_api(tmp_path, mutation, reason):
    name = 'data/model.py' if mutation == 'protected' else 'model.py'
    execution, snapshot = selection_fixture(tmp_path, [name])
    if mutation in {'hash', 'symbol'}:
        (tmp_path / name).write_text('obj.other_api()\n', encoding='utf-8')
        if mutation == 'symbol':
            snapshot['files'][name] = digest(tmp_path / name)
    elif mutation == 'size':
        (tmp_path / name).write_text('obj.legacy_api()\n' + '#' * 100000, encoding='utf-8')
        snapshot['files'][name] = digest(tmp_path / name)
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert not selected['selected'] and selected['rejected'][0]['reason'] == reason


def test_keyword_scope_uses_ast_keyword_evidence_not_a_comment(tmp_path):
    execution, snapshot = selection_fixture(tmp_path, ['caller.py'], kind='keyword')
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert selected['selected'] == ['caller.py'] and selected['compatibility_kind'] == 'keyword'
    (tmp_path / 'caller.py').write_text('# call(legacy_api=1)\ncall(current_api=1)\n', encoding='utf-8')
    snapshot['files']['caller.py'] = digest(tmp_path / 'caller.py')
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']


@pytest.mark.parametrize('symbol', ['random_state', 'learning_rate', 'dropout', 'n_estimators',
                                    'max_iter', 'num_layers', 'hidden_channels', 'momentum',
                                    'randomState', 'learningRate', 'numLayers', 'maxIter',
                                    'n_components', 'num_leaves', 'perplexity', 'tol', 'C', 'bagging_fraction'])
@pytest.mark.parametrize('kind', ['attribute', 'keyword'])
def test_hyperparameter_names_do_not_enter_automatic_compatibility_scope(tmp_path, symbol, kind):
    execution, snapshot = selection_fixture(tmp_path, ['model.py'], symbol=symbol, kind=kind)
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert not selected['selected'] and '协议敏感' in selected['reason']


def test_unlisted_traceback_files_never_enter_auto_scope(tmp_path):
    execution, snapshot = selection_fixture(tmp_path, ['known.py', 'not_frozen.py'])
    snapshot['files'].pop('not_frozen.py')
    assert repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected'] == ['known.py']


def test_recovered_traceback_followed_by_other_output_is_not_a_terminal_cause(tmp_path):
    execution, snapshot = selection_fixture(tmp_path, ['model.py'])
    execution['final']['stderr'] += '\nRecovered optional API; later processing failed without diagnostics'
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert not selected['selected'] and '最终失败' in selected['reason']


def test_real_auto_protocol_change_is_rejected_and_original_sources_stay_frozen(repair_case):
    case = repair_case
    enable_auto(case)
    llm = CompatibilityLLM(lambda context, number: {
        'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()'),
        'train.py': TRAIN_SOURCE.replace('.iteritems()', '.items()').replace('answer=7', 'answer=100')})
    result = case['run'](llm)
    assert not result['success'] and len(llm.calls) == 2
    assert all(item['status'] == 'rejected' for item in result['repair_attempts'])
    state = case['data']['source_repair']
    assert all(digest(Path(state['workspace']) / name) == sha for name, sha in case['snapshot']['files'].items())
    case['verify']()
