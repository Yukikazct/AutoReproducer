"""Source-bound restoration of old NumPy heterogeneous result-table reductions."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.experiments.author_numeric_compatibility import diagnose_ragged_summary, run_entrypoint
from test_discovered_repository_service import cpu_dependency_cache


SOURCE = '''import numpy as np
results = [[.7, .8, np.array([1., 2.], dtype=np.float32)],
           [.9, .6, np.array([3., 4.], dtype=np.float32)]]
original = [row[2].copy() for row in results]
test_mean, val_mean, weights_mean = np.mean(results, axis=0)
test_var, val_var, weights_var = np.var(results, axis=0)
assert all(np.array_equal(a, row[2]) for a, row in zip(original, results))
'''


def failed_summary(tmp_path, source=SOURCE):
    script = tmp_path / 'train.py'
    script.write_text(source, encoding='utf-8')
    sources = {script.name: hashlib.sha256(script.read_bytes()).hexdigest()}
    try:
        run_entrypoint(script, [], tmp_path, sources)
    except ValueError as error:
        return script, sources, diagnose_ragged_summary(error, script, tmp_path, sources)
    pytest.fail('The original heterogeneous reduction must genuinely fail')


def test_scalar_and_vector_means_and_variances_preserve_legacy_statistics(tmp_path):
    script, sources, fix = failed_summary(tmp_path)
    assert fix['kind'] == 'numpy_ragged_summary'
    assert [call['operation'] for call in fix['calls']] == ['mean', 'var']
    before = script.read_bytes()
    result = run_entrypoint(script, [fix], tmp_path, sources)
    np.testing.assert_allclose([result['test_mean'], result['val_mean']], [.8, .7], rtol=0, atol=1e-14)
    np.testing.assert_allclose([result['test_var'], result['val_var']], [.01, .01], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result['weights_mean'], [2., 3.])
    np.testing.assert_array_equal(result['weights_var'], [1., 1.])
    assert script.read_bytes() == before


def test_changed_source_cannot_receive_the_recorded_summary_repair(tmp_path):
    script, sources, fix = failed_summary(tmp_path)
    script.write_text(script.read_text() + '\nprint("changed")\n')
    with pytest.raises(ValueError, match='pinned entrypoint'):
        run_entrypoint(script, [fix], tmp_path, sources)


def test_inconsistent_vector_shapes_remain_an_original_error(tmp_path):
    source = SOURCE.replace('np.array([3., 4.]', 'np.array([3., 4., 5.]')
    _, _, fix = failed_summary(tmp_path, source)
    assert fix is None


def test_other_reduction_axis_is_not_automatically_changed(tmp_path):
    _, _, fix = failed_summary(tmp_path, SOURCE.replace('axis=0', 'axis=1'))
    assert fix is None


def test_structure_change_between_recorded_reductions_is_rejected(tmp_path):
    source = SOURCE.replace('test_var, val_var, weights_var =',
                            "results = [[1., 2., np.array([1., 2., 3.])], [3., 4., np.array([4., 5., 6.])]]\n"
                            'test_var, val_var, weights_var =')
    script, sources, fix = failed_summary(tmp_path, source)
    with pytest.raises(ValueError, match='changed shape or dtype'):
        run_entrypoint(script, [fix], tmp_path, sources)


def test_explicit_dtype_and_other_keywords_are_not_overridden(tmp_path):
    _, _, fix = failed_summary(tmp_path, SOURCE.replace('axis=0', 'axis=0, dtype=np.float64'))
    assert fix is None


def test_same_named_function_argument_keeps_its_original_regular_reduction(tmp_path):
    source = SOURCE + '\ndef ordinary(results):\n    return np.mean(results, axis=0)\nregular = ordinary([[1., 2.], [3., 4.]])\n'
    script, sources, fix = failed_summary(tmp_path, source)
    assert len(fix['calls']) == 2
    result = run_entrypoint(script, [fix], tmp_path, sources)
    np.testing.assert_array_equal(result['regular'], [2., 3.])


def test_full_service_can_repair_post_training_summary_and_complete_original_command(
        tmp_path, monkeypatch, cpu_dependency_cache):
    from test_author_execution_plan import AuthorLLM
    from test_discovered_repository_service import setup_service, use_preinstalled_cache

    fixture, service, request, _ = setup_service(tmp_path, monkeypatch)
    script = fixture['workspace'] / 'train.py'
    source = script.read_text('utf-8') + '\n' + SOURCE
    script.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    service.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, service, cpu_dependency_cache)
    result = service.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert len(data['numeric_repair_attempts']) == 1
    assert data['numeric_repair_attempts'][0]['kind'] == 'numpy_ragged_summary'
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert data['validation']['paper_metrics_verified'] is False
    assert hashlib.sha256((Path(data['run_dir']) / 'repo/train.py').read_bytes()).hexdigest() == fixture['snapshot']['files']['train.py']
