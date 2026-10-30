"""Frozen-source statistical context must not relabel confidence intervals as SD."""
import hashlib
from pathlib import Path

import pytest

from src.author_metric_context import annotate_uncertainty
from src.author_claim_comparison import collect_author_metrics, compare_author_claims
from src.runtime_metrics import parse_runtime_metrics


SOURCE = '''import numpy as np
import seaborn as sns
values = np.asarray(results, dtype=object)[:,0]
uncertainty = np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000),95)-values.mean()))
print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \\t val acc mean = {val_acc_mean:.4f}')
'''
LINE = 'test acc mean = 71.5974 ± 1.5536  \t val acc mean = 71.3407'


def records():
    parsed, errors = parse_runtime_metrics(LINE, preserve_contexts=True)
    assert not errors
    return parsed


def test_exact_bootstrap_half_width_retains_value_and_mean_but_is_not_std():
    original = records()
    result = annotate_uncertainty(original, SOURCE, 'train.py', 'a' * 64)
    test, validation = result
    assert test['value'] == 71.5974
    assert 'std' not in test
    assert test['reported_uncertainty'] == 1.5536
    assert test['uncertainty_kind'] == 'bootstrap_confidence_interval'
    assert test['confidence_level'] == 95
    assert test['source_quote'] == original[0]['source_quote']
    assert original[0]['std'] == 1.5536
    assert validation['value'] == 71.3407 and 'reported_uncertainty' not in validation


def test_real_std_definition_keeps_original_standard_deviation():
    source = SOURCE.replace('np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000),95)-values.mean()))',
                            'np.std(values)')
    assert annotate_uncertainty(records(), source, 'train.py', 'b' * 64)[0]['std'] == 1.5536


def test_unrelated_bootstrap_or_derived_statistic_does_not_establish_print_context():
    source = SOURCE.replace('uncertainty =', 'other_ci =').replace('{uncertainty*100:', '{std*100:')
    assert annotate_uncertainty(records(), source, 'train.py', 'b' * 64)[0]['std'] == 1.5536
    derived = SOURCE.replace('uncertainty = np.max', 'uncertainty = 2 * np.max')
    assert annotate_uncertainty(records(), derived, 'train.py', 'b' * 64)[0]['std'] == 1.5536


def test_value_mismatch_does_not_apply_source_context():
    observed = records()
    observed[0]['std'] = 2.0
    assert annotate_uncertainty(observed, SOURCE, 'train.py', 'a' * 64)[0]['std'] == 2.0


def test_collector_requires_original_source_hash_and_preserves_explicit_structured_std(tmp_path):
    source = tmp_path / 'train.py'
    source.write_text(SOURCE, encoding='utf-8')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = {'steps': [{'id': 'train', 'entrypoint': {'source_path': 'train.py'}}]}
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0, 'stdout': LINE}]}
    result, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan, sources={'train.py': expected})
    assert not errors and result[0]['reported_uncertainty'] == 1.5536 and 'std' not in result[0]
    comparison = compare_author_claims(None, {}, plan, tmp_path, execution, sources={'train.py': expected})
    assert comparison['metric_records'][0]['reported_uncertainty'] == 1.5536
    assert 'std' not in comparison['metric_records'][0]
    source.write_text(SOURCE + '\n# changed\n')
    unchanged, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan, sources={'train.py': expected})
    assert errors and unchanged[0]['std'] == 1.5536


def test_structured_metric_protocol_is_not_overridden_by_text_source(tmp_path):
    source = tmp_path / 'train.py'
    source.write_text(SOURCE, encoding='utf-8')
    values = [{'name': 'accuracy', 'value': 71.5974, 'std': 1.5536, 'unit': 'percent', 'split': 'test', 'stage': 'final'}]
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0, 'stdout': LINE, 'metric_records': values}]}
    plan = {'steps': [{'id': 'train', 'entrypoint': {'source_path': 'train.py'}}]}
    observed, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan,
        sources={'train.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    assert not errors and observed[0]['std'] == 1.5536 and 'uncertainty_kind' not in observed[0]


SUMMARY_SOURCE = '''import numpy as np
values = np.array(results)
print(f"{args.model}: AUC: {values[:, 0].mean():.4f} ({values[:, 0].std():.4f}) ")
print(f"{args.model}: MacroF1: {values[:, 1].mean():.4f} ({values[:, 1].std():.4f}) ")
'''
SUMMARY_OUTPUT = 'SGCN: AUC: 0.8730 (0.0098)\nSGCN: MacroF1: 0.6859 (0.0133)'


def test_parenthesized_statistics_require_matching_original_mean_std_source():
    parsed, _ = parse_runtime_metrics(SUMMARY_OUTPUT, preserve_contexts=True)
    result = annotate_uncertainty(parsed, SUMMARY_SOURCE, 'examples/train.py', 'a' * 64)
    assert len(result) == 2
    assert [(r['name'], r['value'], r['std'], r['aggregation']) for r in result] == [
        ('auc', 0.8730, 0.0098, 'mean'), ('f1_macro', 0.6859, 0.0133, 'mean')]
    assert all(r['uncertainty_source']['source_sha256'] == 'a' * 64 for r in result)
    assert all('runs' not in r for r in result)
    assert all('std' not in r for r in parsed)


@pytest.mark.parametrize('source', [
    SUMMARY_SOURCE.replace('.std()', '.mean()'),
    SUMMARY_SOURCE.replace('values[:, 0].std()', 'values[:, 1].std()'),
    SUMMARY_SOURCE.replace('AUC:', 'Accuracy:'),
    SUMMARY_SOURCE.replace('({values[:, 0].std():.4f})', '(CI {values[:, 0].std():.4f})'),
])
def test_parentheses_without_matching_statistical_evidence_do_not_become_std(source):
    parsed, _ = parse_runtime_metrics(SUMMARY_OUTPUT, preserve_contexts=True)
    assert 'std' not in annotate_uncertainty(parsed, source, 'train.py', 'a' * 64)[0]


def test_unlabelled_parentheses_remain_unknown_without_source():
    parsed, errors = parse_runtime_metrics(SUMMARY_OUTPUT, preserve_contexts=True)
    assert not errors and all('std' not in r and 'aggregation' not in r for r in parsed)


def test_parenthesized_summary_cannot_use_changed_source(tmp_path):
    path = tmp_path / 'train.py'
    path.write_text(SUMMARY_SOURCE, encoding='utf-8')
    plan = {'steps': [{'id': 'train', 'entrypoint': {'source_path': 'train.py'}}]}
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0, 'stdout': SUMMARY_OUTPUT}]}
    result, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan, sources={'train.py': 'b' * 64})
    assert errors and all('std' not in r for r in result)


@pytest.mark.parametrize('source,line,metric', [
    ('import numpy as arrays\nprint(f"Estimator-42: RMSE: {arrays.mean(errors):.3f} ({arrays.std(errors):.3f})")',
     'Estimator-42: RMSE: 1.250 (0.125)', 'rmse'),
    ('def evaluate(measurements):\n    print(f"Final MAE: {measurements.mean():.3f} ({measurements.std():.3f})")',
     'Final MAE: 1.250 (0.125)', 'mae'),
    ('print(f"Benchmark: latency={timings.mean():.3f} ({timings.std():.3f})")',
     'Benchmark: latency=1.250 (0.125)', 'latency'),
])
def test_statistical_rule_applies_to_unrelated_metrics_names_and_function_scopes(source, line, metric):
    records, errors = parse_runtime_metrics(line, expected_names=[metric])
    result = annotate_uncertainty(records, source, 'arbitrary/evaluate.py', 'c' * 64)
    assert not errors and len(result) == 1
    assert (result[0]['name'], result[0]['value'], result[0]['std'], result[0]['aggregation']) == (
        metric, 1.25, 0.125, 'mean')


def test_equal_metric_values_on_one_line_keep_their_own_deviation():
    source = 'print(f"AUC: {a.mean():.2f} ({a.std():.2f}); Macro-F1: {b.mean():.2f} ({b.std():.2f})")'
    parsed, _ = parse_runtime_metrics('AUC: 0.80 (0.01); Macro-F1: 0.80 (0.03)')
    result = annotate_uncertainty(parsed, source, 'report.py', 'd' * 64)
    assert {r['name']: r['std'] for r in result} == {'auc': 0.01, 'f1_macro': 0.03}


@pytest.mark.parametrize('spread', ['np.std(scores)', 'np.round(np.std(scores), 2)',
                                  'np.around(np.std(scores), 2)'])
def test_comma_summary_requires_matching_original_reductions(spread):
    source = 'import numpy as np\nprint(f"Test accuracy: {np.mean(scores)}, {' + spread + '}")'
    parsed, errors = parse_runtime_metrics('Test accuracy: 87.34, 1.19', preserve_contexts=True)
    assert not errors and 'std' not in parsed[0]
    result = annotate_uncertainty(parsed, source, 'evaluate.py', 'a' * 64)
    assert (result[0]['value'], result[0]['std'], result[0]['aggregation']) == (87.34, 1.19, 'mean')
    assert result[0]['uncertainty_source']['std_expression'] == spread
    assert 'runs' not in result[0] and 'std' not in parsed[0]


@pytest.mark.parametrize('spread', ['np.var(scores)', 'np.std(other_scores)',
    'np.round(np.std(scores), precision)', 'custom_round(np.std(scores), 2)',
    'np.clip(np.std(scores), 0, 2)'])
def test_comma_summary_cannot_guess_variance_or_transformed_uncertainty(spread):
    source = 'import numpy as np\nprint(f"Test accuracy: {np.mean(scores)}, {' + spread + '}")'
    parsed, _ = parse_runtime_metrics('Test accuracy: 87.34, 1.19', preserve_contexts=True)
    assert 'std' not in annotate_uncertainty(parsed, source, 'evaluate.py', 'a' * 64)[0]


def test_same_comma_output_from_std_and_other_statistic_is_ambiguous():
    source = ('import numpy as np\n'
              'def a(scores):\n    print(f"Test accuracy: {np.mean(scores)}, {np.round(np.std(scores), 2)}")\n'
              'def b(scores):\n    print(f"Test accuracy: {np.mean(scores)}, {np.var(scores)}")\n')
    parsed, _ = parse_runtime_metrics('Test accuracy: 87.34, 1.19', preserve_contexts=True)
    assert 'std' not in annotate_uncertainty(parsed, source, 'evaluate.py', 'a' * 64)[0]
