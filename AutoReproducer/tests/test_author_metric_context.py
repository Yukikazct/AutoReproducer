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


NAMED_SOURCE = '''import torch
@torch.no_grad()
def evaluate(model, data):
    logits, accs = model(data.x), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
def main(args):
    results = []
    for run in range(args.runs):
        best_val_acc = test_acc = 0
        for epoch in range(args.epochs):
            train_acc, val_acc, tmp_test_acc = evaluate(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
        results.append(test_acc)
    results = 100 * torch.Tensor(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \\pm {results.std():.2f}')
'''
NAMED_LINE = r'Averaged test accuracy for 10 runs: 82.34 \pm 1.23'


def test_named_source_binds_final_scaled_array_and_real_standard_deviation():
    parsed, errors = parse_runtime_metrics(NAMED_LINE, preserve_contexts=True)
    assert not errors and parsed[0]['unit'] == '' and 'std' not in parsed[0]
    result = annotate_uncertainty(parsed, NAMED_SOURCE, 'main.py', 'a' * 64)[0]
    assert (result['value'], result['unit'], result['std'], result['aggregation'], result['runs']) == (
        82.34, 'percent', 1.23, 'mean', 10)
    assert result['source'] == 'stdout:1' and result['source_quote'] == NAMED_LINE
    assert 'reported_uncertainty' not in result and 'uncertainty_kind' not in result
    assert result['uncertainty_source'] == {'path': 'main.py', 'source_sha256': 'a' * 64,
        'print_line': 21, 'mean_expression': 'results.mean()', 'std_expression': 'results.std()'}
    assert result['unit_source']['kind'] == 'static_original_array_dataflow'
    assert parsed[0]['unit'] == '' and parsed[0]['uncertainty_kind'] == 'unknown'


@pytest.mark.parametrize('constructor', ['torch.Tensor(results)', 'torch.tensor(results)', 'np.array(results)', 'np.asarray(results)'])
@pytest.mark.parametrize('scale,unit', [('', 'fraction'), ('100 * ', 'percent')])
def test_named_array_unit_rule_applies_to_torch_numpy_and_final_scale(constructor, scale, unit):
    source = NAMED_SOURCE.replace('import torch', 'import torch\nimport numpy as np').replace(
        '100 * torch.Tensor(results)', scale + constructor)
    result = annotate_uncertainty(parse_runtime_metrics(NAMED_LINE)[0], source, 'main.py', 'a' * 64)[0]
    assert result['unit'] == unit and result['std'] == 1.23


@pytest.mark.parametrize('change', [
    lambda source: source.replace('results.std()', 'other.std()'),
    lambda source: source.replace('results.std()', 'results.var()'),
    lambda source: source.replace('results.mean():.2f} \\pm {results.std()', 'results.std():.2f} \\pm {results.mean()'),
    lambda source: source.replace('torch.Tensor(results)', 'unknown_array(results)'),
    lambda source: source.replace('import torch', 'import torch\ntorch.Tensor = unknown_array'),
    lambda source: source.replace('import torch', 'import torch\ntorch = unknown_module'),
    lambda source: source.replace('    results = 100 * torch.Tensor(results)',
        '    results = 100 * torch.Tensor(results)\n    results = unknown_values'),
    lambda source: source.replace('@torch.no_grad()', '@unknown_decorator()'),
    lambda source: source.replace('def main(args):', 'evaluate = unknown_evaluate\ndef main(args):'),
    lambda source: source.replace('results.mean():.2f', 'results.mean():.{precision}f'),
    lambda source: source.replace('    results = 100 * torch.Tensor(results)',
        '    alias = results\n    alias.append(unknown_metric())\n    results = 100 * torch.Tensor(results)'),
    lambda source: source.replace('    print(f\'Averaged', '    results[:] = unknown_metric()\n    print(f\'Averaged'),
])
def test_unknown_rebindings_functions_or_swapped_columns_do_not_prove_named_context(change):
    result = annotate_uncertainty(parse_runtime_metrics(NAMED_LINE)[0], change(NAMED_SOURCE), 'main.py', 'a' * 64)[0]
    assert result['unit'] == '' and 'unit_source' not in result
    # Unknown evaluator units can coexist with a genuine Torch array SD.
    if 'std' in result:
        assert result['std'] == 1.23 and 'uncertainty_source' in result
    else:
        assert result['reported_uncertainty'] == 1.23 and result['uncertainty_kind'] == 'unknown'


def test_unknown_metric_function_cannot_supply_the_final_printed_unit():
    source = NAMED_SOURCE.replace('pred.eq(data.y[mask]).sum().item() / mask.sum().item()', 'unknown_metric(pred)')
    result = annotate_uncertainty(parse_runtime_metrics(NAMED_LINE)[0], source, 'main.py', 'a' * 64)[0]
    assert result['unit'] == '' and 'unit_source' not in result


@pytest.mark.parametrize('tail', [' (95% CI)', ' SEM'])
def test_explicit_ci_or_sem_output_cannot_be_upgraded_to_std_by_source(tail):
    source = NAMED_SOURCE.replace("runs: {results.mean():.2f} \\pm {results.std():.2f}'",
                                  "runs: {results.mean():.2f} \\pm {results.std():.2f}" + tail + "'")
    parsed, errors = parse_runtime_metrics(NAMED_LINE + tail)
    result = annotate_uncertainty(parsed, source, 'main.py', 'a' * 64)[0]
    assert not errors and 'std' not in result and result['unit'] == ''


def test_named_source_binding_needs_one_unambiguous_matching_print():
    alternative = NAMED_SOURCE.replace('def main(args):', 'def alternative(args):').replace('results.std()', 'results.var()')
    result = annotate_uncertainty(parse_runtime_metrics(NAMED_LINE)[0], NAMED_SOURCE + alternative, 'main.py', 'a' * 64)[0]
    assert 'std' not in result and result['unit'] == ''


def test_same_valued_named_summaries_use_their_own_literal_split_and_scaled_array():
    source = NAMED_SOURCE.replace('    results = 100 * torch.Tensor(results)',
        '    fractions = torch.Tensor(results)\n    results = 100 * torch.Tensor(results)').replace(
        "    print(f'Averaged test accuracy", "    print(f'Averaged train accuracy for {args.runs} runs: {fractions.mean():.2f} \\pm {fractions.std():.2f}')\n    print(f'Averaged test accuracy")
    line = NAMED_LINE.replace('test', 'train') + '\n' + NAMED_LINE
    parsed, errors = parse_runtime_metrics(line, preserve_contexts=True)
    result = annotate_uncertainty(parsed, source, 'main.py', 'a' * 64)
    assert not errors and {(item['split'], item['unit'], item['std']) for item in result} == {
        ('train', 'fraction', 1.23), ('test', 'percent', 1.23)}


def test_source_context_never_overrides_an_explicit_output_unit():
    line = NAMED_LINE.replace('82.34', '82.34%')
    source = NAMED_SOURCE.replace('100 * torch.Tensor(results)', 'torch.Tensor(results)').replace(
        'results.mean():.2f} \\pm', 'results.mean():.2f}% \\pm')
    parsed, _ = parse_runtime_metrics(line)
    result = annotate_uncertainty(parsed, source, 'main.py', 'a' * 64)[0]
    assert result['unit'] == 'percent' and 'std' not in result and 'unit_source' not in result


def test_named_source_binding_is_hash_checked_by_the_production_collector(tmp_path):
    path = tmp_path / 'main.py'
    path.write_text(NAMED_SOURCE, encoding='utf-8')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {'steps': [{'id': 'train', 'entrypoint': {'source_path': 'main.py'}}]}
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0, 'stdout': NAMED_LINE}]}
    result, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan, sources={'main.py': sha})
    assert not errors and result[0]['unit'] == 'percent' and result[0]['std'] == 1.23
    assert result[0]['source'] == 'train:stdout:1' and result[0]['runs_source']['source'] == 'train:stdout:1'
    path.write_text(NAMED_SOURCE + '\n# changed source\n', encoding='utf-8')
    result, errors = collect_author_metrics(execution, workspace=tmp_path, plan=plan, sources={'main.py': sha})
    assert errors and result[0]['unit'] == '' and 'std' not in result[0]
