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
