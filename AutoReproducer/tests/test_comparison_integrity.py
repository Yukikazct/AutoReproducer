"""Adversarial comparisons must not turn unrelated original text into proof.

These invented inputs exercise evidence semantics without a model, network,
training run, or paper-specific rule. A verifier may reject a contradictory
proposal or retain it as reference-only; neither permits an acceptance verdict.
"""
from copy import deepcopy
import hashlib

import pytest

from src.author_claim_comparison import validate_comparisons
from src.author_metric_context import annotate_uncertainty
from src.runtime_metrics import parse_runtime_metrics


PAPER = """Evaluation of Estimator on ToyGrid.
Method Dataset Split Accuracy (percent)
Estimator ToyGrid test 80.0
The score is a mean over 10 independent runs on the fixed held-out test split.
"""
SOURCE = """# Method: Estimator; dataset: ToyGrid.
# Evaluate the fixed held-out test split. Report mean over 10 independent runs.
# The dimensionless accuracy is a fraction.
print('Test accuracy over 10 runs: mean 0.80')
"""


def _case(tmp_path, paper=PAPER, source=SOURCE):
    """Use real hash-checked source and an immutable extracted-page packet."""
    workspace = tmp_path / 'author'
    workspace.mkdir()
    raw = source.encode('utf-8')
    (workspace / 'evaluate.py').write_bytes(raw)
    packet = {
        'pdf': {'sha256': 'a' * 64, 'pages': [
            {'page': 1, 'text': paper, 'sha256': hashlib.sha256(paper.encode()).hexdigest()}]},
        'repository': {'revision': 'b' * 40, 'files': {
            'evaluate.py': {'text': source, 'sha256': hashlib.sha256(raw).hexdigest()}}},
        'file_inventory': {},
    }
    records, errors = parse_runtime_metrics(
        'Test accuracy over 10 runs: mean 0.80', preserve_contexts=True)
    assert not errors and len(records) == 1
    records[0].update(id='evaluation:stdout:0', source='evaluation:stdout:1')
    payload = {
        'citations': [
            {'id': 'paper', 'source': 'pdf', 'page': 1, 'quote': paper},
            {'id': 'author', 'source': 'repository', 'path': 'evaluate.py', 'quote': source}],
        'comparisons': [{
            'actual_id': records[0]['id'], 'direction': 'higher',
            'reference': {
                'name': 'accuracy', 'value': 80.0, 'unit': 'percent',
                'dataset': 'ToyGrid', 'method': 'Estimator', 'split': 'test',
                'aggregation': 'mean', 'runs': 10,
                'value_quote': 'Estimator ToyGrid test 80.0', 'citations': ['paper']},
            'actual_context': {
                'unit': 'fraction', 'dataset': 'ToyGrid', 'method': 'Estimator',
                'split': 'test', 'aggregation': 'mean', 'runs': 10, 'citations': ['author']},
            'settings': {
                'status': 'matched', 'reason': 'Original evaluation settings match.',
                'citations': ['paper', 'author']}}],
    }
    return payload, packet, workspace, records


def _no_verdict(case, *, no_difference=False):
    """Both explicit rejection and conservative retention are safe outcomes."""
    payload, packet, workspace, records = case
    try:
        rows, _ = validate_comparisons(payload, packet, workspace, records)
    except ValueError:
        return
    assert len(rows) == 1
    assert rows[0]['status'] != 'comparable'
    assert rows[0]['within_tolerance'] is None
    assert rows[0]['meets_reference'] is None
    if no_difference:
        assert rows[0]['difference'] is None


def test_generic_settings_citations_cannot_prove_protocol_equivalence(tmp_path):
    case = _case(tmp_path)
    payload = case[0]
    # Both are genuine original citations, but neither describes the protocol.
    payload['citations'][0]['quote'] = 'Evaluation of Estimator on ToyGrid.'
    payload['citations'][1]['quote'] = '# Method: Estimator; dataset: ToyGrid.'
    payload['citations'].extend([
        {'id': 'results', 'source': 'pdf', 'page': 1, 'quote': PAPER},
        {'id': 'context', 'source': 'repository', 'path': 'evaluate.py', 'quote': SOURCE}])
    payload['comparisons'][0]['reference']['citations'] = ['results']
    payload['comparisons'][0]['actual_context']['citations'] = ['context']
    _no_verdict(case)


def test_epoch_count_is_not_a_reference_repetition_count(tmp_path):
    paper = PAPER.replace('10 independent runs', '10 optimizer epochs')
    case = _case(tmp_path, paper=paper)
    _no_verdict(case)


def test_epoch_count_cannot_complete_unobserved_actual_repetitions(tmp_path):
    source = (SOURCE.replace('mean over 10 independent runs', '10 optimizer epochs')
              .replace('Test accuracy over 10 runs: mean 0.80', 'Test accuracy: 0.80'))
    case = _case(tmp_path, source=source)
    records = case[3]
    records[0].pop('runs')
    records[0].pop('aggregation')
    _no_verdict(case)


def test_best_reference_cannot_be_relabelled_as_a_mean(tmp_path):
    paper = PAPER.replace('a mean over 10 independent runs', 'the best of 10 independent runs')
    case = _case(tmp_path, paper=paper)
    _no_verdict(case)


def test_reference_training_row_cannot_be_relabelled_as_test(tmp_path):
    paper = PAPER.replace('Estimator ToyGrid test 80.0', 'Estimator ToyGrid train 80.0')
    case = _case(tmp_path, paper=paper)
    case[0]['comparisons'][0]['reference']['value_quote'] = 'Estimator ToyGrid train 80.0'
    _no_verdict(case, no_difference=True)


def test_fraction_proposal_cannot_override_explicit_paper_percent(tmp_path):
    case = _case(tmp_path)
    case[0]['comparisons'][0]['reference']['unit'] = 'fraction'
    _no_verdict(case, no_difference=True)


def test_scalar_proposal_cannot_override_explicit_paper_percent(tmp_path):
    case = _case(tmp_path)
    case[0]['comparisons'][0]['reference']['unit'] = 'scalar'
    case[0]['comparisons'][0]['actual_context']['unit'] = 'scalar'
    _no_verdict(case, no_difference=True)


def test_model_column_flag_is_not_verification_of_same_row_multiple_metrics(tmp_path):
    paper = PAPER.replace(
        'Method Dataset Split Accuracy (percent)\nEstimator ToyGrid test 80.0',
        'Method Dataset Split Accuracy (percent) Recall (percent)\nEstimator ToyGrid test 80.0 90.0')
    case = _case(tmp_path, paper=paper)
    reference = case[0]['comparisons'][0]['reference']
    reference.update(value=90.0, value_quote='Estimator ToyGrid test 80.0 90.0', column_verified=True)
    # ToyGrid occurs in the row; that must not bypass metric-column grounding.
    _no_verdict(case)


@pytest.mark.parametrize('field,observed', [('dataset', 'OtherGrid'), ('method', 'OtherEstimator')])
def test_observed_metric_context_cannot_be_replaced_by_cited_default_context(tmp_path, field, observed):
    case = _case(tmp_path)
    # Structured measurements can explicitly carry their dataset and method.
    # A real README mentioning a default experiment cannot override those fields.
    case[3][0][field] = observed
    _no_verdict(case, no_difference=True)


def test_plus_suffixed_method_variant_cannot_be_merged_with_the_base_method(tmp_path):
    paper = PAPER.replace('Estimator', 'Estimator+')
    case = _case(tmp_path, paper=paper)
    case[0]['comparisons'][0]['reference'].update(
        method='Estimator+', value_quote='Estimator+ ToyGrid test 80.0')
    _no_verdict(case, no_difference=True)


@pytest.mark.parametrize('prefix', ['95% CI', 'standard error', 'SEM'])
def test_prefix_uncertainty_label_cannot_become_standard_deviation(prefix):
    records, errors = parse_runtime_metrics(f'{prefix} Test accuracy: 0.80 +/- 0.02')
    assert not errors and len(records) == 1
    assert records[0].get('std') is None
    assert records[0]['reported_uncertainty'] == 0.02
    assert records[0]['uncertainty_kind'] == (
        'confidence_interval' if prefix == '95% CI' else 'standard_error')
    assert records[0].get('aggregation') is None
    if prefix == '95% CI':
        assert records[0]['confidence_level'] == 95


@pytest.mark.parametrize('quantity', ['-0.02', 'nan', 'inf'])
def test_malformed_explicit_ci_uncertainty_is_not_a_valid_observation(quantity):
    records, errors = parse_runtime_metrics(f'95% CI Test accuracy: 0.80 +/- {quantity}')
    assert errors and len(records) == 1
    assert records[0]['value'] == 0.80
    assert records[0]['reported_uncertainty'] is None
    assert records[0].get('std') is None


def test_ambiguous_original_print_templates_do_not_establish_standard_deviation():
    source = '''def report_sd(values):
    print(f"Test accuracy: {values.mean():.2f} ({values.std():.2f})")

def report_sem(mean, standard_error):
    print(f"Test accuracy: {mean:.2f} ({standard_error:.2f})")

report_sem(0.80, 0.02)
'''
    records, errors = parse_runtime_metrics('Test accuracy: 0.80 (0.02)')
    assert not errors and 'std' not in records[0]
    result = annotate_uncertainty(records, source, 'evaluate.py', 'c' * 64)
    assert result[0].get('std') is None
    assert result[0]['value'] == records[0]['value']
    assert result[0]['source_quote'] == records[0]['source_quote']


@pytest.mark.parametrize('unproven_print', [
    'print("Test accuracy: 0.80 (0.02)")',
    'print(f"{message}")',
])
def test_untyped_matching_print_source_cannot_borrow_sd_from_another_helper(unproven_print):
    source = ('def other_helper(values):\n'
              '    print(f"Test accuracy: {values.mean():.2f} ({values.std():.2f})")\n'
              + unproven_print)
    records, _ = parse_runtime_metrics('Test accuracy: 0.80 (0.02)')
    result = annotate_uncertainty(records, source, 'metrics.py', 'd' * 64)
    assert result[0].get('std') is None
    assert 'std' not in records[0]


def test_unrelated_print_source_keeps_a_unique_original_mean_std_derivation():
    source = ('print(f"Progress: {message}")\n'
              'def evaluate(values):\n'
              '    print(f"Test accuracy: {values.mean():.2f} ({values.std():.2f})")\n')
    records, _ = parse_runtime_metrics('Test accuracy: 0.80 (0.02)')
    result = annotate_uncertainty(records, source, 'metrics.py', 'e' * 64)
    assert result[0]['value'] == 0.80
    assert result[0]['std'] == 0.02 and result[0]['aggregation'] == 'mean'
    assert result[0]['uncertainty_source']['source_sha256'] == 'e' * 64
    assert 'std' not in records[0]


def test_comparison_validation_never_mutates_the_source_proposal(tmp_path):
    case = _case(tmp_path)
    before = deepcopy(case[0])
    validate_comparisons(*case[:3], case[3])
    assert case[0] == before
