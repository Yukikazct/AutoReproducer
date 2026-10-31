"""Field meanings are verified independently of model-provided status flags."""
from copy import deepcopy

import pytest

from src.author_comparison_semantics import context_evidence, reference_uncertainty, settings_evidence


def evidence(text, **node):
    return context_evidence({'name': 'accuracy', 'unit': 'percent', 'split': 'test',
                             'aggregation': 'mean', 'runs': 7, **node},
                            [{'id': 'original', 'source': 'pdf', 'quote': text}])


@pytest.mark.parametrize('count', ['7 runs', '7 independent trials', 'runs=7', '7 random seeds', 'number of seeds: 7'])
def test_only_explicit_repetition_counts_bind_the_runs_field(count):
    check = evidence('We report mean test accuracy over ' + count + '.')['runs']
    assert check['status'] == 'verified' and check['value'] == 7


@pytest.mark.parametrize('literal,count', [('ten', 10), ('seven', 7), ('twenty-five', 25), ('thirty two', 32)])
def test_english_number_words_are_bound_to_repetition_context(literal, count):
    check = evidence('We report mean test accuracy over ' + literal + ' runs.', runs=count)['runs']
    assert check['status'] == 'verified' and check['value'] == count
    assert evidence('We train for ' + literal + ' epochs.', runs=count)['runs']['status'] == 'unknown'
    assert evidence('We use ' + literal + ' runs.', runs=count + 1)['runs']['status'] == 'contradicted'


@pytest.mark.parametrize('count', ['7 epochs', '7 optimizer updates', 'batch_size=7', 'seed=7'])
def test_training_counts_and_one_seed_value_do_not_prove_repetitions(count):
    assert evidence('Mean accuracy. ' + count)['runs']['status'] == 'unknown'


@pytest.mark.parametrize('statement', [
    'seeds: 7', 'seeds: 7, 11, 21', 'seeds: 7 and 11', 'runs: 7/10',
    'runs: 7.5', '1.7 runs', '-7 runs',
])
def test_seed_identifiers_progress_and_noninteger_counts_do_not_prove_repetitions(statement):
    assert evidence('Mean test accuracy; ' + statement)['runs']['status'] == 'unknown'


def test_two_conflicting_repetition_protocols_require_a_selected_exact_scope():
    text = 'Method A uses 7 runs.\nMethod B uses 12 runs.'
    assert evidence(text)['runs']['status'] == 'unknown'
    check = evidence(text, context_quotes={'runs': 'Method A uses 7 runs.'})['runs']
    assert check['status'] == 'verified'
    with pytest.raises(ValueError, match='own original citations'):
        evidence(text, context_quotes={'runs': 'Method A uses 5 runs.'})


@pytest.mark.parametrize('text,count', [
    ('We run each experiment 100 times with multiple random splits and different initializations.', 100),
    ('We repeated the experiment seven times.', 7),
    ('Each experiment was repeated 7 times.', 7),
    ('The experiment is run seven times.', 7),
    ('We conduct each experiment 7 times.', 7),
    ('We perform every experiment twenty-five times.', 25),
    ('We run each experiment 100\ntimes with different initializations.', 100),
])
def test_explicit_experiment_times_bind_counts_and_preserve_original_quote(text, count):
    check = evidence(text, runs=count)['runs']
    assert check['status'] == 'verified' and check['value'] == count
    assert check['quote'] in text and 'times' in check['quote']
    assert evidence(text, runs=count + 1)['runs']['status'] == 'contradicted'


@pytest.mark.parametrize('text', [
    'We use at most 7 runs.', 'We use at least 7 runs.',
    'We use up to 7 runs.', 'We use approximately 7 trials.',
    'We use approx. 7 runs.',
    'We use around seven runs.', 'We use more than 7 runs.',
    'We do not use 7 runs.', 'We never use seven runs.',
    'We use 5 to 7 runs.', 'We use 5-7 runs.', 'We use 3/7 runs.',
    'We use between five and seven runs.', 'We use 5 or 7 runs.',
    'We use 7 runs or fewer.', 'We use 7 runs at most.', 'runs: 7+',
    'We will run each experiment 7 times.',
    'We did not run each experiment 7 times.',
    'We run each experiment at most 7 times.',
    'Each experiment was not repeated 7 times.',
    'We run each experiment 5 to 7 times.',
    'We run each experiment 7 times faster.',
    'We run each experiment 7 times per epoch.', 'We use 7 runs if time permits.',
    'We train for 7 epochs.', 'We repeat each minibatch 7 times.',
    'We report accuracy over 7 random splits.',
])
def test_qualified_ranges_negation_and_other_counts_do_not_become_exact_runs(text):
    assert evidence(text)['runs']['status'] == 'unknown'


def test_conflicting_experiment_times_need_an_exact_selected_scope():
    first = 'We run each experiment 7 times.'
    second = 'Each experiment was repeated 10 times.'
    text = first + '\n' + second
    assert evidence(text)['runs']['status'] == 'unknown'
    check = evidence(text, context_quotes={'runs': first})['runs']
    assert check['status'] == 'verified' and check['value'] == 7


def test_qualifier_in_another_sentence_does_not_disqualify_explicit_runs():
    assert evidence('We use at most 200 epochs. We report accuracy over 7 runs.')['runs']['status'] == 'verified'


@pytest.mark.parametrize('text,quote,count', [
    ('We use at most 7 runs.', '7 runs', 7),
    ('We did not run each experiment 7 times.', 'run each experiment 7 times', 7),
    ('We did not perform a single run.', 'a single run', 1),
    ('We use 7 runs or fewer.', '7 runs', 7),
    ('We run each experiment 7 times faster.', 'run each experiment 7 times', 7),
    ('We run each experiment 7 times per epoch.', 'run each experiment 7 times', 7),
    ('We use at most 7 runs. A different method uses 7 runs.', '7 runs', 7),
])
def test_cropped_context_cannot_hide_repetition_qualifiers(text, quote, count):
    assert evidence(text, runs=count, context_quotes={'runs': quote})['runs']['status'] == 'unknown'


def test_context_quote_can_recover_only_original_whitespace():
    original = 'We report mean test accuracy over 7\nruns.'
    check = evidence(original, context_quotes={'runs': 'We report mean test accuracy over 7 runs.'})['runs']
    assert check['status'] == 'verified'
    assert check['quote'] == original
    with pytest.raises(ValueError, match=r'context_quotes.runs.*own original citations'):
        evidence(original, context_quotes={'runs': 'We report mean test accuracy over 8 runs.'})


def test_context_quote_recovers_escaped_newline_and_keeps_original_evidence():
    original = 'We report mean test accuracy over 7\nruns.'
    proposed = {'runs': r'We report mean test accuracy over 7\nruns.'}
    check = evidence(original, context_quotes=proposed)['runs']
    assert check['status'] == 'verified'
    assert check['quote'] == original
    assert proposed['runs'] == r'We report mean test accuracy over 7\nruns.'


def test_only_pdf_context_can_recover_printing_hyphenation():
    original = 'The mean clas-\nsification accuracy over 7 runs.'
    node = {'name': 'accuracy', 'aggregation': 'mean',
            'context_quotes': {'aggregation': 'The mean classification accuracy over 7 runs.'}}
    refs = [{'id': 'source', 'source': 'pdf', 'quote': original}]
    result = context_evidence(node, refs)['aggregation']
    assert result['status'] == 'verified'
    assert result['quote'] in original
    refs[0]['source'] = 'repository'
    with pytest.raises(ValueError, match='own original citations'):
        context_evidence(node, refs)


@pytest.mark.parametrize('quoted,proposed', [('best of 7 runs', 'mean'), ('mean over 7 runs', 'single')])
def test_one_statistic_cannot_be_relabelled_as_another(quoted, proposed):
    check = evidence('We report test accuracy: ' + quoted, aggregation=proposed)['aggregation']
    assert check['status'] == 'contradicted'


def test_sample_accuracy_mean_does_not_prove_a_mean_over_experiments():
    assert evidence('accuracy = correct.float().mean().item()')['aggregation']['status'] == 'unknown'


def test_explicit_paper_unit_is_not_overridden_by_a_model_scale():
    check = evidence('Test Accuracy (percent)', unit='fraction')['unit']
    assert check['status'] == 'contradicted' and check['value'] == 'percent'


def test_no_unit_can_be_inferred_from_only_the_magnitude():
    assert evidence('Test Accuracy 93.1')['unit']['status'] == 'unknown'
    assert evidence('Test Accuracy 0.931', unit='fraction')['unit']['status'] == 'unknown'


@pytest.mark.parametrize('text,metric,unit', [
    ('Accuracy (%)', 'auc', 'percent'),
    ('Accuracy (fraction)', 'mse', 'fraction'),
    ('Accuracy (%) and Macro-F1', 'f1_macro', 'percent'),
    ('AUC (fraction) and RMSE', 'rmse', 'fraction'),
])
def test_another_metric_cannot_supply_the_selected_metrics_unit(text, metric, unit):
    assert evidence(text, name=metric, unit=unit)['unit']['status'] == 'unknown'


@pytest.mark.parametrize('metric,unit', [('accuracy', 'percent'), ('f1_macro', 'fraction')])
def test_distinct_metric_units_remain_bound_to_their_own_header_segment(metric, unit):
    text = 'Accuracy (%) and Macro-F1 (fraction)'
    check = evidence(text, name=metric, unit=unit)['unit']
    assert check['status'] == 'verified' and check['value'] == unit


@pytest.mark.parametrize('expression,unit', [
    ('(predictions == labels).float().mean().item()', 'fraction'),
    ('100 * (predictions == labels).float().mean().item()', 'percent'),
    ('predictions.eq(labels).sum().item() / mask.sum().item()', 'fraction'),
    ('(predictions == labels).float().mean().item() * 100', 'percent'),
])
def test_original_boolean_accuracy_arithmetic_supplies_its_scale(expression, unit):
    check = evidence('accuracy = ' + expression, unit=unit)['unit']
    assert check['status'] == 'verified' and check['value'] == unit
    assert check['reason'] == 'Static original metric arithmetic'


def test_arbitrary_tensor_mean_or_numeric_constant_cannot_prove_an_accuracy_unit():
    for source in ['accuracy = values.mean().item()', 'accuracy = 99.5', 'accuracy = 0.995']:
        assert evidence(source)['unit']['status'] == 'unknown'


def test_mean_reduction_aliases_are_scoped_and_conflicting_units_stay_unknown():
    text = ('def a():\n    accuracy = (p == y).float().mean()\n'
            'def b():\n    accuracy = 100 * (p == y).float().mean()\n')
    assert evidence(text)['unit']['status'] == 'unknown'


@pytest.mark.parametrize('later', [
    'accuracy = custom_scale(accuracy)',
    'accuracy *= 100',
    'accuracy = unknown_output',
])
def test_stale_accuracy_arithmetic_does_not_prove_the_final_unit(later):
    text = 'accuracy = (p == y).float().mean()\n' + later
    assert evidence(text, unit='fraction')['unit']['status'] == 'unknown'


def test_rebound_metric_import_cannot_supply_a_library_unit_contract():
    text = ('from sklearn.metrics import accuracy_score as score\n'
            'score = custom_count\naccuracy = score(y, p)')
    assert evidence(text, unit='fraction')['unit']['status'] == 'unknown'


def test_a_selected_arithmetic_quote_cannot_hide_later_rebinding_in_its_complete_source():
    first = 'accuracy = (p == y).float().mean()'
    text = first + '\naccuracy = custom_scale(accuracy)'
    assert evidence(text, unit='fraction', context_quotes={'unit': first})['unit']['status'] == 'unknown'


def test_standard_metric_library_contract_needs_an_actual_import():
    source = 'from sklearn.metrics import accuracy_score as score\naccuracy = score(y, p)'
    assert evidence(source, unit='fraction')['unit']['status'] == 'verified'
    assert evidence('accuracy = accuracy_score(y, p)', unit='fraction')['unit']['status'] == 'unknown'


@pytest.mark.parametrize('declaration,call', [
    ('from sklearn import metrics', 'metrics.accuracy_score'),
    ('import sklearn.metrics as evaluation', 'evaluation.accuracy_score'),
    ('import sklearn as library', 'library.metrics.accuracy_score'),
])
def test_qualified_metric_contracts_require_their_original_imports(declaration, call):
    source = declaration + '\naccuracy = ' + call + '(y, p)'
    assert evidence(source, unit='fraction')['unit']['status'] == 'verified'
    assert evidence('accuracy = ' + call + '(y, p)', unit='fraction')['unit']['status'] == 'unknown'
    assert evidence(source.replace('(y, p)', '(y, p, normalize=False)'), unit='fraction')['unit']['status'] == 'unknown'


def test_auc_library_contract_supports_an_original_score_variable():
    source = 'from sklearn import metrics\nauc_score = metrics.roc_auc_score(y, p)'
    assert evidence(source, name='auc', unit='fraction')['unit']['status'] == 'verified'


def test_authoritative_runtime_fields_override_proposed_context_and_fake_flags():
    node = {'name': 'accuracy', 'unit': 'fraction', 'split': 'test', 'aggregation': 'mean', 'runs': 7,
            'field_evidence': {'runs': {'status': 'verified'}}}
    original = deepcopy(node)
    checks = context_evidence(node, [{'id': 'x', 'quote': 'Only the metric name: Accuracy.'}],
                              observed={'runs': 7, 'aggregation': 'mean', 'source': 'stdout:1'})
    assert checks['runs']['status'] == checks['aggregation']['status'] == 'verified'
    assert checks['unit']['status'] == 'unknown'
    assert node == original


def test_generic_method_titles_do_not_prove_experimental_settings():
    refs = {'a': {'id': 'a', 'source': 'pdf', 'quote': 'Evaluation of Method A on Dataset B.'},
            'b': {'id': 'b', 'source': 'repository', 'quote': 'Method A; Dataset B.'}}
    check = settings_evidence({'status': 'matched', 'citations': ['a', 'b']}, refs)
    assert check['protocol_context_present'] is False
    assert check['independently_verified'] is False
    assert check['assessment_basis'] == 'model_with_citations'


@pytest.mark.parametrize('definition,kind', [('95% confidence interval', 'confidence_interval'),
                                           ('standard error', 'standard_error'), ('SEM', 'standard_error')])
def test_paper_uncertainty_type_comes_from_its_original_definition(definition, kind):
    row = 'Estimator 81.5 ±0.6'
    caption = 'Test accuracy (%), mean ± ' + definition + '.'
    refs = [{'id': 'paper', 'quote': caption + '\n' + row}]
    result = reference_uncertainty({'value_quote': row, 'std': 0.6,
        'context_quotes': {'uncertainty': caption}}, refs)
    assert result['uncertainty_kind'] == kind
    assert result['uncertainty_source']['quote'] == caption
    if definition.startswith('95'):
        assert result['confidence_level'] == 95


def test_unrelated_confidence_interval_elsewhere_is_not_the_selected_cell_uncertainty():
    row = 'Estimator 81.5 ±0.6'
    refs = [{'id': 'paper', 'quote': 'Other results use 95% confidence intervals.\n' + row}]
    assert reference_uncertainty({'value_quote': row, 'std': 0.6}, refs) is None


def test_paper_uncertainty_definition_cannot_be_fabricated():
    with pytest.raises(ValueError, match='own original citations'):
        reference_uncertainty({'context_quotes': {'uncertainty': '95% confidence interval'}},
                              [{'id': 'paper', 'quote': 'Results are mean ± standard deviation.'}])


def _followup_repetitions(text, count=7, quote=None):
    node = {'name': 'accuracy', 'runs': count}
    if quote is not None:
        node['context_quotes'] = {'runs': quote}
    return context_evidence(node, [{'id': 'original', 'source': 'pdf', 'quote': text}])['runs']


@pytest.mark.parametrize('text', [
    '7 runs were not performed.', '7 runs were never conducted.',
    '7 runs have not been performed.', "7 runs weren't conducted.",
    '7 runs are planned for future work.', '7 runs were proposed.',
    '7 runs will be performed.', '7 runs might be sufficient.',
    '7 runs would be required.', '7 runs in total were not performed.',
    'No 7 runs were performed.', 'We can perform 7 runs.',
])
def test_post_count_negation_and_future_plans_are_not_completed_counts(text):
    assert _followup_repetitions(text)['status'] == 'unknown'
    assert _followup_repetitions(text, quote='7 runs')['status'] == 'unknown'


@pytest.mark.parametrize('text,count', [
    ('We report accuracy over 1,007 runs.', 1007),
    ('We report accuracy over 1,000 runs.', 1000),
    ('We report accuracy over 12,345 runs.', 12345),
    ('runs: 1,007', 1007),
    ('We run each experiment 1,007 times.', 1007),
])
def test_comma_grouped_count_is_read_as_one_integer(text, count):
    result = _followup_repetitions(text, count)
    assert result['status'] == 'verified' and result['value'] == count
    assert _followup_repetitions(text, 7)['status'] == 'contradicted'


@pytest.mark.parametrize('text', [
    'We used 1,07 runs.', 'We used 1, 7 runs.', 'We used 1 007 runs.',
    'We used one thousand seven runs.', 'runs: 1,007, 21',
])
def test_incomplete_number_groups_and_lists_are_not_exact_counts(text):
    assert _followup_repetitions(text)['status'] == 'unknown'


@pytest.mark.parametrize('text,quote,count', [
    ('We used 1,007 runs.', '7 runs', 7),
    ('We used 1,001 run.', '1 run', 1),
    ('There were 70 runs.', '0 runs', 0),
    ('runs: 70', 'runs: 7', 7),
    ('These are 7 runstats.', '7 runs', 7),
    ('runs: 7, 11', 'runs: 7', 7),
    ('runs: 1,007, 21', 'runs: 1,007', 1007),
    ('runs: 7.5', 'runs: 7', 7),
    ('runs: 7/10', 'runs: 7', 7),
])
def test_cropped_context_cannot_create_a_number_or_word_boundary(text, quote, count):
    assert _followup_repetitions(text, count, quote)['status'] == 'unknown'


@pytest.mark.parametrize('text', [
    '7 runs were performed.', '7 runs have been completed.',
    'We report accuracy over 7 runs. Later runs were not performed.',
    '7 runs were completed; future work will examine other methods.',
    'Each experiment was repeated 7 times with different initializations.',
])
def test_affirmative_counts_and_separate_sentences_remain_supported(text):
    assert _followup_repetitions(text)['status'] == 'verified'


@pytest.mark.parametrize('statement,field,proposed,phrase', [
    ('We report accuracy over 7 runs.', 'runs', 7, '7 runs'),
    ('We report a single run.', 'runs', 1, 'single run'),
    ('We run each experiment 7 times.', 'runs', 7, 'experiment 7 times'),
    ('We report test accuracy.', 'split', 'test', 'test'),
    ('We report mean accuracy.', 'aggregation', 'mean', 'mean'),
    ('We report median accuracy.', 'aggregation', 'median', 'median'),
])
def test_long_line_evidence_keeps_the_supporting_match(statement,field,proposed,phrase):
    text='Background detail. '*35+statement
    result=context_evidence({'name':'accuracy',field:proposed},[{'id':'paper','source':'pdf','quote':text}])[field]
    assert result['status']=='verified'
    assert phrase in result['quote']
    assert result['quote'] in text
    assert len(result['quote'])<=400


def test_long_wrapped_count_keeps_its_whole_original_match():
    text='We report accuracy over 7'+'\n '*220+'runs.'
    result=context_evidence({'name':'accuracy','runs':7},[{'id':'paper','source':'pdf','quote':text}])['runs']
    assert result['status']=='verified'
    assert '7'+'\n '*220+'runs' in result['quote']
    assert result['quote'] in text


@pytest.mark.parametrize('same_citation,metric',[(False,'accuracy'),(True,'AUC')])
def test_other_metric_or_citation_cannot_supply_inferred_uncertainty(same_citation,metric):
    row='Estimator 81.5 ±0.6'
    caption=f'Test {metric} (%) with 95% confidence interval.'
    refs=([{'id':'p','source':'pdf','quote':caption+'\n'+row}] if same_citation else
          [{'id':'p','source':'pdf','quote':row},{'id':'other','source':'pdf','quote':caption}])
    assert reference_uncertainty({'name':'accuracy','value_quote':row,'std':0.6,
        'context_quotes':{'unit':caption}},refs) is None
