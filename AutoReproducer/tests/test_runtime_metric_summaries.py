"""Author summaries outrank individual trials and keep their own statistics."""
import json

import pytest

from src.author_claim_comparison import collect_author_metrics
from src.runtime_metrics import METRICS_MARKER, parse_runtime_metrics


DGC_OUTPUT = """Test accuracy: 0.8320, train time: 0.0572s
Test accuracy: 0.8330, train time: 0.0572s
Dataset: cora Test accuracy of 10 runs: mean 0.83280, std 0.00040
"""


@pytest.mark.parametrize('label', ['Test acc.', 'Test accuracy.', 'Eval F1.', 'Test MSE.'])
def test_abbreviation_period_before_separator_keeps_the_final_metric(label):
    records, errors = parse_runtime_metrics(label + ':8.53e1', preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['value'] == 85.3 and records[0]['split'] == 'test'
    assert records[0]['stage'] == 'final' and records[0]['unit'] == ''
    assert records[0]['source_quote'] == label + ':8.53e1'


def test_adjacent_metrics_inherit_only_their_own_explicit_split_clause():
    text = ('Epoch:17 train loss:0.23 acc:71.0 | val loss:0.41 acc:62.0\n'
            'Test acc.:73.2\n')
    records, errors = parse_runtime_metrics(text, preserve_contexts=True)
    assert not errors
    assert {(r['name'], r['split']): r['value'] for r in records} == {
        ('loss', 'train'): .23, ('accuracy', 'train'): 71.0,
        ('loss', 'validation'): .41, ('accuracy', 'validation'): 62.0,
        ('accuracy', 'test'): 73.2}
    final = next(row for row in parse_runtime_metrics(text)[0] if row['name'] == 'accuracy')
    assert final['value'] == 73.2 and final['split'] == 'test'


def test_split_context_never_crosses_clause_or_line_and_uses_nearest_label():
    records, _ = parse_runtime_metrics('train loss:1 acc:0.7 val loss:2 acc:0.6 | MSE:0.2\nF1:0.5',
                                       preserve_contexts=True)
    assert {(row['name'], row['split']) for row in records} == {
        ('loss', 'train'), ('accuracy', 'train'), ('loss', 'validation'),
        ('accuracy', 'validation'), ('mse', ''), ('f1_score', '')}


def test_dotted_object_attributes_and_reference_claims_are_not_measurements():
    records, errors = parse_runtime_metrics('model.acc:0.99\nPaper test acc.:99 | Test acc.:81',
                                            preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['value'] == 81 and records[0]['split'] == 'test'


def test_dgc_final_ten_trial_summary_replaces_last_trial():
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0,
                            'stdout': DGC_OUTPUT, 'stderr': ''}]}
    records, errors = collect_author_metrics(execution)
    assert not errors and len(records) == 1
    row = records[0]
    assert row['name'] == 'accuracy' and row['value'] == 0.8328
    assert row['aggregation'] == 'mean' and row['runs'] == 10 and row['std'] == 0.0004
    assert row['source'] == 'train:stdout:3' and row['source_quote'] == DGC_OUTPUT.splitlines()[-1]


def test_digcn_saved_output_keeps_test_mean_separate_from_validation():
    line = 'Val Acc: 0.8055, Val Loss: 0.6815, Test Accuracy: 0.7719 ± 0.0044, Duration: 2.4038'
    records, errors = parse_runtime_metrics(line, preserve_contexts=True)
    assert not errors and len(records) == 3
    test = next(row for row in records if row['split'] == 'test')
    assert test['value'] == 0.7719 and test['std'] == 0.0044 and test['aggregation'] == 'mean'
    assert 'runs' not in test
    assert all('std' not in row for row in records if row['split'] == 'validation')


@pytest.mark.parametrize('line,expected', [
    ('Test accuracy: 0.7719 ± 0.0044', (0.7719, 0.0044, '', None)),
    ('Test Accuracy: 77.19 ± 0.44%', (77.19, 0.44, 'percent', None)),
    ('Test accuracy: 77.19% +/- 0.44%', (77.19, 0.44, 'percent', None)),
    ('Mean Test accuracy: 0.82, std: 0.02, runs=10', (0.82, 0.02, '', 10)),
    ('Test accuracy over 20 seeds: mean=0.82, std=0.02', (0.82, 0.02, '', 20)),
])
def test_explicit_summary_statistics_and_units(line, expected):
    records, errors = parse_runtime_metrics(line, preserve_contexts=True)
    assert not errors and len(records) == 1
    row = records[0]
    assert (row['value'], row['std'], row['unit'], row.get('runs')) == expected
    assert row['aggregation'] == 'mean'


@pytest.mark.parametrize('aggregation', ['mean', 'median', 'best'])
def test_reported_summary_is_not_overwritten_by_a_later_trial(aggregation):
    records, errors = parse_runtime_metrics(
        f'Test accuracy of 10 runs: {aggregation} 0.82\nTest accuracy: 0.91', preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['aggregation'] == aggregation and records[0]['value'] == 0.82


def test_plain_measurement_does_not_invent_repetitions_or_statistics():
    records, errors = parse_runtime_metrics('Test accuracy: 0.82', preserve_contexts=True)
    assert not errors
    assert not any(key in records[0] for key in ('aggregation', 'runs', 'std'))


def test_unrelated_summary_does_not_abort_final_metric_extraction():
    records, errors = parse_runtime_metrics(
        'latency mean 2.1; Test accuracy: 0.8', preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['name'] == 'accuracy' and records[0]['value'] == 0.8
    assert 'aggregation' not in records[0]


def test_explicitly_requested_custom_summary_remains_supported():
    records, errors = parse_runtime_metrics('latency mean 2.1', expected_names=['latency'])
    assert not errors and len(records) == 1
    assert records[0]['name'] == 'latency' and records[0]['value'] == 2.1
    assert records[0]['aggregation'] == 'mean'


def test_confidence_interval_is_not_a_standard_deviation():
    records, errors = parse_runtime_metrics('Test accuracy: 0.82 ± 0.02 (95% CI)', preserve_contexts=True)
    assert not errors and 'std' not in records[0] and 'aggregation' not in records[0]


def test_two_metrics_on_a_line_cannot_borrow_statistics():
    records, errors = parse_runtime_metrics(
        'Mean Validation accuracy: 0.88, std: 0.01; Test accuracy: 0.81', preserve_contexts=True)
    assert not errors and len(records) == 2
    validation, test = records
    assert validation['aggregation'] == 'mean' and validation['std'] == 0.01
    assert test['split'] == 'test' and 'aggregation' not in test and 'std' not in test


def test_paper_reference_summary_never_overwrites_actual_output():
    records, errors = parse_runtime_metrics(
        DGC_OUTPUT + 'Paper test accuracy of 10 runs: mean 0.99, std 0.01', preserve_contexts=True)
    assert not errors and records[0]['value'] == 0.8328


@pytest.mark.parametrize('extra', [
    {'runs': True}, {'runs': 0}, {'runs': 1.5}, {'std': -0.01},
    {'std': 'nan'}, {'aggregation': []}, {'aggregation': 'invented'},
])
def test_invalid_structured_statistics_remain_an_error_without_text_fallback(extra):
    records, errors = parse_runtime_metrics(DGC_OUTPUT, structured=[
        {'name': 'accuracy', 'value': 0.5, 'split': 'test', **extra}])
    assert errors and records[0]['value'] == 0.5


def test_structured_summary_remains_authoritative_and_keeps_provenance():
    payload = {'metrics': [{'name': 'accuracy', 'value': 0.9, 'unit': 'fraction',
                           'aggregation': 'mean', 'runs': 20, 'std': 0.01}]}
    records, errors = parse_runtime_metrics(METRICS_MARKER + json.dumps(payload) + '\n' + DGC_OUTPUT)
    assert not errors
    assert records[0]['value'] == 0.9 and records[0]['runs'] == 20
    assert records[0]['source'] == 'stdout:1'


def logger_output(ids=(1, 2, 3, 4, 5), metric='acc', total=5):
    lines = [f"Namespace(metric='{metric}', runs={total})"]
    for run in ids:
        lines += [f'Run {run:02d}:', 'Highest Test: 99.99', 'Final Train: 100.00', 'Final Test: 89.66']
    lines += ['All runs:', 'Highest Test: 99.99 ± 0.10',
              '  Final Train: 100.00 ± 0.00', '   Final Test: 89.48 ± 0.40']
    return '\n'.join(lines)


def test_typed_logger_final_test_summary_beats_last_trial_and_highest_test():
    text = logger_output()
    records, errors = parse_runtime_metrics(text, preserve_contexts=True)
    assert not errors
    row = next(row for row in records if row['split'] == 'test')
    assert row['name'] == 'accuracy' and row['value'] == 89.48
    assert row['aggregation'] == 'mean' and row['std'] == 0.40 and row['runs'] == 5
    assert row['unit'] == ''  # Printed logger source, not magnitude, supplies units later.
    assert row['source_quote'] == 'Final Test: 89.48 ± 0.40'
    assert row['source'] == f'stdout:{len(text.splitlines())}'
    assert next(row for row in records if row['split'] == 'train')['value'] == 100


@pytest.mark.parametrize('ids', [(), (1,), (1, 2, 4, 5), (1, 2, 2, 3, 4, 5)])
def test_declared_runs_alone_or_incomplete_logger_blocks_do_not_prove_a_count(ids):
    records, _ = parse_runtime_metrics(logger_output(ids), preserve_contexts=True)
    test = next(row for row in records if row['split'] == 'test')
    assert test['value'] == 89.48 and test['aggregation'] == 'mean'
    assert 'runs' not in test


@pytest.mark.parametrize('metric,expected', [('rocauc', 'roc_auc'), ('f1', 'f1_score')])
def test_logger_uses_the_explicit_metric_name_instead_of_assuming_accuracy(metric, expected):
    records, errors = parse_runtime_metrics(logger_output(metric=metric), preserve_contexts=True)
    assert not errors and {record['name'] for record in records} == {expected}


def test_ambiguous_bare_logger_labels_do_not_invent_a_metric_identity():
    records, _ = parse_runtime_metrics('All runs:\nFinal Test: 89.48 ± 0.40', preserve_contexts=True)
    assert not records


def test_per_run_final_logger_output_is_not_a_five_run_mean():
    records, errors = parse_runtime_metrics("Namespace(metric='acc', runs=5)\nRun 05:\nFinal Test: 89.66", preserve_contexts=True)
    assert not errors
    assert records[0]['aggregation'] == 'single' and records[0]['runs'] == 1


def test_logger_explicit_ci_and_paper_reference_never_become_standard_deviation_or_measurement():
    text = "Namespace(metric='acc', runs=5)\nAll runs:\nFinal Test: 89.48 ± 0.40 (95% CI)\nPaper Final Test: 99.99 ± 0.01"
    records, errors = parse_runtime_metrics(text, preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['value'] == 89.48 and 'std' not in records[0]


def test_explicit_structured_measurement_still_wins_over_typed_logger():
    records, _ = parse_runtime_metrics(logger_output(), structured=[{
        'name': 'accuracy', 'value': 0.7, 'unit': 'fraction', 'split': 'test'}])
    assert records[0]['value'] == 0.7 and records[0]['unit'] == 'fraction'


def test_configuration_and_layer_repr_are_not_runtime_measurements():
    output = ("Namespace(epochs=500, lr=0.01, runs=5, loss=1.0)\n"
              "(lin): Linear(in_features=40, out_features=3, bias=True)\n"
              "Test AUC: 0.8730\nlatency=2.1")
    records, errors = parse_runtime_metrics(output, preserve_contexts=True)
    assert not errors and [(r['name'], r['value']) for r in records] == [('auc', 0.8730)]
    custom, _ = parse_runtime_metrics(output, expected_names=['latency'])
    assert next(r for r in custom if r['name'] == 'latency')['value'] == 2.1


@pytest.mark.parametrize('macro,micro', [('MacroF1', 'MicroF1'), ('Macro-F1', 'Micro-F1'),
    ('f1_macro', 'f1_micro'), ('Macro F1', 'Micro F1'), ('macro_f1_score', 'micro_f1_score')])
def test_f1_averaging_methods_remain_distinct(macro, micro):
    records, errors = parse_runtime_metrics(f'F1: 0.9702, {macro}: 0.6859, {micro}: 0.9433')
    assert not errors
    assert {r['name']: r['value'] for r in records} == {
        'f1_score': 0.9702, 'f1_macro': 0.6859, 'f1_micro': 0.9433}


@pytest.mark.parametrize('header,count', [
    ('GPRGNN on dataset cora, in 100 repeated experiment:', 100),
    ('Estimator-42 on dataset benchmark_v2, in 7 repeated experiments:', 7),
    ('Final results over 5 completed trials:', 5),
    ('Summary of 12 independent seeds:', 12),
    ('Results across 3 runs:', 3),
])
def test_adjacent_reported_summary_header_preserves_count_and_its_exact_provenance(header, count):
    line = 'test acc mean = 79.6319 \t test acc std = 1.7389 \t val acc mean = 81.2941'
    rows, errors = parse_runtime_metrics(header + '\n' + line, preserve_contexts=True)
    assert not errors and len(rows) == 2
    assert {row['runs'] for row in rows} == {count}
    assert rows[0]['value'] == 79.6319 and rows[0]['std'] == 1.7389
    assert all(row['source_quote'] == line for row in rows)
    assert all(row['runs_source'] == {'value': count, 'source': 'stdout:1',
        'source_quote': header, 'kind': 'reported_summary'} for row in rows)
    assert all(row['unit'] == '' for row in rows)


@pytest.mark.parametrize('header', [
    'Planned results over 100 runs:', 'Summary of up to 100 runs:',
    'Summary of at least 100 runs:', 'Summary of 100 unfinished runs:',
    'Not a summary of 100 runs:', 'Paper results over 100 runs:',
    'Reference on dataset cora, in 100 repeated experiment:',
    'Summary of 0 completed runs:', 'Summary of 100 or 200 runs:',
    'Namespace(runs=100)', '--runs 100', 'for RP in range(100):',
])
def test_plans_bounds_negation_configuration_and_reference_headers_do_not_prove_runs(header):
    rows, errors = parse_runtime_metrics(header + '\nTest accuracy mean=0.8', preserve_contexts=True)
    assert not errors and len(rows) == 1
    assert 'runs' not in rows[0] and 'runs_source' not in rows[0]


@pytest.mark.parametrize('gap', ['\n', 'Training started\n', 'Epoch: 1\n',
    'Summary of 20 runs:\n', 'Summary of 10 runs:\n'])
def test_summary_header_cannot_cross_a_gap_or_competing_header(gap):
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\n' + gap + 'Test accuracy mean=0.8')
    assert not errors and len(rows) == 1 and 'runs' not in rows[0]


def test_summary_header_does_not_turn_a_single_measurement_into_a_repeated_summary():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy:0.8')
    assert not errors and 'runs' not in rows[0] and 'aggregation' not in rows[0]
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8; Loss:0.1')
    assert not errors and all('runs' not in row for row in rows)


def test_summary_header_is_consumed_by_one_physical_line_only():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8\nTest F1 mean=0.7')
    assert not errors
    assert rows[0]['runs'] == 10 and 'runs' not in rows[1]


def test_conflicting_header_and_metric_counts_remain_an_error():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy of 20 runs: mean 0.8')
    assert errors and 'different repetition counts' in errors[0]
    assert 'runs' not in rows[0] and 'runs_source' not in rows[0]


def test_same_line_explicit_count_can_agree_with_header():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy of 10 runs: mean 0.8')
    assert not errors and rows[0]['runs'] == 10


@pytest.mark.parametrize('line,values', [
    ('Test accuracy mean=0.8; 10 runs; 20 runs', [0.8]),
    ('Test accuracy mean=0.8; runs=10; runs=20', [0.8]),
    ('Test accuracy mean=0.8; 20 runs; 30 runs', [0.8]),
    ('Test accuracy of 10 runs: mean 0.8; Test F1 of 20 runs: mean 0.7', [0.8, 0.7]),
])
def test_summary_header_cannot_resolve_conflicting_inline_counts(line, values):
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\n' + line, preserve_contexts=True)
    assert [row['value'] for row in rows] == values
    assert any('conflicting repetition counts' in error for error in errors)
    assert all('runs' not in row and 'runs_source' not in row for row in rows)


@pytest.mark.parametrize('counts', ['10 runs; 10 trials', 'runs=10; trials=10'])
def test_repeated_matching_inline_counts_keep_the_summary_provenance(counts):
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8; ' + counts)
    assert not errors and rows[0]['value'] == 0.8 and rows[0]['runs'] == 10
    assert rows[0]['runs_source']['source'] == 'stdout:1'


def test_later_summary_never_borrows_an_earlier_header():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8\n'
        'New evaluation\nTest accuracy mean=0.9')
    assert not errors and rows[0]['value'] == 0.9 and 'runs' not in rows[0]


def test_structured_output_does_not_borrow_text_header_count():
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8',
        structured=[{'name': 'accuracy', 'value': 0.9, 'aggregation': 'mean'}])
    assert not errors and rows[0]['value'] == 0.9 and 'runs' not in rows[0]


@pytest.mark.parametrize('count', [
    '-2 runs', '+2 runs', '−2 runs', '5/10 runs', '5-10 runs', '5 to 10 runs',
    'at least 10 runs', 'at most 10 runs', 'up to 10 runs', 'approximately 10 trials',
    'about 10 repeats', '10 runs or fewer', '10 runs at most', 'runs: 7+',
    'runs: 7/10', 'runs: 7.5', 'runs: 7,8', '100, 007 runs', 'seeds: 42', 'n=10',
    'planned 10 runs', 'will use 10 runs', '10 runs were not completed',
    'paper reports 10 runs', 'reference: 10 trials', 'target: 10 repeats',
])
def test_runtime_count_qualifiers_identifiers_and_number_fragments_stay_unknown(count):
    rows, errors = parse_runtime_metrics('Test accuracy mean=0.8; ' + count)
    assert not errors and rows[0]['value'] == 0.8
    assert 'runs' not in rows[0]


@pytest.mark.parametrize('count', ['at most 10 runs', '-2 runs', '5/10 runs', 'seeds: 42',
                                 'paper reports 10 runs'])
def test_summary_header_does_not_hide_an_uncertain_inline_run_count(count):
    rows, errors = parse_runtime_metrics('Summary of 10 runs:\nTest accuracy mean=0.8; ' + count)
    assert not errors and rows[0]['value'] == 0.8
    assert 'runs' not in rows[0] and 'runs_source' not in rows[0]


@pytest.mark.parametrize('suffix,count', [
    ('over 10 runs', 10), ('runs=10', 10), ('10 independent runs', 10),
    ('1,007 trials', 1007), ('number of seeds: 10', 10),
])
def test_complete_exact_inline_counts_remain_bound_to_the_metric(suffix, count):
    rows, errors = parse_runtime_metrics('Test accuracy mean=0.8, std=0.01, ' + suffix)
    assert not errors and rows[0]['runs'] == count and rows[0]['value'] == 0.8


@pytest.mark.parametrize('style', ['summary_header', 'inline', 'logger'])
def test_unparseable_large_counts_do_not_discard_valid_metric_values(style):
    huge = '9' * 5000
    if style == 'summary_header':
        text = 'Summary of ' + huge + ' runs:\nTest accuracy mean=0.8'
    elif style == 'inline':
        text = 'Test accuracy mean=0.8; runs=' + huge
    else:
        text = "Namespace(metric='acc', runs=2)\nRun " + huge + ':\nFinal Test:0.8'
    rows, errors = parse_runtime_metrics(text)
    assert not errors and rows[0]['value'] == 0.8
    assert 'runs' not in rows[0]


@pytest.mark.parametrize('source', [10, None, 'stdout:999'])
def test_structured_producer_cannot_inject_derived_header_provenance(source):
    payload = [{'name': 'accuracy', 'value': 0.9, 'runs': 5, 'runs_source': {
        'kind': 'reported_summary', 'value': 5, 'source': source, 'source_quote': 'invented'}}]
    rows, errors = collect_author_metrics({'steps': [{'id': 'eval', 'success': True, 'exit_code': 0,
        'metric_records': payload}]})
    assert not errors and rows[0]['runs'] == 5 and 'runs_source' not in rows[0]


def test_summary_count_does_not_cross_failed_steps_other_steps_or_output_streams():
    execution = {'steps': [
        {'id': 'old', 'success': False, 'exit_code': 1,
         'stdout': 'Summary of 100 runs:\nTest accuracy mean=0.99'},
        {'id': 'header_only', 'success': True, 'exit_code': 0, 'stdout': 'Summary of 100 runs:'},
        {'id': 'current', 'success': True, 'exit_code': 0,
         'stderr': 'Summary of 100 runs:', 'stdout': 'Test accuracy mean=0.8'}]}
    rows, errors = collect_author_metrics(execution)
    assert not errors and len(rows) == 1 and rows[0]['value'] == 0.8
    assert 'runs' not in rows[0]


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_summary_count_evidence_keeps_its_step_stream_and_header_line(stream):
    from src.author_comparison_semantics import context_evidence
    header = 'Final results over 10 completed trials:'
    rows, errors = collect_author_metrics({'steps': [{'id': 'evaluate', 'success': True, 'exit_code': 0,
        stream: header + '\nTest accuracy mean=0.8'}]})
    assert not errors and len(rows) == 1
    row = rows[0]
    assert row['source'] == f'evaluate:{stream}:2'
    assert row['runs_source']['source'] == f'evaluate:{stream}:1'
    evidence = context_evidence({'runs': 10}, [], observed=row)['runs']
    assert evidence['status'] == 'verified' and evidence['value'] == 10
    assert evidence['quote'] == header and evidence['citation'] == f'evaluate:{stream}:1'
    assert 'not independently counted' in evidence['reason']
