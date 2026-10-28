"""Real failed-table evidence and adversarial positional table checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.author_table_evidence import locate_table_value


def table(reference=None, *, header='Method Alpha Beta', row='Signal 81.5 ± 0.6 79.4 ± 1.9',
          prefix='', between='', suffix=''):
    reference = {'dataset': 'Alpha', 'method': 'Signal', 'name': 'accuracy',
                 'value': 81.5, 'std': 0.6, **(reference or {})}
    page = prefix + header + '\n' + between + row + suffix
    return reference, row, header, page


def locate(case):
    ref, row, header, page = case
    return locate_table_value(ref, row, header, page_text=page)


def pyg_case():
    path = Path(__file__).resolve().parents[1] / 'docs/evidence/pyg_execution_20261019_regression_05.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    comparison = record['author_claim_comparison']
    reference = deepcopy(comparison['comparisons'][0]['reference'])
    page = next(c['quote'] for c in comparison['citations'] if c['source'] == 'pdf')
    header = 'Cora CiteSeer PubMed\nMethod\nFixed Random Fixed Random Fixed Random'
    return reference, reference['value_quote'], header, page


def test_real_pyg_repeated_child_columns_are_resolved_from_original_full_header():
    ref, row, header, page = pyg_case()
    ref['column_labels'] = ['Cora', 'Fixed']
    result = locate_table_value(ref, row, header, page_text=page)
    assert result['status'] == 'verified'
    assert result['column_index'] == 0 and result['column_labels'] == ['Cora', 'Fixed']
    assert result['value'] == 81.5 and result['std'] == 0.6


def test_real_pyg_dataset_header_alone_does_not_prove_the_subcolumn():
    ref, row, _, page = pyg_case()
    ref.update(column_verified=True, column_labels=['Cora', 'Fixed'])
    result = locate_table_value(ref, row, 'Cora CiteSeer PubMed', page_text=page)
    assert result['status'] == 'unknown'


def test_real_pyg_correct_value_without_subcolumn_selector_stays_unknown():
    ref, row, header, page = pyg_case()
    result = locate_table_value(ref, row, header, page_text=page)
    assert result['status'] == 'unknown'


def test_flat_table_returns_exact_cell_offsets_and_preserves_inputs():
    case = table()
    before = deepcopy(case)
    result = locate(case)
    assert result['status'] == 'verified'
    assert result['column_index'] == 0 and result['column_labels'] == ['Alpha']
    assert result['value'] == 81.5 and result['std'] == 0.6
    assert case == before
    assert case[1][result['cell_span'][0]:result['cell_span'][1]] == result['cell_quote']
    assert case[3][result['row_span'][0]:result['row_span'][1]] == case[1]
    assert result['basis']


def test_dataset_rows_select_method_columns_by_label_not_value():
    case = table({'column_labels': ['Signal']}, header='Datasets Baseline Signal',
                 row='Alpha 79.4 ± 1.9 81.5 ± 0.6')
    result = locate(case)
    assert result['status'] == 'verified'
    assert result['orientation'] == 'datasets_in_rows'
    assert result['column_index'] == 1 and result['value'] == 81.5
    case[0]['value'] = 79.4
    assert locate(case)['status'] == 'rejected'


@pytest.mark.parametrize('header,row', [
    ('Datasets Baseline Signal+', 'Alpha 79.4 ± 1.9 81.5 ± 0.6'),
    ('Datasets Baseline Signal', 'Alpha-v2 79.4 ± 1.9 81.5 ± 0.6'),
    ('Datasets Signal Signal', 'Alpha 81.5 ± 0.6 81.5 ± 0.6'),
])
def test_transposed_variants_and_duplicate_headers_cannot_borrow_values(header, row):
    assert locate(table(header=header, row=row))['status'] != 'verified'


@pytest.mark.parametrize('separator', ['±', '+/-', '\x06'])
def test_uncertainty_separators_preserve_the_same_column(separator):
    result = locate(table(row=f'Signal 81.5 {separator} 0.6 79.4 {separator} 1.9'))
    assert result['status'] == 'verified' and result['std'] == 0.6


def test_wrong_column_mean_is_rejected_after_location_not_selected_by_value():
    result = locate(table({'value': 79.4, 'std': 1.9}))
    assert result['status'] == 'rejected'
    assert result['value'] == 81.5 and result['column_index'] == 0


def test_wrong_column_standard_deviation_is_rejected():
    result = locate(table({'std': 1.9}))
    assert result['status'] == 'rejected' and result['std'] == 0.6


def test_equal_neighboring_values_still_need_a_unique_label_path():
    case = table({'column_verified': True}, header='Method Alpha Alpha',
                 row='Signal 81.5 ±0.6 81.5 ±0.6')
    assert locate(case)['status'] == 'unknown'


@pytest.mark.parametrize('missing', ['-', '—', '–', 'N/A', 'NA', 'nan'])
def test_selected_missing_cell_is_rejected_instead_of_borrowing_its_neighbor(missing):
    assert locate(table(row=f'Signal {missing} 81.5 ±0.6'))['status'] == 'rejected'


@pytest.mark.parametrize('missing', ['-', '—', 'N/A'])
def test_missing_previous_cell_does_not_shift_a_later_dataset(missing):
    result = locate(table({'dataset': 'Beta'}, row=f'Signal {missing} 81.5 ±0.6'))
    assert result['status'] == 'verified' and result['column_index'] == 1


def test_multi_level_selector_uses_a_label_from_the_bound_header_only():
    header = 'Alpha Beta\nFixed Random Fixed Random'
    row = 'Signal 81.5 ±0.6 79.4 ±1.9 70.1 ±0.2 69.8 ±0.3'
    result = locate(table({'column_subgroup': 'Random', 'value': 79.4, 'std': 1.9}, header=header, row=row))
    assert result['status'] == 'verified' and result['column_index'] == 1
    assert result['column_labels'] == ['Alpha', 'Random']


def test_equal_values_in_repeated_dataset_children_do_not_choose_a_child():
    case = table(header='Alpha Beta\nFixed Random Fixed Random',
                 row='Signal 81.5 ±0.6 81.5 ±0.6 70.1 ±0.2 69.8 ±0.3')
    assert locate(case)['status'] == 'unknown'


def test_unrelated_page_selector_words_cannot_prove_a_missing_header_level():
    case = table({'column_labels': ['Alpha', 'Fixed']}, prefix='Our training split is Fixed.\n')
    assert locate(case)['status'] == 'unknown'


@pytest.mark.parametrize('header', ['Alpha Beta\nFixed Random',
                                    'Alpha Beta\nFixed Random Extra Fixed Random',
                                    'Alpha Beta\nFixed Fixed Fixed Fixed'])
def test_nonuniform_or_duplicate_child_structure_stays_unknown(header):
    case = table({'column_subgroup': 'Fixed'}, header=header,
                 row='Signal 81.5 ±0.6 79.4 ±1.9 70.1 ±0.2 69.8 ±0.3')
    assert locate(case)['status'] == 'unknown'


def test_three_header_levels_map_all_dimensions_in_reading_order():
    header = 'Alpha Beta\nFixed Random Fixed Random\nAUC F1 AUC F1 AUC F1 AUC F1'
    case = table({'dataset': 'Beta', 'name': 'f1_score',
                  'column_labels': ['Beta', 'Random', 'F1'], 'value': 8, 'std': None},
                 header=header, row='Signal 1 2 3 4 5 6 7 8')
    result = locate(case)
    assert result['status'] == 'verified' and result['column_index'] == 7


def test_explicit_repeated_dataset_header_has_unique_child_paths():
    case = table({'column_labels': ['Alpha', 'Random'], 'value': 79.4, 'std': 1.9},
                 header='Method Alpha Alpha Beta Beta\nFixed Random Fixed Random',
                 row='Signal 81.5 ±0.6 79.4 ±1.9 70.1 ±0.2 69.8 ±0.3')
    result = locate(case)
    assert result['status'] == 'verified' and result['column_index'] == 1


def test_truncated_row_even_with_matching_partial_header_is_unknown():
    ref, row, header, page = table()
    result = locate_table_value(ref, 'Signal 81.5 ± 0.6', 'Method Alpha', page_text=page)
    assert result['status'] == 'unknown'


def test_quote_spanning_another_method_row_is_not_one_complete_row():
    ref, row, header, page = table(suffix='\nOther 83.1 ±0.4 80.2 ±0.1')
    result = locate_table_value(ref, row + '\nOther 83.1 ±0.4 80.2 ±0.1', header, page_text=page)
    assert result['status'] == 'unknown'


def test_multiple_identical_method_rows_on_the_page_are_ambiguous():
    ref, row, header, page = table()
    assert locate_table_value(ref, row, header, page_text=page + '\n' + page)['status'] == 'unknown'


def test_a_header_from_a_different_table_cannot_ground_the_row():
    ref, row, header, page = table(between='Different table\nMethod Gamma Delta\n')
    assert locate_table_value(ref, row, header, page_text=page)['status'] == 'unknown'


def test_neighboring_complete_method_rows_do_not_block_header_binding():
    case = table(between='Baseline 50 ±1 40 ±2\n')
    assert locate(case)['status'] == 'verified'


def test_without_page_context_completeness_cannot_be_claimed():
    ref, row, header, _ = table()
    assert locate_table_value(ref, row, header)['status'] == 'unknown'


def test_fabricated_header_is_rejected_even_if_labels_exist_elsewhere():
    ref, row, header, page = table(prefix='Gamma Alpha and Beta\n')
    assert locate_table_value(ref, row, 'Method Alpha Gamma', page_text=page)['status'] == 'rejected'


def test_tuple_metric_and_dataset_are_both_located():
    caption = 'Test results (AUC, Macro-F1) in percent.\n'
    case = table({'name': 'f1_macro', 'value': 68.9, 'std': 0.8, 'metric_quote': caption.strip()},
                 prefix=caption, row='Signal (87.5 ±0.9, 68.9 ±0.8) (90.2 ±0.4, 70.1 ±1.2)')
    result = locate(case)
    assert result['status'] == 'verified' and result['metric_index'] == 1
    assert result['value'] == 68.9 and result['std'] == 0.8


@pytest.mark.parametrize('metric_quote', [None, 'Results (Macro-F1, AUC).'])
def test_tuple_metric_order_cannot_come_from_a_missing_or_fabricated_caption(metric_quote):
    case = table({'name': 'f1_macro', 'value': 68.9, 'std': 0.8, 'metric_quote': metric_quote},
                 prefix='Test results (AUC, Macro-F1) in percent.\n',
                 row='Signal (87.5 ±0.9, 68.9 ±0.8) (90.2 ±0.4, 70.1 ±1.2)')
    assert locate(case)['status'] in {'unknown', 'rejected'}


def test_tuple_missing_metric_is_not_replaced_by_another_component():
    caption = 'Results (AUC, Macro-F1).'
    case = table({'name': 'f1_macro', 'metric_quote': caption}, prefix=caption+'\n',
                 row='Signal (81.5 ±0.6, —) (79.4 ±1.9, 70.1 ±1.2)')
    assert locate(case)['status'] == 'rejected'


def test_tuple_wrong_component_is_rejected_even_when_present_in_same_dataset():
    caption = 'Results (AUC, Macro-F1).'
    case = table({'name': 'f1_macro', 'metric_quote': caption}, prefix=caption+'\n',
                 row='Signal (81.5 ±0.6, 68.9 ±0.8) (79.4 ±1.9, 70.1 ±1.2)')
    assert locate(case)['status'] == 'rejected'


def test_metadata_placeholders_and_real_method_typography_keep_offsets():
    case = table({'method': 'SIG NAL LIN EAR'},
                 header='Type Model Non-linearity PDE-solver Input-G Alpha Beta',
                 row='SIG NAL LIN EAR - - - 81.5 ±0.6 79.4 ±1.9')
    result = locate(case)
    assert result['status'] == 'verified' and result['column_index'] == 3


def test_method_variant_cannot_be_selected_by_substring():
    assert locate(table(row='Signal-PPR 81.5 ±0.6 79.4 ±1.9'))['status'] == 'rejected'


@pytest.mark.parametrize('metadata', ['Variant', 'OtherDataset', 'Variant test'])
def test_arbitrary_row_prefix_cannot_be_discarded_as_if_it_were_known_metadata(metadata):
    assert locate(table(row=f'Signal {metadata} 81.5 ±0.6 79.4 ±1.9'))['status'] == 'unknown'


def test_delimited_multiword_dataset_labels_preserve_cell_positions():
    case = table({'dataset': 'Alpha Set'}, header='Method | Alpha Set | Beta Set',
                 row='Signal | 81.5 ±0.6 | 79.4 ±1.9')
    result = locate(case)
    assert result['status'] == 'verified' and result['column_index'] == 0


@pytest.mark.parametrize('row', ['Signal 81.5 ±0.6 noise 79.4 ±1.9',
                                'Signal 81.5 ±0.6 79.4 ±1.9 100',
                                'Signal 81.5(0.6) 79.4(1.9)'])
def test_unparsed_or_extra_cells_cannot_silently_shift_positions(row):
    assert locate(table(row=row))['status'] == 'unknown'


def test_numeric_substrings_in_header_are_not_targets_or_paper_specific_rules():
    case = table({'dataset': 'Graph-42', 'method': 'Model-2027'},
                 header='Method Graph-42 Graph-108', row='Model-2027 81.5 ±0.6 79.4 ±1.9')
    assert locate(case)['status'] == 'verified'


def test_inline_dataset_with_multiple_metrics_needs_the_exact_metric_header():
    ref, row, _, page = table({'dataset': 'ToyGrid', 'method': 'Estimator', 'std': None, 'value': 90},
                             header='Accuracy(percent) Recall(percent)', row='Estimator ToyGrid test 80.0 90.0')
    assert locate_table_value(ref, row, None, page_text=page)['status'] == 'unknown'


def test_inline_dataset_wrong_metric_value_is_rejected():
    case = table({'dataset': 'ToyGrid', 'method': 'Estimator', 'std': None, 'value': 90},
                 header='Accuracy(percent) Recall(percent)', row='Estimator ToyGrid test 80.0 90.0')
    result = locate(case)
    assert result['status'] == 'rejected' and result['value'] == 80
    assert result['column_index'] == 0


def test_inline_dataset_can_select_an_explicit_matching_metric_header():
    case = table({'dataset': 'ToyGrid', 'method': 'Estimator', 'std': None, 'value': 90, 'name': 'recall'},
                 header='Accuracy(percent) Recall(percent)', row='Estimator ToyGrid test 80.0 90.0')
    result = locate(case)
    assert result['status'] == 'verified' and result['value'] == 90
    assert result['column_index'] == 1


def test_a_metric_leaf_header_cannot_be_overridden_by_a_different_metric_name():
    case = table({'column_labels': ['Alpha', 'Recall']},
                 header='Alpha Beta\nAccuracy Recall Accuracy Recall',
                 row='Signal 79.4 ±1.9 81.5 ±0.6 70.1 ±0.2 69.8 ±0.3')
    assert locate(case)['status'] in {'unknown', 'rejected'}


def test_tuple_caption_from_a_previous_table_cannot_select_a_component():
    caption = 'Results (AUC, Macro-F1).'
    previous = caption + '\nMethod Gamma Delta\nOther (1, 2) (3, 4)\n'
    case = table({'name': 'f1_macro', 'value': 68.9, 'std': 0.8, 'metric_quote': caption},
                 prefix=previous+'Results (Macro-F1, AUC).\n',
                 row='Signal (87.5 ±0.9, 68.9 ±0.8) (90.2 ±0.4, 70.1 ±1.2)')
    assert locate(case)['status'] == 'unknown'


def test_empty_delimited_cells_cannot_shift_values_into_a_missing_dataset():
    case = table(header='Method | Alpha | Beta | Gamma',
                 row='Signal | | 81.5 ±0.6 | 79.4 ±1.9 | 70.1 ±0.2')
    assert locate(case)['status'] == 'unknown'


def test_multiple_values_in_one_delimited_cell_do_not_become_extra_columns():
    case = table(header='Method | Alpha | Beta | Gamma',
                 row='Signal | 81.5 ±0.6 79.4 ±1.9 | 70.1 ±0.2')
    assert locate(case)['status'] == 'unknown'


@pytest.mark.parametrize('unparsed', ['inf', '-infinity', '81.5*'])
def test_an_invalid_numeric_first_cell_cannot_be_skipped_as_row_metadata(unparsed):
    case = table(header='Method Alpha Beta Gamma',
                 row=f'Signal {unparsed} 81.5 ±0.6 79.4 ±1.9 70.1 ±0.2')
    assert locate(case)['status'] == 'unknown'


@pytest.mark.parametrize('field,value', [('value', True), ('std', False), ('std', float('inf'))])
def test_invalid_proposed_numbers_are_not_certified(field, value):
    assert locate(table({field: value}))['status'] == 'rejected'


def test_identical_values_in_distinct_columns_still_have_explicit_positions():
    case = table({'dataset': 'Beta'}, row='Signal 81.5 ±0.6 81.5 ±0.6')
    result = locate(case)
    assert result['status'] == 'verified' and result['column_index'] == 1


@pytest.mark.parametrize('dataset_index,child_index', [(i, j) for i in range(4) for j in range(3)])
def test_arbitrary_names_and_reordered_protocol_labels_use_positions_not_values(dataset_index, child_index):
    datasets = ['Network-27', 'Network-3', 'Network-90', 'Network-6']
    children = ['Protocol-Z', 'Protocol-C', 'Protocol-A']
    header = ' '.join(datasets) + '\n' + ' '.join(children * len(datasets))
    row = 'Method-2028 ' + ' '.join('7.25 ±0.125' for _ in range(12))
    case = table({'dataset': datasets[dataset_index], 'method': 'Method-2028',
                  'column_labels': [datasets[dataset_index], children[child_index]], 'value': 7.25, 'std': 0.125},
                 header=header, row=row)
    result = locate(case)
    assert result['status'] == 'verified'
    assert result['column_index'] == dataset_index * len(children) + child_index
