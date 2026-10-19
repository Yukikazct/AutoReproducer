"""Presentation must preserve the boundaries of source-checked comparisons."""
from copy import deepcopy
import io
from unittest.mock import Mock
import zipfile

import pytest

from frontend.report_renderer import render_report, report_downloads
from src.agents.report_generator import _author_claim_comparison_lines, _metric_value_text


def comparison():
    return {'relative_tolerance': 0.05, 'reason': '基于作者输出的信息性比较',
            'comparisons': [{'metric': 'accuracy', 'status': 'reference_only',
                'reference': {'value': 84.0, 'std': 0.4, 'unit': 'percent',
                    'aggregation': 'mean', 'runs': 20, 'split': 'test',
                    'dataset': 'Example', 'method': 'Model', 'value_quote': 'Model 84.0 ± 0.4',
                    'column_quote': 'Example Fixed Random', 'citations': ['paper']},
                'actual': {'value': 0.837, 'std': 0.005, 'unit': 'fraction',
                    'aggregation': 'mean', 'runs': 20, 'split': 'test',
                    'source': 'train:stdout:2', 'source_quote': 'Test Accuracy: 0.837 ± 0.005'},
                'settings': {'status': 'matched', 'reason': '源码与引文均说明固定划分、20 次运行。',
                    'assessment_basis': 'model_with_citations'},
                'difference': -0.3, 'difference_unit': 'percentage_points',
                'relative_difference': -0.003571, 'limitations': []}],
            'citations': [{'id': 'paper', 'source': 'pdf', 'page': 4}]}


def render(value):
    return '\n'.join(_author_claim_comparison_lines(value))


def test_verified_column_path_is_shown_from_structured_evidence():
    value = comparison()
    value['comparisons'][0]['reference']['column_evidence'] = {
        'status': 'verified', 'column_index': 0, 'column_labels': ['Example', 'Fixed'],
        'basis': 'source_table_headers', 'reason': '完整表头确定该单元格。'}
    text = render(value)
    assert '**论文列位置核验**: 已核验' in text
    assert '**已核验列路径**: Example → Fixed' in text
    assert '完整表头确定该单元格。' in text


@pytest.mark.parametrize('state,label', [('unknown', '未完成'), ('rejected', '未通过')])
def test_unverified_column_never_presents_candidate_path_as_verified(state, label):
    value = comparison()
    value['comparisons'][0]['reference'].update(column_verified=True, column_evidence={
        'status': state, 'column_labels': ['Example', 'Random'], 'reason': '多层表头证据不足。'})
    text = render(value)
    assert '**论文列位置核验**: ' + label in text
    assert '**已核验列路径**' not in text
    assert '多层表头证据不足。' in text
    assert '仅供参考，尚未验收' in text


@pytest.mark.parametrize('legacy,label', [(True, '已核验'), (False, '未完成'), (None, '未记录')])
def test_legacy_column_status_does_not_invent_a_path(legacy, label):
    value = comparison()
    if legacy is not None:
        value['comparisons'][0]['reference']['column_verified'] = legacy
    text = render(value)
    assert '**论文列位置核验**: ' + label in text
    assert '**已核验列路径**' not in text
    assert 'Example Fixed Random' in text


def test_matched_model_assessment_is_separate_from_independent_protocol_verification():
    text = render(comparison())
    assert '所引设置相符' in text
    assert '带引文的模型分析' in text
    assert '不等于独立协议核验通过' in text
    assert '尚未独立复算' in text
    assert '20 次' in text and '± 0.005' in text


def test_legacy_missing_settings_and_column_evidence_remain_unknown():
    value = comparison()
    value['comparisons'][0].pop('settings')
    text = render(value)
    assert '**论文列位置核验**: 未记录' in text
    assert '设置证据不足或未记录' in text
    assert '带引文的模型分析' not in text
    assert '**字段核对' not in text


@pytest.mark.parametrize('kind,level,label', [
    ('confidence_interval', 95.0, '95% 置信区间半宽'),
    ('confidence_interval', None, '置信区间半宽'),
    ('bootstrap_confidence_interval', 95, '95% bootstrap 置信区间半宽'),
    ('standard_error', None, '标准误'),
    ('unknown', None, '误差含义未明确'),
])
def test_reported_uncertainty_retains_its_declared_kind(kind, level, label):
    node = {'value': 0.837, 'reported_uncertainty': 0.015, 'uncertainty_kind': kind}
    if level is not None:
        node['confidence_level'] = level
    assert _metric_value_text(node, '（比例）') == f'0.837 ± 0.015（比例）（{label}）'
    assert node.get('std') is None


def test_standard_deviation_remains_separate_from_confidence_and_standard_error():
    assert _metric_value_text({'value': 0.837, 'std': 0.005}) == '0.837 ± 0.005'


def test_field_statuses_and_distinct_original_quotations_are_preserved_concisely():
    value = comparison()
    row = value['comparisons'][0]
    support = {'status': 'verified', 'quote': 'mean accuracy over 20 runs', 'citation': 'paper'}
    row['reference']['field_evidence'] = {'unit': {'status': 'unknown'},
        'split': {'status': 'unknown'}, 'aggregation': support, 'runs': deepcopy(support)}
    row['actual']['field_evidence'] = {'split': {'status': 'contradicted',
        'quote': 'validation accuracy: 0.837', 'citation': 'train:stdout:2'}}
    text = render(value)
    assert '**字段核对（论文）**: 单位：未知；划分：未知；统计方式：已核验；重复次数：已核验' in text
    assert '**字段依据（论文：统计方式、重复次数）**: paper；mean accuracy over 20 runs' in text
    assert text.count('mean accuracy over 20 runs') == 1
    assert '**字段核对（实际）**: 单位：未记录；划分：与原文矛盾' in text
    assert 'train:stdout:2；validation accuracy: 0.837' in text


def test_long_field_quotation_is_explicitly_an_excerpt_and_escapes_markup():
    value = comparison()
    quote = '|[original] ' + 'x' * 450
    value['comparisons'][0]['reference']['field_evidence'] = {
        'unit': {'status': 'verified', 'quote': quote, 'citation': 'paper'}}
    text = render(value)
    assert r'paper；\|\[original\]' in text
    assert '（引用节选）' in text
    assert quote not in text


def test_column_evidence_plain_text_cannot_break_markdown_or_change_input():
    value = comparison()
    value['comparisons'][0]['reference']['column_evidence'] = {
        'status': 'verified', 'column_labels': ['Example | Other', '[Fixed](https://example.org)'],
        'reason': 'line one\n**line two**'}
    before = deepcopy(value)
    text = render(value)
    assert r'Example \| Other → \[Fixed\]\(https://example.org\)' in text
    assert r'line one \*\*line two\*\*' in text
    assert value == before


def test_page_markdown_and_zip_downloads_share_column_and_protocol_explanation(tmp_path):
    value = comparison()
    value['comparisons'][0]['reference']['column_evidence'] = {
        'status': 'verified', 'column_labels': ['Example', 'Fixed']}
    report = render(value)
    path = tmp_path / 'comparison.md'
    path.write_text(report, encoding='utf-8')
    result = {'state': 'COMPLETED', 'report_path': str(path), 'data': {'report': report}}
    page, downloads = Mock(), Mock()
    render_report(report, path, st_module=page)
    report_downloads(result, key='comparison', st_module=downloads)
    shown = ''.join(call.args[0] for call in page.markdown.call_args_list)
    assert shown == report
    payloads = {call.kwargs['mime']: call.kwargs['data']
                for call in downloads.download_button.call_args_list}
    assert payloads['text/markdown'] == report
    with zipfile.ZipFile(io.BytesIO(payloads['application/zip'])) as archive:
        assert archive.read(path.name).decode('utf-8') == report
    assert 'Example → Fixed' in shown and '不等于独立协议核验通过' in shown
