"""Presentation must preserve the boundaries of source-checked comparisons."""
from copy import deepcopy
import io
from unittest.mock import Mock
import zipfile

import pytest

from frontend.report_renderer import render_report, report_downloads
from src.agents.report_generator import (
    ReportGeneratorAgent, _author_claim_comparison_lines, _metric_value_text,
    _reproduction_assessment_lines,
)


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


def ai_assessment(verdict='reproduced'):
    return {'status': 'completed', 'source': 'ai', 'scope': 'executed_experiment',
            'verdict': verdict, 'confidence': 0.86,
            'reason': '实测准确率接近论文，允许随机种子造成的小幅波动。',
            'metric_assessments': [{'actual_id': 'train:stdout:0', 'verdict': verdict,
                'reason': '准确率相差 0.3 个百分点。',
                'tolerance': {'absolute': 0.5, 'relative': 0.01,
                              'reason': '结合论文标准差与单次运行的随机性。'}}],
            'limitations': ['实际为单次运行，论文报告多次均值。']}


@pytest.mark.parametrize('verdict,label', [
    ('reproduced', 'AI 判断复现成功'),
    ('partially_reproduced', 'AI 判断部分复现'),
    ('not_reproduced', 'AI 判断尚未复现'),
    ('insufficient_evidence', 'AI 判断证据不足'),
])
def test_ai_decision_controls_report_summary_without_claiming_strict_verification(verdict, label):
    value = comparison()
    value['comparisons'][0]['actual']['id'] = 'train:stdout:0'
    assessment = ai_assessment(verdict)
    data = {'execution': {'mode': 'repository', 'success': True},
            'grounded_repository_plan': {'scope': 'author_code_execution'},
            'validation': {'status': 'author_code_completed', 'is_reproduced': None,
                           'paper_metrics_verified': False, 'author_claim_comparison': value,
                           'reproduction_assessment': assessment}}
    before = deepcopy(data)
    report = ReportGeneratorAgent()._build_report(data)
    summary = report.split('## 5. 运行结果', 1)[1].split('\n', 2)[1]
    assert label in summary
    assert '**指标判断**: accuracy；' + label in report
    assert '模型置信度**: 86%' in report
    assert '相对误差 1.00%' in report and '绝对误差 0.5' in report
    assert '结合论文标准差与单次运行的随机性。' in report
    assert '实际为单次运行，论文报告多次均值。' in report
    assert '固定 5% 参考线不作为普通复现的硬门槛' in report
    assert '本次实际执行的实验' in report and '尚未独立复算' in report
    assert '仅供参考，尚未验收' not in report
    assert '不进行论文、协议或数值审核' not in report
    assert data == before


@pytest.mark.parametrize('state', ['unavailable', 'blocked'])
def test_unavailable_ai_preserves_completion_and_explains_missing_decision(state):
    assessment = {'status': state, 'reason': '模型响应尚未通过结构校验。'}
    report = ReportGeneratorAgent()._build_report({
        'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'reproduction_assessment': assessment}})
    assert '作者仓库代码运行完成' in report
    assert 'AI 判断尚未完成' in report and assessment['reason'] in report
    assert 'AI 判断复现成功' not in report


def test_ai_explanations_are_plain_text_and_do_not_change_the_assessment():
    value = ai_assessment()
    value['reason'] = '**override** [link](https://example.org)\n| new row'
    value['metric_assessments'][0]['tolerance']['reason'] = '<script>alert(1)</script>'
    value['limitations'] = ['[unsafe](file://C:/secret)']
    before = deepcopy(value)
    text = '\n'.join(_reproduction_assessment_lines(value))
    assert r'\*\*override\*\* \[link\]\(https://example.org\)' in text
    assert r'\| new row' in text and r'\<script\>' in text
    assert r'\[unsafe\]\(file://C:/secret\)' in text
    assert value == before
