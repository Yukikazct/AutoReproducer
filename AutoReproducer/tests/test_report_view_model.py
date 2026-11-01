"""V6 presents observed recovery and science separately across saved channels."""
from copy import deepcopy
import io
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

from frontend.report_renderer import build_report_bundle, render_report
from src.agents.report_generator import ReportGeneratorAgent
from src.report_details import report_metric_lines
from src.report_revisions import regenerate_report
from src.report_view_model import ReportViewModel, build_report_view_model


def example_data():
    return {'paper_info': {'title': 'Graph experiment', 'method': 'GCN', 'dataset': 'Cora'},
            'execution': {'executed': True, 'mode': 'repository',
                          'final': {'success': True, 'exit_code': 0, 'stdout': 'unabridged log'}},
            'validation': {'status': 'author_code_completed', 'paper_metrics_verified': False,
                           'metric_records': [{'name': 'accuracy', 'value': 0.81, 'unit': 'fraction', 'split': 'test'}]},
            'intelligent_recovery': {'enabled': True, 'mode': 'collaborative', 'status': 'observed',
                'usage': {'roles': {'Investigator': {'model_requests': 1, 'total_tokens': None,
                                                     'duration_seconds': 2.0}}},
                'actions': [
                    {'id': 'one', 'kind': 'repair_environment', 'phase': 'environment',
                     'role': 'Repairer', 'problem': 'missing module', 'evidence': ['install:stderr'],
                     'status': 'verified', 'review': {'status': 'accepted'},
                     'execution': {'status': 'completed', 'original_command_rerun': True},
                     'verification': {'success': True, 'reason': 'original command completed'}},
                    {'id': 'two', 'kind': 'apply_patch', 'phase': 'execution', 'status': 'rejected',
                     'review': {'action': 'reject', 'reason': 'would alter training protocol'}},
                    {'id': 'three', 'kind': 'read_source', 'phase': 'planning', 'status': 'verified',
                     'verification': {'success': True}},  # No actual execution receipt.
                    {'id': 'four', 'kind': 'retry', 'phase': 'execution', 'status': 'failed',
                     'execution': {'status': 'failed', 'error': 'same error'},
                     'verification': {'success': False}},
                ]}}


def test_recovery_requires_execution_receipt_and_never_upgrades_paper_success():
    data = example_data()
    data['report_view_model'] = {'conclusion': {'is_reproduced': True}}
    original = deepcopy(data)
    view = ReportViewModel.from_data(data)
    assert data == original
    assert view.conclusion['is_reproduced'] is None
    assert view.conclusion['independent']['status'] == 'unavailable'
    assert [row['executed'] for row in view.recovery_contributions] == [True, False, False, True]
    assert [row['verified'] for row in view.recovery_contributions] == [True, False, False, False]
    assert view.recovery_contributions[3]['status'] == 'failed'
    assert view.recovery_contributions[2]['status'] == 'proposed'
    assert '执行 2 项' in view.answers['recovery'] and '1 项已验证' in view.answers['recovery']


def test_action_payloads_stay_in_details_while_main_uses_evidence_references(tmp_path):
    data = example_data()
    raw = 'RAW PROBE OUTPUT ' * 400
    data['intelligent_recovery']['actions'][0]['evidence'] = {'diagnostic': {'stdout': raw}, 'path': 'prepare.log'}
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    assert 'prepare.log' in report and 'RAW PROBE OUTPUT' not in report
    assert raw in (tmp_path / 'report_details.md').read_text(encoding='utf-8')


def test_metric_table_retains_zero_missing_units_split_runs_and_comparability():
    data = {'validation': {'author_claim_comparison': {'comparisons': [
        {'metric': 'error', 'status': 'comparable', 'difference': -0.01,
         'reference': {'value': 0.01, 'unit': 'fraction', 'split': 'test', 'runs': 3},
         'actual': {'value': 0.0, 'unit': 'fraction', 'split': 'test', 'run_count': 3}},
        {'metric': 'accuracy', 'status': 'reference_only',
         'reference': {'value': 0.8, 'unit': 'fraction'}, 'actual': {'value': None}},
    ]}}}
    view = build_report_view_model(data)
    assert view['metrics'][0]['actual'] == 0.0
    assert view['metrics'][1]['actual'] is None
    shown = '\n'.join(report_metric_lines(view['metrics']))
    assert '| error | 0.01 | 0.0 | fraction / fraction | test / test | 3 / 3 | -0.01 | 可比较 |' in shown
    assert '| accuracy | 0.8 | 未取得 |' in shown
    assert '仅供参考' in shown


def test_unpaired_metric_does_not_gain_difference_or_comparability_and_frozen_reference_survives():
    view = build_report_view_model({'paper_info': {'metrics': {'accuracy': 0.9}},
        'validation': {'metric_records': [{'name': 'accuracy', 'value': 0.81}]},
        'grounded_repository_plan': {'metrics': [{'name': 'loss', 'reference': 0.5, 'unit': 'scalar'}]}})
    accuracy, loss = view['metrics']
    assert accuracy['reference'] == 0.9 and accuracy['actual'] == 0.81
    assert accuracy['comparability'] == 'unrecorded' and accuracy['difference'] is None
    assert loss['reference'] == 0.5 and loss['actual'] is None


def author_coverage_data(records, *, actual=None, other_records=None):
    actual = deepcopy(records[0] if actual is None else actual)
    return {'validation': {'metric_records': deepcopy(records if other_records is None else other_records),
        'author_claim_comparison': {'metric_records': deepcopy(records), 'comparisons': [{
            'metric': actual['name'], 'status': 'reference_only', 'actual': actual,
            'reference': {'name': actual['name'], 'value': .9, 'unit': 'fraction'},
            'difference': None, 'limitations': ['Reference comparison only.']}]}}}


def test_paired_metric_does_not_hide_other_actual_outputs_or_guess_paper_values(tmp_path):
    accuracy = {'id': 'train:stdout:1', 'name': 'accuracy', 'value': .81,
                'unit': 'fraction', 'split': 'test', 'runs': 3, 'aggregation': 'mean'}
    loss = {'id': 'train:stdout:2', 'name': 'loss', 'value': 0., 'unit': 'scalar', 'split': 'test'}
    data = author_coverage_data([accuracy, loss])
    data['paper_info'] = {'metrics': {'loss': 99.}}
    original = deepcopy(data)
    rows = build_report_view_model(data)['metrics']
    assert [row['metric'] for row in rows] == ['accuracy', 'loss']
    assert rows[0]['actual'] == .81
    assert rows[1]['actual'] == 0. and rows[1]['reference'] is None
    assert rows[1]['actual_unit'] == 'scalar' and rows[1]['actual_split'] == 'test'
    assert rows[1]['difference'] is None and rows[1]['comparability'] == 'unrecorded'
    assert '尚未找到对应论文参考' in rows[1]['reason']
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    assert '| loss | 未取得 | 0.0 |' in report
    assert data['validation'] == original['validation']


def test_same_metric_name_with_a_different_split_or_id_is_retained():
    test = {'id': 'train:stdout:1', 'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test'}
    train = {**test, 'id': 'train:stdout:2', 'split': 'train'}
    second_test = {**test, 'id': 'train:stdout:3'}
    data = author_coverage_data([test, train, second_test], other_records=[test, train, second_test, train])
    rows = build_report_view_model(data)['metrics']
    assert len(rows) == 3
    assert [(row['actual'], row['actual_split']) for row in rows] == [(.81, 'test'), (.81, 'train'), (.81, 'test')]
    assert all(row['reference'] is None and row['difference'] is None for row in rows[1:])


def test_legacy_without_ids_only_deduplicates_a_unique_exact_context():
    accuracy = {'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test',
                'run_count': 3, 'std': .01, 'aggregation': 'mean'}
    loss = {'name': 'loss', 'value': .2, 'unit': 'scalar', 'split': 'test'}
    data = author_coverage_data([accuracy, loss], other_records=[{**accuracy, 'runs': 3}, loss])
    rows = build_report_view_model(data)['metrics']
    assert [(row['metric'], row['actual']) for row in rows] == [('accuracy', .81), ('loss', .2)]
    assert rows[1]['reference'] is None
    # A missing split is insufficient to consume an otherwise unpaired output.
    data['validation']['author_claim_comparison']['comparisons'][0]['actual'].pop('split')
    assert [row['metric'] for row in build_report_view_model(data)['metrics']] == ['accuracy', 'accuracy', 'loss']


def test_ambiguous_legacy_records_are_retained_instead_of_consumed_by_one_pair():
    actual = {'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test'}
    data = author_coverage_data([actual, deepcopy(actual)], actual=actual, other_records=[])
    rows = build_report_view_model(data)['metrics']
    assert len(rows) == 3 and all(row['actual'] == .81 for row in rows)
    assert all(row['reference'] is None for row in rows[1:])


def test_id_matched_pair_cannot_also_consume_an_ambiguous_record_without_id():
    actual = {'id': 'train:stdout:1', 'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test'}
    legacy = {key: value for key, value in actual.items() if key != 'id'}
    rows = build_report_view_model(author_coverage_data([actual, legacy], other_records=[]))['metrics']
    assert len(rows) == 2 and rows[1]['actual'] == .81 and rows[1]['reference'] is None


def test_cross_collection_legacy_duplicate_retains_the_available_record_id():
    legacy = {'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test'}
    identified = {**legacy, 'id': 'train:stdout:1'}
    data = author_coverage_data([legacy], actual=identified, other_records=[identified])
    assert len(build_report_view_model(data)['metrics']) == 1
    data = author_coverage_data([identified], actual=identified, other_records=[legacy])
    assert len(build_report_view_model(data)['metrics']) == 1


def test_legacy_records_with_different_datasets_or_methods_are_not_merged():
    actual = {'name': 'accuracy', 'value': .81, 'unit': 'fraction', 'split': 'test',
              'dataset': 'Cora', 'method': 'GCN'}
    other_dataset = {**actual, 'dataset': 'Citeseer'}
    other_method = {**actual, 'method': 'GAT'}
    data = author_coverage_data([actual], other_records=[other_dataset, other_method])
    rows = build_report_view_model(data)['metrics']
    assert len(rows) == 3
    assert all(row['reference'] is None for row in rows[1:])


def test_unpaired_records_can_come_from_either_authoritative_collection():
    actual = {'id': 'test:1', 'name': 'accuracy', 'value': .81}
    loss = {'id': 'test:2', 'name': 'loss', 'value': .2}
    data = author_coverage_data([actual], other_records=[actual, loss])
    assert [row['metric'] for row in build_report_view_model(data)['metrics']] == ['accuracy', 'loss']
    data = author_coverage_data([actual, loss], other_records=[actual])
    assert [row['metric'] for row in build_report_view_model(data)['metrics']] == ['accuracy', 'loss']


def test_reference_only_difference_remains_visible_without_an_acceptance_verdict(tmp_path):
    data = example_data()
    row = {'metric': 'accuracy', 'status': 'reference_only', 'difference': -2.35,
           'difference_unit': 'percentage_points', 'within_tolerance': None, 'meets_reference': None,
           'reference': {'value': 87.84, 'unit': 'percent', 'split': 'test', 'runs': 10,
                         'method': 'Variant-Backbone'},
           'actual': {'value': .8549, 'unit': 'fraction', 'split': 'test', 'runs': None,
                      'method': 'Variant_Wrapper'},
           'method_alignment': {'status': 'source_bound_static', 'independently_verified': False},
           'limitations': ['静态方法组件对应；运行设置和统计次数尚未核对。']}
    data['validation']['author_claim_comparison'] = {
        'status': 'partial', 'comparisons': [row], 'basis': 'author_output'}
    view = build_report_view_model(data)
    assert view['metrics'][0]['difference'] == -2.35
    assert view['metrics'][0]['comparability'] == 'reference_only'
    assert view['conclusion']['comparison']['status'] == 'reference_only'
    assert view['conclusion']['is_reproduced'] is None
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    assert '-2.35 个百分点' in report and '仅供参考' in report
    assert 'Variant-Backbone' in report and 'Variant\\_Wrapper' in report
    assert '未证明运行期有效配置' in report
    assert data['validation']['paper_metrics_verified'] is False
    row['status'] = 'not_comparable'
    assert build_report_view_model(data)['metrics'][0]['difference'] is None


def test_different_metric_names_stay_visible_without_subtracting_values(tmp_path):
    data = example_data()
    data['validation']['author_claim_comparison'] = {'comparisons': [{
        'metric': 'accuracy', 'status': 'not_comparable', 'difference': None,
        'reference': {'name': 'f1_macro', 'value': 82.5, 'unit': 'percent'},
        'actual': {'name': 'accuracy', 'value': .84, 'unit': 'fraction'},
        'limitations': ['论文与实际指标不同，未计算差值']} ]}
    view = build_report_view_model(data)
    row = view['metrics'][0]
    assert row['metric'] == '论文 f1_macro / 实际 accuracy'
    assert row['reference'] == 82.5 and row['actual'] == .84 and row['difference'] is None
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    assert '论文 f1\\_macro / 实际 accuracy' in report
    assert '指标不同，未计算差值' in report
    assert view['conclusion']['is_reproduced'] is None


def test_scalar_author_comparisons_and_strict_only_rows_remain_visible():
    author = build_report_view_model({'validation': {'author_claim_comparison': {'comparisons': [
        {'metric': 'accuracy', 'reference': 80.0, 'actual': 0.82, 'status': 'comparable',
         'reference_unit': 'percent', 'actual_unit': 'fraction', 'difference': 2.0,
         'difference_unit': 'percentage_points'}]}}})
    row = author['metrics'][0]
    assert row['reference'] == 80.0 and row['actual'] == 0.82
    assert '2.0 个百分点' in '\n'.join(report_metric_lines(author['metrics']))
    strict = build_report_view_model({'validation': {'comparisons': [
        {'name': 'accuracy', 'reference': {'value': 0.84, 'unit': 'fraction', 'split': 'test', 'runs': 3},
         'actual': {'value': 0.85, 'unit': 'fraction', 'split': 'test', 'runs': 3}, 'pass': True},
        {'name': 'loss', 'reference': 0.5, 'actual': None, 'pass': False, 'unit': 'scalar'}]}})
    accuracy, loss = strict['metrics']
    assert accuracy['reference'] == 0.84 and accuracy['actual'] == 0.85
    assert accuracy['actual_runs'] == 3 and accuracy['reference_split'] == 'test'
    assert abs(accuracy['difference'] - 0.01) < 1e-12
    assert loss['reference'] == 0.5 and loss['actual'] is None


def test_deterministic_comparisons_override_older_metric_payloads():
    view = build_report_view_model({'paper_info': {'metrics': {'accuracy': 0.9}},
        'validation': {'metrics_comparison': {'actual': {'accuracy': 0.1}},
                       'comparisons': [{'name': 'accuracy', 'reference': 0.84,
                                        'actual': 0.85, 'pass': True}]}})
    assert view['metrics'][0]['reference'] == 0.84
    assert view['metrics'][0]['actual'] == 0.85


def test_unavailable_strict_verification_retains_author_completion_and_visible_reason(tmp_path):
    data = example_data()
    data['strict_validation'] = {'requested': True, 'status': 'unavailable',
                                'reason': 'No independent capture for this framework'}
    view = build_report_view_model(data)
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    assert view['answers']['completion'] == '已完成运行'
    assert view['conclusion']['independent']['status'] == 'unavailable'
    assert view['conclusion']['independent']['reason'] == data['strict_validation']['reason']
    assert '独立核验尚未完成' in report and '作者仓库代码运行完成' in report
    assert data['strict_validation']['reason'] in report


def test_stopped_and_no_progress_actions_retain_executed_evidence_without_success():
    data = example_data()
    recovery = data['intelligent_recovery']
    recovery['actions'] = [
        {'id': 'one', 'kind': 'repair', 'status': 'stopped',
         'execution': {'status': 'stopped', 'original_command_rerun': True},
         'verification': {'success': True, 'reason': 'stale candidate acceptance'}},
        {'id': 'two', 'kind': 'repair', 'status': 'no_progress',
         'execution': {'status': 'completed'}, 'verification': {'success': False}},
        {'id': 'three', 'kind': 'read', 'status': 'verified',
         'execution': {'status': 'not_executed'}, 'verification': {'success': True}},
    ]
    recovery['unresolved_issues'] = [{'status': 'unresolved', 'problem': 'original data still missing'}]
    view = build_report_view_model(data)
    assert [row['executed'] for row in view['recovery_contributions']] == [True, True, False]
    assert [row['status'] for row in view['recovery_contributions']] == ['stopped', 'no_progress', 'proposed']
    assert not any(row['verified'] for row in view['recovery_contributions'])
    assert 'original data still missing' in view['answers']['remaining']


def test_failed_action_shows_actual_error_before_unverified_placeholder():
    view = build_report_view_model({'intelligent_recovery': {'enabled': True, 'actions': [
        {'kind': 'replan', 'status': 'failed', 'execution': {'status': 'failed', 'error': 'entrypoint is missing'},
         'verification': {'success': None, 'reason': 'Not verified'}}]}})
    assert view['recovery_contributions'][0]['result'] == 'entrypoint is missing'


def test_preparation_and_planning_receipts_cannot_claim_an_author_command_rerun():
    phases = ['environment_preparation', 'dataset_preparation', 'entrypoint_planning',
              'paper_parsing', 'repository_discovery', 'result_evidence', 'environment']
    actions = [{'phase': phase, 'status': 'verified',
                'execution': {'status': 'completed', 'original_command_rerun': True},
                'verification': {'success': True}} for phase in phases]
    # A recovered-stage record also overrides an older, contradictory stage.
    actions.append({'phase': 'author_execution', 'recovered_phase': 'dataset_preparation',
                    'execution': {'status': 'completed', 'original_command_rerun': True},
                    'verification': {'success': True}})
    actions.append({'phase': 'author_execution',
                    'execution': {'status': 'completed', 'original_command_rerun': True},
                    'verification': {'success': True}})
    actions.append({'execution': {'status': 'completed', 'original_command_rerun': True},
                    'verification': {'success': True}})
    view = build_report_view_model({'intelligent_recovery': {'enabled': True, 'actions': actions}})
    rows = view['recovery_contributions']
    assert all(row['rerun'] is False for row in rows[:-2])
    assert all(row['verified'] is True for row in rows)
    assert [row['rerun'] for row in rows[-2:]] == [True, True]


def test_preparation_recovery_does_not_repeat_contradictory_author_rerun_reason():
    for phase, reason in [('environment_preparation', 'Original author command rerun; numerical equivalence unknown'),
                          ('dataset_preparation', '修复后的原作者命令已实际重跑并正常退出。')]:
        data = {'intelligent_recovery': {'enabled': True, 'actions': [{
            'phase': phase, 'status': 'verified',
            'execution': {'status': 'completed', 'original_command_rerun': True},
            'verification': {'success': True, 'reason': reason}}]}}
        before = deepcopy(data)
        row = build_report_view_model(data)['recovery_contributions'][0]
        assert row['rerun'] is False
        assert '作者原命令重跑未核验' in row['result'] and reason not in row['result']
        assert row['verification']['reason'] == reason and data == before
        data['intelligent_recovery']['actions'][0].pop('phase')
        assert build_report_view_model(data)['recovery_contributions'][0]['result'] == reason


def test_history_upgrade_keeps_preparation_recovery_separate_from_author_execution(tmp_path):
    path = tmp_path / 'report.md'
    path.write_text('# Legacy', encoding='utf-8')
    data = example_data()
    recovery = data['intelligent_recovery']
    action = recovery['actions'][0]
    action['phase'] = 'environment_preparation'
    recovery['tasks'] = [{'trigger': 'execution_failure', 'status': 'command_recovered',
                          'recovered_phase': 'environment_preparation'}]
    data.update(report='# Legacy', report_path=str(path), self_repair={
        'status': 'resolved', 'reason': '修复后的原命令已实际重跑并正常退出。',
        'attempts': [{'round': 1, 'status': 'resolved', 'action': 'install_missing_dependency',
                      'reason': '重试原命令'}]})
    source = {'state': 'COMPLETED', 'report_path': str(path), 'data': data}
    original = deepcopy(source)
    upgraded = regenerate_report(source)
    report = upgraded['data']['report']
    assert source == original and path.read_text(encoding='utf-8') == '# Legacy'
    assert upgraded['data']['report_view_model']['recovery_contributions'][0]['rerun'] is False
    assert '环境准备已恢复，作者实验运行另行记录' in report
    assert '环境预检执行失败' in report and '作者命令执行失败' not in report
    assert '原命令运行已恢复' not in report and '原命令已实际重跑' not in report
    assert '受控步骤命令重跑通过' in report and '历史记录未保存修复阶段' in report
    assert 'original command completed' not in report and '作者原命令重跑未核验' in report
    details = (Path(upgraded['report_path']).parent / (Path(upgraded['report_path']).stem + '_details.md'))
    assert 'original command completed' in details.read_text(encoding='utf-8')
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, upgraded['report_path'], result=upgraded))) as bundle:
        evidence = json.loads(next(bundle.read(name) for name in bundle.namelist() if name.endswith('.evidence.json')))
        assert evidence['data']['report_view_model']['recovery_contributions'][0]['rerun'] is False


def test_failed_receipts_and_rejected_actions_cannot_be_verified_even_with_stale_true_flag():
    view = build_report_view_model({'intelligent_recovery': {'enabled': True, 'actions': [
        {'kind': 'repair', 'status': 'failed', 'execution': {'status': 'failed'},
         'verification': {'success': True}},
        {'kind': 'repair', 'status': 'rejected', 'execution': {'status': 'completed'},
         'verification': {'success': True}},
        {'kind': 'repair', 'status': 'verified', 'execution': {'status': 'completed',
             'result': {'process_cleanup': {'confirmed': False}}}, 'verification': {'success': True}},
    ]}})
    assert not any(row['verified'] for row in view['recovery_contributions'])
    assert [row['status'] for row in view['recovery_contributions']] == ['failed', 'rejected', 'stopped']


def test_generic_main_uses_one_evidence_template_and_no_string_filter(tmp_path, monkeypatch):
    import src.report_details as details
    raw_renderer = ReportGeneratorAgent._build_report
    calls = []
    def render(*args, **kwargs):
        calls.append(True)
        return raw_renderer(*args, **kwargs)
    monkeypatch.setattr(ReportGeneratorAgent, '_build_report', staticmethod(render))
    monkeypatch.setattr(details, 'readable_text', lambda *args: (_ for _ in ()).throw(AssertionError('legacy filter')))
    report = ReportGeneratorAgent.build_report(example_data(), report_path=tmp_path / 'report.md')
    assert len(calls) == 1
    assert 'format: 6' in report
    for prompt in ('完成到哪一步', '是否达到参考', '自动修复解决了什么', '仍需处理什么'):
        assert prompt in report
    assert 'GCN' in report and 'Cora' in report
    assert 'DLinear' not in report and 'ETTh1' not in report
    assert 'MSE、MAE' not in report
    assert 'unabridged log' not in report
    assert 'unabridged log' in (tmp_path / 'report_details.md').read_text(encoding='utf-8')
    assert '已否决，未执行' in report and '执行失败' in report


def test_historical_overlay_ui_and_zip_share_v6_model_without_mutating_source(tmp_path):
    path = tmp_path / 'report.md'
    path.write_text('# Legacy', encoding='utf-8')
    data = example_data()
    data.update(report='# Legacy', report_path=str(path))
    source = {'state': 'COMPLETED', 'report_path': str(path), 'data': data}
    original = deepcopy(source)
    upgraded = regenerate_report(source)
    assert source == original and path.read_text(encoding='utf-8') == '# Legacy'
    view = upgraded['data']['report_view_model']
    assert view == build_report_view_model(data)
    ui = Mock()
    render_report(upgraded['data']['report'], upgraded['report_path'], st_module=ui)
    assert '恢复行动与协作贡献' in ''.join(call.args[0] for call in ui.markdown.call_args_list)
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(upgraded['data']['report'], upgraded['report_path'], result=upgraded))) as bundle:
        evidence = json.loads(next(bundle.read(name) for name in bundle.namelist() if name.endswith('.evidence.json')))
        assert evidence['data']['report_view_model'] == view
    old = build_report_view_model({'execution': data['execution'], 'validation': data['validation']})
    assert old['answers']['recovery'] == '未记录' and not old['recovery_recorded']


def test_late_cancellation_refreshes_saved_model_and_all_top_answers(tmp_path):
    from src.run_finalization import finalize_report_text
    data = example_data()
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    data['report_view_model'] = build_report_view_model(data)
    assert data['report_view_model']['answers']['completion'] == '已完成运行'
    data['cancelled'] = True
    report = finalize_report_text(report, data)
    assert data['report_view_model'] == build_report_view_model(data)
    assert data['report_view_model']['conclusion']['is_reproduced'] is None
    overview = report.split('<!-- autoreproducer-overview:start -->')[1].split('<!-- autoreproducer-overview:end -->')[0]
    assert '已完成运行' not in overview
    assert '取消' in overview or '中断' in overview
    assert report.count('## 任务结果与恢复') == 1
    assert finalize_report_text(report, data) == report


def test_late_cleanup_is_retained_outside_refreshed_overview(tmp_path):
    from src.run_finalization import finalize_report_text
    data = example_data()
    report = ReportGeneratorAgent.build_report(data, report_path=tmp_path / 'report.md')
    data.update(environment_cleanup={'status': 'error', 'errors': [{'reason': 'still open'}]},
                run_finalization={'state': 'ERROR'})
    report = finalize_report_text(report, data)
    assert report.count('### 任务收尾') == 1
    assert '**环境释放**: 失败' in report
    assert data['report_view_model']['conclusion']['lifecycle']['cleanup_status'] == 'error'
    assert report.index('## 任务结果与恢复') < report.index('## 本次复现结论')
    assert report.index('## 本次复现结论') < report.index('### 任务收尾')
    assert report.index('### 任务收尾') < report.index('## 选定实验与冻结依据')
    assert finalize_report_text(report, data) == report


def test_web_and_downloads_present_identical_model_derived_main_report(tmp_path):
    from frontend.report_renderer import report_downloads
    from frontend.display_text import report_display_text
    data = example_data()
    path = tmp_path / 'report.md'
    report = ReportGeneratorAgent.build_report(data, report_path=path)
    path.write_text(report, encoding='utf-8')
    data.update(report=report, report_path=str(path), report_view_model=build_report_view_model(data))
    result = {'state': 'COMPLETED', 'data': data, 'report_path': str(path)}
    web, downloads = Mock(), Mock()
    render_report(report, path, st_module=web)
    visible = ''.join(call.args[0] for call in web.markdown.call_args_list)
    assert visible == report_display_text(report)
    assert 'autoreproducer-overview' not in visible
    report_downloads(result, key='shared_vm', st_module=downloads)
    markdown = next(call.kwargs['data'] for call in downloads.download_button.call_args_list
                    if call.kwargs['file_name'] == 'report.md')
    zipped = next(call.kwargs['data'] for call in downloads.download_button.call_args_list
                  if call.kwargs['file_name'] == 'report.zip')
    assert markdown == report
    with zipfile.ZipFile(io.BytesIO(zipped)) as bundle:
        assert bundle.read('report.md').decode('utf-8') == report
        assert json.loads(bundle.read('report.evidence.json'))['data']['report_view_model'] == data['report_view_model']
