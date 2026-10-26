"""Readable experiment reports retain unabridged evidence as verified details."""
from copy import deepcopy
import hashlib
import io
import json
import re
from unittest.mock import Mock
import zipfile

import pytest

from frontend.report_renderer import build_report_bundle, report_downloads
from src.agents.report_generator import ReportGeneratorAgent
from src.report_revisions import REPORT_FORMAT_VERSION, report_format_version


@pytest.fixture(autouse=True)
def forbid_models_and_execution(monkeypatch):
    from src.agents.code_executor import CodeExecutorAgent
    from src.llm.llm_client import LLMClient
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError('Report presentation must not call models or start experiments')

    monkeypatch.setattr(ReportGeneratorAgent, '__init__', forbidden)
    monkeypatch.setattr(ReportGeneratorAgent, 'run', forbidden)
    monkeypatch.setattr(LLMClient, '__init__', forbidden)
    monkeypatch.setattr(CodeExecutorAgent, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


@pytest.fixture
def experiment_data():
    quote = '\n'.join(f'逐字原文长段第{index}句，包含论文与作者实现的完整依据。' for index in range(600))
    stdout = '\n'.join(f'raw epoch {index} loss=0.384 final-output-marker-{index}' for index in range(600))
    code = '\n'.join(f"print('only-detail-code-{index}')" for index in range(600))
    source = {'source_id': 'paper_table2', 'locator': 'Table 2',
              'url': 'https://example.org/paper#table2'}
    reference = {'source_id': 'paper_table2', 'locator': 'Table 2', 'quote': quote}
    comparison = {
        'status': 'compared', 'basis': 'author_output', 'relative_tolerance': 0.05,
        'reason': '仅对照原作者输出，不替代独立模型复算。',
        'comparisons': [{
            'metric': 'mse', 'status': 'comparable', 'meets_reference': True,
            'difference': 0.009, 'relative_difference': 0.024,
            'reference': {'value': 0.375, 'unit': 'scalar', 'aggregation': 'single',
                          'dataset': 'ETTh1', 'method': 'DLinear', 'split': 'test',
                          'value_quote': quote, 'column_quote': 'ETTh1 / 96 / MSE',
                          'citations': ['paper'],
                          'field_evidence': {'unit': {'status': 'verified', 'citation': 'paper', 'quote': quote}}},
            'actual': {'value': 0.384, 'unit': 'scalar', 'aggregation': 'single',
                       'split': 'test', 'source': 'train:stdout:600', 'source_quote': stdout},
            'settings': {'status': 'matched', 'assessment_basis': 'model_with_citations',
                         'reason': '相同方法、数据集和预测范围。'},
            'limitations': ['仅覆盖 ETTh1 的 336→96 实验，不代表论文全部结果。'],
        }],
        'citations': [{'id': 'paper', 'source': 'pdf', 'page': 3}],
    }
    return {
        'paper_info': {'title': 'Readable selected experiment', 'method': 'DLinear', 'dataset': 'ETTh1'},
        'env_config': {'python_version': '3.12', 'device': 'cpu', 'requirements_txt': 'numpy==1.26.4'},
        'grounded_repository_plan': {'scope': 'author_code_execution', 'steps': [
            {'id': 'train', 'cwd': '.', 'argv': ['python', 'train.py', '--epochs', '40']}]},
        'execution': {'mode': 'repository', 'executed': True, 'code': code,
                      'steps': [{'id': 'train', 'cwd': '.', 'argv': ['python', 'train.py', '--epochs', '40'],
                                 'executed': True, 'success': True, 'exit_code': 0}],
                      'final': {'success': True, 'exit_code': 0, 'stdout': stdout,
                                'stderr': 'raw diagnostic detail marker'}},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None,
                       'paper_metrics_verified': False, 'reviews_removed': True,
                       'author_claim_comparison': comparison,
                       'metric_records': [{'name': 'mse', 'value': 0.384, 'unit': 'scalar',
                                           'split': 'test', 'stage': 'author', 'source': 'stdout'}]},
        'repository_analysis': {'status': 'accepted', 'model': 'saved-analysis-model',
                                'sources': [source], 'analyses': {
            'reader': {'protocol': {'seq_len': 336, 'pred_len': 96, 'epochs': 40},
                       'evidence': {'protocol.seq_len': [reference]}}}},
        'analysis_status': 'completed',
        'agent_collaboration': {'run_id': 'readable-report', 'messages': [
            {'id': 'message1', 'sender': 'PaperReader', 'recipient': 'ResourceFinder',
             'content': {'long_evidence': 'FULL COLLABORATION MESSAGE ' + 'retained message payload ' * 500}}]},
        'result_analysis': {'status': 'accepted',
                            'summary': '测试误差略高于论文参考，现有证据不能将偏差归因于单一原因。',
                            'limitations': ['仅覆盖所选实验，不能推广到整篇论文。'],
                            'sources': [source], 'evidence': {'summary': [reference]}},
    }


def generated(tmp_path, data):
    path = tmp_path / 'report.md'
    report = ReportGeneratorAgent.build_report(data, report_path=path)
    path.write_bytes(report.encode('utf-8'))
    return path, report


def test_readable_report_keeps_settings_metrics_analysis_and_moves_unabridged_evidence(
        tmp_path, experiment_data):
    original = deepcopy(experiment_data)
    path, report = generated(tmp_path, experiment_data)
    details = path.with_name('report_details.md').read_text(encoding='utf-8')
    assert experiment_data == original
    assert report_format_version(report) == REPORT_FORMAT_VERSION
    assert REPORT_FORMAT_VERSION >= 5
    assert '论文复现实验报告' in report
    for value in ('ETTh1', 'DLinear', '336', '96', 'train.py', '--epochs', '40', '0.375', '0.384'):
        assert value in report
    assert 'Table 2' in report or '第 3 页' in report
    assert experiment_data['result_analysis']['summary'] in report
    assert '仅覆盖所选实验，不能推广到整篇论文。' in report
    assert '尚未独立核验' in report or '独立核验尚未完成' in report
    for marker in ('逐字原文长段第0句', '逐字原文长段第599句',
                   'final-output-marker-0', 'final-output-marker-599',
                   'only-detail-code-0', 'only-detail-code-599', 'raw diagnostic detail marker'):
        assert marker not in report
        assert marker in details
    assert experiment_data['execution']['final']['stdout'] in details
    assert experiment_data['execution']['code'] in details
    assert experiment_data['agent_collaboration']['messages'][0]['content']['long_evidence'] in details
    assert 'FULL COLLABORATION MESSAGE' not in report
    assert len(report) > 800  # Preserve an actual experiment report, not only a short status line.
    assert re.search(r'\[完整依据与执行详情\]\(report_details\.md\) <!-- sha256:[0-9a-f]{64} -->', report)


@pytest.mark.parametrize('change', ['modified', 'missing'])
def test_details_attachment_rejects_missing_or_changed_content(tmp_path, experiment_data, change):
    from src.report_details import report_details_bytes

    path, report = generated(tmp_path, experiment_data)
    details_path = path.with_name('report_details.md')
    assert report_details_bytes(report, path) == {'report_details.md': details_path.read_bytes()}
    if change == 'modified':
        details_path.write_bytes(b'modified evidence')
    else:
        details_path.unlink()
    with pytest.raises((OSError, ValueError)):
        report_details_bytes(report, path)
    with pytest.raises((OSError, ValueError)):
        build_report_bundle(report, path, result={'state': 'COMPLETED', 'data': experiment_data})


def test_downloads_and_zip_use_same_readable_report_and_verified_details(tmp_path, experiment_data):
    path, report = generated(tmp_path, experiment_data)
    data = deepcopy(experiment_data)
    data.update(report=report, report_path=str(path), report_format_version=REPORT_FORMAT_VERSION)
    result = {'state': 'COMPLETED', 'error': None, 'report_path': str(path), 'data': data}
    ui = Mock()
    report_downloads(result, key='readable', st_module=ui)
    downloads = {call.kwargs['file_name']: call.kwargs for call in ui.download_button.call_args_list}
    assert downloads['report.md']['data'] == report
    details = downloads['report_details.md']['data']
    details = details.encode('utf-8') if isinstance(details, str) else details
    assert details == (tmp_path / 'report_details.md').read_bytes()
    with zipfile.ZipFile(io.BytesIO(downloads['report.zip']['data'])) as bundle:
        assert bundle.read('report.md').decode('utf-8') == result['data']['report']
        assert bundle.read('report_details.md') == details
        assert b'final-output-marker-599' in details
        evidence = json.loads(bundle.read('report.evidence.json'))
        assert evidence['data']['validation'] == result['data']['validation']
        assert evidence['data']['reproduction_conclusion']['is_reproduced'] is None
        manifest = json.loads(bundle.read('report.manifest.json'))
        entry = next(item for item in manifest['files'] if item['path'] == 'report_details.md')
        assert entry['sha256'] == hashlib.sha256(details).hexdigest()
        assert entry['bytes'] == len(details)
    ui.warning.assert_not_called()
