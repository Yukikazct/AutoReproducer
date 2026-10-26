"""Recovery reports preserve full saved investigations without upgrading success."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from frontend.report_renderer import build_report_bundle
from src.agents.report_generator import ReportGeneratorAgent
from src.report_details import intelligent_recovery_summary, report_details_bytes
from src.report_revisions import regenerate_report
from src.result_conclusion import build_reproduction_conclusion
from test_report_readability import experiment_data, forbid_models_and_execution


@pytest.fixture
def recovery():
    return {
        'version': 1, 'enabled': True, 'mode': 'collaborative', 'status': 'stopped',
        'identity': {'source_sha256': 'frozen-source', 'protocol_sha256': 'frozen-protocol'},
        'budget': {'max_model_requests': None, 'max_tokens': None,
                   'max_probes': 4, 'max_source_repair_rounds': 2, 'max_repair_actions': 8},
        'tasks': [{'id': 'task-1', 'trigger': '作者环境安装失败', 'status': 'stopped',
                   'hypotheses': [{'text': 'full saved hypothesis ' * 700}],
                   'probes': [{'evidence': {'value': 'RAW INVESTIGATION EVIDENCE ' * 700},
                               'status': 'completed'}],
                   'decisions': [{'action': 'stop', 'reason': '受限调查仍未定位可接受的兼容性修复'}]}],
        'usage': {'model_requests': 2, 'total_tokens': None}, 'requires_human': True,
        'stop_reason': '无进展', 'artifact_path': '/saved/run/intelligent_recovery.json',
    }


def test_readable_summary_and_zip_preserve_full_recovery_without_upgrading_conclusion(
        tmp_path, experiment_data, recovery):
    experiment_data['intelligent_recovery'] = recovery
    experiment_data['validation']['is_reproduced'] = None
    original = deepcopy(experiment_data)
    expected_conclusion = build_reproduction_conclusion(experiment_data)
    path = tmp_path / 'report.md'
    report = ReportGeneratorAgent.build_report(experiment_data, report_path=path)
    path.write_text(report, encoding='utf-8')
    assert experiment_data == original
    assert '## 智能调查与处理结果' in report
    for value in ('作者环境安装失败', '受限调查仍未定位可接受的兼容性修复', '无进展', '需要人工处理', 'unknown'):
        assert value in report
    assert 'RAW INVESTIGATION EVIDENCE' not in report
    assert 'full saved hypothesis' not in report
    summary = report.split('## 智能调查与处理结果', 1)[1].split('## 完整依据与日志', 1)[0]
    assert len(summary) < 1000
    details = report_details_bytes(report, path)['report_details.md']
    assert recovery['tasks'][0]['probes'][0]['evidence']['value'].encode() in details
    assert recovery['tasks'][0]['hypotheses'][0]['text'].encode() in details
    assert recovery['identity']['protocol_sha256'].encode() in details
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, path, result={
            'state': 'COMPLETED', 'data': experiment_data}))) as archive:
        assert archive.read('report_details.md') == details
        evidence = json.loads(archive.read('report.evidence.json'))
        assert evidence['data']['intelligent_recovery'] == recovery
        assert evidence['data']['reproduction_conclusion'] == expected_conclusion
        assert evidence['data']['reproduction_conclusion']['is_reproduced'] is None
        manifest = json.loads(archive.read('report.manifest.json'))
        entry = next(item for item in manifest['files'] if item['path'] == 'report_details.md')
        assert entry['sha256'] == hashlib.sha256(details).hexdigest()


def test_historical_regeneration_preserves_saved_recovery_and_original_result(
        tmp_path, experiment_data, recovery):
    report_path = tmp_path / 'report.md'
    report_path.write_text('# Legacy report', encoding='utf-8')
    experiment_data.update(intelligent_recovery=recovery, report='# Legacy report',
                           report_path=str(report_path))
    original = {'state': 'COMPLETED', 'report_path': str(report_path), 'data': experiment_data}
    saved = json.dumps(original, ensure_ascii=False).encode('utf-8')
    result_path = tmp_path / 'result.json'
    result_path.write_bytes(saved)
    before = deepcopy(original)
    upgraded = regenerate_report(original)
    assert original == before
    assert result_path.read_bytes() == saved
    assert report_path.read_text(encoding='utf-8') == '# Legacy report'
    assert upgraded['data']['intelligent_recovery'] == recovery
    assert upgraded['data']['validation'] == experiment_data['validation']
    assert upgraded['data']['reproduction_conclusion']['is_reproduced'] is None
    assert '智能调查与处理结果' in upgraded['data']['report']
    attachments = report_details_bytes(upgraded['data']['report'], Path(upgraded['report_path']))
    assert any(recovery['tasks'][0]['hypotheses'][0]['text'].encode() in raw
               for raw in attachments.values())


def test_recovery_summary_is_bounded_plain_text_and_disabled_history_is_supported(recovery):
    assert intelligent_recovery_summary(None) == ''
    assert '未启用' in intelligent_recovery_summary({'enabled': False})
    recovery['tasks'] *= 30
    recovery['tasks'][0]['trigger'] = 'unsafe <script>\n[click](https://invalid) ' * 1000
    summary = intelligent_recovery_summary(recovery)
    assert len(summary) < 1600
    assert '\\<script\\>' in summary
    assert '<script>' not in summary
    assert '[click](' not in summary
    assert '其余 27 项' in summary
