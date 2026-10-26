"""The exported record must reconcile with the report and portable payloads."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile
from unittest.mock import Mock

from frontend.report_renderer import build_report_bundle, report_downloads
from src.agents.report_generator import ReportGeneratorAgent


def completed():
    return {'state': 'COMPLETED', 'error': None, 'data': {
        'paper_info': {'title': 'A saved experiment'},
        'execution': {'executed': True, 'final': {'stage': 'eval', 'exit_code': 0,
                                                 'success': True, 'stdout': 'accuracy=0.845'}},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None,
                       'paper_metrics_verified': False},
        'reproduction_conclusion': {'status': 'reproduced', 'is_reproduced': True},
    }}


def test_bundle_hashes_all_payloads_and_copies_sanitized_current_evidence(tmp_path):
    result = completed()
    result['data']['execution']['final']['stdout'] += '\nsk-fixture-12345678901234567890'
    result['data']['env_config'] = {'nested': {'api_key': 'private-fixture-credential'}}
    original = deepcopy(result)
    path = tmp_path / '报告.md'
    report = '# Saved report\n'
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, path, result=result))) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('报告.manifest.json'))
        evidence_raw = archive.read('报告.evidence.json')
        evidence = json.loads(evidence_raw)
        assert evidence['data']['validation'] == original['data']['validation']
        assert evidence['data']['reproduction_conclusion']['is_reproduced'] is None
        assert 'private-fixture-credential' not in evidence_raw.decode('utf-8')
        assert 'sk-fixture-12345678901234567890' not in evidence_raw.decode('utf-8')
        assert evidence['report']['sha256'] == hashlib.sha256(archive.read(path.name)).hexdigest()
        assert {row['path'] for row in manifest['files']} == set(archive.namelist()) - {'报告.manifest.json'}
        for row in manifest['files']:
            raw = archive.read(row['path'])
            assert row['bytes'] == len(raw)
            assert row['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result == original


def test_evidence_does_not_promote_reference_match_or_late_cancellation(tmp_path):
    result = completed()
    result['data']['validation'].update(status='reproduced', is_reproduced=True,
        paper_metrics_verified=True, protocol_pass=True, independent_metrics_pass=True)
    result['data']['strict_validation'] = {'requested': True, 'status': 'verified'}
    result['data']['cancelled'] = True
    report = ReportGeneratorAgent._build_report(result['data'])
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, tmp_path / 'r.md', result=result))) as archive:
        evidence = json.loads(archive.read('r.evidence.json'))
        assert evidence['data']['reproduction_conclusion']['is_reproduced'] is None
        assert evidence['data']['reproduction_conclusion']['status'] == 'interrupted'
        assert evidence['data']['validation']['is_reproduced'] is True  # Preserve recorded evidence.


def test_web_download_uses_same_result_for_markdown_and_evidence(tmp_path):
    result = completed()
    report = ReportGeneratorAgent._build_report(result['data'])
    result['data']['report'] = report
    result['report_path'] = str(tmp_path / 'final.md')
    ui = Mock()
    report_downloads(result, key='final', st_module=ui)
    downloads = {call.kwargs['mime']: call.kwargs['data'] for call in ui.download_button.call_args_list}
    assert downloads['text/markdown'] == report
    with zipfile.ZipFile(io.BytesIO(downloads['application/zip'])) as archive:
        assert archive.read('final.md').decode('utf-8') == report
        evidence = json.loads(archive.read('final.evidence.json'))
        assert evidence['data']['validation'] == result['data']['validation']
        assert evidence['state'] == result['state']
