import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from frontend.report_renderer import build_report_bundle
from src.agents.report_generator import ReportGeneratorAgent
from src import source_repair_evidence as evidence


@pytest.fixture
def archived(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'source_repair.json').write_text('{"status":"failed"}', encoding='utf-8')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'train.stderr.log').write_bytes(('actual failure\n' + 'sk-' + 'x' * 24).encode())
    records = run / '.autorepro_repository_repairs' / 'fixture'
    records.mkdir(parents=True)
    (records / 'repairs.json').write_text('{}', encoding='utf-8')
    (records / 'repair_report.md').write_text('Status: failed', encoding='utf-8')
    attempt = records / 'round_1'
    attempt.mkdir()
    (attempt / 'candidate.diff').write_bytes(b'-legacy\n+modern')
    (attempt / 'result.json').write_text('{"status":"rejected"}', encoding='utf-8')
    data = {'run_dir': str(run), 'source_repair': {'requested': True, 'status': 'failed',
        'editable': ['model.py'], 'records_path': str(records / 'repairs.json'),
        'attempts': [{'round': 1, 'status': 'rejected', 'reason': 'actual diagnostic'}]},
        'original_execution': {'run_dir': str(logs), 'steps': [{'id': 'train'}]}}
    evidence.archive_repair_evidence(run, data)
    return data


def test_archive_preserves_diff_complete_logs_and_redaction_manifest(archived):
    artifact = archived['source_repair']['evidence_archive']
    raw = Path(artifact['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == artifact['sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        manifest = json.loads(bundle.read('evidence_manifest.json'))
        log = bundle.read('logs/original/train.stderr.log')
        assert log == b'actual failure\n[REDACTED]'
        assert manifest['logs/original/train.stderr.log']['redacted'] is True
        assert bundle.read('repairs/round_1/candidate.diff') == b'-legacy\n+modern'
        for name, entry in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == entry['sha256']


def test_report_download_contains_verified_portable_repair_attachment(archived, tmp_path):
    report = tmp_path / 'report.md'
    text = ReportGeneratorAgent._build_report(archived, report_path=report)
    report.write_text(text, encoding='utf-8')
    assert '源码修复证据 ZIP' in text
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(text, report))) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith('.zip'))
        assert bundle.read(member) == Path(archived['source_repair']['evidence_archive']['path']).read_bytes()
    regenerated = tmp_path / 'revisions' / 'v2.md'
    regenerated.parent.mkdir()
    revised = ReportGeneratorAgent._build_report(archived, report_path=regenerated)
    assert evidence.report_evidence_bytes(revised, regenerated)


def test_corrupt_companion_is_rejected_by_download(archived, tmp_path):
    path = tmp_path / 'report.md'
    text = ReportGeneratorAgent._build_report(archived, report_path=path)
    companion = next((tmp_path / 'report_assets').glob('*.zip'))
    companion.write_bytes(b'changed')
    with pytest.raises(ValueError, match='哈希'):
        build_report_bundle(text, path)


@pytest.mark.parametrize('target', ['../private.zip', '/private.zip', 'report_assets/private.zip',
                                   'report_assets/../private.zip'])
def test_untrusted_report_attachment_cannot_read_arbitrary_paths(tmp_path, target):
    text = f'[源码修复证据 ZIP]({target}) <!-- sha256:{"a" * 64} -->'
    with pytest.raises(ValueError, match='路径'):
        evidence.report_evidence_bytes(text, tmp_path / 'report.md')


def test_archive_budget_preserves_originals_without_truncation(archived, monkeypatch):
    monkeypatch.setattr(evidence, 'MAX_EVIDENCE_BYTES', 1)
    before = Path(archived['source_repair']['evidence_archive']['path']).read_bytes()
    with pytest.raises(ValueError, match='64 MiB'):
        evidence.archive_repair_evidence(archived['run_dir'], archived)
    assert Path(archived['source_repair']['evidence_archive']['path']).read_bytes() == before


def test_json_credentials_are_redacted_but_public_repository_identity_is_preserved():
    from src.repository_repairer import _private
    text = '{"password":"plain-password", "OPENAI_API_KEY":"non-token-provider-value", "url":"https://github.com/example/project"}'
    safe = json.loads(_private(text))
    assert safe['password'] == '[REDACTED]'
    assert safe['OPENAI_API_KEY'] == '[REDACTED]'
    assert safe['url'] == 'https://github.com/example/project'
    assert _private({'password': 'plain-password'}) == {'password': '[REDACTED]'}
    assert 'token=value' not in _private('https://github.com/example/project?token=value')


def test_report_diagnostics_do_not_expose_credentials(archived, tmp_path):
    archived['source_repair']['attempts'][0]['error'] = 'Authorization: Bearer example-private-value'
    text = ReportGeneratorAgent._build_report(archived, report_path=tmp_path / 'report.md')
    assert 'example-private-value' not in text


def test_companion_replacement_does_not_write_through_existing_hardlink(archived, tmp_path):
    import os
    report = tmp_path / 'report.md'
    artifact = archived['source_repair']['evidence_archive']
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'outside original')
    assets = tmp_path / 'report_assets'
    assets.mkdir()
    target = assets / f"source_repair_{artifact['sha256']}.zip"
    os.link(outside, target)
    evidence.evidence_target(archived, report)
    assert outside.read_bytes() == b'outside original'
    assert target.read_bytes() == Path(artifact['path']).read_bytes()
