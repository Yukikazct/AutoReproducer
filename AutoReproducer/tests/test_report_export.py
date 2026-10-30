"""Offline publication keeps saved evidence, portable assets and source files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest
from PIL import Image

from frontend.report_renderer import render_report
from src.agents.report_generator import ReportGeneratorAgent
from src.report_export import export_reproduction_report
from src.report_revisions import REPORT_FORMAT_VERSION, report_format_version
from src.result_conclusion import build_reproduction_conclusion


@pytest.fixture
def saved_result(tmp_path):
    source = tmp_path / '原始实验'
    source.mkdir()
    report = source / 'original.md'
    report.write_text('# Original saved report\n', encoding='utf-8')
    result = {'state': 'COMPLETED', 'error': None, 'report_path': str(report), 'data': {
        'paper_info': {'title': 'Saved selected experiment'},
        'report': report.read_text(encoding='utf-8'), 'report_path': str(report),
        'run_dir': str(source),
        'execution': {'mode': 'repository', 'executed': True,
                      'final': {'success': True, 'exit_code': 0, 'stdout': 'accuracy=0.845'}},
        'strict_validation': {'requested': True, 'status': 'verified'},
        'validation': {'status': 'reproduced', 'is_reproduced': True,
                       'scope': 'selected_paper_experiment', 'paper_metrics_verified': True,
                       'protocol_pass': True, 'independent_metrics_pass': True,
                       'comparisons': [{'name': 'accuracy', 'actual': 0.845, 'reference': 0.84, 'pass': True}]}}}
    path = source / 'result.json'
    path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    return path, result, tmp_path / '离线交付'


@pytest.fixture(autouse=True)
def forbid_execution_and_api(monkeypatch):
    from src.llm.llm_client import LLMClient
    from src.agents.code_executor import CodeExecutorAgent
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline export must not construct agents, call models or start processes')

    monkeypatch.setattr(ReportGeneratorAgent, '__init__', forbidden)
    monkeypatch.setattr(ReportGeneratorAgent, 'run', forbidden)
    monkeypatch.setattr(LLMClient, '__init__', forbidden)
    monkeypatch.setattr(CodeExecutorAgent, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


def save(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


def test_export_preserves_originals_and_matches_report_result_and_hashed_bundle(saved_result):
    source, original, destination = saved_result
    original['data']['reproduction_conclusion'] = {'is_reproduced': False, 'summary': 'STALE'}
    original['data']['experiment_spec'] = {'label': 'Frozen selected experiment'}
    save(source, original)
    before = {path: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
    receipt = export_reproduction_report(source, destination)
    assert receipt['experiment_reexecuted'] is False
    assert receipt['state'] == original['state']
    assert {path: path.read_bytes() for path in before} == before
    exported = json.loads((destination / 'result.json').read_text(encoding='utf-8'))
    report = (destination / 'report.md').read_text(encoding='utf-8')
    assert report == exported['data']['report']
    assert report_format_version(report) == REPORT_FORMAT_VERSION
    assert '离线生成' in report
    assert exported['data']['validation'] == original['data']['validation']
    assert exported['data']['execution'] == original['data']['execution']
    assert exported['data']['run_dir'] == original['data']['run_dir']
    details = (destination / 'report_details.md').read_text(encoding='utf-8')
    assert f"- **独立运行目录**: `{original['data']['run_dir']}`" in details
    assert '.saved-evidence-' not in details and '.report-export-' not in details
    assert exported['data']['reproduction_conclusion'] == build_reproduction_conclusion(exported['data'])
    assert exported['data']['reproduction_conclusion']['is_reproduced'] is True
    assert exported['report_export']['source_result_sha256'] == hashlib.sha256(before[source]).hexdigest()
    with zipfile.ZipFile(destination / 'report.zip') as bundle:
        assert bundle.read('report.md').decode('utf-8') == report
        evidence = json.loads(bundle.read('report.evidence.json'))
        assert evidence['data']['validation'] == original['data']['validation']
        assert evidence['data']['reproduction_conclusion'] == exported['data']['reproduction_conclusion']
        manifest = json.loads(bundle.read('report.manifest.json'))
        assert {row['path'] for row in manifest['files']} == set(bundle.namelist()) - {'report.manifest.json'}
        for row in manifest['files']:
            raw = bundle.read(row['path'])
            assert row['sha256'] == hashlib.sha256(raw).hexdigest()
            assert row['bytes'] == len(raw)


@pytest.mark.parametrize('update', [
    {'state': 'RUNNING'}, {'data': {'execution': {'final': {'running': True}}}},
    {'data': {'execution': {'stages': [{'status': 'running'}]}}},
])
def test_export_refuses_active_results_without_creating_output(saved_result, update):
    source, result, destination = saved_result
    if 'data' in update:
        result['data']['execution'] = update['data']['execution']
    else:
        result.update(update)
    save(source, result)
    with pytest.raises(ValueError, match='尚未结束'):
        export_reproduction_report(source, destination)
    assert not destination.exists()


def test_export_refuses_existing_output_without_changing_it(saved_result):
    source, _, destination = saved_result
    destination.mkdir()
    sentinel = destination / 'report.md'
    sentinel.write_bytes(b'previous delivery')
    with pytest.raises(FileExistsError, match='已存在'):
        export_reproduction_report(source, destination)
    assert sentinel.read_bytes() == b'previous delivery'
    assert len(list(destination.iterdir())) == 1


def test_export_redacts_configuration_and_free_text_credentials(saved_result):
    source, result, destination = saved_result
    secret = 'sk-offlinefixture123456789012'
    result['data']['model_settings'] = {'DEEPSEEK_API_KEY': secret, 'password': 'config-password'}
    result['data']['execution']['final']['stdout'] += '\n' + secret + '\nBearer saved-bearer-token'
    save(source, result)
    export_reproduction_report(source, destination)
    for path in (destination / 'report.md', destination / 'result.json'):
        raw = path.read_bytes()
        assert secret.encode() not in raw
        assert b'config-password' not in raw
        assert b'saved-bearer-token' not in raw
    with zipfile.ZipFile(destination / 'report.zip') as bundle:
        for name in bundle.namelist():
            raw = bundle.read(name)
            assert secret.encode() not in raw
            assert b'config-password' not in raw
            assert b'saved-bearer-token' not in raw
    assert secret in source.read_text(encoding='utf-8')  # Original evidence is unchanged.


@pytest.mark.parametrize('legacy_without_artifacts', [False, True])
def test_export_moves_verified_figures_and_unabridged_log_after_collection_cleanup(
        saved_result, legacy_without_artifacts):
    from unittest.mock import Mock

    source, result, destination = saved_result
    buffer = io.BytesIO()
    Image.new('RGB', (12, 10), 'orange').save(buffer, format='PNG')
    raw = buffer.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    assets = source.parent / 'original_assets'
    assets.mkdir()
    (assets / (digest + '.png')).write_bytes(raw)
    old_report = '# Original\n\n![运行结果图 1](original_assets/' + digest + '.png)\n\n'
    old_report += '> 📎 **完整执行输出附件**: `original_execution.txt`（同一目录下，未被截断）\n'
    result['data']['report'] = old_report
    Path(result['report_path']).write_text(old_report, encoding='utf-8')
    if not legacy_without_artifacts:
        result['data']['execution']['final']['artifacts'] = [
            {'path': str(source.parent / 'removed_collection' / 'figure.png'),
             'name': 'Actual figure', 'sha256': digest}]
    full_log = 'FULL SAVED LOG\n' + ('unabridged epoch output\n' * 2500) + 'FINAL ORIGINAL LINE\n'
    (source.parent / 'original_execution.txt').write_bytes(full_log.encode('utf-8'))
    save(source, result)
    export_reproduction_report(source, destination)
    moved = destination.parent / '搬迁后的交付'
    destination.rename(moved)
    assert source.parent.resolve().is_relative_to(destination.parent.resolve())
    shutil.rmtree(source.parent)
    report = (moved / 'report.md').read_text(encoding='utf-8')
    ui = Mock()
    render_report(report, moved / 'report.md', st_module=ui)
    ui.image.assert_called_once_with(raw, caption='运行结果图 1')
    ui.warning.assert_not_called()
    assert (moved / 'report_execution.txt').read_bytes() == full_log.encode('utf-8')
    with zipfile.ZipFile(moved / 'report.zip') as bundle:
        assert bundle.read('report_execution.txt') == full_log.encode('utf-8')
        assert bundle.read('report_assets/' + digest + '.png') == raw
        assert bundle.read('report.md').decode('utf-8') == report
    # A moved delivery remains an input to the same offline exporter, even
    # though the original run and artifact collection no longer exist.
    repeated = moved.parent / '再次导出'
    export_reproduction_report(moved / 'result.json', repeated)
    assert (repeated / 'report_execution.txt').read_bytes() == full_log.encode('utf-8')
    assert (repeated / 'report_assets' / (digest + '.png')).read_bytes() == raw


def test_export_refuses_unsafe_or_missing_declared_images(saved_result):
    source, result, destination = saved_result
    private = source.parent / 'private.png'
    private.write_bytes(b'arbitrary sibling contents')
    result['data']['report'] = '# Original\n![运行结果图 1](private.png)\n'
    save(source, result)
    with pytest.raises(ValueError, match='图片缺失或校验失败'):
        export_reproduction_report(source, destination)
    assert not destination.exists()
    assert private.read_bytes() == b'arbitrary sibling contents'


def test_export_checks_pdf_markdown_and_preserves_portable_reading_evidence(saved_result):
    source, result, destination = saved_result
    markdown = b'<!-- PDF page 1 -->\n\nCora accuracy: 84.5%\n'
    record = {'artifact_sha256': hashlib.sha256(markdown).hexdigest(), 'pdf_sha256': 'f' * 64}
    result['data']['paper_markdown'] = record
    result['data']['pdf_input'] = {'sha256': 'f' * 64, 'extraction': {'status': 'partial', 'page_count': 1}}
    (source.parent / 'paper_markdown.md').write_bytes(markdown)
    (source.parent / 'paper_markdown.json').write_text(json.dumps(record), encoding='utf-8')
    save(source, result)
    export_reproduction_report(source, destination)
    with zipfile.ZipFile(destination / 'report.zip') as bundle:
        reading = [name for name in bundle.namelist() if name.startswith('report_assets/paper_markdown_')]
        assert len(reading) == 2
        assert any(bundle.read(name) == markdown for name in reading)
    (source.parent / 'paper_markdown.md').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='摘要不一致'):
        export_reproduction_report(source, destination.parent / 'tampered-export')


def test_cli_reports_success_and_errors_without_starting_processes(saved_result, capsys):
    source, _, destination = saved_result
    script = Path(__file__).parents[1] / 'scripts' / 'export_reproduction_report.py'
    spec = importlib.util.spec_from_file_location('offline_report_cli', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ['--result', str(source), '--output-dir', str(destination)]
    assert module.main(args) == 0
    assert json.loads(capsys.readouterr().out)['experiment_reexecuted'] is False
    assert module.main(args) == 1
    assert '已存在' in capsys.readouterr().err
