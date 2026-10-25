"""The paper text and its receipt remain portable and bound to the same PDF."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
from unittest.mock import Mock

import pytest

from frontend.report_renderer import build_report_bundle, render_report
from src.agents.report_generator import ReportGeneratorAgent
from src.pdf_markdown import write_markdown_evidence, report_markdown_bytes, report_markdown_targets


@pytest.fixture
def paper_data(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    document = SimpleNamespace(sha256=hashlib.sha256(b'original pdf').hexdigest(),
        page_texts=('Complete paper page 1', 'Results: Signal 81.5'),
        extraction={'engine': 'markitdown', 'status': 'converted', 'page_count': 2, 'converted_pages': 2})
    record = write_markdown_evidence(run, document)
    return {'run_dir': str(run), 'paper_markdown': record,
            'pdf_input': {'sha256': document.sha256, 'extraction': document.extraction}}


def test_report_bundle_and_revision_preserve_hash_bound_markdown_and_receipt(paper_data, tmp_path):
    report = tmp_path / 'reports' / 'report.md'
    report.parent.mkdir()
    text = ReportGeneratorAgent._build_report(paper_data, report_path=report)
    companion = report_markdown_bytes(text, report)
    assert len(companion) == 2
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(text, report))) as archive:
        for name, raw in companion.items():
            assert archive.read(name) == raw
        markdown = next(raw for name, raw in companion.items() if name.endswith('.md'))
        receipt = json.loads(next(raw for name, raw in companion.items() if name.endswith('.json')))
        assert hashlib.sha256(markdown).hexdigest() == receipt['artifact_sha256']
    revised = tmp_path / 'revision' / 'v2.md'
    revised_text = ReportGeneratorAgent._build_report(paper_data, report_path=revised)
    assert set(report_markdown_bytes(revised_text, revised).values()) == set(companion.values())


@pytest.mark.parametrize('mutation', ['text', 'receipt', 'pdf_identity'])
def test_modified_source_artifact_or_pdf_identity_is_not_exported(paper_data, tmp_path, mutation):
    if mutation == 'pdf_identity':
        paper_data['pdf_input']['sha256'] = 'f' * 64
    else:
        path = Path(paper_data['run_dir']) / ('paper_markdown.md' if mutation == 'text' else 'paper_markdown.json')
        path.write_text('changed' if mutation == 'text' else '{}', encoding='utf-8')
    with pytest.raises(ValueError, match='摘要'):
        report_markdown_targets(paper_data, tmp_path / 'report.md')


def test_changed_report_companion_prevents_zip_export(paper_data, tmp_path):
    report = tmp_path / 'report.md'
    text = ReportGeneratorAgent._build_report(paper_data, report_path=report)
    companion = next(name for name in report_markdown_bytes(text, report) if name.endswith('.md'))
    (tmp_path / companion).write_bytes(b'changed')
    with pytest.raises(ValueError, match='哈希'):
        build_report_bundle(text, report)


@pytest.mark.parametrize('suffix', ['.md', '.json'])
def test_missing_report_companion_is_a_recoverable_download_error(paper_data, tmp_path, suffix):
    report = tmp_path / 'report.md'
    text = ReportGeneratorAgent._build_report(paper_data, report_path=report)
    companion = next(name for name in report_markdown_bytes(text, report) if name.endswith(suffix))
    (tmp_path / companion).unlink()
    with pytest.raises(ValueError, match='附件缺失'):
        report_markdown_bytes(text, report)
    with pytest.raises(ValueError, match='附件缺失'):
        build_report_bundle(text, report)
    from frontend.report_renderer import report_downloads
    ui = Mock()
    report_downloads({'data': {'report': text, 'report_path': str(report)}},
                     key='missing_paper', st_module=ui)
    ui.warning.assert_called_once()
    assert '附件缺失' in ui.warning.call_args.args[0]
    assert ui.download_button.call_count == 1  # The report itself is still available.


def test_web_uses_downloads_and_hides_metadata_without_changing_offline_evidence(paper_data, tmp_path):
    path = tmp_path / 'report.md'
    report = ReportGeneratorAgent._build_report(paper_data, report_path=path)
    report += '\n```text\n<!-- autoreproducer-report-format: 3 -->\n```\n'
    ui = Mock()
    render_report(report, path, st_module=ui)
    visible = ''.join(call.args[0] for call in ui.markdown.call_args_list)
    assert '论文 Markdown：可在报告下方下载。' in visible
    assert 'PDF 转换依据：随报告 ZIP 一并下载。' in visible
    assert 'report_assets/paper_markdown_' not in visible and '<!-- sha256:' not in visible
    assert visible.count('<!-- autoreproducer-report-format: 3 -->') == 1  # Code remains literal.
    assert '<!-- autoreproducer-run-finalization:' not in visible
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, path))) as archive:
        assert archive.read('report.md').decode('utf-8') == report
        assert len(archive.namelist()) == 3


@pytest.mark.parametrize('target', ['../private.md', '/private.md', 'report_assets/private.md',
                                   'report_assets/../private.md'])
def test_paper_link_cannot_export_arbitrary_local_paths(tmp_path, target):
    text = f'[论文 Markdown]({target}) <!-- sha256:{"a" * 64} -->'
    with pytest.raises(ValueError, match='路径'):
        report_markdown_bytes(text, tmp_path / 'report.md')
