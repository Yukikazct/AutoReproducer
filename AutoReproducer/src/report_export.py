"""Export a saved terminal result without model calls or experiment execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from src.author_claim_comparison import _redact_evidence
from src.report_revisions import (REPORT_FORMAT_VERSION, can_regenerate_report,
                                  sanitize_report_result)
from src.result_conclusion import build_reproduction_conclusion
from src.safety.paths import workspace_path


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_report(source, data, result):
    # A saved absolute path is provenance, not permission to read other folders.
    # Only the corresponding sibling report may supply companion attachments.
    saved = result.get('report_path') or data.get('report_path')
    name = Path(saved).name if isinstance(saved, str) and saved else (
        source.name[:-len('.result.json')] + '.md'
        if source.name.endswith('.result.json') else 'report.md')
    if Path(name).suffix.lower() != '.md':
        raise ValueError('保存结果中的报告文件名无效')
    path = workspace_path(source.parent, name, 'source report')
    text = data.get('report') or ''
    if not text and path.is_file():
        text = path.read_text(encoding='utf-8')
    return path, text


def _image_resolver(source_report, destination):
    from src.execution_artifacts import (IMAGE_TYPES, verified_image_bytes,
                                         verified_report_image_bytes)

    def resolve(artifact, _report_path):
        digest = artifact.get('sha256', '')
        suffix = Path(artifact.get('path', '')).suffix.lower()
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)) or suffix not in IMAGE_TYPES:
            raise ValueError('报告图片缺少有效的文件类型或 SHA-256')
        name = digest + suffix
        raw = verified_report_image_bytes(source_report.stem + '_assets/' + name, source_report)
        if raw is None:
            # The shared collector has its own fixed root and hash boundary.
            raw = verified_image_bytes(artifact)
        if raw is None:
            raise ValueError('报告图片缺失或校验失败；未导出不完整报告')
        relative = destination.stem + '_assets/' + name
        target = workspace_path(destination.parent, relative, 'export image')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(raw)
        return quote(relative, safe='/')

    return resolve


def _restore_legacy_images(report, original, source_report, destination):
    """Retain saved figures even if a legacy result lost its artifact list."""
    from frontend.report_renderer import report_image_bytes, report_parts

    existing = {part['target'] for part in report_parts(report) if part['type'] == 'image'}
    for part in report_parts(original):
        if part['type'] != 'image':
            continue
        raw = report_image_bytes(part['target'], source_report)
        if raw is None:
            raise ValueError('原报告图片缺失或校验失败；未导出不完整报告')
        suffix = '.png' if part['target'].startswith('data:image/png;') else (
            '.jpg' if part['target'].startswith('data:image/jpeg;') else
            '.gif' if part['target'].startswith('data:image/gif;') else
            '.webp' if part['target'].startswith('data:image/webp;') else Path(part['target']).suffix.lower())
        relative = destination.stem + '_assets/' + _digest(raw) + suffix
        target = workspace_path(destination.parent, relative, 'export legacy image')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(raw)
        url = quote(relative, safe='/')
        if url not in existing:
            report += f'\n\n![运行结果图 {len(existing) + 1}]({url})\n'
            existing.add(url)
    return report


def _prepare_reading_and_repair(data, original, source_report, evidence_root):
    """Bind attachment reads to known sibling names or verified companions."""
    from src.pdf_markdown import report_markdown_bytes
    from src.source_repair_evidence import MAX_EVIDENCE_BYTES, report_evidence_bytes

    rendered = deepcopy(data)
    rendered['run_dir'] = str(evidence_root)
    if data.get('paper_markdown'):
        companions = None
        for name in ('paper_markdown.md', 'paper_markdown.json'):
            source = workspace_path(source_report.parent, name, 'saved reading evidence')
            if source.is_file():
                if source.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('论文 Markdown 附件超限')
                raw = source.read_bytes()
            else:
                if companions is None:
                    companions = report_markdown_bytes(original, source_report)
                suffix = Path(name).suffix
                candidates = [raw for path, raw in companions.items() if path.endswith(suffix)]
                if len(candidates) != 1:
                    raise ValueError('论文 Markdown 附件缺失')
                raw = candidates[0]
            (evidence_root / name).write_bytes(raw)
        record = data['paper_markdown']
        if (_digest((evidence_root / 'paper_markdown.md').read_bytes()) != record.get('artifact_sha256')
                or json.loads((evidence_root / 'paper_markdown.json').read_bytes()) != record
                or record.get('pdf_sha256') != (data.get('pdf_input') or {}).get('sha256')):
            raise ValueError('论文 Markdown 附件与本次 PDF 的摘要不一致')
    repair = rendered.get('source_repair') or {}
    artifact = repair.get('evidence_archive')
    if artifact:
        source = workspace_path(source_report.parent, 'source_repair_evidence.zip', 'saved repair evidence')
        if source.is_file():
            if source.stat().st_size > MAX_EVIDENCE_BYTES:
                raise ValueError('源码修复证据附件超限')
            raw = source.read_bytes()
        else:
            companions = report_evidence_bytes(original, source_report)
            candidates = [raw for name, raw in companions.items()
                          if name.endswith('source_repair_' + str(artifact.get('sha256')) + '.zip')]
            if len(candidates) != 1:
                raise ValueError('源码修复证据附件缺失')
            raw = candidates[0]
        if len(raw) != artifact.get('bytes') or _digest(raw) != artifact.get('sha256'):
            raise ValueError('源码修复证据附件哈希不一致')
        target = evidence_root / 'source_repair_evidence.zip'
        target.write_bytes(raw)
        artifact['path'] = str(target)
    return rendered


def export_reproduction_report(result_path, output_dir):
    """Create a new folder with latest Markdown, safe result JSON and ZIP.

    Only saved terminal evidence is rendered. Original files remain untouched;
    the caller must choose a fresh output directory for every export.
    """
    source = Path(result_path).expanduser().absolute()
    if not source.is_file():
        raise ValueError('未找到已保存的 result JSON 文件')
    raw = source.read_bytes()
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError('保存结果不是有效 JSON') from exc
    if not can_regenerate_report(result):
        raise ValueError('缺少可重建报告的结构化结果，或任务尚未结束')
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('导出目录已存在；请选择新目录，避免覆盖')
    if not destination.parent.is_dir():
        raise ValueError('导出目录的父目录不存在')
    safe = _redact_evidence(sanitize_report_result(result), '')
    data = safe['data']
    source_report, original = _source_report(source, result['data'], result)
    # The temporary tree is created under the explicitly selected parent, and
    # only its direct children can be moved into the exclusively created folder.
    with tempfile.TemporaryDirectory(prefix='.report-export-', dir=destination.parent) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != destination.parent.resolve():
            raise ValueError('导出临时目录越界')
        report_path = stage / 'report.md'
        with tempfile.TemporaryDirectory(prefix='.saved-evidence-', dir=stage) as evidence:
            rendered = _prepare_reading_and_repair(data, original, source_report, Path(evidence))
            from src.agents.report_generator import ReportGeneratorAgent
            report = ReportGeneratorAgent._build_report(
                rendered, report_path=report_path,
                artifact_resolver=_image_resolver(source_report, report_path))
            report = report.replace(f"- **独立运行目录**: `{rendered['run_dir']}`",
                                    f"- **独立运行目录**: `{data.get('run_dir', '')}`")
        report = _restore_legacy_images(report, original, source_report, report_path)
        report = report.replace('\n\n', '\n\n> 本报告由已保存的终态结果离线生成；未重新调用模型或运行训练，原数值判定保留。\n\n', 1)
        old_name = source_report.stem + '_execution.txt'
        footer = f'> 📎 **完整执行输出附件**: `{old_name}`（同一目录下，未被截断）'
        if original.rstrip().endswith(footer):
            attachment = workspace_path(source_report.parent, old_name, 'saved execution attachment', must_exist=True)
            if not attachment.is_file():
                raise ValueError('完整执行输出附件缺失或无法读取')
            text = attachment.read_bytes().decode('utf-8')
            (stage / 'report_execution.txt').write_bytes(_redact_evidence(text, '').encode('utf-8'))
            report = report.rstrip() + '\n\n> 📎 **完整执行输出附件**: `report_execution.txt`（同一目录下，未被截断）\n'
        report = _redact_evidence(report, '')
        final_report = destination / 'report.md'
        data.update(report=report, report_path=str(final_report), report_length=len(report),
                    report_format_version=REPORT_FORMAT_VERSION,
                    reproduction_conclusion=build_reproduction_conclusion(data))
        safe['report_path'] = str(final_report)
        safe['report_export'] = {'mode': 'offline_saved_result', 'source_result_filename': source.name,
                                 'source_result_sha256': _digest(raw),
                                 'source_report_sha256': _digest(original.encode('utf-8')),
                                 'experiment_reexecuted': False}
        from src.report_bundle import json_bytes
        from frontend.report_renderer import build_report_bundle
        report_path.write_bytes(report.encode('utf-8'))
        (stage / 'result.json').write_bytes(json_bytes(safe))
        (stage / 'report.zip').write_bytes(build_report_bundle(report, report_path, result=safe))
        destination.mkdir(exist_ok=False)
        for member in stage.iterdir():
            target = destination / member.name
            if member.parent != stage or target.parent != destination:
                raise ValueError('导出文件越界')
            member.rename(target)
    return {'report_path': str(final_report), 'result_path': str(destination / 'result.json'),
            'bundle_path': str(destination / 'report.zip'), 'state': safe.get('state'),
            'source_result_sha256': _digest(raw), 'experiment_reexecuted': False}
