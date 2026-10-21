"""Portable, hashed repair evidence without importing a model or executor."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, unquote
import zipfile

from src.safety.paths import filesystem_path, workspace_path

MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
EVIDENCE_ROW = re.compile(r'^\[源码修复证据 ZIP\]\(([^\r\n)]+)\) <!-- sha256:([0-9a-f]{64}) -->$', re.M)


def _replace_bytes(path, raw):
    path = Path(filesystem_path(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def archive_repair_evidence(run_dir, data):
    """Freeze proposals, rollback receipts and complete owned-process text logs."""
    from src.repository_repairer import _private
    root = Path(run_dir)
    state = data['source_repair']
    members, total = {}, 0

    def add(name, path):
        nonlocal total
        size = path.stat().st_size
        total += size
        if total > MAX_EVIDENCE_BYTES:
            raise ValueError('源码修复证据超过 64 MiB；完整原件仍保留在运行目录。')
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='replace')
        private = _private(text).encode('utf-8')
        members[name] = (private, private != raw, hashlib.sha256(raw).hexdigest())

    for name in ('source_repair.json', 'source_repair_execution_plan.json', 'source_repair_manifest.json',
                 'original_execution.json', 'repository.json', 'experiment_spec.json', 'execution_plan.json'):
        path = workspace_path(root, name, 'repair evidence')
        if path.is_file():
            add(name, path)
    records_path = state.get('records_path')
    if records_path:
        relative = Path(filesystem_path(Path(records_path).absolute())).relative_to(
            Path(filesystem_path(root.resolve())))
        records = workspace_path(root, relative, 'repair records', must_exist=True)
        for name in ('repairs.json', 'repair_report.md'):
            add('repairs/' + name, workspace_path(records.parent, name, 'repair record', must_exist=True))
        for attempt in state.get('attempts', []):
            for name in ('proposal.json', 'response.txt', 'candidate.diff', 'result.json'):
                relative = f"round_{attempt['round']}/{name}"
                path = workspace_path(records.parent, relative, 'repair attempt evidence')
                if path.is_file():
                    add('repairs/' + relative, path)
    executions = [('original', data.get('original_execution', {}))]
    executions += [(f"round_{attempt['round']}", attempt.get('execution', {})) for attempt in state.get('attempts', [])]
    for label, execution in executions:
        if not execution.get('run_dir'):
            continue
        log_root = Path(execution['run_dir'])
        for step in execution.get('steps', []):
            for stream in ('stdout', 'stderr'):
                name = f"{step['id']}.{stream}.log"
                path = workspace_path(log_root, name, 'owned repair execution log')
                if path.is_file():
                    add(f'logs/{label}/{name}', path)
    manifest = {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'redacted': redacted, 'original_sha256': original_sha}
                for name, (raw, redacted, original_sha) in members.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (raw, _, _) in members.items():
            archive.writestr(name, raw)
        archive.writestr('evidence_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    raw = buffer.getvalue()
    path = workspace_path(root, 'source_repair_evidence.zip', 'repair evidence archive')
    _replace_bytes(path, raw)
    state['evidence_archive'] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def evidence_target(data, report_path):
    """Copy a verified receipt beside this report, including regenerated reports."""
    artifact = (data.get('source_repair') or {}).get('evidence_archive')
    if not artifact or report_path is None:
        return ''
    root = Path(data['run_dir']).resolve()
    path = workspace_path(root, Path(artifact['path']).relative_to(root), 'repair archive', must_exist=True)
    if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError('源码修复证据附件缺失或超限')
    raw = path.read_bytes()
    if len(raw) != artifact['bytes'] or hashlib.sha256(raw).hexdigest() != artifact['sha256']:
        raise ValueError('源码修复证据附件哈希不一致')
    report = Path(report_path)
    name = f"{report.stem}_assets/source_repair_{artifact['sha256']}.zip"
    target = workspace_path(report.parent, name, 'report repair evidence')
    target.parent.mkdir(exist_ok=True)
    _replace_bytes(target, raw)
    return f'[源码修复证据 ZIP]({quote(name, safe="/")}) <!-- sha256:{artifact["sha256"]} -->'


def report_evidence_bytes(report, report_path):
    """Only generated, hash-bound companion ZIPs may enter report downloads."""
    result = {}
    root = Path(filesystem_path(Path(report_path).absolute().parent))
    for target, expected in EVIDENCE_ROW.findall(report):
        name = unquote(target)
        relative = Path(name)
        if (len(relative.parts) != 2 or not relative.parts[0].endswith('_assets')
                or relative.parts[1] != 'source_repair_' + expected + '.zip'):
            raise ValueError('源码修复证据附件路径无效')
        path = workspace_path(root, name, 'report repair evidence', must_exist=True)
        if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError('源码修复证据附件缺失或超限')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('源码修复证据附件哈希不一致')
        result[name] = raw
    return result
