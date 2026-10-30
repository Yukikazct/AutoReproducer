"""Parse only the current run's verified, newly appended author log bytes."""
import csv
import hashlib
from io import StringIO
from pathlib import Path

from src.runtime_metrics import parse_runtime_metrics
from src.safety.paths import filesystem_path, workspace_path


def read_output_metrics(receipts, workspace, sources):
    workspace = Path(filesystem_path(Path(workspace).absolute()))
    records, errors = [], []
    if not isinstance(receipts, list):
        return [], ['Author output receipts must be a list']
    seen = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or not isinstance(receipt.get('step'), dict):
            errors.append('Author output receipt is missing its command')
            continue
        step = receipt.get('step') or {}
        outputs = receipt.get('output_files') or []
        if not isinstance(outputs, list) or len(outputs) > 16:
            errors.append('Author output file list is invalid')
            continue
        for output in outputs:
            try:
                if not isinstance(output, dict):
                    raise ValueError('Author output file receipt must be an object')
                if output.get('step_id') != step.get('id') or receipt.get('natural_return') is not True:
                    raise ValueError('Output receipt is not bound to the completed author command')
                source = output['source_path']
                if sources.get(source) != output['source_sha256']:
                    raise ValueError('Output writer does not match the frozen author source')
                writer = workspace_path(workspace, source, 'output writer', must_exist=True, forbid_git=True)
                if hashlib.sha256(writer.read_bytes()).hexdigest() != output['source_sha256']:
                    raise ValueError('Output writer source changed')
                relative = output['captured_path']
                if not isinstance(relative, str) or not relative.startswith('_autorepro_capture/output_logs/' + step['id'] + '/'):
                    raise ValueError('Output capture is outside this command output directory')
                if relative in seen:
                    continue
                path = workspace_path(workspace, relative, 'captured author output', must_exist=True, forbid_git=True)
                if path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError('Author text output is too large to compare')
                raw = path.read_bytes()
                offset = output['original_bytes']
                if type(offset) is not int or not 0 <= offset <= len(raw):
                    raise ValueError('Invalid original output byte count')
                suffix = raw[offset:]
                if (len(raw) != output['captured_bytes'] or len(suffix) != output['appended_bytes']
                        or hashlib.sha256(raw).hexdigest() != output['captured_sha256']
                        or hashlib.sha256(raw[:offset]).hexdigest() != output['original_sha256']
                        or hashlib.sha256(suffix).hexdigest() != output['appended_sha256']):
                    raise ValueError('Author output bytes differ from the completed execution receipt')
                seen.add(relative)
                text = suffix.decode('utf-8-sig')
                parsed, failures = parse_runtime_metrics(text, preserve_contexts=True)
                if not parsed:
                    # Repeated header/value CSV blocks are common in author
                    # logs. Column labels come from new bytes, never old rows.
                    header = None
                    for cells in csv.reader(StringIO(text)):
                        if len(cells) < 2:
                            continue
                        if any(parse_runtime_metrics(name.strip() + ': 0', preserve_contexts=True)[0] for name in cells):
                            header = cells
                            continue
                        if header and len(header) == len(cells):
                            for name, value in zip(header, cells):
                                values, failures = parse_runtime_metrics(name.strip() + ': ' + value.strip(), preserve_contexts=True)
                                for value in values:
                                    value['source_quote'] = ','.join(header) + '\n' + ','.join(cells)
                                    parsed.append(value)
                for metric in parsed:
                    records.append({**metric, 'id': f"{step['id']}:file:{len(records)}", 'step_id': step['id'],
                        'source': f"{step['id']}:file:{output['path']}@bytes{offset}:{len(raw)}",
                        'output_file_evidence': dict(output)})
                errors.extend(f"{output['path']}: {error}" for error in failures)
            except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                errors.append('Author output could not be verified: ' + str(exc))
    return records, errors
