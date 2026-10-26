"""Evidence and verified local-import recovery shared by author execution retries."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from src.agents.code_executor import _safe_install_diagnostic
from src.author_dependencies import declared_requirements
from src.method_adapters import write_json
from src.safety.paths import workspace_path


def failure_diagnostic(result):
    final = result.get('final') or {}
    # Some original programs report exceptions on stdout. Keep both streams so
    # a generic supervisor message cannot hide the actual Python traceback.
    parts = [str(final.get('stdout') or '')[-3000:], str(final.get('stderr') or '')[-8000:]]
    if not any(parts):
        parts.append(str(result.get('reason') or ''))
    return _safe_install_diagnostic('\n'.join(part for part in parts if part))


def failure_fingerprint(result):
    diagnostic = failure_diagnostic(result)
    missing_import = terminal_missing_import(diagnostic)
    diagnostic = re.sub(r'\b[0-9a-f]{32}\b', '<run>', diagnostic)
    # Progress bars and elapsed times change on every retry. The exception and
    # traceback locations identify a repeated failure more reliably.
    trace = diagnostic.rfind('Traceback (most recent call last):')
    if trace >= 0:
        diagnostic = diagnostic[trace:]
    # Identical wrappers can hide different missing dependencies on successive
    # attempts. Keep the proven cause so those attempts count as progress.
    value = [(result.get('final') or {}).get('id'), diagnostic, missing_import]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def terminal_missing_import(diagnostic):
    """Identify a terminal missing import or its explicit ImportError cause."""
    return _missing_import_traceback(diagnostic)[0]


def _missing_import_traceback(diagnostic):
    """Do not infer causes from caught errors, implicit context, or log quotes."""
    diagnostic = diagnostic.replace('\r\n', '\n')
    while True:
        traces = list(re.finditer(r'(?m)^Traceback \(most recent call last\):$', diagnostic))
        start = traces[-1].start() if traces else 0
        trace = diagnostic[start:]
        exceptions = list(re.finditer(r'(?m)^(?:\w+\.)*\w*(?:Error|Exception|Interrupt|Exit):[^\n]*', trace))
        if not exceptions:
            return '', ''
        exception = exceptions[-1].group()
        if re.match(r'(?:ModuleNotFoundError|ImportError):\s*No module named', exception):
            return exception, trace
        if not traces or not exception.startswith('ImportError:'):
            return '', ''
        # Python emits this exact separator for ``raise ImportError(...) from
        # error``. Implicit exception context alone does not prove causality.
        direct = re.search(r'\n\nThe above exception was the direct cause of the following exception:\n\n\Z',
                           diagnostic[:start])
        if not direct:
            return '', ''
        cause = diagnostic[:direct.start()].rstrip()
        cause_traces = list(re.finditer(r'(?m)^Traceback \(most recent call last\):$', cause))
        if not cause_traces:
            return '', ''  # A truncated/quoted earlier error is insufficient.
        ending = re.search(r'(?m)^(?:\w+\.)*\w*(?:Error|Exception|Interrupt|Exit):[^\n]*\Z', cause)
        if not ending or ending.start() < cause_traces[-1].start():
            return '', ''
        diagnostic = cause


def traceback_sources(diagnostic, workspace, inventory):
    """Only select files actually belonging to the pinned source inventory."""
    found = []
    for filename, line in re.findall(r'File "([^"\n]+)", line (\d+)', diagnostic):
        try:
            path = Path(filename)
            relative = path.resolve().relative_to(Path(workspace).resolve()).as_posix()
            if relative in inventory:
                found.append((relative, int(line)))
        except (OSError, ValueError):
            continue
    return list(dict.fromkeys(found))


def repair_local_import(result, workspace, packet, requirements=None):
    """Restore a unique pinned module's import root, never install its namesake."""
    diagnostic = failure_diagnostic(result)
    exception, import_trace = _missing_import_traceback(diagnostic)
    errors = re.findall(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", exception)
    if not errors or not re.fullmatch(r'[A-Za-z_]\w*', errors[-1]):
        return None
    module = errors[-1]
    if declared_requirements(module, (requirements or {}).get('author', [])):
        return None  # A same-named helper does not identify an external distribution.
    inventory = packet.get('file_inventory') or {}
    importing = []
    for name, line in traceback_sources(import_trace, workspace, inventory):
        try:
            path = workspace_path(workspace, name, 'failed import source', must_exist=True)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != inventory[name].get('sha256'):
                continue
            nodes = ast.walk(ast.parse(raw))
            if any(node.lineno <= line <= getattr(node, 'end_lineno', node.lineno)
                   and (isinstance(node, ast.Import) and any(alias.name.split('.')[0] == module for alias in node.names)
                        or isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                        and node.module.split('.')[0] == module)
                   for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))):
                importing.append(name)
        except (OSError, ValueError, SyntaxError):
            continue
    if not importing:
        return None
    candidates = []
    for name, record in inventory.items():
        relative = PurePosixPath(name)
        if relative.name == module + '.py':
            parent = relative.parent
        elif relative.name == '__init__.py' and relative.parent.name == module:
            parent = relative.parent.parent
        else:
            continue
        try:
            path = workspace_path(workspace, name, 'pinned missing module', must_exist=True)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.get('sha256'):
                return None
            candidates.append((parent.as_posix(), name, record['sha256']))
        except (OSError, ValueError):
            return None
    # A root-level module is already on the runner's path; that is a different
    # error. Ambiguous same-named modules require source investigation.
    if len(candidates) != 1 or candidates[0][0] == '.':
        return None
    root, source, sha256 = candidates[0]
    # Promoting a package's internal helper to a top-level module changes its
    # identity. Only legacy sibling imports can justify this package boundary.
    if (any((parent / '__init__.py').as_posix() in inventory
            for parent in [PurePosixPath(root), *PurePosixPath(root).parents] if parent.as_posix() != '.')
            and not any(PurePosixPath(name).parent.as_posix() == root for name in importing)):
        return None
    return {'action': 'restore_repository_import', 'module': module, 'root': root,
            'source_path': source, 'source_sha256': sha256, 'importing_sources': importing,
            'reason': f'固定仓库中的 {source} 未在原命令的模块路径中；加入 {root} 后重试原命令。'}


def save_repair_state(run_dir, data, on_event, *, status, reason, current=None):
    state = data.setdefault('self_repair', {'attempts': []})
    state.update(status=status, reason=reason)
    if current is not None:
        state['last_failure'] = failure_diagnostic(current)[-4000:]
    write_json(Path(run_dir) / 'self_repair.json', state)
    on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'self_repair',
        'status': 'RUNNING' if status == 'retrying' else 'SUCCESS' if status == 'resolved' else 'WARNING',
        'detail': reason, 'data': {'status': status, 'attempts': len(state['attempts'])}}})


def record_repair(run_dir, data, on_event, result, action, reason):
    from src.intelligent_recovery import current_investigation
    if current_investigation():
        current_investigation().consume_repair()
    state = data.setdefault('self_repair', {'attempts': []})
    state['attempts'].append({'round': len(state['attempts']) + 1, 'action': action,
        'reason': reason, 'status': 'retrying', 'failed_execution_dir': result.get('run_dir'),
        'failure_fingerprint': failure_fingerprint(result),
        'diagnostic': failure_diagnostic(result)[-4000:]})
    save_repair_state(run_dir, data, on_event, status='retrying', reason=reason, current=result)


def verify_repair_result(run_dir, data, on_event, result):
    state = data.get('self_repair') or {}
    attempts = state.get('attempts') or []
    if not attempts or attempts[-1].get('status') != 'retrying':
        return
    attempt = attempts[-1]
    attempt['verification_execution_dir'] = result.get('run_dir')
    if result.get('success'):
        attempt['status'] = 'resolved'
        save_repair_state(run_dir, data, on_event, status='resolved',
                          reason='修复后的原命令已实际重跑并正常退出。')
    elif result.get('cancelled') or result.get('interrupted') or (result.get('final') or {}).get('cancelled'):
        attempt['status'] = 'interrupted'
    else:
        attempt['next_failure_fingerprint'] = failure_fingerprint(result)
        attempt['status'] = ('no_progress' if attempt['next_failure_fingerprint'] == attempt['failure_fingerprint']
                             else 'failed_with_new_error')
    write_json(Path(run_dir) / 'self_repair.json', state)
