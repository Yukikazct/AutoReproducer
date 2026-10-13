"""Read literal author environment declarations without executing shell code."""
from pathlib import Path, PureWindowsPath
import os
import re
import shlex

from src.discovered_repository_plan import _load_execution_source


def author_environment(packet, workspace):
    workspace = Path(workspace).resolve()
    files = packet['repository']['files']
    needed = {'PYTHONPATH'}
    for source in list(files.values()):
        needed.update(re.findall(r'(?:os\.)?environ\s*\[\s*[\'"]([A-Za-z_]\w*)[\'"]', source['text']))
        needed.update(re.findall(r'(?:os\.)?(?:getenv|environ\.get)\s*\(\s*[\'"]([A-Za-z_]\w*)[\'"]', source['text']))
    queue = [(name, '.') for name in files if Path(name).name.lower().startswith('readme')]
    queue.extend((name, Path(name).parent.as_posix()) for name in packet['file_inventory']
                 if name.endswith('.sh') and any(name in files[readme]['text'] for readme, _ in queue))
    visited, bindings, records, warnings = set(), {}, {}, []
    while queue:
        name, cwd = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        try:
            source = _load_execution_source(packet, workspace, name)
        except (ValueError, OSError, UnicodeError):
            continue
        for line in source['text'].splitlines():
            stripped = line.strip().removeprefix('$ ').strip('`')
            try:
                tokens = shlex.split(stripped, comments=True)
            except ValueError:
                continue
            if len(tokens) == 2 and tokens[0] in ('source', '.'):
                child = (Path(cwd) / tokens[1]).as_posix()
                if child in packet['file_inventory'] and child.endswith('.sh'):
                    queue.append((child, cwd))
                continue
            if len(tokens) == 2 and tokens[0] == 'export':
                assignment = tokens[1]
            elif len(tokens) == 1:
                assignment = tokens[0]
            else:
                continue
            match = re.fullmatch(r'([A-Za-z_]\w*)=(.*)', assignment)
            if not match:
                continue
            key, value = match.groups()
            value = value.replace('$(pwd)', str(workspace / cwd)).replace('`pwd`', str(workspace / cwd))
            value = re.sub(r'\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))',
                           lambda m: bindings.get(m[1] or m[2], '' if (m[1] or m[2]) == 'PYTHONPATH' else m[0]), value)
            if re.search(r'[$`\n\r;|&<>]', value):
                if key in needed:
                    warnings.append('Unresolved author environment declaration: ' + key + ' in ' + name)
                continue
            bindings[key] = value
            records[key] = {'value': value, 'source_path': name, 'source_sha256': source['sha256'], 'quote': line}
    accepted = {}
    for key in sorted(needed & records.keys()):
        record = records[key]
        value = record['value']
        if key == 'PYTHONPATH':
            # Shell uses ':' even on Windows; never copy host environment data.
            parts = value.replace(str(workspace), '${WORKSPACE}').split(':')
            paths = [part.replace('${WORKSPACE}', str(workspace)) for part in parts if part]
        else:
            paths = [value] if ('/' in value or '\\' in value or PureWindowsPath(value).drive or value in ('.', '..', '.git')) else []
        if paths:
            resolved = [(workspace / cwd_path).resolve() for cwd_path in paths]
            if any(not path.is_relative_to(workspace) or '.git' in path.relative_to(workspace).parts for path in resolved):
                warnings.append('Author environment path outside the repository was omitted: ' + key)
                continue
            value = os.pathsep.join(map(str, resolved)) if key == 'PYTHONPATH' else str(resolved[0])
        if value:
            accepted[key] = {**record, 'value': value}
    return accepted, warnings
