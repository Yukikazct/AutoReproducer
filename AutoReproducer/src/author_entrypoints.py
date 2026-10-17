"""Resolve original Python scripts, modules and notebooks against pinned files."""
import ast
import json
from pathlib import Path
import re
import sys

from src.safety.paths import workspace_path


MODULE = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z')


def notebook_cells(packet, workspace, name):
    entry = packet['file_inventory'].get(name)
    if not entry:
        raise ValueError('Original notebook is absent from the pinned repository: ' + name)
    path = workspace_path(workspace, name, 'original notebook', must_exist=True, forbid_git=True)
    raw = path.read_bytes()
    import hashlib
    if len(raw) > 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Original notebook changed or exceeds its read limit: ' + name)
    notebook = json.loads(raw.decode('utf-8-sig'))
    if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
        raise ValueError('Only complete version 4 author notebooks are supported')
    language = (notebook.get('metadata', {}).get('kernelspec') or {}).get('language', 'python')
    if language.casefold() != 'python' or len(notebook['cells']) > 256:
        raise ValueError('Author notebook kernel or cell count is unsupported')
    cells = []
    for index, cell in enumerate(notebook['cells']):
        if not isinstance(cell, dict) or cell.get('cell_type') not in {'code', 'markdown', 'raw'}:
            raise ValueError('Author notebook has a malformed cell')
        if cell['cell_type'] != 'code':
            continue
        source = cell.get('source', '')
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = ''.join(source)
        if not isinstance(source, str):
            raise ValueError('Author notebook code cell is malformed')
        if re.search(r'(?m)^\s*(?:!|%%(?:bash|sh|script)|%pip\b)', source):
            raise ValueError(f'Notebook cell {index + 1} uses an unsupported shell or installer command')
        cells.append({'index': index, 'source': source})
    if not cells or sum(len(cell['source']) for cell in cells) > 400000:
        raise ValueError('Author notebook has no bounded executable Python cells')
    return cells


def fixed_cuda_in_notebook(cells):
    """Only a top-level device assignment is conclusive without running code."""
    for cell in cells:
        try:
            tree = ast.parse(cell['source'])
        except SyntaxError:
            continue
        for node in tree.body:
            if (isinstance(node, (ast.Assign, ast.AnnAssign))
                    and isinstance(node.value, ast.Constant) and node.value.value == 'cuda'):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == 'device' for target in targets):
                    return True
    return False


def resolve_entrypoint(item, packet, workspace):
    argv = item.get('argv')
    explicit = item.get('entrypoint')
    if explicit is not None:
        if (not isinstance(explicit, dict) or set(explicit) != {'kind', 'target'}
                or explicit['kind'] not in {'script', 'module', 'notebook'}
                or not isinstance(explicit['target'], str) or not explicit['target']):
            raise ValueError('Author entrypoint needs a script, module or notebook target')
    if explicit and explicit['kind'] == 'notebook':
        if argv is not None:
            raise ValueError('Notebook entrypoint must not masquerade as a Python file command')
        kind, target, args = 'notebook', explicit['target'], []
    else:
        if (not isinstance(argv, list) or len(argv) < 2 or argv[0] not in {'python', 'python3'}
                or any(not isinstance(arg, str) or not arg or any(c in arg for c in '\x00\r\n') for arg in argv)):
            raise ValueError('Only structured original Python argv is supported')
        index = 2 if argv[1] == '-u' else 1
        if len(argv) <= index:
            raise ValueError('Author Python command lacks an entrypoint')
        if argv[index] == '-m':
            if len(argv) <= index + 1 or not MODULE.fullmatch(argv[index + 1]):
                raise ValueError('Author -m needs a literal repository module')
            kind, target, args = 'module', argv[index + 1], argv[index + 2:]
        elif argv[index].endswith('.py') and not argv[index].startswith('-'):
            kind, target, args = 'script', argv[index], argv[index + 1:]
        else:
            raise ValueError('Generated code and unsupported Python command cannot replace an author entrypoint')
        if explicit and explicit['kind'] != kind:
            raise ValueError('Author entrypoint differs from the original argv')
    cwd = item.get('cwd', '.')
    directory = workspace_path(workspace, cwd, 'author working directory', must_exist=True, forbid_git=True)
    if kind == 'module':
        relative = Path(*target.split('.'))
        choices = [relative.with_suffix('.py'), relative / '__main__.py']
        present = [workspace_path(directory, candidate.as_posix(), 'author module', forbid_git=True)
                   for candidate in choices]
        matches = [path for path in present if path.is_file() and
                   path.relative_to(Path(workspace).resolve()).as_posix() in packet['file_inventory']]
        if len(matches) != 1:
            raise ValueError('Author module must resolve to exactly one pinned repository entrypoint')
        path = matches[0]
    else:
        suffix = '.ipynb' if kind == 'notebook' else '.py'
        if not target.endswith(suffix):
            raise ValueError('Author entrypoint has an unexpected file type')
        try:
            path = workspace_path(directory, target, 'author entrypoint', must_exist=True, forbid_git=True)
        except FileNotFoundError as exc:
            # Return navigation evidence to the planner instead of guessing a
            # cwd when several author experiments use the same script name.
            matches = sorted(name for name in packet['file_inventory']
                             if Path(name).name == Path(target).name)[:8]
            raise ValueError(f'Author entrypoint {target!r} is missing under cwd {cwd!r}. '
                'Matching pinned inventory paths: ' + json.dumps(matches) +
                '. Read the relevant original source and choose its working directory; '
                'do not invent a root-level script.') from exc
    name = path.relative_to(Path(workspace).resolve()).as_posix()
    if name not in packet['file_inventory']:
        raise ValueError('Author entrypoint is absent from the pinned repository inventory')
    # Inventory paths are repository-relative, whereas the unchanged argv is
    # relative to cwd. Accept both only when they name this same pinned script.
    if (explicit and explicit['target'] != target
            and not (kind == 'script' and explicit['target'] == name)):
        raise ValueError('Author entrypoint differs from the original argv')
    return {'kind': kind, 'target': target, 'source_path': name}, argv or [], args


def notebook_imports(cells, packet, workspace, notebook_path):
    imports, sources = [], {notebook_path}
    local_modules = set()
    for cell in cells:
        try:
            tree = ast.parse(cell['source'])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split('.')[0]]
            else:
                continue
            for module in names:
                if module in sys.stdlib_module_names:
                    continue
                path = module + '.py'
                if path in packet['file_inventory']:
                    local_modules.add(path)
                elif module + '/__init__.py' in packet['file_inventory']:
                    local_modules.add(module + '/__init__.py')
                elif module not in {entry['module'] for entry in imports}:
                    imports.append({'module': module, 'path': notebook_path,
                                    'line': cell['index'] + 1, 'statement': module,
                                    'sha256': packet['file_inventory'][notebook_path]['sha256']})
    for path in sorted(local_modules):
        from src.discovered_repository_plan import _author_imports
        sources.add(path)
        discovered = _author_imports(packet, {'argv': ['python', path], 'cwd': '.'},
                                     include_sources=True, workspace=workspace)
        imports.extend(discovered['imports'])
        sources.update(discovered['sources'])
    return {'imports': imports, 'sources': sorted(sources),
            'excluded_device_branches': [], 'repository_namespaces': []}
