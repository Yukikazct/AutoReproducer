"""Bound model reading to the executed experiment; keep original evidence IDs."""
import ast
from collections import deque
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re

from src.discovered_repository_plan import _planning_excerpts, _read_source, _text_candidate


MAX_REPOSITORY_CHARACTERS = 160000
MAX_REPOSITORY_EXCERPTS = 48
MAX_ADDITIONAL_FILES = 64
MAX_ADDITIONAL_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 100000  # Same bound used when validating a source citation.


def comparison_plan(plan):
    """Import-discovery telemetry is not an experimental setting."""
    result = deepcopy(plan)
    for key in ('import_evidence', 'deferred_runtime_imports', 'repository_namespaces'):
        result.get('requirements', {}).pop(key, None)
    return result


def _frozen_step_cli_inputs(packet, step, workspace):
    from src.author_entrypoints import resolve_entrypoint
    from src.author_experiment_selection import _entrypoint_cli

    entrypoint = step['entrypoint']
    if entrypoint['kind'] not in {'script', 'module'}:
        raise ValueError('No supported original argparse entrypoint')
    bare = {**step, 'entrypoint': {key: entrypoint[key] for key in ('kind', 'target')}}
    resolved, _, arguments = resolve_entrypoint(bare, packet, workspace)
    if resolved != entrypoint:
        raise ValueError('The frozen entrypoint differs from its original argv')
    cli, provenance = _entrypoint_cli(packet, workspace, step)
    if cli is None or cli.get('missing') or not cli.get('options'):
        raise ValueError('No unique complete static argparse contract')
    parser_tree = ast.parse(packet['repository']['files'][provenance['source_path']]['text'])
    rebound = set()
    for node in ast.walk(parser_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_defaults':
            raise ValueError('Parser default mutation order has no static input contract')
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            rebound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
        elif isinstance(node, ast.arg):
            rebound.add(node.arg)
        elif isinstance(node, ast.alias):
            rebound.add(node.asname or node.name.split('.')[0])
            if node.name == '*':
                rebound.update({'str', 'int', 'float', 'bool'})
        elif isinstance(node, ast.ExceptHandler) and node.name:
            rebound.add(node.name)
    options, flags, defaults = {}, {}, {}
    for item in cli['options']:
        dest = item['dest']
        if (dest in options or item['nargs'] is not None
                or item['action'] not in {'store', 'store_true', 'store_false'}
                or not item['flags'] or any(not flag.startswith('-') or flag in flags for flag in item['flags'])
                or len(set(item['flags'])) != len(item['flags'])):
            raise ValueError('Ambiguous or unsupported argparse declaration')
        # Custom converters are never executed or interpreted as literal values.
        declaration = ast.parse(item['quote'], mode='eval').body
        for keyword in declaration.keywords:
            if keyword.arg == 'default':
                try:
                    ast.literal_eval(keyword.value)
                except (ValueError, TypeError, SyntaxError):
                    raise ValueError('A nonliteral argparse default has unknown binding time') from None
        converter = next((keyword.value for keyword in declaration.keywords if keyword.arg == 'type'), None)
        if converter is not None and not (isinstance(converter, ast.Name)
                and converter.id in {'str', 'int', 'float', 'bool'} and converter.id not in rebound):
            raise ValueError('Custom argparse conversion has unknown input semantics')
        options[dest] = item
        flags.update({flag: item for flag in item['flags']})
    for item in cli['defaults']:
        if item['dest'] in defaults:
            raise ValueError('No unique static argparse default')
        defaults[item['dest']] = item
    explicit, index = {}, 0
    while index < len(arguments):
        token = arguments[index]
        flag, equals, value = token.partition('=')
        item = flags.get(flag)
        if item is None or token == '--' or item['dest'] in explicit:
            raise ValueError('Unknown or repeated original argparse input')
        tokens = [token]
        if item['action'] == 'store':
            if not equals:
                index += 1
                if index >= len(arguments) or arguments[index] in flags or arguments[index].startswith('--'):
                    raise ValueError('The original option lacks an unambiguous scalar value')
                value = arguments[index]
                tokens.append(value)
        elif equals:
            raise ValueError('Boolean argparse input cannot use an assigned value')
        else:
            value = item['action'] == 'store_true'
        explicit[item['dest']] = (value, tokens)
        index += 1
    values, unknown = {}, []
    for dest, item in options.items():
        default = defaults.get(dest)
        if dest in explicit:
            value, tokens = explicit[dest]
            origin = 'explicit'
        elif default is not None and not (type(default['value']) is float and not math.isfinite(default['value'])):
            value, tokens, origin = deepcopy(default['value']), [], 'parser_default'
        else:
            unknown.append(dest)
            continue
        values[dest] = {'value': value, 'origin': origin, 'flags': list(item['flags']),
            'explicit_tokens': tokens, 'value_encoding': ('raw_argv_string' if origin == 'explicit'
                and item['action'] == 'store' else 'static_argparse_value'),
            'declaration': {key: item[key] for key in ('line', 'quote')},
            'default_evidence': deepcopy(default)}
    return {'status': 'partial' if unknown else 'observed', 'parser': provenance,
            'values': values, 'unknown_values': unknown}


def frozen_cli_inputs(packet, plan, workspace):
    """Describe frozen argv and unique static parser inputs, never runtime values.

    Source reads use an isolated packet and verify the actual workspace bytes.
    Defaults belong to this invocation; launchers are documentation, not input.
    """
    result = {'version': 1, 'status': 'unavailable', 'effective_runtime_verified': False,
        'scope': 'static_argparse_input', 'steps': [],
        'limitations': ['Static argparse input only; assignments or configuration overrides after parsing are not tracked.']}
    working = deepcopy(packet)
    for step in plan.get('steps') or []:
        record = {key: deepcopy(step.get(key)) for key in ('argv', 'cwd', 'entrypoint')}
        record.update(step_id=step.get('id'), status='unavailable', values={})
        try:
            record.update(_frozen_step_cli_inputs(working, step, workspace))
        except (ValueError, TypeError, KeyError, OSError, SyntaxError, UnicodeError, AttributeError) as error:
            record.update(reason='The frozen original argv has no verified unique static argparse input contract.',
                          error_type=type(error).__name__)
        result['steps'].append(record)
    statuses = [step['status'] for step in result['steps']]
    if statuses and all(status == 'observed' for status in statuses):
        result['status'] = 'observed'
    elif any(status in {'observed', 'partial'} for status in statuses):
        result['status'] = 'partial'
    return result


def _mentioned_paths(value, files):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'path', 'source_path'} and isinstance(item, str) and item in files:
                found.add(item)
            else:
                found.update(_mentioned_paths(item, files))
    elif isinstance(value, list):
        for item in value:
            found.update(_mentioned_paths(item, files))
    return found


def _local_imports(path, files, available=None):
    """Static import locations aid reading, never assert runtime execution."""
    try:
        nodes = ast.walk(ast.parse(files[path]['text']))
    except (KeyError, SyntaxError, TypeError, ValueError):
        return set()
    available = files if available is None else available
    parent = PurePosixPath(path).parent
    found = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            bases = [parent, *parent.parents]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or '']
            if node.level:
                base = parent
                for _ in range(node.level - 1):
                    base = base.parent
                bases = [base]
            else:
                bases = [parent, *parent.parents]
        else:
            continue
        for module in modules:
            for base in bases:
                stem = base.joinpath(*module.split('.')) if module else base
                choices = [stem.as_posix() + '.py', (stem / '__init__.py').as_posix()]
                if isinstance(node, ast.ImportFrom):
                    choices += [(stem / (alias.name + '.py')).as_posix() for alias in node.names]
                matches = {candidate for candidate in choices if candidate in available}
                if matches:
                    found.update(matches)
                    break  # Prefer the nearest local import root.
    return found


def _path_priorities(files, plan, records, available=None):
    available = files if available is None else available
    entrypoints = {step.get('entrypoint', {}).get('source_path') for step in plan.get('steps', [])}
    entrypoints = {path for path in entrypoints if path in available}
    required = entrypoints | _mentioned_paths(plan.get('citations', []), available)
    required |= _mentioned_paths(plan.get('command_source', {}), available)
    required |= _mentioned_paths(records, available)
    directories = {PurePosixPath(path).parent for path in entrypoints}
    priorities = {path: (0, 0) for path in required}
    for path in available:
        pure = PurePosixPath(path)
        if pure.parent in directories:
            priorities.setdefault(path, (2, 0))
        if (pure.name.lower().startswith('readme')
                and any(pure.parent == directory or pure.parent in directory.parents for directory in directories)):
            priorities.setdefault(path, (2, 1))
        if (pure.name.lower() in {'pyproject.toml', 'setup.cfg', 'setup.py', 'requirements.txt', 'environment.yml'}
                and any(pure.parent == directory or pure.parent in directory.parents for directory in directories)):
            priorities.setdefault(path, (3, 0))
    # Follow only a few import levels. Transitive framework implementation can
    # be huge; its presence in a runtime snapshot is not relevance to a metric.
    queue = deque((path, 0) for path in sorted(entrypoints))
    visited = set(entrypoints)
    while queue:
        path, depth = queue.popleft()
        if depth >= 3:
            continue
        for imported in sorted(_local_imports(path, files, available)):
            priorities[imported] = min(priorities.get(imported, (1, depth)), (1, depth))
            if imported not in visited:
                visited.add(imported)
                queue.append((imported, depth + 1))
    return priorities


def prepare_comparison_packet(packet, plan, workspace):
    """Read bounded frozen local helpers into an independent analysis packet.

    Runtime import discovery need not have loaded a training/metric helper.
    Navigate the full immutable inventory, then verify original bytes before
    supplying additional evidence. Never execute sources or alter the saved
    packet. Failed reads remain visible and cannot support a citation.
    """
    result = deepcopy(packet)
    files = result['repository']['files']
    inventory = result['file_inventory']
    available = {path: entry for path, entry in inventory.items() if _text_candidate(path)}
    attempted, loaded, failures = set(), [], []
    used = 0
    while len(attempted) < MAX_ADDITIONAL_FILES:
        priorities = _path_priorities(files, plan, [], available)
        candidates = [path for path in priorities if path not in files and path not in attempted]
        if not candidates:
            break
        path = min(candidates, key=lambda name: (*priorities[name], name))
        attempted.add(path)
        size = available[path].get('bytes')
        if type(size) is not int or not 0 <= size <= min(MAX_SOURCE_BYTES, MAX_ADDITIONAL_BYTES - used):
            failures.append({'path': path, 'reason': 'source_read_budget'})
            continue
        try:
            source = _read_source(workspace, path, available[path]['sha256'],
                                  max_bytes=min(MAX_SOURCE_BYTES, MAX_ADDITIONAL_BYTES - used))
        except (OSError, ValueError, UnicodeError) as error:
            failures.append({'path': path, 'reason': type(error).__name__})
            continue
        files[path] = source
        loaded.append({'path': path, 'sha256': source['sha256'], 'bytes': source['bytes']})
        used += source['bytes']
    result['comparison_source_discovery'] = {'added_sources': loaded, 'read_failures': failures,
        'additional_bytes': used, 'additional_byte_limit': MAX_ADDITIONAL_BYTES,
        'attempted_files': len(attempted), 'additional_file_limit': MAX_ADDITIONAL_FILES,
        'purpose': 'static_navigation_only_not_execution_evidence'}
    return result


def comparison_view(packet, plan, records):
    """Project source evidence for analysis without modifying the full packet.

    All PDF spans remain available. Repository spans are selected whole, using
    IDs generated from the original text and offsets, so normal full-packet
    citation and hash validation can still verify each selected quotation.
    """
    files = packet['repository']['files']
    priorities = _path_priorities(files, plan, records)
    excerpts = _planning_excerpts(packet)
    selected = {key: deepcopy(item) for key, item in excerpts.items() if item['source'] == 'pdf'}
    # Printed labels can prioritize the right span inside a large source file;
    # this is navigation only, not evidence that the source produced a value.
    labels = set()
    for record in records:
        labels.update(re.findall(r'[A-Za-z][A-Za-z_]{2,}', str(record.get('source_quote', ''))))
    labels = {label.casefold() for label in labels}
    candidates = []
    dataset_paths = _mentioned_paths(plan.get('datasets', []), packet.get('file_inventory', {}))
    for key, item in excerpts.items():
        path = item.get('path')
        if item['source'] == 'repository' and path in priorities:
            related = any(re.search(r'\b' + re.escape(label) + r'\b', item['quote'], re.I) for label in labels)
            candidates.append(((*priorities[path], 0 if related else 1, path, int(key.rsplit('_', 1)[-1])), key, item))
        elif item['source'] == 'dataset' and path in dataset_paths:
            candidates.append(((1, 0, 0, path, 0), key, item))
    used, count = 0, 0
    for _, key, item in sorted(candidates):
        length = len(item['quote'])
        if used + length > MAX_REPOSITORY_CHARACTERS or count >= MAX_REPOSITORY_EXCERPTS:
            continue
        selected[key] = deepcopy(item)
        used += length
        count += 1
    paths = {item['path'] for item in selected.values() if item.get('path')}
    repository = {key: deepcopy(value) for key, value in packet['repository'].items() if key != 'files'}
    repository['files'] = {path: {key: deepcopy(value) for key, value in files[path].items() if key != 'text'}
                           for path in sorted(paths) if path in files}
    pdf = deepcopy(packet['pdf'])
    for page in pdf['pages']:
        page.pop('text', None)
    inventory = {path: {key: value for key, value in packet['file_inventory'][path].items()
                         if key in {'sha256', 'bytes', 'text_available'}}
                 for path in sorted(paths) if path in packet['file_inventory']}
    return {'version': packet.get('version'), 'pdf': pdf, 'repository': repository,
            'file_inventory': inventory, 'evidence_excerpts': selected,
            'source_discovery': deepcopy(packet.get('comparison_source_discovery', {})),
            'source_selection': {'scope': 'executed_experiment_analysis',
                'full_pdf_retained': True, 'repository_quote_characters': used,
                'repository_quote_character_limit': MAX_REPOSITORY_CHARACTERS,
                'repository_excerpt_count': count, 'repository_excerpt_limit': MAX_REPOSITORY_EXCERPTS,
                'available_repository_files': len(files), 'selected_repository_files': len(repository['files']),
                'omitted_sources_are_not_negative_evidence': True}}
