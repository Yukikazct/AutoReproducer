"""Read original entrypoints and defaults when README prose omits a command.

This is source navigation, never author-code execution. Inferred commands retain
native defaults, and unresolved required values remain questions for planning.
"""
import ast
import hashlib
from pathlib import Path, PurePosixPath
import re

from src.safety.paths import workspace_path


_ENTRY = re.compile(r'^(?:train(?:ing)?|main|run|fit)(?:[_-].*)?\.py$', re.I)
_CONFIG_SUFFIXES = {'.yaml', '.yml', '.json', '.toml', '.cfg'}


def _mentioned(text, path):
    return bool(re.search(r'(?<![\w./-])' + re.escape(path) + r'(?![\w./-])', text))


def _mentioned_paths(text, paths):
    """A basename in prose identifies a file only when inventory makes it unique."""
    basenames = {}
    for name in paths:
        basenames.setdefault(PurePosixPath(name).name, []).append(name)
    return {name for name in paths if _mentioned(text, name) or (
        len(basenames[PurePosixPath(name).name]) == 1 and _mentioned(text, PurePosixPath(name).name))}


def _source(packet, workspace, name):
    from src.discovered_repository_plan import _load_execution_source
    return _load_execution_source(packet, workspace, name)


def _reachable(tree, defaults=()):
    """Walk executed module statements and local functions they actually call."""
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    pending = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    visited, result = set(), []
    values = {item['dest']: item['value'] for item in defaults}
    namespaces = {node.targets[0].id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
        and (isinstance(node.value.func, ast.Attribute) and node.value.func.attr in {'parse_args', 'parse_known_args'}
             or isinstance(node.value.func, ast.Name) and node.value.func.id in {'get_args', 'parse_args', 'parse_config'})}
    unknown = object()

    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id == '__name__':
            return '__main__'
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in namespaces:
            return values.get(node.attr, unknown)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = literal(node.operand)
            return not value if value is not unknown else unknown
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = literal(node.left), literal(node.comparators[0])
            if left is not unknown and right is not unknown:
                if isinstance(node.ops[0], ast.Eq):
                    return left == right
                if isinstance(node.ops[0], ast.NotEq):
                    return left != right
        return unknown

    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        result.append(node)
        if isinstance(node, (ast.If, ast.While)):
            condition = literal(node.test)
            # Unknown control flow is model navigation evidence, not proof that
            # a default invocation reaches training. Resolve only static branches.
            if condition is not unknown:
                pending.extend(node.body if condition else node.orelse)
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = functions.get(node.func.id)
            if function and function.name not in visited:
                visited.add(function.name)
                pending.extend(function.body)
        pending.extend(ast.iter_child_nodes(node))
    return result


def _training_evidence(tree, text, defaults=()):
    result = []
    for node in _reachable(tree, defaults):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ''
        if name in {'backward', 'fit', 'fit_generator', 'train', 'training', 'train_model', 'train_and_eval'}:
            # model.train() only changes module mode; it does not prove training.
            if name == 'train' and isinstance(function, ast.Attribute) and not node.args and not node.keywords:
                continue
            result.append({'line': node.lineno, 'quote': ast.get_source_segment(text, node) or name})
    return result[:6]


def _input_paths(tree, text):
    result = []
    for node in _reachable(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
        if name not in {'open', 'load', 'loadtxt', 'genfromtxt', 'read_csv', 'read_table', 'read_parquet', 'read_json'}:
            continue
        if name == 'open':
            mode = node.args[1] if len(node.args) > 1 else next((v.value for v in node.keywords if v.arg == 'mode'), ast.Constant('r'))
            if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str) or any(c in mode.value for c in 'wax+'):
                continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value and not re.match(r'\w+://', value.value):
            result.append({'value': value.value, 'line': node.lineno, 'quote': ast.get_source_segment(text, node)})
    return result[:16]


def _cwd_and_argv(name, tree, paths, inventory, *, cwd_hint=None):
    path = PurePosixPath(name)
    package = []
    parent = path.parent
    while parent.as_posix() != '.' and (parent / '__init__.py').as_posix() in inventory:
        package.insert(0, parent.name)
        parent = parent.parent
    relative_imports = any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree))
    if package and relative_imports:
        module = '.'.join([*package, path.stem])
        return parent.as_posix(), ['python', '-m', module], 'package_relative_imports'
    if cwd_hint is not None:
        directory = PurePosixPath(cwd_hint)
        target = path.relative_to(directory).as_posix() if directory.as_posix() != '.' else name
        return directory.as_posix(), ['python', target], 'readme_working_directory'
    choices = list(dict.fromkeys([path.parent.as_posix(), '.']))
    scores = {}
    for directory in choices:
        hits = 0
        for item in paths:
            candidate = PurePosixPath(directory) / item['value']
            parts = []
            safe = True
            for part in candidate.parts:
                if part == '..':
                    if not parts:
                        safe = False
                        break
                    parts.pop()
                elif part != '.':
                    parts.append(part)
            normalized = '/'.join(parts)
            if safe and any(key == normalized or key.startswith(normalized.rstrip('/') + '/') for key in inventory):
                hits += 1
        scores[directory] = hits
    cwd = max(choices, key=lambda directory: scores[directory])
    target = path.relative_to(PurePosixPath(cwd)).as_posix() if cwd != '.' else name
    return cwd, ['python', target], 'original_input_paths' if scores[cwd] else 'entrypoint_directory'


def discover_author_commands(packet, workspace, *, max_files=24, max_bytes=1024 * 1024):
    """Return bounded candidates plus their real README/parser/config context."""
    inventory = packet['file_inventory']
    readmes, consumed = [], 0
    for name in sorted(inventory):
        if not Path(name).name.lower().startswith('readme') or len(readmes) >= 8:
            continue
        size = inventory[name].get('bytes', 0)
        if size > 100000 or consumed + size > max_bytes:
            continue
        try:
            source = _source(packet, workspace, name)
        except (ValueError, OSError, UnicodeError):
            continue
        readmes.append((name, source))
        consumed += size
    prose = '\n'.join(source['text'] for _, source in readmes)
    mentioned = _mentioned_paths(prose, inventory)
    configs = [name for name in inventory if Path(name).suffix.lower() in _CONFIG_SUFFIXES
               and any(word in name.lower() for word in ('config', 'setting'))]
    preferred_configs = [name for name in configs if name in mentioned]
    names = [name for name in inventory if name.endswith('.py') and (
        _ENTRY.fullmatch(Path(name).name) or name in mentioned)
        and not any(part.lower() in {'tests', 'test', 'vendor', 'third_party'} for part in Path(name).parts)]
    names.sort(key=lambda name: (name not in mentioned, len(Path(name).parts), name))
    candidates, failures = [], []
    cwd_quotes = []
    for readme_name, readme in readmes:
        for number, line in enumerate(readme['text'].splitlines(), 1):
            hint = re.search(r'\b(?:navigate\s+(?:into|to)|change\s+(?:the\s+)?(?:working\s+)?directory\s+to)\s+`([^`\r\n]+)`', line, re.I)
            root_hint = re.search(r'\b(?:run|execute|launch|training)\b.*\bfrom\s+(?:the\s+)?(?:repository|project)\s+root\b', line, re.I)
            if hint or root_hint:
                cwd_quotes.append({'value': hint[1] if hint else '.', 'path': readme_name,
                    'sha256': readme['sha256'], 'line': number, 'quote': line[:1800]})
    from src.author_cli_inference import infer_cli
    for name in names[:max_files]:
        entry = inventory[name]
        if entry.get('bytes', 0) > 200000 or consumed + entry.get('bytes', 0) > max_bytes:
            continue
        consumed += entry.get('bytes', 0)
        try:
            source = _source(packet, workspace, name)
            tree = ast.parse(source['text'])
            cli = infer_cli(source['text'], inventory=inventory, source_path=name,
                            preferred_configs=preferred_configs)
            training = _training_evidence(tree, source['text'], (cli or {}).get('defaults', []))
            if not training:
                continue
            inputs = _input_paths(tree, source['text'])
            if cli:
                inputs.extend({'value': item['value'], 'quote': item.get('quote', ''), 'line': item.get('line')}
                              for item in cli['defaults'] if isinstance(item.get('value'), str)
                              and re.search(r'data|path|root|config|cfg', str(item.get('dest', item.get('name', item.get('option', '')))), re.I))
            candidate_cwd_quotes = []
            for hint in cwd_quotes:
                directory = PurePosixPath(hint['value'])
                names_in_quote = _mentioned_paths(hint['quote'], inventory)
                named_scripts = {path for path in names_in_quote if path.endswith('.py')}
                associated = name in named_scripts
                if not named_scripts:
                    training_context = re.search(r'\b(?:train(?:ing)?|experiment|reproduc\w*)\b', hint['quote'], re.I)
                    inside_directory = directory.as_posix() != '.' and directory in PurePosixPath(name).parents
                    associated = bool(training_context or inside_directory)
                if associated:
                    candidate_cwd_quotes.append(hint)
            cwd_values = {item['value'] for item in candidate_cwd_quotes}
            cwd_hint = next(iter(cwd_values)) if len(cwd_values) == 1 else None
            if cwd_hint is not None:
                workspace_path(workspace, cwd_hint, 'author working directory', must_exist=True, forbid_git=True)
            cwd, argv, cwd_reason = _cwd_and_argv(name, tree, inputs, inventory, cwd_hint=cwd_hint)
            if cli:
                cli = infer_cli(source['text'], inventory=inventory, source_path=name,
                                preferred_configs=preferred_configs, cwd=cwd)
                argv.extend(cli['arguments'])
            # A script consuming sys.argv or an unknown parser needs model/source
            # analysis; a bare filename is not a complete inferred invocation.
            dynamic_cli = cli is None and any(
                isinstance(node, ast.Attribute) and node.attr in {'argv', 'parse_args', 'parse_known_args'}
                or isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and re.fullmatch(r'(?:get|parse)_(?:args|arguments|config|options)', node.func.id)
                for node in ast.walk(tree))
            missing = list(cli['missing']) if cli else ['unresolved command-line interface'] if dynamic_cli else []
            if len(cwd_values) > 1:
                missing.append('ambiguous README working directories')
            readme_evidence = []
            for readme_name, readme in readmes:
                for number, line in enumerate(readme['text'].splitlines(), 1):
                    line_mentions = _mentioned_paths(line, inventory)
                    if name in line_mentions or any(config in line_mentions for config in preferred_configs):
                        readme_evidence.append({'path': readme_name, 'sha256': readme['sha256'], 'line': number, 'quote': line[:1800]})
            context = []
            for config in (cli or {}).get('config_paths', [])[:4]:
                try:
                    size = inventory[config].get('bytes', 0)
                    if size > 64000 or consumed + size > max_bytes:
                        continue
                    file = workspace_path(workspace, config, 'original configuration', must_exist=True, forbid_git=True)
                    raw = file.read_bytes()
                    if len(raw) > 64000 or hashlib.sha256(raw).hexdigest() != inventory[config]['sha256']:
                        continue
                    text = raw.decode('utf-8-sig')
                    consumed += len(raw)
                    context.append({'path': config, 'sha256': inventory[config]['sha256'], 'text': text[:16000]})
                    if Path(config).suffix.lower() != '.json':
                        _source(packet, workspace, config)
                except (ValueError, OSError, UnicodeError):
                    continue
            entry_hint = name in mentioned
            score = 10 * entry_hint + 2 * bool(context) + bool(inputs)
            candidates.append({'source_path': name, 'source_sha256': source['sha256'],
                'argv': argv, 'cwd': cwd, 'cwd_evidence': cwd_reason,
                'complete': not missing and (cli is None or cli['complete']), 'missing': missing,
                'training_evidence': training, 'readme_evidence': readme_evidence[:8],
                'cwd_readme_evidence': candidate_cwd_quotes[:8],
                'cli': cli, 'input_paths': inputs, 'configuration_sources': context, 'selection_score': score})
        except (ValueError, OSError, UnicodeError, SyntaxError) as exc:
            failures.append({'path': name, 'reason': str(exc)[:500]})
    candidates.sort(key=lambda item: (-item['selection_score'], item['source_path']))
    return {'candidates': candidates[:8], 'files_examined': min(len(names), max_files),
            'bytes_examined': consumed, 'failures': failures[:8]}


def select_author_command(discovery, *, required_entrypoint=None, dataset=None):
    candidates = [item for item in discovery['candidates'] if item['complete']
                  and (not required_entrypoint or item['source_path'] == required_entrypoint)]
    if dataset:
        from src.author_experiment_selection import dataset_cli_from_metadata, _name
        def selection(item):
            argv = item['argv']
            position = 2 if argv[1] == '-u' else 1
            position += argv[position] == '-m'
            return dataset_cli_from_metadata(item.get('cli'), argv[position + 1:])
        selections = [(item, selection(item)) for item in candidates]
        matches = [item for item, selection in selections
                   if selection['status'] == 'observed_cli_selection' and _name(selection['value']) == _name(dataset)]
        candidates = matches or [item for item, selection in selections if selection['status'] == 'unverified']
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0]['selection_score'] == candidates[1]['selection_score']:
        return None  # The model may disambiguate using the paper; never guess a protocol.
    return candidates[0]
