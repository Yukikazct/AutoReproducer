"""Check literal author CLI dataset choices without asserting runtime identity."""
import ast
from pathlib import PurePosixPath
import re

from src.author_cli_inference import infer_cli
from src.experiment_request import normalize_experiment_request


def _name(value):
    return re.sub(r'[\W_]+', '', value.casefold()) if isinstance(value, str) else ''


def _entrypoint_cli(packet, workspace, step):
    """Follow one directly invoked, pinned local parser function without imports.

    The appended call exists only in the static analyzer input. The original
    entrypoint supplies that exact zero-argument call; no author file is edited
    or executed, and all declaration/default quotes retain original line numbers.
    """
    from src.author_cli_inference import _main_guard
    from src.discovered_repository_plan import _load_execution_source
    name = step['entrypoint']['source_path']
    source = _load_execution_source(packet, workspace, name)
    cli = infer_cli(source['text'], inventory=packet['file_inventory'], source_path=name, cwd=step['cwd'])
    provenance = {'source_path': name, 'source_sha256': source['sha256']}
    if cli is not None:
        return cli, provenance
    try:
        tree = ast.parse(source['text'])
    except SyntaxError:
        return None, provenance
    imported, calls = {}, []

    def scan(body):
        for node in body:
            if _main_guard(node):
                scan(node.body)
                continue
            if isinstance(node, ast.ImportFrom) and node.module and node.level in (0, 1):
                for alias in node.names:
                    imported[alias.asname or alias.name] = (node, alias.name)
                continue
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in imported and not value.args and not value.keywords):
                calls.append((imported[value.func.id], value))
            # A rebound or conditionally changed alias cannot justify a call.
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    imported.pop(item.id, None)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    imported.pop(item.name, None)
                elif isinstance(item, ast.alias):
                    imported.pop(item.asname or item.name, None)
    scan(tree.body)
    found = []
    for (statement, function), call in calls:
        relative = PurePosixPath(*statement.module.split('.')).with_suffix('.py')
        parent = PurePosixPath(name).parent
        if statement.level:
            if step['entrypoint']['kind'] != 'module':
                continue  # Relative imports require an actual package invocation.
            candidates = {str(parent / relative)}
        else:
            base = PurePosixPath(step['cwd']) if step['entrypoint']['kind'] == 'module' else parent
            candidates = {str(base / relative)}
        candidates.intersection_update(packet['file_inventory'])
        if len(candidates) != 1:
            continue
        helper_name = candidates.pop()
        helper = _load_execution_source(packet, workspace, helper_name)
        helper_tree = ast.parse(helper['text'])
        if (any(_main_guard(node) for node in helper_tree.body)
                or infer_cli(helper['text'], inventory=packet['file_inventory'], source_path=helper_name) is not None):
            continue  # Imported helpers never run their __main__ demonstrations.
        if sum(isinstance(node, ast.FunctionDef) and node.name == function for node in helper_tree.body) != 1:
            continue
        parsed = infer_cli(helper['text'] + '\n' + function + '()\n', inventory=packet['file_inventory'],
                           source_path=helper_name, cwd=step['cwd'])
        if parsed is not None:
            found.append((parsed, {'source_path': helper_name, 'source_sha256': helper['sha256'],
                'parser_call': {**provenance, 'line': call.lineno,
                                'quote': ast.get_source_segment(source['text'], call)}}))
    return found[0] if len(found) == 1 else (None, provenance)


def dataset_cli_selection(source, arguments, *, inventory=(), source_path='train.py', cwd='.'):
    """Read one static scalar dataset selector; dynamic/config selectors stay unknown.

    This describes argparse input only. Author code may transform it or choose
    data elsewhere, so this is never evidence of the actual loaded dataset.
    """
    cli = infer_cli(source, inventory=inventory, source_path=source_path, cwd=cwd)
    return dataset_cli_from_metadata(cli, arguments)


def dataset_cli_from_metadata(cli, arguments):
    """Use only metadata produced by the source reader, never model labels."""
    unknown = {'status': 'unverified', 'reason': 'No single static scalar dataset option in the entrypoint.'}
    if cli is None or any(item.startswith('<') for item in cli['missing']):
        return unknown
    declarations = cli.get('options', [])
    # Do not scan through values, positionals, REMAINDER or dynamic actions.
    # Unknown/abbreviated flags also stay unknown rather than falling back to
    # a default that the real parser may override.
    if any(not item['flags'] or any(not flag.startswith('-') for flag in item['flags'])
           or item['nargs'] is not None or item['action'] not in {
               'store', 'store_true', 'store_false', 'store_const', 'count'} for item in declarations):
        return unknown
    options = [item for item in declarations if item['dest'] in {'dataset', 'dataset_name'}]
    if len(options) != 1 or not options[0]['scalar_string']:
        return unknown
    option = options[0]
    flags = option['flags']
    if not flags or any(not flag.startswith('-') for flag in flags):
        return unknown
    defaults = [item for item in cli['defaults'] if item['dest'] == option['dest']]
    value = defaults[0]['value'] if len(defaults) == 1 else None
    origin = 'source_default'
    by_flag = {flag: item for item in declarations for flag in item['flags']}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == '--':
            break
        flag, equals, supplied = token.partition('=')
        declaration = by_flag.get(flag)
        if declaration is None:
            return unknown
        if declaration['action'] != 'store':
            if equals:
                return unknown
            index += 1
            continue
        if not equals:
            index += 1
            if index >= len(arguments) or arguments[index].startswith('--'):
                return unknown
            supplied = arguments[index]
        if flag in flags:
            value, origin = supplied, 'argv'
        index += 1
    if not _name(value):
        return unknown
    return {'status': 'observed_cli_selection', 'value': value, 'origin': origin,
            'flags': flags, 'declaration': {'line': option['line'], 'quote': option['quote']},
            'default_evidence': defaults}


def bind_author_dataset_selection(steps, packet, workspace):
    """Reject known request conflicts before freezing any author command."""
    from src.author_entrypoints import resolve_entrypoint
    from src.discovered_repository_plan import PlanEvidenceError

    requested = normalize_experiment_request(packet.get('experiment_request'))
    if not requested:
        return None
    observations = []
    for step in steps:
        original = {**step, 'entrypoint': {key: step['entrypoint'][key] for key in ('kind', 'target')}}
        entrypoint, _, arguments = resolve_entrypoint(original, packet, workspace)
        record = {'step': step['id'], 'entrypoint': entrypoint['source_path'], 'status': 'unverified'}
        if entrypoint['kind'] in {'script', 'module'} and requested.get('dataset'):
            cli, provenance = _entrypoint_cli(packet, workspace, step)
            record.update(dataset_cli_from_metadata(cli, arguments), **provenance)
            if record['status'] == 'observed_cli_selection':
                if _name(record['value']) != _name(requested['dataset']):
                    raise PlanEvidenceError('Author command dataset differs from the user request: '
                        + record['value'] + ' (requested ' + requested['dataset'] + '). '
                        'Select a matching original invocation; do not change its training settings.',
                        planning_evidence={'stage': 'experiment_request_binding', 'request': requested,
                                           'observation': record, 'argv': step['argv'], 'cwd': step['cwd']})
                record['status'] = 'matched_cli_selection'
        observations.append(record)
    return {'request': requested, 'dataset_status': ('matched_cli_selection'
            if requested.get('dataset') and all(item['status'] == 'matched_cli_selection' for item in observations)
            else 'unverified'), 'method_status': 'unverified', 'steps': observations,
            'runtime_dataset_identity_verified': False}


def select_dataset_loop_bindings(argv, choices, packet, workspace, cwd):
    """Select only an author-enumerated dataset; leave seeds/settings alone."""
    from src.author_entrypoints import resolve_entrypoint

    bindings = {name: values[0] for name, values in choices.items()}
    requested = normalize_experiment_request(packet.get('experiment_request')).get('dataset')
    if not requested or not choices:
        return bindings
    expanded = [re.sub(r'\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))',
                       lambda match: bindings.get(match[1] or match[2], match[0]), token) for token in argv]
    entrypoint, _, arguments = resolve_entrypoint({'argv': expanded, 'cwd': cwd}, packet, workspace)
    cli, _ = _entrypoint_cli(packet, workspace, {'entrypoint': entrypoint, 'cwd': cwd})
    if cli is None or any(item.startswith('<') for item in cli['missing']):
        return bindings
    observed = dataset_cli_from_metadata(cli, arguments)
    if observed['status'] != 'observed_cli_selection' or observed['origin'] != 'argv':
        return bindings
    options = [item for item in cli['options'] if item['dest'] in {'dataset', 'dataset_name'}]
    if len(options) != 1 or not options[0]['scalar_string']:
        return bindings
    flags = options[0]['flags']
    for index, token in enumerate(argv):
        if token == '--':
            break
        flag, equals, value = token.partition('=')
        if flag not in flags:
            continue
        value = value if equals else argv[index + 1] if index + 1 < len(argv) else ''
        variable = re.fullmatch(r'\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))', value)
        if variable:
            name = variable[1] or variable[2]
            matches = [item for item in choices.get(name, []) if _name(item) == _name(requested)]
            references = sum(len(re.findall(r'\$(?:\{' + re.escape(name) + r'\}|' + re.escape(name)
                                           + r'(?!\w))', token)) for token in argv)
            if len(matches) == 1 and references == 1:
                bindings[name] = matches[0]
    return bindings
