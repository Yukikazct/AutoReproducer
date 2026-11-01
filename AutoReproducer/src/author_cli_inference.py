"""Read original argparse defaults and config choices without importing author code.

This is a navigation aid for execution planning, not proof that a paper protocol
has been reproduced. Optional arguments stay omitted so their author defaults
remain authoritative. Required values need a literal declaration or an
unambiguous checked-in configuration; arbitrary parameter values are not guessed.
"""
import ast
import math
from pathlib import PurePosixPath
import posixpath
import re


_UNKNOWN = object()
_CONFIG_NAMES = {'config', 'cfg', 'configuration', 'config_file', 'config_path', 'cfg_file', 'cfg_path'}


def _literal(node, constants):
    if node is None:
        return _UNKNOWN
    if isinstance(node, ast.Name):
        return constants.get(node.id, _UNKNOWN)
    if sum(1 for _ in ast.walk(node)) > 128:
        return _UNKNOWN
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNKNOWN
    def bounded(item):
        if item is None or type(item) is bool:
            return True
        if type(item) in (int, float):
            return not isinstance(item, float) or math.isfinite(item)
        if isinstance(item, str):
            return len(item) <= 4096 and not any(c in item for c in '\x00\r\n')
        if isinstance(item, (list, tuple)):
            return len(item) <= 64 and all(bounded(entry) for entry in item)
        return False
    return value if bounded(value) else _UNKNOWN


def _main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == '__main__')


def _quote(source, node):
    return {'line': node.lineno, 'quote': (ast.get_source_segment(source, node) or '')[:8192]}


def _config_choices(inventory, source_path, preferred):
    """Prefer the entrypoint's own project, never unrelated experiment configs."""
    names = []
    source_parent = PurePosixPath(source_path).parent
    for raw in inventory:
        path = PurePosixPath(raw)
        if path.is_absolute() or '..' in path.parts or any(part.startswith('.') for part in path.parts):
            continue
        if path.suffix.lower() not in {'.yaml', '.yml', '.json', '.toml'}:
            continue
        if path.name.lower() in {'pyproject.toml', 'package.json', 'package-lock.json', 'dataset.json'}:
            continue
        if path.suffix.lower() == '.json' and not (
                re.search(r'config|cfg', path.stem, re.I) or any(p.lower() in {'config', 'configs'} for p in path.parts[:-1])):
            continue
        names.append(path.as_posix())
    local = [name for name in names if source_parent == PurePosixPath('.')
             or source_parent in PurePosixPath(name).parents]
    if source_parent != PurePosixPath('.'):
        names = local or [name for name in names if len(PurePosixPath(name).parts) <= 2
                          and PurePosixPath(name).parts[0].lower() in {'config', 'configs'}]
    selected = sorted(set(names).intersection(preferred))
    return selected if selected else sorted(set(names))


def _relative_config(path, cwd):
    """Both inputs are checked repository-relative POSIX paths."""
    left, right = list(PurePosixPath(cwd).parts), list(PurePosixPath(path).parts)
    while left and right and left[0] == right[0]:
        left.pop(0)
        right.pop(0)
    return '/'.join(['..'] * len(left) + right)


def _existing_config(value, cwd, inventory):
    if not isinstance(value, str) or ':' in value:
        return None
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute():
        return None
    name = posixpath.normpath((PurePosixPath(cwd) / path).as_posix())
    if name == '..' or name.startswith('../') or name not in inventory:
        return None
    return name


def infer_cli(source, *, inventory, source_path, preferred_configs=(), cwd=None):
    """Describe invoked literal argparse parsers, or None when none is observed.

    ``arguments`` contains only required values grounded in declarations or a
    unique configuration path. ``defaults`` records original optional values.
    Config paths in evidence are repository-relative, while argv paths are
    relative to ``cwd`` (the entrypoint directory by default). Dynamic parser
    mutations produce an incomplete result rather than a supposedly valid argv.
    No module, function, config file or author expression is ever executed.
    """
    if not isinstance(source, str) or len(source) > 2 * 1024 * 1024:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    rebound_str = any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == 'str'
        or isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.name == 'str'
        or isinstance(node, ast.arg) and node.arg == 'str'
        or isinstance(node, ast.alias) and (node.asname or node.name) == 'str' for node in ast.walk(tree))
    cwd = PurePosixPath(cwd if cwd is not None else PurePosixPath(source_path).parent).as_posix()
    modules, constructors, constants, functions = set(), set(), {}, {}
    invocations = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node

    def scan(body, parsers=None, depth=0, called=()):
        parsers = {} if parsers is None else dict(parsers)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if _main_guard(node):
                scan(node.body, parsers, depth, called)
                continue
            if isinstance(node, ast.Import):
                modules.update(alias.asname or alias.name for alias in node.names if alias.name == 'argparse')
                continue
            if isinstance(node, ast.ImportFrom) and node.module == 'argparse' and node.level == 0:
                constructors.update(alias.asname or alias.name for alias in node.names if alias.name == 'ArgumentParser')
                continue
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)) else None
            assigned = (node.targets if isinstance(node, ast.Assign) else
                        [node.target] if isinstance(node, ast.AnnAssign) else [])
            for destination in assigned:
                if isinstance(destination, ast.Attribute) and isinstance(destination.value, ast.Name):
                    modules.discard(destination.value.id)
                    constructors.discard(destination.value.id)
            if target:
                functions.pop(target, None)
            if target and not isinstance(value, ast.Call):
                constants[target] = _literal(value, constants)
                if target in parsers:
                    parsers[target]['valid'] = False
                if isinstance(value, ast.Name) and value.id in parsers:
                    parsers[value.id]['valid'] = False
                modules.discard(target)
                constructors.discard(target)
            if not isinstance(value, ast.Call):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Assign, ast.AnnAssign)):
                    touched = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
                    for name in touched.intersection(parsers):
                        parsers[name]['valid'] = False
                continue
            func = value.func
            construct = (isinstance(func, ast.Name) and func.id in constructors
                         or isinstance(func, ast.Attribute) and func.attr == 'ArgumentParser'
                         and isinstance(func.value, ast.Name) and func.value.id in modules)
            if target:
                modules.discard(target)
                constructors.discard(target)
                if target in parsers and not construct:
                    parsers[target]['valid'] = False
            if target and construct:
                keywords = {item.arg: item.value for item in value.keywords}
                valid = not value.args and not any(key not in {
                    'prog', 'description', 'epilog', 'formatter_class', 'add_help', 'allow_abbrev',
                    'argument_default', 'usage'} for key in keywords)
                if 'argument_default' in keywords:
                    valid = valid and _literal(keywords['argument_default'], constants) is None
                parsers[target] = {'declarations': [], 'defaults': {}, 'valid': valid,
                                   'evidence': [_quote(source, value)]}
                continue
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in parsers:
                parser = parsers[func.value.id]
                if func.attr == 'add_argument':
                    parser['declarations'].append(value)
                elif func.attr == 'set_defaults':
                    if value.args or any(item.arg is None for item in value.keywords):
                        parser['valid'] = False
                    for item in value.keywords:
                        parser['defaults'][item.arg] = (_literal(item.value, constants), _quote(source, value))
                elif func.attr == 'add_argument_group' and target:
                    parsers[target] = parser
                elif func.attr in {'parse_args', 'parse_known_args'}:
                    implicit = (not value.args or len(value.args) == 1
                                and isinstance(value.args[0], ast.Constant) and value.args[0].value is None)
                    parser['valid'] = parser['valid'] and implicit and not value.keywords
                    snapshot = {**parser, 'declarations': list(parser['declarations']),
                                'defaults': dict(parser['defaults']), 'evidence': list(parser['evidence'])}
                    invocations.append((snapshot, _quote(source, value), dict(constants)))
                else:
                    parser['valid'] = False
                continue
            # Read only local functions that are visibly called without a
            # custom argument binding. Uninvoked helper parsers are irrelevant.
            calls = [item for item in ast.walk(value) if isinstance(item, ast.Call)]
            for call in calls:
                if isinstance(call.func, ast.Name) and call.func.id in functions and not call.args and not call.keywords:
                    definition = functions[call.func.id]
                    if (depth < 5 and call.func.id not in called and not definition.decorator_list
                            and not definition.args.posonlyargs and not definition.args.kwonlyargs
                            and len(definition.args.args) <= len(definition.args.defaults)):
                        scan(definition.body, parsers, depth + 1, (*called, call.func.id))
            for item in ast.walk(value):
                if isinstance(item, ast.Name) and item.id in parsers:
                    parsers[item.id]['valid'] = False

    scan(tree.body)
    if not invocations:
        return None
    result = {'arguments': [], 'missing': [], 'defaults': [], 'options': [], 'evidence': [],
              'complete': True, 'parsed': True, 'config_paths': []}
    if len(invocations) > 1:
        result['missing'].append('<multiple invoked argparse parsers>')
    choices = _config_choices(inventory, source_path, preferred_configs)
    for parser, parse_quote, bindings in invocations:
        result['evidence'].extend([*parser['evidence'], parse_quote])
        if not parser['valid']:
            result['missing'].append('<dynamic argparse configuration>')
        for declaration in parser['declarations']:
            tokens = [_literal(arg, bindings) for arg in declaration.args]
            if not tokens or any(not isinstance(arg, str) for arg in tokens):
                result['missing'].append('<dynamic argparse option>')
                continue
            flags = [arg for arg in tokens if arg.startswith('-')]
            positional = not flags
            if positional and (len(tokens) != 1 or not tokens[0]):
                result['missing'].append('<unsupported positional argument>')
                continue
            keywords = {item.arg: item.value for item in declaration.keywords}
            label = flags[0] if flags else tokens[0]
            if None in keywords:
                result['missing'].append(label)
                continue
            dest = _literal(keywords.get('dest'), bindings)
            if dest is _UNKNOWN and 'dest' in keywords:
                result['missing'].append(label)
                continue
            if dest is _UNKNOWN:
                dest = (next((flag for flag in flags if flag.startswith('--')), label)
                        .lstrip('-').replace('-', '_'))
            if not isinstance(dest, str):
                result['missing'].append(label)
                continue
            action = _literal(keywords.get('action'), bindings) if 'action' in keywords else 'store'
            nargs = _literal(keywords.get('nargs'), bindings) if 'nargs' in keywords else None
            required = (_literal(keywords.get('required'), bindings) if 'required' in keywords else False)
            default = _literal(keywords.get('default'), bindings)
            implicit = 'default' not in keywords
            if implicit:
                default = False if action == 'store_true' else True if action == 'store_false' else None
            quote = _quote(source, declaration)
            if dest in parser['defaults']:
                default, default_quote = parser['defaults'][dest]
                implicit = False
                result['evidence'].append(default_quote)
            # argparse applies an option's type converter to string defaults.
            # Resolve built-ins only; custom author callables are never run.
            converter = keywords.get('type')
            if isinstance(default, str) and converter is not None:
                conversions = {'str': str, 'int': int, 'float': float, 'bool': bool}
                convert = conversions.get(converter.id) if isinstance(converter, ast.Name) else None
                if convert is None:
                    default = _UNKNOWN
                else:
                    try:
                        default = convert(default)
                    except (ValueError, TypeError, OverflowError):
                        result['missing'].append(label)
                        default = _UNKNOWN
                    if isinstance(default, float) and not math.isfinite(default):
                        result['missing'].append(label)
                        default = _UNKNOWN
            if default is not _UNKNOWN:
                result['defaults'].append({'flags': tokens, 'dest': dest, 'value': default,
                                           'implicit': implicit, **quote})
            result['options'].append({'flags': tokens, 'dest': dest,
                'action': action if isinstance(action, str) else None,
                'nargs': nargs if nargs is not _UNKNOWN else '<dynamic>',
                'scalar_string': action == 'store' and nargs is None and (
                    converter is None or isinstance(converter, ast.Name) and converter.id == 'str' and not rebound_str),
                **quote})
            result['evidence'].append(quote)
            if action not in {'store', 'store_true', 'store_false', 'count', 'append', 'extend', 'store_const', 'append_const'}:
                result['missing'].append(label)
                continue
            if required is _UNKNOWN or nargs is _UNKNOWN:
                result['missing'].append(label)
                continue
            is_required = (positional and nargs not in ('?', '*') or required is True)
            if not is_required:
                # Existing literal defaults are evidence, not explicit argv
                # overrides. This preserves author seeds, epochs and paths.
                if dest in _CONFIG_NAMES and isinstance(default, str):
                    path = _existing_config(default, cwd, inventory)
                    if path:
                        result['config_paths'].append(path)
                continue
            scalar = lambda value: type(value) in (str, int, float) and value != ''
            supplied = default if scalar(default) else _UNKNOWN
            if supplied is _UNKNOWN:
                declared_choices = _literal(keywords.get('choices'), bindings)
                if isinstance(declared_choices, (list, tuple)) and len(declared_choices) == 1 and scalar(declared_choices[0]):
                    supplied = declared_choices[0]
            config = None
            if supplied is _UNKNOWN and dest in _CONFIG_NAMES and len(choices) == 1:
                config = choices[0]
                supplied = _relative_config(config, cwd)
            if supplied is not _UNKNOWN and dest in _CONFIG_NAMES:
                config = config or _existing_config(supplied, cwd, inventory)
                if config is None:
                    result['missing'].append(label)
                    continue  # A literal but missing author config cannot justify a different one.
            if supplied is _UNKNOWN or action not in {'store', 'append'} or nargs not in (None, '?', 1):
                result['missing'].append(label)
                continue
            if config:
                result['config_paths'].append(config)
                result['evidence'].append({'path': config, 'kind': 'configuration_inventory',
                                           'option': label, 'argument': str(supplied)})
            if not positional:
                result['arguments'].append(next((flag for flag in flags if flag.startswith('--')), label))
            result['arguments'].append(str(supplied))
    result['missing'] = list(dict.fromkeys(result['missing']))
    result['config_paths'] = list(dict.fromkeys(result['config_paths']))
    result['complete'] = not result['missing']
    return result
