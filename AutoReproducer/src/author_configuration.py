"""Resolve literal configuration dictionaries at original selected calls, without execution."""
import ast
from dataclasses import dataclass
import math
from pathlib import PurePosixPath


class ConfigurationUnsupported(ValueError):
    pass


@dataclass
class Cell:
    kind: str
    value: object
    path: str = ''
    node: object = None


def resolve_configured_parameters(files, entry, cli, revision):
    """Trace root bindings and known branches; never run imported or authored code."""
    cache, visiting, configured = {}, set(), {}
    namespace = Cell('namespace', {key: Cell('scalar', item['value']) for key, item in cli.items()})

    def source(cell):
        item = files[cell.path]
        return {'source': 'repository', 'path': cell.path,
                'quote': ast.get_source_segment(item['text'], cell.node),
                'sha256': item['sha256'], 'revision': revision}

    def evaluate(node, env, path):
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ConfigurationUnsupported('Unknown configuration binding: ' + node.id)
            return env[node.id]
        if isinstance(node, ast.Attribute):
            base = evaluate(node.value, env, path)
            if base.kind not in {'namespace', 'module'} or node.attr not in base.value:
                raise ConfigurationUnsupported('Configuration attributes must be known CLI/module bindings')
            return base.value[node.attr]
        if isinstance(node, ast.Subscript):
            base, key = evaluate(node.value, env, path), evaluate(node.slice, env, path)
            if base.kind not in {'dict', 'sequence'} or key.kind != 'scalar':
                raise ConfigurationUnsupported('Configuration lookup must index a literal container')
            try:
                return base.value[key.value]
            except (KeyError, IndexError, TypeError) as exc:
                raise ConfigurationUnsupported('Selected original configuration key is absent') from exc
        if isinstance(node, ast.Dict):
            values = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    raise ConfigurationUnsupported('Literal dictionary unpacking is unsupported')
                key = evaluate(key, env, path)
                if key.kind != 'scalar' or type(key.value) not in (str, int) or key.value in values:
                    raise ConfigurationUnsupported('Configuration keys must be unique literal strings/integers')
                values[key.value] = evaluate(value, env, path)
            return Cell('dict', values, path, node)
        if isinstance(node, (ast.List, ast.Tuple)):
            return Cell('sequence', [evaluate(value, env, path) for value in node.elts], path, node)
        if isinstance(node, ast.BoolOp):
            values = [evaluate(value, env, path) for value in node.values]
            if any(value.kind != 'scalar' for value in values):
                raise ConfigurationUnsupported('Configuration branches need primitive conditions')
            value = all(item.value for item in values) if isinstance(node.op, ast.And) else any(item.value for item in values)
            return Cell('scalar', value, path, node)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = evaluate(node.left, env, path), evaluate(node.comparators[0], env, path)
            if left.kind != 'scalar' or right.kind != 'scalar' or type(node.ops[0]) not in (ast.Eq, ast.NotEq):
                raise ConfigurationUnsupported('Configuration branch comparison is unsupported')
            value = left.value == right.value
            return Cell('scalar', value if isinstance(node.ops[0], ast.Eq) else not value, path, node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = evaluate(node.operand, env, path)
            if value.kind != 'scalar':
                raise ConfigurationUnsupported('Configuration condition must be primitive')
            return Cell('scalar', not value.value, path, node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'parse_args':
            # CLI declarations and the selected invocation were already compiled.
            if node.args or node.keywords:
                raise ConfigurationUnsupported('An alternate parse_args input cannot reuse the selected CLI')
            return namespace
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError) as exc:
            raise ConfigurationUnsupported('Dynamic configuration expressions are unsupported') from exc
        if type(value) not in (str, int, float, bool, type(None)) or isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationUnsupported('Configuration values must be finite primitive literals')
        return Cell('scalar', value, path, node)

    def module_path(node, path):
        parent = PurePosixPath(path).parent
        if node.level:
            for _ in range(node.level - 1):
                parent = parent.parent
            roots = [parent]
        else:
            roots = [parent, PurePosixPath(entry).parent, PurePosixPath('.')]
        relative = PurePosixPath(*(node.module or '').split('.'))
        candidates = {str(root / relative) + suffix for root in roots for suffix in ('.py', '/__init__.py')}
        existing = sorted(candidates.intersection(files))
        if len(existing) > 1:
            raise ConfigurationUnsupported('Configuration import resolves to ambiguous source files')
        return existing[0] if existing else None

    def inspect_calls(node, env, path):
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not any(keyword.arg is None for keyword in call.keywords):
                continue
            if not isinstance(call.func, ast.Name):
                continue
            function = env.get(call.func.id)
            if function is None or function.kind != 'function':
                continue
            if call.args:
                raise ConfigurationUnsupported('Configured author calls with positional arguments are unsupported')
            fn = function.node
            if fn.args.posonlyargs or fn.args.vararg:
                raise ConfigurationUnsupported('Configured author function signature is unsupported')
            provided, bindings = {}, []
            for keyword in call.keywords:
                value = evaluate(keyword.value, env, path)
                if keyword.arg is None:
                    if value.kind != 'dict':
                        raise ConfigurationUnsupported('Author keyword expansion must resolve to a literal dictionary')
                    pairs = value.value.items()
                    bindings.append(source(value))
                else:
                    pairs = [(keyword.arg, value)]
                for name, cell in pairs:
                    if not isinstance(name, str) or not name.isidentifier() or name in provided:
                        raise ConfigurationUnsupported('Author keyword expansion has invalid or duplicate names')
                    provided[name] = cell
            parameters = {}
            offset = len(fn.args.args) - len(fn.args.defaults)
            for index, argument in enumerate(fn.args.args):
                if argument.arg in provided:
                    parameters[argument.arg] = provided.pop(argument.arg)
                elif index >= offset:
                    parameters[argument.arg] = evaluate(fn.args.defaults[index - offset], {}, function.path)
                else:
                    raise ConfigurationUnsupported('Configured author call omits a required argument')
            for argument, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
                if argument.arg in provided:
                    parameters[argument.arg] = provided.pop(argument.arg)
                elif default is not None:
                    parameters[argument.arg] = evaluate(default, {}, function.path)
                else:
                    raise ConfigurationUnsupported('Configured author call omits a required keyword')
            if provided and fn.args.kwarg is None:
                raise ConfigurationUnsupported('Author configuration contains unsupported function keywords')
            parameters.update(provided)
            for name, cell in parameters.items():
                if cell.kind != 'scalar' or not cell.path:
                    raise ConfigurationUnsupported('Selected function parameters need original literal source values')
                key = function.node.name + '.' + name if name in cli else name
                declaration = {'value': cell.value, 'action': 'function_configuration', 'type': type(cell.value).__name__,
                    'required': False, 'source': source(cell),
                    'configuration_binding': {'call': source(Cell('call', None, path, call)), 'dictionaries': bindings}}
                if key in configured and configured[key]['value'] != cell.value:
                    raise ConfigurationUnsupported('Multiple author calls have incompatible unqualified parameters: ' + key)
                configured[key] = declaration

    def invalidate_target(node, env):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if isinstance(node, ast.Name):
            affected = env.get(node.id)
            if affected is None:
                return
            def descendants(cell):
                yield id(cell)
                if cell.kind in {'dict', 'namespace', 'module'}:
                    for child in cell.value.values():
                        yield from descendants(child)
                elif cell.kind == 'sequence':
                    for child in cell.value:
                        yield from descendants(child)
            identifiers = set(descendants(affected))
            for name, cell in list(env.items()):
                if identifiers.intersection(descendants(cell)):
                    del env[name]

    def invalidate_calls(node, env):
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if isinstance(call.func, ast.Attribute):
                root = call.func.value
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name) and env.get(root.id, Cell('', None)).kind in {'dict', 'sequence', 'module'}:
                    invalidate_target(root, env)
            if isinstance(call.func, ast.Name) and call.func.id not in {'print', 'len'}:
                for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
                    try:
                        affected = evaluate(argument, env, '')
                    except ConfigurationUnsupported:
                        affected = None
                    if affected is not None and affected.kind in {'dict', 'sequence', 'module'}:
                        invalidate_target(argument, env)
                function = env.get(call.func.id)
                if function is not None and function.kind == 'function':
                    # An intervening local call can mutate a module dictionary
                    # without receiving it as an argument. Do not assume purity.
                    referenced = {child.id for child in ast.walk(function.node) if isinstance(child, ast.Name)}
                    for name in referenced:
                        if env.get(name, Cell('', None)).kind in {'dict', 'sequence', 'module'}:
                            invalidate_target(ast.Name(id=name), env)

    def scan(nodes, env, path, collect):
        for node in nodes:
            if isinstance(node, ast.FunctionDef):
                if not node.decorator_list:
                    env[node.name] = Cell('function', None, path, node)
                else:
                    env.pop(node.name, None)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    env.pop(alias.asname or alias.name.split('.')[0], None)
            elif isinstance(node, ast.ImportFrom):
                target = module_path(node, path)
                if target:
                    imported = globals_for(target)
                    for alias in node.names:
                        if alias.name == '*':
                            continue
                        if alias.name in imported:
                            env[alias.asname or alias.name] = imported[alias.name]
                        else:
                            env.pop(alias.asname or alias.name, None)
                else:
                    for alias in node.names:
                        if alias.name == '*':
                            env.clear()
                        else:
                            env.pop(alias.asname or alias.name, None)
            elif isinstance(node, ast.If):
                try:
                    condition = evaluate(node.test, env, path)
                    if condition.kind != 'scalar':
                        raise ConfigurationUnsupported('Dynamic author condition')
                except ConfigurationUnsupported:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                invalidate_target(target, env)
                        elif isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                            env.pop(child.name, None)
                        elif isinstance(child, ast.Call):
                            invalidate_calls(child, env)
                    continue
                scan(node.body if condition.value else node.orelse, env, path, collect)
            elif isinstance(node, ast.Assign):
                if collect:
                    inspect_calls(node.value, env, path)
                try:
                    value = evaluate(node.value, env, path)
                except ConfigurationUnsupported:
                    value = None
                for target in node.targets:
                    if isinstance(target, ast.Name) and value is not None:
                        env[target.id] = value
                    else:
                        invalidate_target(target, env)
            elif isinstance(node, ast.Expr):
                if collect:
                    inspect_calls(node.value, env, path)
                invalidate_calls(node.value, env)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                invalidate_target(node.target, env)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    invalidate_target(target, env)
            elif isinstance(node, ast.ClassDef):
                env.pop(node.name, None)
                invalidate_calls(node, env)
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                        for target in child.targets if isinstance(child, ast.Assign) else [child.target]:
                            invalidate_target(target, env)
            elif isinstance(node, (ast.For, ast.While, ast.Try, ast.With)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                        for target in child.targets if isinstance(child, ast.Assign) else [child.target]:
                            invalidate_target(target, env)
                    elif isinstance(child, ast.Call):
                        invalidate_calls(child, env)

    def globals_for(path):
        if path in cache:
            return cache[path]
        if path in visiting:
            raise ConfigurationUnsupported('Cyclic literal configuration imports are unsupported')
        visiting.add(path)
        env = {'__name__': Cell('scalar', path.removesuffix('.py').replace('/', '.'))}
        scan(ast.parse(files[path]['text']).body, env, path, False)
        visiting.remove(path)
        cache[path] = env
        return env

    env = {'__name__': Cell('scalar', '__main__')}
    scan(ast.parse(files[entry]['text']).body, env, entry, True)
    return configured
