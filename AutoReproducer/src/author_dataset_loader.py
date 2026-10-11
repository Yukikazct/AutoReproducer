"""Compile an original pure Planetoid loading call; never guess URLs or arguments."""
import ast
from pathlib import Path, PurePosixPath

from src.safety.paths import workspace_path


class AuthorLoaderUnsupported(ValueError):
    pass


def compile_author_loader(declaration, files, workspace, train, effective):
    """Resolve the selected source call, root, dataset and transforms statically.

    This currently reviews Planetoid's public split with an optional original
    NormalizeFeatures transform. It does not execute the entrypoint or accept
    model-supplied loader arguments. The original function runs during data
    preparation after its framework dependencies have been fixed.
    """
    path, function = declaration.get('path'), declaration.get('function')
    if (not isinstance(path, str) or path not in files or not path.endswith('.py')
            or not isinstance(function, str) or not function.isidentifier()):
        raise AuthorLoaderUnsupported('Author loader needs an already-read original Python function')
    original = files[path]
    source_path = workspace_path(workspace, path, 'author loader', must_exist=True, forbid_git=True)
    source = original['text']
    import hashlib
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != original['sha256']:
        raise AuthorLoaderUnsupported('Author loader source changed after discovery')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            annotations = [argument.annotation for argument in [*node.args.posonlyargs, *node.args.args,
                          *node.args.kwonlyargs] if argument.annotation is not None]
            if node.returns is not None:
                annotations.append(node.returns)
            if node.decorator_list or annotations or getattr(node, 'type_params', []):
                raise AuthorLoaderUnsupported('Loader module function headers must not execute decorators or annotations')
            for default in [*node.args.defaults, *(item for item in node.args.kw_defaults if item is not None)]:
                try:
                    value = ast.literal_eval(default)
                except (ValueError, TypeError) as exc:
                    raise AuthorLoaderUnsupported('Loader module function defaults must be primitive literals') from exc
                if type(value) not in (str, int, bool, type(None)):
                    raise AuthorLoaderUnsupported('Loader module function defaults must be primitive literals')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(functions) != 1 or functions[0].decorator_list:
        raise AuthorLoaderUnsupported('Author loader must be a unique undecorated top-level function')
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {'os.path', 'torch_geometric.transforms'}:
                    raise AuthorLoaderUnsupported('Loader module has an unreviewed top-level import')
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module != 'torch_geometric.datasets':
                raise AuthorLoaderUnsupported('Loader module has an unreviewed top-level import')
            for alias in node.names:
                if alias.name not in {'Planetoid', 'Coauthor', 'Amazon'}:
                    raise AuthorLoaderUnsupported('Loader module imports an unreviewed dataset class')
                aliases[alias.asname or alias.name] = node.module + '.' + alias.name
        elif not isinstance(node, ast.FunctionDef) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise AuthorLoaderUnsupported('Loader module must contain only reviewed imports and function definitions')

    argv = train['argv']
    position = 2 if argv[1] == '-u' else 1
    entry = str(PurePosixPath(train['cwd']) / argv[position])
    entry_source = files[entry]['text']
    entry_tree = ast.parse(entry_source)
    call_names = set()
    for node in entry_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        roots = [PurePosixPath(entry).parent, PurePosixPath('.')]
        if node.level:
            roots = [PurePosixPath(entry).parent]
            for _ in range(node.level - 1):
                roots = [root.parent for root in roots]
        module = PurePosixPath(*(node.module or '').split('.'))
        matches = {str(root / module) + '.py' for root in roots}
        if path not in matches:
            continue
        for alias in node.names:
            if alias.name == function:
                call_names.add(alias.asname or function)
            elif alias.name == '*':
                call_names.add(function)
    if not call_names:
        raise AuthorLoaderUnsupported('Selected entrypoint does not import the original loader function')
    cli = {name: item['value'] for name, item in effective.items() if item.get('action') != 'function_configuration'}
    cli_assignments = [node for node in entry_tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                 and node.value.func.attr == 'parse_args' and not node.value.args and not node.value.keywords]
    cli_names = {target.id for node in cli_assignments for target in node.targets if isinstance(target, ast.Name)}
    if len(cli_names) != 1 or len(cli_assignments) != 1:
        raise AuthorLoaderUnsupported('Loader arguments require one original compiled CLI namespace')

    def primitive(node, env):
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool, type(None)):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in cli_names:
            if node.attr in cli:
                return cli[node.attr]
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = primitive(item, env)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not primitive(node.operand, env)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = primitive(node.left, env), primitive(node.comparators[0], env)
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
        raise AuthorLoaderUnsupported('Loader branch or argument is not fixed by original literals/CLI')

    selected = []
    def inspect_entry(nodes, env):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                call_names.discard(node.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id in cli_names:
                    if node not in cli_assignments:
                        raise AuthorLoaderUnsupported('Original CLI namespace was rebound before data loading')
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    root = child.value
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in cli_names:
                        raise AuthorLoaderUnsupported('Original loader CLI values were mutated')
                if isinstance(child, ast.Call) and any(isinstance(arg, ast.Name) and arg.id in cli_names
                        for arg in [*child.args, *(kw.value for kw in child.keywords)]):
                    raise AuthorLoaderUnsupported('An opaque call can mutate the original loader CLI namespace')
            if isinstance(node, ast.If):
                try:
                    condition = primitive(node.test, env)
                except AuthorLoaderUnsupported:
                    if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id in call_names for call in ast.walk(node)):
                        raise
                    continue
                inspect_entry(node.body if condition else node.orelse, env)
                continue
            calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                     and isinstance(call.func, ast.Name) and call.func.id in call_names]
            if calls and not (isinstance(node, ast.Assign) and len(node.targets) == 1
                              and isinstance(node.targets[0], ast.Name) and node.value in calls):
                raise AuthorLoaderUnsupported('Original loader invocation must be a direct selected assignment')
            selected.extend((call, dict(env)) for call in calls)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in cli_names and not (isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args'):
                            raise AuthorLoaderUnsupported('Original CLI namespace was rebound before data loading')
                        call_names.discard(target.id)
                        try:
                            env[target.id] = primitive(node.value, env)
                        except AuthorLoaderUnsupported:
                            env.pop(target.id, None)
    inspect_entry(entry_tree.body, {'__name__': '__main__'})
    if len(selected) != 1:
        raise AuthorLoaderUnsupported('Selected author invocation needs exactly one original loader call')
    (call, call_env), fn = selected[0], functions[0]
    if fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise AuthorLoaderUnsupported('Loader signature needs fixed ordinary arguments and literal defaults')
    names = [arg.arg for arg in fn.args.args]
    if len(call.args) > len(names) or any(isinstance(arg, ast.Starred) for arg in call.args):
        raise AuthorLoaderUnsupported('Loader invocation cannot expand dynamic arguments')
    arguments = dict(zip(names, [primitive(arg, call_env) for arg in call.args]))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names or keyword.arg in arguments:
            raise AuthorLoaderUnsupported('Loader keyword is dynamic, unknown or duplicated')
        arguments[keyword.arg] = primitive(keyword.value, call_env)
    for name, default in zip(names[-len(fn.args.defaults):] if fn.args.defaults else [], fn.args.defaults):
        if name not in arguments:
            arguments[name] = primitive(default, {})
    if set(arguments) != set(names):
        raise AuthorLoaderUnsupported('Loader invocation omits a required original argument')
    if any(value is not None and type(value) not in (str, int, bool) for value in arguments.values()):
        raise AuthorLoaderUnsupported('Loader arguments must be fixed primitive author values')
    dataset, configuration = object(), {}
    env = {**arguments, '__file__': str(source_path)}

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = qualified(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def loader_value(node):
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            values = [loader_value(arg) for arg in node.args]
            if name in {'os.path.join', 'os.path.dirname', 'os.path.realpath'} and not node.keywords:
                if any(not isinstance(value, str) for value in values):
                    raise AuthorLoaderUnsupported('Loader path needs original string components')
                if name == 'os.path.join':
                    if not values:
                        raise AuthorLoaderUnsupported('Loader path join has invalid arity')
                    return str(Path(values[0]).joinpath(*values[1:]))
                if len(values) != 1:
                    raise AuthorLoaderUnsupported('Loader path operation has invalid arity')
                return str(Path(values[0]).parent) if name == 'os.path.dirname' else str(Path(values[0]).resolve())
            if name == 'torch_geometric.transforms.NormalizeFeatures' and not values and not node.keywords:
                return ('NormalizeFeatures',)
            if name == 'torch_geometric.datasets.Planetoid':
                if configuration or len(values) != 2 or node.keywords:
                    raise AuthorLoaderUnsupported('Reviewed Planetoid call needs only original root and dataset name')
                root, data_name = values
                if data_name not in {'Cora', 'CiteSeer', 'PubMed'} or not isinstance(root, str):
                    raise AuthorLoaderUnsupported('Dataset is outside the reviewed public Planetoid inputs')
                resolved = Path(root).resolve()
                if not resolved.is_relative_to(Path(workspace).resolve()) or resolved == Path(workspace).resolve():
                    raise AuthorLoaderUnsupported('Original dataset root must stay within the author workspace')
                relative = resolved.relative_to(Path(workspace).resolve()).as_posix()
                workspace_path(workspace, relative, 'author dataset root', forbid_git=True)
                configuration.update(framework='torch_geometric.datasets.Planetoid', name=data_name,
                                     root=relative, split='public', transform=None)
                return dataset
            raise AuthorLoaderUnsupported('Loader calls an unreviewed operation')
        return primitive(node, env)

    def review_body(nodes):
        for node in nodes:
            if isinstance(node, ast.If):
                review_body(node.body if primitive(node.test, env) else node.orelse)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = loader_value(node.value)
                if isinstance(target, ast.Name):
                    env[target.id] = value
                elif (isinstance(target, ast.Attribute) and target.attr == 'transform'
                      and isinstance(target.value, ast.Name) and env.get(target.value.id) is dataset
                      and value == ('NormalizeFeatures',)):
                    configuration['transform'] = 'NormalizeFeatures'
                else:
                    raise AuthorLoaderUnsupported('Loader mutation is outside the reviewed original transform')
            elif isinstance(node, ast.Return) and loader_value(node.value) is dataset:
                return True
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise AuthorLoaderUnsupported('Original loader contains unreviewed executable statements')
        return False
    if not review_body(fn.body) or not configuration:
        raise AuthorLoaderUnsupported('Original loader does not directly return its reviewed dataset')
    return {'kind': 'author_loader', 'path': path, 'function': function, 'arguments': arguments,
            **configuration, 'source_sha256': original['sha256'],
            'call': {'path': entry, 'quote': ast.get_source_segment(entry_source, call),
                     'sha256': files[entry]['sha256']}, 'definition': ast.get_source_segment(source, fn)}
