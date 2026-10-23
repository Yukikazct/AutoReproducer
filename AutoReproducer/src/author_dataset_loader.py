"""Compile an original pure Planetoid loading call; never guess URLs or arguments."""
import ast
from pathlib import Path, PurePosixPath

from src.safety.paths import workspace_path


class AuthorLoaderUnsupported(ValueError):
    pass


def compile_author_loader(declaration, files, workspace, train, effective):
    """Retain legacy pure contracts; review plain record wrappers separately."""
    try:
        return _compile_author_loader(declaration, files, workspace, train, effective)
    except AuthorLoaderUnsupported:
        source = files.get(declaration.get('path'), {}).get('text', '')
        tree = ast.parse(source)
        definitions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == declaration.get('function')]
        if len(selected) != 1 or not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in definitions - {selected[0].name} for node in ast.walk(selected[0])):
            raise
        return _compile_author_loader(declaration, files, workspace, train, effective, wrapper=True)


def _validate_header(node):
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
            *(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)]
    if (node.decorator_list or node.returns is not None or any(arg.annotation is not None for arg in args)
            or getattr(node, 'type_params', [])):
        raise AuthorLoaderUnsupported('Original wrapper headers cannot execute decorators or annotations')
    for default in [*node.args.defaults, *(item for item in node.args.kw_defaults if item is not None)]:
        try:
            value = ast.literal_eval(default)
        except (ValueError, TypeError) as exc:
            raise AuthorLoaderUnsupported('Original wrapper defaults must be primitive literals') from exc
        if type(value) not in (str, int, float, bool, type(None)):
            raise AuthorLoaderUnsupported('Original wrapper defaults must be primitive literals')


def _validate_record_class(node):
    if (node.decorator_list or node.keywords or getattr(node, 'type_params', [])
            or len(node.bases) > 1 or any(not isinstance(base, ast.Name) or base.id != 'object' for base in node.bases)):
        raise AuthorLoaderUnsupported('Original record class needs the plain object base without metaclasses')
    names = set()
    for child in node.body:
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
            continue
        if not isinstance(child, ast.FunctionDef) or child.name in names:
            raise AuthorLoaderUnsupported('Original record class can contain only inert unique method definitions')
        names.add(child.name)
        _validate_header(child)
        if child.name.startswith('__') and child.name not in {'__init__', '__getitem__', '__len__', '__repr__'}:
            raise AuthorLoaderUnsupported('Original record cannot override attribute access or object construction')
    if '__init__' not in names:
        raise AuthorLoaderUnsupported('Original record needs an explicit reviewed initializer')


def _review_wrapper(source, tree, aliases, source_path, workspace, train, arguments, selected):
    """Interpret one bounded original loading path; execute no author expressions."""
    import hashlib
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    bindings = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    import_bindings = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == '*' for alias in node.names):
                raise AuthorLoaderUnsupported('Original wrapper cannot use wildcard import bindings')
            import_bindings.extend(alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                                   for alias in node.names)
    all_bindings = bindings + import_bindings
    if len(all_bindings) != len(set(all_bindings)) or 'object' in all_bindings:
        raise AuthorLoaderUnsupported('Original wrapper imports and definitions must have unique unshadowed bindings')
    if len(bindings) != len(set(bindings)) or set(bindings) & set(aliases) or 'object' in bindings or 'object' in aliases:
        raise AuthorLoaderUnsupported('Original loading definitions must have unique unshadowed names')
    for node in functions.values():
        _validate_header(node)
    for node in classes.values():
        _validate_record_class(node)
    used, imported, active, configuration = set(), set(), [], {}
    dataset, data = object(), object()

    class Field:
        def __init__(self, name, indices=False):
            self.name, self.indices = name, indices

    class Record:
        def __init__(self, cls):
            self.cls, self.fields = cls, {}

    def bind(fn, values, keywords):
        if fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            raise AuthorLoaderUnsupported('Original loading helpers need fixed ordinary arguments')
        names = [arg.arg for arg in fn.args.args]
        if len(values) > len(names) or any(key not in names for key in keywords):
            raise AuthorLoaderUnsupported('Original loading call has unknown or excess arguments')
        result = dict(zip(names, values))
        if result.keys() & keywords.keys():
            raise AuthorLoaderUnsupported('Original loading call duplicates an argument')
        result.update(keywords)
        for name, default in zip(names[-len(fn.args.defaults):] if fn.args.defaults else [], fn.args.defaults):
            result.setdefault(name, ast.literal_eval(default))
        if set(result) != set(names):
            raise AuthorLoaderUnsupported('Original loading call omitted a required argument')
        return result

    def qualified(node, env):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id not in env and node.id in aliases:
            imported.add(node.id)
            return aliases[node.id] + ''.join('.' + part for part in reversed(parts))
        return ''

    def literal(value):
        return value is None or type(value) in (str, int, float, bool)

    def value(node, env):
        if isinstance(node, ast.Constant) and literal(node.value):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(value(item, env) for item in node.elts)
        if isinstance(node, ast.Dict):
            keys = [value(key, env) if key is not None else None for key in node.keys]
            if any(type(key) is not str or key.startswith('_') for key in keys) or len(keys) != len(set(keys)):
                raise AuthorLoaderUnsupported('Original record dictionaries need unique fixed public string keys')
            return {key: value(item, env) for key, item in zip(keys, node.values)}
        if isinstance(node, ast.JoinedStr):
            parts = []
            for item in node.values:
                if isinstance(item, ast.Constant) and type(item.value) is str:
                    parts.append(item.value)
                elif isinstance(item, ast.FormattedValue) and item.conversion == -1 and item.format_spec is None:
                    part = value(item.value, env)
                    if not literal(part):
                        raise AuthorLoaderUnsupported('Original formatted paths need primitive fixed values')
                    parts.append(str(part))
                else:
                    raise AuthorLoaderUnsupported('Original formatted paths cannot execute a formatter')
            return ''.join(parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = value(node.operand, env)
            if not literal(operand):
                raise AuthorLoaderUnsupported('Original branch needs a fixed primitive condition')
            return not operand
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                result = value(item, env)
                if not literal(result):
                    raise AuthorLoaderUnsupported('Original branch needs a fixed primitive condition')
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left, env), value(node.comparators[0], env)
            if not literal(left) or not (literal(right) or type(right) is tuple and all(literal(x) for x in right)):
                raise AuthorLoaderUnsupported('Original branch cannot compare dynamic data values')
            operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                          ast.Is: lambda: left is right, ast.IsNot: lambda: left is not right,
                          ast.In: lambda: left in right, ast.NotIn: lambda: left not in right}
            if type(node.ops[0]) in operations:
                return operations[type(node.ops[0])]()
        if isinstance(node, ast.Attribute):
            owner = value(node.value, env)
            if owner is data and node.attr in {'x', 'y', 'edge_index', 'num_nodes', 'train_mask', 'val_mask', 'test_mask'}:
                return Field(node.attr)
            if isinstance(owner, Record) and node.attr in owner.fields:
                return owner.fields[node.attr]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            owner, key = value(node.value, env), node.slice.value
            if owner is dataset and type(key) is int and key == 0:
                return data
            if isinstance(owner, Field) and owner.indices and type(key) is int and key == 0:
                return owner
            if type(owner) is dict and type(key) is str and key in owner:
                return owner[key]
        if isinstance(node, ast.Call):
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                raise AuthorLoaderUnsupported('Original loading calls cannot expand arguments')
            values = [value(arg, env) for arg in node.args]
            keywords = {kw.arg: value(kw.value, env) for kw in node.keywords}
            if len(keywords) != len(node.keywords):
                raise AuthorLoaderUnsupported('Original loading call duplicates a keyword')
            name = qualified(node.func, env)
            if name == 'torch_geometric.transforms.NormalizeFeatures' and not values and not keywords:
                return ('NormalizeFeatures',)
            if name == 'torch_geometric.datasets.Planetoid':
                supplied = dict(zip(('root', 'name'), values))
                if (len(values) > 2 or supplied.keys() & keywords.keys()
                        or set(keywords) - {'root', 'name', 'transform'} or configuration):
                    raise AuthorLoaderUnsupported('Original Planetoid loading needs one fixed public constructor')
                supplied.update(keywords)
                root, name = supplied.get('root'), supplied.get('name')
                transform = supplied.get('transform')
                if (type(root) is not str or type(name) is not str or name.casefold() not in {'cora', 'citeseer', 'pubmed'}
                        or transform not in (None, ('NormalizeFeatures',))):
                    raise AuthorLoaderUnsupported('Original Planetoid identity or transform is unsupported')
                resolved = (Path(root) if Path(root).is_absolute() else Path(workspace) / train['cwd'] / root).resolve()
                if not resolved.is_relative_to(Path(workspace).resolve()) or resolved == Path(workspace).resolve():
                    raise AuthorLoaderUnsupported('Original dataset root must stay within the author workspace')
                relative = resolved.relative_to(Path(workspace).resolve()).as_posix()
                workspace_path(workspace, relative, 'author dataset root', forbid_git=True)
                configuration.update(framework='torch_geometric.datasets.Planetoid', name=name,
                    root=relative, split='public', transform='NormalizeFeatures' if transform else None)
                return dataset
            if (name == 'torch.where' and len(values) == 1 and not keywords and isinstance(values[0], Field)
                    and values[0].name in {'train_mask', 'val_mask', 'test_mask'}):
                return Field(values[0].name, indices=True)
            if name in {'os.path.join', 'os.path.dirname', 'os.path.realpath'} and not keywords and all(type(v) is str for v in values):
                if name == 'os.path.join' and values:
                    return str(Path(values[0]).joinpath(*values[1:]))
                if len(values) == 1:
                    original = Path(values[0])
                    return str(original.parent) if name.endswith('dirname') else str(original.resolve())
            if isinstance(node.func, ast.Name) and node.func.id not in env:
                name = node.func.id
                if name in functions:
                    return run(functions[name], bind(functions[name], values, keywords))
                if name in classes:
                    cls = classes[name]
                    used.add((type(cls).__name__, cls.lineno))
                    result = Record(cls.name)
                    init = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == '__init__')
                    bound = bind(init, [result, *values], keywords)
                    body(init.body, {**bound, '__file__': str(source_path)}, initializer=result)
                    return result
            raise AuthorLoaderUnsupported('Original wrapper calls an unreviewed operation')
        raise AuthorLoaderUnsupported('Original wrapper contains a dynamic expression')

    def body(nodes, env, initializer=None):
        for node in nodes:
            if isinstance(node, ast.If) and initializer is None:
                condition = value(node.test, env)
                if type(condition) is not bool:
                    raise AuthorLoaderUnsupported('Original wrapper dispatch needs a fixed boolean branch')
                returned, result = body(node.body if condition else node.orelse, env)
                if returned:
                    return True, result
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                result, target = value(node.value, env), node.targets[0]
                if isinstance(target, ast.Name) and initializer is None:
                    env[target.id] = result
                elif isinstance(target, ast.Attribute):
                    owner = value(target.value, env)
                    if isinstance(owner, Record) and (initializer is None or owner is initializer) and not target.attr.startswith('_'):
                        if initializer is not None and not (literal(result) or type(result) is dict and not result):
                            raise AuthorLoaderUnsupported('Original record initializer can only store primitive arguments or empty dictionaries')
                        owner.fields[target.attr] = result
                    elif owner is dataset and target.attr == 'transform' and result == ('NormalizeFeatures',) and initializer is None:
                        configuration['transform'] = 'NormalizeFeatures'
                    else:
                        raise AuthorLoaderUnsupported('Original wrapper mutates an unreviewed object')
                else:
                    raise AuthorLoaderUnsupported('Original wrapper assignment is unsupported')
            elif isinstance(node, ast.Return) and initializer is None:
                return True, value(node.value, env)
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    continue
                if (initializer is None and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == 'print' and 'print' not in all_bindings and 'print' not in env
                        and not node.value.keywords):
                    # Printing primitive metadata is inert. A node-count field
                    # in a plain f-string remains a reviewed integer at runtime.
                    for arg in node.value.args:
                        pieces = [item.value for item in arg.values if isinstance(item, ast.FormattedValue)] if isinstance(arg, ast.JoinedStr) else [arg]
                        if isinstance(arg, ast.JoinedStr) and any(isinstance(item, ast.FormattedValue)
                                and (item.conversion != -1 or item.format_spec is not None) for item in arg.values):
                            raise AuthorLoaderUnsupported('Original wrapper print cannot execute custom formatting')
                        for piece in pieces:
                            item = value(piece, env)
                            if not literal(item) and not (isinstance(item, Field) and item.name == 'num_nodes'):
                                raise AuthorLoaderUnsupported('Original wrapper can print only primitive metadata')
                    continue
                raise AuthorLoaderUnsupported('Original wrapper contains an unreviewed statement')
            else:
                raise AuthorLoaderUnsupported('Original wrapper contains unreviewed control flow')
        return False, None

    def run(fn, env):
        if fn.name in active or len(active) >= 8 or len(used) >= 16:
            raise AuthorLoaderUnsupported('Original wrapper loading exceeds its bounded acyclic call graph')
        used.add((type(fn).__name__, fn.lineno))
        active.append(fn.name)
        try:
            returned, result = body(fn.body, {**env, '__file__': str(source_path)})
        finally:
            active.pop()
        if not returned:
            raise AuthorLoaderUnsupported('Original loading helper must explicitly return its data')
        return result

    result = run(selected, arguments)
    if not configuration or not (result is dataset or isinstance(result, Record)):
        raise AuthorLoaderUnsupported('Original wrapper must return the one reviewed dataset or a plain record')

    def contract(item):
        if literal(item):
            return {'literal': item}
        if isinstance(item, Field):
            return {'mask_indices' if item.indices else 'data_field': item.name}
        if type(item) is dict:
            return {'dictionary': {key: contract(value) for key, value in item.items()}}
        raise AuthorLoaderUnsupported('Original wrapper result contains an unreviewed stored value')

    record = None
    if isinstance(result, Record):
        record = {'class': result.cls, 'fields': {key: contract(item) for key, item in result.fields.items()}}
        def retained_fields(item):
            if isinstance(item, Field):
                return {item.name}
            if type(item) is dict:
                return set().union(*(retained_fields(child) for child in item.values())) if item else set()
            return set()
        if not {'x', 'y', 'edge_index'} <= retained_fields(result.fields):
            raise AuthorLoaderUnsupported('Original wrapper must preserve full features, labels and edges')
    statements = []
    for node in tree.body:
        retain = (type(node).__name__, node.lineno) in used
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                     for alias in node.names}
            retain = bool(names & imported)
            if retain and any(name not in aliases for name in names):
                raise AuthorLoaderUnsupported('A required import mixes reviewed and unreviewed symbols')
        if retain:
            segment = ast.get_source_segment(source, node)
            statements.append({'kind': type(node).__name__, 'line': node.lineno,
                'sha256': hashlib.sha256(segment.encode()).hexdigest()})
    configuration['source_slice'] = {'version': 1, 'statements': statements,
        'cwd': train['cwd'], 'returned_record': record}
    return configuration


def _compile_author_loader(declaration, files, workspace, train, effective, *, wrapper=False):
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
    if (hashlib.sha256(source_path.read_bytes()).hexdigest() != original['sha256']
            or source_path.read_text('utf-8') != source):
        raise AuthorLoaderUnsupported('Author loader source changed after discovery')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                         *(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)]
            annotations = [argument.annotation for argument in arguments if argument.annotation is not None]
            if node.returns is not None:
                annotations.append(node.returns)
            if node.decorator_list or annotations or getattr(node, 'type_params', []):
                raise AuthorLoaderUnsupported('Loader module function headers must not execute decorators or annotations')
            for default in [*node.args.defaults, *(item for item in node.args.kw_defaults if item is not None)]:
                try:
                    value = ast.literal_eval(default)
                except (ValueError, TypeError) as exc:
                    raise AuthorLoaderUnsupported('Loader module function defaults must be primitive literals') from exc
                if type(value) not in ((str, int, float, bool, type(None)) if wrapper else (str, int, bool, type(None))):
                    raise AuthorLoaderUnsupported('Loader module function defaults must be primitive literals')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(functions) != 1 or functions[0].decorator_list:
        raise AuthorLoaderUnsupported('Author loader must be a unique undecorated top-level function')
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {'os.path', 'torch_geometric.transforms'} and not (wrapper and alias.name == 'torch'):
                    if wrapper:
                        continue
                    raise AuthorLoaderUnsupported('Loader module has an unreviewed top-level import')
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            if wrapper and (node.level or node.module != 'torch_geometric.datasets'):
                continue
            if node.level or node.module != 'torch_geometric.datasets':
                raise AuthorLoaderUnsupported('Loader module has an unreviewed top-level import')
            for alias in node.names:
                if alias.name not in {'Planetoid', 'Coauthor', 'Amazon'}:
                    if wrapper:
                        continue
                    raise AuthorLoaderUnsupported('Loader module imports an unreviewed dataset class')
                aliases[alias.asname or alias.name] = node.module + '.' + alias.name
        elif wrapper and isinstance(node, ast.ClassDef):
            _validate_record_class(node)
        elif not isinstance(node, ast.FunctionDef) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise AuthorLoaderUnsupported('Loader module must contain only reviewed imports and function definitions')

    argv = train['argv']
    position = 2 if argv[1] == '-u' else 1
    entry = str(PurePosixPath(train['cwd']) / argv[position])
    entry_source = files[entry]['text']
    entry_path = workspace_path(workspace, entry, 'author loader entrypoint', must_exist=True, forbid_git=True)
    if (hashlib.sha256(entry_path.read_bytes()).hexdigest() != files[entry]['sha256']
            or entry_path.read_text('utf-8') != entry_source):
        raise AuthorLoaderUnsupported('Author loader entrypoint source changed after discovery')
    entry_tree = ast.parse(entry_source)
    call_names = set()
    loader_imports = {}
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
                loader_imports.setdefault(id(node), set()).add(alias.asname or function)
            elif alias.name == '*':
                call_names.add(function)
                loader_imports.setdefault(id(node), set()).add(function)
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
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(primitive(item, env) for item in node.elts)
        if isinstance(node, ast.JoinedStr):
            parts = []
            for item in node.values:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    parts.append(item.value)
                elif isinstance(item, ast.FormattedValue) and item.conversion == -1 and item.format_spec is None:
                    value = primitive(item.value, env)
                    if type(value) not in (str, int, bool):
                        raise AuthorLoaderUnsupported('Loader formatted path needs primitive literal components')
                    parts.append(str(value))
                else:
                    raise AuthorLoaderUnsupported('Loader formatted path cannot execute a formatter')
            return ''.join(parts)
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
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
        raise AuthorLoaderUnsupported('Loader branch or argument is not fixed by original literals/CLI')

    selected = []
    print_shadowed = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == 'print'
        or isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == 'print'
        or isinstance(node, ast.alias) and (node.asname or node.name.split('.')[0]) == 'print'
        for node in ast.walk(entry_tree))
    entry_helpers = {item.name: item for item in entry_tree.body if isinstance(item, ast.FunctionDef)}

    def reject_dynamic_bindings(node, visiting=None):
        # Review only explicit namespace operations and referenced local helpers.
        # Ordinary author setup calls (for example RNG seeding) remain supported.
        visiting = set() if visiting is None else visiting
        globals_written = set().union(*(set(item.names) for item in ast.walk(node)
                                       if isinstance(item, ast.Global)))
        if globals_written & call_names and any(
                isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
                and item.id in globals_written & call_names for item in ast.walk(node)):
            raise AuthorLoaderUnsupported('Original helper can replace the global loader binding')
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in {'exec', 'eval', 'globals', 'locals', 'vars'}:
                raise AuthorLoaderUnsupported('Original pre-loading code cannot access dynamic namespace bindings')
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in entry_helpers and item.id not in visiting:
                helper = entry_helpers[item.id]
                reject_dynamic_bindings(helper, visiting | {item.id})

    def reject_mutations(node):
        reject_dynamic_bindings(node)
        parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in cli_names:
                parent = parents.get(child)
                fixed_field = isinstance(parent, ast.Attribute) and parent.value is child and not parent.attr.startswith('_')
                inert_print = (isinstance(parent, ast.Call) and child in parent.args
                               and isinstance(parent.func, ast.Name) and parent.func.id == 'print'
                               and not print_shadowed and not parent.keywords)
                if not (fixed_field or inert_print):
                    raise AuthorLoaderUnsupported('Original loader CLI namespace cannot escape into dynamic storage')
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
                if not (isinstance(child.func, ast.Name) and child.func.id == 'print' and not print_shadowed
                        and not child.keywords):
                    raise AuthorLoaderUnsupported('An opaque call can mutate the original loader CLI namespace')
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Name) and child.value.id in cli_names:
                raise AuthorLoaderUnsupported('Original loader CLI namespace cannot escape through an alias')

    def wildcard_bindings(node):
        # Resolve only already-discovered local modules; never import or evaluate them.
        if not isinstance(node, ast.ImportFrom) or node.level:
            raise AuthorLoaderUnsupported('Original wildcard import needs an already-read local module')
        module = PurePosixPath(*(node.module or '').split('.'))
        candidates = [str(PurePosixPath(entry).parent / module) + '.py', str(module) + '.py']
        matched = next((name for name in candidates if name in files), None)
        if matched is None:
            raise AuthorLoaderUnsupported('Original wildcard import has unknown source bindings')
        wildcard_path = workspace_path(workspace, matched, 'wildcard import source', must_exist=True, forbid_git=True)
        wildcard_bytes = wildcard_path.read_bytes()
        if (hashlib.sha256(wildcard_bytes).hexdigest() != files[matched]['sha256']
                or wildcard_path.read_text('utf-8') != files[matched]['text']):
            raise AuthorLoaderUnsupported('Original wildcard import source changed after discovery')
        exports, explicit = set(), None
        for statement in ast.parse(files[matched]['text']).body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == '*' for alias in statement.names):
                    raise AuthorLoaderUnsupported('Original wildcard module cannot re-export unknown bindings')
                exports.update(alias.asname or (alias.name.split('.')[0] if isinstance(statement, ast.Import) else alias.name)
                               for alias in statement.names)
            elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == '__all__'):
                try:
                    explicit = ast.literal_eval(statement.value)
                except (TypeError, ValueError) as exc:
                    raise AuthorLoaderUnsupported('Original wildcard exports must be fixed literal names') from exc
                if type(explicit) not in (list, tuple) or not all(type(name) is str for name in explicit):
                    raise AuthorLoaderUnsupported('Original wildcard exports must be fixed literal names')
            elif not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                      and type(statement.value.value) is str):
                raise AuthorLoaderUnsupported('Original wildcard module has dynamic top-level bindings')
        if explicit is not None and not set(explicit) <= exports:
            raise AuthorLoaderUnsupported('Original wildcard names need direct static bindings')
        return set(explicit) if explicit is not None else {name for name in exports if not name.startswith('_')}

    def inspect_entry(nodes, env):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                call_names.discard(node.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(alias.name == '*' for alias in node.names) and id(node) not in loader_imports:
                    if not selected and wildcard_bindings(node) & call_names:
                        raise AuthorLoaderUnsupported('Original loader binding can be replaced by a wildcard import')
                names_bound = {alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                               for alias in node.names}
                call_names.difference_update(names_bound - loader_imports.get(id(node), set()))
                call_names.update(loader_imports.get(id(node), set()))
                continue
            if isinstance(node, ast.If):
                try:
                    condition = primitive(node.test, env)
                except AuthorLoaderUnsupported:
                    if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id in call_names for call in ast.walk(node)):
                        raise
                    if not selected:
                        reject_mutations(node)
                        for possible in ast.walk(node):
                            if isinstance(possible, ast.Name) and isinstance(possible.ctx, (ast.Store, ast.Del)) and possible.id in call_names:
                                raise AuthorLoaderUnsupported('Unknown original branch can replace the loader binding')
                            if isinstance(possible, (ast.Import, ast.ImportFrom)):
                                possible_names = {alias.asname or (alias.name.split('.')[0] if isinstance(possible, ast.Import) else alias.name)
                                                  for alias in possible.names}
                                if '*' in possible_names or possible_names & call_names:
                                    raise AuthorLoaderUnsupported('Unknown original branch can import another loader binding')
                    continue
                inspect_entry(node.body if condition else node.orelse, env)
                continue
            calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                     and isinstance(call.func, ast.Name) and call.func.id in call_names]
            if selected:
                selected.extend((call, dict(env)) for call in calls)
                continue
            reject_mutations(node)
            if calls and not (isinstance(node, ast.Assign) and len(node.targets) == 1
                              and isinstance(node.targets[0], ast.Name) and node.value in calls):
                raise AuthorLoaderUnsupported('Original loader invocation must be a direct selected assignment')
            selected.extend((call, dict(env)) for call in calls)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
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
    if wrapper:
        configuration = _review_wrapper(source, tree, aliases, source_path, workspace, train, arguments, fn)
        return {'kind': 'author_loader', 'path': path, 'function': function, 'arguments': arguments, 'cwd': train['cwd'],
                **configuration, 'source_sha256': original['sha256'],
                'call': {'path': entry, 'quote': ast.get_source_segment(entry_source, call),
                         'sha256': files[entry]['sha256']}, 'definition': ast.get_source_segment(source, fn)}
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
                resolved = (Path(root) if Path(root).is_absolute() else Path(workspace) / train['cwd'] / root).resolve()
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
    return {'kind': 'author_loader', 'path': path, 'function': function, 'arguments': arguments, 'cwd': train['cwd'],
            **configuration, 'source_sha256': original['sha256'],
            'call': {'path': entry, 'quote': ast.get_source_segment(entry_source, call),
                     'sha256': files[entry]['sha256']}, 'definition': ast.get_source_segment(source, fn)}
