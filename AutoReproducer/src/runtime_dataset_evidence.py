"""Bind author-native dataset construction to observed complete framework files."""
import ast
from pathlib import Path

from src.method_adapters import digest
from src.safety.paths import workspace_path


PYG_CLASSES = {'Planetoid', 'WikiCS', 'Coauthor', 'Amazon', 'CitationFull', 'Reddit',
               'PPI', 'Flickr', 'Actor', 'WebKB', 'HeterophilousGraphDataset'}


def scope_nodes(scope):
    """Visit one Python call scope, including branches but excluding nested defs."""
    yield scope
    for child in ast.iter_child_nodes(scope):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from scope_nodes(child)


def constructor_calls(tree, scope, class_name):
    """Resolve original imports and literal dataset factory tables, without execution.

    A table may contain several dataset classes. This establishes a possible
    constructor only; the runtime observer must still prove the selected class,
    name, calling scope and actual files before the run can pass.
    """
    symbols = set()
    nodes = list(scope_nodes(scope))
    for node in [*tree.body, *nodes]:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if node.module == 'torch_geometric.datasets' and alias.name == class_name:
                    symbols.add(alias.asname or alias.name)
                elif node.module == 'torch_geometric' and alias.name == 'datasets':
                    symbols.add((alias.asname or alias.name) + '.' + class_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'torch_geometric.datasets':
                    symbols.add((alias.asname or alias.name) + '.' + class_name)
                elif alias.name == 'torch_geometric':
                    symbols.add((alias.asname or alias.name) + '.datasets.' + class_name)

    def matches(node):
        return (ast.unparse(node) in symbols or isinstance(node, ast.IfExp)
                and (matches(node.body) or matches(node.orelse)))

    tables = {}
    for node in sorted(nodes, key=lambda part: getattr(part, 'lineno', 0)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                if matches(node.value):
                    symbols.add(target.id)
                elif isinstance(node.value, ast.Dict):
                    tables[target.id] = node.value.values
            elif (isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Name) and node.value.value.id in tables):
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and any(
                            isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts)
                            and matches(value.elts[index]) for value in tables[node.value.value.id]):
                        symbols.add(element.id)
    return [node for node in nodes if isinstance(node, ast.Call) and matches(node.func)]


def discover_runtime_constructors(files):
    """Supply exact original constructor scopes instead of asking a model to guess."""
    options = []
    for path, source in sorted(files.items()):
        if not path.endswith('.py'):
            continue
        try:
            tree = ast.parse(source['text'])
        except SyntaxError:
            continue
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        scopes = [tree, *[node for node in functions if sum(item.name == node.name for item in functions) == 1]]
        for scope in scopes:
            for class_name in sorted(PYG_CLASSES):
                for call in constructor_calls(tree, scope, class_name):
                    options.append({'path': path, 'function': getattr(scope, 'name', '<module>'),
                                    'dataset_class': class_name,
                                    'constructor_quote': ast.get_source_segment(source['text'], call),
                                    'source_sha256': source['sha256']})
    return options


def compile_runtime_dataset(item, packet, workspace, citations):
    from src.discovered_repository_plan import PlanEvidenceError, _references, _has_value
    if packet.get('author_execution_only') is not True:
        raise PlanEvidenceError('Native data observation is only available for explicit author execution')
    if set(item) != {'kind', 'path', 'function', 'dataset_class', 'dataset_name', 'citations'}:
        raise PlanEvidenceError('Native dataset declaration names only original source, class and data identity')
    name, function = item['path'], item['function']
    source = packet['repository']['files'].get(name)
    if not source or not name.endswith('.py'):
        raise PlanEvidenceError('Native dataset construction needs an already-read original Python file')
    path = workspace_path(workspace, name, 'native dataset source', must_exist=True, forbid_git=True)
    if digest(path) != source['sha256']:
        raise PlanEvidenceError('Native dataset source changed after discovery')
    tree = ast.parse(source['text'])
    scopes = [tree] if function == '<module>' else [node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(scopes) != 1:
        raise PlanEvidenceError('Native dataset scope must identify one original function or module')
    class_name = item['dataset_class']
    if class_name not in PYG_CLASSES:
        raise PlanEvidenceError('Native dataset class is outside the observed PyG framework classes')
    calls = constructor_calls(tree, scopes[0], class_name)
    if not calls:
        alternatives = [item['function'] for item in discover_runtime_constructors({name: source})
                        if item['dataset_class'] == class_name]
        raise PlanEvidenceError('Declared native dataset class has no original constructor in its selected scope: '
                                + name + ':' + function + '; actual constructor scopes: '
                                + ', '.join(sorted(set(alternatives)))
                                + '. Select the function making the constructor call, not its caller.')
    refs = _references(item['citations'], citations, 'repository')
    dataset_name = item['dataset_name']
    if not isinstance(dataset_name, str) or not dataset_name.strip() or not _has_value(dataset_name, refs):
        raise PlanEvidenceError('Native data identity needs original repository evidence')
    def covered(quote):
        import re
        quote = re.sub(r'\s+', ' ', quote).strip()
        return any(ref.get('path') == name and quote in re.sub(r'\s+', ' ', ref['quote']).strip() for ref in refs)
    quotes = [ast.get_source_segment(source['text'], call) for call in calls]
    if not any(covered(quote) for quote in quotes):
        raise PlanEvidenceError('Native dataset construction needs its original constructor citation')
    return {'kind': 'author_runtime', 'path': name, 'function': function,
            'dataset_class': class_name, 'dataset_name': dataset_name,
            'source_sha256': source['sha256'], 'citations': list(item['citations']),
            'framework': 'torch_geometric.datasets', 'constructor_quotes': quotes}


def verify_runtime_datasets(plan, receipts, workspace):
    declarations = [item for item in plan['datasets'] if item['kind'] == 'author_runtime']
    records = []
    for declaration in declarations:
        source = workspace_path(workspace, declaration['path'], 'native dataset source', must_exist=True, forbid_git=True)
        if digest(source) != declaration['source_sha256']:
            raise ValueError('Original native data source changed during execution')
        observed = [record for receipt in receipts for record in receipt.get('native_datasets', [])
                    if record.get('declaration') == declaration]
        if not observed:
            raise ValueError('No original native dataset constructor was observed for the frozen declaration')
        # Authors may reconstruct the same dataset for each split/run. Only
        # its first use must precede training; every reconstruction still needs
        # complete, unchanged real files and the reviewed dataset identity.
        if not any(record.get('loaded_before_training') is True for record in observed):
            raise ValueError('Native dataset lacks pre-training construction evidence')
        for record in observed:
            if (record.get('verified') is not True
                    or record.get('dataset_name', '').casefold() != declaration['dataset_name'].casefold()
                    or type(record.get('sample_count')) is not int or record['sample_count'] <= 0
                    or not record.get('files') or not any(value.get('role') == 'raw' for value in record['files'].values())):
                raise ValueError('Native dataset lacks complete real-file and pre-training construction evidence')
            root = Path(record['root']).resolve(strict=True)
            for name, value in record['files'].items():
                path = Path(name).resolve(strict=True)
                if (not path.is_relative_to(root) or not path.is_file()
                        or type(value.get('bytes')) is not int or value['bytes'] <= 0
                        or path.stat().st_size != value['bytes'] or digest(path) != value.get('sha256')):
                    raise ValueError('Native real dataset files changed or escaped their original root')
        records.append({'kind': 'author_runtime', 'loader': declaration, 'verified': True,
                        'observations': observed, 'source': 'original_author_constructor_and_fixed_framework'})
    return records
