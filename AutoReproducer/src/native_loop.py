"""Resolve an original repetition loop and validation selection; never eval text."""
import ast
import re

_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,127}')
_COMPARISONS = {'lt': ast.Lt, 'le': ast.LtE, 'gt': ast.Gt, 'ge': ast.GtE}


def owned_nodes(node):
    """Visit only instructions belonging to the selected original code scope."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from owned_nodes(child)


def selector_expression(selector):
    if isinstance(selector, str) and _NAME.fullmatch(selector):
        return ast.Name(id=selector, ctx=ast.Load())
    if (not isinstance(selector, dict) or set(selector) != {'name', 'keys'}
            or not isinstance(selector['name'], str) or not _NAME.fullmatch(selector['name'])
            or not isinstance(selector['keys'], list) or not 1 <= len(selector['keys']) <= 8):
        raise ValueError('Validation metric must be an original identifier or fixed key selector')
    expression = ast.Name(id=selector['name'], ctx=ast.Load())
    for key in selector['keys']:
        if not (type(key) is str and 1 <= len(key) <= 128 or type(key) is int and 0 <= key < 10000000):
            raise ValueError('Validation metric key must be a fixed literal, without calls or slicing')
        expression = ast.Subscript(value=expression, slice=ast.Constant(value=key), ctx=ast.Load())
    return expression


def resolve_module_loop(source, scope):
    if (not isinstance(scope, dict)
            or set(scope) not in ({'path', 'loop_counter', 'selection'},
                                 {'path', 'function', 'loop_counter', 'selection'})):
        raise ValueError('Native loop scope needs a path, loop counter and validation selection')
    selection = scope['selection']
    fields = {'validation_metric', 'comparison', 'best_metric', 'result', 'result_unit'}
    if (not isinstance(selection, dict) or set(selection) != fields
            or selection.get('comparison') not in _COMPARISONS
            or selection.get('result_unit') not in {'fraction', 'percent'}):
        raise ValueError('Native selection needs a fixed validation comparison and reported result')
    counter = scope['loop_counter']
    if not isinstance(counter, str) or not (counter == '_' or _NAME.fullmatch(counter)):
        raise ValueError('Native loop counter must be an original identifier')
    names = [selection[key] for key in ('best_metric', 'result')]
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in names):
        raise ValueError('Native loop and selection values must be original identifiers')
    validation = selector_expression(selection['validation_metric'])
    tree = ast.parse(source)
    container, function_line = tree, None
    if 'function' in scope:
        name = scope['function']
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError('Native loop function must be an original identifier')
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(functions) != 1 or functions[0] not in tree.body or isinstance(functions[0], ast.AsyncFunctionDef):
            raise ValueError('Native loop must belong to one unique synchronous top-level function')
        container = functions[0]
        function_line = min([container.lineno] + [node.lineno for node in container.decorator_list])
    loops = [node for node in owned_nodes(container) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == scope['loop_counter']]
    if len(loops) != 1 or loops[0] not in container.body:
        raise ValueError('Native loop must resolve to one unique direct author for loop in its scope')
    loop = loops[0]
    guards = [node for node in owned_nodes(loop) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
              and isinstance(node.test.ops[0], _COMPARISONS[selection['comparison']])
              and ast.dump(node.test.left) == ast.dump(validation)
              and isinstance(node.test.comparators[0], ast.Name)
              and node.test.comparators[0].id == selection['best_metric']]
    if len(guards) != 1:
        raise ValueError('Validation selection must resolve to one original source comparison')
    assignments = [node for node in guards[0].body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == selection['result']]
    if len(assignments) != 1:
        raise ValueError('Selected test result must be assigned directly in the original validation branch')
    assignment = assignments[0]
    return {'loop_line': loop.lineno, 'loop_end': loop.end_lineno,
            'selection_start': assignment.lineno, 'selection_end': assignment.end_lineno,
            'scope_name': scope.get('function', '<module>'), 'function_line': function_line}


def discover_native_selections(files):
    """Expose source-resolved capture options, never an experiment acceptance."""
    result = []
    def selector(node):
        keys = []
        while isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            keys.insert(0, node.slice.value)
            node = node.value
        if not isinstance(node, ast.Name):
            raise ValueError('Not an original fixed validation selector')
        value = {'name': node.id, 'keys': keys} if keys else node.id
        selector_expression(value)
        return value
    for path, source in sorted(files.items()):
        if not path.endswith('.py'):
            continue
        try:
            tree = ast.parse(source['text'])
        except SyntaxError:
            continue
        containers = [(tree, {})] + [(node, {'function': node.name}) for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.decorator_list]
        for container, extra in containers:
            for loop in container.body:
                if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
                    continue
                for guard in owned_nodes(loop):
                    if (not isinstance(guard, ast.If) or not isinstance(guard.test, ast.Compare)
                            or len(guard.test.ops) != 1 or not isinstance(guard.test.comparators[0], ast.Name)):
                        continue
                    comparison = next((name for name, kind in _COMPARISONS.items()
                                       if isinstance(guard.test.ops[0], kind)), None)
                    if comparison is None:
                        continue
                    try:
                        validation = selector(guard.test.left)
                    except ValueError:
                        continue
                    best = guard.test.comparators[0].id
                    for assignment in guard.body:
                        if (not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1
                                or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id == best):
                            continue
                        scope = {'path': path, **extra, 'loop_counter': loop.target.id, 'selection': {
                            'validation_metric': validation, 'comparison': comparison, 'best_metric': best,
                            'result': assignment.targets[0].id, 'result_unit': 'fraction'}}
                        try:
                            resolved = resolve_module_loop(source['text'], scope)
                        except ValueError:
                            continue
                        scope['selection'].pop('result_unit')  # Units still require paper/source interpretation.
                        result.append({'scope': scope, 'source_sha256': source['sha256'], **resolved,
                            'validation_branch': ast.get_source_segment(source['text'], guard),
                            'capture_mechanism': 'snapshot_after_original_validation_result_assignment',
                            'author_checkpoint_required': False,
                            'remaining_checks': 'Identify original model, inputs, full labels/test split, metric unit and all repetitions; then verify real optimizer updates and independent predictions at runtime.'})
    return result
