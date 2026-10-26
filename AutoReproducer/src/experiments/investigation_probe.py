"""Fixed isolated helper; only stdlib metadata and restricted original assertions."""
import ast
import hashlib
import importlib.metadata
import json
import math
import operator
from pathlib import Path
import sys
import sysconfig


def runtime_metadata(request):
    executable = Path(sys.executable)
    locations = [executable.parent.parent / 'Lib' / 'site-packages',
                 executable.parent.parent / 'lib' / ('python' + '.'.join(map(str, sys.version_info[:2]))) / 'site-packages']
    # -S can expose base_prefix in older virtualenvs. Do not silently inspect
    # application/base distributions instead of the selected virtualenv.
    if not (executable.parent.parent / 'pyvenv.cfg').is_file():
        locations.append(Path(sysconfig.get_path('purelib')))
    configured = request.get('metadata_paths') or []
    if not isinstance(configured, list) or any(not isinstance(path, str) or not Path(path).is_absolute() for path in configured):
        raise ValueError('Invalid recorded author dependency roots')
    # RepositoryRunner activates these reviewed package directories through
    # PYTHONPATH. Metadata must describe them, not merely interpreter site-packages.
    locations = list(dict.fromkeys(str(path) for path in [*(Path(path) for path in configured), *locations] if path.is_dir()))
    requested = {name.lower().replace('_', '-').replace('.', '-') for name in request['packages']}
    packages = {}
    for distribution in importlib.metadata.distributions(path=locations):
        name = distribution.metadata.get('Name', '')
        normalized = name.lower().replace('_', '-').replace('.', '-')
        if normalized in requested and normalized not in packages:
            packages[normalized] = {'name': name, 'version': distribution.version}
    modules = {}
    for module in request['modules']:
        relative = module.replace('.', '/')
        candidates = []
        for location in locations:
            root = Path(location).resolve()
            for name in (relative + '.py', relative + '/__init__.py'):
                path = root / name
                if path.is_file() and path.resolve().is_relative_to(root) and not path.is_symlink():
                    if path.stat().st_size > 2 * 1024 * 1024:
                        continue
                    content = path.read_bytes()
                    try:
                        tree = ast.parse(content)
                        definitions = [{'name': node.name, 'arguments': ast.unparse(node.args)}
                                       for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][:80]
                    except (ValueError, SyntaxError, UnicodeError):
                        definitions = []
                    candidates.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                                       'definitions': definitions})
        modules[module] = {'candidates': candidates, 'import_executed': False,
                           'native_signature': 'unknown'}
    return {'python_executable': sys.executable, 'python_version': sys.version.split()[0],
            'packages': {name: packages.get(name) for name in sorted(requested)}, 'modules': modules,
            'metadata_roots': locations}


class ContractLimit(ValueError):
    pass


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class PureContract:
    """Small cost-bounded interpreter, never exec/eval of an author expression."""
    def __init__(self, functions, builtins):
        self.functions, self.builtins = functions, builtins
        self.steps, self.elements, self.assertions = 0, 0, 0

    def checked(self, value, depth=0):
        if depth > 32:
            raise ContractLimit('Nested value limit')
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ContractLimit('Integer size limit')
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractLimit('Nonfinite value')
        if isinstance(value, (str, bytes, list, tuple, dict, set)):
            if len(value) > 4096:
                raise ContractLimit('Container size limit')
            self.elements += len(value)
            if self.elements > 32768:
                raise ContractLimit('Total value budget')
            if isinstance(value, dict):
                for key, item in value.items():
                    self.checked(key, depth + 1)
                    self.checked(item, depth + 1)
            elif not isinstance(value, (str, bytes)):
                for item in value:
                    self.checked(item, depth + 1)
        return value

    def tick(self, depth):
        self.steps += 1
        if self.steps > 2000 or depth > 32:
            raise ContractLimit('Operation/depth limit')

    def expression(self, node, local, depth=0):
        self.tick(depth)
        child = lambda item: self.expression(item, local, depth + 1)
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            value = local[node.id]
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if len(node.elts) > 4096:
                raise ContractLimit('Literal container size')
            values = [child(item) for item in node.elts]
            value = tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
        elif isinstance(node, ast.Dict):
            value = {child(key): child(item) for key, item in zip(node.keys, node.values)}
        elif isinstance(node, ast.BinOp):
            left, right = child(node.left), child(node.right)
            if isinstance(node.op, ast.Mod) and isinstance(left, (str, bytes)):
                raise ContractLimit('Unbounded formatting is unsupported')
            if isinstance(node.op, ast.Pow):
                if (not isinstance(right, int) or not 0 <= right <= 64
                        or isinstance(left, int) and max(1, left.bit_length()) * right > 4096):
                    raise ContractLimit('Exponent size limit')
            if isinstance(node.op, ast.Mult):
                sequence, count = (left, right) if isinstance(right, int) else (right, left)
                if isinstance(sequence, (str, bytes, list, tuple)) and isinstance(count, int) and len(sequence) * max(count, 0) > 4096:
                    raise ContractLimit('Sequence allocation limit')
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
            value = operations[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operations = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
            value = operations[type(node.op)](child(node.operand))
        elif isinstance(node, ast.BoolOp):
            value = child(node.values[0])
            for item in node.values[1:]:
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    break
                value = child(item)
        elif isinstance(node, ast.Compare):
            left, value = child(node.left), True
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            for op, item in zip(node.ops, node.comparators):
                right = child(item)
                if not operations[type(op)](left, right):
                    value = False
                    break
                left = right
        elif isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Slice):
                index = slice(child(index.lower) if index.lower else None,
                              child(index.upper) if index.upper else None, child(index.step) if index.step else None)
            else:
                index = child(index)
            value = child(node.value)[index]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [child(item) for item in node.args]
            kwargs = {item.arg: child(item.value) for item in node.keywords if item.arg is not None}
            if len(kwargs) != len(node.keywords):
                raise ContractLimit('Expanded keyword arguments are unsupported')
            if node.func.id in local:
                raise ContractLimit('Locally rebound call targets are unsupported')
            if node.func.id in self.functions:
                value = self.call(node.func.id, args, kwargs, depth + 1)
            elif node.func.id in self.builtins:
                if node.func.id == 'round':
                    digits = args[1] if len(args) > 1 else kwargs.get('ndigits', 0)
                    if not isinstance(digits, int) or abs(digits) > 64:
                        raise ContractLimit('Rounding precision limit')
                value = self.builtins[node.func.id](*args, **kwargs)
            else:
                value = self.call(node.func.id, args, kwargs, depth + 1)
        else:
            raise ContractLimit('Unsupported pure expression')
        return self.checked(value)

    def statements(self, body, local, depth):
        for node in body:
            self.tick(depth)
            if isinstance(node, ast.Return):
                raise Returned(self.expression(node.value, local, depth + 1) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expression(node.value, local, depth + 1)
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        raise ContractLimit('Only local name assignment is supported')
                    local[target.id] = value
            elif isinstance(node, ast.Assert):
                self.assertions += 1
                if not self.expression(node.test, local, depth + 1):
                    raise AssertionError()
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expression(node.test, local, depth + 1) else node.orelse, local, depth + 1)
            elif isinstance(node, ast.Expr):
                self.expression(node.value, local, depth + 1)
            elif not isinstance(node, (ast.Return, ast.Pass)):
                raise ContractLimit('Unsupported pure statement')

    def call(self, name, args, kwargs, depth=0):
        self.tick(depth)
        function = self.functions[name]
        names = [arg.arg for arg in function.args.args]
        assigned = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        called = {node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        if called & (assigned | set(names)):
            raise ContractLimit('Function-local call bindings are unsupported')
        if len(args) > len(names) or set(kwargs) - set(names):
            raise ContractLimit('Unsupported function arguments')
        local = dict(zip(names, args))
        if set(local) & set(kwargs):
            raise ContractLimit('Duplicate function arguments')
        local.update(kwargs)
        if (set(local) != set(names) or function.args.vararg or function.args.kwarg
                or function.args.kwonlyargs or function.args.posonlyargs or len(names) != len(set(names))):
            raise ContractLimit('Unsupported function signature')
        try:
            self.statements(function.body, local, depth + 1)
        except Returned as result:
            return result.value
        return None


def author_contract(request):
    if len(request['source'].encode('utf-8')) > 65536:
        return {'status': 'rejected', 'reason': 'Original pure contract exceeds the source budget.'}
    tree = ast.parse(request['source'])
    if any(not isinstance(node, ast.FunctionDef) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)) for node in tree.body):
        return {'status': 'rejected', 'reason': 'Module bindings or executable setup require a full author test runner.'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len({node.name for node in functions}) != len(functions):
        return {'status': 'rejected', 'reason': 'Repeated module function bindings are unsupported.'}
    selected = next((node for node in functions if node.name == request['function']), None)
    if selected is None or selected.args.args or not any(isinstance(node, ast.Assert) for node in ast.walk(selected)):
        return {'status': 'rejected', 'reason': 'Expected an existing argument-free assertion test.'}
    builtins = {'abs': abs, 'all': all, 'any': any, 'bool': bool, 'float': float, 'int': int,
                'len': len, 'max': max, 'min': min, 'round': round, 'sorted': sorted, 'sum': sum,
                'tuple': tuple, 'list': list}
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.Assert,
               ast.If, ast.Expr, ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
               ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword,
               ast.Subscript, ast.Slice, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
    reachable, pending = {}, [selected.name]
    catalog = {node.name: node for node in functions}
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        node = catalog[name]
        if (node.decorator_list or node.args.defaults or node.args.kw_defaults
                or node.args.posonlyargs or node.args.vararg or node.args.kwarg or node.args.kwonlyargs
                or node.returns or any(arg.annotation for arg in node.args.args)):
            return {'status': 'rejected', 'reason': 'Annotated/default/decorated functions are outside the pure contract language.'}
        for item in ast.walk(node):
            if not isinstance(item, allowed):
                return {'status': 'rejected', 'reason': 'Contract requires unsupported executable constructs.'}
            if isinstance(item, ast.Name) and item.id.startswith('__'):
                return {'status': 'rejected', 'reason': 'Special runtime names are forbidden.'}
            if isinstance(item, ast.Call):
                if not isinstance(item.func, ast.Name) or item.func.id not in {*builtins, *catalog}:
                    return {'status': 'rejected', 'reason': 'Contract calls an unreviewed function.'}
                if item.func.id in catalog:
                    pending.append(item.func.id)
        reachable[name] = node
    try:
        interpreter = PureContract(reachable, builtins)
        interpreter.call(selected.name, [], {})
        if not interpreter.assertions:
            return {'status': 'rejected', 'reason': 'No original assertion was executed.'}
    except AssertionError:
        return {'status': 'failed', 'reason': 'An original assertion failed.'}
    except Exception as exc:
        return {'status': 'rejected', 'reason': 'Original contract unavailable: ' + type(exc).__name__}
    return {'status': 'completed', 'reason': 'Original pure assertion contract passed.',
            'checked_functions': sorted(reachable), 'executed_assertions': interpreter.assertions}


if __name__ == '__main__':
    request = json.loads(sys.stdin.read(2 * 1024 * 1024 + 4096))
    if request.get('tool') == 'runtime_metadata':
        response = runtime_metadata(request)
    elif request.get('tool') == 'author_contract':
        response = author_contract(request)
    else:
        raise ValueError('Unknown fixed helper operation')
    print(json.dumps(response, ensure_ascii=False))
