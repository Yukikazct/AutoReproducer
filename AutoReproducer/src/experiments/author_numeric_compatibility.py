"""A source-bound float reciprocal repair, enabled only after a real failure.

The repository file stays unchanged. The retry compiles one recorded call as
numpy.float_power instead of numpy.power; every other AST node stays original.
"""
import ast
import builtins
import hashlib
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
import runpy
import re
import sys
import types

import numpy as np


ERROR = 'Integers to negative integer powers are not allowed.'
BUILTIN_ALIASES = {'bool', 'int', 'float', 'complex', 'object', 'str'}
RAGGED_KIND = 'numpy_ragged_summary'
RAGGED_HELPER = '__autorepro_legacy_ragged_array__'


def ragged_schema(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 10000:
        raise ValueError('Unsupported ragged result table')
    if not isinstance(values[0], (list, tuple)) or not 2 <= len(values[0]) <= 16:
        raise ValueError('Unsupported ragged result row')
    width = len(values[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in values):
        raise ValueError('Inconsistent ragged result rows')
    columns = []
    for index in range(width):
        column = [row[index] for row in values]
        first = column[0]
        if isinstance(first, np.ndarray):
            if (not first.shape or first.size > 1000000 or first.dtype.kind not in 'fiu'
                    or any(not isinstance(item, np.ndarray) or item.shape != first.shape
                           or item.dtype != first.dtype or not np.isfinite(item).all() for item in column)):
                raise ValueError('Inconsistent result vector column')
            columns.append({'kind': 'array', 'shape': list(first.shape), 'dtype': str(first.dtype)})
        elif all((type(item) in (int, float) or isinstance(item, np.generic) and item.dtype.kind in 'fiu')
                 and bool(np.isfinite(item)) for item in column):
            columns.append({'kind': 'scalar'})
        else:
            raise ValueError('Unsupported result table value')
    if not any(c['kind'] == 'array' for c in columns) or not any(c['kind'] == 'scalar' for c in columns):
        raise ValueError('A legacy result table must mix scalars and uniform vectors')
    return {'rows': len(values), 'columns': columns}


def ragged_calls(tree, variable, alias):
    def module_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                yield from module_nodes(child)
    return [node for node in module_nodes(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in ('mean', 'var')
            and isinstance(node.func.value, ast.Name) and node.func.value.id == alias
            and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == variable
            and len(node.keywords) == 1 and node.keywords[0].arg == 'axis'
            and isinstance(node.keywords[0].value, ast.Constant)
            and type(node.keywords[0].value.value) is int and node.keywords[0].value.value == 0]


def validate_ragged_repair(tree, repair, source_text):
    calls = ragged_calls(tree, repair.get('variable'), repair.get('numpy_alias'))
    actual = [{'line': node.lineno, 'column': node.col_offset, 'operation': node.func.attr,
               'original_expression': ast.get_source_segment(source_text, node)} for node in calls]
    if not calls or actual != repair.get('calls') or not isinstance(repair.get('schema'), dict):
        raise ValueError('Ragged compatibility does not match the original reductions')
    if any(isinstance(node, ast.Name) and node.id == RAGGED_HELPER for node in ast.walk(tree)):
        raise ValueError('Ragged compatibility helper conflicts with original code')
    return calls


def diagnose_ragged_summary(exc, script, workspace, sources):
    if type(exc) is not ValueError or 'inhomogeneous' not in str(exc):
        return None
    script, workspace = Path(script).resolve(), Path(workspace).resolve()
    relative = script.relative_to(workspace).as_posix()
    expected = sources.get(relative)
    if not expected or hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        return None
    tree = ast.parse(script.read_bytes())
    trace = exc.__traceback__
    while trace:
        if Path(trace.tb_frame.f_code.co_filename).resolve() == script and trace.tb_frame.f_code.co_name == '<module>':
            values = {**trace.tb_frame.f_globals, **trace.tb_frame.f_locals}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and node.lineno == trace.tb_lineno
                        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and values.get(node.func.value.id) is np and len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)
                        and node in ragged_calls(tree, node.args[0].id, node.func.value.id)):
                    try:
                        schema = ragged_schema(values.get(node.args[0].id))
                    except (ValueError, TypeError, OverflowError):
                        return None
                    calls = ragged_calls(tree, node.args[0].id, node.func.value.id)
                    return {'kind': RAGGED_KIND, 'path': relative, 'source_sha256': expected,
                        'variable': node.args[0].id, 'numpy_alias': node.func.value.id, 'schema': schema,
                        'calls': [{'line': call.lineno, 'column': call.col_offset, 'operation': call.func.attr,
                                  'original_expression': ast.get_source_segment(script.read_text('utf-8'), call)} for call in calls],
                        'error': str(exc)}
        trace = trace.tb_next
    return None


def _legacy_ragged_array(values, encoded_schema):
    if ragged_schema(values) != json.loads(encoded_schema):
        raise ValueError('Legacy result table changed shape or dtype before reduction')
    result = np.asarray(values, dtype=object)
    if result.shape != (len(values), len(values[0])):
        raise ValueError('Legacy object result table has an unexpected outer shape')
    return result


def diagnose_numpy_alias_error(exc, script, workspace, sources):
    if not isinstance(exc, AttributeError):
        return None
    match = re.search(r"module 'numpy' has no attribute '([^']+)'", str(exc))
    if not match or match[1] not in BUILTIN_ALIASES:
        return None
    alias, workspace = match[1], Path(workspace).resolve()
    trace, candidate = exc.__traceback__, None
    while trace:
        source = Path(trace.tb_frame.f_code.co_filename).resolve()
        if source.is_relative_to(workspace):
            relative = source.relative_to(workspace).as_posix()
            expected = sources.get(relative)
            if expected and source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected:
                tree = ast.parse(source.read_bytes(), filename=str(source))
                values = {**trace.tb_frame.f_globals, **trace.tb_frame.f_locals}
                matches = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                           and node.lineno == trace.tb_lineno and node.attr == alias
                           and isinstance(node.value, ast.Name) and values.get(node.value.id) is np]
                if len(matches) == 1:
                    node = matches[0]
                    candidate = {'kind': 'numpy_builtin_alias', 'alias': alias, 'path': relative,
                        'source_sha256': expected, 'line': node.lineno, 'column': node.col_offset,
                        'original_expression': ast.get_source_segment(source.read_text('utf-8'), node),
                        'replacement': 'builtins.' + alias, 'error': str(exc)}
        trace = trace.tb_next
    return candidate


def power_call(tree, line, column=None):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and node.lineno == line and (column is None or node.col_offset == column)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'power'
               and isinstance(node.func.value, ast.Name) and len(node.args) == 2
               and not node.keywords and isinstance(node.args[0], ast.Name)
               and isinstance(node.args[1], ast.UnaryOp) and isinstance(node.args[1].op, ast.USub)
               and isinstance(node.args[1].operand, ast.Constant)
               and type(node.args[1].operand.value) is int and node.args[1].operand.value == 1]
    return matches[0] if len(matches) == 1 else None


def diagnose_reciprocal_error(exc, script, workspace, sources):
    if type(exc) is not ValueError or str(exc) != ERROR:
        return None
    script, workspace = Path(script).resolve(), Path(workspace).resolve()
    relative = script.relative_to(workspace).as_posix()
    expected = sources.get(relative)
    if not expected or hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        return None
    trace = exc.__traceback__
    while trace.tb_next:
        trace = trace.tb_next
    # Imported functions and third-party packages remain ordinary failures.
    if Path(trace.tb_frame.f_code.co_filename).resolve() != script:
        return None
    tree = ast.parse(script.read_bytes(), filename=str(script))
    call = power_call(tree, trace.tb_lineno)
    if call is None:
        return None
    values = {**trace.tb_frame.f_globals, **trace.tb_frame.f_locals}
    base = values.get(call.args[0].id)
    if values.get(call.func.value.id) is not np or not isinstance(base, np.ndarray):
        return None
    # Require nonnegative, exactly representable integer row sums. A reciprocal
    # is floating point mathematics; do not broaden this to arbitrary powers.
    if base.dtype.kind not in 'iu' or base.size == 0 or np.any(base < 0) or np.any(base > 2**53):
        return None
    return {'kind': 'numpy_float_reciprocal', 'path': relative, 'source_sha256': expected,
            'line': call.lineno, 'column': call.col_offset,
            'original_expression': ast.get_source_segment(script.read_text('utf-8'), call),
            'base_dtype': str(base.dtype), 'base_shape': list(base.shape),
            'replacement': call.func.value.id + '.float_power', 'error': ERROR}


def run_entrypoint(script, repairs, workspace, sources, path_repairs=()):
    for repair in repairs:
        if repair.get('kind') not in ('numpy_builtin_alias', 'numpy_float_reciprocal', RAGGED_KIND):
            raise ValueError('Unknown numeric compatibility operation')
        if repair.get('kind') != 'numpy_builtin_alias':
            continue
        alias = repair.get('alias')
        source = (workspace / repair['path']).resolve(strict=True)
        if (alias not in BUILTIN_ALIASES or not source.is_relative_to(workspace.resolve())
                or repair.get('source_sha256') != sources.get(repair['path'])
                or hashlib.sha256(source.read_bytes()).hexdigest() != repair['source_sha256']
                or repair.get('replacement') != 'builtins.' + alias):
            raise ValueError('NumPy alias repair does not match the original source')
        matches = [node for node in ast.walk(ast.parse(source.read_bytes()))
                   if isinstance(node, ast.Attribute) and node.attr == alias
                   and node.lineno == repair['line'] and node.col_offset == repair['column']
                   and ast.get_source_segment(source.read_text('utf-8'), node) == repair['original_expression']]
        if len(matches) != 1:
            raise ValueError('NumPy alias repair does not match the original expression')
        setattr(np, alias, getattr(builtins, alias))
    relative = script.relative_to(workspace).as_posix()
    ragged = [repair for repair in repairs if repair.get('kind') == RAGGED_KIND and repair.get('path') == relative]
    path_repairs = [repair for repair in path_repairs if repair.get('path') == relative]
    repairs = [repair for repair in repairs if repair.get('kind') == 'numpy_float_reciprocal'
               and repair.get('path') == relative]
    if not repairs and not path_repairs and not ragged:
        return runpy.run_path(str(script), run_name='__main__')
    tree = ast.parse(script.read_bytes(), filename=str(script))
    if path_repairs:
        # This module also runs from a copied observer beside the author code.
        if __package__:
            from .author_path_compatibility import apply_root_repairs
        else:
            from author_path_compatibility import apply_root_repairs
        apply_root_repairs(tree, path_repairs, script, workspace, sources)
    for repair in repairs:
        if (repair.get('kind') != 'numpy_float_reciprocal' or repair.get('path') != relative
                or repair.get('source_sha256') != sources.get(relative)
                or hashlib.sha256(script.read_bytes()).hexdigest() != repair['source_sha256']):
            raise ValueError('Numeric compatibility does not match the pinned entrypoint')
        call = power_call(tree, repair['line'], repair['column'])
        if (call is None or ast.get_source_segment(script.read_text('utf-8'), call) != repair['original_expression']
                or repair.get('replacement') != call.func.value.id + '.float_power'):
            raise ValueError('Numeric compatibility does not match the recorded original call')
        call.func.attr = 'float_power'
    for repair in ragged:
        if (repair.get('source_sha256') != sources.get(relative)
                or hashlib.sha256(script.read_bytes()).hexdigest() != repair['source_sha256']):
            raise ValueError('Ragged compatibility does not match the pinned entrypoint')
        for call in validate_ragged_repair(tree, repair, script.read_text('utf-8')):
            original = call.args[0]
            call.args[0] = ast.copy_location(ast.Call(func=ast.Name(id=RAGGED_HELPER, ctx=ast.Load()),
                args=[original, ast.Constant(value=json.dumps(repair['schema'], sort_keys=True))], keywords=[]), original)
    ast.fix_missing_locations(tree)
    # Preserve normal main-module identity for author classes and imports.
    main = types.ModuleType('__main__')
    main.__dict__.update(__file__=str(script), __package__='', __spec__=None,
                         __loader__=SourceFileLoader('__main__', str(script)), __cached__=None)
    if ragged:
        main.__dict__[RAGGED_HELPER] = _legacy_ragged_array
    previous = sys.modules.get('__main__')
    sys.modules['__main__'] = main
    try:
        exec(compile(tree, str(script), 'exec'), main.__dict__)
        return main.__dict__
    finally:
        if previous is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = previous
