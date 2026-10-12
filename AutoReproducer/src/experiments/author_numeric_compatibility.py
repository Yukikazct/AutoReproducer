"""A source-bound float reciprocal repair, enabled only after a real failure.

The repository file stays unchanged. The retry compiles one recorded call as
numpy.float_power instead of numpy.power; every other AST node stays original.
"""
import ast
import builtins
import hashlib
from importlib.machinery import SourceFileLoader
from pathlib import Path
import runpy
import re
import sys
import types

import numpy as np


ERROR = 'Integers to negative integer powers are not allowed.'
BUILTIN_ALIASES = {'bool', 'int', 'float', 'complex', 'object', 'str'}


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


def run_entrypoint(script, repairs, workspace, sources):
    for repair in repairs:
        if repair.get('kind') not in ('numpy_builtin_alias', 'numpy_float_reciprocal'):
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
    repairs = [repair for repair in repairs if repair.get('kind') == 'numpy_float_reciprocal'
               and repair.get('path') == relative]
    if not repairs:
        return runpy.run_path(str(script), run_name='__main__')
    tree = ast.parse(script.read_bytes(), filename=str(script))
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
    # Preserve normal main-module identity for author classes and imports.
    main = types.ModuleType('__main__')
    main.__dict__.update(__file__=str(script), __package__='', __spec__=None,
                         __loader__=SourceFileLoader('__main__', str(script)), __cached__=None)
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
