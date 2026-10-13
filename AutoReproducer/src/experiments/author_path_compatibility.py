"""Relocate a missing POSIX temporary data root after an actual author failure."""
import ast
import hashlib
from pathlib import Path, PurePosixPath


DATA_ROOT = '_autorepro_capture/author_data'


def temporary_root(value):
    if not isinstance(value, str) or '\\' in value:
        return False
    parts = PurePosixPath(value).parts
    return len(parts) >= 2 and parts[:2] == ('/', 'tmp') and '..' not in parts


def root_literal(tree, line, column):
    matches = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
               for keyword in node.keywords if keyword.arg == 'root'
               and isinstance(keyword.value, ast.Constant)
               and keyword.value.lineno == line and keyword.value.col_offset == column
               and temporary_root(keyword.value.value)]
    return matches[0] if len(matches) == 1 else None


def replacement_root(value):
    return PurePosixPath(DATA_ROOT).joinpath(*PurePosixPath(value).parts[2:]).as_posix()


def diagnose_temporary_root_error(exc, script, workspace, sources):
    """Use the underlying file error even when the loader wraps it as a download error."""
    current, seen, missing = exc, set(), None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, FileNotFoundError) and temporary_root(current.filename):
            missing = current
            break
        current = current.__cause__ or current.__context__
    if missing is None:
        return None
    script, workspace = Path(script).resolve(), Path(workspace).resolve()
    if not script.is_relative_to(workspace):
        return None
    relative = script.relative_to(workspace).as_posix()
    expected = sources.get(relative)
    if not expected or hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        return None
    tree = ast.parse(script.read_bytes())
    trace = exc.__traceback__
    while trace:
        if Path(trace.tb_frame.f_code.co_filename).resolve() == script:
            matches = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and node.lineno <= trace.tb_lineno <= node.end_lineno
                       for keyword in node.keywords if keyword.arg == 'root'
                       and isinstance(keyword.value, ast.Constant)
                       and temporary_root(keyword.value.value)
                       and PurePosixPath(missing.filename).is_relative_to(PurePosixPath(keyword.value.value))]
            if len(matches) == 1:
                literal = matches[0]
                return {'kind': 'temporary_dataset_root', 'path': relative, 'source_sha256': expected,
                        'line': literal.lineno, 'column': literal.col_offset,
                        'original_expression': ast.get_source_segment(script.read_text('utf-8'), literal),
                        'original_root': literal.value, 'replacement_root': replacement_root(literal.value),
                        'failed_filename': missing.filename, 'error': str(missing)}
        trace = trace.tb_next
    return None


def validate_root_repair(tree, repair, source, workspace, sources):
    workspace, source = Path(workspace).resolve(), Path(source).resolve()
    if not source.is_relative_to(workspace):
        raise ValueError('Temporary data root source escaped the workspace')
    relative = source.relative_to(workspace).as_posix()
    if (repair.get('kind') != 'temporary_dataset_root' or repair.get('path') != relative
            or repair.get('source_sha256') != sources.get(relative)
            or hashlib.sha256(source.read_bytes()).hexdigest() != repair.get('source_sha256')
            or not temporary_root(repair.get('original_root'))
            or repair.get('replacement_root') != replacement_root(repair['original_root'])):
        raise ValueError('Temporary data root repair does not match the original source')
    literal = root_literal(tree, repair.get('line'), repair.get('column'))
    if (literal is None or literal.value != repair.get('original_root')
            or ast.get_source_segment(source.read_text('utf-8'), literal) != repair.get('original_expression')
            or not temporary_root(repair.get('failed_filename'))
            or not PurePosixPath(repair['failed_filename']).is_relative_to(PurePosixPath(literal.value))):
        raise ValueError('Temporary data root repair does not match the failing call')
    target = workspace / repair['replacement_root']
    if not target.resolve().is_relative_to(workspace):
        raise ValueError('Temporary data root escaped the workspace')
    return literal, target


def apply_root_repairs(tree, repairs, script, workspace, sources):
    for repair in repairs:
        literal, target = validate_root_repair(tree, repair, script, workspace, sources)
        target.mkdir(parents=True, exist_ok=True)
        literal.value = str(target)
