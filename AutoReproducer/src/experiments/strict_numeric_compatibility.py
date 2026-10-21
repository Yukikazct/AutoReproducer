"""Source-bound numeric compatibility shared by strict capture entrypoints.

Copied beside the frozen adapter. Only an actual author failure emits a repair
candidate; applying it remains the service's bounded, independently checked retry.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def _numeric_module():
    if __package__:
        from . import author_numeric_compatibility
    else:
        import author_numeric_compatibility
    return author_numeric_compatibility


def numeric_compatibility(config, workspace):
    """Revalidate recorded operations against unchanged original source."""
    repairs = config.get('numeric_compatibility', [])
    if not isinstance(repairs, list) or len(repairs) > 16:
        raise ValueError('Strict numeric compatibility must be a bounded operation list')
    workspace = Path(workspace).resolve(strict=True)
    sources = config['source_files']
    numeric = _numeric_module()
    for repair in repairs:
        if not isinstance(repair, dict) or repair.get('kind') not in (
                'numpy_builtin_alias', 'numpy_float_reciprocal', numeric.RAGGED_KIND):
            raise ValueError('Unknown strict numeric compatibility operation')
        relative = repair.get('path')
        expected = repair.get('source_sha256')
        if not isinstance(relative, str) or not expected or sources.get(relative) != expected:
            raise ValueError('Strict numeric compatibility source is not pinned')
        path = (workspace / relative).resolve(strict=True)
        if (not path.is_relative_to(workspace) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Strict numeric compatibility source changed')
        source = path.read_text('utf-8')
        tree = ast.parse(source, filename=str(path))
        if repair['kind'] == numeric.RAGGED_KIND:
            numeric.validate_ragged_repair(tree, repair, source)
        elif repair['kind'] == 'numpy_float_reciprocal':
            call = numeric.power_call(tree, repair.get('line'), repair.get('column'))
            if (call is None or ast.get_source_segment(source, call) != repair.get('original_expression')
                    or repair.get('replacement') != call.func.value.id + '.float_power'):
                raise ValueError('Strict numeric compatibility differs from the original reciprocal')
        else:
            alias = repair.get('alias')
            matches = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                       and node.attr == alias and node.lineno == repair.get('line')
                       and node.col_offset == repair.get('column')
                       and ast.get_source_segment(source, node) == repair.get('original_expression')]
            if (alias not in numeric.BUILTIN_ALIASES or len(matches) != 1
                    or repair.get('replacement') != 'builtins.' + alias):
                raise ValueError('Strict numeric compatibility differs from the original alias')
    return deepcopy(repairs)


def prepare_capture(config, root):
    """Invalidate previous receipts before any new capture can fail or cancel."""
    identifier = config['train']['id']
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', identifier):
        raise ValueError('Strict numeric failure needs the original step identifier')
    root = Path(root)
    for name in ('capture.json', 'independent_metrics.json', 'numeric_failure_' + identifier + '.json'):
        (root / name).unlink(missing_ok=True)
    return numeric_compatibility(config, root.parent)


def prepare_evaluation(root):
    (Path(root) / 'independent_metrics.json').unlink(missing_ok=True)


def verify_numeric_receipts(config, workspace, *receipts):
    repairs = numeric_compatibility(config, workspace)
    if any(record.get('numeric_compatibility') != repairs for record in receipts):
        raise ValueError('Strict numeric compatibility receipts differ from the frozen runtime configuration')
    return repairs


def run_entrypoint(config, script, root, update_count):
    """Run original code and diagnose only its real, source-bound exception."""
    root = Path(root)
    numeric = _numeric_module()
    repairs = numeric_compatibility(config, root.parent)
    try:
        return numeric.run_entrypoint(script, repairs, root.parent, config['source_files'])
    except Exception as exc:
        candidate = (numeric.diagnose_reciprocal_error(exc, script, root.parent, config['source_files'])
                     or numeric.diagnose_numpy_alias_error(exc, script, root.parent, config['source_files'])
                     or numeric.diagnose_ragged_summary(exc, script, root.parent, config['source_files']))
        if candidate is not None:
            updates = update_count()
            if type(updates) is not int or updates < 0:
                raise ValueError('Strict numeric failure has an invalid cumulative update count') from exc
            identifier = config['train']['id']
            (root / ('numeric_failure_' + identifier + '.json')).write_text(json.dumps({
                'spec_sha256': config['spec_sha256'], 'step': identifier,
                'optimizer_steps': updates, 'repair': candidate}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        raise
