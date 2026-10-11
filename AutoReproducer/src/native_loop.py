"""Resolve an original module loop and its validation selection; never eval text."""
import ast
import re

_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,127}')
_COMPARISONS = {'lt': ast.Lt, 'le': ast.LtE, 'gt': ast.Gt, 'ge': ast.GtE}


def resolve_module_loop(source, scope):
    if not isinstance(scope, dict) or set(scope) != {'path', 'loop_counter', 'selection'}:
        raise ValueError('Native module scope needs a path, loop counter and validation selection')
    selection = scope['selection']
    fields = {'validation_metric', 'comparison', 'best_metric', 'result', 'result_unit'}
    if (not isinstance(selection, dict) or set(selection) != fields
            or selection.get('comparison') not in _COMPARISONS
            or selection.get('result_unit') not in {'fraction', 'percent'}):
        raise ValueError('Native selection needs a fixed validation comparison and reported result')
    names = [scope['loop_counter'], *(selection[key] for key in ('validation_metric', 'best_metric', 'result'))]
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in names):
        raise ValueError('Native loop and selection values must be original identifiers')
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == scope['loop_counter']]
    if len(loops) != 1 or loops[0] not in tree.body:
        raise ValueError('Native loop must resolve to one unique top-level author for loop')
    loop = loops[0]
    guards = [node for node in ast.walk(loop) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
              and isinstance(node.test.ops[0], _COMPARISONS[selection['comparison']])
              and isinstance(node.test.left, ast.Name) and node.test.left.id == selection['validation_metric']
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
            'selection_start': assignment.lineno, 'selection_end': assignment.end_lineno}
