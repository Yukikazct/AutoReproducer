"""Deterministic source evidence for otherwise unlabelled printed uncertainty."""
import ast
from copy import deepcopy
import re

from src.runtime_metrics import NUMBER_PATTERN, metric_number


def _module_nodes(node):
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from _module_nodes(child)


def _attribute(node):
    names = []
    while isinstance(node, ast.Attribute):
        names.insert(0, node.attr)
        node = node.value
    return [node.id, *names] if isinstance(node, ast.Name) else []


def _variable(node):
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
            and isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant)
            and type(node.right.value) in (int, float)):
        return node.left.id
    return None


def annotate_uncertainty(records, text, path, source_sha256):
    """Keep observed values; CI half-widths must not masquerade as SD.

    Only a matching printed f-string, a direct variable assignment and the
    original Seaborn ci(bootstrap(...), confidence) call establish this context.
    Unrelated keywords or model suggestions are insufficient.
    """
    result = deepcopy(records)
    tree = ast.parse(text)
    nodes = list(_module_nodes(tree))
    aliases = {alias.asname or alias.name for node in nodes if isinstance(node, ast.Import)
               for alias in node.names if alias.name == 'seaborn'}
    numpy_aliases = {alias.asname or alias.name for node in nodes if isinstance(node, ast.Import)
                     for alias in node.names if alias.name == 'numpy'}
    definitions = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        confidence = []
        value = node.value
        # A CI can also be used to derive other statistics. Only the observed
        # max(abs(ci(bootstrap(values), level) - values.mean())) recipe proves
        # this printed quantity is the interval half-width around the mean.
        if (not isinstance(value, ast.Call) or len(value.args) != 1 or value.keywords
                or len(_attribute(value.func)) != 2
                or _attribute(value.func)[-1:] != ['max'] or _attribute(value.func)[0] not in numpy_aliases):
            definitions[node.targets[0].id] = None
            continue
        absolute = value.args[0]
        if (not isinstance(absolute, ast.Call) or len(absolute.args) != 1 or absolute.keywords
                or _attribute(absolute.func) != [_attribute(value.func)[0], 'abs']):
            definitions[node.targets[0].id] = None
            continue
        difference = absolute.args[0]
        if not isinstance(difference, ast.BinOp) or not isinstance(difference.op, ast.Sub):
            definitions[node.targets[0].id] = None
            continue
        center = difference.right
        if (not isinstance(center, ast.Call) or not isinstance(center.func, ast.Attribute)
                or not isinstance(center.func.value, ast.Name) or center.func.attr != 'mean'
                or center.args or center.keywords):
            definitions[node.targets[0].id] = None
            continue
        for call in [difference.left]:
            if (isinstance(call, ast.Call) and _attribute(call.func)[-2:] == ['utils', 'ci']
                    and _attribute(call.func)[:1] and _attribute(call.func)[0] in aliases
                    and len(call.args) == 2 and isinstance(call.args[1], ast.Constant)
                    and type(call.args[1].value) in (int, float) and 0 < call.args[1].value < 100
                    and isinstance(call.args[0], ast.Call)
                    and _attribute(call.args[0].func) == [_attribute(call.func)[0], 'algorithms', 'bootstrap']
                    and call.args[0].args and isinstance(call.args[0].args[0], ast.Name)
                    and call.args[0].args[0].id == center.func.value.id):
                confidence.append(call.args[1].value)
        definitions[node.targets[0].id] = (confidence[0], node.lineno) if len(confidence) == 1 else None
    for node in nodes:
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'print'
                or len(node.args) != 1 or not isinstance(node.args[0], ast.JoinedStr)):
            continue
        parts, fields, spreads = [], [], []
        previous = ''
        for value in node.args[0].values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                previous = value.value
                parts.append(''.join(r'\s*' if part.isspace() else re.escape(part)
                                     for part in re.split(r'(\s+)', value.value) if part))
            elif isinstance(value, ast.FormattedValue):
                index = len(fields)
                fields.append(value.value)
                parts.append('(?P<v' + str(index) + '>' + NUMBER_PATTERN + ')')
                if previous.rstrip().endswith(('±', '+/-')):
                    spreads.append(index)
                previous = ''
            else:
                parts = []
                break
        if not parts or len(spreads) != 1 or spreads[0] < 1:
            continue
        index = spreads[0]
        definition = definitions.get(_variable(fields[index]))
        if not definition or definition[1] >= node.lineno:
            continue
        for record in result:
            if record.get('std') is None:
                continue
            match = re.fullmatch(r'\s*' + ''.join(parts) + r'\s*', record.get('source_quote', ''))
            if (match and metric_number(match['v' + str(index)]) == record['std']
                    and metric_number(match['v' + str(index - 1)]) == record['value']):
                record['reported_uncertainty'] = record.pop('std')
                record['uncertainty_kind'] = 'bootstrap_confidence_interval'
                record['confidence_level'] = definition[0]
                record['uncertainty_source'] = {'path': path, 'source_sha256': source_sha256,
                    'definition_line': definition[1], 'print_line': node.lineno}
    return result
