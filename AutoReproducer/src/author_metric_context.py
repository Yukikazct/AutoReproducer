"""Deterministic source evidence for otherwise unlabelled printed uncertainty."""
import ast
from copy import deepcopy
import re

from src.runtime_metrics import NUMBER_PATTERN, metric_number, _NAMED_SUMMARY_RE
from src.metric_keys import norm_metric_key


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


def _reduction_operand(node, operation, numpy_aliases):
    """Recognize reductions and explicit NumPy decimal rounding, without guessing."""
    if (isinstance(node, ast.Call) and _attribute(node.func) in
            [[alias, name] for alias in numpy_aliases for name in ('round', 'around')]
            and not node.keywords and 1 <= len(node.args) <= 2
            and (len(node.args) == 1 or isinstance(node.args[1], ast.Constant)
                 and type(node.args[1].value) is int and 0 <= node.args[1].value <= 15)):
        node = node.args[0]
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != operation or node.keywords:
        return None
    if isinstance(node.func.value, ast.Name) and node.func.value.id in numpy_aliases:
        operand = node.args[0] if len(node.args) == 1 else None
    else:
        operand = node.func.value if not node.args else None
    return ast.dump(operand) if operand is not None else None


def _print_template(node, numpy_aliases):
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'print'
            or len(node.args) != 1 or not isinstance(node.args[0], (ast.JoinedStr, ast.Constant))):
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant):
        if not isinstance(argument.value, str):
            return None
        values = [argument]
    else:
        values = argument.values
    parts, fields, literals = [], [], []
    literal = ''
    for part in values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            literal += part.value
            parts.append(''.join(r'\s*' if p.isspace() else re.escape(p)
                                 for p in re.split(r'(\s+)', part.value) if p))
        elif isinstance(part, ast.FormattedValue):
            index = len(fields)
            fields.append(part.value)
            literals.append(literal)
            literal = ''
            numeric = (_reduction_operand(part.value, 'mean', numpy_aliases) is not None
                       or _reduction_operand(part.value, 'std', numpy_aliases) is not None)
            # A textual field may itself contain a complete observed line.
            parts.append(f'(?P<v{index}>' + (NUMBER_PATTERN if numeric else r'[^\r\n]+?') + ')')
        else:
            return None
    if not parts:
        return None
    literals.append(literal)
    return re.compile(r'\s*' + ''.join(parts) + r'\s*'), fields, literals


def _summary_std_candidates(record, match, fields, literals, numpy_aliases):
    candidates = []
    for index in range(1, len(fields)):
        operand = _reduction_operand(fields[index - 1], 'mean', numpy_aliases)
        separator = literals[index].strip()
        supported_separator = (separator == ',' or separator == '('
                               and literals[index + 1].lstrip().startswith(')'))
        if (operand is None or operand != _reduction_operand(fields[index], 'std', numpy_aliases)
                or not supported_separator
                or metric_number(match[f'v{index - 1}']) != record['value']):
            continue
        # Tie the pair to this metric, including equal values on one line.
        label = re.search(r'((?:(?:macro|micro|weighted)[_\s-]?)?'
                          r'[A-Za-z_][A-Za-z0-9_-]*(?:\s+score)?)\s*[:=]\s*$',
                          record.get('source_quote', '')[:match.start(f'v{index - 1}')], re.I)
        std = metric_number(match[f'v{index}'])
        if (not label or norm_metric_key(label[1]) != record['name'] or std is None or std < 0
                or record.get('std', std) != std):
            continue
        candidates.append((std, fields[index - 1], fields[index]))
    return candidates


def _annotate_summary_std(records, nodes, numpy_aliases, path, source_sha256):
    templates = []
    for node in nodes:
        template = _print_template(node, numpy_aliases)
        if template is not None:
            templates.append((node, *template))
    for record in records:
        if record.get('uncertainty_kind'):
            continue
        supported, ambiguous = [], False
        for node, pattern, fields, literals in templates:
            match = pattern.fullmatch(record.get('source_quote', ''))
            if not match:
                continue
            candidates = _summary_std_candidates(record, match, fields, literals, numpy_aliases)
            if not candidates:
                # A different helper can print the exact same line using SEM
                # or an untyped value. Static source matching cannot identify
                # which helper ran, so this line establishes no SD provenance.
                ambiguous = True
                break
            supported.extend((node, *candidate) for candidate in candidates)
        if ambiguous or not supported or len({candidate[1] for candidate in supported}) != 1:
            continue
        node, std, mean_expression, std_expression = supported[0]
        record.update(std=std, aggregation='mean', uncertainty_source={
            'path': path, 'source_sha256': source_sha256, 'print_line': node.lineno,
            'mean_expression': ast.unparse(mean_expression), 'std_expression': ast.unparse(std_expression)})


_UNKNOWN = ('unknown', None)


def _merge_measurement(left, right):
    if left == right:
        return left
    if left[0] == right[0] == 'list' and None not in (left[1], right[1]):
        return right if left[1] == 'empty' else left if right[1] == 'empty' else _UNKNOWN
    if left[0] == right[0] == 'scalar':
        return right if left[1] == 'zero' else left if right[1] == 'zero' else _UNKNOWN
    return _UNKNOWN


def _named_source_states(tree):
    """Small unit/type interpreter; unknown calls and conflicting branches stay unknown.

    It follows homogeneous lists, literal 100 scaling, original boolean-count
    accuracy helpers and NumPy/Torch array constructors. It does not execute an
    author import or infer units from variable names or observed magnitudes.
    """
    libraries = {alias.asname or alias.name: alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names
                 if alias.name in {'numpy', 'torch'}}
    rebound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
               and isinstance(node.ctx, (ast.Store, ast.Del))}
    rebound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    rebound.update(parts[0] for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                   and isinstance(node.ctx, (ast.Store, ast.Del)) and (parts := _attribute(node)))
    libraries = {alias: name for alias, name in libraries.items() if alias not in rebound}
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    helpers = {}
    constructors = {(alias, method) for alias, name in libraries.items() for method in
                    (('array', 'asarray') if name == 'numpy' else ('Tensor', 'tensor', 'as_tensor'))}

    def boolean(node):
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {'eq', 'ne', 'gt', 'lt', 'ge', 'le'}:
                return len(node.args) == 1 and not node.keywords
            if node.func.attr in {'item', 'float', 'double', 'bool', 'int', 'long', 'cpu', 'detach'}:
                return not node.args and not node.keywords and boolean(node.func.value)
        return False

    def count(node):
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name) and node.func.id == 'len':
            return len(node.args) == 1 and not node.keywords
        if isinstance(node.func, ast.Attribute) and not node.args and not node.keywords:
            return node.func.attr == 'sum' or node.func.attr in {'item', 'float', 'double'} and count(node.func.value)
        return False

    def infer(node, environment):
        if isinstance(node, ast.Name):
            return environment.get(node.id, _UNKNOWN)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value == 0:
            return ('scalar', 'zero')
        if isinstance(node, (ast.List, ast.Tuple)):
            if not node.elts:
                return ('list', 'empty')
            values = [infer(item, environment) for item in node.elts]
            return ('list', values[0][1]) if all(item == values[0] and item[0] == 'scalar' for item in values) else _UNKNOWN
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                for scalar, operand in ((node.left, node.right), (node.right, node.left)):
                    measured = infer(operand, environment)
                    if isinstance(scalar, ast.Constant) and type(scalar.value) in (int, float) and scalar.value == 100:
                        if measured[1] == 'fraction':
                            return (measured[0], 'percent')
                        if measured[0] == 'array':
                            return ('array', None)
            if isinstance(node.op, ast.Div) and count(node.right):
                numerator = node.left
                while (isinstance(numerator, ast.Call) and isinstance(numerator.func, ast.Attribute)
                       and numerator.func.attr in {'item', 'float', 'double'} and not numerator.args and not numerator.keywords):
                    numerator = numerator.func.value
                if (isinstance(numerator, ast.Call) and isinstance(numerator.func, ast.Attribute)
                        and numerator.func.attr == 'sum' and not numerator.args and not numerator.keywords
                        and boolean(numerator.func.value)):
                    return ('scalar', 'fraction')
        if isinstance(node, ast.Subscript):
            measured = infer(node.value, environment)
            return measured if measured[0] == 'array' else _UNKNOWN
        if isinstance(node, ast.Call):
            if tuple(_attribute(node.func)) in constructors and len(node.args) == 1:
                measured = infer(node.args[0], environment)
                return ('array', measured[1] if measured[0] in {'list', 'array'} and measured[1] != 'empty' else None)
            if isinstance(node.func, ast.Name) and node.func.id in helpers:
                return helpers[node.func.id]
            if isinstance(node.func, ast.Attribute) and not node.args and not node.keywords:
                measured = infer(node.func.value, environment)
                if node.func.attr in {'item', 'float', 'double', 'cpu', 'detach'}:
                    return measured
        return _UNKNOWN

    def scan(body, initial, prints, returns):
        environment = dict(initial)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = _UNKNOWN if isinstance(node, ast.AugAssign) else infer(node.value, environment)
                if not isinstance(node, ast.AugAssign) and isinstance(node.value, ast.Name) and value[0] in {'list', 'array'}:
                    # Aliased mutable collections need a richer alias analysis;
                    # retain their observations without claiming static units.
                    environment[node.value.id] = value = _UNKNOWN
                for target in targets:
                    if isinstance(target, ast.Name):
                        environment[target.id] = value
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        values = ([infer(item, environment) for item in node.value.elts]
                                  if not isinstance(node, ast.AugAssign) and isinstance(node.value, (ast.Tuple, ast.List))
                                  and len(node.value.elts) == len(target.elts) else None)
                        for index, item in enumerate(target.elts):
                            if isinstance(item, ast.Name):
                                environment[item.id] = (values[index] if values is not None else
                                    ('scalar', value[1]) if value[0] == 'list' else _UNKNOWN)
                    elif isinstance(target, (ast.Subscript, ast.Attribute)):
                        base = target.value
                        while isinstance(base, (ast.Subscript, ast.Attribute)):
                            base = base.value
                        if isinstance(base, ast.Name):
                            environment[base.id] = _UNKNOWN
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == 'print':
                    prints.append((call, dict(environment)))
                elif (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                      and call.func.attr == 'append' and len(call.args) == 1 and not call.keywords):
                    name = call.func.value.id
                    prior, appended = environment.get(name, _UNKNOWN), infer(call.args[0], environment)
                    if prior[0] == 'list' and appended[0] == 'scalar' and appended[1] in {'fraction', 'percent'}:
                        environment[name] = _merge_measurement(prior, ('list', appended[1]))
                    else:
                        environment[name] = _UNKNOWN
                else:
                    # An arbitrary mutator may change a tracked collection.
                    for argument in [*call.args, getattr(call.func, 'value', None)]:
                        if isinstance(argument, ast.Name) and environment.get(argument.id, _UNKNOWN)[0] in {'list', 'array'}:
                            environment[argument.id] = _UNKNOWN
            elif isinstance(node, ast.Return):
                returns.append(infer(node.value, environment))
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                left = scan(node.body, environment, prints, returns)
                right = scan(node.orelse, environment, prints, returns)
                environment = {name: _merge_measurement(left.get(name, _UNKNOWN), right.get(name, _UNKNOWN))
                               for name in left.keys() | right.keys()}
            elif isinstance(node, (ast.With, ast.Try, ast.Global, ast.Nonlocal)):
                # These control-flow/alias effects are deliberately unsupported.
                environment = {name: _UNKNOWN for name in environment}
        return environment

    for _ in range(2):
        for name, nodes in definitions.items():
            if len(nodes) != 1 or name in rebound:
                continue
            function = nodes[0]
            if any(not (isinstance(decorator, ast.Call) and not decorator.args and not decorator.keywords
                        and len(parts := _attribute(decorator.func)) == 2
                        and libraries.get(parts[0]) == 'torch' and parts[1] == 'no_grad')
                   for decorator in function.decorator_list):
                continue
            returned = []
            scan(function.body, {}, [], returned)
            if returned and all(item == returned[0] for item in returned):
                helpers[name] = returned[0]
    prints = []
    scan(tree.body, {}, prints, [])
    for nodes in definitions.values():
        if len(nodes) == 1:
            scan(nodes[0].body, {}, prints, [])
    return prints, {alias for alias, name in libraries.items() if name == 'numpy'}, infer


def _annotate_named_summaries(records, tree, path, source_sha256):
    states, numpy_aliases, infer = _named_source_states(tree)
    for record in records:
        if record.get('uncertainty_kind') != 'unknown' or record.get('reported_uncertainty') is None:
            continue
        supported, ambiguous = [], False
        for node, environment in states:
            template = _print_template(node, numpy_aliases)
            if template is None:
                continue
            pattern, fields, literals = template
            match = pattern.fullmatch(record.get('source_quote', ''))
            if not match:
                continue
            candidates = []
            for index in range(1, len(fields)):
                operand = _reduction_operand(fields[index - 1], 'mean', numpy_aliases)
                if operand is None or operand != _reduction_operand(fields[index], 'std', numpy_aliases):
                    continue
                if literals[index].strip() not in {'±', '+/-', r'\pm'}:
                    continue
                if (metric_number(match[f'v{index - 1}']) != record['value']
                        or metric_number(match[f'v{index}']) != record['reported_uncertainty']):
                    continue
                named = next((item for item in _NAMED_SUMMARY_RE.finditer(record['source_quote'])
                    if item.start('value') == match.start(f'v{index - 1}')
                    and norm_metric_key(item['name']) == record['name']
                    and {'valid': 'validation', 'val': 'validation', 'training': 'train',
                         'eval': 'test', 'evaluation': 'test'}.get((item['split'] or '').lower(),
                        (item['split'] or '').lower()) == record.get('split', '')), None)
                if named is None or record.get('aggregation') != 'mean':
                    continue
                # Dynamic metric/split text and unknown numeric formatters
                # cannot supply a literal named summary or column identity.
                formatted = [part for part in node.args[0].values if isinstance(part, ast.FormattedValue)]
                if any(part.conversion != -1 or part.format_spec is not None and (
                        not all(isinstance(piece, ast.Constant) and isinstance(piece.value, str)
                                for piece in part.format_spec.values)
                        or not re.fullmatch(r'(?:\.\d{1,2})?[fFeEgGd]?', ''.join(piece.value for piece in part.format_spec.values)))
                       for part in formatted):
                    continue
                if any(not re.fullmatch(r'\d+', match[f'v{prior}']) for prior in range(index - 1)):
                    continue
                expression = fields[index - 1]
                # Obtain the actual common reduction operand, not a variable
                # with the same printed value or a different statistic column.
                measured = expression.func.value if not expression.args else expression.args[0]
                kind, unit = infer(measured, environment)
                if kind != 'array' or record.get('unit') not in ('', unit):
                    continue
                candidates.append((unit, fields[index - 1], fields[index]))
            if not candidates:
                ambiguous = True
                break
            supported.extend((node, *candidate) for candidate in candidates)
        if ambiguous or not supported or len({(item[1], ast.dump(item[2]), ast.dump(item[3])) for item in supported}) != 1:
            continue
        node, unit, mean_expression, std_expression = supported[0]
        record['std'] = record.pop('reported_uncertainty')
        record.pop('uncertainty_kind', None)
        record['uncertainty_source'] = {'path': path, 'source_sha256': source_sha256,
            'print_line': node.lineno, 'mean_expression': ast.unparse(mean_expression),
            'std_expression': ast.unparse(std_expression)}
        if unit in {'fraction', 'percent'}:
            record['unit'] = unit
            record['unit_source'] = dict(record['uncertainty_source'], kind='static_original_array_dataflow')


def annotate_uncertainty(records, text, path, source_sha256):
    """Keep observed values; CI half-widths must not masquerade as SD.

    Matching mean/std reductions on the same operand establish parenthesized
    or comma-separated SD, including explicit NumPy rounding and function
    scopes. CI half-widths separately require the
    original Seaborn ci(bootstrap(...), confidence) assignment. Unrelated
    keywords or model suggestions are insufficient for either rule.
    """
    result = deepcopy(records)
    tree = ast.parse(text)
    nodes = list(_module_nodes(tree))
    aliases = {alias.asname or alias.name for node in nodes if isinstance(node, ast.Import)
               for alias in node.names if alias.name == 'seaborn'}
    numpy_aliases = {alias.asname or alias.name for node in nodes if isinstance(node, ast.Import)
                     for alias in node.names if alias.name == 'numpy'}
    _annotate_named_summaries(result, tree, path, source_sha256)
    _annotate_summary_std(result, ast.walk(tree), numpy_aliases, path, source_sha256)
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
