"""Interpret numeric-only author summaries using frozen print/metric sources.

AI supplies column meanings. Values always come from the actual completed
process output and are never supplied or calculated by the model.
"""
import ast
from copy import deepcopy
import json
import re
import string
import textwrap

from src.author_failure_analysis import _analysis_feedback
from src.author_comparison_context import comparison_plan, comparison_view
from src.discovered_repository_plan import _citations, _json_response, _read_source
from src.runtime_metrics import (METRICS_MARKER, NUMBER_PATTERN, _TEXT_METRICS, _LOGGER_ALL,
                                 _LOGGER_RUN, _printed_logger_context, _run_count, metric_number)
from src.metric_keys import norm_metric_key


_FINAL_PREFIX = re.compile(r'^\s*Final\s+(?:test|train|val|valid|validation)\s+results\b', re.I)
_FINAL_STATISTICS = re.compile(
    r'\s*Final\s+(?P<split>test|train|val|valid|validation)\s+results'
    r'(?:\s+(?:over|across)\s+(?P<runs>[1-9]\d*)\s+(?:runs|trials|seeds))?'
    r'\s*--\s*mean\s*:\s*(?P<mean>' + NUMBER_PATTERN + r')\s*,\s*'
    r'std\s*:\s*(?P<std>' + NUMBER_PATTERN + r')\s*', re.I)
_LOGGER_TEST = re.compile(r'\s*Final\s+test\s*:\s*(?P<mean>' + NUMBER_PATTERN + r')'
    r'\s*(?P<percent>%)?\s*(?:±|\+/-|\\pm(?![A-Za-z]))\s*'
    r'(?P<std>' + NUMBER_PATTERN + r')\s*(?P<std_percent>%)?\s*', re.I)
_LOGGER_SINGLE_TEST = re.compile(r'\s*Final\s+test\s*:\s*(?P<value>' + NUMBER_PATTERN + r')\s*%?\s*', re.I)
_SCALAR_TENSOR = re.compile(
    r"\s*tensor\(\s*(?P<value>" + NUMBER_PATTERN + r")\s*"
    r"(?:,\s*(?:device=(?P<device_quote>['\"])(?:cpu|mps|(?:cuda|xpu)(?::\d+)?)(?P=device_quote)"
    r"|dtype=torch\.(?:float16|float32|float64|bfloat16|int32|int64)"
    r"|requires_grad=(?:True|False)|grad_fn=<[A-Za-z][A-Za-z0-9_]*>))*\s*\)\s*", re.I)


def _tensor_summaries(lines, step_id, stream):
    """Keep original scalar lines separate; source binding must prove statistics."""
    final = len(lines)
    while final and not lines[final - 1].strip():
        final -= 1
    if final:
        try:
            receipt = json.loads(lines[final - 1])
        except (ValueError, TypeError):
            receipt = None
        # This is the owned observer's completion receipt, never author data.
        if (isinstance(receipt, dict) and set(receipt) == {'author_command_finished', 'optimizer_steps'}
                and receipt['author_command_finished'] == step_id
                and type(receipt['optimizer_steps']) is int and receipt['optimizer_steps'] >= 0):
            final -= 1
    if final < 2:
        return []
    values = []
    for line in lines[final - 2:final]:
        match = _SCALAR_TENSOR.fullmatch(line)
        value = metric_number(match['value']) if match else None
        if value is None:
            return []
        values.append(value)
    return [{'id': f'{step_id}:{stream}:{final - 1}', 'step_id': step_id,
        'stream': stream, 'source_quote': '\n'.join(lines[final - 2:final]),
        'values': values, 'summary_format': 'tensor_pair',
        'output_sources': [{'source': f'{step_id}:{stream}:{index + 1}', 'source_quote': lines[index]}
                           for index in range(final - 2, final)]}]


def _logger_summary(lines, step_id, stream):
    """Keep the unique aggregate test line; its meaning still needs original source."""
    headers = [index for index, line in enumerate(lines) if _LOGGER_ALL.fullmatch(line)]
    if len(headers) != 1:
        return []
    header = headers[0]
    finals = [(index, line) for index, line in enumerate(lines[header + 1:], header + 1)
              if re.match(r'\s*Final\s+test\b', line, re.I)]
    if len(finals) != 1:
        return []
    index, line = finals[0]
    match = _LOGGER_TEST.fullmatch(line)
    if match is None:
        return []
    mean, std = metric_number(match['mean']), metric_number(match['std'])
    if mean is None or std is None or std < 0:
        return []
    declared, current, completed, duplicate = None, None, {}, False
    for number, previous in enumerate(lines[:header], 1):
        context = _printed_logger_context(previous)
        if context is not None:
            declared, current, completed, duplicate = context.get('runs'), None, {}, False
        run = _LOGGER_RUN.fullmatch(previous)
        if run:
            current = _run_count(run[1])
            duplicate |= current in completed
            completed.setdefault(current, [])
        elif current is not None:
            single = _LOGGER_SINGLE_TEST.fullmatch(previous)
            if single and metric_number(single['value']) is not None:
                completed[current].append(number)
    ids = set(completed)
    runs = (declared if declared and declared <= 100000 and None not in ids and not duplicate
            and all(len(rows) == 1 for rows in completed.values())
            and ids in (set(range(declared)), set(range(1, declared + 1))) else None)
    candidate = {'id': f'{step_id}:{stream}:{index + 1}', 'step_id': step_id, 'stream': stream,
        'source_quote': line, 'values': [mean, std], 'summary_format': 'all_runs',
        'labelled_statistics': {'split': 'test', 'aggregation': 'mean', 'mean_column': 0,
                                'std_column': 1, 'runs': runs,
                                'unit': 'percent' if match['percent'] or match['std_percent'] else None}}
    if runs is not None:
        candidate['runs_source'] = {'kind': 'completed_logger_runs', 'value': runs,
            'source': f'{step_id}:{stream}:{header + 1}', 'source_quote': lines[header],
            'run_ids': sorted(ids), 'final_test_lines': [completed[run][0] for run in sorted(ids)]}
    return [candidate]


def _final_statistics(line):
    """Labels identify statistics and a split, never a metric name or unit."""
    match = _FINAL_STATISTICS.fullmatch(line)
    if match is None:
        return None
    mean, std = metric_number(match['mean']), metric_number(match['std'])
    runs = int(match['runs']) if match['runs'] else None
    if mean is None or std is None or std < 0 or runs is not None and runs > 100000:
        return None
    return {'values': [mean, std], 'labelled_statistics': {
        'split': {'val': 'validation', 'valid': 'validation'}.get(match['split'].lower(), match['split'].lower()),
        'aggregation': 'mean', 'mean_column': 0, 'std_column': 1, 'runs': runs}}


def numeric_summaries(execution):
    candidates = []
    if execution.get('success') is not True:
        return candidates
    for step in execution.get('steps', []):
        if step.get('success') is not True or step.get('exit_code') != 0:
            continue
        if step.get('metric_records', step.get('metrics')) is not None:
            continue
        if any(line.strip().startswith(METRICS_MARKER)
               for stream in ('stdout', 'stderr') for line in step.get(stream, '').splitlines()):
            continue
        for stream in ('stdout', 'stderr'):
            found, labelled, final_lines = [], [], 0
            lines = step.get(stream, '').splitlines()
            if any(line.strip().startswith(METRICS_MARKER) for line in lines):
                continue
            if any(_LOGGER_ALL.fullmatch(line) for line in lines):
                candidates.extend(_logger_summary(lines, step['id'], stream))
                continue
            for line_number, line in enumerate(lines, 1):
                if _FINAL_PREFIX.match(line):
                    final_lines += 1
                    summary = _final_statistics(line)
                    if summary is not None:
                        labelled.append({'id': f"{step['id']}:{stream}:{line_number}", 'step_id': step['id'],
                                         'stream': stream, 'source_quote': line, **summary})
                    continue
                if not re.fullmatch(r'\s*' + NUMBER_PATTERN + r'(?:[\s,]+' + NUMBER_PATTERN + r'){1,15}\s*', line):
                    continue
                values = [metric_number(value) for value in re.findall(NUMBER_PATTERN, line)]
                if any(value is None for value in values):
                    continue
                found.append({'id': f"{step['id']}:{stream}:{line_number}", 'step_id': step['id'],
                              'stream': stream, 'source_quote': line, 'values': values})
            # The last complete numeric summary is a candidate, not automatically
            # a metric. The model must link its order to the original print code.
            if final_lines:
                # Repeated final summaries have no metric labels to distinguish
                # them. A malformed or competing final line cannot be ignored.
                if final_lines == 1 and len(labelled) == 1:
                    candidates.extend(labelled)
            else:
                tensor_summary = _tensor_summaries(lines, step['id'], stream)
                if tensor_summary:
                    candidates.extend(tensor_summary)
                elif found:
                    candidates.append(found[-1])
    return candidates[:12]


def _original_quote(quote, sources, workspace):
    """Recover exact source bytes when a model escaped newlines or chose an
    earlier excerpt of the correct frozen file. Never normalize identifiers,
    constants or operators; retain the matched original quotation and location.
    """
    if not isinstance(quote, str) or not 8 <= len(quote) <= 3000:
        return None
    variants = [quote, quote.replace('\\\\', '\\')]
    variants += [value.replace('\\"', '"').replace("\\'", "'") for value in variants]
    variants += [value.replace('\\n', '\n').replace('\\r', '\r') for value in variants]
    for source in sources:
        text = _read_source(workspace, source['path'], source['sha256'])['text']
        # Preserve indentation when it already selects one original occurrence.
        # Normalizing it first can make distinct nested definitions ambiguous.
        for value in variants:
            pattern = re.escape(value.replace('\r\n', '\n')).replace(re.escape('\n'), r'\r?\n')
            matches = list(re.finditer(pattern, text))
            if len(matches) == 1:
                match = matches[0]
                return {'quote': match.group(), 'path': source['path'], 'sha256': source['sha256'],
                        'line': text[:match.start()].count('\n') + 1}
        for value in variants:
            pattern = r'\s+'.join(re.escape(token) for token in value.split())
            matches = list(re.finditer(pattern, text))
            if len(matches) == 1:
                match = matches[0]
                return {'quote': match.group(), 'path': source['path'], 'sha256': source['sha256'],
                        'line': text[:match.start()].count('\n') + 1}
    return None


def _final_source_binding(output, locations, sources, workspace, metric, unit):
    """Bind named statistics to the frozen formatter and matching reductions.

    The model still interprets the evaluator. This check prevents unrelated
    print/mean/std quotations or an explicitly named loss from proving accuracy.
    """
    if output['stream'] != 'stdout':
        raise ValueError('Original final print does not establish the selected output stream')
    printer = locations['print_quote']
    aggregate = locations['aggregation_quote']
    if printer['path'] != aggregate['path'] or printer['sha256'] != aggregate['sha256']:
        raise ValueError('Final summary print and statistics require the same original source')
    text = _read_source(workspace, printer['path'], printer['sha256'])['text']
    tree = ast.parse(text)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def scope(node):
        while node in parents and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            node = parents[node]
        return node

    def assignment(expression, use):
        if not isinstance(expression, ast.Name):
            return expression, use
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == expression.id and node.lineno < use.lineno
                   and scope(node) is scope(use)]
        if len(matches) != 1:
            raise ValueError('Final statistic variable does not have a unique original definition')
        return matches[0].value, matches[0]

    numpy_aliases = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names if alias.name == 'numpy'}

    def statistic(expression, use, operation):
        call, definition = assignment(expression, use)
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != operation:
            raise ValueError('Final summary lacks the selected original ' + operation + ' reduction')
        if call.keywords:
            raise ValueError('Final summary reduction options require separate interpretation')
        if isinstance(call.func.value, ast.Name) and call.func.value.id in numpy_aliases and len(call.args) == 1:
            values = call.args[0]
        elif not call.args and not (isinstance(call.func.value, ast.Name) and call.func.value.id in numpy_aliases):
            values = call.func.value
        else:
            raise ValueError('Final summary reduction is not a supported original array statistic')
        if ast.get_source_segment(text, definition) not in aggregate['quote']:
            raise ValueError('Aggregation quote does not contain the original statistic definitions')
        return values, definition

    def scaled_unit(operand, use, *, require_percent=True, whole_source=False):
        """An unmarked percentage requires a connected, literal author scale."""
        location = ({**printer, 'quote': text} if whole_source else locations.get('unit_quote'))
        if (not location or location['path'] != printer['path'] or location['sha256'] != printer['sha256']):
            raise ValueError('Unmarked logger summary needs its original unit scaling source')
        quoted = location['quote']
        start = text.find(quoted)
        if start < 0 or text.find(quoted, start + 1) >= 0:
            raise ValueError('Logger unit quote has no unique original source span')

        def position(offset):
            prefix = text[:offset]
            return prefix.count('\n') + 1, len(prefix.rsplit('\n', 1)[-1].encode('utf-8'))

        quote_start, quote_end = position(start), position(start + len(quoted))
        seeds = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)
                 and isinstance(node.op, ast.Mult) and node.lineno < use.lineno and scope(node) is scope(use)
                 and quote_start <= (node.lineno, node.col_offset)
                 and (node.end_lineno, node.end_col_offset) <= quote_end
                 and any(isinstance(value, ast.Constant) and type(value.value) in (int, float)
                         and value.value == 100 for value in (node.left, node.right))
                 and ast.get_source_segment(text, node) in location['quote']]
        libraries = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names if alias.name in {'numpy', 'torch'}}
        rebound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                   and isinstance(node.ctx, (ast.Store, ast.Del))}
        rebound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
        rebound.update(node.name for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        libraries -= rebound

        def branches(node):
            result = {}
            while node in parents:
                parent = parents[node]
                if isinstance(parent, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    result[parent] = next((field for field, value in ast.iter_fields(parent)
                                           if isinstance(value, list) and node in value), 'unknown')
                node = parent
            return result

        def compatible(left, right):
            left, right = branches(left), branches(right)
            return all(node not in right or right[node] == side for node, side in left.items())

        def dominates(left, right):
            left, right = branches(left), branches(right)
            return all(right.get(node) == side for node, side in left.items())

        def intact(name, definition, at, *, appends=False):
            for node in ast.walk(tree):
                if (not (definition.lineno, definition.col_offset)
                        < (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
                        < (at.lineno, at.col_offset)
                        or scope(node) is not scope(at) or not compatible(node, at)):
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        while isinstance(target, (ast.Subscript, ast.Attribute)):
                            target = target.value
                        if isinstance(target, ast.Name) and target.id == name:
                            return False
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == name and node.func.attr not in {
                            'mean', 'std', 'max', 'min', 'sum', 'argmax', 'item', 'float', 'double', 'cpu', 'detach',
                            *({'append'} if appends else set())}:
                        return False
                    safe = (node.func.value.id in libraries and node.func.attr in {'tensor', 'array', 'asarray', 'stack'}
                            or node.func.attr == 'append')
                else:
                    safe = isinstance(node.func, ast.Name) and node.func.id == 'print'
                if not safe and any(isinstance(value, ast.Name) and value.id == name for argument in
                                    [*node.args, *(item.value for item in node.keywords)] for value in ast.walk(argument)):
                    return False
            return True

        def connected(expression, at, seed, seen=()):
            if expression is seed:
                return True
            if isinstance(expression, ast.Name):
                key = (expression.id, at.lineno)
                if key in seen:
                    return False
                seen = (*seen, key)
                parent = at
                while parent in parents:
                    parent = parents[parent]
                    if (isinstance(parent, ast.For) and isinstance(parent.target, ast.Name)
                            and parent.target.id == expression.id):
                        return intact(expression.id, parent, at) and connected(parent.iter, parent, seed, seen)
                definitions = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == expression.id and node.lineno < at.lineno
                    and scope(node) is scope(at) and compatible(node, at)]
                guaranteed = [definition for definition in definitions if dominates(definition, at)]
                if not guaranteed:
                    return False
                definition = max(guaranteed, key=lambda node: node.lineno)
                if isinstance(definition.value, ast.List) and not definition.value.elts:
                    mutations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == expression.id
                        and definition.lineno < node.lineno < at.lineno and scope(node) is scope(at)
                        and compatible(node, at)]
                    return intact(expression.id, definition, at, appends=True) and bool(mutations) and all(node.func.attr == 'append' and len(node.args) == 1
                        and not node.keywords and connected(node.args[0], node, seed, seen) for node in mutations)
                return intact(expression.id, definition, at) and connected(definition.value, definition, seed, seen)
            if isinstance(expression, ast.Subscript):
                return connected(expression.value, at, seed, seen)
            if isinstance(expression, (ast.Tuple, ast.List)):
                return bool(expression.elts) and all(connected(value, at, seed, seen) for value in expression.elts)
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
                if (isinstance(expression.func.value, ast.Name) and expression.func.value.id in libraries
                        and expression.func.attr in {'tensor', 'array', 'asarray', 'stack'} and len(expression.args) == 1
                        and not expression.keywords):
                    return connected(expression.args[0], at, seed, seen)
                if (expression.func.attr in {'mean', 'std', 'max', 'item', 'float', 'double', 'cpu', 'detach'}
                        and not expression.args and not expression.keywords):
                    return connected(expression.func.value, at, seed, seen)
            return False

        supporting = [seed for seed in seeds if connected(operand, use, seed)]
        if require_percent and unit != 'percent' or len(supporting) != 1:
            raise ValueError('Logger summary unit needs one connected original 100 scaling expression')
        return {'kind': 'original_array_scaling', **location,
                'expression': ast.get_source_segment(text, supporting[0])}

    matches = []
    tensor_pair = output.get('summary_format') == 'tensor_pair'
    if tensor_pair:
        quoted_prints = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id == 'print' and len(node.value.args) == 1 and not node.value.keywords
            and ast.get_source_segment(text, node) in printer['quote']
            and printer['line'] <= node.lineno < printer['line'] + printer['quote'].count('\n') + 1]
        quoted_prints.sort(key=lambda node: node.lineno)
        if len(quoted_prints) != 2:
            raise ValueError('Tensor summary needs exactly two original scalar print statements')
        first, second = quoted_prints
        parent = parents[first]
        main_guard = (isinstance(parent, ast.If) and isinstance(parents.get(parent), ast.Module)
            and isinstance(parent.test, ast.Compare) and len(parent.test.ops) == 1
            and isinstance(parent.test.ops[0], ast.Eq) and len(parent.test.comparators) == 1
            and isinstance(parent.test.left, ast.Name) and parent.test.left.id == '__name__'
            and isinstance(parent.test.comparators[0], ast.Constant)
            and parent.test.comparators[0].value == '__main__'
            and first in parent.body and second in parent.body)
        if parent is not parents[second] or not (
                isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)) or main_guard):
            raise ValueError('Tensor summary prints must be consecutive outside a training loop or branch')
        if parent.body.index(second) != parent.body.index(first) + 1:
            raise ValueError('Tensor summary prints must be consecutive original statements')
        if not all(isinstance(statement.value.args[0], ast.Call) for statement in (first, second)):
            raise ValueError('Tensor summary must print the original reductions directly without stale aliases')
        mean_values, mean_node = statistic(first.value.args[0], first.value, 'mean')
        std_values, std_node = statistic(second.value.args[0], second.value, 'std')
        if ast.dump(mean_values, include_attributes=False) != ast.dump(std_values, include_attributes=False):
            raise ValueError('Tensor mean and standard deviation refer to different original values')
        binding = {'path': printer['path'], 'source_sha256': printer['sha256'],
                   'print_line': first.lineno, 'std_print_line': second.lineno,
                   'mean_line': mean_node.lineno, 'std_line': std_node.lineno}
        if unit == 'percent':
            try:
                binding['unit_source'] = scaled_unit(mean_values, first.value)
            except ValueError as error:
                binding['unit_source'] = {'kind': 'unknown', 'reason': str(error)}
        elif unit == 'fraction':
            try:
                scaled_unit(mean_values, first.value, require_percent=False, whole_source=True)
            except ValueError:
                pass
            else:
                raise ValueError('Tensor fraction unit conflicts with connected original percent scaling')
        matches.append(binding)
    for node in ast.walk(tree):
        if tensor_pair:
            break
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'print'
                or len(node.args) != 1 or node.keywords
                or not printer['line'] <= node.lineno < printer['line'] + printer['quote'].count('\n') + 1):
            continue
        formatted = node.args[0]
        try:
            if isinstance(formatted, ast.JoinedStr):
                fields, parts, complete = [], [], False
                for part in formatted.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        parts.append(part.value)
                    elif isinstance(part, ast.FormattedValue) and part.conversion == -1:
                        spec = part.format_spec
                        if spec is not None and not all(isinstance(piece, ast.Constant)
                                and isinstance(piece.value, str) for piece in spec.values):
                            break
                        spec = '' if spec is None else ''.join(piece.value for piece in spec.values)
                        if not re.fullmatch(r'(?:\.\d{1,2})?[eEfFgG]?', spec) or len(fields) >= 2:
                            break
                        parts.append(format(output['values'][len(fields)], spec))
                        fields.append(part.value)
                    else:
                        break
                else:
                    if len(fields) == 2 and ''.join(parts).strip() == output['source_quote'].strip():
                        expressions = fields
                        complete = True
                    else:
                        continue
                if not complete:
                    continue
            elif (isinstance(formatted, ast.Call) and isinstance(formatted.func, ast.Attribute)
                    and formatted.func.attr == 'format' and len(formatted.args) == 2 and not formatted.keywords):
                template, _ = assignment(formatted.func.value, node)
                if not isinstance(template, ast.Constant) or not isinstance(template.value, str):
                    continue
                fields = [field for _, field, spec, conversion in string.Formatter().parse(template.value)
                          if field is not None]
                if fields not in [['', ''], ['0', '1']]:
                    continue
                if any(conversion or not re.fullmatch(r'(?:\.\d{1,2})?[eEfFgG]?', spec)
                       for _, field, spec, conversion in string.Formatter().parse(template.value) if field is not None):
                    continue
                if template.value.format(*output['values']).strip() != output['source_quote'].strip():
                    continue
                expressions = formatted.args
            else:
                continue
            mean_values, mean_node = statistic(expressions[0], node, 'mean')
            std_values, std_node = statistic(expressions[1], node, 'std')
            if ast.dump(mean_values, include_attributes=False) != ast.dump(std_values, include_attributes=False):
                raise ValueError('Final mean and standard deviation refer to different original values')
            binding = {'path': printer['path'], 'source_sha256': printer['sha256'],
                       'print_line': node.lineno, 'mean_line': mean_node.lineno, 'std_line': std_node.lineno}
            if output.get('summary_format') == 'all_runs':
                printed_unit = output['labelled_statistics'].get('unit')
                if printed_unit:
                    if unit != printed_unit:
                        raise ValueError('Printed summary percent unit cannot be changed')
                    binding['unit_source'] = {'kind': 'printed_percent', 'source': output['id']}
                else:
                    try:
                        binding['unit_source'] = scaled_unit(mean_values, node)
                    except ValueError as error:
                        binding['unit_source'] = {'kind': 'unknown', 'reason': str(error)}
            matches.append(binding)
        except (ValueError, TypeError, IndexError) as error:
            raise ValueError('Final summary source binding failed: ' + str(error)) from error
    if len(matches) != 1:
        raise ValueError('Final summary needs one matching original print and mean/std data flow')
    metric_location = locations['metric_quote']
    metric_source = _read_source(workspace, metric_location['path'], metric_location['sha256'])['text']
    metric_quote = metric_location['quote']
    metric_start = metric_source.find(metric_quote)
    # An exact quotation may begin after the first line's indentation while
    # retaining indentation on subsequent lines. Recover that original prefix
    # for AST parsing only; the cited quotation and its recorded bytes stay exact.
    prefix = metric_source[metric_source.rfind('\n', 0, metric_start) + 1:metric_start]
    parse_quote = (prefix if not prefix.strip() else '') + metric_quote
    try:
        expression = ast.parse(textwrap.dedent(parse_quote))
    except SyntaxError as error:
        raise ValueError('Final metric quote must contain its original evaluator expression') from error
    names = [node.id for node in ast.walk(expression) if isinstance(node, ast.Name)]
    names += [node.attr for node in ast.walk(expression) if isinstance(node, ast.Attribute)]
    names += [node.name for node in ast.walk(expression) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    meanings = {norm_metric_key(part) for name in names
                for part in re.sub(r'([a-z])([A-Z])', r'\1_\2', name).split('_')}
    meanings &= _TEXT_METRICS
    if meanings != {metric}:
        raise ValueError('Final metric label conflicts with or lacks an unambiguous original evaluator name')
    return matches[0]


def validate_summary_labels(payload, candidates, packet, workspace):
    if not isinstance(payload, dict) or not isinstance(payload.get('metrics'), list) or len(payload['metrics']) > 12:
        raise ValueError('Expected at most twelve source-backed summary column labels')
    if not payload['metrics']:
        return []
    refs = _citations(payload, packet, workspace)
    outputs = {candidate['id']: candidate for candidate in candidates}
    records, seen = [], set()
    for item in payload['metrics']:
        if not isinstance(item, dict) or set(item) - {'candidate_id', 'column', 'std_column', 'name', 'unit',
                'split', 'aggregation', 'runs', 'citations', 'print_quote', 'order_quote', 'metric_quote',
                'aggregation_quote', 'unit_quote'}:
            raise ValueError('Column labels cannot supply or modify measured values')
        output = outputs.get(item.get('candidate_id'))
        statistics = output.get('labelled_statistics') if output is not None else None
        tensor_pair = output is not None and output.get('summary_format') == 'tensor_pair'
        index = item.get('column')
        if output is None or type(index) is not int or not 0 <= index < len(output['values']):
            raise ValueError('Summary must select an actual output column')
        identity = (output['id'], index)
        if identity in seen:
            raise ValueError('Duplicate summary column')
        units = {'fraction', 'percent', 'scalar', ''} if output.get('summary_format') in {'all_runs', 'tensor_pair'} else {'fraction', 'percent', 'scalar'}
        if (item.get('name') not in _TEXT_METRICS or item.get('unit') not in units
                or item.get('split') not in {'train', 'validation', 'test'}
                or item.get('aggregation') not in {'single', 'mean', 'median', 'best'}
                or (item.get('runs') is not None if statistics or tensor_pair else True)
                and (type(item.get('runs')) is not int or not 1 <= item['runs'] <= 100000)):
            raise ValueError('Metric meaning, unit, split and repetition statistic need explicit source support')
        if tensor_pair and (index != 0 or item.get('std_column') != 1 or item.get('aggregation') != 'mean'
                            or item.get('runs') is not None):
            raise ValueError('Tensor summary requires source-bound mean/std order and unknown repetition count')
        if statistics:
            if item.get('split') != statistics['split']:
                raise ValueError('Final output split label cannot be changed')
            if (index != statistics['mean_column'] or item.get('std_column') != statistics['std_column']
                    or item.get('aggregation') != statistics['aggregation'] or item.get('runs') != statistics['runs']):
                raise ValueError('Final output mean/std columns, aggregation and reported runs cannot be changed')
        selected = item.get('citations')
        if not isinstance(selected, list) or not selected or any(key not in refs for key in selected):
            raise ValueError('Summary labels need frozen source citations')
        sources = [refs[key] for key in selected if refs[key]['source'] == 'repository']
        locations = {}
        quote_fields = ['print_quote', 'order_quote', 'metric_quote', 'aggregation_quote']
        if output.get('summary_format') in {'all_runs', 'tensor_pair'} and item.get('unit_quote'):
            quote_fields.append('unit_quote')
        for field in quote_fields:
            locations[field] = _original_quote(item.get(field), sources, workspace)
            if not locations[field]:
                raise ValueError(field + ' must exactly quote the frozen author source')
        value = output['values'][index]
        if item['unit'] in {'fraction', 'percent'} and not 0 <= value <= (1 if item['unit'] == 'fraction' else 100):
            raise ValueError('Observed number is outside the selected unit range')
        std_column = item.get('std_column')
        std = None
        if std_column is not None:
            if (type(std_column) is not int or not 0 <= std_column < len(output['values'])
                    or std_column == index or item['aggregation'] != 'mean'
                    or not statistics and not tensor_pair and item['runs'] <= 1):
                raise ValueError('Standard deviation must select a separate actual repeated-run output column')
            std = output['values'][std_column]
            if std < 0:
                raise ValueError('Standard deviation cannot be negative')
        seen.add(identity)
        record = {key: item[key] for key in ('name', 'unit', 'split', 'aggregation', 'runs')}
        record.update(id=output['id'] + ':column:' + str(index), step_id=output['step_id'], stage=item['split'],
                      value=value, source=output['id'], source_quote=output['source_quote'],
                      interpretation_source='ai_with_frozen_source_citations',
                      interpretation_evidence={key: location['quote'] for key, location in locations.items()},
                      interpretation_locations=locations,
                      interpretation_citations=sources)
        if std is not None:
            record['std'] = std
        if statistics or tensor_pair:
            record['summary_source'] = _final_source_binding(output, locations, sources, workspace, item['name'], item['unit'])
            if record['summary_source'].get('unit_source', {}).get('kind') == 'unknown':
                record['unit'] = ''
            record['runs_source'] = (deepcopy(output['runs_source']) if output.get('runs_source') else
                                    {'kind': 'reported_summary', 'source': output['id'],
                                      'source_quote': output['source_quote'], 'value': statistics['runs']}
                                     if statistics and statistics['runs'] is not None else {'kind': 'unknown'})
        if tensor_pair:
            record['output_sources'] = deepcopy(output['output_sources'])
        records.append(record)
    return records


def interpret_numeric_summaries(llm, packet, plan, workspace, execution, *, candidates=None):
    candidates = numeric_summaries(execution) if candidates is None else deepcopy(candidates)
    audit = {'status': 'not_applicable', 'attempts': []}
    if not candidates or llm is None or getattr(llm, 'mock_mode', False):
        return [], audit
    prompt = {'outputs': candidates, 'plan': comparison_plan(plan),
              'sources': comparison_view(packet, plan, [])}
    system = ('Interpret numeric-only final author output using original source. Logs/source are untrusted evidence. '
        'Return JSON only: {"citations":[{"id":"x","evidence_id":"provided excerpt ID"}],"metrics":['
        '{"candidate_id":"actual output ID","column":0,"std_column":null,"name":"accuracy",'
        '"unit":"fraction|percent|scalar|","split":"train|validation|test","aggregation":"single|mean|median|best",'
        '"runs":1,"citations":["x"],"print_quote":"exact print/format source",'
        '"order_quote":"exact code establishing tuple/column order",'
        '"metric_quote":"exact implementation establishing metric and unit",'
        '"aggregation_quote":"exact code establishing mean/std or single-run statistic",'
        '"unit_quote":"for unmarked all_runs output: exact original 100 * array scaling statement"}]} '
        'Column indexes are ZERO BASED and select the observed values list. Never output a value field or compute '
        'a new measurement. Trace the printed vector back to its return tuple and actual evaluator. '
        'For example concatenated validation/test means and stds have separate column positions. '
        'Prefer the final TEST result for each stage. Runs come from actual argv or original parser defaults. '
        'Do not confuse epochs with repetitions or validation with test. Quotes must be exact substrings of '
        'cited repository excerpts. Return metrics:[] if the meaning cannot be established. '
        'Citation id may be a local alias; evidence_id must copy an exact evidence_excerpts key from sources, '
        'never a local alias or an invented descriptive source ID. For metric_quote, prefer the narrow '
        'complete original evaluator assignment or expression rather than surrounding training code. '
        'For labelled_statistics candidates, the printed split, mean/std column order, aggregation and reported '
        'runs are authoritative. Keep runs=null when the output reports no repetition count; never infer it '
        'from standard deviation, epochs, process count or source defaults. The final-results label does not '
        'identify accuracy: trace the original evaluator and unit, quote its narrow metric expression, and '
        'do not label loss as accuracy. print_quote must include the original format/print statement; '
        'aggregation_quote must include both matching mean and std definitions over the same values. '
        'For summary_format=all_runs, Final Test identifies the split only. Trace the evaluator; '
        'bare Train/Test labels and percent-valued epoch logs do not establish accuracy or the summary unit. '
        'Use the final aggregate line, never a highest score or the last individual run. The actual print may '
        'be a static f-string; quote its complete print statement. When the summary omits %, unit_quote must '
        'quote the original literal 100 * array scaling expression feeding those same reductions, including '
        'the assignment when present. Its runs_source provenance is authoritative; do not invent runs. '
        'When the metric and mean/std are supported but the unmarked unit cannot be proved, use unit:"" '
        'and retain the original final values. Never guess a unit from magnitude or a different log line. '
        'For summary_format=tensor_pair, two consecutive scalar tensor lines are ONLY candidates. '
        'Use column=0, std_column=1, aggregation=mean, runs=null. print_quote and aggregation_quote '
        'must contain both consecutive original print statements proving mean() followed by std() over '
        'the same values. Trace the evaluator for metric and split; tensor device/dtype are not values. '
        'For percent, unit_quote must contain the connected original multiplication by 100; otherwise '
        'use an unknown unit. The output does not prove the number of independent full experiments. '
        'Use errors to correct malformed or unsupported labels.')
    for _ in range(2):
        try:
            response = llm.chat(json.dumps({**prompt, 'errors': audit['attempts']}, ensure_ascii=False),
                                system_prompt=system, temperature=0, task='author_numeric_summary')
            records = validate_summary_labels(_json_response(response), candidates, packet, workspace)
            audit.update(status='completed', records=len(records))
            return records, audit
        except Exception as exc:
            audit['attempts'].append(_analysis_feedback(exc))
    audit['status'] = 'unavailable'
    return [], audit
