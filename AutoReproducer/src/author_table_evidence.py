"""Conservative, source-only locations for quoted paper table values.

The caller validates PDF identity/hashes and passes unchanged same-page text.
Column selectors describe a requested position, never prove experiment settings.
"""
from __future__ import annotations

import re

from src.metric_keys import norm_metric_key
from src.runtime_metrics import NUMBER_PATTERN, metric_number

_ANCHORS = {'method', 'methods', 'model', 'models', 'dataset', 'datasets'}
_MAX_COLUMNS = 64
_MAX_LEVELS = 6
_MAX_QUOTE = 12000
_MAX_PAGE = 2 * 1024 * 1024
_MISSING = re.compile(r'(?:N/A|NA|nan|[-—–])(?=$|\s|[,|)])', re.I)
_SCALAR = re.compile(r'(?P<value>' + NUMBER_PATTERN + r')\s*(?P<percent>%)?'
                     r'(?:\s*(?:±|\+/-|\x06)\s*(?P<std>' + NUMBER_PATTERN + r')\s*(?P<std_percent>%)?)?'
                     r'(?P<footnote>[†‡*]+)?'
                     r'(?=$|\s|[,|)])', re.I)
_METRIC_LABEL = re.compile(r'(?:'
    r'(?:macro|micro|weighted)[_\s-]?f1(?:[_\s-]?score)?|'
    r'f1(?:[_\s-]?score)?(?:[_\s-](?:macro|micro|weighted))?|'
    r'roc[_\s-]?auc|auc|acc(?:uracy)?|precision|recall|mse|rmse|mae|loss)', re.I)


def _label(value):
    """Fold typography, preserving punctuation that may distinguish variants."""
    return re.sub(r'\s+', ' ', str(value or '').strip()).casefold()


def _result(status, reason, **evidence):
    return {'status': status, 'reason': reason, **evidence}


def _span(text, quote):
    start = text.find(quote)
    if start < 0:
        return None, 'missing'
    if text.find(quote, start + 1) >= 0:
        return None, 'ambiguous'
    return [start, start + len(quote)], None


def _caption_scope(reference, text, row, header):
    """Resolve repetition only inside an exact, uniquely numbered table caption.

    Offsets stay relative to the complete original page. A model-proposed value
    never chooses a table, and another table/figure/section ends the scope.
    """
    caption = reference.get('metric_quote')
    if (not isinstance(caption, str) or len(caption) > _MAX_QUOTE
            or not re.match(r'^Table[ \t]+\d+[A-Za-z]?[ \t]*[:.][ \t]+\S', caption, re.I)):
        return None
    caption_span, error = _span(text, caption)
    if error or text[text.rfind('\n', 0, caption_span[0]) + 1:caption_span[0]].strip():
        return None
    boundary = re.search(r'(?im)^[ \t]*(?:Table|Figure|Fig\.|Section)[ \t]+\d+(?=[\s.:A-Za-z])',
                         text[caption_span[0] + 1:])
    end = caption_span[0] + 1 + boundary.start() if boundary else len(text)
    if caption_span[1] >= end:
        return None
    start = caption_span[1]
    scoped_row, row_error = _span(text[start:end], row)
    scoped_header, header_error = _span(text[start:end], header)
    if row_error or header_error:
        return None
    return ([offset + start for offset in scoped_row],
            [offset + start for offset in scoped_header],
            {'caption_quote': caption, 'caption_span': caption_span,
             'scope_span': [caption_span[0], end], 'basis': 'unique_original_table_caption'})


def _whole_lines(text, span):
    start, end = span
    return (not text[text.rfind('\n', 0, start) + 1:start].strip()
            and not text[end:text.find('\n', end) if '\n' in text[end:] else len(text)].strip())


def _markdown_separator_width(line):
    if '|' not in line:
        return None
    cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
    if len(cells) >= 2 and all(re.fullmatch(r':?-{3,}:?', cell) for cell in cells):
        return len(cells)
    return None


def _header_levels(header):
    levels = []
    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        width = _markdown_separator_width(line)
        if width is not None:
            if not levels or width not in {len(levels[-1]), len(levels[-1]) + 1}:
                return None
            continue
        if '|' in line:
            columns = [part.strip() for part in line.strip('|').split('|')]
            if columns and not columns[0]:
                columns = columns[1:]  # Explicit empty method-header corner.
        elif '\t' in line or re.search(r'\S {2,}\S', line):
            columns = [part.strip() for part in re.split(r'\t+| {2,}', line)]
        else:
            columns = line.split()
        anchors = [i for i, word in enumerate(columns) if _label(word) in _ANCHORS]
        if anchors:
            columns = columns[anchors[-1] + 1:]
        if not columns:
            continue
        # Empty delimited cells, punctuation-only cells and prose are not
        # inferred as implicit spans or guessed dataset names.
        if (len(columns) > _MAX_COLUMNS or any(not part or not re.search(r'[\w]', part) for part in columns)
                or any(re.search(r'[.!?;:]$', part) for part in columns)):
            return None
        levels.append(columns)
    if not levels or len(levels) > _MAX_LEVELS:
        return None
    return levels


def _paths(levels):
    paths = [[label] for label in levels[0]]
    rules = ['original_header_order']
    for children in levels[1:]:
        previous = len(paths)
        if len(children) == previous:
            paths = [path + [child] for path, child in zip(paths, children)]
            rules.append('explicit_per_column_header')
            continue
        if len(children) < previous or len(children) % previous:
            return None, None
        width = len(children) // previous
        pattern = children[:width]
        # Only a fully repeated, nonduplicate child pattern establishes a
        # uniform subdivision in text. Omitted or uneven spans stay unknown.
        if (len({_label(x) for x in pattern}) != width
                or any([_label(x) for x in children[i:i+width]] != [_label(x) for x in pattern]
                       for i in range(0, len(children), width))):
            return None, None
        paths = [path + [children[i * width + j]]
                 for i, path in enumerate(paths) for j in range(width)]
        rules.append('complete_repeated_child_pattern')
    if len(paths) > _MAX_COLUMNS:
        return None, None
    return paths, rules


def _one_cell(text, start):
    missing = _MISSING.match(text, start)
    if missing:
        return {'missing': True, 'value': None, 'std': None,
                'span': [start, missing.end()]}, missing.end()
    match = _SCALAR.match(text, start)
    if match:
        value = metric_number(match['value'])
        std = metric_number(match['std'])
        if value is None or match['std'] is not None and (std is None or std < 0):
            return None, start
        return {'missing': False, 'value': value, 'std': std,
                'percent': bool(match['percent']), 'std_percent': bool(match['std_percent']),
                'footnote': match['footnote'],
                'span': [start, match.end()]}, match.end()
    if start < len(text) and text[start] == '(':
        end = text.find(')', start + 1)
        if end < 0 or '(' in text[start + 1:end]:
            return None, start
        pieces = text[start + 1:end].split(',')
        if len(pieces) < 2:
            return None, start  # (std), confidence intervals, etc. are ambiguous.
        parts, offset = [], start + 1
        for piece in pieces:
            leading = len(piece) - len(piece.lstrip())
            part, stop = _one_cell(text, offset + leading)
            if part is None or 'parts' in part or text[stop:offset + len(piece)].strip():
                return None, start
            parts.append(part)
            offset += len(piece) + 1
        return {'parts': parts, 'missing': False, 'span': [start, end + 1]}, end + 1
    return None, start


def _cells(text, *, allow_prefix=True):
    """Parse every result position; unknown text after the first cell is fatal."""
    delimited_width = None
    if '|' in text:
        parts = text.split('|')
        if not parts[0].strip():
            parts = parts[1:]  # The method-to-result separator is outside cells.
        if parts and not parts[-1].strip():
            parts = parts[:-1]  # Optional closing table delimiter.
        if not parts or any(not part.strip() for part in parts):
            return None, None  # Never discard an empty interior cell and shift.
        delimited_width = len(parts)
    cells, position = [], 0
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == '|'):
            position += 1
        if position == len(text):
            break
        cell, stop = _one_cell(text, position)
        if cell is None:
            if cells or not allow_prefix:
                return None, None
            word = re.match(r'[^\s|()]+', text[position:])
            if (word is None or not re.fullmatch(r'[^\W\d][\w./+-]*', word.group())
                    or word.group().casefold() in {'inf', 'infinity', 'nan'}):
                return None, None
            position += len(word.group())
            continue
        if len(cells) >= _MAX_COLUMNS:
            return None, None
        cells.append(cell)
        position = stop
    if delimited_width is not None and len(cells) != delimited_width:
        return None, None
    return (cells, cells[0]['span'][0]) if cells else (None, None)


def _method_tail(method, row):
    method = str(method or '')
    if not method:
        return None
    # A model variant cannot be shortened to another row label. The method
    # must start the quoted row and end before whitespace or a cell delimiter.
    pattern = (r'^\s*\|?\s*' + r'\s+'.join(re.escape(x) for x in method.split())
               + r'(?:\s+\((?:ours|our method|this work)\))?(?=$|\s|\|)')
    match = re.match(pattern, row, re.I)
    if match and re.match(r'\s*\([A-Za-z]', row[match.end():]):
        return None  # An unselected named parenthetical variant is not metadata.
    return match.end() if match else None


def dataset_row_header(reference, row, header):
    """Recognize the other common orientation from explicit original labels.

    This establishes labels only. Full positional verification still requires
    the complete header, cells and their order in ``locate_table_value``.
    """
    if not isinstance(header, str) or not isinstance(row, str):
        return False
    method = reference.get('method')
    return bool(isinstance(method, str) and method.strip()
        and re.match(r'^\s*\|?\s*datasets?\b', header, re.I)
        and _method_tail(reference.get('dataset'), row) is not None
        and re.search(r'(?<![\w+.-])' + re.escape(method) + r'(?![\w+-]|\.[\w])', header, re.I))


def _metric_name(label):
    value = re.sub(r'\s*\((?:percent|percentage|%)\)\s*$', '', str(label), flags=re.I).strip()
    value = value.rstrip('%').strip()
    return norm_metric_key(value) if _METRIC_LABEL.fullmatch(value) else None


def _known_row_prefix(reference, prefix):
    """Only explicit dataset and evaluation-split labels are row metadata."""
    labels = {str(reference.get('dataset') or '')}
    selected = str(reference.get('split') or '').casefold()
    splits = {'test': ['test', 'testing'], 'train': ['train', 'training'],
              'validation': ['validation', 'valid', 'val']}
    labels.update(splits.get(selected, [label for group in splits.values() for label in group]))
    for label in sorted(filter(None, labels), key=len, reverse=True):
        prefix = re.sub(r'(?<![\w.-])' + re.escape(label) + r'(?![\w.-])', ' ', prefix, flags=re.I)
    return not prefix.strip(' \t\r\n|')


def _select(reference, paths, row_prefix):
    dataset = _label(reference.get('dataset'))
    selected_paths = [i for i, path in enumerate(paths) if any(_label(x) == dataset for x in path)]
    inline_dataset = bool(dataset and re.search(r'(?<![\w.-])' + re.escape(str(reference['dataset']))
                                               + r'(?![\w.-])', row_prefix, re.I))
    if not selected_paths:
        if not inline_dataset:
            return None, 'Dataset has no unique header or row label'
        metric = norm_metric_key(reference.get('name'))
        selected_paths = [i for i, path in enumerate(paths) if any(_metric_name(x) == metric for x in path)]
    else:
        metric_labels = {i: [_metric_name(x) for x in paths[i] if _label(x) != dataset]
                         for i in selected_paths}
        if any(any(names) for names in metric_labels.values()):
            metric = norm_metric_key(reference.get('name'))
            selected_paths = [i for i in selected_paths if metric in metric_labels[i]]
    labels = reference.get('column_labels')
    subgroup = reference.get('column_subgroup')
    if labels is not None:
        if (not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels)):
            return None, 'column_labels must be a nonempty original header path'
        selected_paths = [i for i in selected_paths if [_label(x) for x in paths[i]] == [_label(x) for x in labels]]
    if subgroup is not None:
        if not isinstance(subgroup, str) or not subgroup.strip():
            return None, 'column_subgroup must name an original header label'
        selected_paths = [i for i in selected_paths if any(_label(x) == _label(subgroup) for x in paths[i][1:])]
    if len(selected_paths) != 1:
        return None, 'Requested dataset/metric/header path is missing or ambiguous'
    return selected_paths[0], None


def _between_rows(text, expected):
    if len(text) > 80000 or len(text.splitlines()) > 100:
        return False
    for line in text.splitlines():
        if not line.strip():
            continue
        if re.match(r'^\s*(?:table|figure|section)\b', line, re.I):
            return False
        width = _markdown_separator_width(line)
        if width is not None:
            if width != expected + 1:
                return False
            continue
        if '|' in line:
            columns = [column.strip() for column in line.strip().strip('|').split('|')]
            if len(columns) != expected + 1 or not re.search(r'[^\W\d]', columns[0]):
                return False
            if (columns[0].endswith(':') and not re.search(r'\d', columns[0])
                    and not any(columns[1:])
                    and not re.match(r'^(?:table|figure|section)\b', columns[0], re.I)):
                continue  # An explicit section row preserves the same columns.
            cells, _ = _cells('|'.join(columns[1:]), allow_prefix=False)
            if cells is None or len(cells) != expected:
                return False
            continue
        cells, prefix = _cells(line)
        if cells is None or len(cells) != expected or prefix == 0:
            return False
    return True


def _tuple_metric(reference, cells, page_text, header_start):
    caption = reference.get('metric_quote')
    if not isinstance(caption, str) or not caption.strip():
        return None, 'unknown', 'Tuple cells require an exact ordered metric caption'
    if len(caption) > _MAX_QUOTE or caption not in page_text:
        return None, 'rejected', 'Tuple metric caption is not exact same-page evidence'
    # A later table's caption or unrelated distant text cannot select a tuple
    # component in this table. The caller also validates the cited PDF hash.
    caption_span, caption_error = _span(page_text, caption)
    if (caption_error or caption_span[1] > header_start
            or page_text[caption_span[1]:header_start].strip()):
        return None, 'unknown', 'Tuple metric caption is not bound to this table'
    orders = []
    for group in re.findall(r'\(([^()]*)\)', caption):
        names = [_metric_name(x.strip()) for x in group.split(',')]
        if len(names) >= 2 and all(names) and len(set(names)) == len(names):
            orders.append(names)
    if not orders or any(order != orders[0] for order in orders):
        return None, 'unknown', 'Tuple metric order is absent or ambiguous'
    metric = norm_metric_key(reference.get('name'))
    if metric not in orders[0]:
        return None, 'unknown', 'Requested metric is absent from the tuple caption'
    widths = {len(cell['parts']) for cell in cells if 'parts' in cell}
    if widths != {len(orders[0])} or any('parts' not in cell and not cell.get('missing') for cell in cells):
        return None, 'unknown', 'Tuple width does not match the ordered metric caption'
    return orders[0].index(metric), None, None


def locate_table_value(reference, quote, header, *, page_text=None):
    """Locate a requested column in unchanged, already hash-validated PDF text.

    ``column_labels`` selects a complete original header path, e.g.
    ``['dataset', 'Fixed']``; ``column_subgroup`` selects one explicit child.
    Neither the proposed number nor ``column_verified`` selects a position.
    ``verified`` certifies text position only, never independent reproduction.
    Every offset is zero-based; ``cell_span`` is relative to ``quote``.
    Unsupported/ambiguous structures return ``unknown``. Exact-location
    contradictions and selected missing cells return ``rejected``.
    """
    if not isinstance(reference, dict):
        return _result('rejected', 'Reference must be an object')
    if metric_number(reference.get('value')) is None:
        return _result('rejected', 'Reference value must be a finite number')
    if reference.get('std') is not None:
        deviation = metric_number(reference['std'])
        if deviation is None or deviation < 0:
            return _result('rejected', 'Reference standard deviation must be finite and nonnegative')
    if not all(isinstance(x, str) and x.strip() for x in (quote, header)):
        return _result('unknown', 'An exact complete row and header are required')
    if not isinstance(page_text, str) or not page_text:
        return _result('unknown', 'Same-page text is required to establish complete rows and headers')
    if len(quote) > _MAX_QUOTE or len(header) > _MAX_QUOTE or len(page_text) > _MAX_PAGE:
        return _result('unknown', 'Table evidence exceeds the bounded text budget')
    row_span, row_error = _span(page_text, quote)
    header_span, header_error = _span(page_text, header)
    table_scope = None
    if row_error == 'missing' or header_error == 'missing':
        return _result('rejected', 'Row/header is not exact same-page evidence')
    if row_error or header_error:
        scoped = _caption_scope(reference, page_text, quote, header)
        if scoped is None:
            return _result('unknown', 'Repeated row/header text cannot identify a unique table')
        row_span, header_span, table_scope = scoped
    if not _whole_lines(page_text, row_span) or not _whole_lines(page_text, header_span):
        return _result('unknown', 'A truncated physical row or header cannot establish column positions')
    if header_span[1] >= row_span[0]:
        return _result('unknown', 'The complete header must precede its method row')
    transposed = dataset_row_header(reference, quote, header)
    selected_reference = dict(reference)
    if transposed:
        selected_reference.update(method=reference['dataset'], dataset=reference['method'])
    start = _method_tail(selected_reference.get('method'), quote)
    if start is None:
        return _result('rejected', 'Quoted method row does not match the requested method variant')
    cells, first_cell = _cells(quote[start:])
    if cells is None:
        return _result('unknown', 'The complete result row contains unsupported or unparsed cells')
    if not _known_row_prefix(selected_reference, quote[start:start + first_cell]):
        return _result('unknown', 'Unparsed row labels may identify another method or dataset')
    for cell in cells:
        for part in [cell, *cell.get('parts', [])]:
            part['span'] = [x + start for x in part['span']]
    levels = _header_levels(header)
    paths, rules = _paths(levels) if levels else (None, None)
    if paths is None or len(cells) != len(paths):
        return _result('unknown', 'Complete header/row widths do not establish a unique table layout')
    if not _between_rows(page_text[header_span[1]:row_span[0]], len(paths)):
        return _result('unknown', 'Header is incomplete or separated from the row by another table/prose')
    index, selection_error = _select(selected_reference, paths, quote[start:start + first_cell])
    if index is None:
        return _result('unknown', selection_error)
    selected = cells[index]
    evidence = {'column_index': index, 'column_labels': paths[index], 'column_count': len(paths),
                'orientation': 'datasets_in_rows' if transposed else 'methods_in_rows',
                'row_span': row_span, 'header_span': header_span, 'basis': rules,
                'cell_span': selected['span'], 'cell_quote': quote[slice(*selected['span'])]}
    if table_scope is not None:
        evidence['table_scope'] = table_scope
        evidence['basis'].append(table_scope['basis'])
    if selected.get('footnote'):
        return _result('unknown', 'Selected value has a footnote requiring its original condition', **evidence)
    if selected.get('missing'):
        return _result('rejected', 'Quoted dataset/metric column has no reported value', **evidence)
    if 'parts' in selected:
        metric_index, status, reason = _tuple_metric(reference, cells, page_text, header_span[0])
        if metric_index is None:
            return _result(status, reason, **evidence)
        selected = selected['parts'][metric_index]
        evidence.update(metric_index=metric_index, metric_span=selected['span'],
                        metric_quote=quote[slice(*selected['span'])])
        evidence['basis'].append('exact_ordered_tuple_metric_caption')
        if selected.get('missing'):
            return _result('rejected', 'Quoted dataset/metric column has no reported value', **evidence)
    elif any('parts' in cell for cell in cells):
        return _result('unknown', 'Mixed scalar and tuple columns do not establish the metric position', **evidence)
    evidence.update(value=selected['value'], std=selected['std'])
    proposed = metric_number(reference.get('value'))
    if proposed is not None and proposed != selected['value']:
        return _result('rejected', 'Reference value does not belong to the quoted dataset/metric column', **evidence)
    if reference.get('std') is not None and metric_number(reference['std']) != selected['std']:
        return _result('rejected', 'Reference standard deviation does not belong to the quoted dataset/metric column', **evidence)
    return _result('verified', 'Complete same-page header and row identify one explicit value position', **evidence)
