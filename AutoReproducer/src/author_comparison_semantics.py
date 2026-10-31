"""Conservative semantic checks on already hash-verified comparison quotations.

These checks bind units, splits and repetition statistics to their meanings.
They do not certify that the executed experiment implements a paper protocol.
"""
import ast
import re

from src.metric_keys import norm_metric_key
from src.evidence_quotes import original_quote_spans


_SPLITS = {
    'test': r'\b(?:test(?:ing)?|held[- ]out)\b|测试',
    'validation': r'\b(?:validation|valid|val)\b|验证',
    'train': r'\btrain(?:ing)?\b|训练',
}
_COUNT_WORDS = {word: number for number, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split())}
for _tens, _word in enumerate('twenty thirty forty fifty sixty seventy eighty ninety'.split(), 2):
    _COUNT_WORDS[_word] = _tens * 10
    for _one in range(1, 10):
        _unit = next(word for word, number in _COUNT_WORDS.items() if number == _one)
        _COUNT_WORDS[_word + '-' + _unit] = _tens * 10 + _one
        _COUNT_WORDS[_word + ' ' + _unit] = _tens * 10 + _one
_COUNT = r'(?:[1-9]\d{0,2}(?:,\d{3})+|\d+|' + '|'.join(re.escape(word) for word in sorted(_COUNT_WORDS, key=len, reverse=True)) + ')'
_RUNS = re.compile(
    r'(?<![\w.,+-])(?:(?P<before>' + _COUNT + r')\s+(?:independent\s+|random\s+|repeated\s+)?'
    r'(?:runs|trials|seeds|repetitions|repeats)|'
    r'(?P<label>runs|trials|seeds|repetitions|repeats)\s*(?:=|:)\s*(?P<after>' + _COUNT + r'))\b', re.I)
_SINGLE = re.compile(r'(?<![\w.,+-])(?:a\s+single|single|one|1)\s+(?:run|trial|seed)\b', re.I)
_COUNT_LIST_SUFFIX = re.compile(
    r'(?:\s*[,;:/&.±−–—-]\s*|\s+(?:and|or|to|through)\s+|\s+)[+-]?\d', re.I)
_EXPERIMENT_TIMES = re.compile(
    r'\b(?:(?:run|ran|repeat(?:ed)?|conduct(?:ed)?|perform(?:ed)?)\s+'
    r'(?:each|every|the)\s+experiments?\s+|'
    r'(?:each|every|the)\s+experiment\s+(?:is|was|has\s+been)\s+'
    r'(?:run|repeated|conducted|performed)\s+)'
    r'(?P<count>' + _COUNT + r')\s+times\b'
    r'(?!\s+(?:faster|slower|larger|smaller|as)\b)', re.I)
_AGGREGATIONS = {
    'mean': r'\b(?:mean|average|averaged|avg)\b|均值|平均',
    'median': r'\bmedian\b|中位数',
    'best': r'\bbest\s+(?:of|over|across)\s+\d+\s+(?:(?:independent|random)\s+)?(?:runs|trials|seeds)\b',
    'single': _SINGLE.pattern,
}
_METRICS = re.compile(r'(?<![\w])(?:accuracy|acc|f1(?:[_ -]?(?:score|macro|micro|weighted))?|'
                      r'(?:macro|micro|weighted)[_ -]?f1|auc|mse|rmse|mae|loss|precision|recall|'
                      r'psnr|ssim|bleu|perplexity)(?![\w])', re.I)


def _snippet(text, match, limit=400):
    """Keep the actual supporting line, never a model-written explanation."""
    start = text.rfind('\n', 0, match.start()) + 1
    end = text.find('\n', match.end())
    # Keep a visibly wrapped word with its continuation. Otherwise a PDF line
    # ending in "mean clas-" loses the result-statistic context on the next
    # line. The evidence remains the unchanged original substring.
    while (end >= 0 and end - start < limit
            and re.search(r'[A-Za-z]{2}-[ \t]*$', text[start:end])
            and re.match(r'[ \t]*[A-Za-z]{2}', text[end + 1:])):
        end = text.find('\n', end + 1)
    end = end if end >= 0 else len(text)
    if match.end() - start > limit:
        start = max(start, match.start() - limit // 2)
    # Never discard the phrase that supports the field. A whitespace-wrapped
    # match can exceed the normal excerpt budget and must remain verbatim.
    end = max(match.end(), min(end, start + limit))
    return text[start:end]


def _support(status, *, value=None, quote='', source=None, reason=''):
    result = {'status': status}
    if value is not None:
        result['value'] = value
    if quote:
        result['quote'] = quote
    if source is not None:
        result['citation'] = source
    if reason:
        result['reason'] = reason
    return result


def _scopes(node, refs, field):
    selected = node.get('context_quotes') or {}
    if not isinstance(selected, dict):
        raise ValueError('context_quotes must be a mapping of fields to exact original quotations')
    quote = selected.get(field)
    if quote is not None:
        if not isinstance(quote, str) or not 4 <= len(quote) <= 1200:
            raise ValueError('Comparison context quotation must be a bounded exact string')
        recovered = original_quote_spans(quote, ((ref['quote'], ref['id']) for ref in refs))
        if not recovered:
            recovered = original_quote_spans(quote, ((ref['quote'], ref['id']) for ref in refs
                                                     if ref.get('source') == 'pdf'), pdf_wrapping=True)
        if not recovered:
            raise ValueError(f'context_quotes.{field} must occur in its own original citations; '
                             'copy an exact original span with its source citation, or omit this optional '
                             'context field when unavailable. Proposed quotation: ' + repr(quote[:160]))
        return recovered
    if field == 'split' and node.get('value_quote'):
        return [(node['value_quote'], ref['id']) for ref in refs if node['value_quote'] in ref['quote']]
    return [(ref['quote'], ref['id']) for ref in refs]


def _exact_repetition_context(text, start, end, *, number_follows_label=False, preceding_statistic=False):
    """A bound, estimate, range or negated plan is not an exact run count."""
    if re.search(r'\bapprox\.\s*$', text[:start], re.I):
        return False
    prefix = re.split(r'[.!?;]', text[:start])[-1]
    suffix = text[end:]
    if re.search(r"\b(?:no|not|never|neither|without|would|could|should|can|might|may|will|"
                 r"plan(?:ned)?|intend(?:ed)?)\b|\b\w+n['’]t\b", prefix, re.I):
        return False
    if not (number_follows_label or preceding_statistic) and re.search(
            r'\d(?:,\s*|\s+)$|\b(?:hundred|thousand|million|billion)\s*$', prefix, re.I):
        return False  # A trailing number fragment is not the complete count.
    if number_follows_label and _COUNT_LIST_SUFFIX.match(suffix):
        return False
    if re.match(r'\s*(?:in\s+(?:total|all)\s+)?'
                r'(?:(?:is|are|was|were|has|have|had|be|been|being)\s+)*'
                r"(?:not|never|will|would|could|should|can|might|may|planned|proposed|"
                r"intended|required|recommended|expected|sufficient|necessary|hypothetical|"
                r"suggested|\w+n['’]t)\b", suffix, re.I):
        return False
    if re.search(r'\b(?:at\s+(?:least|most)|up\s+to|more\s+than|less\s+than|fewer\s+than|'
                 r'approximately|about|around|roughly|nearly|under)\s*$|'
                 r'\b(?:use[ds]?|perform(?:ed)?|conduct(?:ed)?)\s+over\s*$|'
                 r'(?:\b(?:to|through|and|or)|[/−–—-]|[~≈≤≥<>])\s*$', prefix, re.I):
        return False
    return not re.match(r'\s*(?:[+~≈≤≥<>]|or\s+(?:more|less|fewer)\b|'
                        r'at\s+(?:most|least)\b|maximum\b|minimum\b|'
                        r'faster\b|slower\b|larger\b|smaller\b|as\b|if\b|unless\b|'
                        r'per\s+(?:second|minute|hour|epoch|batch|iteration|step)\b)', suffix, re.I)


def _exact_repetition(text, source, match, refs, source_texts=None):
    # Verify both crop boundaries: the selected context within its citation,
    # and that citation within the hash-checked original page or source file.
    found = False
    for ref in refs:
        if ref['id'] != source:
            continue
        cited = ref['quote']
        original = (source_texts or {}).get(source, cited)
        citation_offset = 0
        while (citation_offset := original.find(cited, citation_offset)) >= 0:
            offset = 0
            while (offset := cited.find(text, offset)) >= 0:
                found = True
                start = citation_offset + offset + match.start()
                end = citation_offset + offset + match.end()
                original_match = match.re.match(original, start)
                if original_match is None or original_match.end() != end:
                    return False  # Cropping cannot create a number/word boundary.
                if original_match.groupdict().get('after') and _COUNT_LIST_SUFFIX.match(original[end:]):
                    return False
                if not _exact_repetition_context(original, start, end):
                    return False
                offset += 1
            citation_offset += 1
    return found


def _repetitions(node, refs, source_texts=None):
    proposed = node.get('runs')
    if proposed is None:
        return _support('unknown', reason='No repetition count was specified')
    candidates = []
    for text, source in _scopes(node, refs, 'runs'):
        for match in _RUNS.finditer(text):
            if not _exact_repetition(text, source, match, refs, source_texts):
                continue
            if (match['label'] and match['label'].casefold() == 'seeds'
                    and not re.search(r'\b(?:number|count)\s+of\s*$', text[:match.start()], re.I)):
                continue  # Even a single listed seed may identify an ID, not a count.
            literal = (match['before'] or match['after']).casefold()
            numeric = literal.replace(',', '')
            count = int(numeric) if numeric.isdigit() else _COUNT_WORDS[literal]
            candidates.append((count, _snippet(text, match), source))
        for match in _EXPERIMENT_TIMES.finditer(text):
            if _exact_repetition(text, source, match, refs, source_texts):
                literal = match['count'].casefold()
                numeric = literal.replace(',', '')
                count = int(numeric) if numeric.isdigit() else _COUNT_WORDS[literal]
                candidates.append((count, _snippet(text, match), source))
        for match in _SINGLE.finditer(text):
            if _exact_repetition(text, source, match, refs, source_texts):
                candidates.append((1, _snippet(text, match), source))
    values = {item[0] for item in candidates}
    if len(values) == 1 and proposed in values:
        value, quote, source = candidates[0]
        return _support('verified', value=value, quote=quote, source=source)
    if len(values) == 1:
        return _support('contradicted', value=next(iter(values)), reason='Original repetition count differs')
    return _support('unknown', reason='Repetitions need an explicit, unambiguous runs/trials/seeds statement')


def repetition_count_supported(node, refs, *, source_texts=None):
    """Support an explicitly scoped count, including English number words."""
    return _repetitions(node, refs, source_texts)['status'] == 'verified'


def _aggregation(node, refs):
    proposed = node.get('aggregation')
    if not proposed or proposed == 'unknown':
        return _support('unknown', reason='No aggregation was specified')
    candidates = []
    for text, source in _scopes(node, refs, 'aggregation'):
        for kind, expression in _AGGREGATIONS.items():
            for match in re.finditer(expression, text, re.I):
                line = _snippet(text, match)
                # Optimizer/model-selection "best" and feature normalization
                # means are not repetition statistics.
                if (kind in {'mean', 'median'} and not (
                        _RUNS.search(line) or _METRICS.search(line)
                        or re.search(r'\b(?:results?|scores?|report(?:s|ed|ing)?)\b', line, re.I))):
                    continue
                if (kind in {'mean', 'median'} and re.search(r'\.\s*(?:mean|median)\s*\(', line)
                        and not _RUNS.search(line)):
                    continue
                candidates.append((kind, line, source))
    values = {item[0] for item in candidates}
    # A single trial can still be printed by a mean reduction. That does not
    # prove the statistical intent of an unlabeled summary.
    if len(values) == 1 and proposed in values:
        value, quote, source = candidates[0]
        return _support('verified', value=value, quote=quote, source=source)
    if len(values) == 1:
        return _support('contradicted', value=next(iter(values)), reason='Original aggregation differs')
    return _support('unknown', reason='Aggregation is not uniquely supported by original result-statistics text')


def _split(node, refs):
    proposed = {'val': 'validation', 'valid': 'validation', 'training': 'train'}.get(
        str(node.get('split', '')).lower(), str(node.get('split', '')).lower())
    candidates = []
    for text, source in _scopes(node, refs, 'split'):
        for kind, pattern in _SPLITS.items():
            for match in re.finditer(pattern, text, re.I):
                candidates.append((kind, _snippet(text, match), source))
    values = {item[0] for item in candidates}
    if len(values) == 1 and proposed in values:
        value, quote, source = candidates[0]
        return _support('verified', value=value, quote=quote, source=source)
    if len(values) == 1:
        return _support('contradicted', value=next(iter(values)), reason='Original result split differs')
    return _support('unknown', reason='Selected result split is not uniquely identified')


def _unit(node, refs):
    proposed = node.get('unit')
    expected = norm_metric_key(node.get('name') or node.get('metric'))
    candidates = []
    for text, source in _scopes(node, refs, 'unit'):
        for line in text.splitlines():
            labels = list(_METRICS.finditer(line))
            if not (labels or re.search(r'\b(?:results?|scores?)\b', line, re.I)):
                continue
            scopes = [line]
            if labels:
                keys = [norm_metric_key(label.group()) for label in labels]
                if expected not in keys:
                    continue  # Another metric's units cannot establish this one.
                if len(set(keys)) > 1:
                    # In a mixed header, keep only the selected metric's own
                    # following segment. A preceding metric's (%) is unrelated.
                    scopes = [line[label.start():labels[i + 1].start() if i + 1 < len(labels) else len(line)]
                              for i, label in enumerate(labels) if keys[i] == expected]
            for scope in scopes:
                # A prose improvement of 5% is not a table's measurement unit.
                if re.search(r'\(\s*%\s*\)|\bpercent(?:age)?\b|\baccuracy\s*\(%\)', scope, re.I):
                    candidates.append(('percent', scope[:400], source))
                if re.search(r'\b(?:fraction|ratio)\b|\b(?:range|scale)\s+(?:of\s+)?[\[(]?0\s*[-,]\s*1', scope, re.I):
                    candidates.append(('fraction', scope[:400], source))
    values = {item[0] for item in candidates}
    if len(values) == 1 and proposed in values:
        value, quote, source = candidates[0]
        return _support('verified', value=value, quote=quote, source=source)
    if len(values) == 1:
        return _support('contradicted', value=next(iter(values)), reason='Original metric unit differs')
    if not values:
        inferred = _implementation_units(node, refs)
        if inferred is not None:
            return inferred
    return _support('unknown', reason='Metric unit needs an explicit result/metric statement or observed output')


def _implementation_units(node, refs):
    """Recognize dimensionless accuracy arithmetic, never numerical magnitude."""
    expected = norm_metric_key(node.get('name') or node.get('metric'))
    candidates = []
    scopes, complete_sources = [], set()
    # A selected expression must not hide a later rewrite in the same cited
    # source. Prefer parseable complete quotations over a shorter code span.
    for ref in refs:
        try:
            tree = ast.parse(ref['quote'])
        except (SyntaxError, ValueError, TypeError):
            continue
        scopes.append((ref['quote'], ref['id'], tree))
        complete_sources.add(ref['id'])
    for text, source in _scopes(node, refs, 'unit'):
        if source in complete_sources:
            continue
        try:
            scopes.append((text, source, ast.parse(text)))
        except (SyntaxError, ValueError, TypeError):
            continue
    functions = {'accuracy_score', 'f1_score', 'roc_auc_score', 'precision_score', 'recall_score'}
    def qualified(expr):
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.insert(0, expr.attr)
            expr = expr.value
        return tuple([expr.id, *parts]) if isinstance(expr, ast.Name) else ()
    for text, source, tree in scopes:
        metric_functions = set()
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom) and item.level == 0 and item.module == 'sklearn.metrics':
                for alias in item.names:
                    if alias.name in functions:
                        metric_functions.add((alias.asname or alias.name,))
            elif isinstance(item, ast.ImportFrom) and item.level == 0 and item.module == 'sklearn':
                for alias in item.names:
                    if alias.name == 'metrics':
                        metric_functions.update((alias.asname or alias.name, fn) for fn in functions)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    if alias.name == 'sklearn':
                        metric_functions.update((alias.asname or 'sklearn', 'metrics', fn) for fn in functions)
                    elif alias.name == 'sklearn.metrics':
                        prefix = (alias.asname,) if alias.asname else ('sklearn', 'metrics')
                        metric_functions.update((*prefix, fn) for fn in functions)
        rebound = {item.id for item in ast.walk(tree) if isinstance(item, ast.Name)
                   and isinstance(item.ctx, (ast.Store, ast.Del))}
        rebound.update(item.arg for item in ast.walk(tree) if isinstance(item, ast.arg))
        rebound.update(item.name for item in ast.walk(tree)
                       if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        metric_functions = {path for path in metric_functions if path[0] not in rebound}
        for scope in [tree, *(item for item in ast.walk(tree) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))]:
            assignments = {}
            def boolean(expr, active):
                if isinstance(expr, ast.Name):
                    return expr.id not in active and expr.id in assignments and boolean(assignments[expr.id], active | {expr.id})
                if isinstance(expr, ast.Compare):
                    return True
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                    if expr.func.attr in {'eq', 'ne', 'gt', 'lt', 'ge', 'le'}:
                        return len(expr.args) == 1
                    if expr.func.attr in {'float', 'double', 'bool', 'int', 'long', 'to', 'item', 'cpu', 'detach'}:
                        return boolean(expr.func.value, active)
                return False
            def count(expr):
                if isinstance(expr, ast.Call):
                    if isinstance(expr.func, ast.Name) and expr.func.id == 'len' and len(expr.args) == 1:
                        return True
                    if isinstance(expr.func, ast.Attribute):
                        if expr.func.attr in {'item', 'float', 'double'} and not expr.args:
                            return count(expr.func.value)
                        return expr.func.attr == 'sum' and not expr.args
                return False
            def unit(expr, active=frozenset()):
                if isinstance(expr, ast.Name):
                    return unit(assignments[expr.id], active | {expr.id}) if expr.id not in active and expr.id in assignments else None
                if isinstance(expr, ast.BinOp):
                    if isinstance(expr.op, ast.Mult):
                        for scalar, operand in ((expr.left, expr.right), (expr.right, expr.left)):
                            if isinstance(scalar, ast.Constant) and scalar.value == 100 and unit(operand, active) == 'fraction':
                                return 'percent'
                    if isinstance(expr.op, ast.Div) and count(expr.right):
                        numerator = expr.left
                        while (isinstance(numerator, ast.Call) and isinstance(numerator.func, ast.Attribute)
                               and numerator.func.attr in {'item', 'float', 'double'} and not numerator.args):
                            numerator = numerator.func.value
                        if (isinstance(numerator, ast.Call) and isinstance(numerator.func, ast.Attribute)
                                and numerator.func.attr == 'sum' and not numerator.args
                                and boolean(numerator.func.value, set())):
                            return 'fraction'
                if isinstance(expr, ast.Call):
                    if qualified(expr.func) in metric_functions:
                        if not any(kw.arg == 'normalize' and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in expr.keywords):
                            return 'fraction'
                    if isinstance(expr.func, ast.Attribute):
                        if expr.func.attr in {'item', 'float', 'double', 'cpu', 'detach'} and not expr.args:
                            return unit(expr.func.value, active)
                        if expr.func.attr == 'mean' and not expr.args and boolean(expr.func.value, set()):
                            return 'fraction'
                return None
            # Keep lexical scopes separate; unrelated helper bindings must not
            # become definitions for the selected metric's implementation.
            def statements(body):
                for item in body:
                    yield item
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        for field in ('body', 'orelse', 'finalbody'):
                            yield from statements(getattr(item, field, []))
            for item in statements(scope.body):
                if not isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    continue
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    inferred = None if isinstance(item, ast.AugAssign) else unit(item.value)
                    key = re.sub(r'^(?:test|val|validation|train)_', '', target.id)
                    if expected in {norm_metric_key(key), norm_metric_key(key.removesuffix('_score'))}:
                        candidates.append((inferred, (ast.get_source_segment(text, item) or '')[:400], source))
                    if isinstance(item, ast.AugAssign):
                        assignments.pop(target.id, None)
                    else:
                        assignments[target.id] = item.value
    values = {item[0] for item in candidates}
    if len(values) != 1 or None in values:
        return None
    value, quote, source = candidates[0]
    return _support('verified' if node.get('unit') == value else 'contradicted',
                    value=value, quote=quote, source=source, reason='Static original metric arithmetic')


def context_evidence(node, refs, *, observed=None, source_texts=None):
    """Return derived field evidence, ignoring any model-provided status flags."""
    checks = {'unit': _unit(node, refs), 'split': _split(node, refs),
              'aggregation': _aggregation(node, refs), 'runs': _repetitions(node, refs, source_texts)}
    if observed is not None:
        for field in checks:
            value = observed.get(field)
            if value not in (None, '', 'unknown'):
                provenance = observed.get('runs_source') if field == 'runs' else None
                if (isinstance(provenance, dict) and provenance.get('kind') == 'reported_summary'
                        and provenance.get('value') == value):
                    checks[field] = _support('verified', value=value,
                        quote=provenance.get('source_quote', ''), source=provenance.get('source'),
                        reason='Run count reported in the adjacent author summary header; not independently counted')
                else:
                    checks[field] = _support('verified', value=value, quote=observed.get('source_quote', ''),
                                             source=observed.get('source'), reason='Authoritative observed metric field')
    return checks


def reference_uncertainty(node, refs):
    """Retain explicit non-SD uncertainty without trusting a model's label."""
    contexts = node.get('context_quotes') or {}
    selected = contexts.get('uncertainty') if isinstance(contexts, dict) else None
    if selected is not None:
        scopes = _scopes(node, refs, 'uncertainty')
    else:
        scopes = [(text, ref['id']) for ref in refs for text in
                  (node.get('metric_quote', ''), node.get('value_quote', '')) if text and text in ref['quote']]
        expected = norm_metric_key(node.get('name') or node.get('metric'))
        for field in ('unit', 'aggregation'):
            if not isinstance(contexts, dict) or contexts.get(field) is None:
                continue
            for text, source in _scopes(node, refs, field):
                # A selected metric caption can define both units/statistics
                # and uncertainty. Require the same cited result row and
                # metric; unrelated captions remain outside this inference.
                if (expected and expected in {norm_metric_key(m.group()) for m in _METRICS.finditer(text)}
                        and node.get('value_quote') and any(ref['id'] == source
                            and node['value_quote'] in ref['quote'] for ref in refs)):
                    scopes.append((text, source))
    matches = []
    for text, source in scopes:
        sem = re.search(r'\b(?:sem|stderr|standard\s+error)\b', text, re.I)
        ci = re.search(r'\b(?:ci|confidence\s+interval)\b', text, re.I)
        if sem or ci:
            kind = 'standard_error' if sem else 'confidence_interval'
            confidence = re.search(r'\b(\d+(?:\.\d+)?)\s*%\s*(?:ci|confidence\s+interval)\b', text, re.I)
            level = float(confidence[1]) if confidence and 0 < float(confidence[1]) < 100 else None
            matches.append((kind, level, _snippet(text, sem or ci, limit=800), source))
    if not matches or len({(item[0], item[1]) for item in matches}) != 1:
        return None
    kind, level, quote, source = matches[0]
    result = {'uncertainty_kind': kind, 'uncertainty_source': {'quote': quote, 'citation': source}}
    if level is not None:
        result['confidence_level'] = level
    return result


def settings_evidence(settings, citations):
    """Distinguish source-backed model assessment from protocol verification."""
    refs = [citations[key] for key in settings.get('citations', []) if key in citations]
    support = {}
    for kind in ('pdf', 'repository'):
        matched = [ref for ref in refs if ref['source'] == kind and re.search(
            r'\b(?:split|epochs?|seeds?|runs|trials|learning[- ]rate|hyperparameters?|'
            r'preprocess(?:ing)?|validation|held[- ]out|test\s+(?:set|split|samples)|'
            r'(?:single|one|1)\s+run)\b', ref['quote'], re.I)]
        support[kind] = [ref['id'] for ref in matched]
    return {'assessment_basis': 'model_with_citations', 'independently_verified': False,
            'protocol_context_present': all(support.values()), 'protocol_context_citations': support}
