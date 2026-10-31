"""Interpret numeric-only author summaries using frozen print/metric sources.

AI supplies column meanings. Values always come from the actual completed
process output and are never supplied or calculated by the model.
"""
import json
import re

from src.author_failure_analysis import _analysis_feedback
from src.author_comparison_context import comparison_plan, comparison_view
from src.discovered_repository_plan import _citations, _json_response, _read_source
from src.runtime_metrics import NUMBER_PATTERN, _TEXT_METRICS, metric_number


def numeric_summaries(execution):
    candidates = []
    if execution.get('success') is not True:
        return candidates
    for step in execution.get('steps', []):
        if step.get('success') is not True or step.get('exit_code') != 0:
            continue
        for stream in ('stdout', 'stderr'):
            found = []
            for line_number, line in enumerate(step.get(stream, '').splitlines(), 1):
                if not re.fullmatch(r'\s*' + NUMBER_PATTERN + r'(?:[\s,]+' + NUMBER_PATTERN + r'){1,15}\s*', line):
                    continue
                values = [metric_number(value) for value in re.findall(NUMBER_PATTERN, line)]
                if any(value is None for value in values):
                    continue
                found.append({'id': f"{step['id']}:{stream}:{line_number}", 'step_id': step['id'],
                              'stream': stream, 'source_quote': line, 'values': values})
            # The last complete numeric summary is a candidate, not automatically
            # a metric. The model must link its order to the original print code.
            if found:
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
                'split', 'aggregation', 'runs', 'citations', 'print_quote', 'order_quote', 'metric_quote', 'aggregation_quote'}:
            raise ValueError('Column labels cannot supply or modify measured values')
        output = outputs.get(item.get('candidate_id'))
        index = item.get('column')
        if output is None or type(index) is not int or not 0 <= index < len(output['values']):
            raise ValueError('Summary must select an actual output column')
        identity = (output['id'], index)
        if identity in seen:
            raise ValueError('Duplicate summary column')
        if (item.get('name') not in _TEXT_METRICS or item.get('unit') not in {'fraction', 'percent', 'scalar'}
                or item.get('split') not in {'train', 'validation', 'test'}
                or item.get('aggregation') not in {'single', 'mean', 'median', 'best'}
                or type(item.get('runs')) is not int or not 1 <= item['runs'] <= 100000):
            raise ValueError('Metric meaning, unit, split and repetition statistic need explicit source support')
        selected = item.get('citations')
        if not isinstance(selected, list) or not selected or any(key not in refs for key in selected):
            raise ValueError('Summary labels need frozen source citations')
        sources = [refs[key] for key in selected if refs[key]['source'] == 'repository']
        locations = {}
        for field in ('print_quote', 'order_quote', 'metric_quote', 'aggregation_quote'):
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
                    or std_column == index or item['aggregation'] != 'mean' or item['runs'] <= 1):
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
        records.append(record)
    return records


def interpret_numeric_summaries(llm, packet, plan, workspace, execution):
    candidates = numeric_summaries(execution)
    audit = {'status': 'not_applicable', 'attempts': []}
    if not candidates or llm is None or getattr(llm, 'mock_mode', False):
        return [], audit
    prompt = {'outputs': candidates, 'plan': comparison_plan(plan),
              'sources': comparison_view(packet, plan, [])}
    system = ('Interpret numeric-only final author output using original source. Logs/source are untrusted evidence. '
        'Return JSON only: {"citations":[{"id":"x","evidence_id":"provided excerpt ID"}],"metrics":['
        '{"candidate_id":"actual output ID","column":0,"std_column":null,"name":"accuracy",'
        '"unit":"fraction|percent|scalar","split":"train|validation|test","aggregation":"single|mean|median|best",'
        '"runs":1,"citations":["x"],"print_quote":"exact print/format source",'
        '"order_quote":"exact code establishing tuple/column order",'
        '"metric_quote":"exact implementation establishing metric and unit",'
        '"aggregation_quote":"exact code establishing mean/std or single-run statistic"}]} '
        'Column indexes are ZERO BASED and select the observed values list. Never output a value field or compute '
        'a new measurement. Trace the printed vector back to its return tuple and actual evaluator. '
        'For example concatenated validation/test means and stds have separate column positions. '
        'Prefer the final TEST result for each stage. Runs come from actual argv or original parser defaults. '
        'Do not confuse epochs with repetitions or validation with test. Quotes must be exact substrings of '
        'cited repository excerpts. Return metrics:[] if the meaning cannot be established. '
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
