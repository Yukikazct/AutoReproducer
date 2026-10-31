"""Retain literal author numbers without assigning metric or protocol meaning."""
import re

from src.runtime_metrics import METRICS_MARKER, NUMBER_PATTERN, _REFERENCE_RE, metric_number


MAX_OBSERVATIONS = 128
MAX_DIAGNOSTICS = 16
_PAIR = re.compile(
    r'^[ \t]*(?P<row>[A-Za-z][A-Za-z0-9_ .+-]{0,60}?)[ \t]+'
    r'(?P<roc_label>ROC)[ \t]*:[ \t]*(?P<roc>' + NUMBER_PATTERN + r')[ \t]*,[ \t]*'
    r'(?P<precision_label>precision[ \t]*@[ \t]*rank[ \t]+n)[ \t]*:[ \t]*'
    r'(?P<precision>' + NUMBER_PATTERN + r')[ \t]*$', re.I)
_TIME = re.compile(
    r'^[ \t]*(?P<label>(?:[A-Za-z][A-Za-z0-9_+.-]{0,30}[ \t]+){1,4}time)'
    r'[ \t]*:[ \t]*(?P<value>' + NUMBER_PATTERN + r')[ \t]*$', re.I)


def collect_author_observations(execution):
    """Read only completed steps; never traverse historical attempts or infer units.

    These narrow printed forms do not establish ROC-AUC, unqualified precision,
    seconds, a principal model, a split, or repeated-run aggregation. Structured
    metrics remain authoritative and suppress this optional text observation.
    """
    observations, errors = [], []

    def error(message):
        if len(errors) < MAX_DIAGNOSTICS:
            errors.append(message)
        else:
            errors[-1] = '原始数值输出诊断超过保留上限，后续诊断已省略。'
    blocked = ('cancelled', 'timed_out', 'interrupted', 'cleanup_unconfirmed')
    if (not isinstance(execution, dict) or execution.get('success') is not True
            or any(execution.get(field) for field in blocked)):
        return observations, errors
    steps = execution.get('steps')
    if not isinstance(steps, list):
        return observations, errors
    for step in steps:
        if (not isinstance(step, dict) or step.get('success') is not True or step.get('exit_code') != 0
                or any(step.get(field) for field in blocked) or not isinstance(step.get('id'), str)):
            continue
        streams = {name: step.get(name, '') for name in ('stdout', 'stderr')}
        if (any(step.get(field) is not None for field in ('metric_records', 'metrics'))
                or any(isinstance(text, str) and any(line.strip().startswith(METRICS_MARKER)
                       for line in text.splitlines()) for text in streams.values())):
            continue
        for stream, text in streams.items():
            if not isinstance(text, str):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if len(line) > 512 or _REFERENCE_RE.search(line):
                    continue
                pair = _PAIR.fullmatch(line)
                timing = None if pair else _TIME.fullmatch(line)
                if pair:
                    values = [(pair['row'].strip(), pair[label], pair[value])
                              for label, value in (('roc_label', 'roc'), ('precision_label', 'precision'))]
                elif timing:
                    values = [('', timing['label'], timing['value'])]
                else:
                    continue
                source = f"{step['id']}:{stream}:{line_number}"
                if any(metric_number(value) is None for _, _, value in values):
                    error(source + ': 原始数值输出含非有限数值，整行未采纳。')
                    continue
                if len(observations) + len(values) > MAX_OBSERVATIONS:
                    error(source + ': 原始数值输出超过保留上限，后续观察值未采纳。')
                    return observations, errors
                for column, (row, label, value) in enumerate(values):
                    observations.append({'id': source + ':observation:' + str(column),
                        'step_id': step['id'], 'row_label': row, 'metric_label': label,
                        'value': metric_number(value), 'value_text': value, 'unit': '',
                        'interpretation_status': 'unresolved', 'source': source, 'source_quote': line})
    return observations, errors
