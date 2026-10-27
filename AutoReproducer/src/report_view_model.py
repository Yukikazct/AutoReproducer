"""One deterministic presentation contract for live and saved experiment evidence.

Building this model is local and side-effect free. Saved derived summaries are
ignored; execution, comparability and independent verification retain their
separate authorities. Missing observations are represented by None, never zero.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import re

from src.metric_keys import norm_metric_key
from src.result_conclusion import build_reproduction_conclusion
from src.execution_safety import cleanup_is_unconfirmed


def _map(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _value(value):
    value = value.get('value') if isinstance(value, dict) else value
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _count(node):
    for key in ('runs', 'run_count', 'n_runs', 'count', 'repetitions'):
        value = node.get(key)
        if isinstance(value, dict):
            value = value.get('count')
        if type(value) is int and value > 0:
            return value
    return None


def _evidence_references(value, fallback):
    references = []
    def collect(node):
        if isinstance(node, str):
            if len(node) <= 180 and '\n' not in node:
                references.append(node)
        elif isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            for key, item in node.items():
                if key in {'id', 'evidence_id', 'source_id', 'path', 'source', 'artifact_path',
                           'execution_dir', 'run_dir', 'sha256', 'source_sha256', 'plan_sha256'}:
                    collect(item)
                elif isinstance(item, (dict, list)):
                    collect(item)
    collect(value)
    return list(dict.fromkeys(references))[:6] or [fallback]


def _metric_rows(data):
    validation = _map(data.get('validation'))
    comparison = _map(validation.get('author_claim_comparison'))
    rows = []
    for item in _rows(comparison.get('comparisons')):
        reference, actual = _map(item.get('reference')), _map(item.get('actual'))
        state = item.get('status') or 'unrecorded'
        rows.append({'metric': item.get('metric') or item.get('name') or '未记录',
                     'reference': _value(item.get('reference')), 'actual': _value(item.get('actual')),
                     'reference_unit': reference.get('unit') or item.get('reference_unit') or item.get('unit'),
                     'actual_unit': actual.get('unit') or item.get('actual_unit') or item.get('unit'),
                     'reference_split': reference.get('split') or item.get('reference_split'),
                     'actual_split': actual.get('split') or item.get('actual_split'),
                     'reference_runs': _count(reference), 'actual_runs': _count(actual),
                     'difference': _value(item.get('difference')) if state == 'comparable' else None,
                     'difference_unit': item.get('difference_unit'),
                     'comparability': state, 'reason': item.get('reason') or
                         _map(item.get('settings')).get('reason'),
                     'evidence': ['validation.author_claim_comparison']})
    if rows:
        return rows
    # Legacy validated results use a different schema; a name match alone does
    # not establish that two numbers are scientifically comparable.
    metrics = _map(validation.get('metrics_comparison'))
    paper, actual = _map(metrics.get('paper')), _map(metrics.get('actual'))
    paired = {}
    for side, values in (('reference', paper), ('actual', actual)):
        for name, value in values.items():
            paired.setdefault(norm_metric_key(name), {'metric': name})[side] = _value(value)
            node = _map(value)
            row = paired[norm_metric_key(name)]
            units = _map(metrics.get('paper_units' if side == 'reference' else 'actual_units'))
            row[side + '_unit'] = node.get('unit') or units.get(name)
            row[side + '_split'] = node.get('split')
            row[side + '_runs'] = _count(node)
    for record in _rows(validation.get('metric_records')):
        name = record.get('name') or record.get('metric')
        if not name:
            continue
        row = paired.setdefault(norm_metric_key(name), {'metric': name})
        if 'actual' not in row:
            row.update(actual=_value(record), actual_unit=record.get('unit'),
                       actual_split=record.get('split'), actual_runs=_count(record))
    plan = _map(data.get('grounded_repository_plan'))
    for metric in _rows(plan.get('metrics')):
        name = metric.get('name')
        if name:
            row = paired.setdefault(norm_metric_key(name), {'metric': name})
            if 'reference' not in row:
                row.update(reference=_value(metric.get('reference')), reference_unit=metric.get('unit'),
                           reference_split=metric.get('split'), reference_runs=_count(metric))
    for name, value in _map(_map(data.get('paper_info')).get('metrics')).items():
        row = paired.setdefault(norm_metric_key(name), {'metric': name})
        if 'reference' not in row:
            row.update(reference=_value(value), reference_unit=_map(value).get('unit'),
                       reference_split=_map(value).get('split'), reference_runs=_count(_map(value)))
    verified = {}
    for item in _rows(validation.get('comparisons')):
        name = item.get('name') or item.get('metric')
        if not name:
            continue
        key = norm_metric_key(name)
        verified[key] = item
        row = paired.setdefault(key, {'metric': name})
        for side in ('reference', 'actual'):
            node = _map(item.get(side))
            if side in item:
                row[side] = _value(item.get(side))
            for suffix, value in (('unit', node.get('unit') or item.get(side + '_unit') or item.get('unit')),
                                  ('split', node.get('split') or item.get(side + '_split') or item.get('split')),
                                  ('runs', _count(node) or item.get(side + '_runs'))):
                if value is not None:
                    row[side + '_' + suffix] = value
    for key, row in paired.items():
        item = verified.get(key, {})
        established = type(item.get('pass')) is bool
        row.update(comparability='comparable' if established else 'unrecorded',
                   difference=_value(item.get('difference')) if established else None,
                   difference_unit=item.get('difference_unit') or item.get('unit'),
                   reason=item.get('reason'), evidence=['validation.comparisons', 'validation.metrics_comparison',
                                                       'validation.metric_records'])
        row.setdefault('reference', None)
        row.setdefault('actual', None)
        if established:
            row['reference'] = row['reference'] if row['reference'] is not None else _value(item.get('reference'))
            row['actual'] = row['actual'] if row['actual'] is not None else _value(item.get('actual'))
        if (established and row.get('difference') is None and _number(row.get('reference'))
                and _number(row.get('actual')) and row.get('reference_unit') == row.get('actual_unit')):
            row['difference'] = row['actual'] - row['reference']
    return list(paired.values())


def _contributions(data):
    recovery = _map(data.get('intelligent_recovery'))
    actions = _rows(recovery.get('actions'))
    rows = []
    for action in actions:
        execution, verification = _map(action.get('execution')), _map(action.get('verification'))
        # "verified" text or an acknowledgement is insufficient by itself.
        executed = execution.get('status') in {'completed', 'succeeded', 'failed', 'executed', 'stopped',
                                               'success', 'error', 'cancelled', 'timed_out'}
        result = _map(execution.get('result'))
        stopped = (action.get('status') == 'stopped' or execution.get('status') in
                   {'stopped', 'cancelled', 'timed_out'} or result.get('cancelled') is True
                   or result.get('timed_out') is True or cleanup_is_unconfirmed([execution, result]))
        no_progress = action.get('status') == 'no_progress'
        rejected = action.get('status') == 'rejected'
        failed = action.get('status') == 'failed' or execution.get('status') in {'failed', 'error'}
        verified = executed and not (stopped or no_progress or rejected or failed) and verification.get('success') is True
        status = ('stopped' if stopped else 'no_progress' if no_progress else 'rejected' if rejected else
                  'verified' if verified else 'failed' if executed and (failed or verification.get('success') is False)
                  else 'executed' if executed else action.get('status') if action.get('status') in
                  {'rejected', 'stopped', 'no_progress', 'executing'} else 'proposed')
        rows.append({'id': action.get('id'), 'phase': action.get('phase'),
                     'problem': action.get('problem'), 'role': action.get('role'),
                     'action': action.get('kind') or action.get('action'),
                     'evidence': _evidence_references(action.get('evidence'),
                         'intelligent_recovery.actions.' + str(action.get('id') or len(rows)) + '.evidence'),
                     'review': deepcopy(_map(action.get('review'))),
                     'status': status,
                     'executed': executed, 'verified': verified,
                     'rerun': execution.get('original_command_rerun'),
                     'result': execution.get('error') or verification.get('reason') or execution.get('status'),
                     'verification': deepcopy(verification)})
    if not actions:
        # Historical decisions have no execution link. Preserve them as advice,
        # even if a decision's status says "accepted" or "resolved".
        for task in _rows(recovery.get('tasks')):
            for decision in _rows(task.get('decisions')):
                rows.append({'id': task.get('id'), 'phase': task.get('phase') or task.get('trigger'),
                             'problem': task.get('trigger'), 'role': decision.get('role') or 'Investigator',
                             'action': decision.get('action'), 'evidence': deepcopy(decision.get('evidence_ids') or []),
                             'review': {}, 'status': 'rejected' if decision.get('status') == 'rejected' else 'proposed',
                             'executed': False, 'verified': False, 'rerun': None,
                             'result': decision.get('reason'), 'verification': {}})
    for review in _rows(recovery.get('reviews')):
        if review.get('action') == 'reject':
            rows.append({'id': review.get('candidate_sha256'), 'phase': review.get('stage'),
                         'problem': '候选补丁审查', 'role': review.get('reviewer'),
                         'action': 'apply_compatibility_patch', 'evidence': deepcopy(review.get('evidence_ids') or []),
                         'review': deepcopy(review), 'status': 'rejected', 'executed': False,
                         'verified': False, 'rerun': None, 'result': review.get('reason'), 'verification': {}})
    return rows


@dataclass
class ReportViewModel:
    version: int
    conclusion: dict
    answers: dict
    metrics: list
    recovery_contributions: list
    recovery_recorded: bool
    recovery_usage: dict
    collaboration: dict
    experiment: dict

    @classmethod
    def from_data(cls, data):
        from src.author_claim_comparison import _redact_evidence
        from src.report_revisions import sanitize_report_result
        data = _map(_redact_evidence(sanitize_report_result(data), ''))
        conclusion = build_reproduction_conclusion(data)
        contributions = _contributions(data)
        recovery = _map(data.get('intelligent_recovery'))
        verified = sum(row['verified'] for row in contributions)
        executed = sum(row['executed'] for row in contributions)
        recorded = isinstance(data.get('intelligent_recovery'), dict)
        repair = ('未记录' if not recorded else '本次未启用智能恢复' if recovery.get('enabled') is False else
                  f'实际执行 {executed} 项恢复动作，其中 {verified} 项已验证效果；论文指标单独核验。')
        pending = [row.get('check') for row in conclusion['next_steps'] if row.get('check')]
        pending += [str(row['problem']) for row in _rows(recovery.get('unresolved_issues'))
                    if row.get('status') != 'resolved' and row.get('problem')]
        if recovery.get('stop_reason'):
            pending.append('恢复停止原因：' + str(recovery['stop_reason']))
        pending = list(dict.fromkeys(pending))
        experiment = {key: deepcopy(data.get(key) or {}) for key in
                      ('paper_info', 'experiment_spec', 'experiment_protocol', 'grounded_repository_plan')}
        collaboration = _map(data.get('agent_collaboration'))
        messages = _rows(collaboration.get('messages'))
        participants = sorted({str(message[key]) for message in messages for key in ('sender', 'recipient')
                               if message.get(key)})
        collaboration = {'messages_recorded': len(messages), 'participants': participants,
                         'dropped_messages': collaboration.get('dropped_messages') or 0,
                         'actions_executed': executed, 'actions_verified': verified}
        return cls(6, conclusion, {'completion': conclusion['execution']['label'],
                   'reference': conclusion['comparison']['label'],
                   'independent': conclusion['independent']['label'], 'recovery': repair,
                   'remaining': pending or ['当前记录中没有待处理项。']}, _metric_rows(data), contributions,
                   recorded, deepcopy(_map(recovery.get('usage'))), collaboration, experiment)

    def to_dict(self):
        return asdict(self)


def build_report_view_model(data):
    """Return a fresh JSON-compatible view, ignoring saved presentation fields."""
    return ReportViewModel.from_data(data).to_dict()


def report_overview_markdown(view):
    """The same four answers for initial rendering and late lifecycle updates."""
    answers = view.answers if isinstance(view, ReportViewModel) else view['answers']
    def text(value):
        value = re.sub(r'\s+', ' ', str(value)).strip()
        return re.sub(r'([\\`*_{}\[\]()<>!|])', r'\\\1', value)
    return '\n'.join(['## 任务结果与恢复', '',
        '- **完成到哪一步**: ' + text(answers['completion']),
        '- **是否达到参考**: ' + text(answers['reference']),
        '- **独立核验状态**: ' + text(answers['independent']),
        '- **自动修复解决了什么**: ' + text(answers['recovery']),
        '- **仍需处理什么**: ' + text('；'.join(answers['remaining']))])
