"""Summarize existing execution and validation evidence without new acceptance rules.

This presentation layer never runs an experiment or asks a model for a verdict.
In particular, author-output comparisons and a successful command do not supply
the protocol and independent-prediction evidence required for paper acceptance.
"""
from src.author_claim_comparison import _redact_text


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _text(value, limit=700):
    return _redact_text(value, '')[:limit] if isinstance(value, str) else ''


def _rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _current_records(execution):
    history = _rows(execution.get('steps') or execution.get('stages'))
    final = (_mapping(execution.get('final')) or (history[-1] if history else {})
             or _mapping(execution.get('execution')))
    if final.get('repair_round') is not None:
        history = [row for row in history if row.get('repair_round') == final['repair_round']]
    current = {}
    for index, row in enumerate(history):
        key = row.get('id') or row.get('stage')
        current[key if isinstance(key, str) else index] = row
    return [execution, final, _mapping(execution.get('execution')),
            *[row for row in current.values() if row.get('required') is not False]]


def _execution_level(data, validation, execution):
    records = [data, validation, *_current_records(execution),
               _mapping(execution.get('effective_env_config')),
               _mapping(data.get('intelligent_recovery'))]
    for key, label in (('cancelled', '已取消'), ('interrupted', '已中断'), ('timed_out', '已超时')):
        code = 124 if key == 'timed_out' else 130
        if any(row.get(key) is True or row.get('status') == key
               or row.get('exit_code') == code
               or (key == 'timed_out' and row.get('dependency_failure_kind') == 'installation_timeout')
               for row in records):
            return {'status': 'timed_out' if key == 'timed_out' else 'interrupted',
                    'label': label, 'reason': '保留已发生的执行记录；未完成部分不作完成判定。'}
    for row in records:
        cleanup = [_mapping(row.get(key)) for key in ('cleanup', 'process_cleanup')]
        if (any(row.get(key) for key in ('cleanup_unconfirmed', 'cleanup_failed', 'cleanup_error', 'cleanup_errors'))
                or row.get('cleanup_confirmed') is False
                or row.get('status') in {'cleanup_unconfirmed', 'cleanup_failed'}
                or any(item.get('confirmed') is False or item.get('status') in
                       {'unconfirmed', 'cleanup_unconfirmed', 'failed', 'cleanup_failed'} for item in cleanup)):
            return {'status': 'cleanup_unconfirmed', 'label': '进程清理未确认',
                    'reason': '所属进程清理尚未确认，不能发布本次执行完成或独立验收结论。'}
    if any(row.get('process_exited') is False or row.get('process_alive') is True
           or row.get('running') is True for row in records):
        return {'status': 'incomplete', 'label': '执行尚未结束',
                'reason': '仍有进程运行或未退出记录，不能沿用先前的完成状态。'}
    status = validation.get('status')
    if status == 'mock_demo' or execution.get('mock_mode') is True:
        return {'status': 'mock_demo', 'label': '模拟流程', 'reason': '本记录没有执行真实论文实验。'}
    if status in {'prepared', 'environment_prepared', 'environment_ready', 'needs_pdf',
                  'insufficient_evidence', 'not_runnable', 'analysis_failed'}:
        return {'status': 'not_run', 'label': '尚未执行实验',
                'reason': _text(validation.get('reason')) or '当前仅有准备或前置证据检查记录。'}
    if status == 'smoke_passed' or validation.get('execution_status') == 'smoke_passed':
        return {'status': 'smoke_passed', 'label': '仅冒烟通过', 'reason': '尚无完整实验完成记录。'}
    if status == 'execution_incomplete' or validation.get('execution_status') == 'incomplete':
        return {'status': 'incomplete', 'label': '完成证据不足', 'reason': '权威核验记录未确认完整执行。'}
    if execution.get('executed') is False and status != 'execution_failed':
        return {'status': 'not_run', 'label': '尚未执行实验', 'reason': '执行记录明确标注本次未运行。'}
    # Reuse the existing executor-level interpretation, including required
    # steps and replaced repair rounds, rather than deriving success from logs.
    from src.agents.result_validator import ResultValidatorAgent
    try:
        observed, _ = ResultValidatorAgent._execution_state(execution)
    except (AttributeError, TypeError, ValueError):
        observed = 'incomplete'  # Malformed historical evidence cannot supply completion.
    if status == 'execution_failed' or validation.get('execution_status') == 'failed' or observed == 'failed':
        return {'status': 'failed', 'label': '执行失败',
                'reason': _text(validation.get('reason')) or '至少一个当前必需执行步骤未正常结束。'}
    if observed == 'smoke_passed':
        return {'status': 'smoke_passed', 'label': '仅冒烟通过', 'reason': '最终记录仅覆盖冒烟检查。'}
    if observed == 'experiment_completed':
        return {'status': 'completed', 'label': '已完成运行',
                'reason': '最终阶段和已记录的必需步骤均正常结束；训练及论文验收另行判断。'}
    return {'status': 'incomplete', 'label': '完成证据不足',
            'reason': '缺少最终正常结束或完整必需步骤记录；不据此判定论文数值失败。'}


def _comparison_level(validation):
    comparison = _mapping(validation.get('author_claim_comparison'))
    rows = _rows(comparison.get('comparisons'))
    counts = {'total': len(rows), 'comparable': 0, 'met': 0, 'not_met': 0,
              'reference_only': 0, 'not_comparable': 0, 'unknown': 0}
    if comparison:
        for row in rows:
            state = row.get('status')
            if state == 'comparable':
                counts['comparable'] += 1
                verdict = row.get('meets_reference')
                verdict = verdict if type(verdict) is bool else row.get('within_tolerance')
                counts['met' if verdict is True else 'not_met' if verdict is False else 'unknown'] += 1
            else:
                counts[state if state in {'reference_only', 'not_comparable'} else 'unknown'] += 1
        complete = (bool(rows) and comparison.get('status') == 'compared'
                    and counts['comparable'] == counts['total'] and not counts['unknown']
                    and not comparison.get('validation_errors'))
        state = ('not_met' if counts['not_met'] else 'met') if complete else (
            'partial' if counts['comparable'] else 'reference_only' if rows else 'unavailable')
        return {'status': state,
                'label': {'met': '已配对指标达到参考规则', 'not_met': '已配对指标未全部达到参考规则',
                          'partial': '部分指标可比较', 'reference_only': '仅供参考，尚不可验收',
                          'unavailable': '暂无可用论文指标配对'}[state],
                'reason': _text(comparison.get('reason')) or '作者运行输出比较，不替代独立复算。',
                'basis': comparison.get('basis') if comparison.get('basis') in
                         {'author_output', 'patched_author_output'} else 'author_output',
                'counts': counts, 'evidence_paths': ['validation.author_claim_comparison']}
    inner = _mapping(validation.get('validation'))
    values = _mapping(validation.get('metrics_comparison'))
    independent_rows = _rows(validation.get('comparisons'))
    state = 'unavailable'
    if (validation.get('status') in {'reproduced', 'not_reproduced', 'reference_not_met', 'inconclusive'}
            and not inner.get('missing_metrics') and not inner.get('invalid_metrics')):
        verdict = inner.get('metrics_within_tolerance')
        if type(verdict) is not bool:
            verdict = validation.get('is_reproduced')
        if type(verdict) is bool and (values.get('paper') or independent_rows):
            state = 'met' if verdict else 'not_met'
            counts['total'] = len(independent_rows) or len(_mapping(values.get('paper')))
            counts['comparable'] = counts['total']
            if independent_rows and all(type(row.get('pass')) is bool for row in independent_rows):
                counts['met'] = sum(row['pass'] for row in independent_rows)
                counts['not_met'] = counts['total'] - counts['met']
                if bool(counts['not_met']) is verdict:
                    state = 'unavailable'  # Contradictory saved evidence cannot become a new verdict.
            elif verdict:
                counts['met'] = counts['total']
            else:
                # An aggregate failure does not identify which metric failed.
                counts['unknown'] = counts['total']
            if (type(validation.get('is_reproduced')) is bool
                    and validation['is_reproduced'] is not verdict):
                state = 'unavailable'
    return {'status': state,
            'label': {'met': '已有核验指标达到参考规则', 'not_met': '已有核验指标未全部达到参考规则',
                      'unavailable': '论文数值比较证据不足'}[state],
            'reason': _text(validation.get('reason')) if state != 'unavailable' else
                      '未取得完整、可对应的论文参考及实际指标判定。',
            'basis': 'validated_metrics' if state != 'unavailable' else 'unavailable',
            'counts': counts, 'evidence_paths': ['validation.metrics_comparison', 'validation.comparisons']}


def build_reproduction_conclusion(data):
    """Build a fresh, JSON-safe conclusion from the current authoritative fields.

    ``data['reproduction_conclusion']`` is deliberately ignored, so a historical
    rendering or a late cancellation cannot reuse a stale successful summary.
    """
    data = _mapping(data)
    validation, execution = _mapping(data.get('validation')), _mapping(data.get('execution'))
    strict = _mapping(data.get('strict_validation'))
    execution_level = _execution_level(data, validation, execution)
    comparison = _comparison_level(validation)
    modified = (validation.get('source_modified') is True
                or _mapping(data.get('source_repair')).get('status') == 'succeeded'
                or data.get('execution_source') == 'patched_author_repository')
    scope = _text(validation.get('scope') or validation.get('reproduction_scope')
                  or data.get('reproduction_scope'))
    method_only = validation.get('status') in {'method_experiment_completed', 'quality_target_not_met'}
    quality_pass = validation.get('quality_pass')
    quality_status = ('not_met' if validation.get('status') == 'quality_target_not_met'
                      or quality_pass is False else 'met' if quality_pass is True else 'not_recorded')
    quality = {'status': quality_status,
               'label': {'met': '已记录质量门槛通过', 'not_met': '已记录质量门槛未达标',
                         'not_recorded': '未记录单独质量门槛'}[quality_status],
               'evidence_paths': ['validation.quality_pass', 'validation.status']}
    scope_label = ('兼容性修复副本的原作者命令与输出' if modified else
                   '官方方法实验' if method_only else '选定论文实验' if scope == 'selected_paper_experiment'
                   else '本次运行与已有指标配对')
    protocol = validation.get('protocol_pass')
    metrics = validation.get('independent_metrics_pass')
    if type(protocol) is not bool:
        protocol = _mapping(execution.get('protocol_verification')).get('pass')
    if type(metrics) is not bool:
        metrics = _mapping(execution.get('independent_metrics')).get('pass')
    protocol = protocol if type(protocol) is bool else None
    metrics = metrics if type(metrics) is bool else None
    stop = execution_level['status'] in {'interrupted', 'timed_out', 'cleanup_unconfirmed'}
    strict_blocked = strict.get('requested') is True and strict.get('status') != 'verified'
    blocked = (stop or modified or validation.get('paper_metrics_verified') is False
               or validation.get('reviews_removed') is True or strict_blocked
               or execution.get('best_effort') is True or execution.get('fallback_used') is True)
    validation_complete = validation.get('status') in {
        'reproduced', 'not_reproduced', 'reference_not_met', 'inconclusive',
        'method_experiment_completed', 'quality_target_not_met'}
    independent_status = ('verified' if execution_level['status'] == 'completed' and validation_complete
                          and protocol is True and metrics is True and not blocked else
                          'failed' if (protocol is False or metrics is False) and not blocked else 'unavailable')
    independent = {'status': independent_status,
        'label': {'verified': '协议与独立指标复算已核验', 'failed': '独立核验未通过',
                  'unavailable': '独立核验尚未完成'}[independent_status],
        'protocol_verified': protocol, 'metrics_recomputed': metrics,
        'reason': ('该核验与论文参考是否达标分开记录。' if independent_status == 'verified' else
                   '本次完成范围仅限修复副本，未独立核验原始论文实验。' if modified else
                   _text(strict.get('reason')) or '完整协议与独立预测/指标复算证据仍未齐备。'),
        'evidence_paths': ['validation.protocol_pass', 'validation.independent_metrics_pass',
                           'execution.protocol_verification', 'execution.independent_metrics']}
    # Preserve established method-only scopes even when their own independent
    # metric computation succeeds: they have no paper-table acceptance target.
    reproduced = (validation.get('is_reproduced') if independent_status == 'verified'
                  and comparison['status'] in {'met', 'not_met'} and not method_only else None)
    reproduced = reproduced if type(reproduced) is bool else None
    if reproduced is not None and (comparison['status'] == 'met') is not reproduced:
        reproduced = None
    status = execution_level['status']
    if status == 'completed':
        status = ('reproduced' if reproduced is True else 'reference_not_met' if reproduced is False
                  else 'quality_target_not_met' if method_only and quality_status == 'not_met'
                  else 'method_completed' if method_only else 'reference_met' if comparison['status'] == 'met'
                  else 'reference_not_met' if comparison['status'] == 'not_met' else 'verification_pending')
    summaries = {'reproduced': '选定论文实验已完成，独立核验与已记录论文参考验收通过。',
        'reference_not_met': '运行已完成，已有数值判定未全部达到论文参考；偏差根因仍需核查。',
        'reference_met': '运行已完成，已配对指标达到参考规则；完整独立论文验收尚未完成。',
        'verification_pending': '运行已完成，论文数值比较或独立核验证据仍待补齐。',
        'method_completed': '方法实验已运行，结论限于该实验的指标与协议范围。',
        'quality_target_not_met': '方法实验已完成，但已记录质量门槛未达标；不据此判定整篇论文复现成功或失败。',
        'failed': '当前必需执行未正常结束，保留实际日志；暂不作论文数值验收结论。',
        'incomplete': '完成证据不足，保留已有结果并补充终态与指标来源。',
        'not_run': '尚未执行完整实验，当前结论限于准备与前置证据检查。',
        'smoke_passed': '仅完成冒烟检查，尚未完成完整论文实验。',
        'mock_demo': '模拟流程结束，未执行真实论文实验。',
        'interrupted': '任务已取消或中断，保留已有证据，未完成部分不作完成判定。',
        'timed_out': '任务已超时，保留已有证据，未完成部分不作完成判定。',
        'cleanup_unconfirmed': '进程清理尚未确认，完成与独立验收结论暂不发布。'}
    blockers, next_steps = [], []
    def add(code, reason, check, priority, paths):
        blockers.append({'code': code, 'reason': reason, 'evidence_paths': paths})
        next_steps.append({'priority': priority, 'category': code, 'check': check, 'evidence_paths': paths})
    if execution_level['status'] != 'completed':
        add('execution', execution_level['reason'],
            '先核对所属进程清理回执，确认释放后再依据原始命令与完整日志检查未完成步骤。'
            if stop else '按冻结计划核对准备状态、必需命令及实际结束记录。', 1,
            ['execution', 'validation.execution_status'])
    if modified:
        add('source_modified', '本次使用兼容性修复副本。',
            '复核补丁清单及原始来源；修复副本输出继续与原始源码严格验收分开记录。', 2,
            ['source_repair', 'validation.source_modified'])
    if comparison['status'] not in {'met', 'not_met'} and not method_only:
        add('comparison', comparison['reason'],
            '核对论文方法、数据集、划分、单位、重复次数及实际输出来源，补齐尚不可比较的指标配对。', 2,
            comparison['evidence_paths'])
    if independent_status != 'verified':
        add('independent', independent['reason'],
            '补齐冻结协议、完整作者执行、独立模型重载与预测复算证据，再提交独立核验。', 3,
            independent['evidence_paths'])
    if comparison['status'] == 'not_met':
        add('numeric_gap', '已有指标判定未全部达到记录中的参考规则。',
            '先核对单位、统计方式与冻结实验设置；分数偏差本身不作为修改训练参数或容差的依据。', 3,
            comparison['evidence_paths'])
    if quality_status == 'not_met':
        add('quality_target', '已记录质量门槛未达标。',
            '对照固定质量目标及实测指标核查差距；保留原始实验参数与质量判据。', 3,
            quality['evidence_paths'])
    cleanup = _mapping(data.get('environment_cleanup'))
    if cleanup.get('status') == 'error':
        add('environment_cleanup', '任务运行资源释放未完成；已记录实验与数值证据保留。',
            '查看 environment_cleanup 的释放诊断，完成任务资源清理。', 1, ['environment_cleanup'])
    diagnosis = _mapping(data.get('outcome_diagnosis'))
    analysis_status = data.get('analysis_status')
    model_status = _mapping(diagnosis.get('model_analysis')).get('status')
    analysis_failed = analysis_status in {'failed', 'result_analysis_failed'} or model_status == 'failed'
    if analysis_failed:
        next_steps.append({'priority': 4, 'category': 'analysis',
            'check': '需要补充解释时，仅重新分析已保存证据；不重跑训练或更改现有数值判定。',
            'evidence_paths': ['analysis_status', 'outcome_diagnosis.model_analysis']})
    known = {row.get('id') for row in _rows(diagnosis.get('evidence')) if isinstance(row.get('id'), str)}
    for item in _rows(diagnosis.get('next_steps'))[:5]:
        ids = item.get('evidence_ids')
        refs = [ref for ref in ids if isinstance(ref, str) and ref in known] if isinstance(ids, list) else []
        if _text(item.get('check')) and refs:
            priority = item.get('priority')
            if item.get('basis') == 'model_hypothesis':
                priority = 3
            next_steps.append({'priority': priority if type(priority) is int and priority in {1, 2, 3} else 3,
                               'category': _text(item.get('category')),
                               'check': _text(item['check']), 'evidence_ids': refs,
                               'basis': 'model_hypothesis' if item.get('basis') == 'model_hypothesis' else 'observed'})
    return {'version': 1, 'status': status, 'summary': summaries[status], 'is_reproduced': reproduced,
            'scope': {'name': scope, 'label': scope_label, 'source_modified': modified},
            'execution': {**execution_level, 'evidence_paths': ['execution', 'validation.execution_status'],
                          'training_observed': validation.get('training_observed') if
                          type(validation.get('training_observed')) is bool else None},
            'comparison': comparison, 'independent': independent, 'quality': quality,
            'analysis': {'status': 'failed' if analysis_failed else _text(analysis_status or model_status) or 'not_recorded',
                         'affects_numerical_verdict': False},
            'lifecycle': {'cleanup_status': _text(cleanup.get('status')) or 'not_recorded'},
            'blockers': blockers, 'next_steps': sorted(next_steps, key=lambda item: item['priority'])[:10]}


def _markdown(value):
    text = ' '.join(_text(value).split())
    for character in '\\' + chr(96) + '*_{}[]()<>!|':
        text = text.replace(character, '\\' + character)
    return text


def conclusion_markdown(conclusion):
    """Render only established evidence, with actionable checks ahead of detail."""
    lines = ['## 本次复现结论', '', '**' + _markdown(conclusion['summary']) + '**', '',
             '- **结论范围**: ' + _markdown(conclusion['scope']['label']),
             '- **实验执行**: ' + _markdown(conclusion['execution']['label']),
             '- **论文参考比较**: ' + _markdown(conclusion['comparison']['label']),
             '- **独立核验**: ' + _markdown(conclusion['independent']['label'])]
    if conclusion['quality']['status'] != 'not_recorded':
        lines.append('- **质量目标**: ' + _markdown(conclusion['quality']['label']))
    counts = conclusion['comparison']['counts']
    if counts['total']:
        lines.append(f"- **指标覆盖**: 共 {counts['total']} 项；可比较 {counts['comparable']} 项，"
                     f"已知达标 {counts['met']} 项，已知未达标 {counts['not_met']} 项，"
                     f"仅供参考 {counts['reference_only']} 项，不可比较 {counts['not_comparable']} 项，"
                     f"逐项判定未知 {counts['unknown']} 项。")
    if conclusion['analysis']['status'] == 'failed':
        lines.append('- **补充分析**: 解释分析未完成；已有执行与数值判定保留。')
    if conclusion['lifecycle']['cleanup_status'] == 'error':
        lines.append('- **任务收尾**: 环境释放失败；已有实验和数值证据保留，释放问题另行处理。')
    if conclusion['blockers']:
        lines += ['', '**当前限制与未完成项**', '']
        lines += ['- ' + _markdown(item['reason']) for item in conclusion['blockers']]
    if conclusion['next_steps']:
        lines += ['', '**优先核查**', '']
        for item in conclusion['next_steps'][:6]:
            basis = '；待核实假设' if item.get('basis') == 'model_hypothesis' else ''
            lines.append(f"- P{item['priority']}{basis}：" + _markdown(item['check']))
    return '\n'.join(lines)
