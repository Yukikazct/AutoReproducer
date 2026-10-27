"""Readable report presentation with a hash-bound, unabridged details file."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote

from src.safety.paths import workspace_path


_LINK = re.compile(r'^\[完整依据与执行详情\]\(([^\r\n)]+)\) <!-- sha256:([0-9a-f]{64}) -->$', re.M)


def report_details_bytes(report, report_path):
    """Read only an emitted sibling details file whose bytes match its hash."""
    result = {}
    report_file = Path(report_path)
    if not report_file.is_absolute():
        from src.execution_artifacts import PROJECT_ROOT
        report_file = PROJECT_ROOT / report_file
    for target, digest in _LINK.findall(report):
        name = unquote(target)
        if len(Path(name).parts) != 1 or not name.endswith('_details.md'):
            raise ValueError('报告详情附件路径无效')
        try:
            path = workspace_path(report_file.absolute().parent, name,
                                  'report details attachment', must_exist=True)
        except OSError as exc:
            raise ValueError('报告详情附件缺失或无法读取') from exc
        if not path.is_file():
            raise ValueError('报告详情附件缺失或无法读取')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('报告详情附件哈希不一致')
        result[name] = raw
    return result


def presentation_data(data):
    """Keep interpreted settings and results, moving literal payloads to details."""
    def visit(node):
        if isinstance(node, dict):
            return {key: '' if key in {'quote', 'source_quote', 'value_quote', 'column_quote',
                                      'stdout', 'stderr', 'code'} and isinstance(value, str)
                    else visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return deepcopy(node)
    return visit(data)


def readable_text(text, data):
    """Remove empty literal-output scaffolding while retaining explanatory notes."""
    lines, skip_output, skip_collaboration = [], False, False
    for line in text.splitlines():
        if line == '### 执行输出(full)':
            skip_output = True
            final = (data.get('execution') or {}).get('final') or {}
            output_state = '已保存' if final.get('stdout') else '无输出或未记录'
            lines += ['### 执行记录', '', '- **标准输出**: ' + output_state + '。',
                      '完整标准输出、诊断日志和代码保存在详情附件中。', '']
            continue
        if skip_output:
            if line.startswith('## '):
                skip_output = False
            else:
                continue
        if line.startswith('## ') and '协作记录' in line:
            skip_collaboration = True
            messages = (data.get('agent_collaboration') or {}).get('messages') or []
            lines += ['## Agent 协作摘要', '', f'已保存 {len(messages)} 条阶段交接与核查消息，完整记录见详情附件。', '']
            continue
        if skip_collaboration:
            if line.startswith('## '):
                skip_collaboration = False
            else:
                continue
        if (line.strip() in {'>', '**原文依据**'} or line.startswith('- **原文上下文**:')
                or line.strip().startswith('- **原文**:') or line.startswith('- **发现来源**:')
                or line.startswith('- **仓库关联证据**:')):
            continue
        # Source identifiers remain visible when the corresponding quotation is
        # moved, including references emitted as "> source_id: quote".
        if re.fullmatch(r'>\s*[^:：]+[:：]\s*', line):
            continue
        line = line.replace('**计划引用原文**', '**计划依据位置**')
        lines.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).replace(
        '# 论文复现与优化报告', '# 论文复现实验报告', 1)


def intelligent_recovery_summary(recovery):
    """Present saved recovery state without interpreting it as scientific proof."""
    if not isinstance(recovery, dict):
        return ''

    def text(value, limit=160):
        if not isinstance(value, (str, int, float, bool)):
            return ''
        value = re.sub(r'\s+', ' ', str(value)).strip()
        if len(value) > limit:
            value = value[:limit] + '…'
        return re.sub(r'([\\`*_{}\[\]()<>!|])', r'\\\1', value)

    lines = ['## 智能调查与处理结果', '']
    if recovery.get('enabled') is not True:
        return '\n'.join(lines + ['本次未启用智能调查与排障。', ''])
    modes = {'single': '单角色', 'collaborative': '角色协作'}
    mode = modes.get(recovery['mode'], '未记录') if isinstance(recovery.get('mode'), str) else '未记录'
    statuses = {'ready': '待调查', 'investigating': '调查中', 'investigated': '调查已记录',
                'unavailable': '调查不可用', 'stopped': '已停止', 'needs_human': '需要人工处理',
                'repair_proposed': '已提出修复，待原有检查', 'observed': '已记录观察结果',
                'command_recovered': '原命令运行已恢复，数值等价性未确认',
                'environment_recovered': '环境准备已恢复，作者实验运行另行记录',
                'dataset_recovered': '数据准备已恢复，作者实验运行另行记录'}
    triggers = {'execution_failure': '作者命令执行失败', 'dependency_failure': '作者依赖准备失败',
                'result_gap': '结果差距', 'metric_gap': '指标差距'}

    def label(value, labels):
        return text(labels.get(value, value)) if isinstance(value, str) else '未记录'

    def scoped_status(record):
        from src.report_view_model import is_non_author_recovery_phase
        status = record.get('status')
        if status != 'command_recovered':
            return status
        phases = (record.get('recovered_phase'), record.get('phase'))
        phase = next((value for value in phases if is_non_author_recovery_phase(value)), None)
        return {'environment_preparation': 'environment_recovered',
                'environment': 'environment_recovered',
                'dataset_preparation': 'dataset_recovered',
                'dataset': 'dataset_recovered'}.get(phase, 'observed') if phase else status

    saved_tasks = recovery.get('tasks')
    tasks = [task for task in saved_tasks if isinstance(task, dict)] if isinstance(saved_tasks, list) else []
    lines.append(f"- **调查状态**: {label(scoped_status(recovery), statuses)}；{mode}；共 {len(tasks)} 项调查。")
    for task in tasks[:3]:
        trigger = task.get('trigger') or task.get('id') or '未记录触发原因'
        if trigger == 'execution_failure':
            phase = task.get('recovered_phase') or task.get('phase')
            trigger = {'environment_preparation': '环境预检执行失败',
                       'environment': '环境预检执行失败',
                       'dataset_preparation': '数据准备执行失败',
                       'dataset': '数据准备执行失败'}.get(phase, trigger)
        trigger = label(trigger, triggers)
        lines.append(f"- **处理记录**: {trigger}；状态：{label(scoped_status(task), statuses)}。")
        decisions = task.get('decisions')
        if isinstance(decisions, list) and decisions:
            decision = decisions[-1]
            if isinstance(decision, dict):
                # Keep only the saved decision explanation, never raw probes or
                # proposed code. The complete object belongs in the attachment.
                decision = next((decision[key] for key in ('reason', 'summary', 'action', 'status')
                                 if isinstance(decision.get(key), str) and decision[key]), '')
            if text(decision):
                lines.append('  处理依据：' + text(decision) + '。')
    if len(tasks) > 3:
        lines.append(f'- 其余 {len(tasks) - 3} 项调查见完整依据与执行详情。')
    reviews = recovery.get('reviews')
    if isinstance(reviews, list):
        reviews = [row for row in reviews if isinstance(row, dict)]
        model_reviews = [row for row in reviews if row.get('reviewer') in {'independent', 'self_check'}]
        requested_checks = sum(row.get('action') == 'probe' for row in model_reviews)
        vetoed = {row['candidate_sha256'] for row in reviews if row.get('stage') == 'patch'
                  and row.get('action') == 'reject' and isinstance(row.get('candidate_sha256'), str)}
        review_name = '独立反证审查' if recovery.get('mode') == 'collaborative' else '自检'
        lines.append(f'- **审查作用**: {review_name} {len(model_reviews)} 次；请求补查 {requested_checks} 次；'
                     f'否决候选补丁 {len(vetoed)} 个。')
    if recovery.get('requires_human') is True:
        lines.append('- **需要人工处理**: 是；原因和调查证据见详情附件。')
    if text(recovery.get('stop_reason')):
        lines.append('- **停止原因**: ' + text(recovery['stop_reason']))
    usage = recovery.get('usage')
    token_count = usage.get('total_tokens') if isinstance(usage, dict) else None
    shown_tokens = str(token_count) if type(token_count) is int and token_count >= 0 else 'unknown（未记录）'
    lines.append('- **模型 token 用量**: ' + shown_tokens)
    lines += ['调查与处理记录用于解释本次运行；论文复现结论仍以执行、协议与指标核验结果为准。完整调查记录见详情附件。', '']
    return '\n'.join(lines)


def _cell(value, missing='未记录'):
    if value is None or value == '':
        return missing
    text = re.sub(r'\s+', ' ', str(value)).strip()
    if len(text) > 240:
        text = text[:240] + '…（完整记录见详情附件）'
    return re.sub(r'([\\`*_{}\[\]()<>!|])', r'\\\1', text)


def report_metric_lines(metrics):
    """Render the shared metric model without inventing missing observations."""
    if not metrics:
        return ['本次实际指标或论文参考尚未取得。']
    lines = ['| 指标 | 论文参考 | 实际值 | 单位（参考 / 实际） | 划分（参考 / 实际） | 次数（参考 / 实际） | 差值 | 可比较性 |',
             '|---|---|---|---|---|---|---|---|']
    labels = {'comparable': '可比较', 'reference_only': '仅供参考',
              'not_comparable': '不可比较', 'unrecorded': '未记录'}
    for row in metrics:
        difference = _cell(row.get('difference'), '未取得')
        if row.get('difference') is not None and row.get('difference_unit'):
            unit = {'percentage_points': '个百分点'}.get(row['difference_unit'], row['difference_unit'])
            difference += ' ' + _cell(unit)
        cells = [_cell(row.get('metric')), _cell(row.get('reference'), '未取得'),
                 _cell(row.get('actual'), '未取得'),
                 _cell(row.get('reference_unit')) + ' / ' + _cell(row.get('actual_unit')),
                 _cell(row.get('reference_split')) + ' / ' + _cell(row.get('actual_split')),
                 _cell(row.get('reference_runs')) + ' / ' + _cell(row.get('actual_runs')),
                 difference,
                 _cell(labels.get(row.get('comparability'), row.get('comparability')))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '未取得的数值保留为空缺；可比较性依据冻结协议与已有核验记录，指标名称相同不足以确认可比。']
    return lines


def recovery_contribution_lines(rows, *, recorded):
    """Separate proposed, rejected, executed and verified recovery contributions."""
    lines = ['## 恢复行动与协作贡献', '']
    if not recorded:
        return lines + ['未记录：历史结果没有保存智能恢复信息。', '']
    if not rows:
        return lines + ['没有记录恢复动作；阶段交接或收到消息不计为已完成修复。', '']
    lines += ['| 阶段与原始问题 | 调查角色 | 证据 | 采用动作 | 审查结果 | 原命令重跑 | 最终效果 |',
              '|---|---|---|---|---|---|---|']
    statuses = {'proposed': '仅提出建议', 'rejected': '已否决，未执行', 'executing': '执行中',
                'executed': '已执行，效果待核验', 'verified': '已执行且效果已验证',
                'failed': '执行失败', 'stopped': '已停止', 'no_progress': '未取得新进展'}
    for row in rows:
        review = row.get('review') or {}
        review_result = review.get('reason') or review.get('status') or review.get('action') or '未记录'
        evidence = row.get('evidence') or []
        if isinstance(evidence, list):
            evidence = ', '.join(str(item.get('id') or item.get('path') or item.get('source') or '已保存证据')
                                 if isinstance(item, dict) else str(item) for item in evidence)
        rerun = '是' if row.get('rerun') is True else '否' if row.get('rerun') is False else '未记录'
        status = statuses.get(row.get('status'), row.get('status') or '未记录')
        if row.get('status') == 'rejected' and row.get('executed'):
            status = '审查拒绝；保留已有执行记录'
        cells = [str(row.get('phase') or '未记录') + '；' + str(row.get('problem') or '未记录'),
                 row.get('role'), evidence, row.get('action'), review_result, rerun,
                 status + ('；' + str(row['result']) if row.get('result') else '')]
        lines.append('| ' + ' | '.join(_cell(value) for value in cells) + ' |')
    lines += ['', '以上说明实际排障作用；运行恢复、论文参考达标与独立核验分别判断。', '']
    return lines


def save_report_details(full_report, data, report_path):
    """Retain raw quotation values even where the legacy audit used excerpts."""
    values = {}
    def collect(node, path='data'):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {'quote', 'source_quote', 'value_quote', 'column_quote'} and isinstance(value, str) and value:
                    values.setdefault(value, []).append(path + '.' + key)
                else:
                    collect(value, path + '.' + str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                collect(value, f'{path}[{index}]')
    collect(data)
    if values:
        full_report += '\n\n## 完整引用字段\n\n原始引用字段逐项保留，定位键对应保存的结构化结果。\n'
        for value, locations in values.items():
            fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', value)), default=0))
            full_report += '\n### ' + ', '.join(locations) + '\n\n' + fence + 'text\n' + value + '\n' + fence + '\n'
    collaboration = data.get('agent_collaboration') or {}
    if collaboration.get('messages'):
        raw_collaboration = json.dumps(collaboration, ensure_ascii=False, indent=2)
        fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', raw_collaboration)), default=0))
        full_report += ('\n\n## 完整协作记录\n\n保存本次报告生成时的完整协作快照；消息本身不构成核验结论。\n\n'
                        + fence + 'json\n' + raw_collaboration + '\n' + fence + '\n')
    recovery = data.get('intelligent_recovery')
    if isinstance(recovery, dict):
        from src.author_claim_comparison import _redact_evidence
        from src.report_revisions import sanitize_report_result
        raw_recovery = json.dumps(_redact_evidence(sanitize_report_result(recovery), ''),
                                  ensure_ascii=False, indent=2)
        fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', raw_recovery)), default=0))
        full_report += ('\n\n## 完整智能调查与处理记录\n\n'
                        '以下为保存的调查快照；假设、探针与决策不替代执行和独立指标核验。\n\n'
                        + fence + 'json\n' + raw_recovery + '\n' + fence + '\n')
    path = Path(report_path).absolute()
    if not Path(report_path).is_absolute():
        from src.execution_artifacts import PROJECT_ROOT
        path = PROJECT_ROOT / report_path
    name = path.stem + '_details.md'
    target = workspace_path(path.parent, name, 'report details attachment')
    target.parent.mkdir(parents=True, exist_ok=True)
    from src.source_repair_evidence import _replace_bytes
    raw = full_report.encode('utf-8')
    _replace_bytes(target, raw)
    return '[完整依据与执行详情](' + quote(name, safe='') + ') <!-- sha256:' + hashlib.sha256(raw).hexdigest() + ' -->'
