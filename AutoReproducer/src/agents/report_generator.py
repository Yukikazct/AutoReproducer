"""ReportGeneratorAgent - 报告生成 Agent，生成 Markdown 复现与优化报告。

报告覆盖：论文信息 / 资源定位 / 环境配置与依赖诊断 / 代码执行 /
实际输出与结束状态 / 调用统计；历史运行保留当时的验收与优化记录。
"""
from datetime import datetime
import json
import re
from src.base_agent import BaseAgent
from src.metric_keys import norm_metric_key
from src.runtime_metrics import metric_unit
from src.storage_usage import format_bytes
from src.execution_artifacts import default_report_path, markdown_image_target
from src.report_revisions import REPORT_FORMAT_VERSION, REPORT_FORMAT_MARKER


def _fmt(value, spec: str = ".2f", default: float = 0.0) -> str:
    """数值格式化：容忍字符串/None 等非数值（LLM 可能把 confidence 填成 "0.1"）。"""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return format(float(default), spec)


def _txt(value, default: str = "") -> str:
    """文本字段兜底：`dict.get` 的默认值对「键存在且值为 None」不生效。

    docker 执行路径曾因 GBK 解码失败把 stdout 置成 None，报告整段崩在
    `"\\n".join(lines)`（TypeError: expected str instance, NoneType found）。
    非字符串值一律转成字符串，保证拼进 lines 的永远是 str。
    """
    if isinstance(value, str):
        return value
    return default if value is None else str(value)


def _markdown_text(value: str) -> str:
    """Treat generated filenames and artifact diagnostics as plain text."""
    return re.sub(r"([\\`*_{}\[\]()<>!|])", r"\\\1",
                  value.replace("\n", " ").replace("\r", " "))


def _inline_code(value):
    value = str(value).replace('\n', ' ').replace('\r', ' ')
    fence = '`' * (1 + max((len(run) for run in re.findall(r'`+', value)), default=0))
    return fence + ' ' + value + ' ' + fence


def _collaboration_summary(content):
    """Keep the report readable; the run artifact retains full message payloads."""
    if isinstance(content, dict):
        if content.get('reason'):
            summary = str(content['reason'])
        else:
            labels = {'action': '动作', 'status': '状态', 'validation_status': '核验状态',
                      'scope': '范围', 'error': '错误'}
            parts = [label + '：' + str(content[key]) for key, label in labels.items() if content.get(key)]
            paths = content.get('source_paths') or content.get('paths')
            if isinstance(paths, list):
                parts.append('来源：' + '、'.join(str(path) for path in paths[:6]))
            commands = content.get('commands') or content.get('steps')
            if isinstance(commands, list):
                parts.append(f'交接 {len(commands)} 个步骤')
            files = content.get('files')
            if isinstance(files, list):
                parts.append(f'补读 {len(files)} 个来源文件')
            repository = content.get('repository')
            if isinstance(repository, dict) and repository.get('url'):
                parts.append('仓库：' + str(repository['url']))
            summary = '；'.join(parts) or '已交接结构化记录；完整内容保存在运行目录。'
    else:
        summary = str(content)
    summary = re.sub(r'\b[0-9a-f]{32,64}\b', lambda match: match[0][:12] + '…', summary)
    return _markdown_text(summary[:180] + ('…' if len(summary) > 180 else ''))


def _numeric_compatibility_lines(validation):
    """Disclose adopted source-bound operations from verified runtime receipts."""
    labels = {
        'numpy_float_reciprocal': '浮点倒数兼容修复',
        'numpy_builtin_alias': '旧 NumPy 类型别名兼容修复',
        'numpy_ragged_summary': '旧 NumPy 不规则结果表汇总兼容修复',
    }
    records = validation.get('numeric_compatibility')
    if not isinstance(records, list):
        return []
    records = [item for item in records if isinstance(item, dict) and item.get('kind') in labels]
    if not records:
        return []
    kinds = {item['kind'] for item in records}
    note = ('执行包含实际报错后记录的' + '、'.join(label for kind, label in labels.items() if kind in kinds)
            + '；原仓库文件未改写。')
    lines = [] if note in (validation.get('warnings') or []) else ['- **运行说明**: ' + note]
    for item in records:
        locations = [item.get('line'), *(call.get('line') for call in item.get('calls', [])
                                        if isinstance(call, dict))]
        locations = sorted({line for line in locations if type(line) is int and line > 0})
        source = _txt(item.get('path'))
        if locations:
            source += '；行 ' + ', '.join(str(line) for line in locations)
        if item.get('source_sha256'):
            source += '；SHA-256: ' + _txt(item['source_sha256'])
        if source:
            lines.append('- **数值兼容来源**: ' + _markdown_text(source))
    return lines


def _source_repair_lines(data, report_path):
    from src.repository_repairer import _private
    state = _private(data.get('source_repair') or {})
    if not state.get('requested'):
        return []
    lines = ['', '### 兼容性源码修复', '',
        '- **状态**: ' + _markdown_text(_txt(state.get('status'))),
        '- **授权范围**: ' + ', '.join(_markdown_text(name) for name in state.get('editable', [])),
        f"- **补丁轮数**: {len(state.get('attempts', []))} / 2；每轮整体校验、完整原命令重跑，失败或取消整体回滚。",
        '- 原始源码、来源包与冻结计划保留；补丁仅作用于独立修复副本。修改后执行不构成原始源码严格复现或独立论文数值核验。']
    for key, label in (('original_workspace', '原始副本'), ('workspace', '修复副本'),
                       ('reason', '说明'), ('report_path', '修复结果与失败报告'), ('records_path', '结构化审计记录')):
        if state.get(key):
            lines.append('- **' + label + '**: ' + _markdown_text(_txt(state[key])))
    if state.get('manifest_sha256'):
        lines.append('- **修复清单 SHA-256**: `' + state['manifest_sha256'] + '`')
    for attempt in state.get('attempts', []):
        lines += ['', f"- 第 {attempt['round']} 轮：" + _markdown_text(_txt(attempt.get('status')))
                  + '；' + _markdown_text(_txt(attempt.get('reason') or attempt.get('error')))]
        if attempt.get('reason') and attempt.get('error'):
            lines.append('  错误：' + _markdown_text(_txt(attempt['error'])))
        if attempt.get('rollback_error'):
            lines.append('  回滚失败：' + _markdown_text(_txt(attempt['rollback_error'])))
        for change in attempt.get('changed_files', []):
            lines.append('  ' + _markdown_text(change['path']) + '：`' + change['base_sha256'] + '` → `' + change['sha256'] + '`')
    if state.get('evidence_archive'):
        from src.source_repair_evidence import evidence_target
        try:
            target = evidence_target(data, report_path)
            if target:
                lines += ['', target,
                    '附件包含逐轮 proposal、diff、结果、回滚记录与完整进程日志；成员哈希与脱敏状态见 evidence_manifest.json。']
        except (OSError, ValueError, KeyError) as exc:
            lines += ['', '源码修复证据附件不可用：' + _markdown_text(str(exc))]
    if state.get('evidence_error'):
        lines.append('- **附件生成失败**: ' + _markdown_text(state['evidence_error']))
    return lines


def _metric_value_text(node, suffix=''):
    value = _txt(node.get('value'))
    uncertainty = node.get('reported_uncertainty')
    if uncertainty is not None:
        kind = node.get('uncertainty_kind')
        if kind in {'bootstrap_confidence_interval', 'confidence_interval'}:
            level = node.get('confidence_level')
            level_text = format(level, 'g') if type(level) in (int, float) else _txt(level)
            label = (f'{level_text}% ' if level is not None else '')
            label += ('bootstrap ' if kind == 'bootstrap_confidence_interval' else '') + '置信区间半宽'
        elif kind == 'standard_error':
            label = '标准误'
        else:
            label = '误差含义未明确'
        return f'{value} ± {uncertainty}{suffix}（{label}）'
    deviation = f" ± {node['std']}" if node.get('std') is not None else ''
    return value + deviation + suffix


def _outcome_diagnosis_lines(data):
    diagnosis = data.get('outcome_diagnosis')
    if not isinstance(diagnosis, dict) or not diagnosis:
        return []
    from src.author_claim_comparison import _redact_evidence
    diagnosis = _redact_evidence(diagnosis, '')
    lines = ['', '### 数值结果与证据自动诊断', '',
             '- **诊断状态**: ' + {'evidence_limited': '比较证据仍有限',
                                    'unavailable': '暂不可用'}.get(diagnosis.get('status'), '已生成诊断'),
             '- **说明**: ' + _markdown_text(_txt(diagnosis.get('reason'))),
             '- 本诊断核查已有结果与冻结设置；未修改实验参数、源码、参考值或重新训练。']
    known = {item.get('id'): item for item in diagnosis.get('evidence', []) if isinstance(item, dict)}
    assessment = diagnosis.get('assessment') or {}
    coverage = assessment.get('comparison_coverage') or {}
    if coverage:
        lines += [f"- **比较覆盖**: 共 {coverage.get('total', 0)} 项，可比较 {coverage.get('comparable', 0)} 项，"
                  f"仅供参考 {coverage.get('reference_only', 0)} 项，不可比较 {coverage.get('not_comparable', 0)} 项。",
                  f"- **已记录参考差距**: {coverage.get('outside_reference', 0)} 项；根因尚未确认。"]
    reused = (diagnosis.get('efficiency') or {}).get('cache_hits', 0)
    if type(reused) is int and reused > 0:
        lines.append(f'- **诊断复用**: 完整输入与来源哈希未变化，复用已有分析 {reused} 次。')

    def references(item):
        return '；依据：' + ', '.join(_markdown_text(ref) for ref in item.get('evidence_ids', []) if ref in known)

    for key, label in (('facts', '已观察事实与缺失证据'), ('hypotheses', '待核实假设')):
        rows = diagnosis.get(key) or []
        if rows:
            lines += ['', '**' + label + '**', '']
            for item in rows:
                lines.append('- ' + _markdown_text(_txt(item.get('message'))) + references(item))
    if diagnosis.get('next_steps'):
        lines += ['', '**下一步具体核查**', '']
        for item in diagnosis['next_steps']:
            priority = 3 if item.get('basis') == 'model_hypothesis' else item.get('priority')
            prefix = f'P{priority}；' if type(priority) is int and priority in {1, 2, 3} else ''
            if item.get('basis') == 'model_hypothesis':
                prefix += '待核实假设；'
            elif item.get('blocker') is True:
                prefix += '比较证据待补齐；'
            lines.append('- ' + prefix + _markdown_text(_txt(item.get('check'))) + references(item))
    model = diagnosis.get('model_analysis') or {}
    if model.get('status') != 'completed':
        lines += ['', '- **模型补充分析**: ' + _markdown_text(_txt(model.get('reason'), '未运行'))]
    if known:
        lines += ['', '**诊断证据索引**', '']
        for identifier, item in known.items():
            lines.append('- ' + _markdown_text(_txt(identifier)) + '：' + _markdown_text(_txt(item.get('source'))))
    if diagnosis.get('path'):
        lines.append('- **结构化诊断记录**: ' + _markdown_text(_txt(diagnosis['path'])))
    if diagnosis.get('storage_error'):
        lines.append('- **诊断文件保存失败**: ' + _markdown_text(_txt(diagnosis['storage_error'])))
    return lines


def _comparison_column_lines(reference):
    """Display recorded column evidence without deriving a path from prose."""
    evidence = reference.get('column_evidence')
    evidence = evidence if isinstance(evidence, dict) else {}
    if evidence:
        state = {'verified': '已核验', 'unknown': '未完成', 'rejected': '未通过'}.get(
            evidence.get('status'), '未完成')
    elif reference.get('column_verified') is True:
        state = '已核验（历史记录；列路径未记录）'
    elif reference.get('column_verified') is False:
        state = '未完成（历史记录）'
    else:
        state = '未记录'
    lines = ['- **论文列位置核验**: ' + state]
    labels = evidence.get('column_labels')
    if (evidence.get('status') == 'verified' and isinstance(labels, list) and labels
            and all(isinstance(label, str) and label.strip() for label in labels)):
        lines.append('- **已核验列路径**: ' + _markdown_text(' → '.join(labels)))
    if evidence.get('reason'):
        lines.append('- **列核验说明**: ' + _markdown_text(_txt(evidence['reason'])))
    return lines


def _comparison_field_lines(node, side):
    """Keep semantic field status and distinct supporting quotations together."""
    evidence = node.get('field_evidence')
    if not isinstance(evidence, dict) or not evidence:
        return []  # Old reports do not acquire evidence they never recorded.
    statuses, quotations = [], {}
    for field, label in (('unit', '单位'), ('split', '划分'), ('aggregation', '统计方式'), ('runs', '重复次数')):
        info = evidence.get(field)
        info = info if isinstance(info, dict) else {}
        status = {'verified': '已核验', 'unknown': '未知', 'contradicted': '与原文矛盾'}.get(
            info.get('status'), '未记录')
        statuses.append(label + '：' + status)
        quote = info.get('quote')
        if isinstance(quote, str) and quote and info.get('status') in {'verified', 'contradicted'}:
            key = (_txt(info.get('citation')), quote)
            quotations.setdefault(key, []).append(label)
    lines = ['- **字段核对（' + side + '）**: ' + '；'.join(statuses)]
    for (source, quote), labels in quotations.items():
        excerpt = quote[:400] + ('（引用节选）' if len(quote) > 400 else '')
        lines.append('- **字段依据（' + side + '：' + '、'.join(labels) + '）**: '
                     + _markdown_text((source + '；' if source else '') + excerpt))
    return lines


def _author_claim_comparison_lines(comparison, *, include_table=True):
    if not comparison:
        return []
    lines = ['', '### 作者输出与论文宣称对比', '',
             _markdown_text(_txt(comparison.get('reason'))), '',
             '本表使用作者程序输出，尚未独立复算。数值对比不影响代码完成状态。']
    rows = comparison.get('comparisons') or []
    if not rows:
        note = ('实际输出中的指标保留在下方证据表；未将其他数据集或基线指标作为参考。'
                if comparison.get('metric_records') else '作者输出中未识别到可比较的数值指标。')
        diagnostic = comparison.get('diagnostic')
        saved_error = comparison.get('persistence_error')
        return (lines + [note] + (['未完成原因：' + _markdown_text(str(diagnostic)[:2000])] if diagnostic else [])
                + (['- **对比记录保存诊断**: ' + _markdown_text(str(saved_error))] if saved_error else []) + [''])
    if include_table:
        lines += ['', '| 指标 / 数据集 / 方法 | 论文宣称 | 作者运行输出 | 差值（实际 − 论文） | 结论 |',
                  '| --- | --- | --- | --- | --- |']
    def value_cell(node):
        suffix = {'percent': '%', 'fraction': '（比例）'}.get(node.get('unit'), '')
        aggregation = {'single': '单次', 'mean': '均值', 'median': '中位数', 'best': '最佳', 'unknown': '统计方式未明确'}.get(node.get('aggregation'), '')
        runs = node.get('runs')
        return _metric_value_text(node, suffix) + f"；{aggregation}" + (f"，{runs} 次" if runs else '')
    for row in rows:
        paper, actual = row.get('reference'), row.get('actual')
        paper = paper if isinstance(paper, dict) else {'value': paper}
        actual = actual if isinstance(actual, dict) else {'value': actual}
        difference = row.get('difference')
        delta = '无法比较' if difference is None else f"{difference:+.4g}"
        if difference is not None:
            delta += {'percentage_points': ' 个百分点', 'fraction': '（比例）'}.get(row.get('difference_unit'), '')
            relative = row.get('relative_difference')
            if relative is not None:
                delta += f"；相对差异 {relative:+.2%}"
        verdict = {'not_comparable': '设置或指标不对应', 'reference_only': '仅供参考，尚未验收'}.get(row.get('status'))
        if verdict is None:
            verdict = ('达到参考' if row['meets_reference'] else '未达到参考') if row.get('meets_reference') is not None else (
                '在误差范围内' if row.get('within_tolerance') is True else
                '超出误差范围' if row.get('within_tolerance') is False else '逐项判定未记录')
        if include_table:
            cells = [f"{row.get('metric')} / {paper.get('dataset')} / {paper.get('method')}",
                     value_cell(paper), value_cell(actual), delta, verdict]
            lines.append('| ' + ' | '.join(_markdown_text(cell) for cell in cells) + ' |')
        else:
            lines.append('- **对比结果**: ' + _markdown_text(str(row.get('metric')) + '；' + verdict + '；差值 ' + delta))
    lines += ['', f"参考阈值为项目约定的 {comparison.get('relative_tolerance', 0.05):.0%} 相对误差，非论文提供的标准。"]
    citations = {ref['id']: ref for ref in comparison.get('citations', [])}
    for row in rows:
        paper, actual = row.get('reference'), row.get('actual')
        paper = paper if isinstance(paper, dict) else {'value': paper}
        actual = actual if isinstance(actual, dict) else {'value': actual}
        settings = row.get('settings')
        settings = settings if isinstance(settings, dict) else {}
        assessment = {'matched': '所引设置相符', 'different': '所引设置不同',
                      'unknown': '设置证据不足或未记录'}.get(settings.get('status'), '设置证据不足或未记录')
        basis = ('带引文的模型分析' if settings.get('assessment_basis') == 'model_with_citations'
                 else '判断来源未记录')
        lines += ['', '- **对比范围**: ' + _markdown_text(f"{row['metric']}；论文划分 {paper.get('split')} / 实际划分 {actual.get('split')}"),
                  '- **设置分析**: ' + assessment + '；' + basis + '，不等于独立协议核验通过。',
                  '- **实验设置**: ' + _markdown_text(_txt(settings.get('reason'), '未记录')),
                  '- **实际来源**: ' + _markdown_text(_txt(actual.get('source')))]
        lines += _comparison_column_lines(paper)
        lines += _comparison_field_lines(paper, '论文')
        lines += _comparison_field_lines(actual, '实际')
        for limitation in row.get('limitations', []):
            lines.append('- **对比限制**: ' + _markdown_text(limitation))
        pages = [str(citations[key]['page']) for key in paper.get('citations', []) if key in citations and citations[key]['source'] == 'pdf']
        if pages:
            lines += ['- **论文来源**: PDF 第 ' + '、'.join(dict.fromkeys(pages)) + ' 页']
        if paper.get('value_quote'):
            lines.append('> ' + _markdown_text(paper['value_quote']))
        if paper.get('column_quote'):
            lines += ['- **论文列标题**: ' + _markdown_text(paper['column_quote'])]
        if actual.get('source_quote'):
            lines += ['- **实际输出原文**: ' + _markdown_text(actual['source_quote'])]
    for error in comparison.get('validation_errors') or []:
        lines.append('- **未完成的配对**: ' + _markdown_text(str(error.get('error', ''))[:600]))
    if comparison.get('persistence_error'):
        lines.append('- **对比记录保存诊断**: ' + _markdown_text(str(comparison['persistence_error'])))
    return lines + ['']


def _runtime_dataset_lines(evidence):
    if not evidence:
        return []
    lines = ['', '### 原作者数据加载观测', '',
             '以下记录核对原加载器及其输入文件，官方参考数据与完整训练协议尚未独立核对。']
    for source in evidence.get('sources', []):
        observations = source.get('observations') or []
        if not observations:
            continue
        record = observations[0]
        loader = source.get('loader') or {}
        identity = (f"{record.get('dataset_name', '未知')}；{loader.get('dataset_class', '')}；"
                    f"框架 {loader.get('framework', '')} {record.get('framework_version', '')}；"
                    f"观测 {len(observations)} 次")
        lines += ['', '- **加载观测已核对**: ' + _markdown_text(identity)]
        counts = '；'.join(f'{label} {record[field]}' for field, label in (
            ('num_nodes', '节点'), ('num_edges', '边'), ('num_features', '特征'), ('num_classes', '类别')) if field in record)
        if counts:
            lines.append('- **原生规模**: ' + _markdown_text(counts))
        if record.get('mask_counts'):
            lines.append('- **原生划分样本数**: ' + _markdown_text('；'.join(
                f'{name} {count}' for name, count in record['mask_counts'].items())))
        if record.get('source_url'):
            lines.append('- **框架下载来源**: ' + _markdown_text(str(record['source_url'])))
        lines += ['', '| 输入文件 | 类型 | 字节数 | SHA-256 |', '| --- | --- | --- | --- |']
        for name, info in sorted(record.get('files', {}).items()):
            cells = [str(name).replace('\\', '/').rsplit('/', 1)[-1], info.get('role'), info.get('bytes'), info.get('sha256')]
            lines.append('| ' + ' | '.join(_markdown_text(_txt(value)) for value in cells) + ' |')
    for error in evidence.get('errors', []):
        lines.append('- **观测未核对**: ' + _markdown_text(str(error.get('reason', '未知'))))
        for reason in error.get('observation_errors', []):
            lines.append('  - ' + _markdown_text(str(reason)))
    for error in evidence.get('persistence_errors', []):
        lines.append('- **保存说明**: ' + _markdown_text(str(error.get('reason', '未知'))))
    return lines + ['']


def _disk_usage_lines(usage):
    components = (usage or {}).get("components") or []
    if not components:
        return ["- **磁盘统计**: 未测量（安装与执行后读取实际大小，不展示模型估值）"]
    state = {"measured": "已测量", "partial": "部分已测量"}.get(usage.get("status"), "未测量")
    lines = [f"- **磁盘统计**: {state}（各项分别列出，不合计为环境总占用）"]
    for component in components:
        name = _markdown_text(_txt(component.get("label"), "文件体积"))
        value = format_bytes(component.get("bytes"))
        status = component.get("status")
        if status not in {"measured", "partial"}:
            value = "未测量"
        elif status == "partial":
            value = f"已读取部分 {value}，统计不完整"
        if component.get("basis") == "file_stat" and status in {"measured", "partial"}:
            note = "共享缓存，含已有文件" if component.get("shared") else (
                "清理前快照" if not component.get("retained") else "执行后目录，含已有文件")
            value += f"（文件体积实测；{note}）"
        elif component.get("basis") == "docker_image_inspect" and status == "measured":
            value += "（Docker 实测的镜像逻辑大小，含共享层）"
        lines.append(f"  - **{name}**: {value}")
        if status != "measured" and component.get("note"):
            lines.append("    " + _markdown_text(str(component["note"])))
    lines.append("  > 文件体积、镜像大小均不等于本次新增磁盘占用；缓存和镜像层可能由多次任务共享。")
    excluded = usage.get("excluded")
    if excluded:
        lines.append("  > 未计入：" + _markdown_text(str(excluded)) + "。")
    return lines


def _analysis_sources(data):
    sources = {}
    for key in ("repository_analysis", "result_analysis"):
        for source in (data.get(key) or {}).get("sources") or []:
            if not isinstance(source, dict):
                continue
            source_id, locator, url = (source.get(field) for field in ("source_id", "locator", "url"))
            if all(isinstance(value, str) for value in (source_id, locator, url)):
                sources[(source_id, locator)] = url
    return sources


def _analysis_evidence_lines(analysis, sources):
    """Render only exact (source ID, locator) matches as public source links."""
    evidence = analysis.get("evidence") or {}
    if not isinstance(evidence, dict) or not evidence:
        return []
    lines = ["", "**原文依据**", ""]
    for claim, refs in evidence.items():
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            source_id, locator = _txt(ref.get("source_id")), _txt(ref.get("locator"))
            label = _markdown_text(source_id + " · " + locator)
            url = sources.get((source_id, locator))
            if isinstance(url, str) and re.fullmatch(r"https://[^\s<>]+", url):
                citation = f"[{label}](<{url}>)"
            else:
                citation = label + "（引用定位未匹配来源清单）"
            lines += [f"- **{_markdown_text(_txt(claim))}**: {citation}", ""]
            lines += ["> " + _markdown_text(line) for line in _txt(ref.get("quote")).splitlines()]
            lines.append("")
    return lines


def _analysis_stage_lines(analysis):
    stages = analysis.get("stages") or []
    lines = [f"- **真实调用数**: {analysis.get('calls', 0)}",
             "", "| Agent | 已尝试 | 返回完成 | 证据接受 | 实际调用数 | Tokens |",
             "|------|--------|----------|----------|------------|--------|"]
    labels = {"reader": "PaperReader", "finder": "ResourceFinder", "builder": "EnvBuilder",
              "verifier": "Verifier", "result_validator": "ResultValidator"}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = _txt(stage.get("name"))
        states = ["是" if stage.get(field) is True else "否" for field in ("attempted", "completed", "accepted")]
        usage = stage.get("usage") or {}
        tokens = usage.get("total_tokens", "未返回") if isinstance(usage, dict) else "未返回"
        cells = [labels.get(name, name), *states, _txt(stage.get("calls", 0)), _txt(tokens)]
        lines.append("| " + " | ".join(_markdown_text(cell) for cell in cells) + " |")
    return lines + [""]


def _analysis_rejection_lines(analysis, sources):
    """Show typed, sanitized rejection details without presenting them as facts."""
    labels = {"reader": "PaperReader", "finder": "ResourceFinder", "builder": "EnvBuilder",
              "verifier": "Verifier", "result_validator": "ResultValidator"}
    lines = []
    for stage in analysis.get("stages") or []:
        if not isinstance(stage, dict) or stage.get("accepted") is True:
            continue
        diagnostics = stage.get("rejection_diagnostics")
        if not isinstance(diagnostics, dict) or not diagnostics:
            continue
        name = _txt(stage.get("name"))
        attempt = stage.get("attempt")
        suffix = f"（第 {attempt} 次）" if type(attempt) is int and attempt > 0 else ""
        lines += ["#### " + _markdown_text(labels.get(name, name)) + "：分析未接受" + suffix, ""]
        if stage.get("reason"):
            lines.append("- **本地核验拒绝原因**: " + _markdown_text(_txt(stage["reason"])))
        if isinstance(diagnostics.get("status"), str):
            lines.append("- **模型返回状态**: " + _markdown_text(diagnostics["status"]))
        if isinstance(diagnostics.get("pass"), bool):
            lines.append("- **模型自查结论**: " + ("通过" if diagnostics["pass"] else "未通过"))
        reviewed = diagnostics.get("reviewed_stages")
        if isinstance(reviewed, list):
            selected = [_markdown_text(name) for name in reviewed if isinstance(name, str)]
            if selected:
                lines.append("- **模型审查对象**: " + ", ".join(selected))
        checks = diagnostics.get("checks")
        if isinstance(checks, dict):
            selected = [(key, value) for key, value in checks.items() if isinstance(value, bool)]
            if selected:
                lines += ["", "| 审查项 | 模型自查 |", "|--------|----------|"]
                for key, value in selected:
                    lines.append("| " + _markdown_text(_txt(key)) + " | " + ("通过" if value else "未通过") + " |")
                lines.append("")
        issues = diagnostics.get("issues")
        if isinstance(issues, list):
            selected = [issue for issue in issues if isinstance(issue, str) and issue.strip()]
            if selected:
                lines += ["", "**模型问题说明**", ""]
                lines += ["- " + _markdown_text(issue) for issue in selected]
        lines += _analysis_evidence_lines(diagnostics, sources)
        lines += ["", "> 以上为未被接受的分析诊断；原文引用保留供复核，不代表训练或数值复现结论。", ""]
    return lines


def _provenance_sources(references, sources):
    """Render deterministic source origins, linking only registered locators."""
    if not isinstance(references, list):
        return ""
    labels = {"paper_text": "论文原文", "author_script": "作者实验脚本",
              "author_code_setting_or_default": "作者代码设置或默认值",
              "author_statement": "作者公开说明", "author_source": "作者公开源码",
              "public_source": "公开来源"}
    rendered = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        source_id, locator = ref.get("source_id"), ref.get("locator")
        if not isinstance(source_id, str) or not isinstance(locator, str):
            continue
        origin = ref.get("origin")
        origin = labels.get(origin, _txt(origin, "来源类型未标注")) if isinstance(origin, str) else "来源类型未标注"
        label = _markdown_text(origin + " · " + source_id + " · " + locator)
        url = sources.get((source_id, locator))
        if isinstance(url, str) and re.fullmatch(r"https://[^\s<>]+", url):
            rendered.append(f"[{label}](<{url}>)")
        else:
            rendered.append(label + "（引用定位未匹配来源清单）")
    return "；".join(rendered)


def _reader_source_context_lines(reader, sources):
    provenance = reader.get("protocol_provenance")
    lines = []
    if isinstance(provenance, dict):
        context_rows = []
        for field, item in provenance.items():
            if not isinstance(item, dict):
                continue
            context = _provenance_sources(item.get("context_sources"), sources)
            if context:
                context_rows.append("| " + _markdown_text(_txt(field)) + " | " + context + " |")
        if context_rows:
            lines += ["", "**补充来源上下文**", "",
                      "这些引用提供背景；参数数值的依据列在参数表中。", "",
                      "| 字段 | 上下文来源 |", "|------|------------|", *context_rows, ""]
    context = reader.get("reference_context")
    if not isinstance(context, dict) or not context:
        return lines
    labels = {"method": "方法", "dataset": "数据集", "features": "特征模式",
              "input_length": "输入长度", "forecast_horizon": "预测长度", "scope": "实验范围"}
    lines += ["", "**论文表格定位（本地确定性选择）**", "",
              "来自已核验协议与公开引用的定位元数据；本次运行证据另行记录。", "",
              "| 定位项 | 值 |", "|--------|----|"]
    for key, label in labels.items():
        if key in context:
            lines.append("| " + label + " | " + _markdown_text(_txt(context[key])) + " |")
    metric_columns = context.get("metric_columns")
    if isinstance(metric_columns, dict):
        for metric in ("mse", "mae"):
            if isinstance(metric_columns.get(metric), str):
                lines.append("| " + metric.upper() + " 列 | " + _markdown_text(metric_columns[metric]) + " |")
    lines.append("")
    evidence = {field: context[field] for field in ("table_headers", "table_row_evidence")
                if isinstance(context.get(field), list)}
    lines += _analysis_evidence_lines({"evidence": evidence}, sources)
    return lines


def _discovered_plan_lines(data):
    plan = data.get("grounded_repository_plan") or {}
    if not plan:
        return []
    if plan.get('scope') == 'author_code_execution':
        lines = ['### 本次 PDF 与作者源码确定的运行计划', '',
                 '- **运行范围**: 所选作者入口及其原始参数。',
                 '- **结束条件**: 全部所选命令正常返回、退出码为 0，所属进程已结束并清理。',
                 '- **运行结果**: 记录作者输出和实际训练更新；不进行论文、协议或数值审核。',
                 '', '**原始作者命令**', '']
        for step in plan.get('steps', []):
            lines += ['- ' + _markdown_text(_txt(step.get('id'))) + '，目录：' +
                      _markdown_text(_txt(step.get('cwd'))),
                      '```json', json.dumps(step.get('argv', []), ensure_ascii=False), '```']
        return lines + ['']
    lines = ["### 本次 PDF 与作者源码确定的实验计划", "",
             "- **范围**: 引用所确定的单项完整实验。",
             "- **验收规则**: 训练开始前冻结参考值与 5% 相对容差；该容差是项目规则。",
             "- **实际运行要求**: 作者入口完成、原始源码校验通过、完整测试划分及独立模型重载评估通过。",
             "", "| 指标 | 论文参考 | 单位 | 引用 |", "|---|---|---|---|"]
    for item in plan.get("metrics", []):
        lines.append("| " + " | ".join(_markdown_text(_txt(value)) for value in (
            item.get("name"), item.get("reference"), item.get("unit"),
            ", ".join(item.get("citations", [])))) + " |")
    lines += ["", "**原始作者命令**", ""]
    for step in plan.get("steps", []):
        lines += ["- " + _markdown_text(_txt(step.get("id"))) + "，目录：" +
                  _markdown_text(_txt(step.get("cwd"))),
                  "```json", json.dumps(step.get("argv", []), ensure_ascii=False), "```"]
    capture = plan.get('capture') or {}
    protocol = plan.get('experiment_protocol') or data.get('experiment_protocol') or {}
    selection = protocol.get('model_selection') or {}
    rule = capture.get('completion') or {'kind': 'fixed_steps'}
    training_rule = (f"每次完整执行 {capture.get('expected_optimizer_steps')} 次优化器更新" if rule.get('kind') == 'fixed_steps'
        else f"作者原始停止规则；实际更新数须等于 ({rule.get('epoch_counter')} + {rule.get('epoch_offset')}) × {rule.get('updates_per_epoch')}，上限 {capture.get('expected_optimizer_steps')}")
    lines += ['', '**各阶段完成依据**', '', '| 阶段 | 必须核对的证据 |', '|---|---|',
              '| 训练 | ' + _markdown_text(training_rule) + '；作者进程正常退出 |',
              '| 模型选择 | ' + _markdown_text(str(selection.get('policy', '按独立协议审查确定的作者规则'))
                  + '；' + str(selection.get('criterion', '')) + '；' + str(selection.get('evaluation_weights', '实际捕获模型与作者报告一致'))) + ' |',
              f"| 重复实验 | 完成全部 {capture.get('expected_runs', 1)} 次原生运行，每次均有训练和独立评估记录 |",
              '| 测试数据 | 每次完整测试样本数：' + _markdown_text(str(capture.get('expected_test_samples'))) + '；标签、索引与模型输入对应 |',
              '| 独立评估 | 新进程重载实际模型，重新推理；NumPy 与 PyTorch 计算交叉核验，全部重复按原协议汇总 |',
              '| 数值验收 | 作者报告与独立评估一致，再按冻结论文参考值与容差判定；实验完成和数值达标分别记录 |']
    review = plan.get("semantic_review") or {}
    if review:
        lines += ["", "**执行前参考审查**" if review.get('advisory') else "**执行前独立协议审查**", "",
                  "- **状态**: " + ("通过" if review.get("accepted") is True else "未通过")]
        if review.get('advisory'):
            lines += ['- 审查意见作为参考；本次允许尝试运行，执行完成由真实进程和运行证据确认。']
        for name, check in (review.get("checks") or {}).items():
            status = "警告，须实际运行验证；" if check.get('severity') == 'warning' else ''
            lines.append("- " + _markdown_text(name) + "：" + status + _markdown_text(_txt(check.get("reason"))))
    lines += ["", "**计划引用原文**", ""]
    for citation in plan.get("citations", []):
        location = (f"PDF 第 {citation.get('page')} 页" if citation.get("source") == "pdf"
                    else _txt(citation.get("path")))
        lines += ["- " + _markdown_text(_txt(citation.get("id"))) + "：" + _markdown_text(location),
                  "", "> " + _markdown_text(_txt(citation.get("quote"))).replace("\n", "\n> "), ""]
    return lines


def _repository_analysis_lines(data):
    analysis = data.get("repository_analysis") or {}
    if not analysis:
        return []
    sources = _analysis_sources(data)
    state = "通过" if analysis.get("status") == "accepted" else "未完成或未通过"
    lines = ["### 公开来源多 Agent 分析", "",
             f"- **分析状态**: {state}",
             f"- **模型**: {_markdown_text(_txt(analysis.get('model')))}",
             "- **输入范围**: 真实公开论文摘录与固定作者源码；本阶段未发送训练结果。"]
    lines += _analysis_stage_lines(analysis)
    if (analysis.get("gate") or {}).get("reason"):
        lines += ["- **分析未通过原因**: " + _markdown_text(_txt(analysis["gate"]["reason"])), ""]
    lines += _analysis_rejection_lines(analysis, sources)
    analyses = analysis.get("analyses") or {}
    reader = analyses.get("reader") or {}
    if reader:
        provenance = reader.get("protocol_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        lines += ["#### PaperReader：选定实验协议", "",
                  "| 参数 | 抽取值 | 数值依据 |" if provenance else "| 参数 | 抽取值 |",
                  "|------|--------|----------|" if provenance else "|------|--------|"]
        for name, value in (reader.get("protocol") or {}).items():
            row = f"| {_markdown_text(_txt(name))} | {_markdown_text(_txt(value))} |"
            if provenance:
                item = provenance.get("protocol." + name)
                refs = item.get("value_sources") if isinstance(item, dict) else []
                row += " " + (_provenance_sources(refs, sources) or "未记录") + " |"
            lines.append(row)
        lines += ["", "- **论文参考指标**: " + _markdown_text(_txt(reader.get("reference_metrics")))]
        metric_rows = []
        for name, value in (reader.get("reference_metrics") or {}).items():
            item = provenance.get("reference_metrics." + name)
            if isinstance(item, dict):
                origin = _provenance_sources(item.get("value_sources"), sources)
                if origin:
                    metric_rows.append("| " + _markdown_text(_txt(name)) + " | "
                                       + _markdown_text(_txt(value)) + " | " + origin + " |")
        if metric_rows:
            lines += ["", "| 参考指标 | 数值 | 数值依据 |", "|----------|------|----------|", *metric_rows, ""]
        lines += _reader_source_context_lines(reader, sources)
        lines += _analysis_evidence_lines(reader, sources)
    finder = analyses.get("finder") or {}
    if finder:
        lines += ["#### ResourceFinder：作者源码入口", "",
                  "| 职责 | 固定源码路径 |", "|------|--------------|"]
        for name, value in (finder.get("entrypoints") or {}).items():
            lines.append(f"| {_markdown_text(_txt(name))} | {_markdown_text(_txt(value))} |")
        lines += _analysis_evidence_lines(finder, sources)
    builder = analyses.get("builder") or {}
    if builder:
        lines += ["#### EnvBuilder：原始依赖与兼容解释", "",
                  "- **作者原始依赖**: " + _markdown_text(", ".join(_txt(value) for value in builder.get("original_requirements") or [])),
                  "- **兼容解释**: " + _markdown_text(_txt(builder.get("compatibility_note"))),
                  "- **兼容候选状态**: 未执行；执行环境仍由冻结预设确定。"]
        if builder.get("original_requirements_basis") == "documented_author_setup_not_verified_runtime":
            lines.append("- **原始依赖证据范围**: 作者声明的安装清单；未核实论文实验当时实际安装的环境。")
        if builder.get("changes_algorithm_role") == "constraint_on_unexecuted_proposals":
            lines.append("- **算法变更约束**: 兼容候选应保留冻结算法；这是对未执行建议的约束，不提供运行核验证据。")
        for proposal in builder.get("compatibility_proposals") or []:
            if isinstance(proposal, dict):
                lines.append("  - " + _markdown_text(_txt(proposal.get("package"))
                             + _txt(proposal.get("suggested_constraint")) + ": " + _txt(proposal.get("reason"))))
        lines += _analysis_evidence_lines(builder, sources)
    verifier = analyses.get("verifier") or {}
    if verifier:
        lines += ["#### Verifier：独立准备条件审查", "",
                  f"- **准备条件审查**: {'通过' if verifier.get('pass') is True else '未通过'}",
                  "- **审查对象**: PaperReader、ResourceFinder、EnvBuilder 的公开事实及一致性；最终数值结论由本地确定性核验给出。"]
        for name, accepted in (verifier.get("checks") or {}).items():
            lines.append("  - " + _markdown_text(_txt(name)) + ": " + ("通过" if accepted is True else "未通过"))
        for issue in verifier.get("issues") or []:
            lines.append("  - " + _markdown_text(_txt(issue)))
        lines += _analysis_evidence_lines(verifier, sources)
    lines += ["", "> 分析范围由本次冻结的方法、数据集与实验设置确定，不代表论文全部实验。", ""]
    return lines


def _result_analysis_lines(data):
    analysis = data.get("result_analysis") or {}
    status = data.get("analysis_status")
    if not analysis and not status:
        return []
    states = {"completed": "已完成", "public_readiness_accepted": "公开准备条件已通过；未执行结果摘要解释",
              "public_readiness_rejected": "在线来源预审未通过；训练尚未开始",
              "result_analysis_failed": "结果摘要解释失败", "failed": "LLM 分析失败"}
    lines = ["### LLM 分析状态与结果摘要解释", "",
             "- **LLM 分析状态**: " + states.get(status, _txt(status, "未记录")),
             "- **结论职责**: API 分析状态与训练、协议核验、独立指标复算分别记录；API 解释失败不会覆盖已完成实验的确定性结论。"]
    if analysis:
        lines += ["- **结果解释输入范围**: 本次已记录的实际指标、训练进展、协议核验与独立复算状态摘要；参考依据为公开来源。"]
        lines += _analysis_stage_lines(analysis)
        accepted = (analysis.get("analyses") or {}).get("result_validator") or analysis
        if analysis.get("status") == "accepted":
            lines += ["**结果解释**", "", _markdown_text(_txt(accepted.get("summary"))), ""]
            differences = accepted.get("differences") or []
            if differences:
                lines += ["| 指标 | 论文值 | 实际值 | 绝对差值 | 相对差异 |",
                          "|------|--------|--------|----------|----------|"]
                for item in differences:
                    if isinstance(item, dict):
                        cells = [_txt(item.get(key)) for key in ("metric", "paper_value", "actual_value", "absolute_difference")]
                        cells.append(_fmt(item.get("relative_difference"), ".2%"))
                        lines.append("| " + " | ".join(_markdown_text(cell) for cell in cells) + " |")
                lines.append("")
                for item in differences:
                    if isinstance(item, dict) and item.get("explanation"):
                        lines.append("- " + _markdown_text(_txt(item.get("metric")) + ": " + _txt(item["explanation"])))
            for limitation in accepted.get("limitations") or []:
                lines.append("- **解释局限**: " + _markdown_text(_txt(limitation)))
            lines += _analysis_evidence_lines(accepted, _analysis_sources(data))
        elif (analysis.get("gate") or {}).get("reason"):
            lines.append("- **API 解释失败原因**: " + _markdown_text(_txt(analysis["gate"]["reason"])))
    if data.get("analysis_error"):
        lines.append("- **分析诊断**: " + _markdown_text(_txt(data["analysis_error"])))
    return lines + [""]


def _legacy_repair_lines(data):
    lines = []
    self_repair = data.get('self_repair') or {}
    if self_repair:
        outcomes = {'resolved': '原命令重跑通过', 'blocked': '缺少可验证的自动修复方案',
                    'retrying': '正在重跑验证', 'no_progress': '重跑后错误相同',
                    'failed_with_new_error': '重跑后出现新错误', 'interrupted': '已取消',
                    'timed_out': '已超时', 'budget_exhausted': '已达到修复次数上限'}
        lines += ['', '### 自动诊断与修复', '',
                  '- 状态：' + outcomes.get(self_repair.get('status'), '未完成') + '；'
                  + _markdown_text(_txt(self_repair.get('reason'))),
                  '- 每次修复都重跑原命令；此处记录运行恢复情况，论文数值核验另行记录。']
        for item in self_repair.get('attempts') or []:
            lines.append('- 第 ' + str(item.get('round')) + ' 轮：'
                + _markdown_text(_txt(item.get('action'))) + '；'
                + outcomes.get(item.get('status'), '未完成') + '；'
                + _markdown_text(_txt(item.get('reason'))))
    return lines


def _optimization_lines(optimization, *, is_rezero=False):
    lines = []
    # 6. 智能优化
    lines += ["## 6. 智能优化"]
    if optimization.get("mode") in {"suggest", "validate"}:
        lines += [f"- **状态**: {_markdown_text(_txt(optimization.get('status')))}",
                  f"- **说明**: {_markdown_text(_txt(optimization.get('reason')))}",
                  f"- **已验证提升**: {'是' if optimization.get('optimized') else '否'}"]
        for index, suggestion in enumerate(optimization.get("suggestions", []), 1):
            suggestion_status = {"untested": "尚未验证", "tested": "已完成候选试验", "failed": "试验失败",
                                 "validated_gain": "两种子留出确认通过"}.get(suggestion.get("status"), "尚未验证")
            lines += ["", f"### 建议 {index}（{suggestion_status}）",
                      f"- **参数**: {suggestion['parameter']}：{suggestion['baseline_value']} → {suggestion['value']}"]
            for key, label in (("hypothesis", "假设"), ("expected_effect", "预期效果"),
                               ("cost", "成本"), ("validation_plan", "验证方法")):
                lines.append(f"- **{label}**: {_markdown_text(suggestion[key])}")
            for evidence in suggestion.get("evidence", []):
                lines += [f"- **依据**: [{evidence['source_id']}]({evidence['url']}) · {evidence['locator']}",
                          "", "> " + _markdown_text(evidence["quote"]).replace("\n", "\n> ")]
        if optimization.get("trials"):
            lines += ["", "### 真实候选记录", "", "| 候选 | 状态 | 验证指标 | 耗时（秒） |",
                      "|---|---|---|---|"]
            for trial in optimization["trials"]:
                lines.append("| " + " | ".join(_markdown_text(_txt(trial.get(k))) for k in
                             ("candidate", "status", "metrics", "elapsed_s")) + " |")
        if optimization.get("confirmation"):
            lines += ["", "### 两种子留出确认", "", "```json",
                      json.dumps(optimization["confirmation"], ensure_ascii=False, indent=2), "```"]
        if optimization.get("mode") == "validate":
            lines += ["", "- **优化评估范围**: 固定验证集选优后，仅对选定候选做两个种子的留出确认；与官方全图/全轨迹拟合指标分别报告。",
                      f"- **优化协议基线**: {optimization.get('baseline', {})}",
                      f"- **已用预算**: {_fmt(optimization.get('budget_used_s'), '.2f')} / {optimization.get('budget_seconds', 7200)} 秒"]
    elif not optimization.get("optimized"):
        if is_rezero:
            lines.append("- **优化状态**: 关闭；本实验按冻结作者协议运行。")
        else:
            lines.append(f"- **优化状态**: 未触发("
                         f"{optimization.get('reason', '复现未成功或未运行优化')})")
    else:
        # 模拟优化必须与真实执行区分开：`_simulate_trial` 是拿方向名的
        # 哈希当"改进潜力"的假数据，与代码、指标都无关。不标注的话，
        # "改进幅度 1.80% / 最优结果 0.0908" 会被当成实测结果读。
        records = optimization.get("optimization_report", []) or []
        sim_types = {(r.get("detail") or {}).get("type") for r in records}
        simulated = bool(records) and "real_exec" not in sim_types

        state_text = "⚠️ 已优化（模拟）" if simulated else "✅ 已优化"
        lines += [f"- **优化状态**: {state_text}",
                  f"- **基线指标**: {optimization.get('baseline', 'N/A')}"]
        if simulated:
            lines += [
                "  > ⚠️ **本轮优化未真实执行**：以下「改进幅度 / 最优结果」"
                "由方向名哈希模拟得出（`ucb_mock`），**不是**跑出来的实测值，"
                "不能作为代码改进效果的依据。",
            ]
        lines += [f"- **最优方向**: {optimization.get('best_arm', 'N/A')}",
                  f"- **改进幅度**: {optimization.get('improvement', 0):.2%}"
                  + ("（模拟）" if simulated else ""),
                  f"- **最优结果**: {optimization.get('best_result', 'N/A')}"
                  + ("（模拟）" if simulated else ""),
                  f"- **预算使用**: {optimization.get('budget_used', 'N/A')} / "
                  f"{optimization.get('budget', 'N/A')} 次尝试",
                  "",
                  "### 尝试记录(UCB 预算调度)",
                  "| 方向 | 改进幅度 | 依据 | 判定 |",
                  "|------|----------|------|------|"]
        for r in records:
            mark = "✅ Keep" if r.get("kept") else "❌ Reject"
            rd = r.get("detail") or {}
            basis = {"real_exec": "真实执行",
                     "ucb_mock": "⚠️ 哈希模拟"}.get(rd.get("type") or "",
                                                   rd.get("type") or "未知")
            # 真实执行时补上"按哪个指标、哪个方向"判的——退回启发式选键
            # 时方向可能是错的，不写清楚就无从分辨。
            rb = rd.get("reward_basis") or {}
            if rb.get("metric_key"):
                arrow = "↓" if rb.get("direction") == "越小越好" else "↑"
                note = "" if rb.get("metric_source") == "论文声明指标" \
                    else "，启发式选键"
                basis += f" · {rb['metric_key']}{arrow}{note}"
            lines.append(f"| {r.get('arm')} | {r.get('improvement'):.4f} "
                         f"| {basis} | {mark} |")
    lines.append("")

    return lines


# 审计渲染保留完整代码与执行输出，不设截断上限。
# 面向阅读的正文保留实验设置、结果与解释；长原文、代码、日志写入校验过的
# 详情附件，可单独下载并随报告 ZIP 携带。


class ReportGeneratorAgent(BaseAgent):
    """生成 Markdown 格式的复现 + 优化报告。"""

    format_version = REPORT_FORMAT_VERSION

    def __init__(self, logger=None):
        super().__init__("ReportGenerator", logger)

    def run(self, input_data: dict, *, report_path=None) -> dict:
        """生成完整复现报告（含优化与审计信息）。"""
        from src.agent_collaboration import current_collaboration
        collaboration = current_collaboration()
        if collaboration:
            for message in self.collaboration_context():
                collaboration.reply(message, self.name, {
                    'status': 'received',
                    'reason': '已接收结果交接；报告以实际执行记录和核验字段为准。'}, kind='acknowledgement')
            collaboration.attach(input_data)
        self.log("generate_report", "START", "开始生成复现报告", input_data)

        report = self.build_report(input_data, report_path=report_path)

        self.log("generate_report", "SUCCESS",
                 f"报告生成完成 ({len(report)} 字符)",
                 {"report_length": len(report)})
        from src.result_conclusion import build_reproduction_conclusion
        from src.report_view_model import build_report_view_model
        return {"report": report, "report_length": len(report),
                "reproduction_conclusion": build_reproduction_conclusion(input_data),
                "report_view_model": build_report_view_model(input_data),
                "report_format_version": REPORT_FORMAT_VERSION}

    @staticmethod
    def build_report(data: dict, *, report_path=None, artifact_resolver=None) -> str:
        """Build the experiment report and retain full evidence in a companion."""
        from src.author_claim_comparison import _redact_evidence
        from src.report_revisions import sanitize_report_result
        from src.report_details import save_report_details
        from src.report_view_model import ReportViewModel
        path = report_path or data.get('report_path') or default_report_path()
        safe = _redact_evidence(sanitize_report_result(data), '')
        view = ReportViewModel.from_data(safe)
        full = ReportGeneratorAgent._build_report(safe, report_path=path, artifact_resolver=artifact_resolver)
        displayed = ReportGeneratorAgent._build_readable_report(
            view, safe, report_path=path, artifact_resolver=artifact_resolver)
        link = save_report_details(full, safe, path)
        return displayed.rstrip() + '\n\n## 完整依据与日志\n\n' + link + '\n'

    @staticmethod
    def _build_readable_report(view, data, *, report_path, artifact_resolver=None):
        """Render named presentation fields directly; never trim a full template."""
        from src.report_details import (intelligent_recovery_summary, presentation_data,
                                        recovery_contribution_lines, report_metric_lines)
        from src.result_conclusion import conclusion_markdown
        from src.report_view_model import report_overview_markdown
        artifact_resolver = artifact_resolver or markdown_image_target
        readable = presentation_data(data)
        paper = view.experiment['paper_info']
        spec, protocol = view.experiment['experiment_spec'], view.experiment['experiment_protocol']
        execution = data.get('execution') or {}
        final = execution.get('final') or {}
        env = execution.get('effective_env_config') or data.get('env_config') or {}
        validation = data.get('validation') or {}
        lines = ['# 论文复现实验报告', REPORT_FORMAT_MARKER, '',
                 f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", '',
                 '<!-- autoreproducer-overview:start -->', report_overview_markdown(view),
                 '<!-- autoreproducer-overview:end -->', '',
                 '<!-- autoreproducer-conclusion:start -->', conclusion_markdown(view.conclusion),
                 '<!-- autoreproducer-conclusion:end -->', '',
                 '## 选定实验与冻结依据', '',
                 '- **标题**: ' + _markdown_text(_txt(paper.get('title'), '未记录')),
                 '- **方法**: ' + _markdown_text(_txt(paper.get('method'), '未记录')),
                 '- **数据集**: ' + _markdown_text(_txt(paper.get('dataset'), '未记录'))]
        if spec:
            lines += ['- **实验定义**: ' + _markdown_text(_txt(spec.get('label') or spec.get('id'), '未记录')),
                      '- **冻结参数**: ' + _markdown_text(_txt(spec.get('parameters'), '未记录'))]
        if protocol:
            for key, label in (('method', '方法与变体'), ('dataset', '数据与划分'),
                               ('training', '训练协议'), ('repetitions', '重复次数与汇总')):
                if protocol.get(key):
                    lines.append('- **' + label + '**: ' + _markdown_text(_txt(protocol[key])))
        coverage = data.get('reading_coverage') or {}
        if coverage:
            lines.append('- **论文阅读覆盖**: ' + ('完成' if coverage.get('complete') else '未完成')
                         + f"；{coverage.get('read_characters', '未记录')}/{coverage.get('total_characters', '未记录')} 字符")
        pdf = data.get('pdf_input') or data.get('pdf_resolution') or {}
        if pdf.get('sha256'):
            lines.append('- **来源 PDF SHA-256**: `' + _markdown_text(str(pdf['sha256'])) + '`')
        if data.get('paper_markdown'):
            from src.pdf_markdown import report_markdown_targets
            try:
                lines += report_markdown_targets(data, report_path)
            except (OSError, ValueError, KeyError):
                lines.append('- **论文 Markdown 附件**: 未能核对附件，原运行目录与读取结果保留。')
        resources, repository = data.get('resources') or {}, data.get('repository') or {}
        lines += ['', '## 资源与执行准备', '',
                  '- **代码仓库**: ' + _markdown_text(_txt(resources.get('code_repo_url') or repository.get('url'), '未取得')),
                  '- **固定源码版本**: ' + _markdown_text(_txt(repository.get('resolved_sha'), '未记录')),
                  '- **Python**: ' + _markdown_text(_txt(env.get('python_executable') or env.get('python_version'), '未记录')),
                  '- **运行设备**: ' + _markdown_text(_txt(env.get('device'), '未记录'))]
        if repository.get('archive_sha256'):
            lines.append('- **源码归档 SHA-256**: ' + _markdown_text(str(repository['archive_sha256'])))
        if repository.get('path'):
            lines.append('- **独立源码副本**: ' + _inline_code(repository['path']))
        if repository:
            lines.append('- 源码已获取；实际执行来源以执行记录为准。')
        source_request = data.get('source_request') or {}
        if source_request.get('status') == 'needs_pdf':
            lines += ['- **执行准备**: 待补充论文 PDF；尚未准备执行命令。',
                      '- **执行范围**: 仓库实验尚未执行。',
                      '- **下一步**: ' + _markdown_text(_txt(source_request.get('next_action') or source_request.get('reason')))]
            if source_request.get('selected_repository'):
                lines.append('- **候选仓库（作者归属未核验）**: '
                             + _markdown_text(_txt(source_request['selected_repository'])))
        identity = resources.get('repository_identity') or {}
        if identity.get('reason'):
            lines.append('- **仓库关联证据**: ' + _markdown_text(str(identity['reason'])))
        evidence = resources.get('selection_evidence') or identity.get('evidence') or {}
        if evidence:
            page = evidence.get('page')
            location = f'PDF 第 {page} 页' if type(page) is int and page > 0 else '页码未记录'
            lines += ['- **仓库来源位置**: ' + location + '；来源：' + _inline_code(evidence.get('source') or '未记录'),
                      '- **仓库证据类型**: ' + _inline_code(evidence.get('evidence_type') or '未记录')]
            if evidence.get('url'):
                lines.append('- **来源仓库链接**: ' + _markdown_text(str(evidence['url'])))
        if env.get('requirements_txt'):
            lines.append('- **依赖清单**: ' + _markdown_text(env['requirements_txt']))
        lines += _discovered_plan_lines(readable)
        lines += _source_repair_lines(readable, report_path)
        lines += _repository_analysis_lines(readable)
        lines += ['', '## 实际执行记录', '',
                  '- **运行状态**: ' + _markdown_text(view.conclusion['execution']['label']),
                  '- **退出码**: ' + _markdown_text(_txt(final.get('exit_code'), '未记录')),
                  '- 完整原始命令、标准输出、诊断日志和补丁保存在详情与证据附件中。']
        if not final.get('stdout'):
            lines.append('- **标准输出**: 无输出或未记录。')
        scope = execution.get('reproduction_scope')
        if scope == 'generated_reconstruction':
            lines.append('- **执行对象**: 模型依据可用论文证据生成的单文件实现；未运行已发现仓库。')
        if execution.get('not_runnable'):
            lines += ['- **执行状态**: ⚠️ 未运行；无法验证（代码未运行）。',
                      '- **未运行原因**: ' + _markdown_text(_txt(execution.get('reason')))]
        if execution.get('evidence_status') == 'insufficient_evidence':
            lines.append('- **证据状态**: 无法核验（论文证据不足，未执行论文实验）。')
        if execution.get('best_effort'):
            state = '✅ 成功' if final.get('success') else '❌ 失败'
            lines += ['- **执行状态**: ' + state + '（尽力而为：论文信息不足）',
                      '- **复现状态**: ⚠️ 无法核对；本次输出不是论文结论。',
                      '- **生成方式**: ' + ('Mock 本地演示脚本' if execution.get('fallback_used') else '依据可用证据生成的实现')]
            for assumption in execution.get('assumptions') or []:
                lines.append('  - 假设: ' + _txt(assumption))
        stats = execution.get('sanitize_stats') or {}
        dropped = int(stats.get('code_dropped') or 0)
        prose = int(stats.get('prose_dropped') or 0)
        if dropped:
            lines.append(f'- **⚠️ 清洗丢弃**: {dropped} 行疑似代码行；生成代码可能已被截短。')
        elif prose:
            lines.append(f'- **清洗丢弃**: 叙述行 {prose} 行（预期行为，未触碰代码）。')
        steps = execution.get('steps') or execution.get('stages') or (data.get('execution_plan') or {}).get('steps') or []
        for step in steps:
            argv = step.get('argv') or (step.get('step') or {}).get('argv')
            if argv:
                lines.append('- **命令**: ' + _markdown_text(json.dumps(argv, ensure_ascii=False)))
            if step.get('skipped') or step.get('not_run'):
                lines.append('- **未执行原因**: ' + _markdown_text(_txt(step.get('reason') or step.get('not_run'))))
        for step in execution.get('skipped') or []:
            lines.append('- **未执行步骤**: ' + _markdown_text(_txt(step.get('id'))) + '；'
                         + _markdown_text(_txt(step.get('not_run') or step.get('blocked_by') or step.get('reason'), '未记录原因')))
        lines += ['', '## 指标与参考比较', '', *report_metric_lines(view.metrics)]
        if validation.get('status') == 'author_code_completed':
            boundary = ('作者仓库代码运行完成（已对比作者输出，尚未独立核验论文数值）'
                        if (validation.get('author_claim_comparison') or {}).get('comparisons') else
                        '作者仓库代码运行完成（未进行论文数值审核）')
            lines.append('- **作者执行范围**: ' + boundary)
        lines += _author_claim_comparison_lines((readable.get('validation') or {}).get('author_claim_comparison'), include_table=False)
        inner = validation.get('validation') or {}
        if inner.get('differences'):
            lines += ['', '**指标差异**', *['- ' + _markdown_text(str(value)) for value in inner['differences']]]
        if inner.get('missing_metrics'):
            lines += ['', '**无法比对的指标**', *['- ' + _markdown_text(str(value)) for value in inner['missing_metrics']]]
        if validation.get('reason'):
            lines.append('- **判定依据**: ' + _markdown_text(str(validation['reason'])))
        records = validation.get('metric_records') or []
        if records:
            lines += ['', '### 实际指标证据', '', '| 指标 | 实测值 | 单位 | 数据划分 | 阶段 | 来源 |',
                      '|---|---|---|---|---|---|']
            for record in records:
                cells = [record.get('name'), _metric_value_text(record) if record.get('value') is not None else '未取得',
                         record.get('unit') or '未记录', record.get('split') or '未记录',
                         record.get('stage') or '未记录', record.get('source') or '未记录']
                lines.append('| ' + ' | '.join(_markdown_text(_txt(cell)) for cell in cells) + ' |')
        lines += _numeric_compatibility_lines(validation)
        lines += _runtime_dataset_lines(readable.get('runtime_dataset_evidence') or validation.get('runtime_dataset_evidence'))
        lines += _outcome_diagnosis_lines(readable)
        lines += _result_analysis_lines(readable)
        lines += _legacy_repair_lines(readable)
        if data.get('optimization'):
            lines += _optimization_lines(readable['optimization'], is_rezero=spec.get('adapter_id') == 'rezero')
        if view.collaboration['messages_recorded']:
            lines += ['', '## Agent 协作摘要', '',
                f"已保存 {view.collaboration['messages_recorded']} 条阶段交接与核查消息；"
                f"实际执行 {view.collaboration['actions_executed']} 项恢复动作，"
                f"其中 {view.collaboration['actions_verified']} 项效果已验证。",
                '收到消息或确认交接不计为完成修复；采用动作及验证见恢复贡献表，完整消息见详情附件。',
                '- **参与角色**: ' + _markdown_text('、'.join(view.collaboration['participants']))]
        lines += ['', *recovery_contribution_lines(view.recovery_contributions, recorded=view.recovery_recorded)]
        summary = intelligent_recovery_summary(data.get('intelligent_recovery'))
        if summary:
            lines += ['', summary]
        usage = view.recovery_usage
        if usage.get('roles'):
            lines += ['', '## 角色用量与耗时', '', '| 角色 | 模型请求 | token | 耗时（秒） | 费用 |', '|---|---|---|---|---|']
            for role, item in usage['roles'].items():
                if not isinstance(item, dict):
                    continue
                cells = [role, item.get('model_requests'), item.get('total_tokens'),
                         item.get('duration_seconds'), item.get('monetary_cost')]
                lines.append('| ' + ' | '.join('未知' if value is None else _markdown_text(str(value)) for value in cells) + ' |')
        audit = data.get('audit_stats') or {}
        if audit:
            lines += ['', '## 运行与用量统计', '',
                      '- **总步骤数**: ' + _markdown_text(_txt(audit.get('total_steps'), '未记录')),
                      '- **执行时长（秒）**: ' + _markdown_text(_txt(audit.get('duration_sec'), '未记录')),
                      '- **LLM 调用次数**: ' + _markdown_text(_txt(audit.get('llm_calls'), '未记录'))]
        artifacts = final.get('artifacts') or []
        if artifacts:
            lines += ['', '## 本次运行生成的图片', '']
        for index, artifact in enumerate(artifacts, 1):
            target = artifact_resolver(artifact, report_path)
            if target:
                lines += ['**图 ' + str(index) + '**：' + _markdown_text(_txt(artifact.get('name'), f'图 {index}')),
                          '', f'![运行结果图 {index}]({target})', '']
            else:
                lines.append(f'> 图 {index} 无法读取，图片文件可能已被清理。')
        for warning in final.get('artifact_warnings') or []:
            lines.append('> ' + _markdown_text(str(warning)))
        from src.run_finalization import finalize_report_text
        return finalize_report_text('\n'.join(lines), dict(data))

    @staticmethod
    def _build_report(data: dict, *, report_path=None, artifact_resolver=None) -> str:
        report_path = report_path or data.get("report_path") or default_report_path()
        artifact_resolver = artifact_resolver or markdown_image_target
        paper_info = data.get("paper_info", {}) or {}
        resources = data.get("resources", {}) or {}
        execution = data.get("execution", {}) or {}
        env_config = execution.get("effective_env_config") or data.get("env_config", {}) or {}
        validation = data.get("validation", {}) or {}
        reviews_removed = bool(validation.get('reviews_removed') or
                               ((data.get('experiment_spec') or {}).get('validation') or {}).get('reviews_removed') or
                               ((data.get('grounded_repository_plan') or {}).get('scope') == 'author_code_execution'))
        optimization = data.get("optimization", {}) or {}
        audit = data.get("audit_stats", {}) or {}
        is_rezero = (data.get("experiment_spec") or {}).get("adapter_id") == "rezero"

        lines = ["# 论文复现与优化报告",
                 REPORT_FORMAT_MARKER,
                 "",
                 f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"**工具**: AutoReproducer v0.2.0",
                 ""]
        if data.get("interrupted"):
            lines += ["> **本次任务已中断。** 已完成的基线及试验记录保留，未完成部分不构成成功结论。", ""]

        # 1. 论文信息
        lines += ["## 1. 论文信息",
                  f"- **标题**: {paper_info.get('title', '未知')}",
                  f"- **方法**: {paper_info.get('method', '未知')}",
                  f"- **数据集**: {paper_info.get('dataset', '未知')}",
                  f"- **{'作者公开参考指标' if is_rezero else '声明指标'}**: {paper_info.get('metrics', {})}",
                  ""]
        coverage = data.get('reading_coverage') or {}
        extraction = (data.get('pdf_input') or {}).get('extraction') or {}
        if extraction:
            labels = {'converted': '全部页面已转换', 'partial': '部分页面使用原提取文本',
                      'fallback': '已回退到原提取文本'}
            lines += ['- **PDF 读取格式**: Markdown（MarkItDown）；'
                      + labels.get(extraction.get('status'), '转换状态未知')
                      + f"，转换 {extraction.get('converted_pages', 0)}/{extraction.get('page_count', 0)} 页。",
                      f"- **表格结构补充**: {extraction.get('table_pages', 0)} 页保留完整原文并附带按 PDF 坐标核对的 Markdown 表格；重复展示不算新增证据。",
                      '- **数值来源**: 保留原 PDF 哈希与页码；逐页核对数值和可读内容，转换异常时使用完整原文。', '']
            from src.pdf_markdown import report_markdown_targets
            try:
                lines += report_markdown_targets(data, report_path)
            except (OSError, ValueError, KeyError):
                lines += ['- **论文 Markdown 附件**: 未能核对附件，原运行目录与读取结果保留。']
            lines.append('')
        if coverage:
            title_only = coverage.get('scope') == 'provided_title'
            lines += [f"- **{'标题输入阅读' if title_only else '论文全文阅读'}**: {'完成' if coverage.get('complete') else '未完成'}；"
                      f"{coverage.get('read_characters', 0)}/{coverage.get('total_characters', 0)} 字符；"
                      f"{coverage.get('completed_batches', 0)}/{coverage.get('total_batches', 0)} 批",
                      ("- **阅读范围**: 当前仅有标题，尚未取得论文全文。" if title_only else
                       "- **阅读完成依据**: 连续覆盖全部提取文本，所有批次均返回有效结果；汇总笔记用于定位，实验判定引用原文。"), ""]

        # 2. 资源定位
        lines += ["## 2. 资源定位",
                  f"- **代码仓库**: {resources.get('code_repo_url', '未找到')}",
                  f"- **数据集**: {resources.get('dataset_url', '未找到')}"]
        from src.repository_confidence import repository_confidence_lines
        lines += repository_confidence_lines(data)
        if resources.get("discovery_repository_url"):
            lines += [f"- **论文原文发现仓库**: {resources['discovery_repository_url']}",
                      f"- **实际固定训练仓库**: {resources.get('training_repository_url', resources.get('code_repo_url', '未找到'))}"]
        relationship = resources.get("repository_relationship") or {}
        if relationship.get("description"):
            lines.append("- **作者仓库关联链**: " + _markdown_text(_txt(relationship["description"])))
        for index, hop in enumerate(relationship.get("hops") or [], 1):
            if not isinstance(hop, dict):
                continue
            lines += [f"- **作者链接第 {index} 跳**: {hop.get('source_url', '')} → {hop.get('target_repository_url', '')}",
                      f"  - **固定来源 SHA-256**: `{hop.get('sha256', '')}`",
                      "  - **原文**: " + _markdown_text(_txt(hop.get("quote")))]
        identity = resources.get("repository_identity") or {}
        if identity:
            lines += [f"- **仓库关联证据**: {identity.get('reason', '未核验')}",
                      "- **执行范围**: 定位仓库是资源发现结果；是否运行仓库由下方执行对象和实际步骤确认。"]
            if identity.get("source") == "user_supplied_pinned_git_llm_review":
                lines += ["- **仓库提供方式**: 用户在自动确认失败后手动提供",
                          "- **作者身份**: " + ("经证据确认" if identity.get("author_identity_confirmed") else "未确认；仅确认论文与实现内容匹配")]
        review = identity.get("review") or {}
        if review.get("accepted"):
            lines += [f"- **仓库语义核验**: {review.get('verdict', '')}",
                      f"- **核验 Git commit**: `{review.get('repository_revision', '')}`"]
            for gate, check in (review.get("checks") or {}).items():
                lines.append(f"- **{gate}**: " + _markdown_text(str(check.get("reason", ""))))
                for citation in check.get("citations") or []:
                    source = (f"PDF 第 {citation.get('page')} 页" if citation.get("source") == "pdf"
                              else citation.get("url", citation.get("path", "")))
                    lines.append("  - " + _markdown_text(source) + "：" + _markdown_text(str(citation.get("quote", ""))))
        evidence = resources.get("selection_evidence") or identity.get("evidence") or {}
        if evidence:
            page = evidence.get("page")
            lines += [f"- **PDF 仓库来源**: {evidence.get('source', '未记录')}"
                      + (f"，第 {page} 页" if page else ""),
                      f"- **原文仓库链接**: {evidence.get('url', '')}",
                      f"- **证据类型**: {evidence.get('evidence_type', 'repository_link')}"]
            if evidence.get("context"):
                lines.append("- **原文上下文**: " + _markdown_text(str(evidence["context"])))
            pdf_input = data.get("pdf_input") or data.get("pdf_resolution") or {}
            if pdf_input.get("sha256"):
                lines.append(f"- **来源 PDF SHA-256**: `{pdf_input['sha256']}`")
        urls = resources.get("extracted_urls", []) or []
        if urls:
            lines.append(f"- **从论文中提取URL**: {len(urls)} 个")
            for u in urls[:5]:
                lines.append(f"  - {u}")
        lines.append("")

        repository = data.get("repository") or {}
        protocol = data.get('experiment_protocol') or {}
        if protocol:
            method, dataset = protocol.get('method', {}), protocol.get('dataset', {})
            repetitions = protocol.get('repetitions', {})
            lines += ['### 选定实验的协议判别',
                      '- **方法与变体**: ' + _markdown_text(str(method.get('name', '')) + ' / ' + str(method.get('variant', ''))),
                      '- **数据集与划分**: ' + _markdown_text(str(dataset.get('name', '')) + ' / ' + str(dataset.get('split', ''))),
                      '- **预处理**: ' + _markdown_text(str(dataset.get('preprocessing', ''))),
                      '- **训练协议**: ' + _markdown_text(str(protocol.get('training', {}).get('description', ''))),
                      f"- **完整运行次数与汇总**: {repetitions.get('count')} / {repetitions.get('aggregation')}",
                      '- **论文与源码对应关系**: ' + _markdown_text(str(protocol.get('alignment', {}).get('reason', ''))),
                      f"- **协议 SHA-256**: `{protocol.get('protocol_sha256', '')}`", '']
        spec = data.get("experiment_spec") or {}
        if spec:
            lines += ["### 用户提供的仓库实验来源" if identity.get("source") == "user_supplied_pinned_git_llm_review" else "### 官方仓库实验来源",
                      f"- **实验定义**: {spec.get('label', spec.get('id', ''))}",
                      f"- **论文版本/参考位置**: {paper_info.get('reference_source', '')}",
                      f"- **固定源码版本**: `{repository.get('resolved_sha', '未获取')}`",
                      f"- **实验定义SHA-256**: `{data.get('spec_sha256', '')}`",
                      f"- **独立运行目录**: `{data.get('_report_source_run_dir', data.get('run_dir', ''))}`",
                      f"- **参数**: {spec.get('parameters', {})}"]
            dataset = data.get("dataset_provenance") or {}
            if dataset:
                if dataset.get('verified'):
                    lines += [f"- **真实数据SHA-256**: `{dataset.get('sha256', '')}`",
                              f"- **数据切分**: {dataset.get('split', '')}"]
                else:
                    lines += ['- **数据来源核验**: 尚未完整核验；作者程序负责加载输入，不据此认定数据规模或论文数值通过。']
            lines += [f"- **协议说明**: {(spec.get('validation') or {}).get('note', '')}", ""]
            lines += [f"- **作者协议来源**: {(spec.get('validation') or {}).get('aggregation_source', '')}",
                      f"- **实现版本说明**: {(spec.get('validation') or {}).get('implementation_note', '')}", ""]
        elif repository.get("resolved_sha"):
            lines += ["### 固定源码获取记录",
                      f"- **获取仓库**: {repository.get('url', '')}",
                      f"- **请求版本**: `{repository.get('revision') or '远端 HEAD（获取时解析）'}`",
                      f"- **固定源码版本**: `{repository['resolved_sha']}`",
                      f"- **源码归档SHA-256**: `{repository.get('archive_sha256', '')}`",
                      f"- **独立源码副本**: `{repository.get('path', '')}`",
                      "- 此记录证明源码已获取；实际执行来源以执行记录为准。", ""]
        lines += _repository_analysis_lines(data)
        lines += _discovered_plan_lines(data)
        lines += _source_repair_lines(data, report_path)

        # 3. 环境配置 + 依赖诊断
        lines += ["## 3. 环境配置",
                  f"- **Python版本**: {env_config.get('python_version', 'N/A')}",
                  f"- **依赖数**: "
                  f"{len(_txt(env_config.get('requirements_txt')).splitlines())}"]
        if env_config.get("note"):
            lines.append(f"- **环境适配**: {env_config['note']}")
        if data.get('training_python_attempts'):
            lines += ['', '### 训练解释器尝试', '']
            for chosen in data['training_python_attempts']:
                conflict = chosen.get('selected_for') or {}
                lines.append('- CPython ' + _markdown_text(str(chosen.get('python_version', '?')))
                    + '：' + _markdown_text(str(chosen.get('source', '已选择')))
                    + ('；根据实际请求包 ' + _markdown_text(str(conflict.get('package')))
                       + ' 的 Requires-Python 约束切换' if conflict else ''))
        runtime_recovery = data.get("runtime_preparation") or {}
        prepared_env = execution.get("environment") or {}
        health_checks = prepared_env.get("dependency_health_attempts") or []
        cache_repairs = prepared_env.get("dependency_cache_repairs") or []
        if runtime_recovery or health_checks or cache_repairs:
            lines += ["", "### 自动环境恢复", ""]
            if runtime_recovery:
                lines.append("- **运行环境**: " + _markdown_text(runtime_recovery.get("reason", "已自动准备兼容环境")))
            if health_checks:
                passed = sum(check.get("success") is True for check in health_checks)
                lines.append(f"- **实际依赖检查**: {passed}/{len(health_checks)} 次通过（导入、版本与原生运算）")
            if cache_repairs:
                succeeded = sum(repair.get("status") == "succeeded" for repair in cache_repairs)
                lines.append(f"- **缓存修复**: {succeeded}/{len(cache_repairs)} 项完成；依赖版本与实验协议保持冻结")
        source_recoveries = data.get("public_source_recovery") or []
        lines += _legacy_repair_lines(data)
        runtime_repairs = data.get('dependency_repair_attempts') or []
        if runtime_repairs:
            lines += ['', '### 根据运行报错补充依赖', '']
            for repair in runtime_repairs:
                lines.append('- 第 ' + str(repair['attempt']) + ' 轮：缺少 ' +
                    _markdown_text(repair['module']) + '，安装 ' + _markdown_text(repair['requirement']) +
                    ' 后重试；原失败日志保留在运行目录。')
        if data.get('model_dependency_repairs'):
            lines += ['', '### 模型建议并通过静态校验的依赖调整', '']
            for repair in data['model_dependency_repairs']:
                lines.append('- 第 ' + str(repair['attempt']) + ' 轮：'
                    + _markdown_text(repair['previous']) + ' → '
                    + _markdown_text(repair['requirement']) + '；仅调整项目推导版本，重新验证原命令。')
        if data.get('failure_analyses'):
            lines += ['', '### 执行失败分析', '']
            for analysis in data['failure_analyses']:
                lines.append('- ' + _markdown_text(str(analysis.get('failure_kind', 'execution_failure')))
                    + '；动作：' + _markdown_text(str(analysis.get('action', 'blocked')))
                    + '；' + _markdown_text(str(analysis.get('reason') or
                        analysis.get('analysis_error') or '保留原始错误与所需条件')))
        if data.get('path_repair_attempts'):
            lines += ['', '### 根据运行报错修复数据目录', '']
            for repair in data['path_repair_attempts']:
                lines.append('- 第 ' + str(repair['attempt']) + ' 轮：作者临时目录 ' +
                    _markdown_text(repair['original_root']) + ' 不存在，改用任务内目录 ' +
                    _markdown_text(repair['replacement_root']) + ' 后重试；原源码与失败日志保留。')
        if source_recoveries:
            lines += ["", "### 公开证据自动恢复", "",
                      f"- **恢复来源数**: {len(source_recoveries)}；从固定官方网页读取原文并校验身份、定位与哈希"]
        disk_usage = (execution.get("final") or {}).get("disk_usage") or env_config.get("disk_usage")
        lines += _disk_usage_lines(disk_usage)
        if env_config.get("dependency_corpus"):
            lines.append("- **语料依赖来源**: " + _markdown_text(
                str(env_config["dependency_corpus"])))
        diag = env_config.get("dependency_diagnosis", {}) or {}
        if diag:
            lines += ["", "### 依赖诊断（5轮循环）",
                      f"- **状态**: {diag.get('overview', 'N/A')}",
                      f"- **诊断轮数**: {diag.get('rounds', 0)} / {diag.get('max_rounds', 0)}"]
            for rnd in diag.get("rounds_detail", []):
                issues = rnd.get("issues", [])
                if not issues:
                    lines.append(f"- 第 {rnd.get('round')} 轮: ✅ 无问题,依赖验证通过")
                    continue
                descs = "; ".join(
                    f"{i.get('type')}({i.get('package', '')})" for i in issues)
                fixs = "; ".join(a.get("action", "")
                                 for a in rnd.get("fix_actions", []))
                lines.append(
                    f"- 第 {rnd.get('round')} 轮: 🔧 发现 {descs} -> {fixs}")
        lines += ["", "### requirements.txt", "```",
                  _txt(env_config.get("requirements_txt"), "无"), "```", ""]
        selection = env_config.get("runtime_dependency_selection") or {}
        if selection.get("added") or selection.get("removed"):
            lines += ["### 执行前依赖校准", ""]
            if selection.get("added"):
                lines.append("- **按代码补全**: " + _markdown_text(", ".join(selection["added"])))
            if selection.get("removed"):
                lines.append("- **移除未使用的模型猜测包**: " + _markdown_text(", ".join(selection["removed"])))
            lines.append("")

        # 4. 代码执行（smoke + full）
        lines += ["## 4. 代码执行"]
        needs_pdf = validation.get("status") == "needs_pdf"
        if needs_pdf:
            lines.append("- **执行准备**: 待补充论文 PDF；尚未准备执行命令。")
            candidate = (data.get("source_request") or {}).get("selected_repository")
            if candidate:
                lines.append("- **候选仓库（作者归属未核验）**: " + _markdown_text(_txt(candidate)))
        elif execution.get("mode") == "repository":
            lines.append("- **执行对象**: " + ('兼容补丁后的独立作者源码副本（多文件）；原始源码保留'
                if data.get('execution_source') == 'patched_author_repository' else '固定版本作者仓库中的实验入口（多文件）'))
        else:
            lines.append(f"- **代码长度**: {len(_txt(execution.get('code')))} 字符")
            scope = execution.get("reproduction_scope")
            if scope:
                lines.append("- **执行对象**: " + {
                    "generated_reconstruction": "模型依据可用论文证据生成的单文件实现；未运行已发现仓库。",
                    "user_supplied_code": "用户提供的代码；系统未据此确认作者仓库执行。",
                    "mock_demo": "Mock 演示代码，不能用作真实论文复现证据。",
                }.get(scope, _txt(scope)))
        # 清洗记录：清洗层碰过代码就必须让人看见——丢掉疑似代码行是"结果可能
        # 已被洗残"的信号，静默吞掉正是此前"代码不完整却查不出来"的成因。
        sstats = execution.get("sanitize_stats") or {}
        prose_n = int(sstats.get("prose_dropped", 0) or 0)
        code_n = int(sstats.get("code_dropped", 0) or 0)
        if code_n:
            lines.append(f"- **⚠️ 清洗丢弃**: {code_n} 行疑似代码行"
                         f"（其余叙述行 {prose_n} 行）——"
                         f"生成代码可能已被清洗截短，请对照审计日志核对")
        elif prose_n:
            lines.append(f"- **清洗丢弃**: 叙述行 {prose_n} 行"
                         f"（预期行为，未触碰代码）")
        stages = execution.get("stages", []) or []
        final = execution.get("final", {}) or {}
        best_effort = bool(execution.get("best_effort"))
        if execution.get("not_runnable"):
            lines.append("- **执行状态**: ⚠️ 未运行" +
                         ("（仓库实验尚未执行）" if execution.get("mode") == "repository" else
                           "（论文证据不足，未获得实现）" if execution.get("evidence_status") == "insufficient_evidence" else
                          "（代码未通过执行前检查）"))
            lines.append(f"- **未运行原因**: {execution.get('reason', 'N/A')}")
        elif execution.get("mode") == "repository" and execution.get("executed") is False:
            lines.append("- **执行状态**: ⏳ 未运行（训练尚未开始）")
        else:
            state = "⏹ 已中断" if final.get("cancelled") else "✅ 成功" if final.get("success") else "❌ 失败"
            if best_effort:
                state += "（尽力而为：论文信息不足）"
            lines.append(f"- **执行状态**: {state}")
        if best_effort:
            # An attempted reconstruction remains inconclusive while its paper
            # evidence is incomplete, including when the script exits cleanly.
            lines += [
                "- **⚠️ 信息不足**: "
                + _txt(execution.get("best_effort_reason"), "论文信息不足"),
                "- **生成方式**: "
                + ("Mock 本地演示脚本（不属于论文实现）"
                   if execution.get("fallback_used")
                    else "模型尝试依据已知论文证据重建；关键证据不足时明确记录未运行"),
            ]
            for a in execution.get("assumptions") or []:
                lines.append(f"  - 假设: {a}")
        for st in stages:
            st_ok = st.get("success")
            lines.append(
                f"  - {st.get('stage')}: {'✅ 通过' if st_ok else '❌ 失败'} "
                f"(退出码 {st.get('exit_code')})")
        if execution.get("mode") == "repository" and not needs_pdf:
            lines += ["", "### 官方仓库执行命令", ""]
            steps = stages or (data.get("execution_plan") or {}).get("steps", [])
            for step in steps:
                lines += [f"- **{step.get('id', step.get('stage', '步骤'))}** · cwd: `{step.get('cwd', '.')}`",
                          "```json", json.dumps(step.get("argv", []), ensure_ascii=False), "```"]
                if step.get("stdout_path"):
                    lines.append(f"- 完整日志: `{step['stdout_path']}`")
            skipped = execution.get("skipped") or []
            if skipped:
                # 被前置条件挡住的步骤不会进 attempts，上面的命令列表里因此只剩
                # 真正跑过的那些。不写出来，报告会被读成"这几个不必跑"。
                lines += ["", "### 未执行的步骤", ""]
                for item in skipped:
                    if item.get("not_run") == "interrupted":
                        why = "任务已中断"
                    elif item.get("not_run") == "missing_requirement":
                        why = f"缺少前置产物 `{_txt(item.get('requirement'), '未知')}`"
                    else:
                        why = ("前置步骤 "
                               + _markdown_text(_txt(item.get("blocked_by"), "失败")) + " 未通过")
                    lines.append(f"- **{_markdown_text(_txt(item.get('id'), '步骤'))}**：未执行（{why}）")
                lines += ["", "> 被挡住的步骤按失败计入结论，不代表它可选。", ""]
        artifacts = final.get("artifacts") or []
        if artifacts:
            lines += ["", "### 运行生成的图片", ""]
            for index, artifact in enumerate(artifacts, 1):
                name = _txt(artifact.get("name"), f"图 {index}")
                name = _markdown_text(name)
                url = artifact_resolver(artifact, report_path)
                if url:
                    # Use a fixed alt label so sandbox filenames cannot inject Markdown.
                    lines += [f"**图 {index}**：{name}", "",
                              f"![运行结果图 {index}]({url})", ""]
                else:
                    lines += [f"> 图 {index} 无法读取，图片文件可能已被清理。", ""]
        for warning in final.get("artifact_warnings") or []:
            lines += [f"> {_markdown_text(str(warning))}", ""]
        if execution.get("code"):
            lines += ["", "### 生成代码", "```python",
                      _txt(execution["code"]), "```"]
        lines += ["", "### 执行输出(full)", "```",
                  _txt(final.get("stdout"), "无输出"), "```"]
        if final.get("stderr"):
            lines += ["### 诊断输出（stderr）"]
            if final.get("success") and final.get("exit_code") == 0:
                lines.append("> 本阶段退出码为 0，执行成功；stderr 也可能包含提示或警告。")
            lines += ["```", _txt(final["stderr"]), "```"]
        lines.append("")

        # 5. 验证结果 + 指标对比
        # 四态：复现成功 / 复现失败 / 无法验证（代码没跑起来，不能算复现失败）/
        # 无法核对（跑了，但论文信息不足、代码是占位实现——既不判成功也不判失败）
        if validation.get("status") == "needs_pdf":
            state_text = "⏳ 待补充论文 PDF（作者代码尚未执行）"
        elif validation.get("status") == "insufficient_evidence":
            state_text = ("⚠️ 无法核验（运行环境不受支持，未执行论文实验）"
                          if validation.get('failure_kind') == 'runtime_unsupported' else
                          "⚠️ 无法核验（论文证据不足，未执行论文实验）")
        elif validation.get("status") == "not_runnable":
            state_text = "⚠️ 无法验证（代码未运行）"
        elif validation.get("status") == "best_effort":
            # 必须排在 is_reproduced 之前：best_effort 的 is_reproduced 是 None，
            # 落到下面会被判成 ❌ 失败——那等于把"我们没拿到论文信息"说成
            # "论文复现失败"。
            state_text = "⚠️ 无法核对（论文证据不足或使用演示代码，非论文结论）"
        elif validation.get("status") == "no_reference_metrics":
            state_text = "⚠️ 无法核验（论文未声明参考指标数值）"
        elif validation.get("status") == "execution_failed":
            state_text = "❌ 执行失败（不能判定论文数值复现）"
        elif validation.get("status") == "mock_demo":
            state_text = "模拟流程已完成（未执行真实论文实验）"
        elif validation.get("status") == "execution_incomplete":
            state_text = "⚠️ 无法核验（执行证据不足）"
        elif validation.get("status") == "invalid_metrics":
            state_text = "❌ 无法核验（最终指标无效）"
        elif validation.get("status") == "smoke_passed":
            state_text = "⚠️ 仅冒烟通过（尚未完成论文数值核验）"
        elif validation.get("status") == "prepared":
            state_text = "⏳ 已准备（尚未执行训练）"
        elif validation.get("status") == "environment_prepared":
            state_text = "✅ 实验环境已准备（尚未训练）"
        elif validation.get("status") == "method_experiment_completed":
            state_text = "✅ 官方方法实验完成（不判定整篇论文数值复现）"
        elif validation.get('status') == 'author_code_completed':
            state_text = ('作者仓库代码运行完成（已对比作者输出，尚未独立核验论文数值）'
                          if (validation.get('author_claim_comparison') or {}).get('comparisons') else
                          '作者仓库代码运行完成（未进行论文数值审核）')
        elif validation.get("status") == "quality_target_not_met":
            state_text = "⚠️ 方法实验已运行，工程效果门槛未达到"
        elif validation.get("status") == "reference_not_met":
            state_text = "⚠️ 选定论文实验已完成，数值参考未达到"
        elif validation.get("status") == "analysis_failed":
            state_text = "⏳ 未进入训练（公开协议分析未通过）"
        elif validation.get("status") == "inconclusive":
            state_text = "⚠️ 参考实验已完成（实验协议仍待核验）"
        elif validation.get("is_reproduced"):
            state_text = ("✅ 选定论文实验数值复现通过" if execution.get("mode") == "repository"
                          else "✅ 论文数值核验通过")
        else:
            state_text = "❌ 失败"
        lines += ["## 5. 运行结果",
                  f"- **复现状态**: {state_text}"]
        strict = data.get('strict_validation')
        if isinstance(strict, dict) and strict.get('requested'):
            status = {'pending': '等待检查', 'eligible': '协议已固定，尚未完成独立运行',
                      'unavailable': '未完成；保留普通作者执行结果',
                      'verified': '完整独立复算证据已核验', 'failed': '未通过'}.get(
                          strict.get('status'), '未完成')
            lines.append('- **严格验收**: ' + status)
            if strict.get('reason'):
                lines.append('- **严格验收说明**: ' + _markdown_text(_txt(strict['reason'])))
        if execution.get("mode") != "repository":
            lines.append(f"- **置信度**: {_fmt(validation.get('confidence', 0.0))}")
        lines.append(f"- **分析**: {(validation.get('validation') or {}).get('analysis', '无')}")
        # 判定原因与上面的"分析"往往是同一句话（analysis 由 reason 拼成），
        # 重复印一遍只是噪音——已包含在分析里就不再单列。
        reason = _txt(validation.get("reason")).strip()
        if reason and reason not in _txt(
                (validation.get("validation") or {}).get("analysis")):
            lines.append(f"- **判定原因**: {reason}")
        source_request = data.get("source_request") or {}
        if source_request.get("next_action"):
            lines.append("- **下一步**: " + _markdown_text(_txt(source_request["next_action"])))
        for warning in validation.get('warnings', []):
            lines.append('- **运行说明**: ' + _markdown_text(_txt(warning)))
        lines.extend(_numeric_compatibility_lines(validation))
        for receipt in validation.get('author_execution', []):
            step = receipt.get('step') or {}
            lines.append('- **作者命令结束记录**: ' + _markdown_text(_txt(step.get('id'))) +
                         '；自然返回：' + ('是' if receipt.get('natural_return') else '否') +
                         '；实际优化器更新：' + _markdown_text(_txt(receipt.get('optimizer_steps'), '未观察到')))
        inner = validation.get("validation") or {}
        execution_level = {"failed": "执行失败", "incomplete": "执行证据不足",
                           "smoke_passed": "仅冒烟通过",
                           "author_code_completed": "原作者命令运行完成",
                           "experiment_completed": "完整执行已完成"}.get(validation.get("execution_status"))
        if execution_level:
            lines.append(f"- **执行级别**: {execution_level}")
        if inner.get("verdict_source") == "deterministic":
            tolerance = inner.get("relative_tolerance", 0.05)
            if execution.get("mode") == "repository":
                lines.append(f"- **判定依据**: 完整执行协议、独立指标复算及项目事前约定的 {tolerance:.0%} 相对误差规则（非论文给出的阈值）；API 负责协议审查和授权范围内的结果解释，最终结论由确定性核验给出。")
            else:
                lines.append(f"- **判定依据**: 执行状态、必需指标完整性、显式单位及 {tolerance:.0%} 相对误差规则；模型仅解释偏差")
        # 逐项数值差异（"声明 X vs 实际 Y，相对差异 Z%"）——判定结论的依据，
        # 只给"成功/失败"而不给差异，用户无法判断判定是否合理。
        diffs = inner.get("differences") or []
        if diffs:
            lines.append("- **指标差异**:")
            lines += [f"  - {d}" for d in diffs]
        missing = inner.get("missing_metrics") or []
        if missing:
            lines.append("- **无法比对的指标**:")
            lines += [f"  - {m}" for m in missing]
        metrics_comp = validation.get("metrics_comparison", {}) or {}
        if metrics_comp:
            paper_m = metrics_comp.get("paper", {}) or {}
            actual_m = metrics_comp.get("actual", {}) or {}
            if paper_m or actual_m:
                # 按归一化键配对：论文声明 `MSE`、运行输出 `mse` 是同一个指标，
                # 不配对就会在表里排成两行各缺一半，看起来像"没跑出来"。
                rows: dict = {}
                for k, v in paper_m.items():
                    rows.setdefault(norm_metric_key(k), {})["paper"] = (k, v)
                for k, v in actual_m.items():
                    rows.setdefault(norm_metric_key(k), {})["actual"] = (k, v)
                lines += ["", "### 指标对比",
                          "| 指标 | 作者公开参考 | 独立评估实测 |" if is_rezero else "| 指标 | 论文声明 | 实际运行 |",
                          "|------|----------|----------|"]
                for nk in sorted(rows):
                    cell = rows[nk]
                    pk, pv = cell.get("paper", (nk, "N/A"))
                    ak, av = cell.get("actual", (nk, "N/A"))
                    # 两侧键名写法不同则标注原写法，避免"对不上号"的疑惑
                    name = pk if pk == ak else f"{pk} / {ak}"
                    def metric_cell(value, unit_map, key):
                        explicit_unit = unit_map.get(key)
                        if is_rezero and key == "top1_accuracy_pct":
                            explicit_unit = "percent"
                        unit = metric_unit(value, explicit_unit)
                        if isinstance(value, dict):
                            value = value.get("value", "N/A")
                        suffix = {"percent": "%", "fraction": " (比例)"}.get(unit, " " + unit if unit else "")
                        return f"{value}{suffix}"
                    pv = metric_cell(pv, metrics_comp.get("paper_units") or {}, pk)
                    av = metric_cell(av, metrics_comp.get("actual_units") or {}, ak)
                    lines.append(f"| {name} | {pv} | {av} |")
        lines += _runtime_dataset_lines(validation.get('runtime_dataset_evidence'))
        lines += _author_claim_comparison_lines(validation.get('author_claim_comparison'))
        lines += _outcome_diagnosis_lines(data)
        records = validation.get("metric_records") or []
        if records:
            lines += ["", "### 实际指标证据", "| 指标 | 实测值 | 单位 | 数据划分 | 阶段 | 来源 |",
                      "|------|------|------|----------|------|------|"]
            for record in records:
                cells = [record.get("name"), _metric_value_text(record), record.get("unit") or "未标注",
                         record.get("split") or "未标注", record.get("stage") or "未标注",
                         record.get("source")]
                lines.append("| " + " | ".join(_markdown_text(_txt(c)) for c in cells) + " |")
        protocol = execution.get("protocol_verification") or {}
        independent = execution.get("independent_metrics") or {}
        if protocol:
            lines += ["", "### 完整实验核验",
                      f"- **协议核验**: {'通过' if protocol.get('pass') else '未通过'}"]
            if spec.get("adapter_id") == "discovered":
                lines += [f"- **实际优化器更新次数**: {protocol.get('optimizer_steps', '未知')}",
                          f"- **实际测试样本数**: {protocol.get('test_samples', '未知')}",
                          "- **独立评估方式**: 重载本次模型重新推理；NumPy 与 PyTorch 交叉核验指标。"]
            else:
                note = ("（最多10轮；保留作者早停）" if spec.get("adapter_id") == "dlinear"
                        or str(spec.get("id", "")).startswith("dlinear_") else "")
                lines.append(f"- **实际训练轮数**: {protocol.get('epochs_completed', '未知')}" + note)
            for check in protocol.get("checks", []):
                if isinstance(check, dict):
                    lines.append(f"  - {check.get('name', '')}: {'通过' if check.get('pass') else '未通过'}")
            if independent:
                values = independent.get("metrics") or {name: independent[name]
                    for name in ("accuracy", "cross_entropy") if name in independent}
                lines += [f"- **独立指标复算**: {independent.get('reason', independent.get('evaluation', ''))}",
                          f"- **复算值**: {values}"]
        api_analysis = data.get("llm_analysis") or {}
        if api_analysis:
            lines += ["", "### 真实 API 公开论文协议解析",
                      f"- **模型**: {api_analysis.get('model', '')}",
                      _txt(api_analysis.get("summary"))]
            for limitation in api_analysis.get("limitations", []):
                lines.append(f"- {_txt(limitation)}")
        lines += _result_analysis_lines(data)
        if is_rezero and not reviews_removed:
            summary = validation.get("training_summary") or {}
            actual = (validation.get("metrics_comparison") or {}).get("actual") or {}
            lines += ["", "### ReZero 选定论文实验范围与验收",
                      "- **结论范围**: CIFAR-10 上 ReZero PreActResNet18 超收敛；不覆盖 enwiki8 或整篇论文。",
                      "- **冻结训练协议**: seed 6892、FP32 CUDA、batch 512、45 轮 / 4410 个优化器步骤；完整 50,000 张训练集和 10,000 张测试集。",
                      "- **固定验收参考**: 作者公开 notebook 的 94.00% top-1；独立评估须 ≥94.00%，不降低阈值。cross_entropy 仅记录本次实测值。",
                      f"- **参考来源**: {paper_info.get('reference_source', spec.get('paper', {}).get('reference_source', ''))}",
                      f"- **实验状态**: {validation.get('status', '尚未执行')}",
                      f"- **协议核验 / 独立复算**: {validation.get('protocol_pass', False)} / {validation.get('independent_metrics_pass', False)}"]
            if summary:
                def measured(value, precision=".2f"):
                    return "未提供" if value is None else _fmt(value, precision)
                lines += [f"- **实际完成**: {summary.get('epochs_completed', '未知')} 轮 / {summary.get('steps_completed', '未知')} 个优化器步骤",
                          f"- **实际数据规模**: 训练 {summary.get('train_samples', '未知')} / 测试 {summary.get('test_samples', '未知')}",
                          f"- **最佳 checkpoint**: 按作者协议从各轮测试准确率选取，第 {summary.get('best_epoch', '未知')} 轮；训练记录 {measured(summary.get('best_accuracy_pct'))}%",
                          f"- **独立重载评估**: top-1 {measured(actual.get('top1_accuracy_pct'))}% / 固定参考 94.00%；完整测试集 cross_entropy {measured(actual.get('cross_entropy'), '.6f')}"]
            else:
                lines.append("- **本次训练证据**: 尚无完成协议核验的训练摘要，不能根据冻结协议推断已完成。")
            lines += [f"- **基线耗时**: {_fmt(data.get('baseline_elapsed_s'), '.2f')} 秒",
                      f"- **本次总耗时（含环境准备）**: {_fmt(data.get('run_elapsed_s'), '.2f')} 秒",
                      f"- **运行兼容性**: {spec.get('environment', {}).get('note', '')}"]
        if spec.get("adapter_id") in {"siren", "neural_ode"}:
            lines += ["", "### 方法实验范围与耗时",
                      f"- **实验状态**: {validation.get('status', '尚未执行')}",
                      "- **结论范围**: 官方方法实验；没有对应论文表格数值验收，不宣称整篇论文数值复现。",
                      f"- **基线耗时**: {_fmt(data.get('baseline_elapsed_s'), '.2f')} 秒",
                      f"- **本次总耗时（含环境准备）**: {_fmt(data.get('run_elapsed_s'), '.2f')} 秒",
                      f"- **实验与报告耗时（环境准备后）**: {_fmt(data.get('execution_elapsed_s'), '.2f')} 秒"]
            if not reviews_removed:
                lines.append(f"- **协议/独立复算**: {validation.get('protocol_pass', False)} / {validation.get('independent_metrics_pass', False)}")
        if spec.get("adapter_id") in {"siren", "neural_ode", "rezero"}:
            method_analysis = data.get("method_analysis") or {}
            if method_analysis.get("status") == "rejected":
                labels = {"reader": "论文方法分析", "finder": "资源核对",
                          "builder": "依赖分析", "verifier": "证据预审"}
                failed_role = method_analysis.get("failed_role", "")
                lines += ["", "### 在线分析未通过",
                          "- **失败阶段**: " + _markdown_text(labels.get(failed_role, failed_role)),
                          "- **拒绝说明（不代表已核验的实验事实）**: " +
                          _markdown_text(_txt(method_analysis.get("reason"))),
                          "- **训练状态**: 在线预审未通过，训练尚未开始。",
                          "- **诊断记录**: 运行目录中的 `method_analysis.json` 保留逐阶段结果与模型响应。"]
            for review in method_analysis.get("reviews", []):
                lines += ["", f"### 在线来源分析：{review['role']}", _markdown_text(review["summary"])]
                for evidence in review["evidence"]:
                    lines += ["", f"> {_markdown_text(evidence['source_id'])}: " + _markdown_text(evidence["quote"]).replace("\n", "\n> ")]
        lines.append("")

        lines += _optimization_lines(optimization, is_rezero=is_rezero)

        collaboration = data.get('agent_collaboration') or {}
        messages = collaboration.get('messages') or []
        if messages:
            lines += ['## Agent 协作记录', '',
                      '以下记录本次任务的证据交接、补充请求和校验反馈；消息本身不是实验核验结论。', '',
                      '完整消息保存在运行目录的 `agent_collaboration.json` 中。', '']
            positions = {message.get('id'): message.get('sequence', index)
                         for index, message in enumerate(messages, 1) if message.get('id')}
            kinds = {'request': '请求', 'response': '回复', 'handoff': '交接', 'evidence': '证据',
                     'feedback': '反馈', 'acknowledgement': '确认', 'result': '结果'}
            for message in messages[-40:]:
                route = _markdown_text(f"{message.get('sender', '')} → {message.get('recipient', '')}")
                kind = message.get('kind', '')
                detail = kinds.get(kind, _markdown_text(str(kind)))
                if message.get('reply_to'):
                    reply = positions.get(message['reply_to'])
                    detail += f'；回复 #{reply}' if reply is not None else '；回复较早消息'
                lines += [f"**{message.get('sequence', '')}. {route}**（{detail}）", '',
                          _collaboration_summary(message.get('content')), '']
            if collaboration.get('dropped_messages'):
                lines += ['', '部分消息已达到协作记录预算；完整执行与修复证据另行保存。']
            lines.append('')

        # 8. 审计与用量统计
        from src.agent_collaboration import current_collaboration
        session = current_collaboration()
        cache = (session.snapshot() if session is not None else collaboration).get('evidence_cache') or {}
        lines += ["## 7. 运行与用量统计",
                  f"- **总步骤数**: {audit.get('total_steps', 'N/A')}",
                  f"- **成功步数**: {audit.get('success', 'N/A')}",
                  f"- **错误步数**: {audit.get('errors', 'N/A')}",
                  f"- **执行时长**: {audit.get('duration_sec', 'N/A')} 秒",
                  f"- **LLM 调用次数**: {audit.get('llm_calls', 0)} "
                  f"（用于用量审计；全文阅读不设总调用预算）"]
        if cache:
            lines += [f"- **PDF 证据复用**: 命中 {cache.get('hits', 0)} 次，重新提取 {cache.get('misses', 0)} 次；"
                      "仅复用同一运行中内容与解析器身份均未变化的完整证据。",
                      "- **复用范围**: 减少重复解析；不代表训练耗时减少或论文复现成功率已经提升。"]
        lines += ["",
                  "---",
                  "*报告由 AutoReproducer 自动生成*"]
        from src.run_finalization import finalize_report_text
        # Rendering saved evidence must not mutate its source. Runtime
        # finalization separately persists the refreshed derived conclusion.
        return finalize_report_text("\n".join(lines), dict(data))
