"""ReportGeneratorAgent - 报告生成 Agent，生成 Markdown 复现与优化报告。

报告覆盖：论文信息 / 资源定位 / 环境配置与依赖诊断 / 代码执行（smoke+full）/
验证结果（指标对比） / 智能优化（UCB 尝试记录） / 验证闭环 / 审计与预算统计。
"""
from datetime import datetime
import json
import re
from src.base_agent import BaseAgent
from src.metric_keys import norm_metric_key
from src.runtime_metrics import metric_unit
from src.storage_usage import format_bytes
from src.execution_artifacts import default_report_path, markdown_image_target


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
    lines += ["", "> 分析范围与实验范围均为选定的 DLinear / ETTh1 336→96 多变量实验，不代表论文全部实验。", ""]
    return lines


def _result_analysis_lines(data):
    analysis = data.get("result_analysis") or {}
    status = data.get("analysis_status")
    if not analysis and not status:
        return []
    states = {"completed": "已完成", "public_readiness_accepted": "公开准备条件已通过；未执行结果摘要解释",
              "result_analysis_failed": "结果摘要解释失败", "failed": "LLM 分析失败"}
    lines = ["### LLM 分析状态与结果摘要解释", "",
             "- **LLM 分析状态**: " + states.get(status, _txt(status, "未记录")),
             "- **结论职责**: API 分析状态与训练、协议核验、独立指标复算分别记录；API 解释失败不会覆盖已完成实验的确定性结论。"]
    if analysis:
        lines += ["- **结果解释输入范围**: 用户授权的 MSE、MAE、已完成轮数、协议核验与独立复算状态摘要；参考依据为公开来源。"]
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


# 报告内代码与执行输出**全文内嵌**，不设展示上限。
# 此前是截断到 6000/6000/3000 字符 + 一句「完整内容见 *_execution.txt 附件」，
# 但报告本身才是用户真正在读的东西：实测一次 10627 字符的输出被砍到 6000，
# 读报告的人拿到的是半截内容，还得去翻附件才知道后一半是什么。
# 前端按原生 Markdown 渲染（代码块 = 灰底 `<pre>`），全文内嵌意味着长输出会
# 把页面拉长——这是刻意换来的：可读性优先于版面，附件仍提供纯文本旁路。


class ReportGeneratorAgent(BaseAgent):
    """生成 Markdown 格式的复现 + 优化报告。"""

    def __init__(self, logger=None):
        super().__init__("ReportGenerator", logger)

    def run(self, input_data: dict, *, report_path=None) -> dict:
        """生成完整复现报告（含优化与审计信息）。"""
        self.log("generate_report", "START", "开始生成复现报告", input_data)

        report = self._build_report(input_data, report_path=report_path)

        self.log("generate_report", "SUCCESS",
                 f"报告生成完成 ({len(report)} 字符)",
                 {"report_length": len(report)})
        return {"report": report, "report_length": len(report)}

    def _build_report(self, data: dict, *, report_path=None) -> str:
        report_path = report_path or data.get("report_path") or default_report_path()
        paper_info = data.get("paper_info", {}) or {}
        resources = data.get("resources", {}) or {}
        execution = data.get("execution", {}) or {}
        env_config = execution.get("effective_env_config") or data.get("env_config", {}) or {}
        validation = data.get("validation", {}) or {}
        optimization = data.get("optimization", {}) or {}
        audit = data.get("audit_stats", {}) or {}

        lines = ["# 论文复现与优化报告",
                 "",
                 f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"**工具**: AutoReproducer v0.2.0",
                 ""]

        # 1. 论文信息
        lines += ["## 1. 论文信息",
                  f"- **标题**: {paper_info.get('title', '未知')}",
                  f"- **方法**: {paper_info.get('method', '未知')}",
                  f"- **数据集**: {paper_info.get('dataset', '未知')}",
                  f"- **声明指标**: {paper_info.get('metrics', {})}",
                  ""]

        # 2. 资源定位
        lines += ["## 2. 资源定位",
                  f"- **代码仓库**: {resources.get('code_repo_url', '未找到')}",
                  f"- **数据集**: {resources.get('dataset_url', '未找到')}",
                  f"- **置信度**: {_fmt(resources.get('confidence', 0.0))}"]
        urls = resources.get("extracted_urls", []) or []
        if urls:
            lines.append(f"- **从论文中提取URL**: {len(urls)} 个")
            for u in urls[:5]:
                lines.append(f"  - {u}")
        lines.append("")

        repository = data.get("repository") or {}
        spec = data.get("experiment_spec") or {}
        if spec:
            lines += ["### 官方仓库实验来源",
                      f"- **预设**: {spec.get('label', spec.get('id', ''))}",
                      f"- **论文版本/参考位置**: {paper_info.get('reference_source', '')}",
                      f"- **固定源码版本**: `{repository.get('resolved_sha', '未获取')}`",
                      f"- **实验定义SHA-256**: `{data.get('spec_sha256', '')}`",
                      f"- **独立运行目录**: `{data.get('run_dir', '')}`",
                      f"- **参数**: {spec.get('parameters', {})}"]
            dataset = data.get("dataset_provenance") or {}
            if dataset:
                lines += [f"- **真实数据SHA-256**: `{dataset.get('sha256', '')}`",
                          f"- **数据切分**: {dataset.get('split', '')}"]
            lines += [f"- **协议说明**: {(spec.get('validation') or {}).get('note', '')}", ""]
            lines += [f"- **作者协议来源**: {(spec.get('validation') or {}).get('aggregation_source', '')}",
                      f"- **实现版本说明**: {(spec.get('validation') or {}).get('implementation_note', '')}", ""]
        lines += _repository_analysis_lines(data)

        # 3. 环境配置 + 依赖诊断
        lines += ["## 3. 环境配置",
                  f"- **Python版本**: {env_config.get('python_version', 'N/A')}",
                  f"- **依赖数**: "
                  f"{len(_txt(env_config.get('requirements_txt')).splitlines())}"]
        if env_config.get("note"):
            lines.append(f"- **环境适配**: {env_config['note']}")
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
        if execution.get("mode") == "repository":
            lines.append("- **执行对象**: 固定版本完整官方仓库（多文件）")
        else:
            lines.append(f"- **代码长度**: {len(_txt(execution.get('code')))} 字符")
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
            # 只剩两道真门（语法错误 / 危险调用）会走到这里
            lines.append("- **执行状态**: ⚠️ 未运行" +
                         ("（仓库实验尚未执行）" if execution.get("mode") == "repository" else
                          "（代码未通过执行前检查）"))
            lines.append(f"- **未运行原因**: {execution.get('reason', 'N/A')}")
        else:
            state = "✅ 成功" if final.get("success") else "❌ 失败"
            if best_effort:
                state += "（尽力而为：论文信息不足）"
            lines.append(f"- **执行状态**: {state}")
        if best_effort:
            # 信息不足也必须跑——但跑出来的东西不能被读成论文结论。这两句是
            # 报告里唯一防止"占位数字被当成复现结果"的拦网，措辞不能省。
            lines += [
                "- **⚠️ 信息不足**: "
                + _txt(execution.get("best_effort_reason"), "论文信息不足"),
                "- **生成方式**: "
                + ("系统本地兜底脚本（模型未给出可用代码）"
                   if execution.get("fallback_used")
                   else "模型按占位约定生成的最小可运行脚本"),
            ]
            for a in execution.get("assumptions") or []:
                lines.append(f"  - 假设: {a}")
        for st in stages:
            st_ok = st.get("success")
            lines.append(
                f"  - {st.get('stage')}: {'✅ 通过' if st_ok else '❌ 失败'} "
                f"(退出码 {st.get('exit_code')})")
        if execution.get("mode") == "repository":
            lines += ["", "### 官方仓库执行命令", ""]
            steps = stages or (data.get("execution_plan") or {}).get("steps", [])
            for step in steps:
                lines += [f"- **{step.get('id', step.get('stage', '步骤'))}** · cwd: `{step.get('cwd', '.')}`",
                          "```json", json.dumps(step.get("argv", []), ensure_ascii=False), "```"]
                if step.get("stdout_path"):
                    lines.append(f"- 完整日志: `{step['stdout_path']}`")
        artifacts = final.get("artifacts") or []
        if artifacts:
            lines += ["", "### 运行生成的图片", ""]
            for index, artifact in enumerate(artifacts, 1):
                name = _txt(artifact.get("name"), f"图 {index}")
                name = _markdown_text(name)
                url = markdown_image_target(artifact, report_path)
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
        if validation.get("status") == "not_runnable":
            state_text = "⚠️ 无法验证（代码未运行）"
        elif validation.get("status") == "best_effort":
            # 必须排在 is_reproduced 之前：best_effort 的 is_reproduced 是 None，
            # 落到下面会被判成 ❌ 失败——那等于把"我们没拿到论文信息"说成
            # "论文复现失败"。
            state_text = "⚠️ 无法核对（论文信息不足，代码为占位实现，非论文结论）"
        elif validation.get("status") == "no_reference_metrics":
            state_text = "⚠️ 无法核验（论文未声明参考指标数值）"
        elif validation.get("status") == "execution_failed":
            state_text = "❌ 执行失败（不能判定论文数值复现）"
        elif validation.get("status") == "execution_incomplete":
            state_text = "⚠️ 无法核验（执行证据不足）"
        elif validation.get("status") == "invalid_metrics":
            state_text = "❌ 无法核验（最终指标无效）"
        elif validation.get("status") == "smoke_passed":
            state_text = "⚠️ 仅冒烟通过（尚未完成论文数值核验）"
        elif validation.get("status") == "prepared":
            state_text = "⏳ 已准备（尚未执行训练）"
        elif validation.get("status") == "analysis_failed":
            state_text = "⏳ 未进入训练（公开协议分析未通过）"
        elif validation.get("status") == "inconclusive":
            state_text = "⚠️ 参考实验已完成（实验协议仍待核验）"
        elif validation.get("is_reproduced"):
            state_text = ("✅ 选定论文实验数值复现通过" if execution.get("mode") == "repository"
                          else "✅ 论文数值核验通过")
        else:
            state_text = "❌ 失败"
        lines += ["## 5. 验证结果",
                  f"- **复现状态**: {state_text}"]
        if execution.get("mode") != "repository":
            lines.append(f"- **置信度**: {_fmt(validation.get('confidence', 0.0))}")
        lines.append(f"- **分析**: {(validation.get('validation') or {}).get('analysis', '无')}")
        # 判定原因与上面的"分析"往往是同一句话（analysis 由 reason 拼成），
        # 重复印一遍只是噪音——已包含在分析里就不再单列。
        reason = _txt(validation.get("reason")).strip()
        if reason and reason not in _txt(
                (validation.get("validation") or {}).get("analysis")):
            lines.append(f"- **判定原因**: {reason}")
        inner = validation.get("validation") or {}
        execution_level = {"failed": "执行失败", "incomplete": "执行证据不足",
                           "smoke_passed": "仅冒烟通过",
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
                          "| 指标 | 论文声明 | 实际运行 |",
                          "|------|----------|----------|"]
                for nk in sorted(rows):
                    cell = rows[nk]
                    pk, pv = cell.get("paper", (nk, "N/A"))
                    ak, av = cell.get("actual", (nk, "N/A"))
                    # 两侧键名写法不同则标注原写法，避免"对不上号"的疑惑
                    name = pk if pk == ak else f"{pk} / {ak}"
                    def metric_cell(value, unit_map, key):
                        unit = metric_unit(value, unit_map.get(key))
                        if isinstance(value, dict):
                            value = value.get("value", "N/A")
                        suffix = {"percent": "%", "fraction": " (比例)"}.get(unit, " " + unit if unit else "")
                        return f"{value}{suffix}"
                    pv = metric_cell(pv, metrics_comp.get("paper_units") or {}, pk)
                    av = metric_cell(av, metrics_comp.get("actual_units") or {}, ak)
                    lines.append(f"| {name} | {pv} | {av} |")
        records = validation.get("metric_records") or []
        if records:
            lines += ["", "### 实际指标证据", "| 指标 | 单位 | 数据划分 | 阶段 | 来源 |",
                      "|------|------|----------|------|------|"]
            for record in records:
                cells = [record.get("name"), record.get("unit") or "未标注",
                         record.get("split") or "未标注", record.get("stage") or "未标注",
                         record.get("source")]
                lines.append("| " + " | ".join(_markdown_text(_txt(c)) for c in cells) + " |")
        protocol = execution.get("protocol_verification") or {}
        independent = execution.get("independent_metrics") or {}
        if protocol:
            lines += ["", "### 完整实验核验",
                      f"- **实际训练轮数**: {protocol.get('epochs_completed', '未知')}（最多10轮；保留作者早停）",
                      f"- **协议核验**: {'通过' if protocol.get('pass') else '未通过'}"]
            for check in protocol.get("checks", []):
                if isinstance(check, dict):
                    lines.append(f"  - {check.get('name', '')}: {'通过' if check.get('pass') else '未通过'}")
            if independent:
                lines += [f"- **独立指标复算**: {independent.get('reason', '')}",
                          f"- **复算值**: {independent.get('metrics', {})}"]
        api_analysis = data.get("llm_analysis") or {}
        if api_analysis:
            lines += ["", "### 真实 API 公开论文协议解析",
                      f"- **模型**: {api_analysis.get('model', '')}",
                      _txt(api_analysis.get("summary"))]
            for limitation in api_analysis.get("limitations", []):
                lines.append(f"- {_txt(limitation)}")
        lines += _result_analysis_lines(data)
        if spec.get("adapter_id") in {"siren", "neural_ode"}:
            lines += ["", "### 方法实验范围与耗时",
                      f"- **实验状态**: {validation.get('status', '尚未执行')}",
                      "- **结论范围**: 官方方法实验；没有对应论文表格数值验收，不宣称整篇论文数值复现。",
                      f"- **基线耗时**: {_fmt(data.get('baseline_elapsed_s'), '.2f')} 秒",
                      f"- **本次累计耗时**: {_fmt(data.get('run_elapsed_s'), '.2f')} 秒（首次环境准备另计）",
                      f"- **协议/独立复算**: {validation.get('protocol_pass', False)} / {validation.get('independent_metrics_pass', False)}"]
        lines.append("")

        # 6. 智能优化
        lines += ["## 6. 智能优化"]
        if optimization.get("mode") in {"suggest", "validate"}:
            lines += [f"- **状态**: {_markdown_text(_txt(optimization.get('status')))}",
                      f"- **说明**: {_markdown_text(_txt(optimization.get('reason')))}",
                      f"- **已验证提升**: {'是' if optimization.get('optimized') else '否'}"]
            for index, suggestion in enumerate(optimization.get("suggestions", []), 1):
                lines += ["", f"### 建议 {index}（尚未验证）",
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
        elif not optimization.get("optimized"):
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

        # 7. Prompt-Free 验证闭环
        lines += ["## 7. 质量验证(Prompt-Free)",
                  f"- **验证记录数**: {len(data.get('verifications', []) or [])}"]
        fix_records = data.get("fix_records", []) or []
        if fix_records:
            lines.append(f"- **修正闭环触发**: {len(fix_records)} 次")
            for fr in fix_records:
                lines.append(f"  - {fr.get('agent')} 第 {fr.get('round')} 轮修正: "
                             f"{'; '.join(fr.get('issues', [])) or '结构问题'}")
        else:
            lines.append("- **修正闭环触发**: 0 次（各步骤一次通过）")
        lines.append("")

        # 8. 审计与预算统计
        lines += ["## 8. 审计与预算统计",
                  f"- **总步骤数**: {audit.get('total_steps', 'N/A')}",
                  f"- **成功步数**: {audit.get('success', 'N/A')}",
                  f"- **错误步数**: {audit.get('errors', 'N/A')}",
                  f"- **执行时长**: {audit.get('duration_sec', 'N/A')} 秒",
                  f"- **LLM 调用次数**: {audit.get('llm_calls', 0)} "
                  f"（方案预算上限 100 次）",
                  "",
                  "---",
                  "*报告由 AutoReproducer 自动生成*"]
        return "\n".join(lines)
