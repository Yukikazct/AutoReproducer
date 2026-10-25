"""Present lifecycle outcomes without changing experiment or numeric evidence."""
import re


_START = "<!-- autoreproducer-run-finalization:start -->"
_END = "<!-- autoreproducer-run-finalization:end -->"
_CONCLUSION_START = "<!-- autoreproducer-conclusion:start -->"
_CONCLUSION_END = "<!-- autoreproducer-conclusion:end -->"

def _lines_outside_fences(report):
    fence = None
    offset = 0
    for line in report.splitlines(keepends=True):
        if fence is None:
            yield offset, line
        match = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not line[match.end():].strip():
                fence = None
        offset += len(line)


def _remove_summary(report, start_marker=_START, end_marker=_END):
    start = None
    for offset, line in _lines_outside_fences(report):
        if line.strip() == start_marker:
            start = offset
        elif start is not None and line.strip() == end_marker:
            end = offset + len(line)
            while end < len(report) and report[end] == "\n":
                end += 1
            return report[:start] + report[end:]
    return report


def _text(value):
    value = " ".join(str(value).split())
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", value)


def _insert_block(report, lines, *, before_markers=()):
    visible = list(_lines_outside_fences(report))
    position = next((offset for offset, line in visible
                     if line.startswith("## ") or line.strip() in before_markers), None)
    if position is None:
        position = next((offset for offset, line in visible
                         if line.startswith("> 📎 **完整执行输出附件**:")
                         and not report[offset + len(line):].strip()), None)
    if position is None:
        return report.rstrip() + "\n\n" + "\n".join(lines)
    return report[:position].rstrip() + "\n\n" + "\n".join(lines) + "\n" + report[position:]


def _with_conclusion(report, data):
    from src.result_conclusion import build_reproduction_conclusion, conclusion_markdown
    conclusion = build_reproduction_conclusion(data)
    data["reproduction_conclusion"] = conclusion
    lines = [_CONCLUSION_START, conclusion_markdown(conclusion), _CONCLUSION_END, ""]
    return _insert_block(report, lines, before_markers=(_START,))


def finalize_report_text(report, data):
    """Refresh current conclusions, retaining original evidence, assets and footer."""
    receipt = data.get("environment_cleanup") or {}
    finalization = data.get("run_finalization") or {}
    if not report:
        return report
    report = _remove_summary(report, _CONCLUSION_START, _CONCLUSION_END)
    if not receipt:
        return _with_conclusion(report, data)
    state = finalization.get("state")
    lines = [_START, "### 任务收尾", "",
             "- **总任务状态**: " + {"COMPLETED": "已完成", "ERROR": "异常结束"}.get(state, "已结束"),
             "- **环境释放**: " + {"released": "已释放", "error": "失败", "in_use": "由其他任务持有"}.get(
                 receipt.get("status"), "未确认")]
    if finalization.get("error"):
        lines.append("- **终态错误**: " + _text(finalization["error"]))
    if receipt.get("status") == "released":
        lines.append("- 本次实验依赖与本任务拥有的运行资源已释放；报告、日志、数据和源码证据保留。")
    for item in receipt.get("errors") or []:
        reason = (str(item.get("path", "")) + ": " + str(item.get("reason", ""))).strip(": ") if isinstance(item, dict) else str(item)
        lines.append("- **释放诊断**: " + _text(reason))
    if (receipt.get("supervisor_cleanup") or {}).get("status") == "in_use":
        lines.append("- **工作进程收尾**: 依赖缓存已由其他任务持有，当前持有者的环境已保留。")
    lines += ["- 实验执行与论文数值结论保留在下方核验记录中。", _END, ""]
    report = _remove_summary(report)
    # Keep the execution attachment footer at the end so ZIP export includes it.
    return _with_conclusion(_insert_block(report, lines), data)
