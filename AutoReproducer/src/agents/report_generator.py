"""ReportGeneratorAgent - 报告生成 Agent，生成 Markdown 复现与优化报告。

报告覆盖：论文信息 / 资源定位 / 环境配置与依赖诊断 / 代码执行（smoke+full）/
验证结果（指标对比） / 智能优化（UCB 尝试记录） / 验证闭环 / 审计与预算统计。
"""
from datetime import datetime
from src.base_agent import BaseAgent


def _fmt(value, spec: str = ".2f", default: float = 0.0) -> str:
    """数值格式化：容忍字符串/None 等非数值（LLM 可能把 confidence 填成 "0.1"）。"""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return format(float(default), spec)


# 报告内嵌代码/输出的展示上限（字符数）。
# 完整内容由流水线落盘为 `<报告名>_execution.txt` 附件，报告内超长仅做展示截断
# （不再静默丢弃：截断处标注实际长度与附件位置，避免「输出一半」）。
CODE_SHOW_LIMIT = 6000
STDOUT_SHOW_LIMIT = 6000
STDERR_SHOW_LIMIT = 3000


def _clip(text: str, limit: int) -> str:
    """截断展示用长文本：超长时在末尾标注实际长度与附件提醒。"""
    if not text or len(text) <= limit:
        return text
    return (f"{text[:limit]}\n\n"
            f"[⚠️ 输出过长，此处仅截断展示前 {limit} / {len(text)} 字符——"
            f"完整内容见随报告生成的 `*_execution.txt` 附件]")


class ReportGeneratorAgent(BaseAgent):
    """生成 Markdown 格式的复现 + 优化报告。"""

    def __init__(self, logger=None):
        super().__init__("ReportGenerator", logger)

    def run(self, input_data: dict) -> dict:
        """生成完整复现报告（含优化与审计信息）。"""
        self.log("generate_report", "START", "开始生成复现报告", input_data)

        report = self._build_report(input_data)

        self.log("generate_report", "SUCCESS",
                 f"报告生成完成 ({len(report)} 字符)",
                 {"report_length": len(report)})
        return {"report": report, "report_length": len(report)}

    def _build_report(self, data: dict) -> str:
        paper_info = data.get("paper_info", {}) or {}
        resources = data.get("resources", {}) or {}
        env_config = data.get("env_config", {}) or {}
        execution = data.get("execution", {}) or {}
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

        # 3. 环境配置 + 依赖诊断
        lines += ["## 3. 环境配置",
                  f"- **Python版本**: {env_config.get('python_version', 'N/A')}",
                  f"- **依赖数**: {len(env_config.get('requirements_txt', '').splitlines())}",
                  f"- **预估磁盘**: {env_config.get('estimated_disk_gb', 'N/A')} GB"]
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
                  env_config.get("requirements_txt", "无"), "```", ""]

        # 4. 代码执行（smoke + full）
        lines += ["## 4. 代码执行",
                  f"- **代码长度**: {len(execution.get('code', ''))} 字符"]
        stages = execution.get("stages", []) or []
        final = execution.get("final", {}) or {}
        if execution.get("not_runnable"):
            lines.append("- **执行状态**: ⚠️ 未运行（代码未通过执行前检查）")
            lines.append(f"- **未运行原因**: {execution.get('reason', 'N/A')}")
        else:
            lines.append(f"- **执行状态**: "
                         f"{'✅ 成功' if final.get('success') else '❌ 失败'}")
        for st in stages:
            st_ok = st.get("success")
            lines.append(
                f"  - {st.get('stage')}: {'✅ 通过' if st_ok else '❌ 失败'} "
                f"(退出码 {st.get('exit_code')})")
        if execution.get("code"):
            lines += ["", "### 生成代码", "```python",
                      _clip(execution["code"], CODE_SHOW_LIMIT), "```"]
        lines += ["", "### 执行输出(full)", "```",
                  _clip(final.get("stdout", "无输出"), STDOUT_SHOW_LIMIT), "```"]
        if final.get("stderr"):
            lines += ["### 错误输出", "```",
                      _clip(final["stderr"], STDERR_SHOW_LIMIT), "```"]
        lines.append("")

        # 5. 验证结果 + 指标对比
        # 三态：复现成功 / 复现失败 / 无法验证（代码没跑起来，不能算复现失败）
        if validation.get("status") == "not_runnable":
            state_text = "⚠️ 无法验证（代码未运行）"
        elif validation.get("is_reproduced"):
            state_text = "✅ 成功"
        else:
            state_text = "❌ 失败"
        lines += ["## 5. 验证结果",
                  f"- **复现状态**: {state_text}",
                  f"- **置信度**: {_fmt(validation.get('confidence', 0.0))}",
                  f"- **分析**: "
                  f"{(validation.get('validation') or {}).get('analysis', '无')}"]
        missing = (validation.get("validation") or {}).get("missing_metrics") or []
        if missing:
            lines.append("- **无法比对的指标**:")
            lines += [f"  - {m}" for m in missing]
        metrics_comp = validation.get("metrics_comparison", {}) or {}
        if metrics_comp:
            paper_m = metrics_comp.get("paper", {}) or {}
            actual_m = metrics_comp.get("actual", {}) or {}
            if paper_m or actual_m:
                lines += ["", "### 指标对比",
                          "| 指标 | 论文声明 | 实际运行 |",
                          "|------|----------|----------|"]
                for k in sorted(set(paper_m) | set(actual_m)):
                    lines.append(f"| {k} | {paper_m.get(k, 'N/A')} "
                                 f"| {actual_m.get(k, 'N/A')} |")
        lines.append("")

        # 6. 智能优化
        lines += ["## 6. 智能优化"]
        if not optimization.get("optimized"):
            lines.append(f"- **优化状态**: 未触发("
                         f"{optimization.get('reason', '复现未成功或未运行优化')})")
        else:
            lines += ["- **优化状态**: ✅ 已优化",
                      f"- **基线指标**: {optimization.get('baseline', 'N/A')}",
                      f"- **最优方向**: {optimization.get('best_arm', 'N/A')}",
                      f"- **改进幅度**: {optimization.get('improvement', 0):.2%}",
                      f"- **最优结果**: {optimization.get('best_result', 'N/A')}",
                      f"- **预算使用**: {optimization.get('budget_used', 'N/A')} / "
                      f"{optimization.get('budget', 'N/A')} 次尝试",
                      "",
                      "### 尝试记录(UCB 预算调度)",
                      "| 方向 | 改进幅度 | 判定 |",
                      "|------|----------|------|"]
            for r in optimization.get("optimization_report", []) or []:
                mark = "✅ Keep" if r.get("kept") else "❌ Reject"
                lines.append(f"| {r.get('arm')} | {r.get('improvement'):.4f} "
                             f"| {mark} |")
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