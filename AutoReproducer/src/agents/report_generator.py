"""ReportGeneratorAgent - 报告生成 Agent，生成 Markdown 复现与优化报告。

报告覆盖：论文信息 / 资源定位 / 环境配置与依赖诊断 / 代码执行（smoke+full）/
验证结果（指标对比） / 智能优化（UCB 尝试记录） / 验证闭环 / 审计与预算统计。
"""
from datetime import datetime
from src.base_agent import BaseAgent
from src.metric_keys import norm_metric_key


def _fmt(value, spec: str = ".2f", default: float = 0.0) -> str:
    """数值格式化：容忍字符串/None 等非数值（LLM 可能把 confidence 填成 "0.1"）。"""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return format(float(default), spec)


# 报告内代码与执行输出**全文内嵌**，不设展示上限。
# 此前是截断到 6000/6000/3000 字符 + 一句「完整内容见 *_execution.txt 附件」，
# 但报告本身才是用户真正在读的东西：实测一次 10627 字符的输出被砍到 6000，
# 读报告的人拿到的是半截内容，还得去翻附件才知道后一半是什么。
# 前端 `frontend/markdown_render.py` 会把代码块渲染成可滚动的深色面板，
# 长度不再等于页面高度，全文内嵌没有排版代价。


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
                      execution["code"], "```"]
        lines += ["", "### 执行输出(full)", "```",
                  final.get("stdout", "无输出"), "```"]
        if final.get("stderr"):
            lines += ["### 错误输出", "```", final["stderr"], "```"]
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
        inner = validation.get("validation") or {}
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
                    lines.append(f"| {name} | {pv} | {av} |")
        lines.append("")

        # 6. 智能优化
        lines += ["## 6. 智能优化"]
        if not optimization.get("optimized"):
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