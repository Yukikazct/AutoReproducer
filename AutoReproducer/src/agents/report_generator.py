"""ReportGeneratorAgent - 报告生成Agent，生成Markdown复现报告"""
from datetime import datetime
from src.base_agent import BaseAgent


class ReportGeneratorAgent(BaseAgent):
    """生成Markdown格式的复现报告"""

    def __init__(self, logger=None):
        super().__init__("ReportGenerator", logger)

    def run(self, input_data: dict) -> dict:
        """生成完整复现报告
        input_data: {
            "paper_info": dict,
            "resources": dict,
            "env_config": dict,
            "execution": dict,
            "validation": dict,
            "audit_summary": dict
        }
        """
        self.log("generate_report", "START", "开始生成复现报告", input_data)

        report = self._build_report(input_data)

        self.log("generate_report", "SUCCESS",
                 f"报告生成完成 ({len(report)} 字符)",
                 {"report_length": len(report)})

        return {"report": report, "report_length": len(report)}

    def _build_report(self, data: dict) -> str:
        """构建Markdown报告"""
        paper_info = data.get("paper_info", {})
        resources = data.get("resources", {})
        env_config = data.get("env_config", {})
        execution = data.get("execution", {})
        validation = data.get("validation", {})

        lines = []
        lines.append(f"# 论文复现报告")
        lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**工具**: AutoReproducer v0.1.0")
        lines.append("")

        # 1. 论文信息
        lines.append("## 1. 论文信息")
        lines.append(f"- **标题**: {paper_info.get('title', '未知')}")
        lines.append(f"- **方法**: {paper_info.get('method', '未知')}")
        lines.append(f"- **数据集**: {paper_info.get('dataset', '未知')}")
        lines.append(f"- **声明指标**: {paper_info.get('metrics', {})}")
        lines.append("")

        # 2. 资源定位
        lines.append("## 2. 资源定位")
        res = resources if resources else {}
        lines.append(f"- **代码仓库**: {res.get('code_repo_url', '未找到')}")
        urls = res.get("extracted_urls", [])
        if urls:
            lines.append(f"- **提取URL**: {len(urls)} 个")
            for u in urls[:5]:
                lines.append(f"  - {u}")
        lines.append("")

        # 3. 环境配置
        lines.append("## 3. 环境配置")
        env = env_config if env_config else {}
        lines.append(f"- **Python版本**: {env.get('python_version', '3.12')}")
        lines.append(f"- **依赖数**: {len(env.get('requirements_txt', '').split('\\n'))}")
        lines.append(f"- **预估磁盘**: {env.get('estimated_disk_gb', 'N/A')} GB")
        lines.append("")
        lines.append("### requirements.txt")
        lines.append("```")
        lines.append(env.get("requirements_txt", "无"))
        lines.append("```")
        lines.append("")

        # 4. 代码执行
        lines.append("## 4. 代码执行")
        exec_data = execution if execution else {}
        code = exec_data.get("code", "")
        lines.append(f"- **代码长度**: {len(code)} 字符")
        lines.append(f"- **执行状态**: {'✅ 成功' if exec_data.get('execution', {}).get('success') else '❌ 失败'}")
        lines.append("")
        if code:
            lines.append("### 生成代码")
            lines.append("```python")
            lines.append(code[:1000])
            lines.append("```")
            lines.append("")
        lines.append("### 执行输出")
        lines.append("```")
        lines.append(exec_data.get("execution", {}).get("stdout", "无输出")[:500])
        lines.append("```")
        if exec_data.get("execution", {}).get("stderr"):
            lines.append("### 错误输出")
            lines.append("```")
            lines.append(exec_data.get("execution", {}).get("stderr", "")[:500])
            lines.append("```")
        lines.append("")

        # 5. 验证结果
        lines.append("## 5. 验证结果")
        val = validation if validation else {}
        lines.append(f"- **复现状态**: {'✅ 成功' if val.get('is_reproduced') else '❌ 失败'}")
        lines.append(f"- **置信度**: {val.get('confidence', 0):.2f}")
        lines.append(f"- **分析**: {val.get('validation', {}).get('analysis', '无')}")
        lines.append("")
        metrics_comp = val.get("metrics_comparison", {})
        if metrics_comp:
            lines.append("### 指标对比")
            lines.append("| 指标 | 论文声明 | 实际运行 |")
            lines.append("|------|----------|----------|")
            paper_m = metrics_comp.get("paper", {})
            actual_m = metrics_comp.get("actual", {})
            all_keys = set(list(paper_m.keys()) + list(actual_m.keys()))
            for k in sorted(all_keys):
                pv = paper_m.get(k, "N/A")
                av = actual_m.get(k, "N/A")
                lines.append(f"| {k} | {pv} | {av} |")
        lines.append("")

        # 6. 审计统计
        lines.append("## 6. 审计统计")
        audit = data.get("audit_summary", {})
        lines.append(f"- **总步骤数**: {audit.get('total_steps', 'N/A')}")
        lines.append(f"- **成功步数**: {audit.get('success', 'N/A')}")
        lines.append(f"- **错误步数**: {audit.get('errors', 'N/A')}")
        lines.append(f"- **执行时长**: {audit.get('duration_sec', 'N/A')} 秒")
        lines.append("")

        lines.append("---")
        lines.append("*报告由 AutoReproducer 自动生成*")

        return "\n".join(lines)