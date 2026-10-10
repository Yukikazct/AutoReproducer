"""PaperReaderAgent - 论文解析 Agent，从 PDF/标题提取结构化信息。

流程：
1. 优先解析上传/本地 PDF（PyPDF2 主用，pdfplumber 兜底）；
2. 无 PDF 时使用论文标题作为输入；
3. LLM 提取结构化 JSON；LLM 不可用/解析失败时降级为本地规则提取。
"""
import json
import re
from typing import Dict
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient


class PaperReaderAgent(BaseAgent):
    """从论文 PDF 或标题中提取标题、方法、依赖、数值声明、数据集等结构化信息。"""

    system_prompt = "从论文中提取标题、作者、方法、依赖列表、数值声明与数据集,输出结构化 JSON"

    # LLM 输出中可能包裹 JSON 的常见噪音
    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
    # 判定字段是否为空/占位值（"未知""N/A" 及其变体，如
    # "未知（LLM 解析失败，本地降级提取）"），与 CodeExecutor 的判据一致
    _UNKNOWN_RE = re.compile(
        r"^\s*(|未知.*|未找到|无|n/?a|none|null)\s*$", re.IGNORECASE)

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("PaperReader", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """解析论文。

        input_data 支持:
          - "pdf_path": 本地 PDF 路径（优先）
          - "paper_title": 论文标题（无 PDF 时使用）
        """
        self.log("parse_paper", "START", "开始解析论文", input_data)

        pdf_path = input_data.get("pdf_path", "") or ""
        paper_title = input_data.get("paper_title", "") or ""

        pdf_text = ""
        pdf_input = None
        if pdf_path:
            from src.pdf_input import extract_pdf_input, supported_pdf_title
            # Reject an unreadable upload before making any model request. A
            # filename or user title cannot stand in for its missing content.
            pdf_input = extract_pdf_input(pdf_path)
            pdf_text = pdf_input.text

        # 标题-only：如实标注"没有正文"，绝不编造摘要。编造出来的占位摘要
        # 会被下游当成真实论文信息，进而生成与论文无关的代码。
        title_only = not pdf_text and bool(paper_title)
        if title_only:
            pdf_text = (f"论文标题: {paper_title}\n"
                        "（未获取到论文正文，以下仅有标题）")

        prompt = f"""请从以下论文内容中提取结构化信息，返回JSON格式：
{{
    "title": "论文标题",
    "authors": ["作者列表"],
    "method": "方法描述",
    "dependencies": ["依赖库列表"],
    "metrics": {{"指标名": 数值}},
    "metric_units": {{"指标名": "percent / fraction / 原始单位"}},
    "dataset": "数据集名称",
    "code_url": "代码仓库URL或'未找到'",
    "insufficient_info": false
}}

【信息不足时的处理 - 必须严格遵守】
若上面只有标题、没有正文摘要，只做**保守**推断：标题里明确写出的
方法/领域可以填写；推断不出的字段一律留空（method/dataset 填 ""、
metrics 填 {{}}），并把 "insufficient_info" 设为 true。
严禁编造摘要、数据集、指标数值等任何未经证实的信息。
指标单位必须来自原文：百分数记 percent，明确的 0~1 比例记 fraction；
MSE/MAE 等保留原始单位。单位未知时留空，不按数值大小猜百分比。

论文内容：
{pdf_text[:3000]}
"""
        llm_result = self.llm.chat(prompt, task="paper_reader")
        parsed = self._parse_json(llm_result)
        if not parsed or not parsed.get("title"):
            parsed = self._fallback_extract(pdf_text, paper_title)

        # For an upload, its visible identity takes priority over unrelated title
        # textbox state or a model guess. Title-only requests retain user input.
        visible_title = supported_pdf_title(pdf_input) if pdf_input is not None else None
        if visible_title:
            parsed["title"] = visible_title
        elif paper_title and pdf_input is None:
            parsed["title"] = paper_title
        # 透传"信息是否足以生成针对性复现代码"给下游
        # （CodeExecutor 据此拒绝编造代码，ResultValidator 据此判"无法验证"）
        if title_only and not getattr(self.llm, "mock_mode", False):
            parsed["insufficient_info"] = True
        elif not isinstance(parsed.get("insufficient_info"), bool):
            parsed["insufficient_info"] = self._judge_insufficient(
                parsed, title_only)
        parsed["info_sufficient"] = not parsed["insufficient_info"]

        self.log_experiment(
            "READ_PAPER", "解析论文并生成结构化信息（真相来源）",
            inputs={"pdf_path": pdf_path, "paper_title": paper_title},
            outputs=parsed,
        )
        self.log("parse_paper", "SUCCESS",
                 f"成功解析论文: {parsed.get('title', '未知')[:50]}", parsed)

        result = {
            "paper_info": parsed,
            "raw_text": pdf_text,
            "extracted_code_urls": re.findall(
                r"https?://github\.com/[^\s\)\]}\"]+", pdf_text),
            "llm_calls": self._delta_llm_calls(),
        }
        if pdf_input is not None:
            result["pdf_input"] = {"sha256": pdf_input.sha256,
                                   "pages": pdf_input.page_count, "bytes": pdf_input.byte_size,
                                   "readable": True}
        return result

    # ---------------- 内部工具 ----------------

    def _delta_llm_calls(self) -> int:
        """返回本次 run 消耗的 LLM 调用数（基于累计计数）。"""
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    @classmethod
    def _judge_insufficient(cls, parsed: Dict, title_only: bool) -> bool:
        """本地判定：这份结构化信息是否足以生成针对性复现代码。

        方法、数据集、声明指标三者皆空 -> 信息不足。此时下游不应编造
        代码，而应诚实报"无法复现"。
        """
        method = str(parsed.get("method", "") or "")
        dataset = str(parsed.get("dataset", "") or "")
        metrics = parsed.get("metrics") or {}
        if title_only and not method.strip():
            return True      # 只有标题，且连方法都没推断出来
        return bool(cls._UNKNOWN_RE.match(method)
                    and cls._UNKNOWN_RE.match(dataset) and not metrics)

    def _parse_json(self, text: str) -> Dict:
        """宽容解析 LLM 返回的 JSON（容忍代码围栏与首尾噪音）。"""
        for candidate in (text, self._strip_fence(text)):
            if not candidate or not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        return {}

    @staticmethod
    def _strip_fence(text: str) -> str:
        m = PaperReaderAgent._JSON_FENCE_RE.search(text)
        return m.group(1) if m else text

    @staticmethod
    def _fallback_extract(text: str, title: str) -> Dict:
        """本地规则降级：从文本中提取标题、指标与代码链接。

        method 必须保持「未知」开头（下游 `_judge_insufficient` / CodeExecutor
        据此判定信息不足），但括号内如实说明降级原因，避免把「PDF 文本提取失败」
        误报成「LLM 不可用」。
        """
        text = text or ""
        if text.strip():
            method = "未知（LLM 解析失败，本地降级提取）"
        else:
            method = ("未知（未获取到论文正文：PDF 文本提取失败，"
                      "可能缺少 PyPDF2/pdfplumber 依赖，或为扫描件/图片型 PDF）")
        info: Dict = {
            "title": title or "未知标题",
            "authors": [],
            "method": method,
            "dependencies": [],
            "metrics": {},
            "metric_units": {},
            "dataset": "未知",
            "code_url": "未找到",
        }
        if text:
            title_line = next(
                (ln for ln in text.splitlines()
                 if ln.strip().startswith("论文标题") or ln.startswith("Title")),
                "")
            if title_line:
                info["title"] = title_line.split(":", 1)[-1].strip() or info["title"]
            code_urls = re.findall(r"https?://[^\s\)\]}\"]*github[^\s\)\]}\"]*", text)
            if code_urls:
                info["code_url"] = code_urls[0]
            metrics_re = re.findall(
                r"(?:accuracy|acc|准确率)\s*[:：]\s*([\d.]+)\s*(%)?", text,
                re.IGNORECASE)
            if metrics_re:
                info["metrics"]["accuracy"] = float(metrics_re[0][0])
                if metrics_re[0][1]:
                    info["metric_units"]["accuracy"] = "percent"
        return info

    def _extract_text(self, pdf_path: str) -> str:
        """Extract readable content; never silently substitute an empty PDF."""
        from src.pdf_input import extract_pdf_input
        return extract_pdf_input(pdf_path).text
