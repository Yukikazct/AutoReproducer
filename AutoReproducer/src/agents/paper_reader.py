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
from src.repository_evidence import extract_repository_links


READING_TEXT_LIMIT = 120000


def reading_context(text):
    """Read ordinary papers in full; label distributed excerpts for long ones."""
    if len(text) <= READING_TEXT_LIMIT:
        return text, {"complete": True, "total_characters": len(text),
                      "spans": [[0, len(text)]]}
    # Retain both the identity and late experiments/appendices. No guessed
    # section boundaries or silent first-page truncation can hide the results.
    width, count = 12000, 10
    starts = [round(i * (len(text) - width) / (count - 1)) for i in range(count)]
    spans = [[start, start + width] for start in starts]
    context = "\n\n".join(f"[原文字符 {start}:{end}，全文 {len(text)} 字符]\n{text[start:end]}"
                          for start, end in spans)
    return context, {"complete": False, "total_characters": len(text), "spans": spans}


def repository_reading_context(text, links):
    """Identity header plus code-availability locations from the entire paper."""
    spans = [[0, min(10000, len(text))]]
    markers = re.compile(r"(?i)(?:source\s+code|code\s+(?:is\s+)?avail|implement(?:ation|ed)|"
                         r"github\s*\.?\s*com|github|software\s+(?:is\s+)?avail)")
    for match in markers.finditer(text):
        start, end = max(0, match.start() - 600), min(len(text), match.end() + 1200)
        if spans[-1][0] <= start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    # Candidate evidence always survives this auxiliary prose bound, including
    # numbered footnotes and URI annotations outside the selected text windows.
    selected, size = [], 0
    for start, end in spans:
        end = min(end, start + 40000 - size)
        if end <= start:
            break
        selected.append([start, end])
        size += end - start
    context = "\n\n".join(f"[原文字符 {start}:{end}]\n{text[start:end]}" for start, end in selected)
    context += "\n\n[全 PDF 页面和 URI 注释提取的仓库位置，保留页码、代码声明与脚注]\n"
    context += json.dumps(links, ensure_ascii=False)
    return context, {"complete": False, "purpose": "repository_discovery",
                     "total_characters": len(text), "spans": selected,
                     "repository_locations": len(links)}


class PaperReaderAgent(BaseAgent):
    """从论文 PDF 或标题中提取标题、方法、依赖、数值声明、数据集等结构化信息。"""

    system_prompt = ("依据实际论文提取标题、作者、方法、软件依赖、实验指标与数据集，返回结构化对象。"
                     "dependencies 只指原文明确提及的软件库，不包括基线模型、引用方法或论文。"
                     "原文未说明软件库时空列表是正确结果。metrics 只包含带实验语境和单位的结果指标，"
                     "不能把层数、训练轮数或基线结果当成本方法指标。没有证据的字段应留空，禁止补造。")
    repository_system_prompt = ("当前任务是从实际 PDF 定位仓库候选：保留论文身份、实际 GitHub 地址、"
        "所在页及代码声明/脚注证据。引用/基线仓库可以是候选但不能冒充作者实现。"
        "本阶段不要求数据集、软件依赖或实验数值；它们留空不构成错误。"
        "作者身份将在拉取固定 Git 源码后独立核验，不能只凭链接提前确认。")

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

        # The language-model viewport is intentionally bounded, but repository
        # evidence comes from every page and link annotation, independently.
        repository_links = list(pdf_input.repository_links) if pdf_input is not None else []
        purpose = input_data.get("reading_purpose", "experiment")
        context, coverage = (repository_reading_context(pdf_text, repository_links)
                             if purpose == "repository_discovery" else reading_context(pdf_text))

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
dependencies 只填写原文明确命名的软件库（如 PyTorch），不是基线模型或引用方法。
正文没有软件依赖声明时填 []，不为满足字段而推测依赖。
metrics 只提取本论文方法的实验结果，保留数据集/方法变体语境（可写在指标名称中），
不把模型层数、训练轮数、超参数或他人基线的数字作为结果指标。
如果给的是标明范围的原文节选，未展示的内容不等于原文没有；如实保留信息缺口。
指标单位必须来自原文：百分数记 percent，明确的 0~1 比例记 fraction；
MSE/MAE 等保留原始单位。单位未知时留空，不按数值大小猜百分比。

论文内容：
{context}

从完整 PDF 确定性提取的仓库链接及上下文（引用链接不等于作者实现）：
{json.dumps(repository_links[:12], ensure_ascii=False)}
"""
        if purpose == "repository_discovery":
            prompt += ("\n当前阶段仅定位论文仓库。依据上述完整文档扫描所得的位置核对论文身份、"
                       "代码声明和候选地址；method 可简要说明，dependencies、metrics、dataset "
                       "可以留空。实验细节将在实际仓库确认之后从完整论文与源码重新提取。")
        feedback = input_data.get("retry_feedback")
        if isinstance(feedback, dict) and feedback.get("agent") == self.name:
            prompt += ("\n上一轮审查反馈（只依据上述实际原文修正；无证据的建议不能执行）：\n"
                       + json.dumps(feedback, ensure_ascii=False))
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
        if pdf_input is not None:
            # A guessed URL is not paper evidence. Only an actual author-code
            # declaration can fill this field; weaker links remain candidates.
            code_evidence = next((link for link in repository_links if link["is_author_code"]), None)
            parsed["code_url"] = code_evidence["url"] if code_evidence else "未找到"
            parsed["code_url_evidence"] = code_evidence
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
            "reading_context": context,
            "reading_coverage": coverage,
            "reading_purpose": purpose,
            "extracted_code_urls": list(dict.fromkeys(link["url"] for link in repository_links)),
            "extracted_repository_links": repository_links,
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
            code_evidence = next((link for link in extract_repository_links([text])
                                  if link["is_author_code"]), None)
            if code_evidence:
                info["code_url"] = code_evidence["url"]
                info["code_url_evidence"] = code_evidence
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
