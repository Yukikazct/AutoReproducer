"""ResourceFinderAgent - 资源查找 Agent，定位代码仓库与数据集。

流程（确定性优先，LLM 补充）：
1. 确定性仓库发现链（P1-⑨，无 LLM，见 src/agents/repo_discovery.py）：
   用户显式 URL -> Papers with Code -> GitHub 搜索 -> curated 回退；
2. 从论文原文中提取 URL（正则）作为额外线索；
3. LLM 综合论文信息补全数据集等资源；LLM 失败时使用提取结果降级；
4. 复现模式决策（auto/smoke/full）写入结果，供编排与存储分层使用。
"""
import json
import math
import re
from copy import deepcopy
from typing import Dict

from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.agents.repo_discovery import (
    build_repo_discovery_query,
    clean_paper_title,
    decide_reproduction_mode,
    discover_repositories,
    normalize_github_repo_url,
    probe_memory_gb,
)
from src.resource_manager import _is_placeholder_url  # noqa: F401  复用占位判定
from src.repository_evidence import extract_repository_links, normalize_repository_url

_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


class ResourceFinderAgent(BaseAgent):
    """从论文信息中查找代码仓库、数据集和相关资源。"""

    system_prompt = "根据论文信息定位代码仓库与数据集,输出仓库 URL、数据源列表及置信度"

    def __init__(self, llm_client: LLMClient, logger=None,
                 offline: bool = False):
        super().__init__("ResourceFinder", logger)
        self.llm = llm_client
        self.offline = offline

    def run(self, input_data: dict) -> dict:
        """查找论文相关资源。

        input_data: {"paper_info": dict, "raw_text": str,
                     "preferred_repo_url": str, "repro_mode": str}
        """
        self.log("find_resources", "START", "开始查找资源", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        raw_text = input_data.get("raw_text", "") or ""

        urls = list(dict.fromkeys([
            *self._extract_urls(raw_text),
            *(url for url in input_data.get("extracted_code_urls", []) if isinstance(url, str)),
        ]))
        repository_links = self._repository_links(input_data, raw_text)
        github_urls = list(dict.fromkeys(link["url"] for link in repository_links))
        urls = list(dict.fromkeys([*urls, *github_urls]))
        user_repo = (input_data.get("preferred_repo_url")
                     or input_data.get("code_repo_url") or "")
        paper_repo = (paper_info.get("code_url")
                      or paper_info.get("code_repo_url") or "")
        user_repo = self._normalize_repository(user_repo)
        paper_repo = self._normalize_repository(paper_repo)

        # A paper's own code link precedes lexical search results. Preserve its
        # source separately: model metadata alone does not prove authorship.
        query = build_repo_discovery_query(paper_info)
        paper_link = next((link for link in repository_links
                           if link.get("evidence_type") != "reference"), None)
        metadata_is_reference = bool(paper_repo and any(link["url"] == paper_repo for link in repository_links)
                                     and all(link.get("evidence_type") == "reference"
                                             for link in repository_links if link["url"] == paper_repo))
        metadata_preferred = "" if metadata_is_reference else paper_repo
        preferred = user_repo or (paper_link["url"] if paper_link else "") or metadata_preferred
        selection_evidence = next((link for link in repository_links if link["url"] == preferred), None)
        discovery = discover_repositories(
            query, preferred_url=preferred,
            timeout=_int_env("PWC_HTTP_TIMEOUT", 8),
            offline=self.offline)
        selected = discovery.get("selected_repo", "")

        if selected and preferred:
            source = "user_preference" if user_repo else (
                "paper_code_url" if preferred == paper_repo else "paper_text_url")
            discovery["selection_source"] = source
            if not user_repo:
                discovery["discovery_chain"] = [source]
                for candidate in discovery.get("candidates", []):
                    candidate["source"] = source
                    candidate["title"] = "Repository linked in paper" if selection_evidence else "Parsed repository candidate"
                    candidate["description"] = ("Repository URL grounded in paper evidence."
                                                if selection_evidence else "Unverified repository URL from parsed metadata.")
            if selection_evidence is not None:
                discovery["selection_evidence"] = deepcopy(selection_evidence)
            if paper_repo and paper_repo != preferred:
                discovery["ignored_repository_guesses"] = [{
                    "url": paper_repo, "source": "paper_info.code_url",
                    "reason": "论文原文仓库证据优先于模型元数据候选。"}]

        # 1.1 pin revision：paper_info 显式字段或文本中 commit/<sha> 线索
        pinned_revision = self._extract_pinned_revision(
            paper_info, raw_text, repo_url=selected)
        if pinned_revision:
            discovery["pinned_revision"] = pinned_revision

        # 2. 从文本提取 URL（真实线索，补充候选）
        if paper_link and not selected:
            selected = paper_link["url"]
            discovery["selected_repo"] = selected
            discovery["fallback_used"] = True
            discovery["selection_source"] = "paper_text_url"

        # 3. Model guesses supplement candidates; search does not prove that a
        # repository is maintained by the paper's authors.
        prompt = f"""根据论文信息补全代码仓库候选和使用的数据集。
论文标题: {paper_info.get('title', '未知')}
方法: {paper_info.get('method', '未知')}
已知代码仓库: {selected or '未找到'}
论文原文仓库证据: {json.dumps(selection_evidence, ensure_ascii=False) if selection_evidence else '无'}
若已知仓库有论文原文证据，保留该URL；其他候选不能替代原文链接。

返回JSON格式:
{{
    "code_repo_url": "最可能的GitHub URL或'未找到'",
    "alternative_repos": ["备用仓库1", "备用仓库2"],
    "dataset_url": "数据集URL或'未找到'",
    "weights_url": "预训练权重URL或'未找到'",
    "confidence": 0.0-1.0
}}
"""
        llm_result = self.llm.chat(prompt, task="resource_finder")
        parsed = self._parse_json(llm_result)
        if not parsed or not parsed.get("code_repo_url"):
            parsed = {
                "code_repo_url": selected or "未找到",
                "alternative_repos": [],
                "dataset_url": "未找到",
                "weights_url": "未找到",
                "confidence": 0.7 if github_urls or selected else 0.3,
            }

        # 4. Retain a candidate while making its identity evidence explicit.
        code_repo_url = selected or parsed.get("code_repo_url", "未找到")
        if _is_placeholder_url(code_repo_url or ""):
            code_repo_url = "未找到"
        linked_implementation_urls = [link["url"] for link in repository_links
                                      if link.get("evidence_type") != "reference"]
        identity = self._repository_identity(code_repo_url, discovery, user_repo, linked_implementation_urls)
        if selection_evidence and selection_evidence["url"] == self._normalize_repository(code_repo_url):
            identity["evidence"] = deepcopy(selection_evidence)
            if selection_evidence.get("is_author_code") and identity["status"] == "paper_linked":
                page = selection_evidence.get("page")
                location = f"PDF 第 {page} 页" if page else "论文原文"
                identity["reason"] = location + "的代码公开声明直接链接此仓库；作者身份和实际执行另行核验。"
        discovery["repository_identity"] = identity

        # 5. 复现模式决策（auto -> smoke，full 需显式确认）
        requested_mode = (input_data.get("repro_mode")
                          or paper_info.get("repro_mode") or "auto")
        repro_mode = decide_reproduction_mode(
            requested=str(requested_mode),
            full_requested=_as_bool(input_data.get("full_reproduction")),
            memory_gb=probe_memory_gb())

        resources: Dict = {
            "code_repo_url": code_repo_url,
            "selected_repo": code_repo_url,
            "alternative_repos": parsed.get("alternative_repos", []),
            "dataset_url": parsed.get("dataset_url", "未找到"),
            "weights_url": parsed.get("weights_url", "未找到"),
            "confidence": max(_confidence(parsed.get("confidence")), 0.8)
                if identity["status"] in {"user_selected", "paper_linked"}
                else min(_confidence(parsed.get("confidence")), 0.5),
            "repository_identity": identity,
            "repo_discovery": discovery,
            "repro_mode": repro_mode,
            "extracted_urls": urls[:10],
            "github_urls": github_urls,
            "repository_links": repository_links,
            "selection_evidence": deepcopy(selection_evidence),
        }

        self.log_experiment(
            "FIND_RESOURCES", "定位代码仓库与数据集",
            inputs={"paper_info": paper_info},
            outputs=resources,
        )
        self.log("find_resources", "SUCCESS",
                 f"发现链={discovery.get('discovery_chain', [])},"
                 f"仓库线索 {code_repo_url or '无'}（{identity['status']}），"
                 f"复现模式 {repro_mode.get('effective_mode')}",
                 {"selected": code_repo_url,
                  "chain": discovery.get("discovery_chain", []),
                  "repro_mode": repro_mode.get("effective_mode")})

        return {
            "resources": resources,
            "llm_calls": self._delta_llm_calls(),
        }

    # ---------------- 内部工具 ----------------

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        for candidate in (text, re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)):
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
    def _extract_urls(text: str) -> list:
        """从文本中提取 URL。"""
        url_pattern = r'https?://[^\s\)\]}"]+'
        urls = [url for url in re.findall(url_pattern, text)
                if not re.match(r"https?://(?:www\.)?github(?:\.|/)", url, re.I)
                or normalize_repository_url(url)]
        return list(dict.fromkeys([*urls, *(link["url"] for link in extract_repository_links([text]))]))

    @staticmethod
    def _normalize_repository(value):
        if not isinstance(value, str):
            return ""
        normalized = normalize_repository_url(value)
        if normalized:
            return normalized
        return normalize_github_repo_url(value) if re.fullmatch(r"[\w.-]+/[\w.-]+", value.strip()) else ""

    @classmethod
    def _repository_links(cls, input_data, raw_text):
        """Keep page-grounded evidence ahead of metadata and lexical guesses."""
        links = []
        for item in input_data.get("extracted_repository_links", []) or []:
            if not isinstance(item, dict):
                continue
            url = cls._normalize_repository(item.get("url", ""))
            if url:
                links.append({**deepcopy(item), "url": url})
        if not links:
            # Flattened legacy text has no reliable page boundaries. Never
            # reinterpret the complete PDF when page-grounded records exist.
            links.extend({**item, "page": None, "source": "paper_text"}
                         for item in extract_repository_links([raw_text]))
        for raw_url in input_data.get("extracted_code_urls", []) or []:
            if not isinstance(raw_url, str):
                continue
            url = cls._normalize_repository(raw_url)
            if url and not any(item["url"] == url for item in links):
                links.append({"url": url, "raw_url": raw_url, "source": "paper_extracted_url",
                              "page": None, "context": "", "evidence_type": "repository_link",
                              "is_author_code": False})
        ranks = {"author_code_statement": 0, "repository_link": 1, "reference": 2}
        links.sort(key=lambda item: ranks.get(item.get("evidence_type"), 1))
        result, seen = [], set()
        for item in links:
            key = (item["url"], item.get("page"), item.get("source"), item.get("evidence_type"))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _extract_pinned_revision(paper_info: dict, raw_text: str, repo_url: str = "") -> str:
        """提取 pin revision：paper_info 显式字段 > 文本 commit 线索。

        只接受 GitHub 标准 commit URL（.../commit/<sha>）与显式
        "commit <40位sha>" 模式，避免误抓普通数字/版本号。
        """
        for key in ("code_revision", "revision", "commit_sha"):
            value = (paper_info or {}).get(key)
            if value and isinstance(value, str) and _SHA_RE.fullmatch(value):
                return value
        for match in re.finditer(
                r"(https?://github\.com/[^/\s]+/[^/\s]+)/commit/([0-9a-fA-F]{7,40})", raw_text):
            if not repo_url or normalize_github_repo_url(match.group(1)) == repo_url:
                return match.group(2)
        match = re.search(r"\bcommit\s+([0-9a-fA-F]{40})\b", raw_text)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _repository_identity(repo_url, discovery, user_repo, paper_urls):
        normalized = normalize_github_repo_url(repo_url)
        source = discovery.get("selection_source") or next(
            (candidate.get("source", "") for candidate in discovery.get("candidates", [])
             if normalized in candidate.get("repo_urls", [])), "llm_guess")
        status = "not_found"
        if normalized:
            status = "user_selected" if normalized == user_repo else (
                "paper_linked" if normalized in {normalize_github_repo_url(url) for url in paper_urls}
                else "candidate_unverified")
        return {"status": status, "source": source, "is_official": None,
                "repository_executed": False,
                "reason": {"not_found": "未定位到可用仓库。",
                           "user_selected": "用户指定仓库；作者身份仍需独立核验。",
                           "paper_linked": "论文文本包含仓库链接；未据此宣称为作者官方实现。",
                           "candidate_unverified": "搜索或模型只提供候选，尚未核验与论文作者的关联。"}[status]}


def _int_env(name: str, default: int) -> int:
    try:
        return int(__import__("os").environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _confidence(value) -> float:
    try:
        parsed = float(value)
        return min(max(parsed, 0.0), 1.0) if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)
