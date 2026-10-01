"""ResourceFinderAgent - 资源查找 Agent，定位代码仓库与数据集。

流程（确定性优先，LLM 补充）：
1. 确定性仓库发现链（P1-⑨，无 LLM，见 src/agents/repo_discovery.py）：
   用户显式 URL -> Papers with Code -> GitHub 搜索 -> curated 回退；
2. 从论文原文中提取 URL（正则）作为额外线索；
3. LLM 综合论文信息补全数据集等资源；LLM 失败时使用提取结果降级；
4. 复现模式决策（auto/smoke/full）写入结果，供编排与存储分层使用。
"""
import json
import re
from typing import Dict

from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.agents.repo_discovery import (
    build_repo_discovery_query,
    clean_paper_title,
    decide_reproduction_mode,
    discover_repositories,
    probe_memory_gb,
)
from src.resource_manager import _is_placeholder_url  # noqa: F401  复用占位判定

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

        # 1. 确定性仓库发现链（用户 URL 优先，网络尽力而为）
        query = build_repo_discovery_query(paper_info)
        preferred = (input_data.get("preferred_repo_url")
                     or paper_info.get("code_repo_url") or "")
        discovery = discover_repositories(
            query, preferred_url=preferred,
            timeout=_int_env("PWC_HTTP_TIMEOUT", 8),
            offline=self.offline)
        selected = discovery.get("selected_repo", "")

        # 1.1 pin revision：paper_info 显式字段或文本中 commit/<sha> 线索
        pinned_revision = self._extract_pinned_revision(
            paper_info, raw_text)
        if pinned_revision:
            discovery["pinned_revision"] = pinned_revision

        # 2. 从文本提取 URL（真实线索，补充候选）
        urls = self._extract_urls(raw_text)
        github_urls = [u for u in urls if "github.com" in u]
        if github_urls and not selected:
            selected = github_urls[0]
            discovery["selected_repo"] = selected
            discovery["fallback_used"] = True

        # 3. LLM 推测（数据集等补充信息；仓库已由确定性链保证）
        prompt = f"""根据论文信息，推测代码仓库URL和使用的数据集。
论文标题: {paper_info.get('title', '未知')}
方法: {paper_info.get('method', '未知')}
已知代码仓库: {selected or '未找到'}

返回JSON格式:
{{
    "code_repo_url": "最可能的GitHub URL或'未找到'",
    "alternative_repos": ["备用仓库1", "备用仓库2"],
    "benchmark_framework_url": "论文使用的基准/评测框架仓库URL或'未找到'",
    "dataset_url": "数据集URL或'未找到'",
    "weights_url": "预训练权重URL或'未找到'",
    "confidence": 0.0-1.0
}}

备用仓库必须实现同一篇论文；不要把论文对比的其他模型仓库列为备用。
benchmark_framework_url 只填写运行该实现必需的独立评测框架；没有则填"未找到"。
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

        # 4. 确定性发现的仓库为权威主线；LLM 按论文信息兜底
        code_repo_url = selected or parsed.get("code_repo_url", "未找到")
        if _is_placeholder_url(code_repo_url or ""):
            code_repo_url = "未找到"
        if code_repo_url != "未找到" and not discovery.get("selected_repo"):
            from src.agents.repo_discovery import normalize_github_repo_url
            code_repo_url = normalize_github_repo_url(code_repo_url) or "未找到"
            if code_repo_url != "未找到":
                discovery["selected_repo"] = code_repo_url
                discovery["fallback_used"] = True
                discovery["discovery_chain"].append("llm_inferred")
                discovery["code_units"] = [{"unit_id": "main", "role": "main",
                                            "url": code_repo_url, "source": "llm_inferred"}]

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
            "confidence": max(
                parsed.get("confidence", 0.0), 0.8 if selected else 0.0),
            "repo_discovery": discovery,
            "repro_mode": repro_mode,
            "extracted_urls": urls[:10],
            "github_urls": github_urls,
        }

        # 6. 多代码单元组装：发现链 main + LLM 备选/基准框架 + 数据集
        #    所在仓库（iTransformer 的数据集 URL 指向 Time-Series-Library
        #    即为 library 单元），供 ResourceManager 多仓库拉取与
        #    ExecutionPlanner 规划使用。
        resources["code_units"] = self._assemble_code_units(
            discovery, parsed, pinned_revision)

        self.log_experiment(
            "FIND_RESOURCES", "定位代码仓库与数据集",
            inputs={"paper_info": paper_info},
            outputs=resources,
        )
        self.log("find_resources", "SUCCESS",
                 f"发现链={discovery.get('discovery_chain', [])},"
                 f"选中 {code_repo_url or '无'}，"
                 f"复现模式 {repro_mode.get('effective_mode')}",
                 {"selected": code_repo_url,
                  "chain": discovery.get("discovery_chain", []),
                  "repro_mode": repro_mode.get("effective_mode")})

        return {
            "resources": resources,
            "llm_calls": self._delta_llm_calls(),
        }

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _assemble_code_units(discovery: Dict, parsed: Dict,
                             pinned_revision: str = "") -> list:
        """组装 code_units 列表（字典形态，落 resources.code_units）。

        来源优先级：发现链（selected=main + 可信候选）> LLM 备选仓库
        > LLM benchmark 框架 > 数据集所在 GitHub 仓库。全部按 URL 去重，
        保序；unit_id 生成规则与 src.code_units 一致（main / alt_N /
        lib_N）。
        """
        from src.agents.repo_discovery import normalize_github_repo_url
        from src.code_units import dedupe_units, next_unit_id, CodeUnit

        units: list = []
        for raw in (discovery.get("code_units") or []):
            unit = CodeUnit.from_dict(raw) if isinstance(raw, dict) \
                else CodeUnit()
            if not unit.url:
                continue
            if unit.unit_id == "main" and pinned_revision:
                unit.revision = pinned_revision
            units.append(unit)

        def _append(url: str, role: str, source: str, notes: str = "") -> None:
            normalized = normalize_github_repo_url(url or "")
            if not normalized:
                return
            if any(u.url == normalized for u in units):
                return
            prefix = "lib" if role == "library" else "alt"
            units.append(CodeUnit(
                unit_id=next_unit_id(units, prefix),
                role=role, url=normalized, source=source, notes=notes))

        # LLM 备选仓库（最多 3 个）
        for alt in (parsed.get("alternative_repos") or [])[:3]:
            _append(alt, "alternative", "llm_inferred", "LLM 推断备选实现")
        # LLM 基准框架（若给出）
        benchmark = parsed.get("benchmark_framework_url") or ""
        if benchmark and benchmark != "未找到":
            _append(benchmark, "benchmark", "llm_inferred", "基准评测框架")
        # 数据集 URL 本身是 GitHub 仓库（数据集与代码同仓的场景）
        dataset_url = parsed.get("dataset_url") or ""
        if dataset_url and dataset_url != "未找到":
            normalized = normalize_github_repo_url(dataset_url)
            if normalized and not any(u.url == normalized for u in units):
                units.append(CodeUnit(
                    unit_id=next_unit_id(units, "lib"),
                    role="library", url=normalized, source="llm_inferred",
                    notes="数据集位于该仓库（README 数据集链接指向它）"))
        return [u.to_dict() for u in dedupe_units(units)]

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
        return list(dict.fromkeys(re.findall(url_pattern, text)))

    @staticmethod
    def _extract_pinned_revision(paper_info: dict, raw_text: str) -> str:
        """提取 pin revision：paper_info 显式字段 > 文本 commit 线索。

        只接受 GitHub 标准 commit URL（.../commit/<sha>）与显式
        "commit <40位sha>" 模式，避免误抓普通数字/版本号。
        """
        for key in ("code_revision", "revision", "commit_sha"):
            value = (paper_info or {}).get(key)
            if value and isinstance(value, str) and _SHA_RE.fullmatch(value):
                return value
        match = re.search(
            r"github\.com/[^/\s]+/[^/\s]+/commit/([0-9a-fA-F]{7,40})",
            raw_text)
        if match:
            return match.group(1)
        match = re.search(r"\bcommit\s+([0-9a-fA-F]{40})\b", raw_text)
        if match:
            return match.group(1)
        return ""


def _int_env(name: str, default: int) -> int:
    try:
        return int(__import__("os").environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)
