"""确定性仓库发现链（P1-⑨，对齐 ScholarAgent repo_ops 设计）。

发现链（全部无 LLM，网络尽力而为，失败逐层降级不回退）：
1. 用户显式 URL（preferred_repo_url）：最高优先，直接采用；
2. Papers with Code（HuggingFace Papers API）按论文查询：返回候选仓库；
3. GitHub 搜索 API 回退：用标题衍生的多个查询检索，按分数排序；
4. curated 内置回退：内置少量知名论文映射 + AUTOREPRO_CURATED_REPOS
   环境变量 JSON 扩展，保证离线/受限网络下仍有确定性候选。

配套决策：
- decide_reproduction_mode：auto/smoke/full 复现模式决策（auto→smoke，
  full 需显式确认否则降级 smoke，防误伤）；
- fetch 语义：浅克隆 + pin revision + 源标记写入（见 resource_manager）。

所有网络调用走 urllib（标准库，无第三方依赖）；任何异常只记录错误
字符串，不抛出，保证确定性链路不因网络中断。
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?"
    r"(?:\s|$|/|\)|\")")
ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- 基础工具 ----------------

def clean_paper_title(text: str) -> str:
    """清洗论文标题：去引号/标点，压缩空白。"""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r\"'`")


def normalize_search_query(text: str, max_len: int = 200) -> str:
    return clean_paper_title(text)[:max_len]


def normalize_github_repo_url(value: str) -> str:
    """把 github 链接归一为 https://github.com/<owner>/<repo> 或空串。"""
    if not value:
        return ""
    value = value.strip()
    match = GITHUB_URL_RE.search(value)
    if not match:
        # 支持 owner/repo 裸形式（限 github.com 语境）
        bare = re.match(r"^([\w.-]+)/([\w.-]+)$", value)
        if bare:
            return f"https://github.com/{bare.group(1)}/{bare.group(2)}"
        return ""
    owner, repo = match.group(1), match.group(2)
    return f"https://github.com/{owner}/{repo}"


def extract_plain_github_urls(text: str) -> List[str]:
    """从任意文本中提取全部去重、归一化的 GitHub URL。"""
    if not text:
        return []
    urls: List[str] = []
    for candidate in GITHUB_URL_RE.findall(text):
        url = f"https://github.com/{candidate[0]}/{candidate[1].rstrip('/')}"
        if url not in urls:
            urls.append(url)
    return urls


# ---------------- RepoCandidate ----------------

@dataclass
class RepoCandidate:
    title: str = ""
    repo_name: str = ""
    description: str = ""
    repo_urls: List[str] = field(default_factory=list)
    source: str = ""          # user_preference / papers_with_code / github_search / curated_fallback
    score_hint: int = 0
    stars: int = 0
    paper_id: str = ""

    def to_dict(self) -> Dict:
        payload = {
            "repo_urls": self.repo_urls,
            "source": self.source,
            "score_hint": self.score_hint,
        }
        for key in ("title", "repo_name", "description", "stars", "paper_id"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        return payload


def significant_tokens(text: str) -> List[str]:
    """提取查询的显著 token（用于宽松匹配）。"""
    stop = {"is", "all", "you", "the", "a", "an", "of", "and",
            "for", "with", "paper", "implementation", "learning",
            "using", "towards", "via", "on"}
    out: List[str] = []
    for token in re.split(r"[^a-z0-9]+", clean_paper_title(text).lower()):
        if len(token) < 3 or token in stop or token in out:
            continue
        out.append(token)
    return out


def candidate_search_text(candidate: RepoCandidate) -> str:
    parts = [candidate.repo_name, candidate.description, *candidate.repo_urls]
    for url in candidate.repo_urls:
        parsed = urllib.parse.urlparse(url)
        parts.append(parsed.path.strip("/"))
    return " ".join(part for part in parts if part)


def is_trusted_repo_candidate(query: str, candidate: RepoCandidate) -> bool:
    """判定候选是否与查询主题可信匹配（词面重合启发式，无 LLM）。"""
    if candidate.source == "user_preference" \
            or candidate.source == "curated_fallback":
        return True
    lower_query = clean_paper_title(query).lower()
    lower_text = candidate_search_text(candidate).lower()
    if not lower_query or not lower_text:
        return False
    tokens = significant_tokens(lower_query)
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in lower_text)
    return matched >= 2 if len(tokens) >= 4 else matched >= 1


def build_repo_discovery_query(paper_info: Dict) -> str:
    """从论文信息构造确定性查询：arxiv id > 标题 > 方法名。"""
    for key in ("arxiv_id", "arxivId", "paper_id"):
        value = paper_info.get(key) or ""
        if value and re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$",
                              str(value).strip()):
            return str(value).strip()
    title = paper_info.get("title") or ""
    title_q = normalize_search_query(title)
    if title_q:
        return title_q
    method = paper_info.get("method") or ""
    method_q = normalize_search_query(method)
    if method_q:
        return method_q
    return ""


# ---------------- curated 回退 ----------------

def curated_repo_fallback_candidates(query: str) -> List[RepoCandidate]:
    """内置 curated 回退：知名论文 -> 稳定实现仓库；支持环境变量扩展。

    环境变量 AUTOREPRO_CURATED_REPOS 为一个 JSON 映射：
    {"关键词片段": [{"repo_url": "...", "repo_name": "...", "description": "..."}]}
    关键词片段命中查询的干净标题（子串匹配，均小写）。
    """
    lower = clean_paper_title(query).lower()
    out: List[RepoCandidate] = []
    if not lower:
        return out

    builtin = {
        "attention is all you need": [{
            "repo_name": "harvardnlp/annotated-transformer",
            "description": "Annotated PyTorch implementation of the Transformer paper.",
            "repo_url": "https://github.com/harvardnlp/annotated-transformer",
        }],
        "transformer": [{
            "repo_name": "harvardnlp/annotated-transformer",
            "description": "Annotated PyTorch implementation of the Transformer paper.",
            "repo_url": "https://github.com/harvardnlp/annotated-transformer",
        }],
        "pinn": [{
            "repo_name": "maziarraissi/PINNs",
            "description": "Physics-informed neural networks (PINNs).",
            "repo_url": "https://github.com/maziarraissi/PINNs",
        }],
        "physics-informed": [{
            "repo_name": "maziarraissi/PINNs",
            "description": "Physics-informed neural networks (PINNs).",
            "repo_url": "https://github.com/maziarraissi/PINNs",
        }],
        "u-net": [{
            "repo_name": "milesial/Pytorch-UNet",
            "description": "PyTorch implementation of the U-Net for image segmentation.",
            "repo_url": "https://github.com/milesial/Pytorch-UNet",
        }],
        "unet": [{
            "repo_name": "milesial/Pytorch-UNet",
            "description": "PyTorch implementation of the U-Net for image segmentation.",
            "repo_url": "https://github.com/milesial/Pytorch-UNet",
        }],
    }
    extra_json = os.environ.get("AUTOREPRO_CURATED_REPOS", "").strip()
    extra: Dict = {}
    if extra_json:
        try:
            loaded = json.loads(extra_json)
            if isinstance(loaded, dict):
                extra = loaded
        except json.JSONDecodeError:
            extra = {}

    merged = dict(builtin)
    merged.update(extra)
    for keyword, entries in merged.items():
        if keyword.lower() not in lower:
            continue
        for entry in entries:
            url = normalize_github_repo_url(entry.get("repo_url", ""))
            name = entry.get("repo_name") or ""
            if not name and url:
                name = url.rstrip("/").split("github.com/")[-1]
            if not url:
                continue
            out.append(RepoCandidate(
                title=clean_paper_title(query),
                repo_name=name,
                description=entry.get("description", ""),
                repo_urls=[url],
                source="curated_fallback",
                score_hint=180,
            ))
    return out


# ---------------- GitHub 搜索评分 ----------------

def github_repo_search_score(query: str, full_name: str,
                             description: str, stars: int) -> int:
    score = 0
    lower_query = clean_paper_title(query).lower()
    full_name_l, desc_l = full_name.lower(), description.lower()
    for token in significant_tokens(lower_query):
        if token in full_name_l:
            score += 6
        if token in desc_l:
            score += 3
    if stars >= 10000:
        score += 12
    elif stars >= 3000:
        score += 8
    elif stars >= 500:
        score += 4
    return score


def build_github_fallback_queries(title: str) -> List[str]:
    title = clean_paper_title(title)
    if not title:
        return []
    queries = [title, f"{title} implementation"]
    return queries[:4]


# ---------------- 网络调用（尽力而为） ----------------

def _http_json(url: str, timeout: float, headers: Optional[Dict] = None):
    """GET 并解析 JSON；返回 (data, error)。异常不抛出。"""
    req_headers = {"Accept": "application/json",
                   "User-Agent": "autoreproducer/1.0"}
    req_headers.update(headers or {})
    try:
        request = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace")), ""
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _search_pwc(query: str, limit: int = 5,
                timeout: float = 8.0) -> tuple:
    """Papers with Code（HuggingFace Papers API）搜索，返回 (候选, 错误)。"""
    base = os.environ.get(
        "PWC_API_BASE_URL", "https://huggingface.co").strip().rstrip("/")
    data, err = _http_json(
        f"{base}/api/papers/search"
        f"?q={urllib.parse.quote(query)}", timeout)
    if err or data is None:
        return [], err or "search failed"
    items = [] if not isinstance(data, list) else data[:limit]
    candidates: List[RepoCandidate] = []
    for item in items:
        paper = (item or {}).get("paper", {}) or {}
        paper_id = str(paper.get("id", "")).strip()
        title = str(paper.get("title", "")).strip()
        if not paper_id:
            continue
        info, _err = _http_json(
            f"{base}/api/papers/{urllib.parse.quote(paper_id)}", timeout)
        repos: List[str] = []
        if isinstance(info, dict):
            repos = extract_plain_github_urls(json.dumps(info))
        candidates.append(RepoCandidate(
            paper_id=paper_id,
            title=title,
            repo_urls=repos[:3],
            source="papers_with_code",
            score_hint=3 if repos else 0,
        ))
    return candidates, ""


def _search_github(query: str, limit: int = 5,
                   timeout: float = 8.0) -> tuple:
    """GitHub 搜索 API，返回 (候选, 错误)。"""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data, err = _http_json(
        "https://api.github.com/search/repositories"
        f"?q={urllib.parse.quote(query)}&sort=stars&order=desc"
        f"&per_page={limit}",
        timeout, headers)
    if err or data is None:
        return [], err or "search failed"
    repos = data.get("items", []) if isinstance(data, dict) else []
    candidates: List[RepoCandidate] = []
    for repo in repos[:limit]:
        full_name = str(repo.get("full_name", "") or "")
        url = f"https://github.com/{full_name}" if full_name else ""
        if not url:
            continue
        candidates.append(RepoCandidate(
            title=str(repo.get("name", "") or ""),
            repo_name=full_name,
            description=str(repo.get("description", "") or ""),
            repo_urls=[url],
            source="github_search",
            stars=int(repo.get("stargazers_count") or 0),
            score_hint=github_repo_search_score(
                query, full_name,
                str(repo.get("description", "") or ""),
                int(repo.get("stargazers_count") or 0)),
        ))
    return candidates, ""


# ---------------- 主发现函数 ----------------

def discover_repositories(query: str,
                          preferred_url: str = "",
                          max_results: int = 8,
                          timeout: float = 8.0,
                          offline: bool = False) -> Dict:
    """确定性仓库发现链主入口（无 LLM）。返回结构化结果。

    链：用户URL（最高优先，直接采用） -> PwC -> GitHub 搜索 ->
    curated 回退。前一层有可用候选（带 URL）即停。严格信任筛选：
    非 arxiv id 查询时，只有可信匹配（词面重合）的候选才能被选为主仓库。
    offline=True 时跳过 PwC/GitHub 网络层（模拟模式/断网演示），
    仅用户 URL 与 curated 回退可用。
    """
    chain: List[str] = []
    candidates: List[RepoCandidate] = []
    selected = ""
    fallback_used = False
    pwc_error = gh_error = ""

    # 1. 用户 URL 最高优先
    pref = normalize_github_repo_url(preferred_url)
    if pref:
        chain.append("user_preference")
        candidates.append(RepoCandidate(
            title="User-preferred repository",
            repo_name=pref.rstrip("/").split("github.com/")[-1],
            description="Repository explicitly requested by the user.",
            repo_urls=[pref],
            source="user_preference",
            score_hint=1_000_000,
        ))
        selected = pref
        return _discovery_result(
            query, candidates, selected, chain, fallback_used, "")

    if not query.strip():
        return _discovery_result(query, [], "", chain, False,
                                 "empty query")

    chain.append("papers_with_code")
    if offline:
        pwc_error = "offline: papers_with_code skipped"
    else:
        # 2. PwC（HF Papers API）
        pwc_candidates, pwc_error = _search_pwc(query, timeout=timeout)
        candidates.extend(pwc_candidates)

    # 3. GitHub 搜索回退（PwC 无有效 URL 时）
    if not any(c.repo_urls for c in candidates):
        chain.append("github_search")
        if offline:
            gh_error = "offline: github_search skipped"
        else:
            gh_candidates, gh_error = _search_github(
                build_github_fallback_queries(query)[0] or query,
                timeout=timeout)
            if gh_candidates:
                fallback_used = True
                candidates.extend(gh_candidates)

    # 4. curated 回退（前两层全无有效候选时）
    if not any(c.repo_urls for c in candidates):
        chain.append("curated_fallback")
        curated = curated_repo_fallback_candidates(query)
        if curated:
            fallback_used = True
            candidates.extend(curated)

    # 排序与选择：非 arxiv id 查询用可信匹配筛选，否则取分数最高
    candidates.sort(key=lambda c: c.score_hint, reverse=True)
    strict_trusted = ARXIV_ID_RE.search(query) is None
    if strict_trusted:
        selected = next(
            (c.repo_urls[0] for c in candidates
             if c.repo_urls and is_trusted_repo_candidate(query, c)),
            "")
    else:
        selected = next((c.repo_urls[0] for c in candidates
                         if c.repo_urls), "")

    errors = " | ".join(e for e in (pwc_error, gh_error) if e)
    return _discovery_result(query, candidates, selected, chain,
                             fallback_used, errors)


def _discovery_result(query: str, candidates: List[RepoCandidate],
                      selected: str, chain: List[str],
                      fallback_used: bool, errors: str) -> Dict:
    return {
        "query": query,
        "discovery_chain": chain,
        "candidates": [c.to_dict() for c in candidates][:8],
        "selected_repo": selected,
        "fallback_used": fallback_used,
        "error": errors,
    }


# ---------------- 复现模式决策 ----------------

def decide_reproduction_mode(requested: str = "auto",
                             full_requested: bool = False,
                             cpu: Optional[int] = None,
                             memory_gb: float = 0.0) -> Dict:
    """auto/smoke/full 复现模式决策。

    - auto：默认 smoke（有界验证，控制资源）；显式 full 且确认时升级 full；
    - smoke/full：遵循请求，但 full 未显式确认时降级 smoke（防无预案全量跑）；
    - 返回含 effective_mode 与 reason，供编排与报告使用。
    """
    effective = requested
    reason = "explicit"
    if requested == "auto":
        effective = "smoke"
        reason = "auto-selected smoke mode for bounded verification"
    elif requested == "full" and not full_requested:
        effective = "smoke"
        reason = "full requires explicit confirmation, degraded to smoke"
    if full_requested or (requested == "full" and full_requested):
        effective = "full"
        reason = "full reproduction explicitly confirmed"
    return {
        "requested_mode": requested,
        "effective_mode": effective,
        "reason": reason,
        "cpu_count": cpu if cpu is not None else (os.cpu_count() or 1),
        "memory_gb": memory_gb,
    }


def probe_memory_gb() -> float:
    """探测可用内存（GB）。psutil 优先，/proc/meminfo 兜底，失败 0.0。"""
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except OSError:
        pass
    return 0.0