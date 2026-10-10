"""Authentic public inputs for the fixed DLinear protocol reader.

The packet is deliberately distinct from execution evidence: it contains only
published excerpts, public locators and a fixed author commit. Its local cache
manifest records the hashes used to construct it, without entering the prompt.
No repository file outside PUBLIC_REPOSITORY_FILES is opened.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.repository_profiles import PAPER_TITLE, REPO_SHA, REPO_URL

PAPER_URL = "https://arxiv.org/html/2205.13504v3"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_COMMENT_BYTES = 64 * 1024
MAX_REPOSITORY_FILE_BYTES = 256 * 1024
MAX_PACKET_CHARACTERS = 120_000
MAX_DOWNLOAD_ATTEMPTS = 3
MAX_RETRY_DELAY_S = 5

PUBLIC_REPOSITORY_FILES = {
    "README.md": "repo_readme",
    "requirements.txt": "repo_requirements",
    "models/DLinear.py": "repo_model",
    "run_longExp.py": "repo_entrypoint",
    "data_provider/data_loader.py": "repo_data_split",
    "exp/exp_main.py": "repo_training",
    "utils/metrics.py": "repo_metrics",
    "scripts/EXP-LongForecasting/Linear/etth1.sh": "repo_author_command",
}
PAPER_SECTIONS = {
    "abstract1": "paper_abstract",
    "S4": "paper_method",
    "S5.SS1": "paper_experiment_settings",
    "S5.SS2": "paper_comparison",
    "S5.T2": "paper_table2",
    "A2.SS2": "paper_implementation_b2",
}
AUTHOR_COMMENTS = {
    1331937601: {"source_id": "author_single_seed", "issue": 33},
    1398345611: {"source_id": "author_initialization", "issue": 39},
}
for _comment_id, _spec in AUTHOR_COMMENTS.items():
    _spec["id"] = _comment_id
    _spec["api_url"] = f"https://api.github.com/repos/cure-lab/LTSF-Linear/issues/comments/{_comment_id}"
    _spec["html_url"] = f"{REPO_URL}/issues/{_spec['issue']}#issuecomment-{_comment_id}"
    _spec["web_url"] = f"{REPO_URL}/issues/{_spec['issue']}"
    _spec["issue_url"] = f"https://api.github.com/repos/cure-lab/LTSF-Linear/issues/{_spec['issue']}"
PUBLIC_DOWNLOAD_URLS = {PAPER_URL, *(spec["api_url"] for spec in AUTHOR_COMMENTS.values()),
                        *(spec["web_url"] for spec in AUTHOR_COMMENTS.values())}
GITHUB_API_URLS = {spec["api_url"] for spec in AUTHOR_COMMENTS.values()}


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    _write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.part"
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_limited(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("公开来源超过字节上限")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("公开来源超过字节上限")
    return content


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError("公开来源下载不允许重定向")


class PublicSourceDownloadError(RuntimeError):
    """A credential-free, bounded-download diagnostic usable by recovery."""

    def __init__(self, url: str, *, status: int | None = None,
                 attempts: int = 1, rate_limited: bool = False,
                 retry_after_s: float | None = None, rate_limit: str | None = None,
                 rate_remaining: str | None = None, rate_reset_utc: str | None = None,
                 retryable: bool = False):
        self.url = url
        self.status = status
        self.attempts = attempts
        self.rate_limited = rate_limited
        self.retry_after_s = retry_after_s
        self.rate_limit = rate_limit
        self.rate_remaining = rate_remaining
        self.rate_reset_utc = rate_reset_utc
        self.retryable = retryable
        detail = f"HTTP {status}" if status is not None else "网络连接或超时错误"
        if rate_limited:
            detail += "，GitHub/API 请求限额已触发"
        if rate_remaining is not None and rate_limit is not None:
            detail += f"（剩余 {rate_remaining}/{rate_limit}）"
        if rate_reset_utc is not None:
            detail += f"，额度重置于 {rate_reset_utc}"
        if retry_after_s is not None:
            detail += f"，服务端建议 {retry_after_s:g} 秒后重试"
        super().__init__(f"公开来源下载失败: {url}: {detail}；已尝试 {attempts} 次")

    def as_dict(self) -> dict:
        return {key: value for key, value in {
            "url": self.url, "status": self.status, "attempts": self.attempts,
            "rate_limited": self.rate_limited, "retry_after_s": self.retry_after_s,
            "rate_limit": self.rate_limit, "rate_remaining": self.rate_remaining,
            "rate_reset_utc": self.rate_reset_utc,
        }.items() if value is not None}


def _header_number(headers, name: str) -> str | None:
    value = headers.get(name) if headers is not None else None
    return value if isinstance(value, str) and re.fullmatch(r"\d{1,12}", value) else None


def _download_error(url: str, error, attempts: int) -> PublicSourceDownloadError:
    if not isinstance(error, urllib.error.HTTPError):
        return PublicSourceDownloadError(url, attempts=attempts, retryable=True)
    headers = error.headers
    limit = _header_number(headers, "X-RateLimit-Limit")
    remaining = _header_number(headers, "X-RateLimit-Remaining")
    reset = _header_number(headers, "X-RateLimit-Reset")
    reset_utc = None
    if reset is not None:
        try:
            reset_utc = datetime.fromtimestamp(int(reset), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (OverflowError, OSError, ValueError):
            pass
    retry_after = headers.get("Retry-After") if headers is not None else None
    retry_after_s = None
    if isinstance(retry_after, str):
        try:
            retry_after_s = float(retry_after)
        except ValueError:
            try:
                retry_date = parsedate_to_datetime(retry_after)
                retry_after_s = max(0, retry_date.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
        if retry_after_s is not None and (not math.isfinite(retry_after_s) or retry_after_s < 0):
            retry_after_s = None
    rate_limited = error.code == 429 or (error.code == 403 and (remaining == "0" or retry_after_s is not None))
    return PublicSourceDownloadError(
        url, status=error.code, attempts=attempts, rate_limited=rate_limited,
        retry_after_s=retry_after_s, rate_limit=limit, rate_remaining=remaining,
        rate_reset_utc=reset_utc,
        retryable=error.code in {408, 429, 500, 502, 503, 504})


def _download_once(url: str, *, max_bytes: int, timeout_s: float,
                   transport: Callable | None, github_token: str | None) -> bytes:
    if transport is not None:
        return transport(url, max_bytes=max_bytes, timeout_s=timeout_s)
    headers = {"User-Agent": "AutoReproducer/1.0"}
    if url in GITHUB_API_URLS:
        headers["Accept"] = "application/vnd.github+json"
        # Authentication is sent only to the exact approved API URLs, never
        # paper/web hosts, redirects, injected transports, logs or manifests.
        token = github_token
        if token is None:
            token = next((os.environ.get(name) for name in
                          ("AUTOREPRO_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
                          if os.environ.get(name)), None)
        if isinstance(token, str) and token.strip() and "\r" not in token and "\n" not in token:
            headers["Authorization"] = f"Bearer {token.strip()}"
    else:
        headers["Accept"] = "text/html"
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout_s) as response:
        if response.geturl() != url:
            raise ValueError("公开来源响应URL不在白名单")
        if response.status != 200:
            raise urllib.error.HTTPError(url, response.status, "unexpected status", response.headers, None)
        return response.read(max_bytes + 1)


def download_public_document(url: str, *, offline: bool = False,
                             max_bytes: int = MAX_DOCUMENT_BYTES,
                             timeout_s: float = 30,
                             transport: Callable | None = None,
                             max_attempts: int = MAX_DOWNLOAD_ATTEMPTS,
                             retry_delay_s: float = 1,
                             sleep: Callable = time.sleep,
                             github_token: str | None = None) -> bytes:
    """Fetch an approved public paper/comment; an injected transport is testable.

    ``transport(url, max_bytes=..., timeout_s=...)`` returns bytes. Both real
    and injected transports are subject to the same URL, offline and size rules.
    The urllib implementation refuses redirects before another host is opened.
    Transient failures receive at most three attempts; a long Retry-After or
    exhausted GitHub quota returns immediately so an approved web source can
    recover without waiting for the hourly API reset.
    """
    if url not in PUBLIC_DOWNLOAD_URLS:
        raise ValueError("公开来源URL不在白名单")
    if offline:
        raise RuntimeError("离线模式禁止下载公开来源")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("公开来源字节上限必须为正整数")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > 120:
        raise ValueError("公开来源下载超时必须在0至120秒内")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_DOWNLOAD_ATTEMPTS:
        raise ValueError("公开来源下载最多允许1至3次尝试")
    if not isinstance(retry_delay_s, (int, float)) or not math.isfinite(retry_delay_s) or not 0 <= retry_delay_s <= MAX_RETRY_DELAY_S:
        raise ValueError("公开来源重试间隔必须在0至5秒内")
    for attempt in range(1, max_attempts + 1):
        try:
            content = _download_once(url, max_bytes=max_bytes, timeout_s=timeout_s,
                                     transport=transport, github_token=github_token)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError) as error:
            diagnostic = _download_error(url, error, attempt)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            # Do not expose HTTPError.reason, response bodies, or nested URL
            # errors: these can contain an Authorization header or proxy key.
            if (not diagnostic.retryable or attempt == max_attempts
                    or (diagnostic.retry_after_s is not None
                        and diagnostic.retry_after_s > MAX_RETRY_DELAY_S)):
                raise diagnostic from None
            delay = min(MAX_RETRY_DELAY_S, retry_delay_s * (2 ** (attempt - 1)))
            if diagnostic.retry_after_s is not None:
                delay = max(delay, diagnostic.retry_after_s)
            sleep(delay)
    if not isinstance(content, bytes):
        raise TypeError("公开来源transport必须返回bytes")
    if len(content) > max_bytes:
        raise ValueError("公开来源超过字节上限")
    return content


class _Node:
    def __init__(self, tag: str, attrs=None, line: int = 1):
        self.tag = tag
        self.attrs = dict(attrs or ())
        self.children: list[_Node | str] = []
        self.line = line
        self.end_line = line


class _PaperParser(HTMLParser):
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self.stack = [self.root]
        self.ids: dict[str, _Node] = {}

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs, self.getpos()[0])
        self.stack[-1].children.append(node)
        identifier = node.attrs.get("id")
        if identifier:
            if identifier in self.ids:
                raise ValueError("论文HTML包含重复节定位")
            self.ids[identifier] = node
        if tag not in self._VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                for node in self.stack[index:]:
                    node.end_line = self.getpos()[0]
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _descendants(node: _Node, tag: str):
    for child in node.children:
        if isinstance(child, _Node):
            if child.tag == tag:
                yield child
            yield from _descendants(child, tag)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _table_text(node: _Node) -> str:
    # Expanding row/column spans keeps every dataset/horizon and method/metric
    # in the same column in Table 2. No metric value is supplied by this module.
    grid = {}
    rows = list(_descendants(node, "tr"))
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row.children:
            if not isinstance(cell, _Node) or cell.tag not in {"td", "th"}:
                continue
            while (row_index, column) in grid:
                column += 1
            try:
                colspan = int(cell.attrs.get("colspan", "1"))
                rowspan = int(cell.attrs.get("rowspan", "1"))
            except ValueError as exc:
                raise ValueError("论文表格跨度无效") from exc
            if not 1 <= colspan <= 64 or not 1 <= rowspan <= 64:
                raise ValueError("论文表格跨度超出上限")
            text = _normalize(_render(cell))
            for relative_row in range(rowspan):
                for relative_column in range(colspan):
                    grid[row_index + relative_row, column + relative_column] = text
            column += colspan
    width = max((column + 1 for _, column in grid), default=0)
    return "\n".join(" | ".join(grid.get((row, column), "") for column in range(width))
                     for row in range(len(rows)))


def _render(node: _Node | str) -> str:
    if isinstance(node, str):
        return node
    if node.tag in {"script", "style", "head", "nav", "footer", "annotation"}:
        return ""
    if node.tag == "math" and node.attrs.get("alttext"):
        return node.attrs["alttext"]
    if node.tag == "table":
        return "\n" + _table_text(node) + "\n"
    text = "".join(_render(child) for child in node.children)
    if node.tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "h5",
                    "h6", "li", "figcaption", "figure", "br"}:
        return "\n" + text + "\n"
    return text


def extract_paper_sections(content: bytes) -> tuple[list[dict], list[dict]]:
    """Extract real text by arXiv anchor; record HTML lines and excerpt hashes."""
    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("论文HTML必须为UTF-8") from exc
    parser = _PaperParser()
    parser.feed(html)
    parser.close()
    titles = [_normalize(_render(node)) for node in _descendants(parser.root, "h1")]
    if PAPER_TITLE not in titles or not list(_descendants(parser.root, "article")):
        raise ValueError("缓存内容不是指定论文正文HTML")
    if not re.search(r"\barXiv:\s*2205\.13504v3\b", _render(parser.root)):
        raise ValueError("论文HTML缺少指定arXiv版本标记2205.13504v3")
    sources, provenance = [], []
    for anchor, source_id in PAPER_SECTIONS.items():
        node = parser.ids.get(anchor)
        if node is None:
            raise ValueError(f"论文HTML缺少真实节定位: {anchor}")
        text = "\n".join(line for line in (_normalize(line) for line in _render(node).splitlines())
                         if line)
        if not text:
            raise ValueError(f"论文定位没有正文: {anchor}")
        source = {"source_id": source_id, "url": f"{PAPER_URL}#{anchor}",
                  "locator": f"#{anchor}", "text": text}
        sources.append(source)
        provenance.append({"source_id": source_id, "url": source["url"],
                           "locator": source["locator"],
                           "html_lines": [node.line, node.end_line],
                           "text_sha256": _digest(text.encode("utf-8")),
                           "characters": len(text)})
    return sources, provenance


def extract_author_comment(content: bytes, comment_id: int) -> tuple[dict, dict]:
    """Read the complete published body, after repository/comment attribution.

    GitHub's public response must match all three repository locators and mark
    the comment author as a repository member/owner. Images linked in a body
    are retained as published markup; this module never fetches their URLs.
    """
    spec = AUTHOR_COMMENTS.get(comment_id)
    if spec is None:
        raise ValueError("作者评论不在白名单")
    try:
        comment = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("作者评论必须为有效公开UTF-8 JSON") from exc
    if (not isinstance(comment, dict) or type(comment.get("id")) is not int
            or comment.get("id") != comment_id or comment.get("url") != spec["api_url"]
            or comment.get("html_url") != spec["html_url"]
            or comment.get("issue_url") != spec["issue_url"]):
        raise ValueError("作者评论来源与固定公开locator不一致")
    user = comment.get("user")
    if (not isinstance(user, dict) or not isinstance(user.get("login"), str)
            or not user["login"].strip()
            or comment.get("author_association") not in {"OWNER", "MEMBER"}):
        raise ValueError("公开评论缺少作者仓库成员身份")
    body = comment.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("公开作者评论缺少原文body")
    locator = f"#issuecomment-{comment_id}"
    source = {"source_id": spec["source_id"], "url": spec["html_url"],
              "locator": locator, "text": body}
    provenance = {"source_id": spec["source_id"], "url": spec["api_url"],
                  "html_url": spec["html_url"], "locator": locator,
                  "id": comment_id, "sha256": _digest(content), "bytes": len(content),
                  "text_sha256": _digest(body.encode("utf-8")), "characters": len(body),
                  "author": user["login"], "author_association": comment["author_association"]}
    return source, provenance


class _GitHubEmbeddedDataParser(HTMLParser):
    """Read inert public JSON only; never interpret scripts or rendered text."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.documents: list[str] = []
        self.parts: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if (tag == "script" and attributes.get("type") == "application/json"
                and attributes.get("data-target") == "react-app.embeddedData"):
            if self.parts is not None or len(self.documents) >= 4:
                raise ValueError("公开GitHub网页内嵌数据结构无效")
            self.parts = []

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.parts is not None:
            self.documents.append("".join(self.parts))
            self.parts = None


def extract_author_comment_html(content: bytes, comment_id: int) -> bytes:
    """Extract the exact Markdown from GitHub's official public issue page.

    A normalized comment document remains compatible with the REST cache, but
    is explicitly attributed to the separately retained/hash-checked raw HTML.
    Repository identity, issue number/URL, exact comment URL/id, public status
    and member association all come from the published embedded data. BodyHTML
    or adjacent discussion is never substituted for a missing original body.
    """
    spec = AUTHOR_COMMENTS.get(comment_id)
    if spec is None:
        raise ValueError("作者评论不在白名单")
    parser = _GitHubEmbeddedDataParser()
    try:
        parser.feed(content.decode("utf-8"))
        parser.close()
    except UnicodeDecodeError as error:
        raise ValueError("公开GitHub网页必须为UTF-8") from error
    matches = []
    for document in parser.documents:
        try:
            data = json.loads(document)
            issue = data["payload"]["issueViewerRoute"]["data"]["repository"]["issue"]
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(issue, dict):
            raise ValueError("公开GitHub网页issue数据结构无效")
        repository = issue.get("repository") if isinstance(issue, dict) else None
        if (type(issue.get("number")) is not int or issue["number"] != spec["issue"]
                or issue.get("url") != spec["web_url"]
                or not isinstance(repository, dict)
                or repository.get("nameWithOwner") != "cure-lab/LTSF-Linear"
                or repository.get("isPrivate") is not False):
            raise ValueError("公开GitHub网页来源与固定仓库/issue locator不一致")
        for timeline_name in ("timelineItems", "backTimelineItems"):
            timeline = issue.get(timeline_name)
            edges = timeline.get("edges") if isinstance(timeline, dict) else None
            if not isinstance(edges, list):
                continue
            for edge in edges:
                node = edge.get("node") if isinstance(edge, dict) else None
                if not isinstance(node, dict) or node.get("databaseId") != comment_id:
                    continue
                comment_issue = node.get("issue")
                comment_repository = node.get("repository")
                author = node.get("author")
                if (node.get("__typename") != "IssueComment"
                        or type(node.get("databaseId")) is not int
                        or node.get("url") != spec["html_url"]
                        or not isinstance(comment_issue, dict)
                        or type(comment_issue.get("number")) is not int
                        or comment_issue["number"] != spec["issue"]
                        or not isinstance(comment_repository, dict)
                        or comment_repository.get("nameWithOwner") != "cure-lab/LTSF-Linear"
                        or comment_repository.get("isPrivate") is not False
                        or not isinstance(author, dict)):
                    raise ValueError("公开GitHub网页评论与固定作者locator不一致")
                comment = {"id": comment_id, "url": spec["api_url"],
                           "html_url": node["url"], "issue_url": spec["issue_url"],
                           "user": {"login": author.get("login")},
                           "author_association": node.get("authorAssociation"),
                           "body": node.get("body")}
                normalized = json.dumps(comment, ensure_ascii=False, sort_keys=True).encode("utf-8")
                extract_author_comment(normalized, comment_id)
                matches.append(normalized)
    if not matches:
        raise ValueError("公开GitHub网页缺少指定作者评论的原始正文/署名")
    if any(document != matches[0] for document in matches[1:]):
        raise ValueError("公开GitHub网页包含不一致的重复作者评论")
    return matches[0]


class RepositoryPublicSources:
    def __init__(self, cache_dir: str | Path, *, transport: Callable | None = None,
                 max_document_bytes: int = MAX_DOCUMENT_BYTES,
                 max_comment_bytes: int = MAX_COMMENT_BYTES,
                 max_repository_file_bytes: int = MAX_REPOSITORY_FILE_BYTES,
                 max_packet_characters: int = MAX_PACKET_CHARACTERS):
        for value in (max_document_bytes, max_comment_bytes, max_repository_file_bytes, max_packet_characters):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("公开来源大小上限必须为正整数")
        self.cache_dir = Path(cache_dir)
        self.transport = transport
        self.max_document_bytes = max_document_bytes
        self.max_comment_bytes = max_comment_bytes
        self.max_repository_file_bytes = max_repository_file_bytes
        self.max_packet_characters = max_packet_characters
        self.manifest_path = self.cache_dir / "packet_manifest.json"

    def _comment_html_provenance(self, content: bytes, comment_id: int) -> dict:
        return {"source_format": "github_issue_html",
                "retrieval_url": AUTHOR_COMMENTS[comment_id]["web_url"],
                "source_sha256": _digest(content), "source_bytes": len(content)}

    def _cached_html_comment(self, metadata: dict, comment_id: int,
                             normalized: bytes) -> dict:
        raw_path = self.cache_dir / f"issuecomment-{comment_id}.source.html"
        if not raw_path.is_file():
            raise ValueError("公开作者评论网页缓存缺少原始来源HTML")
        raw = _read_limited(raw_path, self.max_document_bytes)
        expected = self._comment_html_provenance(raw, comment_id)
        if (metadata.get("origin") != "github_issue_html"
                or any(metadata.get(key) != value for key, value in expected.items())
                or extract_author_comment_html(raw, comment_id) != normalized):
            raise ValueError("公开作者评论网页缓存来源或哈希校验失败")
        if "recovery" in metadata:
            expected["recovery"] = metadata["recovery"]
        return expected

    def _paper(self, *, offline: bool, paper_path: str | Path | None) -> tuple[bytes, dict]:
        cache_path = self.cache_dir / "paper.html"
        metadata_path = self.cache_dir / "paper_manifest.json"
        if paper_path is not None:
            content = _read_limited(Path(paper_path), self.max_document_bytes)
            origin = "verified_local_public_document"
        elif cache_path.is_file():
            if not metadata_path.is_file():
                raise ValueError("公开论文缓存缺少来源manifest")
            metadata = json.loads(_read_limited(metadata_path, 32_768))
            content = _read_limited(cache_path, self.max_document_bytes)
            if (metadata.get("url") != PAPER_URL or metadata.get("sha256") != _digest(content)
                    or metadata.get("bytes") != len(content)):
                raise ValueError("公开论文缓存来源或哈希校验失败")
            extract_paper_sections(content)
            return content, metadata
        else:
            content = download_public_document(PAPER_URL, offline=offline,
                                               max_bytes=self.max_document_bytes,
                                               transport=self.transport)
            origin = "https_download"
        # Identity and required sections are validated before accepting a seed
        # or publishing a download into the cache.
        extract_paper_sections(content)
        metadata = {"url": PAPER_URL, "sha256": _digest(content), "bytes": len(content),
                    "origin": origin, "anchors": list(PAPER_SECTIONS)}
        _write_bytes(cache_path, content)
        _write_json(metadata_path, metadata)
        return content, metadata

    def cache_author_comments(self, *, offline: bool = True,
                              comment_paths: Mapping | None = None) -> tuple[list[dict], list[dict]]:
        """Prepare both public comment sources, accepting explicit JSON seeds.

        ``comment_paths`` maps approved integer comment IDs to local public JSON
        copies. An offline call requires a verified cache or such a seed for
        each comment. Neither missing nor corrupt sources become short profile
        paraphrases. Every cached body is revalidated against its raw JSON hash.
        """
        paths = dict(comment_paths or {})
        if any(type(key) is not int or key not in AUTHOR_COMMENTS for key in paths):
            raise ValueError("作者评论seed ID不在白名单")
        sources, manifests = [], []
        for comment_id, spec in AUTHOR_COMMENTS.items():
            cache_path = self.cache_dir / f"issuecomment-{comment_id}.json"
            metadata_path = self.cache_dir / f"issuecomment-{comment_id}.manifest.json"
            raw_html = None
            extra_provenance = {}
            if comment_id in paths:
                content = _read_limited(Path(paths[comment_id]), self.max_comment_bytes)
                origin = "verified_local_public_comment"
                publish = True
            elif cache_path.is_file():
                if not metadata_path.is_file():
                    raise ValueError("公开作者评论缓存缺少来源manifest")
                metadata = json.loads(_read_limited(metadata_path, 32_768))
                content = _read_limited(cache_path, self.max_comment_bytes)
                if (not isinstance(metadata, dict) or metadata.get("url") != spec["api_url"]
                        or metadata.get("sha256") != _digest(content)
                        or metadata.get("bytes") != len(content)):
                    raise ValueError("公开作者评论缓存来源或哈希校验失败")
                origin = metadata.get("origin", "verified_public_cache")
                if (origin == "github_issue_html"
                        or metadata.get("source_format") == "github_issue_html"):
                    extra_provenance = self._cached_html_comment(metadata, comment_id, content)
                publish = False
            else:
                try:
                    content = download_public_document(spec["api_url"], offline=offline,
                                                       max_bytes=self.max_comment_bytes,
                                                       transport=self.transport)
                    origin = "https_download"
                except PublicSourceDownloadError as api_error:
                    # This is the same authentic comment published on the
                    # author's official GitHub issue, not a missing-evidence
                    # bypass. Integrity/size errors never invoke this fallback.
                    try:
                        raw_html = download_public_document(
                            spec["web_url"], offline=offline,
                            max_bytes=self.max_document_bytes, transport=self.transport)
                    except PublicSourceDownloadError as web_error:
                        raise RuntimeError(f"作者评论自动恢复失败；REST: {api_error}；官方网页: {web_error}") from None
                    content = extract_author_comment_html(raw_html, comment_id)
                    if len(content) > self.max_comment_bytes:
                        raise ValueError("公开作者评论超过字节上限")
                    origin = "github_issue_html"
                    extra_provenance = self._comment_html_provenance(raw_html, comment_id)
                    extra_provenance["recovery"] = api_error.as_dict()
                publish = True
            source, provenance = extract_author_comment(content, comment_id)
            provenance["origin"] = origin
            provenance.update(extra_provenance)
            if publish:
                if raw_html is not None:
                    _write_bytes(self.cache_dir / f"issuecomment-{comment_id}.source.html", raw_html)
                _write_bytes(cache_path, content)
                _write_json(metadata_path, provenance)
            sources.append(source)
            manifests.append(provenance)
        return sources, manifests

    def build_packet(self, workspace: str | Path, snapshot: Mapping,
                     profile: Mapping | None = None, *, offline: bool = True,
                     paper_path: str | Path | None = None,
                     comment_paths: Mapping | None = None) -> dict:
        """Build the API-safe packet and write local ``packet_manifest.json``.

        ``snapshot`` is the provenance returned by ``export_repository``:
        its public URL, revision/resolved_sha and per-file hashes are checked.
        Optional profile data is used only to reject a mismatched repository;
        its local parameters, metrics, paths and environment never enter packet.
        ``paper_path`` explicitly seeds the cache with a validated public HTML
        copy, including in offline mode. Offline never invokes any transport.
        """
        repository = {"url": REPO_URL, "revision": REPO_SHA}
        if (snapshot.get("url") != REPO_URL or snapshot.get("revision") != REPO_SHA
                or snapshot.get("resolved_sha", snapshot.get("revision")) != REPO_SHA):
            raise ValueError("公开源码必须来自固定作者仓库SHA")
        if profile is not None and profile.get("repository") != repository:
            raise ValueError("论文预设与公开源码固定版本不一致")
        files = snapshot.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("公开源码缺少snapshot.files哈希清单")
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("公开源码workspace必须为目录")
        sources, file_provenance = [], []
        for relative, source_id in PUBLIC_REPOSITORY_FILES.items():
            expected = files.get(relative)
            if isinstance(expected, Mapping):
                expected = expected.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError(f"公开源码缺少有效snapshot哈希: {relative}")
            path = root / relative
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"公开源码路径越界: {relative}")
            # Exported author archives contain regular files, never symlinks.
            if any(part.is_symlink() for part in [path, *path.parents] if part != root):
                raise ValueError(f"公开源码不允许symlink: {relative}")
            content = _read_limited(resolved, self.max_repository_file_bytes)
            if _digest(content) != expected:
                raise ValueError(f"公开源码snapshot哈希不符: {relative}")
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"公开源码必须为UTF-8: {relative}") from exc
            if not text.strip():
                raise ValueError(f"公开源码内容为空: {relative}")
            line_count = len(text.splitlines())
            locator = f"{relative}#L1-L{line_count}"
            url = f"{REPO_URL}/blob/{REPO_SHA}/{relative}#L1-L{line_count}"
            sources.append({"source_id": source_id, "url": url,
                            "locator": locator, "text": text})
            file_provenance.append({"source_id": source_id, "url": url,
                                    "locator": locator, "sha256": expected,
                                    "bytes": len(content)})
        paper, paper_metadata = self._paper(offline=offline, paper_path=paper_path)
        paper_sources, section_provenance = extract_paper_sections(paper)
        comment_sources, comment_provenance = self.cache_author_comments(
            offline=offline, comment_paths=comment_paths)
        sources = paper_sources + sources + comment_sources
        if sum(len(source["text"]) for source in sources) > self.max_packet_characters:
            raise ValueError("公开来源packet超过字符上限；不会截断证据")
        packet = {"version": 1, "repository": repository, "sources": sources}
        packet_hash = _digest(json.dumps(packet, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8"))
        _write_json(self.manifest_path, {"version": 1, "paper": paper_metadata,
                                       "repository": repository,
                                       "paper_sections": section_provenance,
                                       "repository_files": file_provenance,
                                       "author_comments": comment_provenance,
                                       "packet_sha256": packet_hash})
        return packet


def build_public_source_packet(workspace: str | Path, snapshot: Mapping,
                               cache_dir: str | Path, profile: Mapping | None = None,
                               *, offline: bool = True,
                               paper_path: str | Path | None = None,
                               comment_paths: Mapping | None = None,
                               transport: Callable | None = None) -> dict:
    """Convenience wrapper; the adjacent manifest stays local to ``cache_dir``."""
    return RepositoryPublicSources(cache_dir, transport=transport).build_packet(
        workspace, snapshot, profile, offline=offline, paper_path=paper_path,
        comment_paths=comment_paths)
