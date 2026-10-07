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
from pathlib import Path
import re
from typing import Callable, Mapping
import urllib.request
import uuid

from src.repository_profiles import PAPER_TITLE, REPO_SHA, REPO_URL

PAPER_URL = "https://arxiv.org/html/2205.13504v3"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_COMMENT_BYTES = 64 * 1024
MAX_REPOSITORY_FILE_BYTES = 256 * 1024
MAX_PACKET_CHARACTERS = 120_000

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
    _spec["issue_url"] = f"https://api.github.com/repos/cure-lab/LTSF-Linear/issues/{_spec['issue']}"
PUBLIC_DOWNLOAD_URLS = {PAPER_URL, *(spec["api_url"] for spec in AUTHOR_COMMENTS.values())}


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


def download_public_document(url: str, *, offline: bool = False,
                             max_bytes: int = MAX_DOCUMENT_BYTES,
                             timeout_s: float = 30,
                             transport: Callable | None = None) -> bytes:
    """Fetch an approved public paper/comment; an injected transport is testable.

    ``transport(url, max_bytes=..., timeout_s=...)`` returns bytes. Both real
    and injected transports are subject to the same URL, offline and size rules.
    The urllib implementation refuses redirects before another host is opened.
    """
    if url not in PUBLIC_DOWNLOAD_URLS:
        raise ValueError("公开来源URL不在白名单")
    if offline:
        raise RuntimeError("离线模式禁止下载公开来源")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("公开来源字节上限必须为正整数")
    if timeout_s <= 0 or timeout_s > 120:
        raise ValueError("公开来源下载超时必须在0至120秒内")
    if transport is not None:
        content = transport(url, max_bytes=max_bytes, timeout_s=timeout_s)
    else:
        headers = {"User-Agent": "AutoReproducer/1.0"}
        if url != PAPER_URL:
            headers["Accept"] = "application/vnd.github+json"
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout_s) as response:
            if response.geturl() != url:
                raise ValueError("公开来源响应URL不在白名单")
            if response.status != 200:
                raise RuntimeError("公开来源下载响应失败")
            content = response.read(max_bytes + 1)
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
                publish = False
            else:
                content = download_public_document(spec["api_url"], offline=offline,
                                                   max_bytes=self.max_comment_bytes,
                                                   transport=self.transport)
                origin = "https_download"
                publish = True
            source, provenance = extract_author_comment(content, comment_id)
            provenance["origin"] = origin
            if publish:
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
