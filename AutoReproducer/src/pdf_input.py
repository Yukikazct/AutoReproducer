"""Validate an uploaded PDF and retain its visible identity and resource evidence.

PDF parsers are imported lazily so the bootstrap can recover missing application
dependencies. Text in the abstract, body or references cannot select a profile.
"""
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re

from src.title_routing import match_paper_title, resolve_title_request
from src.repository_routing import (REZERO_PROFILE_ID, extract_current_repository_links,
                                    match_repository_profile)


PDF_INPUT_API_VERSION = 3
# Record the source generation owned by this module. A long-running web host
# must not mistake the same public API for the same extraction implementation.
PDF_SOURCE_FINGERPRINT = hashlib.sha256(b"".join(
    Path(__file__).with_name(name).read_bytes()
    for name in ("pdf_input.py", "repository_routing.py", "repository_evidence.py")
)).hexdigest()


class PDFInputError(ValueError):
    """The supplied PDF cannot provide readable paper content."""


class PDFParserUnavailable(PDFInputError):
    """No installed PDF parser is available; prepare the app runtime and retry."""


@dataclass(frozen=True)
class PDFInput:
    text: str
    first_page_text: str
    metadata_title: str
    page_count: int
    sha256: str
    byte_size: int
    repository_links: tuple = ()


def _pypdf_page_content(page, page_number):
    fragments = []

    def remember_text(text, cm, tm, font, size):
        if text.strip():
            # PDF annotations and these transformed text origins share the
            # bottom-left PDF coordinate system. Retain nearby lines as context.
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            fragments.append((x, y, text))

    try:
        text = page.extract_text(visitor_text=remember_text) or ""
    except TypeError:  # Older compatible parsers may lack the visitor argument.
        text = page.extract_text() or ""
    annotations = []
    for indirect in page.get("/Annots", ()):
        try:
            annotation = indirect.get_object()
            action = annotation.get("/A", {})
            action = action.get_object() if hasattr(action, "get_object") else action
            uri = action.get("/URI", "")
            if not isinstance(uri, str):
                continue
            rect = annotation.get("/Rect", ())
            nearby = []
            if len(rect) == 4:
                bottom, top = min(float(rect[1]), float(rect[3])), max(float(rect[1]), float(rect[3]))
                nearby = [fragment for fragment in fragments if bottom - 24 <= fragment[1] <= top + 24]
            context = " ".join(fragment[2] for fragment in sorted(nearby, key=lambda item: (-item[1], item[0])))
            annotations.append({"page": page_number, "raw_url": uri,
                                "context": context or str(annotation.get("/Contents", ""))})
        except Exception:
            # A malformed optional annotation must not invalidate readable text.
            continue
    return text, annotations


def _extract_pypdf(payload):
    from PyPDF2 import PdfReader
    reader = PdfReader(BytesIO(payload))
    if reader.is_encrypted and not reader.decrypt(""):
        raise PDFInputError("PDF 已加密，无法读取论文正文")
    contents = [_pypdf_page_content(page, index) for index, page in enumerate(reader.pages, 1)]
    pages = [content[0] for content in contents]
    annotations = [annotation for content in contents for annotation in content[1]]
    metadata = reader.metadata or {}
    return pages, str(metadata.get("/Title") or ""), annotations


def _extract_pdfplumber(payload):
    import pdfplumber
    with pdfplumber.open(BytesIO(payload)) as document:
        pages = [page.extract_text() or "" for page in document.pages]
        annotations = []
        for number, page in enumerate(document.pages, 1):
            try:
                for link in page.hyperlinks:
                    top = max(0, float(link["top"]) - 24)
                    bottom = min(page.height, float(link["bottom"]) + 24)
                    context = page.crop((0, top, page.width, bottom)).extract_text() or ""
                    annotations.append({"page": number, "raw_url": link.get("uri", ""),
                                        "context": context})
            except Exception:
                continue
        metadata = document.metadata or {}
        return pages, str(metadata.get("Title") or ""), annotations


def extract_pdf_input(pdf_path):
    """Read one immutable byte snapshot, trying a second parser on empty text."""
    try:
        payload = Path(pdf_path).read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise PDFInputError("PDF 文件不存在或无法读取，本次未生成或运行代码") from exc
    if b"%PDF-" not in payload[:1024]:
        raise PDFInputError("文件不是有效 PDF，本次未生成或运行代码")

    unavailable, empty = 0, False
    for parser in (_extract_pypdf, _extract_pdfplumber):
        try:
            extracted = parser(payload)
            # Preserve the established two-field parser test/fallback contract.
            pages, metadata_title = extracted[:2]
            annotations = extracted[2] if len(extracted) > 2 else ()
        except ImportError:
            unavailable += 1
            continue
        except Exception:
            continue
        text = "\n".join(pages)
        if not text.strip():
            empty = True
            continue
        return PDFInput(text=text, first_page_text=pages[0] if pages else "",
                        metadata_title=metadata_title, page_count=len(pages),
                        sha256=hashlib.sha256(payload).hexdigest(), byte_size=len(payload),
                        repository_links=tuple(extract_current_repository_links(pages, annotations)))

    if unavailable:
        raise PDFParserUnavailable("PDF 解析依赖缺失，需要自动准备应用运行环境后重试")
    if empty:
        raise PDFInputError("PDF 未提取到可读正文，可能是扫描件或图片型 PDF；本次未生成或运行代码")
    raise PDFInputError("PDF 损坏、加密或无法解析，本次未生成或运行代码")


_PREAMBLE = re.compile(
    r"^(?:arxiv\s*:|published as a conference paper|preprint\b|"
    r"accepted (?:at|by)\b|proceedings of\b|"
    r"(?:iclr|icml|neurips|nips|cvpr|eccv|acl)\s+\d{4}\b)", re.I)
_BODY_HEADING = re.compile(r"^(?:abstract|introduction|references)\b", re.I)


def _leading_title_candidates(document):
    """Yield complete leading lines, stopping before the body or bibliography."""
    lines = [line.strip() for line in document.first_page_text.splitlines() if line.strip()]
    while lines and _PREAMBLE.match(lines[0]):
        lines.pop(0)
    if not lines or _BODY_HEADING.match(lines[0]):
        return
    for count in range(1, min(6, len(lines)) + 1):
        if any(_BODY_HEADING.match(line) for line in lines[:count]):
            break
        yield " ".join(lines[:count])


def supported_pdf_title(document):
    """Match complete leading title lines, never a mention elsewhere in a PDF.

    Metadata can confirm the title but cannot override a different visible title.
    Only complete consecutive lines at the start of the first page are candidates.
    """
    for candidate in _leading_title_candidates(document):
        profile = match_paper_title(candidate)
        if profile is not None:
            # A conflicting known metadata title is not unambiguous evidence.
            metadata_profile = match_paper_title(document.metadata_title)
            if metadata_profile is not None and metadata_profile != profile:
                return None
            return candidate
    return None


def resolve_pdf_request(request):
    """Copy a valid PDF request and select a supported visible paper identity."""
    resolved = dict(request)
    if not request.get("pdf_path"):
        return resolved
    document = extract_pdf_input(request["pdf_path"])
    if request.get("mock_mode") or any(request.get(key) for key in (
        "code", "preferred_repo_url", "code_repo_url", "corpus_paper",
    )):
        return resolved
    # Rebuild preview evidence from the bytes currently supplied to the worker.
    # In particular, a stale selected ReZero profile cannot authorize execution
    # after its PDF was replaced with an unrelated paper or a link-only document.
    resolved.pop("pdf_resolution", None)
    for candidate in _leading_title_candidates(document):
        repository_route = match_repository_profile(candidate, document.repository_links)
        if repository_route is None:
            continue
        metadata_profile = match_paper_title(document.metadata_title)
        if metadata_profile is not None and metadata_profile != repository_route["profile"]:
            continue
        if request.get("experiment_profile") not in (None, "", repository_route["profile"]):
            return resolved
        resolved["experiment_profile"] = repository_route["profile"]
        resolved["pdf_resolution"] = {
            **repository_route, "sha256": document.sha256,
            "pages": document.page_count, "bytes": document.byte_size,
            "repository_links": [dict(link) for link in document.repository_links],
        }
        return resolved
    if request.get("experiment_profile") == REZERO_PROFILE_ID:
        raise PDFInputError("本次 PDF 的完整标题与作者代码声明未共同确认 ReZero 实验；本次未运行训练代码")
    title = supported_pdf_title(document)
    if title is None:
        return resolved
    route = resolve_title_request({"paper_title": title})
    if request.get("experiment_profile") not in (None, "", route["experiment_profile"]):
        return resolved
    resolved["experiment_profile"] = route["experiment_profile"]
    resolved["pdf_resolution"] = {
        "sha256": document.sha256, "title": title, "source": "exact_pdf_header",
        "profile": route["experiment_profile"],
        "scope": route["title_resolution"]["scope"],
        "pages": document.page_count, "bytes": document.byte_size,
    }
    return resolved
