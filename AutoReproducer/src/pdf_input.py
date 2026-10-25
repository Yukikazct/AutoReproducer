"""Validate an uploaded PDF and retain its visible identity and resource evidence.

PDF parsers are imported lazily so the bootstrap can recover missing application
dependencies. Text in the abstract, body or references cannot select a profile.
"""
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from pathlib import Path
import re
import json

from src.title_routing import match_paper_title, resolve_title_request
from src.repository_routing import (REZERO_PROFILE_ID, extract_current_repository_links,
                                    match_repository_profile)
from src.pdf_markdown import convert_pdf_pages


PDF_INPUT_API_VERSION = 3
# Record the source generation owned by this module. A long-running web host
# must not mistake the same public API for the same extraction implementation.
def _source_fingerprint():
    return hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in ("pdf_input.py", "pdf_markdown.py", "repository_routing.py", "repository_evidence.py")
    )).hexdigest()


PDF_SOURCE_FINGERPRINT = _source_fingerprint()


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
    page_texts: tuple = ()
    extraction: dict = field(default_factory=dict)


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
            # PDF string objects subclass str. Keep their exact characters in
            # a plain value so safe evidence reuse never needs custom objects.
            annotations.append({"page": page_number, "raw_url": str.__str__(uri),
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
        # Small-font captions and tables use word gaps below the default 3pt.
        pages = [page.extract_text(x_tolerance=1) or "" for page in document.pages]
        annotations = []
        for number, page in enumerate(document.pages, 1):
            try:
                for link in page.hyperlinks:
                    top = max(0, float(link["top"]) - 24)
                    bottom = min(page.height, float(link["bottom"]) + 24)
                    context = page.crop((0, top, page.width, bottom)).extract_text(x_tolerance=1) or ""
                    annotations.append({"page": number, "raw_url": link.get("uri", ""),
                                        "context": context})
            except Exception:
                continue
        metadata = document.metadata or {}
        return pages, str(metadata.get("Title") or ""), annotations


def _extraction_damage(text):
    """Rank glyph loss before numeric layout damage; never rewrite numbers."""
    glyphs = (sum(ord(char) < 32 and char not in '\n\r\t' or char == '\ufffd' for char in text)
              + len(re.findall(r'\(cid:\d+\)', text)))
    # Adjacent decimal cells can be concatenated, or a font change can split
    # one decimal. These are reasons to retry extraction from page geometry,
    # never enough evidence to insert/remove characters in the extracted text.
    numeric_layout = (len(re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+){2,}(?![\w.])', text))
                      + len(re.findall(r'(?<=\d)[ \t]+\.\d+', text)))
    return glyphs, numeric_layout


def _evidence_cache_identity(digest):
    """Bind reuse to current bytes, loaded implementation and parser versions."""
    from importlib.metadata import PackageNotFoundError, version
    from src import pdf_markdown
    try:
        if _source_fingerprint() != PDF_SOURCE_FINGERPRINT:
            return None  # A stale web module must not reuse old extraction.
        dependencies = {}
        for name in ('PyPDF2', 'pdfplumber', 'pdfminer.six', 'markitdown'):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = ''
        functions = (_extract_pdf_payload, _extract_pypdf, _extract_pdfplumber,
                     _pypdf_page_content, _extraction_damage, convert_pdf_pages,
                     extract_current_repository_links, pdf_markdown._converter,
                     pdf_markdown._page_payloads, pdf_markdown._geometry_tables,
                     pdf_markdown._rejection)
        identity = {'pdf_sha256': digest, 'source': PDF_SOURCE_FINGERPRINT,
                    'functions': [id(function) for function in functions],
                    'dependencies': dependencies}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    except Exception:
        return None  # Metadata/source probing cannot prevent a fresh parse.


def _conversion_cacheable(document):
    if type(document.extraction) is not dict:
        return False
    pages = document.extraction.get('pages', [])
    return type(pages) in (list, tuple) and bool(pages) and all(
        type(page) is dict and type(page.get('reason')) is str
        and page['reason'] not in {'markitdown_dependency_unavailable', 'page_count_mismatch',
                                  'page_serialization_failed', 'conversion_resource_limit'}
        and not page['reason'].startswith('conversion_failed:') for page in pages)


def extract_pdf_input(pdf_path):
    """Read current bytes; reuse matching canonical evidence only within this run."""
    try:
        payload = Path(pdf_path).read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise PDFInputError("PDF 文件不存在或无法读取，本次未生成或运行代码") from exc
    if b"%PDF-" not in payload[:1024]:
        raise PDFInputError("文件不是有效 PDF，本次未生成或运行代码")

    from src.agent_collaboration import current_collaboration
    session = current_collaboration()
    if session is not None and any(not callable(getattr(session, name, None)) for name in (
            'evidence_cache_scope', 'cached_evidence', 'cache_evidence', 'reject_evidence_cache')):
        session = None  # A long-running host may still own the older mailbox.
    if session is None:
        return _extract_pdf_payload(payload)[0]
    digest = hashlib.sha256(payload).hexdigest()
    with session.evidence_cache_scope():
        identity = _evidence_cache_identity(digest)
        if identity is None:
            return _extract_pdf_payload(payload)[0]
        cached = session.cached_evidence(identity)
        if cached is not None:
            cached['repository_links'] = tuple(cached['repository_links'])
            cached['page_texts'] = tuple(cached['page_texts'])
            return PDFInput(**cached)
        document, parser_healthy = _extract_pdf_payload(payload)
        if (parser_healthy and _conversion_cacheable(document)
                and identity == _evidence_cache_identity(digest)):
            session.cache_evidence(identity, vars(document))
        else:
            session.reject_evidence_cache()
        return document


def _extract_pdf_payload(payload):
    """Perform the original complete extraction; failures are never reusable."""

    unavailable, empty, candidate = 0, False, None
    parser_healthy = True
    annotation_sources = []
    for parser in (_extract_pypdf, _extract_pdfplumber):
        try:
            extracted = parser(payload)
            # Preserve the established two-field parser test/fallback contract.
            pages, metadata_title = extracted[:2]
            annotations = extracted[2] if len(extracted) > 2 else ()
        except ImportError:
            unavailable += 1
            parser_healthy = False
            continue
        except Exception:
            parser_healthy = False
            continue
        text = "\n".join(pages)
        if not text.strip():
            empty = True
            continue
        damage = _extraction_damage(text)
        if candidate is not None:
            prior_pages, _, _, prior_damage = candidate
            # A cleaner caption cannot authorize omitting other readable pages.
            if (len(pages) != len(prior_pages)
                    or any(old.strip() and not new.strip() for old, new in zip(prior_pages, pages))):
                continue
            # A geometry parser may recover a table but lose math glyphs on
            # another page. Choose complete original pages independently.
            pages = [new if _extraction_damage(new) < _extraction_damage(old) else old
                     for old, new in zip(prior_pages, pages)]
            damage = _extraction_damage('\n'.join(pages))
            if damage >= prior_damage:
                continue
        # Both parsers read the same bytes. Keep their original annotation
        # contexts: column interleaving can obscure a declaration in one parser.
        annotation_sources.extend(annotations)
        candidate = (pages, metadata_title, annotations, damage)
        if damage == (0, 0):
            break

    if candidate is not None:
        pages, metadata_title, annotations, _ = candidate
        markdown_pages, extraction = convert_pdf_pages(payload, pages)
        document = PDFInput(text="\n".join(markdown_pages), first_page_text=pages[0] if pages else "",
                        metadata_title=metadata_title, page_count=len(pages),
                        sha256=hashlib.sha256(payload).hexdigest(), byte_size=len(payload),
                        repository_links=tuple(extract_current_repository_links(pages, annotation_sources)),
                        page_texts=markdown_pages, extraction=extraction)
        return document, parser_healthy

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
