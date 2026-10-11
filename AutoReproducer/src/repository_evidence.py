"""Find GitHub repositories in document text without treating links as official.

The evidence records are deliberately plain dictionaries so they can travel
through the existing agent, progress, and audit JSON boundaries.
"""
import re
from urllib.parse import urlsplit


REPOSITORY_EVIDENCE_API_VERSION = 1


_REPOSITORY_LINK = re.compile(
    r"(?<![\w.@/-])(?:https?://)?(?:www\.)?github\.com/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"(?:/[^\s<>\"'\)\]\}]+)?(?:[?#][^\s<>\"'\)\]\}]+)?", re.I)
_REFERENCES = re.compile(r"(?im)^\s*(?:\d+\.?\s+)?(?:references|bibliography)\s*$")
_APPENDIX = re.compile(r"(?im)^\s*(?:appendix|supplementary material)\b")
_AUTHOR_CODE = re.compile(
    r"(?<![\w])(?:\d{1,3})?(?:source\s+code|our\s+(?:code|implementation|software|library)|"
    r"(?:code|implementation|software|library)(?:\s+for\s+[^.!?]{1,80})?)"
    r"\s+(?:(?:is|are|was|has\s+been|can\s+be|will\s+be)\s+)?"
    r"(?:publicly\s+|freely\s+|openly\s+)?(?:available|published|released|provided|found|downloaded|accessed)\b"
    r"|\b(?:we|the\s+authors)\s+(?:also\s+)?(?:release|provide|publish|open[- ]source)\b"
    r"[^.!?]{0,100}\b(?:code|implementation|software|library)\b"
    # Footnote extraction can join the marker directly to "Code". A labelled
    # code-for-project link is an explicit declaration even without "available".
    r"|(?<![\w])(?:\d{1,3})?(?:source\s+code|code|implementation|software|library)"
    r"\s+for\s+[^.!?:]{1,140}:\s*(?=(?:https?://)?(?:www\.)?github\b)", re.I)
_EXTERNAL_CODE = re.compile(
    r"\b(?:third[- ]party|prior\s+work|previous\s+work|baseline\s+(?:code|implementation))\b"
    r"|\b(?:code|implementation)\s+for\s+(?:the\s+)?baseline\b", re.I)
_NON_REPOSITORY_OWNERS = {
    "about", "collections", "contact", "enterprise", "events", "explore",
    "features", "issues", "join", "login", "marketplace", "new", "notifications",
    "orgs", "organizations", "pricing", "pulls", "search", "settings", "sponsors",
    "topics", "trending",
}


def _repair_url_wraps(text):
    """Repair only URL syntax boundaries, preserving unrelated prose spacing.

    Arbitrary whitespace is not deleted: a repository at the end of a line must
    not absorb the first word of the next sentence. A broken host or a wrap after
    a slash/hyphen is unambiguous enough to repair.
    """
    text = str(text or "")
    # A position map keeps contexts and raw URLs tied to the original text.
    chars, positions = [], []
    for index, char in enumerate(text):
        if char in ("\u00ad", "\u200b"):
            continue
        chars.append(char)
        positions.append(index)
    pattern = re.compile(
        r"(?<=[/-])[ \t]*\r?\n[ \t]*(?=[A-Za-z0-9_-])|"
        r"(?<=github\.)[ \t]*\r?\n[ \t]*(?=com/)|"
        r"(?<=github)[ \t]*\r?\n[ \t]*(?=\.com/)", re.I)
    # Limit repairs to a URL token (including a not-yet-complete github host).
    cursor = 0
    while True:
        repaired = "".join(chars)
        match = pattern.search(repaired, cursor)
        if match is None:
            break
        prefix = repaired[max(0, match.start() - 240):match.start()]
        token = re.split(r"[\s<>\"'\)\]\}]", prefix)[-1]
        if re.search(r"(?:https?://)?(?:www\.)?github(?:\.|\.com/)", token, re.I):
            del chars[match.start():match.end()]
            del positions[match.start():match.end()]
            cursor = match.start()
        else:
            cursor = match.end()
    return "".join(chars), positions


def normalize_repository_url(raw_url):
    """Return a GitHub repository root, accepting PDF line wraps and bare hosts."""
    if not isinstance(raw_url, str):
        return ""
    candidate = re.sub(r"\s+", "", raw_url.replace("\u00ad", "").replace("\u200b", ""))
    candidate = candidate.strip("<>[](){}\"',.;:!?")
    if re.match(r"^(?:www\.)?github\.com/", candidate, re.I):
        candidate = "https://" + candidate
    try:
        parts = urlsplit(candidate)
        if (parts.scheme.lower() not in ("http", "https") or
                (parts.hostname or "").lower() not in ("github.com", "www.github.com") or
                parts.username or parts.password or parts.port):
            return ""
    except ValueError:
        return ""
    path = parts.path.strip("/").split("/")
    if len(path) < 2:
        return ""
    owner, repository = path[:2]
    repository = repository.rstrip(".,;:!?")
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if (owner.lower() in _NON_REPOSITORY_OWNERS or owner in (".", "..") or
            repository in ("", ".", "..") or
            not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or
            not re.fullmatch(r"[A-Za-z0-9_.-]+", repository)):
        return ""
    return f"https://github.com/{owner}/{repository}"


def _context(text, start, end):
    value = text[max(0, start - 280):min(len(text), end + 160)]
    value = re.sub(r"[\x00-\x08\x0e-\x1f]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def is_author_code_statement(context):
    """Recognize an explicit code declaration without claiming official ownership."""
    return bool(isinstance(context, str) and _AUTHOR_CODE.search(context)
                and not _EXTERNAL_CODE.search(context))


def _evidence(url, raw_url, page, source, context, reference=False, claim_context=None):
    claim = context if claim_context is None else claim_context
    author_code = bool(not reference and is_author_code_statement(claim))
    return {"url": url, "raw_url": raw_url, "page": page, "source": source,
            "context": context, "evidence_type": "reference" if reference else (
                "author_code_statement" if author_code else "repository_link"),
            "is_author_code": author_code}


def extract_repository_links(pages, annotations=()):
    """Extract ranked repository evidence from all pages and URI annotations.

    ``annotations`` entries contain 1-based ``page``, ``raw_url``, and optional
    nearby visible ``context``. Duplicate annotation rectangles for one wrapped
    link are collapsed, while text and annotation provenance remain independent.
    """
    records, page_reference, page_headings = [], {}, {}
    reference_section = False
    for page_number, text in enumerate(pages, 1):
        text = str(text or "")
        page_reference[page_number] = reference_section
        headings = sorted([(m.start(), True) for m in _REFERENCES.finditer(text)] +
                          [(m.start(), False) for m in _APPENDIX.finditer(text)])
        page_headings[page_number] = headings
        repaired, positions = _repair_url_wraps(text)
        matches = list(_REPOSITORY_LINK.finditer(repaired))
        for index, match in enumerate(matches):
            url = normalize_repository_url(match.group())
            if not url:
                continue
            start, end = positions[match.start()], positions[match.end() - 1] + 1
            reference = reference_section
            for offset, state in headings:
                if offset <= start:
                    reference = state
            # A declaration for a previous link must not be inherited merely
            # because a different repository appears in the same paragraph.
            left = positions[matches[index - 1].end() - 1] + 1 if index else 0
            right = positions[matches[index + 1].start()] if index + 1 < len(matches) else len(text)
            context = _context(text[max(left, start - 280):min(right, end + 160)],
                               start - max(left, start - 280), end - max(left, start - 280))
            records.append(_evidence(url, text[start:end], page_number,
                                     "pdf_text", context, reference,
                                     text[max(left, start - 280):end]))
        if headings:
            reference_section = headings[-1][1]

    text_records = list(records)
    for annotation in annotations:
        raw_url = annotation.get("raw_url", "")
        url = normalize_repository_url(raw_url)
        page = annotation.get("page")
        if not url or not isinstance(page, int) or not 1 <= page <= len(pages):
            continue
        matching = next((record for record in text_records
                         if record["page"] == page and record["url"].lower() == url.lower()), None)
        context = (matching["context"] if matching else
                   re.sub(r"\s+", " ", str(annotation.get("context", ""))).strip())
        reference = page_reference.get(page, False)
        if matching:
            reference = matching["evidence_type"] == "reference"
        else:
            page_text = str(pages[page - 1] or "")
            collapsed = re.sub(r"\s+", " ", page_text).strip()
            offset = collapsed.find(context) if context else -1
            if offset >= 0:
                declaration = _AUTHOR_CODE.search(context)
                # Nearby annotation context can begin above a References
                # heading. Classify the declaration itself, not its first line.
                offset += declaration.start() if declaration else max(0, len(context) - 1)
            for heading_offset, state in page_headings[page]:
                # Use the nearby visible text to locate a URI within its page.
                # If that position is unavailable, a references page remains a
                # conservative reference candidate instead of an author claim.
                if offset < 0 or len(re.sub(r"\s+", " ", page_text[:heading_offset]).strip()) <= offset:
                    reference = state
        record = _evidence(url, raw_url, page, "pdf_annotation", context, reference)
        if matching:
            record["evidence_type"] = matching["evidence_type"]
            record["is_author_code"] = matching["is_author_code"]
        records.append(record)

    unique = []
    seen = set()
    for record in records:
        key = (record["url"].lower(), record["page"], record["source"],
               record["raw_url"], record["context"])
        if key not in seen:
            unique.append(record)
            seen.add(key)
    ranks = {"author_code_statement": 0, "repository_link": 1, "reference": 2}
    return sorted(unique, key=lambda record: (ranks[record["evidence_type"]], record["page"]))
