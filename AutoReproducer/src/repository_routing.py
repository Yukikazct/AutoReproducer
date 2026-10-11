"""Select reviewed experiments from a paper's visible identity and code claim.

A GitHub link alone is not authority to run a reviewed paper experiment. Both
the complete visible title and the original author-code declaration must agree.
The discovery repository can differ from the reviewed author training repository.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
import threading
import unicodedata


REZERO_PROFILE_ID = "rezero_cifar10_reference"
REZERO_TITLE = "ReZero is All You Need: Fast Convergence at Large Depth"
REZERO_DISCOVERY_REPOSITORY = "https://github.com/majumderb/rezero"
REZERO_EXAMPLES_REPOSITORY = "https://github.com/tbachlechner/ReZero-examples"
REZERO_TRAINING_REPOSITORY = "https://github.com/tbachlechner/ReZero-Superconvergence"
REZERO_TRAINING_REVISION = "6c0212669ac8c23d3db6f2b99255bcf6e3c5e6e6"

# Reviewed immutable Git blobs, rather than the Windows checkout's CRLF bytes.
# The paper's library README does not link the training repository directly.
_REZERO_README_HOPS = (
    {
        "repository_url": REZERO_DISCOVERY_REPOSITORY,
        "revision": "e2c94a825c5564217e8cf4d75a28d59cab1d7029",
        "path": "README.md",
        "sha256": "b766654a4144964fc3f52943db637788047f0a6c027f4d1fef7c7c29fa56e501",
        "line": 63,
        "quote": "Watch for more tutorials in this [space](https://github.com/tbachlechner/ReZero-examples).",
        "linked_url": REZERO_EXAMPLES_REPOSITORY,
        "target_repository_url": REZERO_EXAMPLES_REPOSITORY,
    },
    {
        "repository_url": REZERO_EXAMPLES_REPOSITORY,
        "revision": "fe7e7ef080df6555018bec3102613c9e4c0d1f1d",
        "path": "README.md",
        "sha256": "6927734c91a0703805d725c2387d4fbbbd89fe73d0ef1d6396f9799e474444ae",
        "line": 7,
        "quote": "- [ReZero speeds up superconvergence](https://github.com/tbachlechner/ReZero-Superconvergence/blob/master/Faster_SuperC.ipynb)",
        "linked_url": REZERO_TRAINING_REPOSITORY + "/blob/master/Faster_SuperC.ipynb",
        "target_repository_url": REZERO_TRAINING_REPOSITORY,
    },
)

_REVIEWED_REPOSITORIES = {
    REZERO_DISCOVERY_REPOSITORY.casefold(): {
        "title": REZERO_TITLE, "profile": REZERO_PROFILE_ID,
        "scope": "selected_paper_experiment",
        "discovery_repository_url": REZERO_DISCOVERY_REPOSITORY,
        "training_repository_url": REZERO_TRAINING_REPOSITORY,
        "training_repository_revision": REZERO_TRAINING_REVISION,
        "repository_relationship": {
            "source": "reviewed_author_readme",
            "url": REZERO_DISCOVERY_REPOSITORY,
            "description": "作者主仓库 README 指向 ReZero-examples；该示例仓库 README 再指向 CIFAR-10 超收敛训练仓库。",
            "hash_basis": "git_blob_bytes",
            "hops": [
                {**hop, "source_url": f"{hop['repository_url']}/blob/{hop['revision']}/{hop['path']}#L{hop['line']}"}
                for hop in _REZERO_README_HOPS
            ],
        },
    },
}
_EVIDENCE_VERSION = 1
_evidence_lock = threading.RLock()


def current_repository_evidence():
    """Isolate the new extractor when an active web task owns an older module."""
    with _evidence_lock:
        source = Path(__file__).with_name("repository_evidence.py")
        # Classification fixes can retain the public API. A version-only check
        # would keep an old extractor that misses the original ReZero footnote.
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        name = f"src._repository_evidence_v{_EVIDENCE_VERSION}_{digest}"
        upgraded = sys.modules.get(name)
        if upgraded is not None:
            return upgraded
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 PDF 仓库证据提取器")
        upgraded = importlib.util.module_from_spec(spec)
        sys.modules[name] = upgraded
        try:
            spec.loader.exec_module(upgraded)
            if vars(upgraded).get("REPOSITORY_EVIDENCE_API_VERSION") != _EVIDENCE_VERSION:
                raise RuntimeError("PDF 仓库证据提取器版本不兼容")
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return upgraded


def extract_current_repository_links(pages, annotations=()):
    return current_repository_evidence().extract_repository_links(pages, annotations)


def _normalized_title(title):
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", title)).strip().casefold()


def match_repository_profile(visible_title, repository_links):
    """Return only a uniquely grounded reviewed route; never infer from metadata.

    ``visible_title`` is a complete leading title candidate from the PDF, not a
    user textbox, search snippet, metadata title, abstract, or reference mention.
    The caller retains responsibility for computing the immutable PDF digest.
    """
    title = _normalized_title(visible_title)
    extractor = current_repository_evidence()
    normalize = extractor.normalize_repository_url
    matches = []
    for evidence in repository_links or ():
        if (not isinstance(evidence, dict) or evidence.get("is_author_code") is not True
                or evidence.get("evidence_type") != "author_code_statement"
                or evidence.get("source") not in {"pdf_text", "pdf_annotation"}
                or type(evidence.get("page")) is not int or evidence["page"] < 1
                or not evidence.get("raw_url") or not evidence.get("context")
                or not extractor.is_author_code_statement(evidence["context"])):
            continue
        url = normalize(evidence.get("url", ""))
        expected = _REVIEWED_REPOSITORIES.get(url.casefold())
        if expected is None or _normalized_title(expected["title"]) != title:
            continue
        # The original URI must confirm the same repository as the normalized
        # record. Caller-supplied labels cannot turn a fork into author evidence.
        if normalize(evidence["raw_url"]).casefold() != url.casefold():
            continue
        matches.append({**deepcopy(expected), "source": "pdf_author_code_repository",
                        "evidence": deepcopy(evidence)})
    profiles = {match["profile"] for match in matches}
    if len(profiles) != 1:
        return None
    # Text is most reviewable; an annotation remains a valid alternative when
    # the actual URI is absent from visible text. Both are kept by PDFInput.
    return min(matches, key=lambda match: (match["evidence"]["page"],
                                          match["evidence"]["source"] != "pdf_text"))
