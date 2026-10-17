"""Shared paper identity and immutable source-freeze evidence for acceptance tools."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import urlsplit

REGISTRY = "docs/paper_acceptance_registry.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def identity_key(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def source_key(value):
    value = value.strip().casefold()
    match = re.fullmatch(r"(?:https?://arxiv.org/(?:abs|pdf)/|arxiv:)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?", value)
    return "arxiv:" + match[1] if match else value.rstrip("/")


def repository_key(value):
    parsed = urlsplit(value)
    path = parsed.path.rstrip("/").removesuffix(".git")
    if parsed.scheme != "https" or parsed.hostname != "github.com" or not re.fullmatch(r"/[\w.-]+/[\w.-]+", path):
        raise ValueError("A canonical HTTPS GitHub repository is required")
    return "https://github.com" + path.casefold()


def load_registry(root, path=None):
    registry = json.loads(Path(path or Path(root) / REGISTRY).read_text("utf-8"))
    if registry.get("version") != 1 or not isinstance(registry.get("papers"), list):
        raise ValueError("Unsupported paper registry")
    claimed, ids = {}, set()
    for paper in registry["papers"]:
        paper_id = paper.get("id", "")
        if not isinstance(paper_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", paper_id) or paper_id in ids:
            raise ValueError("Duplicate or invalid paper id")
        ids.add(paper_id)
        if paper.get("status") not in {"candidate", "regression", "accepted"}:
            raise ValueError("Unknown paper eligibility status")
        if not isinstance(paper.get("title"), str) or not identity_key(paper["title"]):
            raise ValueError("Paper title is required")
        for field in ("aliases", "source_ids", "pdf_sha256", "history"):
            if not isinstance(paper.get(field), list):
                raise ValueError("Missing paper identity/history list: " + field)
        if not paper["history"] or any(not isinstance(h, dict) or not h.get("evidence") or not h.get("outcome") for h in paper["history"]):
            raise ValueError("Paper history requires outcomes and evidence")
        if paper["status"] == "candidate" and (not paper.get("repository_url") or not paper["pdf_sha256"] or not paper["source_ids"]):
            raise ValueError("Candidate identity is incomplete")
        if paper["status"] == "candidate" and any(h["outcome"] not in {
                "candidate_not_executed", "registered_unused_candidate", "preflight_verified"} for h in paper["history"]):
            raise ValueError("Candidate status conflicts with paper trial history")
        if paper["status"] == "regression" and not paper.get("retirement_reason"):
            raise ValueError("Regression paper requires a retirement reason")
        if paper.get("repository_url"):
            repository_key(paper["repository_url"])
        keys = [("name", identity_key(paper["title"])), ("name", identity_key(paper_id))]
        if any(not isinstance(a, str) or not identity_key(a) for a in paper["aliases"]):
            raise ValueError("Invalid title alias")
        keys += [("name", identity_key(a)) for a in paper["aliases"]]
        for value in paper["pdf_sha256"]:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Invalid PDF hash")
            keys.append(("pdf", value))
        for value in paper["source_ids"]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Invalid source identity")
            keys.append(("source", source_key(value)))
        for key in keys:
            if key in claimed and claimed[key] != paper_id:
                raise ValueError("Conflicting paper identity: " + str(key))
            claimed[key] = paper_id
    return registry


def select_paper(registry, paper_id, *, independent=False):
    matches = [p for p in registry["papers"] if identity_key(paper_id) in
               {identity_key(p["id"]), identity_key(p["title"]), *(identity_key(a) for a in p["aliases"])}]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous paper identity: " + paper_id)
    paper = matches[0]
    if independent and paper["status"] != "candidate":
        raise ValueError("Paper is not an unused independent candidate: " + paper["id"])
    if independent and not re.fullmatch(r"[0-9a-f]{40}", paper.get('repository_revision', '')):
        raise ValueError('Independent candidate requires a pinned author repository revision')
    return paper


def product_hashes(root):
    root = Path(root)
    paths = [root / "app.py"]
    for name in ("src", "frontend"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(root.glob("requirements*.txt"))
    paths.extend(root / "scripts" / name for name in ("freeze_holdout_source.py", "verify_browser_pdf_holdout.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def runtime_dependencies():
    distributions = [{'name': item.metadata['Name'], 'version': item.version}
                     for item in metadata.distributions() if item.metadata['Name']]
    return {'python': sys.version, 'implementation': sys.implementation.name, 'platform': sys.platform,
            'distributions': sorted(distributions, key=lambda item: (item['name'].casefold(), item['version']))}


def build_freeze(root, *, paper_id=None, registry_path=None):
    root = Path(root)
    registry = load_registry(root, registry_path)
    paper = select_paper(registry, paper_id, independent=True) if paper_id else None
    hashes = product_hashes(root)
    return {"version": 2, "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "files": hashes, "source_sha256": digest(hashes), "registry_sha256": digest(registry),
            "environment": runtime_dependencies(),
            "paper": paper, "independent_eligible": bool(paper), "objective": "author_code_execution",
            "excluded_regression_papers": [p["title"] for p in registry["papers"] if p["status"] != "candidate"],
            "rule": "A product repair retires the trial paper. Re-freeze and use another unused candidate."}


def verify_freeze(root, freeze, *, paper_id=None, pdf_sha256=None, repository=None,
                  require_independent=False, registry_path=None):
    root = Path(root)
    if freeze.get("version") == 2:
        current = product_hashes(root)
        if current != freeze.get("files") or digest(current) != freeze.get("source_sha256"):
            raise ValueError("Product source or dependency manifests changed after freeze")
    else:
        for name, expected in freeze["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid legacy freeze path")
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                raise ValueError("Product source changed after freeze: " + name)
    if not require_independent:
        return {"independent": False, "reason": "Regression audit; independent eligibility not requested"}
    if freeze.get("version") != 2 or not paper_id or freeze.get("independent_eligible") is not True:
        raise ValueError("Independent acceptance requires a version-2 paper-bound freeze and --paper-id")
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if current_commit != freeze.get('git_commit'):
        raise ValueError('Git commit changed after freeze')
    if runtime_dependencies() != freeze.get('environment'):
        raise ValueError('Application environment changed after freeze')
    registry = load_registry(root, registry_path)
    paper = select_paper(registry, paper_id, independent=True)
    if digest(registry) != freeze.get("registry_sha256") or paper != freeze.get("paper"):
        raise ValueError("Paper registry changed after freeze")
    if pdf_sha256 not in paper["pdf_sha256"]:
        raise ValueError("PDF hash is not registered for this paper")
    if repository_key(repository) != repository_key(paper["repository_url"]):
        raise ValueError("Expected repository does not match the registered paper")
    return {"independent": True, "paper_id": paper["id"], "pdf_sha256": pdf_sha256,
            "source_sha256": freeze["source_sha256"], "registry_sha256": freeze["registry_sha256"]}


def write_freeze(root, tag, record):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,120}", tag):
        raise ValueError("Invalid freeze tag")
    path = Path(root) / "data/runtime" / (tag + "_source_freeze.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
