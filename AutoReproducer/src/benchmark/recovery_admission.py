"""Trusted, bounded exposure review for formal recovery holdouts.

An exposure review records what the operator actually searched. It cannot prove
that an unrecorded use or fork does not exist. Models never receive this record.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

from src.holdout_registry import REGISTRY, digest, identity_key, repository_key, source_key, validate_registry

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_FILES = 128
MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024 * 1024
EVENT_KINDS = {"discovered", "static_review", "author_execution", "product_tuning", "dev_case", "holdout_trial"}
DISQUALIFYING_EVENTS = {"product_tuning", "dev_case", "holdout_trial"}


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Exposure review timestamps must be timezone-aware ISO dates")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Exposure review timestamps must be timezone-aware ISO dates") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Exposure review timestamps must be timezone-aware ISO dates")
    return parsed


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _git_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _identity_keys(paper):
    if (not isinstance(paper, dict) or not _text(paper.get("title")) or not identity_key(paper["title"])
            or not isinstance(paper.get("aliases"), list)
            or any(not _text(alias) or not identity_key(alias) for alias in paper["aliases"])
            or not isinstance(paper.get("source_ids"), list) or not paper["source_ids"]
            or any(not _text(source) for source in paper["source_ids"])
            or not isinstance(paper.get("pdf_sha256"), list) or not paper["pdf_sha256"]
            or any(not _sha(sha) for sha in paper["pdf_sha256"])):
        raise ValueError("Exposure review needs complete paper title/aliases/source/PDF identity")
    return ({("name", identity_key(name)) for name in [paper["title"], *paper["aliases"]]}
            | {("source", source_key(source)) for source in paper["source_ids"]}
            | {("pdf", sha) for sha in paper["pdf_sha256"]})


def _registered_keys(paper):
    return ({("name", identity_key(name)) for name in [paper["id"], paper["title"], *paper["aliases"]]}
            | {("source", source_key(source)) for source in paper["source_ids"]}
            | {("pdf", sha) for sha in paper["pdf_sha256"]})


def _read_bytes(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Exposure documents must be regular files, without symlinks")
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Exposure document or evidence exceeds the size bound")
    return content


def _read_json(path):
    return json.loads(_read_bytes(path, MAX_DOCUMENT_BYTES).decode("utf-8"))


def _verify_evidence(audit, directory):
    total = 0
    content_by_index = {}
    root = Path(directory).resolve(strict=True)
    for index, item in enumerate(audit["evidence"]):
        name = item["path"]
        relative = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("Exposure evidence paths must stay inside the audit directory")
        path = root
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Exposure evidence symlinks are not allowed")
        if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
            raise ValueError("Exposure evidence must be an existing regular file")
        content = _read_bytes(path, MAX_EVIDENCE_BYTES - total)
        total += len(content)
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Exposure evidence changed from its registered digest")
        content_by_index[index] = content
    for group in audit["repository_groups"]:
        for acquisition in group["source_acquisitions"]:
            expected = {key: value for key, value in acquisition.items() if key != "evidence_index"}
            try:
                actual = json.loads(content_by_index[acquisition["evidence_index"]].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Source acquisition evidence must be its original JSON receipt") from exc
            if digest(actual) != digest(expected):
                raise ValueError("Source acquisition differs from its actual evidence receipt")


def _validate_audit(audit, frozen_at):
    if not isinstance(audit, dict) or type(audit.get("version")) is not int or audit["version"] != 1:
        raise ValueError("Exposure audit version must be 1")
    reviewed = _timestamp(audit.get("reviewed_at"))
    frozen = _timestamp(frozen_at)
    if frozen > datetime.now(timezone.utc):
        raise ValueError("A formal holdout freeze cannot be in the future")
    if reviewed > frozen:
        raise ValueError("Exposure review cannot occur after the product trial freeze")
    if not _text(audit.get("auditor")):
        raise ValueError("A trusted exposure auditor must be named")
    scope = audit.get("scope")
    if not isinstance(scope, list) or not scope or any(not _text(item) for item in scope):
        raise ValueError("Exposure review must declare its bounded search scope")
    evidence = audit.get("evidence")
    if (not isinstance(evidence, list) or not 1 <= len(evidence) <= MAX_EVIDENCE_FILES
            or any(not isinstance(item, dict) or not _text(item.get("path"))
                   or not _sha(item.get("sha256")) for item in evidence)):
        raise ValueError("Exposure review needs hash-bound source evidence")
    if len({item["path"].casefold() for item in evidence}) != len(evidence):
        raise ValueError("Exposure evidence paths must be unique")
    groups = audit.get("repository_groups")
    if not isinstance(groups, list) or not groups or len(groups) > 20000:
        raise ValueError("Exposure review needs bounded repository identity groups")
    indexed, identifiers, paper_claims, source_claims = {}, set(), {}, {}
    for group in groups:
        if not isinstance(group, dict) or not _text(group.get("id")) or group["id"] in identifiers:
            raise ValueError("Exposure repository group ids must be unique")
        identifiers.add(group["id"])
        for key in _identity_keys(group.get("paper_identity")):
            if key in paper_claims:
                raise ValueError("Paper identity conflicts across exposure groups")
            paper_claims[key] = group["id"]
        urls = group.get("repository_urls")
        if not isinstance(urls, list) or not urls or len(urls) > 100 or any(not _text(url) for url in urls):
            raise ValueError("Exposure repository group needs reviewed original/fork URLs")
        if group.get("aliases_reviewed") is not True:
            raise ValueError("Repository fork and alias review is incomplete")
        if group.get("finding") not in {"no_tuning_found", "used_for_tuning", "unknown"}:
            raise ValueError("Unknown exposure review finding")
        refs = group.get("evidence_indices")
        if (not isinstance(refs, list) or not refs
                or any(type(index) is not int or not 0 <= index < len(evidence) for index in refs)):
            raise ValueError("Repository review must reference its actual evidence")
        events = group.get("events")
        if not isinstance(events, list):
            raise ValueError("Repository review must record known exposure events")
        for event in events:
            if (not isinstance(event, dict) or event.get("kind") not in EVENT_KINDS
                    or type(event.get("evidence_index")) is not int
                    or event["evidence_index"] not in refs):
                raise ValueError("Exposure events require known kinds and evidence references")
            if _timestamp(event.get("at")) > reviewed:
                raise ValueError("Exposure events cannot occur after their review")
        acquisitions = group.get("source_acquisitions")
        if not isinstance(acquisitions, list) or not acquisitions:
            raise ValueError("Repository review requires trusted source acquisition receipts")
        for acquisition in acquisitions:
            if (not isinstance(acquisition, dict) or type(acquisition.get("version")) is not int
                    or acquisition["version"] != 1 or not _git_sha(acquisition.get("repository_revision"))
                    or not _git_sha(acquisition.get("git_tree"))
                    or type(acquisition.get("evidence_index")) is not int
                    or acquisition["evidence_index"] not in refs):
                raise ValueError("Source acquisition identity or evidence reference is incomplete")
            if repository_key(acquisition.get("repository_url", "")) not in {repository_key(url) for url in urls}:
                raise ValueError("Source acquisition repository is outside its reviewed identity group")
            if _timestamp(acquisition.get("acquired_at")) > reviewed:
                raise ValueError("Source acquisition must precede its exposure review")
            files, blobs = acquisition.get("source_files"), acquisition.get("git_blobs")
            if (not isinstance(files, dict) or not files or len(files) > 20000
                    or any(not isinstance(name, str) or not _sha(sha) for name, sha in files.items())
                    or not isinstance(blobs, dict) or set(blobs) != set(files)
                    or any(not isinstance(blob, dict) or not _git_sha(blob.get("sha1"))
                           or blob.get("mode") not in {"100644", "100755"} for blob in blobs.values())):
                raise ValueError("Source acquisition must bind complete regular Git files and frozen bytes")
            fingerprint = digest(files)
            if fingerprint in source_claims and source_claims[fingerprint] != group["id"]:
                raise ValueError("The same frozen source cannot count as different holdout repositories")
            source_claims[fingerprint] = group["id"]
        for url in urls:
            key = repository_key(url)
            if key in indexed:
                raise ValueError("Repository aliases or forks conflict across exposure groups")
            indexed[key] = group
    return indexed


def validate_admission(manifest):
    """Validate a detached frozen review; absence remains historical/unaudited."""
    admission = manifest.get("holdout_admission")
    if admission is None:
        return None
    if not isinstance(admission, dict) or type(admission.get("version")) is not int or admission["version"] != 1:
        raise ValueError("Holdout admission version must be 1")
    registry, audit = admission.get("registry"), admission.get("audit")
    if (not isinstance(registry, dict) or type(registry.get("version")) is not int or registry["version"] != 1
            or not isinstance(registry.get("papers"), list)
            or digest(registry) != admission.get("registry_sha256")
            or digest(audit) != admission.get("audit_sha256")):
        raise ValueError("Frozen registry or exposure audit digest changed")
    validate_registry(registry)
    groups = _validate_audit(audit, manifest.get("frozen_at"))
    blocked = set()
    for paper in registry["papers"]:
        if not isinstance(paper, dict):
            raise ValueError("Invalid frozen registry entry")
        if paper.get("repository_url") and paper.get("status") != "candidate":
            blocked.add(repository_key(paper["repository_url"]))
    dev_groups = {groups[repository_key(case["repository_url"])]["id"]
                  for case in manifest["cases"] if case["split"] == "dev"
                  and repository_key(case["repository_url"]) in groups}
    for case in manifest["cases"]:
        if case["split"] != "holdout" or case["origin"] != "natural":
            continue
        group = groups.get(repository_key(case["repository_url"]))
        if group is None:
            raise ValueError("Natural holdout repository has no exposure review")
        if group["id"] in dev_groups:
            raise ValueError("Reviewed repository aliases cannot cross dev and holdout splits")
        if (group["finding"] != "no_tuning_found"
                or any(event["kind"] in DISQUALIFYING_EVENTS for event in group["events"])):
            raise ValueError("Holdout repository exposure is unknown or includes prior tuning/trials")
        if any(repository_key(url) in blocked for url in group["repository_urls"]):
            raise ValueError("Holdout repository or reviewed fork is a registered regression/accepted paper")
        keys = _identity_keys(group["paper_identity"])
        for paper in registry["papers"]:
            if keys & _registered_keys(paper):
                if paper["status"] != "candidate":
                    raise ValueError("Holdout paper identity is a registered regression/accepted paper")
                if (paper.get("repository_url") and repository_key(paper["repository_url"])
                        not in {repository_key(url) for url in group["repository_urls"]}):
                    raise ValueError("Registered paper identity conflicts with its reviewed repository")
        _case_acquisition(case, group)
    return admission


def prepare_admission(manifest, *, audit_path, registry_path=None):
    """Read actual operator files before freezing; never invoked by a model."""
    registry = validate_registry(_read_json(registry_path or APPLICATION_ROOT / REGISTRY))
    audit = _read_json(audit_path)
    admission = {"version": 1, "registry": registry, "registry_sha256": digest(registry),
                 "audit": audit, "audit_sha256": digest(audit)}
    value = {**manifest, "holdout_admission": admission}
    validate_admission(value)
    _verify_evidence(audit, Path(audit_path).parent)
    return deepcopy(admission)


def verify_admission(manifest, *, audit_path=None, registry_path=None):
    """Recheck actual registry, review and evidence at each formal trial boundary."""
    admission = validate_admission(manifest)
    if admission is None:
        return None
    if audit_path is None:
        raise ValueError("Audited holdout execution requires the original exposure audit file")
    current = prepare_admission(manifest, audit_path=audit_path, registry_path=registry_path)
    if current != admission:
        raise ValueError("Shared registry or exposure audit changed after freeze")
    return admission_receipt(manifest)


def admission_receipt(manifest):
    admission = validate_admission(manifest)
    if admission is None:
        return None
    return {"version": 1, "verified": True, "admission_sha256": digest(admission),
            "registry_sha256": admission["registry_sha256"], "audit_sha256": admission["audit_sha256"]}


def repository_identity(manifest, case):
    """Count reviewed forks together rather than inflating independent repositories."""
    admission = validate_admission(manifest)
    key = repository_key(case["repository_url"])
    if admission is None:
        return key
    groups = _validate_audit(admission["audit"], manifest["frozen_at"])
    return groups[key]["id"] if key in groups else key


def _case_acquisition(case, group):
    matches = [acquisition for acquisition in group["source_acquisitions"]
               if repository_key(acquisition["repository_url"]) == repository_key(case["repository_url"])
               and acquisition["repository_revision"] == case["repository_revision"]
               and acquisition["source_files"] == case["source_files"]]
    if len(matches) != 1:
        raise ValueError("Case URL/revision/full source map does not match a trusted acquisition receipt")
    return matches[0]


def collect_author_source(root):
    """Read a full checkout into an operator acquisition receipt; never execute authors."""
    root = Path(root).resolve(strict=True)

    def git(*args):
        try:
            # Files avoid unbounded in-memory capture. These Git built-ins do not
            # invoke checkout filters, hooks, setup scripts or author imports.
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
                subprocess.run(["git", "--no-replace-objects", *args], cwd=root,
                               stdout=output, stderr=error, timeout=30, check=True)
                output.seek(0)
                content = output.read(MAX_GIT_OUTPUT_BYTES + 1)
                if len(content) > MAX_GIT_OUTPUT_BYTES:
                    raise ValueError("Author Git metadata exceeds the output size bound")
                return content
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("Formal author source requires its original readable Git checkout") from exc

    top = Path(git("rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve(strict=True)
    if top != root:
        raise ValueError("Formal author source root must be the complete Git repository root")
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    tree = git("rev-parse", "HEAD^{tree}").decode("ascii").strip()
    url = repository_key(git("remote", "get-url", "origin").decode("utf-8").strip())
    if not _git_sha(revision) or not _git_sha(tree):
        raise ValueError("Author checkout requires fixed SHA-1 Git identities")
    actual = {}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, sha = header.decode("ascii").split(" ")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("Formal author source cannot contain submodules or source symlinks")
        actual[name.decode("utf-8")] = {"mode": mode, "sha1": sha}
    if not 1 <= len(actual) <= 20000 or len({name.casefold() for name in actual}) != len(actual):
        raise ValueError("Author checkout files exceed limits or collide on Windows")
    files, total = {}, 0
    for name, blob in actual.items():
        path = root
        relative = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("Author checkout contains unsafe frozen file paths")
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Author checkout source paths cannot contain symlinks")
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Author checkout source path resolves outside its repository")
        content = _read_bytes(path, min(MAX_SOURCE_FILE_BYTES, MAX_SOURCE_BYTES - total))
        total += len(content)
        if hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest() != blob["sha1"]:
            raise ValueError("Author source bytes differ from the acquired Git blob")
        files[name] = hashlib.sha256(content).hexdigest()
    if (git("rev-parse", "HEAD").decode("ascii").strip() != revision
            or git("rev-parse", "HEAD^{tree}").decode("ascii").strip() != tree
            or repository_key(git("remote", "get-url", "origin").decode("utf-8").strip()) != url):
        raise ValueError("Author checkout identity changed during source acquisition")
    return {"version": 1, "acquired_at": datetime.now(timezone.utc).isoformat(),
            "repository_url": url, "repository_revision": revision, "git_tree": tree,
            "source_files": files, "git_blobs": actual}


def verify_author_source(manifest, case, root):
    """Require the original full checkout; a raw export alone cannot enter formal trials."""
    admission = validate_admission(manifest)
    if admission is None or case["split"] != "holdout" or case["origin"] != "natural":
        return None
    groups = _validate_audit(admission["audit"], manifest["frozen_at"])
    acquisition = _case_acquisition(case, groups[repository_key(case["repository_url"])])
    current = collect_author_source(root)
    if any(current[key] != acquisition[key] for key in
           ("repository_url", "repository_revision", "git_tree", "source_files", "git_blobs")):
        raise ValueError("Author checkout URL/revision/full source tree differs from its trusted acquisition")
    return source_receipt(manifest, case)


def source_receipt(manifest, case):
    admission = validate_admission(manifest)
    if admission is None or case["split"] != "holdout" or case["origin"] != "natural":
        return None
    groups = _validate_audit(admission["audit"], manifest["frozen_at"])
    acquisition = _case_acquisition(case, groups[repository_key(case["repository_url"])])
    return {"version": 1, "verified": True, "repository_url": repository_key(acquisition["repository_url"]),
            "repository_revision": acquisition["repository_revision"], "git_tree": acquisition["git_tree"],
            "source_files_sha256": digest(acquisition["source_files"]), "acquisition_sha256": digest(acquisition)}
