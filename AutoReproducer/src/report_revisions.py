"""Versioned, local-only report rendering from an unchanged saved result.

Original reports, result.json and progress/log files are never overwritten.
The manifest is a presentation overlay, committed after its Markdown is saved.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import quote

from src.project_paths import get_data_root


REPORT_FORMAT_VERSION = 5
REPORT_FORMAT_MARKER = f"<!-- autoreproducer-report-format: {REPORT_FORMAT_VERSION} -->"
_VERSION = re.compile(r"^<!-- autoreproducer-report-format: (\d+) -->$", re.MULTILINE)
_PRESENTATION_FIELDS = {"report", "report_path", "report_length", "report_format_version", "report_rendering",
                        "reproduction_conclusion"}
_CREDENTIAL_KEYS = {"apikey", "accesstoken", "refreshtoken", "authorization", "password",
                    "secret", "secretkey", "clientsecret", "bearertoken", "apitoken",
                    "authtoken", "privatekey", "credentials"}


def sanitize_report_result(result):
    """Copy saved results while redacting explicit credential configuration keys.

    This does not rewrite prose, logs, metric names/values or generic token counts.
    It also handles provider-prefixed configuration such as OPENAI_API_KEY.
    """
    def visit(value):
        if isinstance(value, dict):
            copied = {}
            for key, item in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                is_credential = normalized in _CREDENTIAL_KEYS or any(
                    normalized.endswith(suffix) for suffix in ("apikey", "accesstoken", "refreshtoken", "clientsecret"))
                copied[key] = "[REDACTED]" if is_credential else visit(item)
            return copied
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return deepcopy(value)
    return visit(result)


def _is_terminal_result(result):
    if not isinstance(result, dict):
        return False
    data = result.get("data") or {}
    if not isinstance(data, dict):
        return False
    execution = data.get("execution") or {}

    def active(node, *, nested=False):
        if not isinstance(node, dict):
            return False
        if any(node.get(key) is True for key in ("running", "process_alive", "finalization_pending", "completion_pending")):
            return True
        if any(str(node.get(key, "")).lower() in {"running", "pending", "starting", "executing"}
               for key in ("status", "state")):
            return True
        if nested:
            for value in node.values():
                if isinstance(value, dict) and active(value, nested=True):
                    return True
                if isinstance(value, list) and any(active(item, nested=True) for item in value):
                    return True
        return False

    if active(result) or active(data) or active(execution, nested=True):
        return False
    state = result.get("state")
    if state is not None:
        return state in {"COMPLETED", "ERROR"}
    # Old CLI result.json files lacked the wrapper state. A real exit code is
    # terminal evidence; paper/configuration data alone is not.
    final = execution.get("final") or {} if isinstance(execution, dict) else {}
    return (isinstance(final, dict) and type(final.get("exit_code")) is int
            and final.get("process_exited") is not False)


def report_format_version(report):
    match = _VERSION.search(report or "")
    return int(match[1]) if match else 0


def needs_report_upgrade(result):
    report = (result.get("data") or {}).get("report") or ""
    return bool(report) and report_format_version(report) < REPORT_FORMAT_VERSION


def can_regenerate_report(result):
    data = result.get("data") or {} if isinstance(result, dict) else {}
    return (_is_terminal_result(result)
            and any(isinstance(data.get(key), dict) and data[key]
                    for key in ("paper_info", "execution", "validation", "env_config")))


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_digest(result):
    data = {key: value for key, value in (sanitize_report_result(result).get("data") or {}).items()
            if key not in _PRESENTATION_FIELDS}
    return _sha(json.dumps(data, sort_keys=True, ensure_ascii=False))


def _source_path(result, data_root=None):
    data = result.get("data") or {}
    original = (data.get("report_rendering") or {}).get("source_report_path")
    original = original or result.get("report_path") or data.get("report_path")
    if original:
        return Path(original).absolute()
    # A session may have only an in-memory/progress report. Its deterministic
    # location lets a restarted browser find the same rendering without a run.
    return get_data_root(data_root) / "reports" / ("saved_" + _source_digest(result) + ".md")


def _revision_dir(result, data_root=None):
    source = _source_path(result, data_root)
    return source.parent / (source.stem + ".report-revisions")


def _overlay(result, report, path, source):
    updated = deepcopy(result)
    data = updated.setdefault("data", {})
    data.update(report=report, report_path=str(path), report_length=len(report),
                report_format_version=report_format_version(report),
                report_rendering={"source_report_path": str(source), "revision_path": str(path)})
    if data["report_format_version"] >= 4:
        from src.result_conclusion import build_reproduction_conclusion
        data["reproduction_conclusion"] = build_reproduction_conclusion(data)
    updated["report_path"] = str(path)
    return updated


def resolve_report_revision(result, *, data_root=None):
    """Read a matching saved rendering; never regenerate or start any work."""
    if not _is_terminal_result(result):
        return result
    try:
        directory = _revision_dir(result, data_root)
        manifest = json.loads((directory / "current.json").read_text(encoding="utf-8"))
        if manifest["source_digest"] != _source_digest(result):
            return result
        source_report = (result.get("data") or {}).get("report") or ""
        if (_sha(source_report) not in {manifest["source_report_sha256"], manifest["report_sha256"]}
                or Path(manifest["filename"]).name != manifest["filename"]):
            return result
        path = directory / manifest["filename"]
        report = path.read_text(encoding="utf-8")
        if _sha(report) != manifest["report_sha256"]:
            return result
        return _overlay(result, report, path, _source_path(result, data_root))
    except (OSError, ValueError, KeyError, TypeError):
        return result


def regenerate_report(result, *, data_root=None):
    """Render saved data without agents, API calls, execution or status changes."""
    from src.agents import report_generator

    # Streamlit can keep an already imported pre-versioning renderer after the
    # app file changes. Refresh just this template before updating an old tab.
    if getattr(report_generator.ReportGeneratorAgent, "format_version", 0) != REPORT_FORMAT_VERSION:
        import importlib
        report_generator = importlib.reload(report_generator)

    if not can_regenerate_report(result):
        raise ValueError("缺少可重建报告的结构化结果，或任务尚未结束；原报告已保留。")
    current = resolve_report_revision(result, data_root=data_root)
    if not needs_report_upgrade(current):
        return current
    source = _source_path(result, data_root)
    directory = _revision_dir(result, data_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v{REPORT_FORMAT_VERSION}_{uuid.uuid4().hex}.md"
    # This static renderer has no logger, model or executor initialization.
    def artifact_target(artifact, report_path):
        from src.execution_artifacts import markdown_image_target, verified_report_image_bytes

        target = markdown_image_target(artifact, report_path)
        if target:
            return target
        # The original collection can have been cleaned while a downloaded/
        # saved report still owns its verified companion image. Retain it.
        name = str(artifact.get("sha256", "")) + Path(artifact.get("path", "")).suffix.lower()
        old_target = source.stem + "_assets/" + name
        raw = verified_report_image_bytes(old_target, source)
        if raw is None:
            return ""
        assets = path.parent / (path.stem + "_assets")
        assets.mkdir(exist_ok=True)
        (assets / name).write_bytes(raw)
        return quote(assets.name + "/" + name, safe="/")

    report = report_generator.ReportGeneratorAgent.build_report(
        sanitize_report_result(result["data"]), report_path=path, artifact_resolver=artifact_target)
    path.write_text(report, encoding="utf-8")
    original = load_report_revision_source(source)
    if original is None or _source_digest(original) != _source_digest(result):
        original = result
    original = sanitize_report_result(original)
    manifest = {
        "report_format_version": REPORT_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_digest": _source_digest(result),
        "source_report_sha256": _sha(original["data"].get("report") or ""),
        "report_sha256": _sha(report), "filename": path.name,
        "source_report_path": str(source), "source_result": original,
    }
    # Atomic pointer update: a reader sees either the previous complete revision
    # or this complete revision, never a partially written report/manifest.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(manifest, output, ensure_ascii=False, indent=2)
        os.replace(temporary, directory / "current.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _overlay(result, report, path, source)


def load_report_revision_source(report_path):
    """Recover structured input after disposable progress files were cleaned."""
    path = Path(report_path)
    try:
        manifest = json.loads((path.parent / (path.stem + ".report-revisions") /
                               "current.json").read_text(encoding="utf-8"))
        source = manifest["source_result"]
        if (not isinstance(source, dict) or not _is_terminal_result(source)
                or manifest["source_digest"] != _source_digest(source)
                or manifest["source_report_sha256"] != _sha((source.get("data") or {}).get("report") or "")):
            return None
        return sanitize_report_result(source)
    except (OSError, ValueError, KeyError, TypeError):
        return None
