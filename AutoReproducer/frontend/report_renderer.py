"""Native report rendering and portable Markdown/image downloads."""
import base64
import binascii
import io
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

from frontend.display_text import report_display_text
from src.execution_artifacts import (PROJECT_ROOT, MAX_IMAGE_BYTES,
                                     verified_report_image_bytes)
from src.report_revisions import resolve_report_revision


DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "report.md"
_IMAGE = re.compile(r"^!\[运行结果图 (\d+)\]\(([^\r\n)]+)\)\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def report_parts(report):
    """Separate generated figure rows without interpreting fenced code as images."""
    text, fence = [], None
    for line in report.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        match = _IMAGE.fullmatch(line.strip()) if fence is None and not marker else None
        if match:
            if text:
                yield {"type": "markdown", "text": "".join(text)}
                text = []
            yield {"type": "image", "number": int(match[1]), "target": match[2]}
        else:
            text.append(line)
    if text:
        yield {"type": "markdown", "text": "".join(text)}


def report_image_bytes(target, report_path):
    # Old saved reports may contain data URLs. Decode those locally instead of
    # asking a browser/Markdown preview to resolve them as filesystem paths.
    if target.startswith("data:"):
        match = re.fullmatch(r"data:(image/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/=]+)", target)
        if not match or len(match[2]) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
            return None
        from PIL import Image
        try:
            raw = base64.b64decode(match[2], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            return raw if len(raw) <= MAX_IMAGE_BYTES else None
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
            return None
    return verified_report_image_bytes(target, report_path)


def render_report(report, report_path=None, *, st_module=None):
    """Keep text/code as native Markdown and serve verified images via st.image."""
    if st_module is None:
        import streamlit as st_module
    report_path = report_path or DEFAULT_REPORT_PATH
    for part in report_parts(report):
        if part["type"] == "markdown":
            st_module.markdown(report_display_text(part["text"]))
        else:
            raw = report_image_bytes(part["target"], report_path)
            if raw is None:
                st_module.warning(f"图 {part['number']} 无法读取，请检查报告的图片目录。")
            else:
                st_module.image(raw, caption=f"运行结果图 {part['number']}")


def build_report_bundle(report, report_path):
    """Package a report and its verified companion assets for offline viewing."""
    report_file = Path(report_path)
    figures = {}
    for part in report_parts(report):
        if part["type"] != "image" or part["target"].startswith("data:"):
            continue
        target = part["target"]
        raw = verified_report_image_bytes(target, report_file)
        if raw is None:
            raise ValueError(f"图 {part['number']} 的报告图片缺失或校验失败")
        figures[unquote(target)] = raw
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(report_file.name, report)
        for target, raw in figures.items():
            archive.writestr(target, raw)
    return buffer.getvalue()


def report_downloads(result, *, key, st_module=None, data_root=None):
    """Use the same persisted report revision for Markdown and ZIP downloads."""
    if st_module is None:
        import streamlit as st_module
    current = resolve_report_revision(result, data_root=data_root)
    data = current.get("data") or {}
    report = data.get("report") or ""
    if not report:
        return
    path = Path(current.get("report_path") or data.get("report_path") or DEFAULT_REPORT_PATH)
    st_module.download_button("下载报告", data=report, file_name=path.name,
                              mime="text/markdown", key=f"{key}_markdown")
    try:
        st_module.download_button("下载报告和图片（ZIP）", data=build_report_bundle(report, path),
                                  file_name=path.stem + ".zip", mime="application/zip",
                                  key=f"{key}_zip")
    except ValueError as exc:
        st_module.warning(str(exc))
