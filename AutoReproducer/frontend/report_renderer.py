"""Native report rendering and portable Markdown/image downloads."""
import base64
import binascii
import io
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

from src.execution_artifacts import (PROJECT_ROOT, MAX_IMAGE_BYTES,
                                     verified_report_image_bytes)


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
            st_module.markdown(part["text"])
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
