import base64
import hashlib
import io
import shutil
import zipfile
from pathlib import Path
from urllib.parse import unquote
from unittest.mock import Mock

import pytest
from PIL import Image

from src import execution_artifacts
from src.agents.report_generator import ReportGeneratorAgent
from frontend.report_renderer import (build_report_bundle, render_report,
                                      report_image_bytes, report_parts)


@pytest.fixture
def saved_report(tmp_path, monkeypatch):
    root = tmp_path / "collected"
    root.mkdir()
    monkeypatch.setattr(execution_artifacts, "ARTIFACT_ROOT", root)
    source = root / "actual.png"
    Image.new("RGB", (30, 20), "orange").save(source)
    artifact = {"name": "test_forecast.png", "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}
    report_path = tmp_path / "报告目录" / "真实实验报告.md"
    data = {"execution": {"final": {"success": True, "exit_code": 0, "artifacts": [artifact],
                                    "stdout": "last_line_of_real_execution"}}}
    text = ReportGeneratorAgent()._build_report(data, report_path=report_path)
    report_path.write_text(text, encoding="utf-8")
    return report_path, text, source


def test_web_report_serves_verified_image_bytes_instead_of_relative_browser_url(saved_report):
    path, text, image = saved_report
    st = Mock()
    render_report(text, path, st_module=st)
    st.image.assert_called_once_with(image.read_bytes(), caption="运行结果图 1")
    st.warning.assert_not_called()
    assert "last_line_of_real_execution" in "".join(call.args[0] for call in st.markdown.call_args_list)
    assert all("![运行结果图" not in call.args[0] for call in st.markdown.call_args_list)


def test_report_bundle_remains_readable_after_moving_and_removing_original_images(saved_report, tmp_path):
    path, text, original = saved_report
    zipped = build_report_bundle(text, path)
    elsewhere = tmp_path / "另一台电脑"
    elsewhere.mkdir()
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        assert set(archive.namelist()) == {
            path.name, path.stem + '.manifest.json',
            unquote(next(part['target'] for part in report_parts(text) if part['type'] == 'image')),
        }
        archive.extractall(elsewhere)
    original.unlink()
    shutil.rmtree(path.parent)
    st = Mock()
    render_report((elsewhere / path.name).read_text(encoding="utf-8"), elsewhere / path.name, st_module=st)
    st.image.assert_called_once()
    st.warning.assert_not_called()


def test_missing_or_modified_image_is_reported_and_not_packaged(saved_report):
    path, text, original = saved_report
    companion = next(path.parent.glob("*_assets/*.png"))
    companion.write_bytes(b"corrupt image")
    st = Mock()
    render_report(text, path, st_module=st)
    st.image.assert_not_called()
    st.warning.assert_called_once()
    with pytest.raises(ValueError, match="校验失败"):
        build_report_bundle(text, path)


def test_image_syntax_inside_code_is_kept_as_literal_markdown():
    text = "# 原生报告\n\n```python\n![运行结果图 1](private.png)\n```\n\n尾部\n"
    assert list(report_parts(text)) == [{"type": "markdown", "text": text}]


def test_unsafe_report_image_does_not_read_arbitrary_local_file(tmp_path):
    path = tmp_path / "report.md"
    private = tmp_path / "private.png"
    private.write_bytes(b"private file must not be read")
    text = "![运行结果图 1](private.png)\n"
    st = Mock()
    render_report(text, path, st_module=st)
    st.image.assert_not_called()
    st.warning.assert_called_once()


def test_valid_legacy_embedded_image_still_renders_without_companion_files(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(buffer, format="PNG")
    raw = buffer.getvalue()
    target = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    st = Mock()
    render_report(f"![运行结果图 1]({target})\n", tmp_path / "legacy.md", st_module=st)
    st.image.assert_called_once_with(raw, caption="运行结果图 1")
    st.warning.assert_not_called()


def test_legacy_image_decompression_bomb_is_rejected_without_breaking_report(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    target = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    # Exercise Pillow's actual pixel-count guard without allocating a huge image.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    path = tmp_path / "legacy.md"
    assert report_image_bytes(target, path) is None
    st = Mock()
    render_report(f"![运行结果图 1]({target})\n\nReport continues here.\n", path, st_module=st)
    st.image.assert_not_called()
    st.warning.assert_called_once()
    assert any("Report continues here." in call.args[0] for call in st.markdown.call_args_list)


def test_malformed_legacy_base64_is_rejected(tmp_path):
    assert report_image_bytes("data:image/png;base64,====", tmp_path / "legacy.md") is None


def test_report_display_removes_decorations_without_altering_evidence_or_download(tmp_path):
    report = (
        "# 📋 实验报告\n\n- **✅ 结果**: 通过\n"
        "- ⚠️ 限制: 仅此实验\n\n"
        "```python\nprint('✅ 原始执行输出')\n```\n\n"
        "~~~text\n⚠️ 原始诊断\n~~~\n\n"
        "`checkpoint_✅.pt` · $x \\in \\mathbb{R}, x ≥ 0$\n"
        "$$\n\\text{✅} = 1\n$$\n\n"
        "[源码](https://example.org/✅/code)\n"
    )
    st = Mock()
    path = tmp_path / "report.md"
    render_report(report, path, st_module=st)
    displayed = "".join(call.args[0] for call in st.markdown.call_args_list)
    assert "# 实验报告" in displayed
    assert "**结果**: 通过" in displayed
    assert "- 限制: 仅此实验" in displayed
    for evidence in (
        "print('✅ 原始执行输出')", "⚠️ 原始诊断", "`checkpoint_✅.pt`",
        "$x \\in \\mathbb{R}, x ≥ 0$", "\\text{✅} = 1",
        "[源码](https://example.org/✅/code)",
    ):
        assert evidence in displayed
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(report, path))) as archive:
        assert archive.read("report.md").decode("utf-8") == report


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_bundle_rejects_unreadable_declared_execution_attachment(tmp_path, kind):
    report = tmp_path / "report.md"
    attachment = tmp_path / "report_execution.txt"
    text = "> 📎 **完整执行输出附件**: `report_execution.txt`（同一目录下，未被截断）\n"
    if kind == "directory":
        attachment.mkdir()
    with pytest.raises(ValueError, match="完整执行输出附件缺失或无法读取"):
        build_report_bundle(text, report)


def test_bundle_resolves_relative_report_and_includes_full_execution_output(tmp_path, monkeypatch):
    from frontend import report_renderer

    monkeypatch.setattr(report_renderer, "PROJECT_ROOT", tmp_path)
    directory = tmp_path / "reports"
    directory.mkdir()
    raw = "unabridged author output\n准确率: 0.845\n".encode("utf-8")
    (directory / "report_execution.txt").write_bytes(raw)
    text = "> 📎 **完整执行输出附件**: `report_execution.txt`（同一目录下，未被截断）\n"
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(text, Path("reports/report.md")))) as archive:
        assert archive.read("report_execution.txt") == raw
        assert archive.read("report.md").decode("utf-8") == text
