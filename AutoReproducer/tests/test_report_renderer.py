import base64
import hashlib
import io
import shutil
import zipfile
from pathlib import Path
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
    report_path.write_text(text)
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
        assert len(archive.namelist()) == 2
        archive.extractall(elsewhere)
    original.unlink()
    shutil.rmtree(path.parent)
    st = Mock()
    render_report((elsewhere / path.name).read_text(), elsewhere / path.name, st_module=st)
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
