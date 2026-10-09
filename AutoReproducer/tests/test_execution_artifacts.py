"""Real plot lifecycle, file boundaries, and portable Markdown image bundles."""
from pathlib import Path
import re
import shutil
from urllib.parse import unquote
from unittest.mock import Mock

import pytest
from PIL import Image

import src.execution_artifacts as artifacts
from src.agents.code_executor import CodeExecutorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.llm.llm_client import LLMClient


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "reports" / "artifacts")


def test_real_chinese_plot_survives_workspace_cleanup_and_links_in_report(tmp_path):
    pytest.importorskip("matplotlib")
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    code = """import os
from pathlib import Path
import matplotlib.pyplot as plt
Path('plots').mkdir(exist_ok=True)
plt.plot([1, 2, 3], [3, 1, 4])
plt.title('真实气温与预测值')
plt.xlabel('训练数据')
plt.tight_layout()
plt.savefig('plots/气温.png')
plt.close()
print('WORKDIR=' + os.getcwd())
"""
    result = executor.run({"code": code})
    assert result["success"], result["final"]["stderr"]
    final = result["final"]
    assert "Glyph" not in final["stderr"]
    assert "findfont" not in final["stderr"]
    assert "temporary cache" not in final["stderr"]
    assert len(final["artifacts"]) == 1
    assert result["stages"][0]["artifacts"] == []  # no smoke duplicates
    workdir = final["stdout"].split('WORKDIR=')[1].strip()
    assert not Path(workdir).exists()
    image = Path(final["artifacts"][0]["path"])
    assert image.exists()
    report_path = tmp_path / "reports" / "report.md"
    report = ReportGeneratorAgent(logger=Mock())._build_report(
        {"execution": result}, report_path=report_path)
    assert "### 运行生成的图片" in report
    assert "plots/气温.png" in report
    target = re.search(r"!\[运行结果图 1\]\(([^)]+)\)", report).group(1)
    assert "data:image" not in report
    assert artifacts.verified_report_image_bytes(target, report_path) == image.read_bytes()
    assert (report_path.parent / unquote(target)).is_file()
    assert code in report


def test_nested_images_collected_but_symlinks_and_invalid_images_skipped(tmp_path, require_symlinks):
    workspace = tmp_path / "workspace"
    plots = workspace / "plots"
    plots.mkdir(parents=True)
    Image.new("RGB", (12, 12), "blue").save(plots / "real.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (12, 12), "red").save(outside / "secret.png")
    (workspace / "linked").symlink_to(outside, target_is_directory=True)
    (plots / "link.png").symlink_to(outside / "secret.png")
    (plots / "fake.png").write_text("not an image", encoding="utf-8")
    result = artifacts.collect_images(str(workspace), "full", "20261003_110111")
    assert [a["name"] for a in result["artifacts"]] == ["plots/real.png"]
    assert result["artifact_warnings"]
    assert "20261003_110111" in result["artifacts"][0]["path"]
    artifact = result["artifacts"][0]
    assert artifacts.image_data_url(artifact).startswith("data:image/png;base64,")
    Path(artifact["path"]).write_bytes(b"replaced")
    assert artifacts.image_data_url(artifact) == ""
    artifact["path"] = str(outside / "secret.png")
    assert artifacts.image_data_url(artifact) == ""


def test_docker_runtime_uses_writable_cache_and_noninteractive_backend(tmp_path):
    env = artifacts.prepare_plot_runtime(str(tmp_path), docker=True)
    assert env["MPLCONFIGDIR"].startswith("/tmp/")
    assert env["MPLBACKEND"] == "Agg"
    assert env["PYTHONPATH"] == "/app/.autorepro_plot"
    assert "addfont" in (tmp_path / ".autorepro_plot" / "sitecustomize.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("font_setup", [
    "plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']",
    "plt.rcdefaults()",
    "plt.style.use('classic')",
])
def test_cjk_font_survives_generated_font_overrides(font_setup):
    pytest.importorskip("matplotlib")
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    code = f"""import matplotlib.pyplot as plt
from matplotlib import font_manager
{font_setup}
fig, ax = plt.subplots()
ax.plot([1, 2], [2, 1], label='训练数据')
ax.set_title('中文标题：真实气温与预测值', fontsize=19)
ax.set_xlabel('显式字体：残差分布', fontproperties=font_manager.FontProperties(
    fname=font_manager.findfont('DejaVu Sans'), size=13))
ax.text(1, 1.5, 'English only', fontfamily='DejaVu Sans')
ax.legend()
plt.tight_layout()
fig.savefig('plot.png')
assert ax.title.get_fontproperties().get_file().endswith('NotoSansSC.ttf')
assert ax.title.get_fontsize() == 19
assert ax.xaxis.label.get_fontsize() == 13
assert ax.texts[0].get_fontproperties().get_file() is None
print('CJK layout and rendering verified')
"""
    result = executor.run({"code": code})
    assert result["success"], result["final"]["stderr"]
    assert "Glyph" not in result["final"]["stderr"]
    assert "findfont" not in result["final"]["stderr"]
    assert "CJK layout and rendering verified" in result["final"]["stdout"]
    assert len(result["final"]["artifacts"]) == 1


def test_session_cleanup_includes_retained_figures(tmp_path, monkeypatch):
    import frontend.history_manager as history
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    sid = "20261003_110111"
    folder = tmp_path / "reports" / "artifacts" / f"figures_{sid}_abcd"
    folder.mkdir(parents=True)
    image = folder / "01.png"
    image.write_bytes(b"image")
    assert image in history._related_files(sid)
    removed, _ = history.delete_session(sid)
    assert removed == 1
    assert not image.exists()


def _collected_image(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Image.new("RGB", (12, 12), "blue").save(workspace / "plot.png")
    return artifacts.collect_images(str(workspace), "full")["artifacts"][0]


@pytest.mark.parametrize("destination", [
    "data/runs/repository_run/report.md",
    "data/reports/Paper_20261007.md",
])
def test_each_report_location_has_a_relative_portable_image_bundle(tmp_path, destination):
    artifact = _collected_image(tmp_path)
    original = Path(artifact["path"]).read_bytes()
    report_path = tmp_path / destination
    report = ReportGeneratorAgent(logger=Mock()).run(
        {"execution": {"final": {"artifacts": [artifact]}}},
        report_path=report_path)["report"]
    target = re.search(r"!\[运行结果图 1\]\(([^)]+)\)", report).group(1)
    expected = f"{report_path.stem}_assets/{artifact['sha256']}.png"
    assert target == expected
    assert "data:image" not in report
    assert not Path(target).is_absolute()
    assert (report_path.parent / target).read_bytes() == original
    assert artifacts.verified_image_bytes(artifact) == original
    report_path.write_text(report, encoding="utf-8")

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(report_path, moved / report_path.name)
    shutil.copytree(report_path.parent / f"{report_path.stem}_assets",
                    moved / f"{report_path.stem}_assets")
    Path(artifact["path"]).unlink()
    assert artifacts.verified_report_image_bytes(target, moved / report_path.name) == original


def test_relative_report_path_and_unicode_image_target_ignore_cwd(tmp_path, monkeypatch):
    artifact = _collected_image(tmp_path)
    project_root = tmp_path / "project"
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", project_root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    report_path = "data/reports/中文 report [1].md"
    target = artifacts.markdown_image_target(artifact, report_path)
    assert target.startswith("%E4%B8%AD%E6%96%87%20report%20%5B1%5D_assets/")
    assert artifacts.verified_report_image_bytes(target, report_path) == Path(artifact["path"]).read_bytes()
    assert (project_root / "data" / "reports" / unquote(target)).is_file()
    assert not (elsewhere / "data").exists()


def test_report_default_path_and_data_field_ignore_cwd(tmp_path, monkeypatch):
    artifact = _collected_image(tmp_path)
    project_root = tmp_path / "project"
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", project_root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    data = {"execution": {"final": {"artifacts": [artifact]}}}
    generator = ReportGeneratorAgent(logger=Mock())
    report = generator._build_report(data)
    target = re.search(r"!\[运行结果图 1\]\(([^)]+)\)", report).group(1)
    assert artifacts.default_report_path() == project_root / "data" / "reports" / "report.md"
    assert artifacts.verified_report_image_bytes(target, artifacts.default_report_path())
    assert not (elsewhere / "data").exists()

    other_path = tmp_path / "other" / "different.md"
    report = generator._build_report({**data, "report_path": str(other_path)})
    assert "different_assets/" in report
    assert artifacts.verified_report_image_bytes(
        re.search(r"!\[运行结果图 1\]\(([^)]+)\)", report).group(1), other_path)


def test_report_images_reject_source_changes_and_repair_changed_copies(tmp_path):
    artifact = _collected_image(tmp_path)
    report_path = tmp_path / "output" / "report.md"
    original = Path(artifact["path"]).read_bytes()
    target = artifacts.markdown_image_target(artifact, report_path)
    saved = report_path.parent / target
    saved.write_bytes(b"changed report image")
    assert artifacts.verified_report_image_bytes(target, report_path) is None
    assert artifacts.markdown_image_target(artifact, report_path) == target
    assert saved.read_bytes() == original
    Path(artifact["path"]).write_bytes(b"changed source image")
    assert artifacts.verified_image_bytes(artifact) is None
    assert artifacts.markdown_image_target(artifact, report_path) == ""
    # A report bundle remains readable independently of its source run.
    assert artifacts.verified_report_image_bytes(target, report_path) == original


@pytest.mark.parametrize("target", [
    "../secret.png", "%2e%2e/secret.png", "/tmp/secret.png",
    "report_assets/../../secret.png", "report_assets/not-a-hash.png",
    "data:image/png;base64,AA==", "https://example.test/secret.png",
])
def test_report_image_reader_rejects_unsafe_targets(tmp_path, target):
    assert artifacts.verified_report_image_bytes(target, tmp_path / "report.md") is None


def test_report_image_helpers_reject_asset_directory_symlinks(tmp_path, require_symlinks):
    artifact = _collected_image(tmp_path)
    report_path = tmp_path / "output" / "report.md"
    report_path.parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    name = f"{artifact['sha256']}.png"
    (outside / name).write_bytes(Path(artifact["path"]).read_bytes())
    (report_path.parent / "report_assets").symlink_to(outside, target_is_directory=True)
    assert artifacts.markdown_image_target(artifact, report_path) == ""
    assert artifacts.verified_report_image_bytes(f"report_assets/{name}", report_path) is None


def test_report_image_helpers_reject_symlink_files_and_oversized_images(tmp_path, monkeypatch, require_symlinks):
    artifact = _collected_image(tmp_path)
    report_path = tmp_path / "output" / "report.md"
    target = artifacts.markdown_image_target(artifact, report_path)
    saved = report_path.parent / target
    original = Path(artifact["path"])
    saved.unlink()
    saved.symlink_to(original)
    assert artifacts.markdown_image_target(artifact, report_path) == ""
    assert artifacts.verified_report_image_bytes(target, report_path) is None
    saved.unlink()
    target = artifacts.markdown_image_target(artifact, report_path)
    monkeypatch.setattr(artifacts, "MAX_IMAGE_BYTES", original.stat().st_size - 1)
    assert artifacts.verified_image_bytes(artifact) is None
    assert artifacts.markdown_image_target(artifact, report_path) == ""
    assert artifacts.verified_report_image_bytes(target, report_path) is None


def test_report_marks_unavailable_images_without_embedding_unverified_paths(tmp_path):
    artifact = _collected_image(tmp_path)
    Path(artifact["path"]).write_bytes(b"tampered")
    report = ReportGeneratorAgent(logger=Mock())._build_report(
        {"execution": {"final": {"artifacts": [artifact]}}},
        report_path=tmp_path / "report.md")
    assert "图片文件可能已被清理" in report
    assert "![运行结果图" not in report
    assert artifact["path"] not in report
