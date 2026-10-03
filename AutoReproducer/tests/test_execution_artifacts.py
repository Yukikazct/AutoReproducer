"""Real plot lifecycle, sandbox file boundaries, and self-contained reports."""
import base64
from pathlib import Path
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


def test_real_chinese_plot_survives_workspace_cleanup_and_embeds_in_report():
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
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": result})
    assert "### 运行生成的图片" in report
    assert "plots/气温.png" in report
    encoded = report.split("data:image/png;base64,", 1)[1].split(")", 1)[0]
    assert base64.b64decode(encoded) == image.read_bytes()
    assert code in report


def test_nested_images_collected_but_symlinks_and_invalid_images_skipped(tmp_path):
    workspace = tmp_path / "workspace"
    plots = workspace / "plots"
    plots.mkdir(parents=True)
    Image.new("RGB", (12, 12), "blue").save(plots / "real.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (12, 12), "red").save(outside / "secret.png")
    (workspace / "linked").symlink_to(outside, target_is_directory=True)
    (plots / "link.png").symlink_to(outside / "secret.png")
    (plots / "fake.png").write_text("not an image")
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
    assert "addfont" in (tmp_path / ".autorepro_plot" / "sitecustomize.py").read_text()


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
