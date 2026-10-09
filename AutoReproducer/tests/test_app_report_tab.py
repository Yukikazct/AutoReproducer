"""报告 Tab 的端到端测试（AppTest 驱动真实 app.py）。

锁死两条性质：
1. **报告真的上屏**——`session_state.result["data"]["report"]` 经**原生
   `st.markdown`** 渲染。这条线此前断过：自绘深色面板（[2026.09.20-10]）
   的单元测试与预览页全绿，用户打开页面却「看不到代码」，见
   [2026.09.20-12] 的回滚。断言「围栏原样出现在 markdown 值里」正是
   「没有中间层再吃掉它」的证据；
2. **内容不截断**——超长执行输出在页面上完整出现，不带「截断展示」标注。

为什么必须走 AppTest：单元测试能证明 `st.markdown(report)` 这一行本身没
问题，但证明不了 app.py 的那个分支真的被执行到（Tab 2 的渲染在
`st.session_state.result` 有值时才会走）。

数据目录经 monkeypatch 指向 tmp_path，不触碰真实 data/。

运行: python -m pytest tests/test_app_report_tab.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import frontend.history_manager as hm  # noqa: E402

APP_PATH = str(Path(__file__).parent.parent / "app.py")

CODE = "import numpy as np\n\ndef main():\n    print('ok')\n"
TAIL = "final_line_reached = True"
LONG_STDOUT = "\n".join(f"step {i} done" for i in range(500)) + "\n" + TAIL
REPORT = ("# 论文复现与优化报告\n\n## 1. 论文信息\n\n- **标题**: 测试论文\n\n"
          "### 生成代码\n```python\n" + CODE + "```\n\n"
          "### 执行输出(full)\n```\n" + LONG_STDOUT + "\n```\n")


@pytest.fixture
def at_report(tmp_path, monkeypatch):
    """预置一次「已完成」的复现结果，返回已渲染的 AppTest。"""
    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)

    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["result"] = {
        "state": "COMPLETED",
        "data": {"report": REPORT},
    }
    app.run()
    return app


def _markdown_values(app) -> str:
    return "\n".join(m.value for m in app.markdown)


def test_report_is_rendered_through_native_markdown(at_report):
    """报告经原生 st.markdown 上屏：围栏原样保留，无自绘面板残留。"""
    assert not at_report.exception
    blob = _markdown_values(at_report)

    assert "# 论文复现与优化报告" in blob
    assert "```python" in blob                     # 原生渲染保留围栏
    assert "autorepro-code" not in blob            # 自绘面板已回滚，不得复活


def test_report_body_still_markdown(at_report):
    """正文仍按 Markdown 渲染（标题不能变成代码块）。"""
    assert not at_report.exception
    blob = _markdown_values(at_report)
    assert "# 论文复现与优化报告" in blob
    assert "**标题**: 测试论文" in blob


def test_long_output_is_not_truncated_on_page(at_report):
    """500 行输出在页面上完整呈现，无「截断」标注。"""
    assert not at_report.exception
    blob = _markdown_values(at_report)

    assert "step 0 done" in blob
    assert "step 499 done" in blob
    assert TAIL in blob                            # 末行必须在
    assert "截断" not in blob


def test_server_restart_restores_completed_report(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend.backend_pipeline import ProgressStore
    from src.agents.report_generator import ReportGeneratorAgent

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    progress = tmp_path / "finished.jsonl"
    store = ProgressStore(str(progress))
    report = ReportGeneratorAgent()._build_report({
        "env_config": {"python_version": "3.11", "requirements_txt": "numpy\nmatplotlib",
                       "estimated_disk_gb": 1.5},
        "execution": {"final": {"disk_usage": {
            "status": "measured", "components": [{
                "label": "隔离依赖文件体积", "status": "measured", "bytes": 172216759,
                "basis": "file_stat", "retained": False}]}}},
    })
    store.emit({"type": "done", "result": {
        "state": "COMPLETED", "data": {"report": report}}})
    monkeypatch.setenv("AUTOREPRO_RESUME_PROGRESS", str(progress))
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    assert not app.exception
    assert app.session_state.progress_file == str(progress)
    assert app.session_state.running is False
    blob = _markdown_values(app)
    assert "预估磁盘" not in blob
    assert "164.24 MiB" in blob
    assert "**依赖数**: 2" in blob


def test_server_restart_does_not_resume_unfinished_job(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend.backend_pipeline import ProgressStore

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    progress = tmp_path / "unfinished.jsonl"
    ProgressStore(str(progress)).emit({"type": "state", "state": "EXECUTE_CODE"})
    monkeypatch.setenv("AUTOREPRO_RESUME_PROGRESS", str(progress))
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    assert not app.exception
    assert app.session_state.progress_file is None
    assert app.session_state.result is None


def test_saved_report_figures_are_rendered_as_native_images(tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from src import execution_artifacts
    from src.agents.report_generator import ReportGeneratorAgent

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    root = tmp_path / "collected"
    root.mkdir()
    monkeypatch.setattr(execution_artifacts, "ARTIFACT_ROOT", root)
    image = root / "forecast.png"
    Image.new("RGB", (30, 20), "orange").save(image)
    artifact = {"name": "test_forecast.png", "path": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "bytes": image.stat().st_size}
    report_path = tmp_path / "report.md"
    report = ReportGeneratorAgent()._build_report({
        "execution": {"final": {"success": True, "artifacts": [artifact]}},
    }, report_path=report_path)
    report_path.write_text(report, encoding="utf-8")
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["result"] = {
        "state": "COMPLETED", "report_path": str(report_path), "data": {"report": report},
    }
    app.run()
    assert not app.exception
    assert len(app.image) == 1
    assert app.image[0].captions == ["运行结果图 1"]
    assert not any("无法读取" in warning.value for warning in app.warning)
    # 下载入口位于按需加载的历史页，先进入该页再检查打包下载。
    import inspect
    import streamlit as st
    if "on_change" in inspect.signature(st.tabs).parameters:
        app.session_state["workspace_tabs"] = "📂 历史记录"
        app.run()
    else:
        app.button(key="history_load").click().run()
    assert not app.exception
    assert any("报告和图片" in button.label for button in app.get("download_button"))
