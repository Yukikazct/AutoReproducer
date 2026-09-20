"""报告 Tab 的端到端测试（AppTest 驱动真实 app.py）。

锁死两条性质：
1. **渲染器真的被接线**——报告里的围栏代码块必须经过 `render_markdown`
   变成深色 IDE 面板，而不是退回 `st.markdown` 的灰底 `<pre>`；
2. **内容不截断**——超长执行输出在页面上完整出现，不带「截断展示」标注。

为什么必须走 AppTest：`render_markdown` 的分支逻辑由纯函数测试覆盖
（tests/test_markdown_render.py），但「app.py 到底调没调它」只有把
app.py 真跑起来才能证明——这正是此前多次「单元测试全绿、页面上还是旧的」
那类问题的所在。

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


def test_report_code_is_rendered_as_ide_panel(at_report):
    """围栏代码块必须走深色 IDE 面板（接线断言），不是裸 Markdown。"""
    assert not at_report.exception
    blob = _markdown_values(at_report)

    assert 'class="autorepro-code"' in blob        # 面板外壳
    assert 'class="autorepro-code-bar"' in blob    # IDE 标题栏
    assert "autorepro-highlight" in blob           # pygments 高亮结果
    # 令牌颜色的样式表必须一起下发，否则 token 全是默认色
    assert ".autorepro-highlight .k" in blob
    # 原始围栏不能残留成裸文本（那说明又走回 st.markdown 了）
    assert "```python" not in blob


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
