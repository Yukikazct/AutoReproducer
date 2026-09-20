"""报告 Markdown 渲染单元测试：围栏切分 + 代码块深色面板 HTML。

只测纯函数（`split_fenced_blocks` / `code_block_html`），不启动 Streamlit
——`render_markdown` 的接线由 AppTest 覆盖（tests/test_app_report_tab.py）。

运行: python -m pytest tests/test_markdown_render.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.markdown_render import (  # noqa: E402
    code_block_html,
    split_fenced_blocks,
)


# ---------------- 围栏切分 ----------------

def test_text_only_has_single_segment():
    segs = split_fenced_blocks("# 标题\n\n正文一\n正文二")
    assert segs == [("text", "", "# 标题\n\n正文一\n正文二")]


def test_text_and_code_alternate():
    md = "前文\n```python\nprint(1)\n```\n后文"
    assert split_fenced_blocks(md) == [
        ("text", "", "前文"),
        ("code", "python", "print(1)"),
        ("text", "", "后文"),
    ]


def test_lang_and_trailing_space_are_tolerated():
    segs = split_fenced_blocks("```python   \nprint(1)\n```")
    assert segs == [("code", "python", "print(1)")]


def test_untagged_fence_keeps_empty_lang():
    """执行输出的围栏没有语言标签——必须仍被识别为代码段。"""
    segs = split_fenced_blocks("```\nTraining complete.\n```")
    assert segs == [("code", "", "Training complete.")]


def test_indented_fence_is_not_a_fence():
    """4 空格缩进是代码块语法、不是围栏，不得在这里被切开。"""
    md = "    ```python\n    print(1)\n    ```"
    assert split_fenced_blocks(md) == [("text", "", md)]


def test_unclosed_fence_keeps_everything_as_code():
    """围栏没闭合时剩余内容整段按代码渲染——绝不能丢内容。"""
    segs = split_fenced_blocks("前文\n```python\nprint(1)\nprint(2)")
    assert segs == [("text", "", "前文"),
                    ("code", "python", "print(1)\nprint(2)")]


def test_longer_fence_swallows_inner_fence():
    """```` 裹 ``` 时内层反引号不是结束围栏（否则代码会被拦腰截断）。"""
    md = "````\n```python\nprint(1)\n```\n````"
    segs = split_fenced_blocks(md)
    assert len(segs) == 1
    kind, _, text = segs[0]
    assert kind == "code"
    assert "```python" in text and "print(1)" in text


def test_crlf_input_does_not_leak_carriage_returns():
    segs = split_fenced_blocks("前文\r\n```python\r\nprint(1)\r\n```")
    assert segs == [("text", "", "前文"), ("code", "python", "print(1)")]


def test_real_report_shape_round_trips_content():
    """真实报告形状：文本 + requirements 围栏 + python 围栏 + 输出围栏。

    所有非空行必须原样出现在切分结果里（重建后逐行比对），
    避免切分器静默吞行。
    """
    md = ("# 报告\n\n## 3. 环境配置\n\n### requirements.txt\n```\n"
          "numpy>=1.24.0\ntqdm>=4.65.0\n```\n\n"
          "## 4. 代码执行\n\n### 生成代码\n```python\nimport numpy as np\n"
          "print('ok')\n```\n\n### 执行输出(full)\n```\nok\n\n```")
    segs = split_fenced_blocks(md)
    rebuilt = "\n".join(text for _, _, text in segs)
    for line in ("# 报告", "numpy>=1.24.0", "import numpy as np", "print('ok')"):
        assert line in rebuilt


# ---------------- 代码块 HTML ----------------

def test_python_block_is_highlighted_with_line_numbers():
    html = code_block_html("import os\nprint(os.getcwd())", "python")
    assert 'class="autorepro-code"' in html          # 深色面板外壳
    assert 'class="autorepro-code-bar"' in html      # IDE 标题栏
    assert ">python<" in html                        # 标题栏写语言名
    assert "autorepro-highlight" in html
    assert "<span" in html                           # pygments 着色
    assert "linenos" in html or "linenodiv" in html  # 行号


def test_output_block_has_no_line_numbers():
    """无语言标签（执行输出）不加行号：终端里报错行号不该被平移。"""
    html = code_block_html("Training complete.\nFinal loss: 0.31", "")
    assert "autorepro-plain" in html
    assert "linenos" not in html
    assert "Training complete." in html


def test_unknown_language_falls_back_to_plain():
    html = code_block_html("some text", "brainfuck-not-a-lexer")
    assert "autorepro-plain" in html
    assert "brainfuck-not-a-lexer" in html           # 标题栏仍如实标注


def test_code_content_is_html_escaped():
    """LLM 生成的内容必须被转义：报告渲染的是任意模型输出，不能注入 HTML。"""
    html = code_block_html('print("<script>alert(1)</script>")', "python")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "&#39;&lt;script&gt;" in html


def test_plain_output_content_is_html_escaped():
    html = code_block_html("<script>alert(1)</script>", "")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_max_height_is_applied_to_scroll_area():
    html = code_block_html("x = 1", "python", max_height=123)
    assert "max-height:123px" in html


def test_long_output_is_not_truncated_anywhere():
    """渲染器本身不截断：一万字符的输出必须一字不少地出现在 HTML 里。"""
    out = "\n".join(f"line {i} <value>" for i in range(1000))
    html = code_block_html(out, "")
    assert f"line 999" in html
    assert "&lt;value&gt;" in html
    assert "截断" not in html


# ---------------- 主题样式表 ----------------

def test_pygments_css_defines_token_colors():
    """pygments 只给 token 打类名，**颜色在样式表里**。

    不注入这张表的话，满屏 `<span class="kn">` 一个颜色都不变——
    看起来「有高亮」，其实等于没有。这是最容易漏、也最难自查的一步。
    """
    from frontend.markdown_render import pygments_css

    css = pygments_css()
    assert ".autorepro-highlight .k" in css          # 关键字规则
    assert "#" in css and "color:" in css            # 至少一条颜色声明


def test_pygments_css_is_scoped_to_code_blocks():
    """样式必须限定在代码块外壳内，不能污染页面其它 Markdown 元素。"""
    from frontend.markdown_render import pygments_css

    for line in pygments_css().splitlines():
        rule = line.strip()
        if rule.endswith("{") and not rule.startswith(("@", "}")):
            assert rule.startswith(".autorepro-highlight"), rule


def test_pygments_css_overrides_monokai_background():
    """monokai 自带面板底色会盖住本项目的外壳配色，必须显式压成透明。"""
    from frontend.markdown_render import pygments_css

    assert "background: transparent" in pygments_css()
