"""Markdown 渲染：把报告里的围栏代码块渲染成深色 IDE 面板。

复现报告是「带代码的文档」——`st.markdown` 渲染出来的围栏代码块只是一块
灰底 `<pre>`：没有语法高亮、没有行号，长输出还看不出边界。这里把报告按
围栏切成「文本段 + 代码段」，文本段仍走 `st.markdown`，代码段改用服务端
pygments 渲染成套壳式深色面板（行号 + Monokai 配色 + 语言标题栏 + 溢出滚动）。

为什么不用 `st.code`（原生就有高亮和行号）：
1. 浅色主题下它的 token 颜色是 react-syntax-highlighter 写死的内联样式，
   光靠 CSS 改不出稳定的深色面板；
2. `st.code` 的 `height` 只有 "content"/固定像素，拿不到「标题栏 + 滚动区」
   这套套壳结构；
3. pygments 的 `HtmlFormatter` **默认转义全部内容**，LLM 生成的代码/输出
   不可能注入 HTML——渲染任意模型输出时这一点是硬要求。

本模块除 `render_markdown`（唯一依赖 Streamlit 的函数）外均为纯函数，
便于单测。
"""
import functools
import html
from typing import List, Tuple

import streamlit as st
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# 代码段默认最大高度（像素）：超出则面板内部滚动，不把页面撑长。
# 报告里 full 阶段的输出动辄上千行，不封顶会让下方内容永远划不到。
DEFAULT_MAX_HEIGHT = 600

# 报告里出现过的语言标签（实测：python 与无标签两种）。映射到 pygments
# 词法分析器名；"text"（含无标签）刻意**不做高亮**——执行输出是终端文本，
# 当纯文本渲染更接近控制台观感，也不去臆测 stderr 里偶发的代码片段。
_LEXER_ALIASES = {
    "python": "python", "py": "python", "python3": "python",
    "bash": "bash", "sh": "bash", "shell": "bash", "console": "bash",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "sql": "sql", "diff": "diff", "text": "text", "txt": "text",
    "": "text",
}

# pygments 给 token 打的是**类名**（`<span class="kn">import</span>`），
# 具体颜色在主题样式表里，必须显式注入——不注入的话满屏都是没上色的 span，
# 看起来「有高亮」其实一个字都没变色。
_PYGMENTS_SELECTOR = ".autorepro-highlight"
_FORMATTER_KWARGS = {
    "style": "monokai",
    "cssclass": _PYGMENTS_SELECTOR.lstrip("."),
    "wrapcode": True,
}


@functools.lru_cache(maxsize=4)
def pygments_css(selector: str = _PYGMENTS_SELECTOR) -> str:
    """主题样式表（monokai），选择器限定在代码块外壳内。

    lru_cache：Streamlit 每次交互都会重跑整个脚本，而这张表有几十条规则、
    内容恒定，不该每轮重建。

    末尾把 monokai 自带的面板底色压成透明：它的 `.autorepro-highlight`
    规则优先级与本项目 `.autorepro-code` 相同、且出现在文档更后面，
    不压掉就会盖住面板底色，标题栏与代码区出现两种深色。
    """
    css = HtmlFormatter(style="monokai").get_style_defs(selector)
    return f"{css}\n{selector} {{ background: transparent; }}"


def _fence_info(line: str) -> Tuple[int, str]:
    """判断该行是否为围栏行：返回 (反引号个数, 语言标签)，非围栏返回 (0, "")。

    CommonMark 允许围栏前有最多 3 个空格缩进、语言标签后可跟空格。
    """
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return 0, ""
    ticks = len(stripped) - len(stripped.lstrip("`"))
    if ticks < 3:
        return 0, ""
    rest = stripped[ticks:].strip()
    return ticks, rest


def split_fenced_blocks(md: str) -> List[Tuple[str, str, str]]:
    """把 Markdown 切成 (kind, lang, text) 段，kind 为 "text" 或 "code"。

    - 围栏内的内容**原样保留**（不做 `strip`），缩进是代码语义的一部分；
    - 结束围栏要求反引号个数 >= 开围栏个数，因此 ```` 裹 ``` 不会提前收尾；
    - 围栏未闭合时把剩余内容整段当作代码：宁可多渲染成代码块，
      也不能因为格式瑕疵丢掉内容（报告是给人看的产物，不是解析器测试）。
    """
    segments: List[Tuple[str, str, str]] = []
    text_buf: List[str] = []
    code_buf: List[str] = []
    lang = ""
    fence_len = 0

    for line in md.splitlines():
        ticks, info = _fence_info(line)
        if fence_len:                       # 代码段内：只找结束围栏
            if ticks >= fence_len:
                segments.append(("code", lang, "\n".join(code_buf)))
                code_buf, lang, fence_len = [], "", 0
            else:
                code_buf.append(line)
            continue
        if ticks:                           # 文本段内：遇到开围栏
            if text_buf:
                segments.append(("text", "", "\n".join(text_buf)))
                text_buf = []
            lang, fence_len = info, ticks
            continue
        text_buf.append(line)

    if fence_len:                           # 未闭合：剩余内容仍按代码渲染
        segments.append(("code", lang, "\n".join(code_buf)))
    if text_buf:
        segments.append(("text", "", "\n".join(text_buf)))
    return segments


def _lexer_for(lang: str):
    name = _LEXER_ALIASES.get((lang or "").strip().lower(), "")
    if not name or name == "text":
        return None
    try:
        return get_lexer_by_name(name)
    except ClassNotFound:
        return None


def code_block_html(code: str, lang: str = "",
                    max_height: int = DEFAULT_MAX_HEIGHT) -> str:
    """把一段代码渲染成深色 IDE 面板 HTML（含语言标题栏与行号）。

    无语言标签或未知语言走纯文本：不加行号——执行输出里带行号会让
    「第几行报错」这类终端信息多一层对不上号的偏移。
    """
    lexer = _lexer_for(lang)
    if lexer is None:
        body = (f'<pre class="autorepro-plain">'
                f'<code>{html.escape(code)}</code></pre>')
    else:
        body = highlight(code, lexer, HtmlFormatter(
            linenos="table", **_FORMATTER_KWARGS))

    label = (lang or "").strip() or "输出"
    return (f'<div class="autorepro-code">'
            f'<div class="autorepro-code-bar">{html.escape(label)}</div>'
            f'<div class="autorepro-code-body" '
            f'style="max-height:{int(max_height)}px;">{body}</div>'
            f'</div>')


def render_markdown(md: str) -> None:
    """渲染带代码块的 Markdown 报告：文本走 st.markdown，代码走深色面板。

    只有真的存在代码段时才注入 pygments 样式表——纯文字报告不该多背
    几十条用不上的 CSS 规则。样式表随正文一起下发，调用方不需要记得
    另外注入（忘了注入的后果是「代码上了色但全是默认色」，很难自查）。
    """
    if not md:
        return
    segments = split_fenced_blocks(md)
    if any(kind == "code" for kind, _, _ in segments):
        st.markdown(f"<style>{pygments_css()}</style>", unsafe_allow_html=True)

    for kind, lang, text in segments:
        if kind == "code":
            st.markdown(code_block_html(text, lang), unsafe_allow_html=True)
        elif text.strip():
            st.markdown(text)
