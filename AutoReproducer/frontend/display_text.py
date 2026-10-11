"""Presentation-only normalization of decorative UI symbols."""
import re


# Enumerate decorations rather than deleting Unicode ranges: mathematical
# operators, arrows and scientific notation must remain readable.
_DECORATIONS = re.compile(
    "[🔬🔗🔌🧪🐳⚠✅🔄🚀⚪📖🔍🔧⚡🏆📝🎉❌📋📄📜📂📊"
    "🟢🔴🟡🟠⬇📡💾🧹📦🗑☑▫⏳🛡📎⏹✳]\\ufe0f?[ \\t]?"
)
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE_LITERAL = re.compile(
    r"(?P<ticks>`+).*?(?P=ticks)"
    r"|\$\$.*?\$\$|(?<!\\)\$[^\n$]+(?<!\\)\$"
    r"|\\\(.*?\\\)|\\\[.*?\\\]"
    r"|!?\[[^\]\n]*\]\([^\n]*?\)"
)


def display_text(value):
    """Remove known decorations from a display label without changing its source."""
    return _DECORATIONS.sub("", str(value))


def report_display_text(markdown):
    """Normalize prose while preserving code, equations and link destinations."""
    lines, fence, equation = [], None, None
    for line in markdown.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        stripped = line.strip()
        if equation:
            lines.append(line)
            if stripped == equation:
                equation = None
            continue
        if stripped in {"$$", r"\["}:
            equation = "$$" if stripped == "$$" else r"\]"
            lines.append(line)
            continue
        cursor, parts = 0, []
        for literal in _INLINE_LITERAL.finditer(line):
            parts.extend((display_text(line[cursor:literal.start()]), literal.group()))
            cursor = literal.end()
        parts.append(display_text(line[cursor:]))
        lines.append("".join(parts))
    return "".join(lines)
