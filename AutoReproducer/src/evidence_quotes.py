"""Recover a model's quoted text only by locating an original source span."""
import re


_ESCAPES = {'n': '\n', 'r': '\r', 't': '\t', '"': '"', "'": "'", '\\': '\\'}
_LIGATURES = dict(zip('ﬀﬁﬂﬃﬄﬅﬆ', ('ff', 'fi', 'fl', 'ffi', 'ffl', 'st', 'st')))
_WRAP = re.compile(r'(?<=[A-Za-z]{2})-[ \t]*\r?\n[ \t]*(?=[A-Za-z]{2})')


def _pdf_search_text(text):
    """Search typography without losing the offsets of the original glyphs."""
    omitted = {index for match in _WRAP.finditer(text) for index in range(*match.span())}
    characters, offsets = [], []
    for index, char in enumerate(text):
        if index in omitted:
            continue
        replacement = _LIGATURES.get(char, char)
        characters.extend(replacement)
        offsets.extend([index] * len(replacement))
    return ''.join(characters), offsets


def original_quote_spans(quote, originals, *, pdf_wrapping=False):
    """Return exact source spans, allowing whitespace or one extra escape layer.

    Literal text takes precedence, so a quoted Python string containing ``\\n``
    keeps its meaning. Never decode Unicode/hex escapes or synthesize evidence.
    PDF-only matching can join alphabetic line-end wraps and expand printing
    ligatures; returned evidence still contains the original glyphs and breaks.
    Code, numbers and inline hyphenated names are not rewritten for matching.
    Distinct recovered spans are ambiguous.
    """
    originals = list(originals)
    literal = [(quote, identity) for text, identity in originals if quote in text]
    if literal:
        return literal
    decoded = re.sub(r'''\\([nrt"'\\])''', lambda match: _ESCAPES[match[1]], quote)
    matches = []
    for candidate in dict.fromkeys((quote, decoded)):
        parts = candidate.split()
        if not parts:
            continue
        pattern = r'\s+'.join(re.escape(part) for part in parts)
        for text, identity in originals:
            for match in re.finditer(pattern, text):
                item = (match.group(), identity)
                if item not in matches:
                    matches.append(item)
    if not matches and pdf_wrapping:
        for candidate in dict.fromkeys((quote, decoded)):
            search, _ = _pdf_search_text(candidate)
            if not search.split():
                continue
            pattern = r'\s+'.join(re.escape(part) for part in search.split())
            for original, identity in originals:
                folded, offsets = _pdf_search_text(original)
                for match in re.finditer(pattern, folded):
                    # A proposal cannot select half of a multi-character glyph.
                    start, end = match.span()
                    if ((start and offsets[start - 1] == offsets[start])
                            or (end < len(offsets) and offsets[end - 1] == offsets[end])):
                        continue
                    item = (original[offsets[start]:offsets[end - 1] + 1], identity)
                    if item not in matches:
                        matches.append(item)
    return matches if len({span for span, _ in matches}) == 1 else []
