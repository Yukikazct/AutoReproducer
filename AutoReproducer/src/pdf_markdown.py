"""Convert immutable PDF pages to Markdown while retaining their source identity.

MarkItDown supplies layout, not new paper evidence. Each page must preserve the
numbers and readable characters of the existing extraction; otherwise that
complete original page remains the reading/citation source. Conversion is local,
with no plugins, image LLM, OCR service or URI conversion.
"""
from collections import Counter
import hashlib
from html import escape
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import re
import unicodedata


MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PAGES = 200
MAX_MARKDOWN_CHARS = 1000000
# PDF extraction can join a label to its value or leave a sentence period
# immediately after a decimal. Word/period boundaries would hide those values
# and let a moved decimal point pass the separate readable-character check.
_NUMBER = re.compile(r'[+\-\u2212]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?%?')


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _version():
    try:
        return version('markitdown')
    except PackageNotFoundError:
        return ''


def _converter():
    from markitdown import StreamInfo
    from markitdown.converters import PdfConverter, HtmlConverter
    # The format is already validated. Calling the public format converters
    # avoids creating a Magika/ONNX classifier (and its native thread pool)
    # every time a reader or planner reopens the same PDF. No plugin or LLM
    # converter is registered on this local-only path.
    pdf_converter, html_converter = PdfConverter(), HtmlConverter()
    info = StreamInfo(extension='.pdf', mimetype='application/pdf')

    def convert(payload):
        result = pdf_converter.convert(BytesIO(payload), stream_info=info)
        return result.text_content
    def tables(payload):
        def html_to_markdown(html):
            return html_converter.convert(BytesIO(html.encode('utf-8')),
                stream_info=StreamInfo(extension='.html', mimetype='text/html', charset='utf-8')).text_content
        return _geometry_tables(payload, html_to_markdown)
    convert.tables = tables
    return convert


def _geometry_tables(payload, html_to_markdown):
    """Recover simple ruled result tables without merging adjacent paper columns.

    Only a captioned region bounded by at least three matching horizontal
    rules is eligible. Single-level headers establish physical column bands;
    words crossing those bands, merged cells and unsupported rows reject the
    region. The complete page stays intact alongside this alternative view.
    """
    import pdfplumber
    result = []
    with pdfplumber.open(BytesIO(payload)) as document:
        page = document.pages[0]
        groups = []
        for edge in page.edges:
            if edge['orientation'] != 'h' or edge['x1'] - edge['x0'] < 100:
                continue
            group = next((g for g in groups if abs(g[0] - edge['x0']) < 1
                          and abs(g[1] - edge['x1']) < 1), None)
            if group is None:
                group = [edge['x0'], edge['x1'], []]
                groups.append(group)
            group[2].append(edge['top'])
        for left, right, ys in groups[:32]:
            rules = []
            for y in sorted(ys):
                if not rules or y - rules[-1] > 2:
                    rules.append(y)
            if len(rules) < 3 or not 30 < rules[-1] - rules[0] < page.height * .75:
                continue
            caption = page.within_bbox((max(0, left - 16), max(0, rules[0] - 65),
                                        min(page.width, right + 16), rules[0])).extract_text(x_tolerance=1) or ''
            matches = list(re.finditer(r'\bTable\s+\d+[.:]?\s', caption, re.I))
            if not matches:
                continue
            caption = caption[matches[-1].start():].strip()
            region = page.crop((left, rules[0], right, rules[-1]))
            words = region.extract_words(x_tolerance=1, y_tolerance=2)
            header = [w for w in words if w['bottom'] < rules[1] + .5]
            if not 2 <= len(header) <= 12 or max(w['top'] for w in header) - min(w['top'] for w in header) > 2:
                continue
            header.sort(key=lambda w: w['x0'])
            if any(not re.search(r'[^\W\d]', w['text']) for w in header):
                continue
            centers = [(w['x0'] + w['x1']) / 2 for w in header]
            if any(b - a < 25 for a, b in zip(centers, centers[1:])):
                continue
            named_method = header[0]['text'].casefold() in {'method', 'methods', 'model', 'models'}
            labels = [w['text'] for w in header]
            bounds = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
            if not named_method:
                labels.insert(0, '')  # Preserve the PDF's empty top-left cell.
                bounds.insert(0, centers[0] - (centers[1] - centers[0]) / 2)
            bounds = [left, *bounds, right]
            body = [w for w in words if w not in header]
            rows = []
            for word in sorted(body, key=lambda w: (w['bottom'], w['x0'])):
                row = next((r for r in rows if abs(r[0]['bottom'] - word['bottom']) < 2), None)
                if row is None:
                    rows.append([word])
                else:
                    row.append(word)
            cells, rejected, numeric_rows = [labels], False, 0
            for row in rows:
                row.sort(key=lambda w: w['x0'])
                text = ' '.join(w['text'] for w in row)
                if text.endswith(':') and not re.search(r'\d', text):
                    cells.append([text] + [''] * (len(labels) - 1))
                    continue
                values = [''] * len(labels)
                for word in row:
                    indices = [i for i in range(len(labels))
                               if bounds[i] <= word['x0'] + .5 and word['x1'] <= bounds[i + 1] + .5]
                    if len(indices) != 1:
                        rejected = True
                        break
                    i = indices[0]
                    values[i] += (' ' if values[i] else '') + word['text']
                scalar = r'[+\-\u2212]?(?:\d+(?:\.\d+)?|\.\d+)%?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?%?)?[†‡*]*'
                if (rejected or not re.search(r'[^\W\d]', values[0])
                        or any(not re.fullmatch(scalar, value) for value in values[1:])):
                    rejected = True
                    break
                numeric_rows += 1
                cells.append(values)
            if rejected or numeric_rows < 2 or len(cells) > 100:
                continue
            raw = region.extract_text(x_tolerance=1, y_tolerance=2) or ''
            cell_text = '\n'.join(' '.join(row) for row in cells)
            # Every readable character and numeric token in the bounded region
            # must survive the physical cell assignment exactly once.
            if (_rejection(raw, cell_text) or Counter(c for c in raw if c.isalnum())
                    != Counter(c for c in cell_text if c.isalnum())):
                continue
            html = '<table>' + ''.join('<tr>' + ''.join(
                '<' + ('th' if index == 0 else 'td') + '>' + escape(value) + '</'
                + ('th' if index == 0 else 'td') + '>' for value in row) + '</tr>'
                for index, row in enumerate(cells)) + '</table>'
            markdown = html_to_markdown(html).strip()
            if _rejection(cell_text, markdown):
                continue
            result.append({'bbox': [round(value, 3) for value in region.bbox],
                'caption': caption, 'markdown': markdown, 'region_text_sha256': _sha(raw),
                'markdown_sha256': _sha(markdown), 'rows': numeric_rows,
                'columns': len(labels), 'engine': 'pdfplumber_geometry+markitdown_html'})
            if len(result) >= 8:
                break
    return result


def _page_payloads(payload):
    from PyPDF2 import PdfReader, PdfWriter
    reader = PdfReader(BytesIO(payload))
    if reader.is_encrypted and not reader.decrypt(''):
        raise ValueError('encrypted_pdf')
    if len(reader.pages) > MAX_PAGES:
        raise ValueError('page_limit')
    for page in reader.pages:
        try:
            writer = PdfWriter()
            writer.add_page(page)
            stream = BytesIO()
            writer.write(stream)
            payload = stream.getvalue()
        except Exception:
            # Broken optional objects in one page must not discard successful
            # conversions of other pages or change their original numbering.
            payload = None
        yield payload


def _rejection(original, markdown):
    if not isinstance(markdown, str) or (original.strip() and not markdown.strip()):
        return 'empty_conversion'
    # Table separators/whitespace may change. Numeric spelling, signs, units
    # and multiplicity cannot be silently added, corrected or dropped.
    if Counter(_NUMBER.findall(original)) != Counter(_NUMBER.findall(markdown)):
        return 'numeric_content_changed'
    if _NUMBER.findall(original) != _NUMBER.findall(markdown):
        return 'numeric_order_changed'
    source = Counter(char for char in original if char.isalnum())
    converted = Counter(char for char in markdown if char.isalnum())
    if source - converted:
        return 'readable_content_missing'
    # Inventory alone cannot detect a value moved to a different method or
    # dataset. Formatting may add punctuation/spacing, never permute labels.
    if ''.join(c for c in original if c.isalnum()) != ''.join(c for c in markdown if c.isalnum()):
        return 'readable_content_reordered'
    # Equal digits do not imply an equal result: uncertainty, inequalities,
    # powers and footnotes carry conditions too. Keep their positions relative
    # to the labels/numbers, ignoring only generated Markdown table rules.
    def scientific_sequence(text):
        text = re.sub(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$', '', text, flags=re.M)
        return ''.join(c for c in text if c.isalnum() or c in '+-/%^†‡*‰‱°'
                       or (unicodedata.category(c) in {'Sm', 'Pd'} and c != '|'))
    if scientific_sequence(original) != scientific_sequence(markdown):
        return 'scientific_notation_changed'
    if markdown.count('\ufffd') > original.count('\ufffd'):
        return 'undecoded_characters'
    return ''


def convert_pdf_pages(payload, original_pages):
    """Return canonical page texts plus a serializable conversion receipt.

    Failures are explicit fallbacks, never truncated papers. Cancellation and
    process termination propagate to the existing owned-worker lifecycle.
    """
    originals = tuple(original_pages)
    output = list(originals)
    receipt = {'engine': 'markitdown', 'version': _version(),
               'pdf_sha256': hashlib.sha256(payload).hexdigest(),
               'format': 'markdown', 'status': 'fallback', 'converted_pages': 0,
               'page_count': len(originals), 'pages': []}
    global_reason = ''
    converted = []
    supplemental = {}
    if len(payload) > MAX_PDF_BYTES or len(originals) > MAX_PAGES:
        global_reason = 'conversion_resource_limit'
    else:
        try:
            convert = _converter()
            pages = _page_payloads(payload)
            total = 0
            for number, page in enumerate(pages):
                if number >= len(originals):
                    raise ValueError('page_count_mismatch')
                if page is None:
                    converted.append((originals[number], 'page_serialization_failed'))
                    continue
                try:
                    text = convert(page)
                    if not isinstance(text, str):
                        raise ValueError('non_text_conversion')
                    text = text.replace('\r\n', '\n').replace('\r', '\n').strip('\n\f ')
                    total += len(text)
                    if total > MAX_MARKDOWN_CHARS:
                        raise ValueError('conversion_resource_limit')
                    reason = _rejection(originals[number], text)
                    if reason and callable(getattr(convert, 'tables', None)):
                        try:
                            tables = convert.tables(page)
                            if tables:
                                text = originals[number] + '\n\n<!-- Alternative table views from this PDF page; not additional evidence. -->\n\n'
                                text += '\n\n'.join(table['caption'] + '\n\n' + table['markdown'] for table in tables)
                                if total + len(text) <= MAX_MARKDOWN_CHARS:
                                    total += len(text)
                                    supplemental[number] = tables
                        except Exception:
                            pass  # Preserve the complete page on unsupported geometry.
                    if number in supplemental:
                        converted.append((text, reason))
                        continue
                    converted.append((text if not reason else originals[number], reason))
                except Exception as exc:
                    # Do not persist exception bodies: a malformed PDF can
                    # contain local paths or credentials in parser diagnostics.
                    converted.append((originals[number], 'conversion_failed:' + type(exc).__name__))
            if len(converted) != len(originals):
                raise ValueError('page_count_mismatch')
        except ImportError:
            global_reason = 'markitdown_dependency_unavailable'
        except Exception as exc:
            global_reason = ('page_count_mismatch' if str(exc) == 'page_count_mismatch'
                             else 'conversion_failed:' + type(exc).__name__)
    for number, original in enumerate(originals, 1):
        text, reason = ((original, global_reason) if global_reason else converted[number - 1])
        output[number - 1] = text
        receipt['pages'].append({'page': number, 'status': 'fallback' if reason else 'converted',
            'reason': reason, 'source_text_sha256': _sha(original), 'text_sha256': _sha(text)})
        if not global_reason and number - 1 in supplemental:
            receipt['pages'][-1]['status'] = 'fallback_with_tables'
            receipt['pages'][-1]['tables'] = supplemental[number - 1]
    receipt['converted_pages'] = sum(page['status'] == 'converted' for page in receipt['pages'])
    receipt['table_pages'] = sum(page['status'] == 'fallback_with_tables' for page in receipt['pages'])
    receipt['status'] = ('converted' if originals and receipt['converted_pages'] == len(originals)
                         else 'partial' if receipt['converted_pages'] or receipt['table_pages'] else 'fallback')
    return tuple(output), receipt


def write_markdown_evidence(run_dir, document):
    """Save page-addressable Markdown and its conversion receipt in an owned run."""
    from src.method_adapters import write_json
    from src.safety.paths import filesystem_path, workspace_path
    from pathlib import Path
    path = Path(filesystem_path(workspace_path(run_dir, 'paper_markdown.md', 'paper Markdown')))
    content = '\n\n'.join(f'<!-- PDF page {number} -->\n\n{text}'
                           for number, text in enumerate(document.page_texts, 1))
    # Hash the exact bytes also on Windows; write_text would translate LF to
    # CRLF and make a correct artifact fail its download integrity check.
    from src.source_repair_evidence import _replace_bytes
    _replace_bytes(path, content.encode('utf-8'))
    record = {**document.extraction, 'artifact': path.name,
              'artifact_sha256': _sha(content), 'pdf_sha256': document.sha256}
    write_json(workspace_path(run_dir, 'paper_markdown.json', 'paper Markdown receipt'), record)
    return record


def report_markdown_targets(data, report_path):
    """Copy hash-checked reading evidence beside a portable report."""
    import json
    from pathlib import Path
    from urllib.parse import quote
    from src.safety.paths import workspace_path
    from src.source_repair_evidence import _replace_bytes
    record = data.get('paper_markdown')
    if not record or not report_path:
        return []
    root, report = Path(data['run_dir']), Path(report_path)
    files = {}
    for name in ('paper_markdown.md', 'paper_markdown.json'):
        path = workspace_path(root, name, 'paper Markdown evidence', must_exist=True)
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('论文 Markdown 附件缺失或超限')
        files[name] = path.read_bytes()
    if (hashlib.sha256(files['paper_markdown.md']).hexdigest() != record.get('artifact_sha256')
            or json.loads(files['paper_markdown.json']) != record
            or record.get('pdf_sha256') != (data.get('pdf_input') or {}).get('sha256')):
        raise ValueError('论文 Markdown 附件与本次 PDF 的摘要不一致')
    targets = []
    for name, raw in files.items():
        digest = hashlib.sha256(raw).hexdigest()
        relative = f'{report.stem}_assets/paper_markdown_{digest}{Path(name).suffix}'
        target = workspace_path(report.parent, relative, 'report paper Markdown')
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace_bytes(target, raw)
        label = '论文 Markdown' if name.endswith('.md') else 'PDF 转换依据'
        targets.append(f'[{label}]({quote(relative, safe="/")}) <!-- sha256:{digest} -->')
    return targets


def report_markdown_bytes(report, report_path):
    """Allow only generated, hash-bound Markdown/receipt companions in exports."""
    from pathlib import Path
    from urllib.parse import unquote
    from src.safety.paths import filesystem_path, workspace_path
    pattern = r'^\[(论文 Markdown|PDF 转换依据)\]\(([^\r\n)]+)\) <!-- sha256:([0-9a-f]{64}) -->$'
    root = Path(filesystem_path(Path(report_path).absolute().parent))
    result = {}
    for label, target, digest in re.findall(pattern, report, re.M):
        name = unquote(target)
        relative = Path(name)
        suffix = '.md' if label == '论文 Markdown' else '.json'
        if (len(relative.parts) != 2 or not relative.parts[0].endswith('_assets')
                or relative.parts[1] != f'paper_markdown_{digest}{suffix}'):
            raise ValueError('论文 Markdown 附件路径无效')
        try:
            path = workspace_path(root, name, 'report paper Markdown', must_exist=True)
            if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError('论文 Markdown 附件缺失或超限')
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError('论文 Markdown 附件缺失或无法读取') from exc
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('论文 Markdown 附件哈希不一致')
        result[name] = raw
    return result
