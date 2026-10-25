"""Markdown is used by real reader/citation paths without losing PDF evidence."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest

from discovered_fixtures import make_experiment, write_pdf
from src import pdf_markdown as markdown
from src.pdf_input import extract_pdf_input, supported_pdf_title
from src.discovered_repository_plan import build_evidence_packet, _planning_excerpts
from src.author_table_evidence import locate_table_value


def converter(monkeypatch, outputs):
    def factory():
        values = iter(outputs)
        def convert(_payload):
            value = next(values)
            if isinstance(value, BaseException):
                raise value
            return value
        return convert
    monkeypatch.setattr(markdown, '_converter', factory)
    monkeypatch.setattr(markdown, '_page_payloads', lambda _: iter([b'page'] * len(outputs)))


def test_page_conversion_keeps_numbers_source_hashes_and_complete_fallback(monkeypatch):
    originals = ('Method Alpha Beta\nSignal 81.5 79.4', 'Appendix details 12')
    table = '| Method | Alpha | Beta |\n| --- | --- | --- |\n| Signal | 81.5 | 79.4 |'
    converter(monkeypatch, [table, 'Appendix 12'])
    pages, receipt = markdown.convert_pdf_pages(b'original PDF bytes', originals)
    assert pages == (table, originals[1])
    assert receipt['status'] == 'partial' and receipt['converted_pages'] == 1
    assert receipt['pages'][1]['reason'] == 'readable_content_missing'
    assert receipt['pdf_sha256'] == hashlib.sha256(b'original PDF bytes').hexdigest()
    assert receipt['pages'][0]['text_sha256'] == hashlib.sha256(table.encode()).hexdigest()


@pytest.mark.parametrize('converted', ['Signal 91.5 79.4', 'Signal 81.5', 'Signal 81.5 79.4 81.5'])
def test_changed_missing_or_duplicated_numbers_do_not_replace_original(monkeypatch, converted):
    original = 'Signal 81.5 79.4'
    converter(monkeypatch, [converted])
    pages, receipt = markdown.convert_pdf_pages(b'pdf', [original])
    assert pages == (original,)
    assert receipt['pages'][0]['reason'] == 'numeric_content_changed'


@pytest.mark.parametrize('original,converted', [
    ('acc81.5', 'acc8.15'),
    ('result 81.5.', 'result 8.15.'),
    ('coefficient1e-3', 'coefficient1e3'),
    ('x.125', 'x1.25'),
    ('version2.5.1 score81.5', 'version2.5.1 score8.15'),
])
def test_joined_labels_and_sentence_periods_cannot_hide_changed_numbers(monkeypatch, original, converted):
    converter(monkeypatch, [converted])
    pages, receipt = markdown.convert_pdf_pages(b'pdf', [original])
    assert pages == (original,)
    assert receipt['pages'][0]['reason'] == 'numeric_content_changed'


@pytest.mark.parametrize('converted,reason', [
    ('Signal 79.4 81.5', 'numeric_order_changed'),
    ('Baseline 81.5\nSignal 79.4', 'readable_content_reordered'),
])
def test_same_number_inventory_cannot_move_values_to_other_labels(monkeypatch, converted, reason):
    original = 'Signal 81.5\nBaseline 79.4' if reason == 'readable_content_reordered' else 'Signal 81.5 79.4'
    converter(monkeypatch, [converted])
    pages, receipt = markdown.convert_pdf_pages(b'pdf', [original])
    assert pages == (original,)
    assert receipt['pages'][0]['reason'] == reason


def test_dependency_failure_keeps_complete_pages_without_parser_error_leaks(monkeypatch):
    monkeypatch.setattr(markdown, '_converter', Mock(side_effect=ImportError('secret local path')))
    originals = ['First page 1', 'Final appendix 2']
    pages, receipt = markdown.convert_pdf_pages(b'pdf', originals)
    assert pages == tuple(originals)
    assert receipt['status'] == 'fallback'
    assert all(page['reason'] == 'markitdown_dependency_unavailable' for page in receipt['pages'])
    assert 'secret local path' not in json.dumps(receipt)


@pytest.mark.parametrize('original,converted', [
    ('Signal 81.5 ± 0.6', 'Signal 81.5 0.6'),
    ('Signal 81.5†', 'Signal 81.5'),
    ('Threshold < 0.1', 'Threshold > 0.1'),
    ('rate 10 ^ -3', 'rate 10 -3'),
    ('Signal 81.5† Baseline 70.0', 'Signal 81.5 Baseline 70.0†'),
    ('Signal 81.5*', 'Signal 81.5'),
    ('Difference - 0.6', 'Difference + 0.6'),
    ('Interval 0.1–0.5', 'Interval 0.1 0.5'),
    ('Interval 0.1–0.5', 'Interval 0.1—0.5'),
    ('Rate 0.5‰', 'Rate 0.5'),
    ('Rate 0.5‱', 'Rate 0.5'),
    ('Temperature 30°', 'Temperature 30'),
])
def test_equal_digits_cannot_erase_or_move_scientific_conditions(monkeypatch, original, converted):
    converter(monkeypatch, [converted])
    pages, receipt = markdown.convert_pdf_pages(b'pdf', [original])
    assert pages == (original,)
    assert receipt['pages'][0]['reason'] == 'scientific_notation_changed'


def test_page_count_mismatch_never_shifts_pages_or_omits_appendix(monkeypatch):
    converter(monkeypatch, ['First page'])
    originals = ('First page', 'Last appendix')
    pages, receipt = markdown.convert_pdf_pages(b'pdf', originals)
    assert pages == originals
    assert all(page['reason'] == 'page_count_mismatch' for page in receipt['pages'])


def test_one_unserializable_page_does_not_discard_other_pages(monkeypatch):
    converter(monkeypatch, ['## First page', '## Last appendix'])
    monkeypatch.setattr(markdown, '_page_payloads', lambda _: iter([b'first', None, b'last']))
    originals = ('First page', 'Middle page', 'Last appendix')
    pages, receipt = markdown.convert_pdf_pages(b'pdf', originals)
    assert pages == ('## First page', 'Middle page', '## Last appendix')
    assert receipt['converted_pages'] == 2
    assert receipt['pages'][1]['reason'] == 'page_serialization_failed'


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(130)])
def test_conversion_cancellation_propagates_to_owned_worker(monkeypatch, interrupt):
    converter(monkeypatch, [interrupt])
    with pytest.raises(type(interrupt)):
        markdown.convert_pdf_pages(b'pdf', ['Page'])


def test_pdf_reader_planner_and_saved_markdown_share_page_text(monkeypatch, tmp_path):
    fixture = make_experiment(tmp_path)
    # Use real PDF parsing and frozen repository verification; only converter
    # transport is injected so the original extraction is an independent oracle.
    monkeypatch.setattr(markdown, '_converter', Mock(side_effect=ImportError()))
    original = extract_pdf_input(fixture['pdf'])
    outputs = ['## ' + page for page in original.page_texts]
    converter(monkeypatch, outputs)
    document = extract_pdf_input(fixture['pdf'])
    assert document.text == '\n'.join(outputs)
    assert document.sha256 == original.sha256
    assert supported_pdf_title(document) == supported_pdf_title(original)
    assert document.repository_links == original.repository_links
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    assert [page['text'] for page in packet['pdf']['pages']] == outputs
    assert _planning_excerpts(packet)['pdf_p1_0']['quote'] == outputs[0]
    run = tmp_path / 'run'
    run.mkdir()
    saved = markdown.write_markdown_evidence(run, document)
    text = (run / saved['artifact']).read_text('utf-8')
    assert all(page in text for page in outputs)
    assert '<!-- PDF page 3 -->' in text
    assert saved['artifact_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert json.loads((run / 'paper_markdown.json').read_text('utf-8')) == saved


@pytest.mark.parametrize('include_separator', [False, True])
def test_markdown_table_resolves_exact_dataset_cell_including_prior_rows(include_separator):
    heading = '| Method | Alpha | Beta |'
    separator = '| :--- | ---: | ---: |'
    row = '| Signal | 81.5 ± 0.6 | 79.4 ± 1.9 |'
    page = '\n'.join([heading, separator, '| Baseline | 70.0 ± 0.5 | 72.0 ± 0.4 |', row])
    header = heading + '\n' + separator if include_separator else heading
    ref = {'dataset': 'Alpha', 'method': 'Signal', 'name': 'accuracy', 'value': 81.5, 'std': .6}
    before = deepcopy(ref)
    result = locate_table_value(ref, row, header, page_text=page)
    assert result['status'] == 'verified', result
    assert result['column_labels'] == ['Alpha'] and result['column_index'] == 0
    assert row[slice(*result['cell_span'])] == result['cell_quote']
    assert page[slice(*result['row_span'])] == row
    assert ref == before
    wrong = locate_table_value({**ref, 'value': 79.4}, row, header, page_text=page)
    assert wrong['status'] == 'rejected'


def test_markdown_table_malformed_width_cannot_bypass_column_checks():
    heading = '| Method | Alpha | Beta |'
    row = '| Signal | 81.5 | 79.4 |'
    page = heading + '\n| --- | --- |\n' + row
    ref = {'dataset': 'Alpha', 'method': 'Signal', 'name': 'accuracy', 'value': 81.5}
    assert locate_table_value(ref, row, heading, page_text=page)['status'] == 'unknown'


def test_empty_corner_and_section_rows_preserve_columns_and_footnote_limits():
    header = '|  | Alpha | Beta |'
    row = '| Signal | 81.5 | 79.4 |'
    page = '\n'.join([header, '| --- | --- | --- |',
        '| Previous experiments: |  |  |', '| Baseline | 70.0† | 72.0 |',
        '| Our experiments: |  |  |', row])
    ref = {'dataset': 'Alpha', 'method': 'Signal', 'name': 'accuracy', 'value': 81.5}
    assert locate_table_value(ref, row, header, page_text=page)['status'] == 'verified'
    conditional = {'dataset': 'Alpha', 'method': 'Baseline', 'name': 'accuracy', 'value': 70.0}
    assert locate_table_value(conditional, '| Baseline | 70.0† | 72.0 |', header, page_text=page)['status'] == 'unknown'
    for invalid in ('| Table Other: |  |  |', '| Our experiments 2: |  |  |', '| Broken | 70.0 |  |'):
        assert locate_table_value(ref, row, header, page_text=page.replace('| Our experiments: |  |  |', invalid))['status'] == 'unknown'


def test_real_markitdown_preserves_sgc_scientific_table_columns():
    pytest.importorskip('markitdown')
    from pathlib import Path
    document = extract_pdf_input(Path(__file__).resolve().parents[1] /
        'examples/papers/pdf/sgc_1902.07153_holdout_01.pdf')
    page = document.page_texts[5]
    receipt = document.extraction['pages'][5]
    assert receipt['status'] == 'fallback_with_tables'
    assert receipt['tables'][0]['columns'] == 4
    header = '|  | Cora | Citeseer | Pubmed |'
    row = '| SGC | 81.0 ± 0.0 | 71.9 ± 0.1 | 78.9 ± 0.0 |'
    for dataset, value, std in [('Cora', 81.0, 0.0), ('Citeseer', 71.9, 0.1), ('Pubmed', 78.9, 0.0)]:
        ref = {'method': 'SGC', 'dataset': dataset, 'name': 'accuracy', 'value': value, 'std': std}
        assert locate_table_value(ref, row, header, page_text=page)['status'] == 'verified'
        assert locate_table_value({**ref, 'value': 81.5}, row, header, page_text=page)['status'] == 'rejected'
    # The original complete page remains present before the alternative table.
    original = page.split('<!-- Alternative table views')[0].rstrip()
    assert hashlib.sha256(original.encode()).hexdigest() == receipt['source_text_sha256']


def test_known_pdf_conversion_does_not_initialize_format_classifier(monkeypatch, tmp_path):
    engine = pytest.importorskip('markitdown')
    monkeypatch.setattr(engine.MarkItDown, '__init__', Mock(side_effect=AssertionError('unneeded classifier')))
    path = write_pdf(tmp_path / 'simple.pdf', [['Independent paper', 'Signal 81.5 79.4']])
    document = extract_pdf_input(path)
    assert document.extraction['status'] == 'converted'
    assert '81.5' in document.text and '79.4' in document.text


def test_real_ssgc_table_retains_multiword_methods_and_numbered_dataset_labels():
    pytest.importorskip('markitdown')
    from pathlib import Path
    document = extract_pdf_input(Path(__file__).resolve().parents[1] / 'examples/papers/pdf/ssgc_holdout_02.pdf')
    table = document.extraction['pages'][9]['tables'][0]
    assert table['caption'] == 'Table 7: Test accuracy on the document classification task.'
    page = document.page_texts[9]
    header = '| Model | 20NG | R8 | R52 | Ohsumed | MR |'
    row = '| S2GC | 88.6± 0.1 | 97.4 ± 0.1 | 94.5 ± 0.2 | 68.5 ± 0.1 | 76.7 ± 0.0 |'
    ref = {'method': 'S2GC', 'dataset': 'R52', 'name': 'accuracy', 'value': 94.5, 'std': .2}
    assert locate_table_value(ref, row, header, page_text=page)['status'] == 'verified'
    assert locate_table_value({**ref, 'dataset': 'R8'}, row, header, page_text=page)['status'] == 'rejected'


def ruled_pdf(tmp_path, *, crossing=False, missing=False, caption=True):
    """A real two-column page with a small, independently positioned table."""
    from PyPDF2 import PdfWriter
    from PyPDF2.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[-1]
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    commands = []
    def text(x, top, value):
        commands.append(f'BT /F1 10 Tf {x} {792-top} Td ({value}) Tj ET')
    text(50, 160, 'Unrelated left column 92.0 93.0')
    if caption:
        text(300, 140, 'Table 1. Test accuracy percent.')
    for top in (150, 175, 235):
        commands.append(f'300 {792-top} m 560 {792-top} l S')
    for x, label in [(305, 'Method'), (390, 'Alpha'), (490, 'Beta')]:
        text(x, 165, label)
    for top, label, alpha, beta in [(195, 'Baseline', '70.0', '72.0'), (220, 'Signal', '81.5', '79.4')]:
        text(305, top, label)
        text(445 if crossing and label == 'Signal' else 390, top, alpha)
        if not missing or label != 'Signal':
            text(490, top, beta)
    stream = DecodedStreamObject()
    stream.set_data('\n'.join(commands).encode('ascii'))
    page[NameObject('/Contents')] = writer._add_object(stream)
    path = tmp_path / 'ruled.pdf'
    with path.open('wb') as handle:
        writer.write(handle)
    return path.read_bytes()


def test_physical_table_excludes_adjacent_column_and_preserves_cells(tmp_path):
    pytest.importorskip('markitdown')
    tables = markdown._converter().tables(ruled_pdf(tmp_path))
    assert len(tables) == 1
    assert '| Signal | 81.5 | 79.4 |' in tables[0]['markdown']
    assert '92.0' not in tables[0]['markdown']
    assert tables[0]['bbox'] == [300.0, 150.0, 560.0, 235.0]


@pytest.mark.parametrize('options', [{'crossing': True}, {'missing': True}, {'caption': False}])
def test_ambiguous_or_incomplete_physical_table_is_not_promoted(tmp_path, options):
    pytest.importorskip('markitdown')
    assert markdown._converter().tables(ruled_pdf(tmp_path, **options)) == []
