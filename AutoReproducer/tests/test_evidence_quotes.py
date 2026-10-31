import pytest

from src.evidence_quotes import original_quote_spans


@pytest.mark.parametrize('quoted', [
    r'The mean clas-\nsification accuracy on the test nodes after 7 runs.',
    r'The mean clas-\r\nsification accuracy on the test nodes after 7 runs.',
    'The mean clas- sification accuracy on the test nodes after 7 runs.',
])
def test_recovers_only_original_pdf_span(quoted):
    original = 'The mean clas-\nsification accuracy on the test nodes after 7 runs.'
    assert original_quote_spans(quoted, [(original, 'page')]) == [(original, 'page')]


def test_original_code_literal_takes_precedence_over_decoded_text():
    literal = r'print("mean accuracy\n7 runs")'
    assert original_quote_spans(literal, [(literal, 'code'), (literal.replace('\\n', '\n'), 'other')]) == [(literal, 'code')]


def test_model_escaped_code_quotes_recover_original_without_interpreting_unicode():
    original = 'print("Test", "acc.:{:.1f}".format(acc*100))'
    quoted = original.replace('"', '\\"')
    assert original_quote_spans(quoted, [(original, 'code')]) == [(original, 'code')]
    assert not original_quote_spans(r'accuracy: 8\u0035.3', [('accuracy: 85.3', 'code')])


@pytest.mark.parametrize('quoted', [
    r'mean accuracy over 8\nruns',
    r'best accuracy over 7\nruns',
    r'mean accuracy over 7\nruns on Dataset B',
])
def test_recovery_does_not_change_values_or_claims(quoted):
    assert not original_quote_spans(quoted, [('mean accuracy over 7\nruns', 'page')])


def test_distinct_recovered_spans_are_ambiguous():
    assert not original_quote_spans(r'mean accuracy\n7 runs', [
        ('mean accuracy 7 runs', 'one'), ('mean accuracy\n7 runs', 'two')])


@pytest.mark.parametrize('original', [
    'The mean clas-\nsification accuracy over 7 runs.',
    'The mean classiﬁcation accuracy over 7 runs.',
])
def test_pdf_printing_wraps_return_original_glyphs_and_offsets(original):
    quote = 'The mean classification accuracy over 7 runs.'
    assert not original_quote_spans(quote, [(original, 'page')])
    assert original_quote_spans(quote, [(original, 'page')], pdf_wrapping=True) == [(original, 'page')]


@pytest.mark.parametrize('quote,original', [
    ('ModelAB accuracy 85.3', 'Model-AB accuracy 85.3'),
    ('accuracy 853', 'accuracy 85-\n3'),
    ('xy', 'x-\ny'),
    ('classification accuracy over 8 runs', 'classiﬁcation accuracy over 7 runs'),
])
def test_pdf_typography_does_not_change_names_numbers_or_math(quote, original):
    assert not original_quote_spans(quote, [(original, 'page')], pdf_wrapping=True)


def test_pdf_recovery_cannot_quote_half_a_ligature():
    assert not original_quote_spans('f accuracy', [('ﬁ accuracy', 'page')], pdf_wrapping=True)
