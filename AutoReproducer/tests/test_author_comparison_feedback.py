"""Navigation feedback stays bounded and cannot become source evidence."""
from copy import deepcopy
import json

import pytest

from src.author_claim_comparison import _correction_feedback
from src.author_comparison_feedback import context_quote_hints, correction_source_view


def _attempt(side='reference', field='split'):
    return {
        'proposal': {
            'citations': [{'id': 'selected', 'evidence_id': 'paper_selected'}],
            'comparisons': [{
                'actual_id': 'metric_0',
                side: {'citations': ['selected'], 'context_quotes': {
                    field: 'Sparse splitting uses the original setting across both pages.'}},
            }],
        },
        'validation_errors': [{
            'index': 0, 'actual_id': 'metric_0',
            'error': f'{side}.context_quotes.{field} must occur in its own original citations',
        }],
    }


def _excerpts():
    return {
        'paper_other': {'source': 'pdf', 'page': 8,
                        'quote': 'Sparse splitting uses the original setting across both pages in this alternative.'},
        'repo_original': {'source': 'repository', 'path': 'README.md',
                          'quote': 'The repository uses sparse splitting.'},
        'paper_selected': {'source': 'pdf', 'page': 7,
                           'quote': 'For homophilic datasets, sparse splitting is used.'},
    }


def test_declared_alias_prioritizes_its_supplied_original_without_mutation():
    attempt, excerpts = _attempt(), _excerpts()
    before = deepcopy((attempt, excerpts))
    hints = context_quote_hints(attempt, excerpts)
    assert hints[0]['evidence_id'] == 'paper_selected'
    assert hints[0]['already_selected'] is True
    assert hints[1]['already_selected'] is False
    assert (attempt, excerpts) == before


@pytest.mark.parametrize('side, source', [('reference', 'pdf'), ('actual_context', 'repository')])
def test_context_hints_remain_on_the_rejected_source_side(side, source):
    hints = context_quote_hints(_attempt(side), _excerpts())
    assert hints
    assert all(hint['source'] == source for hint in hints)
    assert all(hint['field'] == f'{side}.context_quotes.split' for hint in hints)
    for hint in hints:
        assert hint['quote'] in _excerpts()[hint['evidence_id']]['quote']


def test_unprovided_declared_source_cannot_become_a_hint():
    attempt = _attempt()
    attempt['proposal']['citations'] = [{
        'id': 'selected', 'evidence_id': 'not_supplied', 'source': 'pdf',
        'quote': 'Sparse splitting uses the original setting in an invented source.',
    }]
    hints = context_quote_hints(attempt, _excerpts())
    assert hints and all(hint['evidence_id'] in _excerpts() for hint in hints)
    assert all(not hint['already_selected'] for hint in hints)
    assert all('invented source' not in hint['quote'] for hint in hints)


def test_hints_never_join_original_pages_even_when_rejected_quote_spans_them():
    attempt = _attempt()
    first = 'For homophilic datasets, sparse splitting uses the original setting'
    second = 'used in prior experiments across both pages.'
    proposed = first + ' ' + second
    attempt['proposal']['comparisons'][0]['reference']['context_quotes']['split'] = proposed
    excerpts = {'page_7': {'source': 'pdf', 'page': 7, 'quote': first},
                'page_8': {'source': 'pdf', 'page': 8, 'quote': second}}
    hints = context_quote_hints(attempt, excerpts)
    assert len(hints) == 2
    assert all(hint['quote'] in excerpts[hint['evidence_id']]['quote'] for hint in hints)
    assert all(hint['quote'] != proposed for hint in hints)


def test_hint_count_and_quote_size_are_bounded_across_many_failed_rows():
    attempt = _attempt()
    attempt['proposal']['comparisons'] *= 12
    attempt['validation_errors'] = [dict(attempt['validation_errors'][0], index=i) for i in range(12)]
    excerpts = {f'paper_{i}': {'source': 'pdf', 'page': i,
                'quote': ('padding ' * 200) + ('sparse splitting original setting ' * 100)}
                for i in range(20)}
    hints = context_quote_hints(attempt, excerpts)
    assert len(hints) == 6
    assert [hint['evidence_id'] for hint in hints[:3]] == ['paper_0', 'paper_1', 'paper_2']
    for hint in hints:
        assert len(hint['quote']) <= 600
        assert hint['quote'] in excerpts[hint['evidence_id']]['quote']


def test_navigation_prefers_the_contiguous_passage_over_an_earlier_common_word():
    attempt = _attempt()
    passage = 'Sparse splitting uses the original setting'
    quote = 'The original experiments are discussed here. ' + 'irrelevant ' * 150 + passage
    excerpts = {'paper_selected': {'source': 'pdf', 'page': 7, 'quote': quote}}
    hint, = context_quote_hints(attempt, excerpts)
    assert passage in hint['quote']
    assert hint['quote'] in quote and len(hint['quote']) <= 600


def test_exact_original_on_an_unselected_page_precedes_selected_partial_matches():
    attempt = _attempt(field='runs')
    proposed = 'We run each experiment 100 times with different initializations.'
    attempt['proposal']['comparisons'][0]['reference']['context_quotes']['runs'] = proposed
    excerpts = _excerpts()
    excerpts['paper_selected']['quote'] = 'We run different experiments with random splitting.'
    excerpts['paper_six'] = {'source': 'pdf', 'page': 6,
                             'quote': 'Repetitions. We run each experiment 100 times with\ndifferent initializations.'}
    hints = context_quote_hints(attempt, excerpts)
    assert hints[0]['evidence_id'] == 'paper_six' and hints[0]['already_selected'] is False
    assert hints[0]['quote'] in excerpts['paper_six']['quote']
    assert hints[1]['already_selected'] is True


@pytest.mark.parametrize('proposed', [None, 5, {}, [], 'abc', 'x' * 1201])
def test_malformed_or_oversized_context_quote_does_not_generate_hints(proposed):
    attempt = _attempt()
    attempt['proposal']['comparisons'][0]['reference']['context_quotes']['split'] = proposed
    assert context_quote_hints(attempt, _excerpts()) == []


@pytest.mark.parametrize('error, index', [
    ('context_quotes.split must occur in its own original citations', 0),
    ('reference.value must have evidence', 0),
    ('reference.context_quotes.split must occur in its own original citations', True),
    ('reference.context_quotes.split must occur in its own original citations', -1),
    ('reference.context_quotes.split must occur in its own original citations', 100),
])
def test_ambiguous_or_unrelated_errors_do_not_trigger_navigation(error, index):
    attempt = _attempt()
    attempt['validation_errors'][0].update(error=error, index=index)
    assert context_quote_hints(attempt, _excerpts()) == []


@pytest.mark.parametrize('area, value', [
    ('proposal', None), ('proposal', []), ('proposal', 'not an object'),
    ('comparisons', None), ('comparisons', {}), ('comparisons', 'not rows'),
    ('citations', None), ('citations', 3), ('citations', {'not': 'declarations'}),
    ('selected_ids', None), ('selected_ids', 3), ('selected_ids', [['unhashable']]),
    ('evidence_id', []), ('evidence_id', {}), ('evidence_id', None),
])
def test_malformed_proposals_cannot_break_feedback_generation(area, value):
    attempt = _attempt()
    if area == 'proposal':
        attempt['proposal'] = value
    elif area in {'comparisons', 'citations'}:
        attempt['proposal'][area] = value
    elif area == 'selected_ids':
        attempt['proposal']['comparisons'][0]['reference']['citations'] = value
    else:
        attempt['proposal']['citations'][0]['evidence_id'] = value
    before = deepcopy(attempt)
    hints = context_quote_hints(attempt, _excerpts())
    assert isinstance(hints, list) and len(hints) <= 6
    assert all(hint['evidence_id'] in _excerpts() for hint in hints)
    assert attempt == before


def test_feedback_reserves_navigation_budget_and_redacts_each_section():
    credential = 'opaque_"client_雪_value'
    generic = 'sk-' + 'q' * 32
    secrets = credential + ' ' + generic + ' Bearer ephemeral-secret'
    attempt = _attempt()
    attempt['proposal']['reason'] = secrets + 'Z' * 30000
    attempt['validation_errors'][0]['error'] += secrets + 'E' * 5000
    attempt['context_quote_hints'] = context_quote_hints(attempt, _excerpts())
    attempt['context_quote_hints'][0]['quote'] += secrets
    attempt['context_quote_hints'] *= 10
    attempt['label_evidence_hints'] = [{'quote': secrets + 'L' * 500} for _ in range(6)]
    attempt['error'] = secrets + 'C' * 3000
    attempt['json_error'] = {'message': secrets + 'J' * 3000, 'line': 2, 'column': 3}
    records = [{'id': 'metric_0', 'value': 0.85, 'source_quote': secrets + 'O' * 4000}]
    accepted = {'accepted_' + 'A' * 1000: {}}
    before = deepcopy((attempt, accepted, records))
    feedback = _correction_feedback(attempt, accepted, credential, records)
    assert len(feedback) <= 12000
    assert 'Original excerpt candidates for rejected context quotes' in feedback
    assert 'paper_selected' in feedback and 'navigation only' in feedback
    assert 'Do not join pages or excerpts' in feedback
    assert feedback.endswith('Do not repeat rejected rows unchanged or invent missing evidence.')
    for secret in (credential, json.dumps(credential)[1:-1], generic, 'ephemeral-secret'):
        assert secret not in feedback
    assert '[REDACTED]' in feedback
    assert (attempt, accepted, records) == before


@pytest.mark.parametrize('key, marker', [
    ('context_quote_hints', 'select that evidence_id before quoting it:\n'),
    ('label_evidence_hints', 'Do not combine names or infer a variant from token overlap:\n'),
])
def test_navigation_budget_drops_whole_items_and_keeps_quotes_intact(key, marker):
    attempt = _attempt()
    original_hints = [{
        'actual_id': 'metric_0', 'field': 'reference.context_quotes.split',
        'evidence_id': f'paper_{i}', 'source': 'pdf', 'page': i,
        'quote': f'Original passage {i}: ' + 'q' * 570,
    } for i in range(6)]
    attempt[key] = deepcopy(original_hints)
    feedback = _correction_feedback(attempt, {}, None)
    navigation_json = feedback.split(marker, 1)[1].split('\nCurrent response excerpt', 1)[0]
    selected = json.loads(navigation_json)
    assert 0 < len(selected) < len(original_hints)
    assert all(item in original_hints for item in selected)
    assert '[... omitted ...]' not in navigation_json
    assert attempt[key] == original_hints
    assert len(feedback) <= 12000


def test_feedback_keeps_all_known_quote_faults_visible_at_the_twelve_row_bound():
    errors = []
    expected = {'reference': ['unit', 'split', 'aggregation', 'runs', 'uncertainty'],
                'actual_context': ['unit', 'split', 'aggregation', 'runs']}
    for index in range(12):
        for side, fields in expected.items():
            for field in fields:
                location = f'{side}.context_quotes.{field}'
                errors.append({'index': index, 'actual_id': 'metric_' + str(index), 'field': location,
                    'error': location + ' must occur in its own original citations; Proposed quotation: ' + 'q' * 160})
    attempt = {'validation_errors': errors, 'proposal': {'comparisons': [], 'reason': 'r' * 30000},
               'label_evidence_hints': [{'quote': 'l' * 570} for _ in range(6)],
               'context_quote_hints': [{'quote': 'q' * 570} for _ in range(6)],
               'error': 'e' * 3000, 'json_error': {'message': 'j' * 3000, 'line': 2, 'column': 3}}
    records = [{'id': 'metric_' + str(index), 'source_quote': 'o' * 4000} for index in range(12)]
    feedback = _correction_feedback(attempt, {'accepted_' + 'a' * 1000: {}}, None, records)
    encoded = feedback.split('each must use an exact own-citation span, or be omitted if unavailable): ', 1)[1]
    summary, _ = json.JSONDecoder().raw_decode(encoded)
    assert summary == [{'index': index, **expected} for index in range(12)]
    assert len(feedback) <= 12000
    assert feedback.endswith('Do not repeat rejected rows unchanged or invent missing evidence.')


def test_navigation_budget_includes_each_rejected_field_before_its_alternatives():
    attempt = _attempt()
    fields = ('split', 'runs')
    attempt['context_quote_hints'] = [
        {'actual_id': 'metric_0', 'field': 'reference.context_quotes.' + field,
         'evidence_id': f'{field}_{index}', 'quote': field + 'q' * 570}
        for field in fields for index in range(3)]
    feedback = _correction_feedback(attempt, {}, None)
    encoded = feedback.split('select that evidence_id before quoting it:\n', 1)[1]
    hints, _ = json.JSONDecoder().raw_decode(encoded)
    assert [hint['evidence_id'] for hint in hints[:2]] == ['split_0', 'runs_0']
    assert all(hint in attempt['context_quote_hints'] for hint in hints)


def _correction_projection_inputs():
    attempt = _attempt()
    attempt['proposal']['citations'].append({'id': 'settings_alias', 'evidence_id': 'paper_settings'})
    row = attempt['proposal']['comparisons'][0]
    row.update(actual_context={'citations': ['repo_original']},
               settings={'citations': ['settings_alias', 'repo_original']})
    attempt['context_quote_hints'] = [{'evidence_id': 'paper_other', 'field': 'reference.context_quotes.split'}]
    excerpts = _excerpts()
    excerpts['paper_selected'].update(sha256='page-seven-digest', pdf_sha256='paper-digest',
                                      quote='Full page seven original. ' * 100)
    excerpts['paper_settings'] = {'source': 'pdf', 'page': 9, 'quote': 'Original experimental settings.'}
    excerpts['unused'] = {'source': 'pdf', 'page': 11, 'quote': 'Unrelated original discussion.'}
    view = {'version': 1, 'pdf': {'sha256': 'paper-digest', 'pages': [{'page': 7}]},
            'repository': {'commit': 'frozen-commit', 'files': {'README.md': {'sha256': 'readme-digest'}}},
            'file_inventory': {'README.md': {'sha256': 'readme-digest', 'bytes': 20}},
            'source_discovery': {'status': 'frozen'}, 'evidence_excerpts': excerpts,
            'source_selection': {'full_pdf_retained': True, 'repository_excerpt_count': 1}}
    return attempt, view


def test_correction_projection_retains_alias_direct_settings_and_hint_originals():
    attempt, view = _correction_projection_inputs()
    before = deepcopy((attempt, view))
    projected = correction_source_view(attempt, view)
    assert set(projected['evidence_excerpts']) == {
        'paper_selected', 'repo_original', 'paper_settings', 'paper_other'}
    for key, excerpt in projected['evidence_excerpts'].items():
        assert excerpt == view['evidence_excerpts'][key]
    assert len(projected['evidence_excerpts']['paper_selected']['quote']) > 2000
    for key in ('version', 'pdf', 'repository', 'file_inventory', 'source_discovery'):
        assert projected[key] == view[key]
    assert (attempt, view) == before


def test_correction_projection_covers_every_failed_row_not_only_hint_owner():
    attempt, view = _correction_projection_inputs()
    second = {'actual_id': 'metric_1', 'reference': {'citations': ['unused']},
              'actual_context': {'citations': ['repo_original']},
              'settings': {'citations': ['paper_settings', 'repo_original']}}
    attempt['proposal']['comparisons'].append(second)
    attempt['validation_errors'].append({'index': 1, 'actual_id': 'metric_1',
                                         'error': 'reference.value needs evidence'})
    projected = correction_source_view(attempt, view)
    assert set(projected['evidence_excerpts']) == set(view['evidence_excerpts'])


def test_correction_projection_keeps_unselected_label_navigation_sources_too():
    attempt, view = _correction_projection_inputs()
    attempt['label_evidence_hints'] = [{'evidence_id': 'unused', 'field': 'reference.method'}]
    projected = correction_source_view(attempt, view)
    assert projected['evidence_excerpts']['unused'] == view['evidence_excerpts']['unused']


def test_correction_projection_metadata_describes_selected_complete_sources():
    attempt, view = _correction_projection_inputs()
    view['pdf']['pages'] = [{'page': page, 'sha256': f'page-{page}-digest', 'bytes': 900}
                            for page in (7, 8, 9, 11)]
    before = deepcopy(view)
    projected = correction_source_view(attempt, view)
    assert projected['pdf']['pages'] == before['pdf']['pages'][:3]
    selection = projected['source_selection']
    assert selection['full_pdf_retained'] is False
    assert selection['repository_excerpt_count'] == 1
    assert selection['repository_quote_characters'] == len(view['evidence_excerpts']['repo_original']['quote'])
    assert set(selection['selected_evidence_ids']) == set(projected['evidence_excerpts'])
    assert selection['omitted_sources_are_not_negative_evidence'] is True
    projected['evidence_excerpts']['paper_selected']['quote'] = 'changed result only'
    projected['pdf']['pages'][0]['sha256'] = 'changed result only'
    assert view == before


def test_correction_projection_cannot_resolve_legacy_literal_citation_by_guessing():
    attempt, view = _correction_projection_inputs()
    attempt['proposal']['citations'][0] = {
        'id': 'selected', 'source': 'pdf', 'page': 7,
        'quote': view['evidence_excerpts']['paper_selected']['quote'],
    }
    assert correction_source_view(attempt, view) == view


@pytest.mark.parametrize('hints', [None, []])
def test_correction_projection_without_quote_hints_keeps_original_view(hints):
    attempt, view = _correction_projection_inputs()
    attempt['context_quote_hints'] = hints
    assert correction_source_view(attempt, view) == view


@pytest.mark.parametrize('area', [
    'proposal', 'rows', 'row', 'declarations', 'selected_ids', 'missing_node',
    'unknown_selected_id', 'unknown_alias_target', 'malformed_alias_target',
    'unknown_hint_id', 'missing_hint_id', 'malformed_hint_id',
    'error_index', 'boolean_error_index', 'out_of_range_error_index',
])
def test_correction_projection_falls_back_when_any_required_source_is_unresolved(area):
    attempt, view = _correction_projection_inputs()
    proposal = attempt['proposal']
    if area == 'proposal':
        attempt['proposal'] = None
    elif area == 'rows':
        proposal['comparisons'] = None
    elif area == 'row':
        proposal['comparisons'][0] = None
    elif area == 'declarations':
        proposal['citations'] = None
    elif area == 'selected_ids':
        proposal['comparisons'][0]['settings']['citations'] = None
    elif area == 'missing_node':
        proposal['comparisons'][0].pop('settings')
    elif area == 'unknown_selected_id':
        proposal['comparisons'][0]['settings']['citations'].append('not_supplied')
    elif area == 'unknown_alias_target':
        proposal['citations'][0]['evidence_id'] = 'not_supplied'
    elif area == 'malformed_alias_target':
        proposal['citations'][0]['evidence_id'] = ['unhashable']
    elif area == 'unknown_hint_id':
        attempt['context_quote_hints'][0]['evidence_id'] = 'not_supplied'
    elif area == 'missing_hint_id':
        attempt['context_quote_hints'][0].pop('evidence_id')
    elif area == 'malformed_hint_id':
        attempt['context_quote_hints'][0]['evidence_id'] = ['unhashable']
    else:
        attempt['validation_errors'][0]['index'] = {
            'error_index': None, 'boolean_error_index': True, 'out_of_range_error_index': 100}[area]
    before = deepcopy((attempt, view))
    assert correction_source_view(attempt, view) == view
    assert (attempt, view) == before
