"""Explicit local experiment guides retain frozen, bounded source navigation."""
from copy import deepcopy

import pytest

from src.author_command_inference import (
    _PathMentionMatcher, _mentioned, discover_author_commands, select_author_command,
)
from src.author_execution_plan import propose_author_execution_plan
from test_author_command_inference import TRAIN, packet_for


ENTRYPOINT = 'experiments/node/classification.py'


def linked_packet(tmp_path, sources):
    packet = packet_for(tmp_path, sources)
    packet['repository']['files'] = {
        name: source for name, source in packet['repository']['files'].items()
        if name == 'README.md'}
    return packet


@pytest.mark.parametrize('suffix', ['md', 'rst', 'txt'])
def test_linked_local_guide_reveals_deep_native_entrypoint(tmp_path, suffix):
    guide = 'docs/experiment.' + suffix
    sentinel = tmp_path / 'planning-executed.txt'
    packet = linked_packet(tmp_path, {
        'README.md': f'See the [experiment guide]({guide}) for training.',
        guide: f'The default training is implemented in `{ENTRYPOINT}`.',
        ENTRYPOINT: f"open({str(sentinel)!r}, 'w').write('Author source was executed')\n" + TRAIN,
    })
    discovery = discover_author_commands(packet, tmp_path)
    candidate = select_author_command(discovery)
    assert candidate['source_path'] == ENTRYPOINT
    assert candidate['argv'] == ['python', 'classification.py']
    assert candidate['cwd'] == 'experiments/node'
    assert candidate['readme_evidence'][0]['path'] == guide
    assert discovery['documentation_sources'] == ['README.md', guide]
    assert packet['repository']['files'][guide]['sha256'] == packet['file_inventory'][guide]['sha256']
    assert not (tmp_path / 'experiments/node/__pycache__').exists()
    assert not sentinel.exists()


def test_linked_guide_prose_is_available_to_native_default_planning(tmp_path):
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [experiment guide](docs/experiment.md).',
        'docs/experiment.md': f'The default training is implemented in `{ENTRYPOINT}`.',
        ENTRYPOINT: TRAIN,
    })
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['entrypoint']['source_path'] == ENTRYPOINT
    assert plan['steps'][0]['argv'] == ['python', 'classification.py']
    assert plan['command_source']['readme_evidence'][0]['path'] == 'docs/experiment.md'


def test_linked_guides_follow_relative_paths_and_stop_at_cycles(tmp_path):
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [guide](docs/overview.md).',
        'docs/overview.md': 'See the [protocol](protocol/details.txt) and [root](../README.md).',
        'docs/protocol/details.txt': f'Training uses `{ENTRYPOINT}`. See [overview](../overview.md).',
        ENTRYPOINT: TRAIN,
    })
    discovery = discover_author_commands(packet, tmp_path)
    assert select_author_command(discovery)['source_path'] == ENTRYPOINT
    assert discovery['documentation_sources'] == ['README.md', 'docs/overview.md', 'docs/protocol/details.txt']


@pytest.mark.parametrize('change', ['missing', 'modified'])
def test_missing_or_changed_guide_never_becomes_planning_evidence(tmp_path, change):
    guide = 'docs/experiment.md'
    packet = linked_packet(tmp_path, {
        'README.md': f'See the [guide]({guide}).',
        guide: f'Training uses `{ENTRYPOINT}`.',
        ENTRYPOINT: TRAIN,
    })
    if change == 'missing':
        (tmp_path / guide).unlink()
    else:
        (tmp_path / guide).write_text('Changed source mentions ' + ENTRYPOINT, encoding='utf-8')
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['documentation_sources'] == ['README.md']
    assert not discovery['candidates'] and not discovery['review_candidates']
    assert guide not in packet['repository']['files']


@pytest.mark.parametrize('target', ['../../outside.md', '/outside.md', 'C:/outside.md',
                                   'https://example.invalid/guide.md', 'docs/missing.md'])
def test_guide_links_cannot_escape_or_invent_frozen_inventory(tmp_path, target):
    packet = linked_packet(tmp_path, {
        'README.md': f'See the [guide]({target}).',
        'docs/unlinked.md': f'Training uses `{ENTRYPOINT}`.',
        ENTRYPOINT: TRAIN,
    })
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['documentation_sources'] == ['README.md']
    assert not discovery['candidates'] and not discovery['review_candidates']


def test_guides_share_the_existing_eight_document_limit(tmp_path):
    sources = {'README.md': 'See the [guide](docs/guide0.md).', ENTRYPOINT: TRAIN}
    sources.update({f'docs/guide{index}.md': f'See the [next](guide{index + 1}.md).'
                    for index in range(8)})
    sources['docs/guide8.md'] = f'Training uses `{ENTRYPOINT}`.'
    packet = linked_packet(tmp_path, sources)
    discovery = discover_author_commands(packet, tmp_path)
    assert len(discovery['documentation_sources']) == 8
    assert discovery['documentation_sources'][-1] == 'docs/guide6.md'
    assert not discovery['candidates'] and not discovery['review_candidates']


def test_linked_guide_respects_total_discovery_byte_limit(tmp_path):
    sources = {'README.md': 'See the [guide](docs/experiment.md).',
               'docs/experiment.md': f'Training uses `{ENTRYPOINT}`.', ENTRYPOINT: TRAIN}
    packet = linked_packet(tmp_path, sources)
    limit = sum(len(sources[name].encode()) for name in ('README.md', 'docs/experiment.md')) - 1
    discovery = discover_author_commands(packet, tmp_path, max_bytes=limit)
    assert discovery['bytes_examined'] <= limit
    assert discovery['documentation_sources'] == ['README.md']
    assert not discovery['candidates'] and not discovery['review_candidates']


@pytest.mark.parametrize('budget', ['source_files', 'source_bytes'])
def test_new_guides_cannot_spend_helper_source_budget_past_its_limit(tmp_path, budget):
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [guide](docs/overview.md).',
        'docs/overview.md': 'See the [details](details.md).',
        'docs/details.md': f'Training uses `{ENTRYPOINT}`.', ENTRYPOINT: TRAIN,
        **{f'catalog/{index}/unused.md': 'Unlinked reference.' for index in range(90)},
    })
    root_bytes = packet['file_inventory']['README.md']['bytes']
    guide_bytes = packet['file_inventory']['docs/overview.md']['bytes']
    packet['limits'][budget] = 2 if budget == 'source_files' else root_bytes + guide_bytes
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['documentation_sources'] == ['README.md', 'docs/overview.md']
    assert 'docs/details.md' not in packet['repository']['files']
    assert not discovery['candidates'] and not discovery['review_candidates']


def test_already_published_guide_does_not_spend_source_budget_twice(tmp_path):
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [guide](docs/experiment.md).',
        'docs/experiment.md': 'The original defaults are documented here.',
    })
    inventory = packet['file_inventory']['docs/experiment.md']
    packet['repository']['files']['docs/experiment.md'] = {
        **deepcopy(inventory), 'text': 'The original defaults are documented here.'}
    packet['limits']['source_files'] = len(packet['repository']['files'])
    packet['limits']['source_bytes'] = sum(source['bytes'] for source in packet['repository']['files'].values())
    discovery = discover_author_commands(packet, tmp_path)
    assert discovery['documentation_sources'] == ['README.md', 'docs/experiment.md']


def test_guide_text_is_navigation_and_does_not_prove_executable_training(tmp_path):
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [guide](docs/experiment.md).',
        'docs/experiment.md': f'Training uses `{ENTRYPOINT}` with estimator.fit(records).',
        ENTRYPOINT: "print('configuration only')\n",
    })
    discovery = discover_author_commands(packet, tmp_path)
    assert not discovery['candidates']
    assert discovery['review_candidates'][0]['review_only'] is True
    assert discovery['review_candidates'][0]['training_evidence'] == []


@pytest.mark.parametrize('ending', ['.', '. Next, use the defaults.', '.\nUse defaults.'])
def test_sentence_ending_period_preserves_unquoted_entrypoint_mentions(tmp_path, ending):
    text = 'Training uses ' + ENTRYPOINT + ending
    assert _mentioned(text, ENTRYPOINT)
    assert _PathMentionMatcher([ENTRYPOINT])(text) == {ENTRYPOINT}
    packet = linked_packet(tmp_path, {
        'README.md': 'See the [guide](docs/experiment.md).',
        'docs/experiment.md': text,
        ENTRYPOINT: TRAIN,
    })
    assert select_author_command(discover_author_commands(packet, tmp_path))['source_path'] == ENTRYPOINT


@pytest.mark.parametrize('ending', ['.backup', '.py', '/other.py', '-other', '_other'])
def test_sentence_punctuation_rule_does_not_accept_a_different_source_path(ending):
    text = 'Training uses ' + ENTRYPOINT + ending + '.'
    assert not _mentioned(text, ENTRYPOINT)
    assert _PathMentionMatcher([ENTRYPOINT])(text) == set()
