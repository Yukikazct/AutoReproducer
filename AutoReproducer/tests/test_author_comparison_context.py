"""Analysis selects original experiment evidence without copying the framework."""
from copy import deepcopy
import hashlib

import pytest

from src.author_comparison_context import (
    MAX_ADDITIONAL_BYTES, MAX_ADDITIONAL_FILES, MAX_REPOSITORY_CHARACTERS,
    MAX_REPOSITORY_EXCERPTS, comparison_plan, comparison_view, prepare_comparison_packet,
)
from src.discovered_repository_plan import _citations, _planning_excerpts


def source(text):
    raw = text.encode()
    return {'text': text, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def packet_for(tmp_path, sources, loaded, entrypoint):
    for path, text in sources.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode())
    page = source('Table 1: Signal accuracy on PublicSet\nEncoder 83.2 ± 0.7\n' * 160)
    packet = {'version': 1, 'pdf': {'sha256': 'f' * 64, 'pages': [{'page': 1, **page}]},
        'repository': {'revision': 'a' * 40, 'files': {p: source(sources[p]) for p in loaded}},
        'file_inventory': {p: {k: v for k, v in source(text).items() if k != 'text'}
                           for p, text in sources.items()}}
    plan = {'steps': [{'entrypoint': {'source_path': entrypoint},
                       'argv': ['python', entrypoint, '--epochs', '413']}],
            'parameters': {'epochs': 413, 'runs': 17, 'seed': 91},
            'requirements': {'author': ['package>=1.0'], 'compatibility': ['numpy==1.26.4'],
                             'import_evidence': ['mechanical telemetry'] * 10000,
                             'deferred_runtime_imports': {'other': 'large'},
                             'repository_namespaces': ['framework']}}
    return packet, plan


@pytest.mark.parametrize('directory,entrypoint', [
    ('bench/field', 'model.py'), ('studies/image', 'encoder.py'), ('experiments', 'main.py'),
])
def test_unloaded_local_metric_helper_is_verified_and_citable(tmp_path, directory, entrypoint):
    sources = {f'{directory}/{entrypoint}': 'from helpers import measure\nmeasure()\n',
               f'{directory}/helpers.py': 'def measure():\n    print("Test Accuracy", 0.832)\n'}
    entry = f'{directory}/{entrypoint}'
    packet, plan = packet_for(tmp_path, sources, [entry], entry)
    original = deepcopy(packet)
    analysis = prepare_comparison_packet(packet, plan, tmp_path)
    helper = f'{directory}/helpers.py'
    assert packet == original
    assert analysis['repository']['files'][helper] == source(sources[helper])
    view = comparison_view(analysis, plan, [])
    evidence_id, evidence = next((k, v) for k, v in view['evidence_excerpts'].items()
                                 if v.get('path') == helper)
    assert _planning_excerpts(analysis)[evidence_id] == evidence
    citations = _citations({'citations': [{'id': 'helper', 'evidence_id': evidence_id}]}, analysis, tmp_path)
    assert citations['helper']['sha256'] == source(sources[helper])['sha256']


def test_parent_package_and_relative_imports_follow_full_inventory(tmp_path):
    sources = {'bench/text/model.py': 'from text import evaluate\n',
               'bench/text/__init__.py': 'from .scores import evaluate\n',
               'bench/text/scores.py': 'def evaluate():\n    print("Test Accuracy", 0.832)\n',
               'text/__init__.py': 'raise RuntimeError("unrelated farther root")\n'}
    packet, plan = packet_for(tmp_path, sources, ['bench/text/model.py'], 'bench/text/model.py')
    analysis = prepare_comparison_packet(packet, plan, tmp_path)
    paths = analysis['repository']['files']
    assert 'bench/text/__init__.py' in paths and 'bench/text/scores.py' in paths
    assert 'text/__init__.py' not in paths


def test_large_runtime_packet_keeps_full_pdf_exact_ids_and_small_experiment(tmp_path):
    sources = {'study/model.py': 'from score import evaluate\n',
               'study/score.py': 'def evaluate():\n    print("Test Accuracy", 0.832)\n',
               **{f'framework/module_{i}.py': '# unrelated runtime implementation\n' * 100
                  for i in range(500)}}
    packet, plan = packet_for(tmp_path, sources, list(sources), 'study/model.py')
    original = deepcopy(packet)
    view = comparison_view(packet, plan, [])
    assert packet == original
    all_excerpts = _planning_excerpts(packet)
    assert {k: v for k, v in all_excerpts.items() if v['source'] == 'pdf'} == {
        k: v for k, v in view['evidence_excerpts'].items() if v['source'] == 'pdf'}
    assert all(all_excerpts[k] == v for k, v in view['evidence_excerpts'].items())
    assert set(view['repository']['files']) == {'study/model.py', 'study/score.py'}
    assert all('text' not in s for s in view['repository']['files'].values())
    assert view['source_selection']['omitted_sources_are_not_negative_evidence'] is True
    assert view['source_selection']['repository_quote_characters'] <= MAX_REPOSITORY_CHARACTERS
    assert view['source_selection']['repository_excerpt_count'] <= MAX_REPOSITORY_EXCERPTS


def test_only_mechanical_import_telemetry_removed_from_plan(tmp_path):
    _, plan = packet_for(tmp_path, {'run.py': 'pass\n'}, ['run.py'], 'run.py')
    original = deepcopy(plan)
    compact = comparison_plan(plan)
    assert plan == original
    assert compact['steps'] == plan['steps']
    assert compact['parameters'] == plan['parameters']
    assert compact['requirements'] == {'author': ['package>=1.0'], 'compatibility': ['numpy==1.26.4']}


def test_large_experiment_file_selects_exact_metric_span_within_budget(tmp_path):
    text = '# implementation detail\n' * 2800 + 'print("FinalAccuracy", 0.832)\n'
    sources = {'run.py': text, **{f'helper_{i}.py': '# details\n' * 3500 for i in range(20)}}
    packet, plan = packet_for(tmp_path, sources, list(sources), 'run.py')
    view = comparison_view(packet, plan, [{'source_quote': 'FinalAccuracy: 0.832'}])
    spans = view['evidence_excerpts']
    assert any('print("FinalAccuracy"' in item['quote'] for item in spans.values())
    assert all(_planning_excerpts(packet)[k] == v for k, v in spans.items())
    assert view['source_selection']['repository_quote_characters'] <= MAX_REPOSITORY_CHARACTERS
    assert view['source_selection']['repository_excerpt_count'] <= MAX_REPOSITORY_EXCERPTS


@pytest.mark.parametrize('failure', ['changed', 'missing', 'oversized'])
def test_unavailable_or_changed_helper_never_becomes_evidence(tmp_path, failure):
    sources = {'study/run.py': 'from helper import evaluate\n', 'study/helper.py': 'value = 1\n'}
    packet, plan = packet_for(tmp_path, sources, ['study/run.py'], 'study/run.py')
    helper = tmp_path / 'study/helper.py'
    if failure == 'changed':
        helper.write_text('value = 2\n', encoding='utf-8')
    elif failure == 'missing':
        helper.unlink()
    else:
        packet['file_inventory']['study/helper.py']['bytes'] = 100001
    result = prepare_comparison_packet(packet, plan, tmp_path)
    assert 'study/helper.py' not in result['repository']['files']
    assert result['comparison_source_discovery']['read_failures'][0]['path'] == 'study/helper.py'


def test_inventory_traversal_cannot_read_outside_workspace(tmp_path):
    packet, plan = packet_for(tmp_path, {'run.py': 'pass\n'}, ['run.py'], 'run.py')
    packet['file_inventory']['../outside.py'] = {k: v for k, v in source('secret = 1').items() if k != 'text'}
    plan['citations'] = [{'path': '../outside.py'}]
    result = prepare_comparison_packet(packet, plan, tmp_path)
    assert '../outside.py' not in result['repository']['files']
    assert not result['comparison_source_discovery']['added_sources']


def test_inventory_symlink_cannot_be_used_as_source(tmp_path):
    packet, plan = packet_for(tmp_path, {'run.py': 'pass\n', 'helper.py': 'value = 1\n'}, ['run.py'], 'run.py')
    outside = tmp_path.parent / (tmp_path.name + '_outside.py')
    outside.write_text('value = 1\n', encoding='utf-8')
    helper = tmp_path / 'helper.py'
    helper.unlink()
    try:
        helper.symlink_to(outside)
    except OSError:
        pytest.skip('Creating symlinks is not permitted on this host')
    result = prepare_comparison_packet(packet, plan, tmp_path)
    assert 'helper.py' not in result['repository']['files']
    assert result['comparison_source_discovery']['read_failures']


def test_additional_source_reads_have_count_and_byte_bounds(tmp_path):
    sources = {'study/run.py': 'pass\n', **{f'study/helper_{i}.py': '# original\n' * 8500 for i in range(100)}}
    packet, plan = packet_for(tmp_path, sources, ['study/run.py'], 'study/run.py')
    result = prepare_comparison_packet(packet, plan, tmp_path)
    discovery = result['comparison_source_discovery']
    assert discovery['attempted_files'] <= MAX_ADDITIONAL_FILES
    assert 0 < discovery['additional_bytes'] <= MAX_ADDITIONAL_BYTES
    assert len(result['repository']['files']) < len(sources)
    assert discovery['read_failures']


def test_imported_helper_precedes_large_sibling_catalog(tmp_path):
    sources = {'study/run.py': 'from zz_metrics import evaluate\n',
               'study/zz_metrics.py': 'def evaluate():\n    print("Test Accuracy", 0.832)\n',
               **{f'study/alternative_{i}.py': '# unrelated model\n' * 4000 for i in range(80)}}
    packet, plan = packet_for(tmp_path, sources, ['study/run.py'], 'study/run.py')
    result = prepare_comparison_packet(packet, plan, tmp_path)
    view = comparison_view(result, plan, [])
    assert result['comparison_source_discovery']['added_sources'][0]['path'] == 'study/zz_metrics.py'
    assert 'study/zz_metrics.py' in view['repository']['files']
