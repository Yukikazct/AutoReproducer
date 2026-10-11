"""Correct evidence is selectable without weakening protocol/number checks."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from discovered_fixtures import make_experiment
from src import discovered_repository_plan as planner


def test_selected_source_ids_expand_to_original_text_and_reject_fabricated_ids(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    proposal = deepcopy(fixture['proposal'])
    excerpts = planner._planning_excerpts(packet)
    for item in proposal['citations']:
        selected = next(key for key, excerpt in excerpts.items()
                        if excerpt['source'] == item['source'] and item['quote'] in excerpt['quote']
                        and excerpt.get('page') == item.get('page')
                        and excerpt.get('path') == item.get('path'))
        item.clear()
        item.update(id=selected, evidence_id=selected)
    citations = planner._citations(proposal, packet, fixture['workspace'])
    assert all(item['quote'] == excerpts[key]['quote'] for key, item in citations.items())
    proposal['citations'][0]['evidence_id'] = 'fabricated'
    with pytest.raises(planner.PlanEvidenceError, match='unknown source excerpt'):
        planner._citations(proposal, packet, fixture['workspace'])


def test_inventory_proves_existence_but_not_counts_and_binds_file_bytes(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    proposal = {'citations': [{'id': 'file', 'source': 'inventory', 'path': 'dataset.json', 'quote': 'dataset.json'}]}
    accepted = planner._citations(proposal, packet, fixture['workspace'])['file']
    assert accepted['evidence_scope'] == 'file_existence_only'
    assert accepted['sha256'] == fixture['snapshot']['files']['dataset.json']
    assert not planner._has_value(12, [accepted])
    assert not planner._has_value('dataset.json', [accepted])
    proposal['citations'][0]['source'] = 'dataset'
    with pytest.raises(planner.PlanEvidenceError, match='Use source=inventory'):
        planner._citations(proposal, packet, fixture['workspace'])
    proposal['citations'][0]['source'] = 'inventory'
    (fixture['workspace'] / 'dataset.json').write_bytes(b'{}')
    with pytest.raises(planner.PlanEvidenceError, match='changed after the snapshot'):
        planner._citations(proposal, packet, fixture['workspace'])


def test_only_binary_integer_mask_names_get_selected_sample_counts(tmp_path):
    fixture = make_experiment(tmp_path)
    path = fixture['workspace'] / 'split.npz'
    np.savez(path, test_mask=np.array([0, 1, 1, 0], dtype=np.uint8),
             train_indices=np.array([0, 1], dtype=np.int64),
             invalid_mask=np.array([0, 2, 1]), float_mask=np.array([0., 1.]))
    result = planner._read_dataset_summary(fixture['workspace'], path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    arrays = json.loads(result['text'])['arrays']
    assert arrays['test_mask']['selected_samples'] == 2
    assert 'selected_samples' not in arrays['train_indices']
    assert 'selected_samples' not in arrays['invalid_mask']
    assert 'selected_samples' not in arrays['float_mask']


def test_selected_split_missing_from_initial_budget_can_be_read_on_demand(tmp_path):
    fixture = make_experiment(tmp_path)
    path = fixture['workspace'] / 'split_0.6_0.2.npz'
    np.savez(path, train_mask=np.array([1, 1, 0, 0], dtype=np.uint8),
             val_mask=np.array([0, 0, 1, 0], dtype=np.uint8), test_mask=np.array([0, 0, 0, 1], dtype=np.uint8))
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['file_inventory'][path.name].pop('dataset_evidence')
    view = planner._planning_view(packet)
    assert view['dataset_catalog'][path.name]['summary_available'] is False
    assert path.name not in view['source_catalog']
    planner._add_requested_files([path.name], packet, fixture['workspace'])
    summary = packet['file_inventory'][path.name]['dataset_evidence']
    assert json.loads(summary['text'])['arrays']['test_mask']['selected_samples'] == 1
    assert json.loads(summary['text'])['arrays']['train_mask']['selected_samples'] == 2
    selected = next(key for key, entry in planner._planning_excerpts(packet).items() if entry.get('path') == path.name)
    citation = planner._citations({'citations': [{'id': 'selected_split', 'evidence_id': selected}]}, packet, fixture['workspace'])
    assert citation['selected_split']['sha256'] == fixture['snapshot']['files'][path.name]
    assert path.name not in packet['repository']['files']  # Never execute NPZ as author source.


@pytest.mark.parametrize('failure', ['changed', 'budget', 'object'])
def test_on_demand_dataset_read_never_bypasses_hash_budget_or_no_pickle(tmp_path, failure):
    fixture = make_experiment(tmp_path)
    path = fixture['workspace'] / 'split.npz'
    np.savez(path, test_mask=np.array([0, 1], dtype=np.uint8))
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['file_inventory'][path.name].pop('dataset_evidence')
    if failure == 'budget':
        packet['limits']['dataset_summary_bytes'] = 0
    else:
        np.savez(path, values=np.array([{'unsafe': 'object'}], dtype=object))
        if failure == 'object':
            packet['file_inventory'][path.name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    original = deepcopy(packet)
    with pytest.raises(planner.PlanEvidenceError):
        planner._add_requested_files([path.name], packet, fixture['workspace'])
    assert packet == original


def test_large_split_inventory_is_catalogued_without_unrelated_initial_numbers(tmp_path):
    fixture = make_experiment(tmp_path)
    for index in range(20):
        path = fixture['workspace'] / f'split_{index}.npz'
        np.savez(path, test_mask=np.array([1, 0], dtype=np.uint8))
        fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    view = planner._planning_view(packet)
    assert len(view['dataset_catalog']) == 20
    assert all(not item['summary_available'] for item in view['dataset_catalog'].values())
    assert not any(ref['source'] == 'dataset' for ref in view['evidence_excerpts'].values())
    planner._add_requested_files(['split_19.npz'], packet, fixture['workspace'])
    assert json.loads(packet['file_inventory']['split_19.npz']['dataset_evidence']['text'])['arrays']['test_mask']['selected_samples'] == 1
    assert planner._planning_view(packet)['dataset_catalog']['split_19.npz']['summary_available'] is True


def test_similarly_named_sources_have_distinct_compact_stable_ids(tmp_path):
    fixture = make_experiment(tmp_path)
    for name in ('cpu/train.py', 'gpu/train.py'):
        path = fixture['workspace'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('same = 1\n', encoding='utf-8')
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    refs = {key: ref for key, ref in planner._planning_excerpts(packet).items() if ref.get('path') in {'cpu/train.py', 'gpu/train.py'}}
    assert len(refs) == 2 and all(len(key) == 19 for key in refs)
    for key, ref in refs.items():
        citation = planner._citations({'citations': [{'id': key, 'evidence_id': key}]}, packet, fixture['workspace'])[key]
        assert citation['path'] == ref['path']
    before = set(refs)
    packet['repository']['files'].pop('README.md')
    assert before.issubset(planner._planning_excerpts(packet))


def reader_navigation(packet):
    text = '\n'.join(page['text'] for page in packet['pdf']['pages'])
    start = text.rfind(packet['pdf']['pages'][-1]['text'])
    return {'pdf_input': {'sha256': packet['pdf']['sha256']},
        'reading_coverage': {'complete': True, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'read_characters': len(text), 'spans': [[0, len(text)]]},
        'reading_originals': {'original': {'start': start, 'end': len(text), 'text': text[start:]}},
        'reading_batches': [{'notes': [{'detail': 'The actual result table belongs to the stated experiment.',
                                       'source_ids': ['original']}]}]}


def test_full_reader_navigation_uses_actual_pdf_page_citation_ids(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    planner.bind_full_paper_notes(packet, reader_navigation(packet))
    note = packet['paper_reading']['notes'][0]
    refs = planner._citations({'citations': [{'id': key, 'evidence_id': key} for key in note['evidence_ids']]},
                             packet, fixture['workspace'])
    assert refs and all(ref['source'] == 'pdf' and ref['page'] == 3 for ref in refs.values())
    assert packet['paper_reading']['purpose'] == 'navigation_only_verify_against_original_pdf_excerpts'


@pytest.mark.parametrize('mutation', ['pdf', 'text', 'gap', 'source'])
def test_reading_notes_never_substitute_another_pdf_or_incomplete_original_coverage(tmp_path, mutation):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    reader = reader_navigation(packet)
    if mutation == 'pdf':
        reader['pdf_input']['sha256'] = 'a'*64
    elif mutation == 'text':
        reader['reading_coverage']['text_sha256'] = 'a'*64
    elif mutation == 'gap':
        reader['reading_coverage']['spans'][0][0] = 1
    else:
        reader['reading_originals']['original']['text'] = 'fabricated'
    original = deepcopy(packet)
    with pytest.raises(planner.PlanEvidenceError):
        planner.bind_full_paper_notes(packet, reader)
    assert packet == original
