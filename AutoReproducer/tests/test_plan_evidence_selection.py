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
