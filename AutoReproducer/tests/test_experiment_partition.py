"""Real mask arithmetic overrides a model's assertion that ratios agree."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from discovered_fixtures import make_experiment, write_pdf
from src.discovered_repository_plan import PlanEvidenceError, build_evidence_packet
from src.experiment_partition import verify_partition


def partition_fixture(tmp_path, ratios=(60, 20, 20), per_class=True, skew=False):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    labels = np.repeat(np.arange(4), 50)
    masks = [np.zeros(200, dtype=np.int8) for _ in range(3)]
    if skew:
        starts = [0, ratios[0] * 2, (ratios[0] + ratios[1]) * 2, 200]
        for index, mask in enumerate(masks):
            mask[starts[index]:starts[index + 1]] = 1
    else:
        for label in range(4):
            start = label * 50
            sizes = [int(ratios[0] / 100 * 50), int(ratios[1] / 100 * 50)]
            offsets = [start, start + sizes[0], start + sum(sizes), start + 50]
            for index, mask in enumerate(masks):
                mask[offsets[index]:offsets[index + 1]] = 1
    np.savez(workspace / 'split.npz', train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])
    (workspace / 'labels.txt').write_text('node_id\tfeature\tlabel\n' + ''.join(
        f'{index}\t0\t{label}\n' for index, label in enumerate(labels)), encoding='utf-8')
    for name in ('split.npz', 'labels.txt'):
        fixture['snapshot']['files'][name] = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
    sentence = ('We split nodes of each class into ' if per_class else 'We split nodes into ')
    sentence += '60%, 20%, and 20% for training, validation and testing.'
    pdf = write_pdf(tmp_path / 'partition.pdf', [['Partition Fixture',
        'Our source code is available at ' + fixture['snapshot']['url'], sentence]])
    packet = build_evidence_packet(pdf, workspace, fixture['snapshot'])
    packet['experiment_protocol'] = {'dataset': {'citations': ['paper_split']},
        'citations': [{'id': 'paper_split', 'source': 'pdf', 'page': 1, 'quote': sentence}]}
    plan = {'datasets': [{'kind': 'bundled', 'paths': ['split.npz', 'labels.txt']}]}
    return workspace, packet, plan


def test_actual_stratified_masks_match_paper_percentages(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path)
    result = verify_partition(plan, packet, workspace)
    assert result['status'] == 'verified'
    assert result['rounding_bound_samples'] == 8
    assert result['observed'][0]['counts'] == {'train': 120, 'validation': 40, 'test': 40}
    assert len(result['observed'][0]['per_class_counts']) == 4


def test_48_32_20_is_not_60_20_20_even_when_filename_and_model_claim_match(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path, ratios=(48, 32, 20))
    with pytest.raises(PlanEvidenceError, match='conflict with paper percentages') as exc:
        verify_partition(plan, packet, workspace)
    assert exc.value.category == 'protocol_conflict'
    assert exc.value.planning_evidence['observed']['counts'] == {'train': 96, 'validation': 64, 'test': 40}


def test_matching_global_counts_do_not_prove_per_class_partition(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path, skew=True)
    with pytest.raises(PlanEvidenceError, match='per-class split counts conflict'):
        verify_partition(plan, packet, workspace)


def test_global_partition_does_not_require_labels_or_stratification(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path, per_class=False, skew=True)
    plan['datasets'][0]['paths'].remove('labels.txt')
    assert verify_partition(plan, packet, workspace)['status'] == 'verified'


def test_per_class_partition_cannot_invent_a_rounding_bound_without_labels(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path)
    plan['datasets'][0]['paths'].remove('labels.txt')
    with pytest.raises(PlanEvidenceError, match='complete labels') as exc:
        verify_partition(plan, packet, workspace)
    assert exc.value.category == 'missing_information'


def test_changed_masks_fail_hash_validation(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path)
    (workspace / 'split.npz').write_bytes((workspace / 'split.npz').read_bytes() + b'changed')
    with pytest.raises(PlanEvidenceError, match='changed'):
        verify_partition(plan, packet, workspace)


def test_overlapping_masks_fail_even_with_individually_plausible_counts(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path)
    with np.load(workspace / 'split.npz', allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['val_mask'][0], arrays['val_mask'][30] = 1, 0
    np.savez(workspace / 'split.npz', **arrays)
    snapshot = {'url': packet['repository']['url'], 'revision': packet['repository']['revision'],
                'files': {name: item['sha256'] for name, item in packet['file_inventory'].items()}}
    snapshot['files']['split.npz'] = hashlib.sha256((workspace / 'split.npz').read_bytes()).hexdigest()
    fresh = build_evidence_packet(workspace.parent / 'partition.pdf', workspace, snapshot)
    fresh['experiment_protocol'] = packet['experiment_protocol']
    with pytest.raises(PlanEvidenceError, match='overlapping or unassigned'):
        verify_partition(plan, fresh, workspace)


def test_ambiguous_paper_declarations_cannot_choose_a_convenient_ratio(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path)
    protocol = deepcopy(packet['experiment_protocol'])
    protocol['citations'][0]['quote'] += ' We split nodes into 50%, 25%, and 25% for training, validation and testing.'
    packet['experiment_protocol'] = protocol
    with pytest.raises(PlanEvidenceError, match='ambiguous'):
        verify_partition(plan, packet, workspace)


def test_proven_split_conflict_stops_planner_without_retry_or_semantic_override(tmp_path, monkeypatch):
    from discovered_fixtures import PlannedLLM
    from src import discovered_repository_plan as planner
    workspace, packet, plan = partition_fixture(tmp_path, ratios=(48, 32, 20))
    monkeypatch.setattr(planner, 'validate_plan', lambda candidate, evidence, root:
                        verify_partition(candidate, evidence, root))
    llm = PlannedLLM(plan)
    with pytest.raises(PlanEvidenceError) as exc:
        planner.propose_plan(llm, packet, workspace)
    assert exc.value.category == 'protocol_conflict'
    assert exc.value.planning_evidence['stage'] == 'partition_validation'
    assert [kwargs['task'] for _, kwargs in llm.calls] == ['discovered_repository_plan']


def test_integer_rounding_is_allowed_without_percentage_tolerance(tmp_path):
    workspace, packet, plan = partition_fixture(tmp_path, per_class=False)
    with np.load(workspace / 'split.npz', allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['train_mask'][0], arrays['test_mask'][0] = 0, 1
    np.savez(workspace / 'split.npz', **arrays)
    from src.discovered_repository_plan import _read_dataset_summary
    entry = packet['file_inventory']['split.npz']
    entry['sha256'] = hashlib.sha256((workspace / 'split.npz').read_bytes()).hexdigest()
    entry['dataset_evidence'] = _read_dataset_summary(workspace, 'split.npz', entry['sha256'])
    result = verify_partition(plan, packet, workspace)
    assert result['status'] == 'verified'
    assert result['observed'][0]['counts'] == {'train': 119, 'validation': 40, 'test': 41}
