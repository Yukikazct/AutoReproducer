"""Bundled file discovery and feedback mechanics, never paper reproduction."""
from copy import deepcopy
import hashlib

import pytest

from discovered_fixtures import make_experiment
from src import discovered_repository_plan as planner


SUFFIXES = ('x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index')


def raw_paths(dataset='cora', root='main/data/cora/cora/raw'):
    return [f'{root}/ind.{dataset}.{suffix}' for suffix in SUFFIXES]


def bundled_experiment(tmp_path):
    fixture = make_experiment(tmp_path)
    for name in raw_paths():
        path = fixture['workspace'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # These are invented inventory bytes, not usable Planetoid data.
        raw = b'invented file inventory fixture'
        path.write_bytes(raw)
        fixture['snapshot']['files'][name] = hashlib.sha256(raw).hexdigest()
    fixture['packet'] = planner.build_evidence_packet(
        fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    return fixture


@pytest.mark.parametrize('dataset', ['cora', 'citeseer', 'pubmed'])
def test_complete_same_directory_files_offer_bundled_data(dataset):
    paths = raw_paths(dataset)
    options = planner._bundled_planetoid_options({'file_inventory': dict.fromkeys(paths)})
    assert len(options) == 1
    assert options[0]['kind'] == 'bundled'
    assert options[0]['dataset'] == dataset
    assert set(options[0]['paths']) == set(paths)


@pytest.mark.parametrize('change', ['missing_file', 'different_directory', 'different_dataset'])
def test_incomplete_or_mixed_sets_do_not_offer_bundled_data(change):
    paths = raw_paths()
    last = paths.pop()
    if change == 'different_directory':
        paths.append(last.replace('main/data/', 'other/data/'))
    elif change == 'different_dataset':
        paths.append(last.replace('ind.cora.', 'ind.citeseer.'))
    assert planner._bundled_planetoid_options({'file_inventory': dict.fromkeys(paths)}) == []


def test_planning_view_recomputes_capabilities_without_mutating_packet(tmp_path):
    fixture = bundled_experiment(tmp_path)
    packet = fixture['packet']
    packet['program_detected_bundled_planetoid'] = [{'paths': ['forged.pkl']}]
    before = deepcopy(packet)
    view = planner._planning_view(packet)
    assert set(view['program_detected_bundled_planetoid'][0]['paths']) == set(raw_paths())
    assert packet == before
    del packet['file_inventory'][raw_paths()[0]]
    assert planner._planning_view(packet)['program_detected_bundled_planetoid'] == []


def test_bundled_option_cannot_establish_split_counts(tmp_path):
    fixture = bundled_experiment(tmp_path)
    refs = [{'source': 'inventory', 'path': name, 'quote': name} for name in raw_paths()]
    assert not planner._has_value(1000, refs)
    assert not planner._has_value('random split', refs)


def test_bundled_data_still_requires_source_citations_and_unchanged_bytes(tmp_path):
    fixture = bundled_experiment(tmp_path)
    declaration = {'kind': 'bundled', 'paths': raw_paths(), 'citations': ['loader']}
    refs = {'loader': {'source': 'repository', 'quote': 'original loader fixture'}}
    result = planner._datasets([declaration], refs, fixture['packet'], fixture['workspace'])
    assert result[0]['files_sha256'] == {
        name: fixture['snapshot']['files'][name] for name in raw_paths()}
    with pytest.raises(planner.PlanEvidenceError, match='repository citation'):
        planner._datasets([declaration], {'loader': {'source': 'inventory', 'quote': raw_paths()[0]}},
                          fixture['packet'], fixture['workspace'])
    (fixture['workspace'] / raw_paths()[0]).write_bytes(b'changed')
    with pytest.raises(planner.PlanEvidenceError, match='changed'):
        planner._datasets([declaration], refs, fixture['packet'], fixture['workspace'])


@pytest.mark.parametrize('complete', [True, False])
def test_unsupported_loader_feedback_offers_only_existing_complete_sets(tmp_path, monkeypatch, complete):
    from src import author_dataset_loader
    fixture = bundled_experiment(tmp_path)
    if not complete:
        del fixture['packet']['file_inventory'][raw_paths()[0]]
    declaration = {'kind': 'author_loader', 'path': 'train.py', 'function': 'load', 'citations': ['loader']}
    refs = {'loader': {'source': 'repository', 'quote': 'original loader fixture'}}
    train = fixture['proposal']['steps']
    before = deepcopy(train)

    def unsupported(*args):
        raise author_dataset_loader.AuthorLoaderUnsupported('Loader module has an unreviewed top-level import')

    monkeypatch.setattr(author_dataset_loader, 'compile_author_loader', unsupported)
    monkeypatch.setattr(planner, '_effective_author_parameters', lambda *args: {})
    with pytest.raises(ValueError, match='unreviewed top-level import') as error:
        planner._datasets([declaration], refs, fixture['packet'], fixture['workspace'], selected_steps=train)
    assert ('kind=bundled' in str(error.value)) is complete
    if complete:
        assert all(name in str(error.value) for name in raw_paths())
        assert 'Preserve the selected author command and split' in str(error.value)
    assert train == before
