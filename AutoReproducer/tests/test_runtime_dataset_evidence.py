"""Invented framework objects test observation mechanics, never real-paper success."""
from copy import deepcopy
import hashlib
from pathlib import Path
import runpy
import sys
import types

import pytest

from src import discovered_repository_plan as planner
from src.runtime_dataset_evidence import compile_runtime_dataset, verify_runtime_datasets
from src.experiments.observe_datasets import DatasetObserver
from discovered_fixtures import make_experiment


def test_constructor_catalog_distinguishes_helper_from_its_wrapper(tmp_path):
    from src.runtime_dataset_evidence import discover_runtime_constructors
    workspace, packet, item, citations = setup(tmp_path)
    source = '''from torch_geometric.datasets import WikiCS
def actual_loader(path):
    return WikiCS(root=path)
def get_dataset(path):
    return actual_loader(path)
'''
    (workspace / 'data_loader.py').write_bytes(source.encode())
    stored = packet['repository']['files']['data_loader.py']
    stored.update(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    citations['loader']['quote'] = source
    options = discover_runtime_constructors({'data_loader.py': stored})
    assert [option['function'] for option in options] == ['actual_loader']
    with pytest.raises(ValueError, match='actual constructor scopes: actual_loader'):
        compile_runtime_dataset(item, packet, workspace, citations)
    item['function'] = 'actual_loader'
    assert compile_runtime_dataset(item, packet, workspace, citations)['function'] == 'actual_loader'


@pytest.mark.parametrize('declaration,constructor', [
    ('from torch_geometric import datasets as original', 'original.WikiCS'),
    ('import torch_geometric.datasets as original', 'original.WikiCS'),
    ('import torch_geometric', 'torch_geometric.datasets.WikiCS'),
])
def test_original_qualified_constructor_is_observed(tmp_path, monkeypatch, declaration, constructor):
    workspace, packet, item, citations = setup(tmp_path)
    source = f'{declaration}\ndef get_dataset(path):\n    return {constructor}(root=path)\n'
    (workspace / 'data_loader.py').write_bytes(source.encode())
    packet['repository']['files']['data_loader.py'].update(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    citations['loader']['quote'] = source
    assert compile_runtime_dataset(item, packet, workspace, citations)['dataset_class'] == 'WikiCS'


def test_literal_factory_selection_still_requires_actual_original_constructor(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    source = '''from torch_geometric import datasets
def get_dataset(path):
    factories = {'original': (datasets.WikiCS, 'WikiCS')}
    dataset_class, name = factories['original']
    return dataset_class(root=path)
dataset = get_dataset(DATA_ROOT)
'''
    (workspace / 'data_loader.py').write_bytes(source.encode())
    packet['repository']['files']['data_loader.py'].update(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    citations['loader']['quote'] = source
    compiled = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    observer = DatasetObserver([compiled], workspace, lambda: 0)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'real_constructor'})
        records = observer.finish()
    finally:
        observer.close()
    assert len(verify_runtime_datasets({'datasets': [compiled]}, [{'native_datasets': records}], workspace)) == 1
    # Mentioning a framework class in a table does not make an unrelated call
    # the original constructor. This rejected source is never executed.
    invalid = source.replace('return dataset_class(root=path)', 'return unrelated(root=path)')
    (workspace / 'data_loader.py').write_bytes(invalid.encode())
    packet['repository']['files']['data_loader.py'].update(text=invalid, sha256=hashlib.sha256(invalid.encode()).hexdigest())
    citations['loader']['quote'] = invalid
    with pytest.raises(ValueError, match='no original constructor'):
        compile_runtime_dataset(item, packet, workspace, citations)


SOURCE = '''from torch_geometric.datasets import WikiCS
def get_dataset(path):
    return WikiCS(root=path)
dataset = get_dataset(DATA_ROOT)
'''


def setup(tmp_path):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    (workspace / 'data_loader.py').write_bytes(SOURCE.encode())
    fixture['snapshot']['files']['data_loader.py'] = hashlib.sha256(SOURCE.encode()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    if 'data_loader.py' not in packet['repository']['files']:
        planner._add_requested_files(['data_loader.py'], packet, workspace)
    packet['author_execution_only'] = True
    item = {'kind': 'author_runtime', 'path': 'data_loader.py', 'function': 'get_dataset',
            'dataset_class': 'WikiCS', 'dataset_name': 'WikiCS', 'citations': ['loader']}
    citations = {'loader': {'source': 'repository', 'path': 'data_loader.py', 'quote': SOURCE}}
    return workspace, packet, item, citations


def fake_framework(monkeypatch):
    class WikiCS:
        __module__ = 'torch_geometric.datasets.invented_test'
        def __init__(self, root):
            self.root = str(root)
            root = Path(root)
            (root / 'raw').mkdir(parents=True, exist_ok=True)
            (root / 'processed').mkdir(exist_ok=True)
            (root / 'raw/data.json').write_bytes(b'invented raw test fixture')
            (root / 'processed/data.pt').write_bytes(b'invented processed test fixture')
            self.raw_paths, self.processed_paths = [str(root / 'raw/data.json')], [str(root / 'processed/data.pt')]
        def __len__(self):
            return 1
    module = types.ModuleType('torch_geometric.datasets')
    module.WikiCS = WikiCS
    monkeypatch.setitem(sys.modules, 'torch_geometric', types.ModuleType('torch_geometric'))
    monkeypatch.setitem(sys.modules, 'torch_geometric.datasets', module)
    return module


def test_native_constructor_records_complete_files_before_updates(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    module = fake_framework(monkeypatch)
    original = module.WikiCS.__init__
    observer = DatasetObserver([declaration], workspace, lambda: 0)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'actual_data'})
        records = observer.finish()
    finally:
        observer.close()
    assert module.WikiCS.__init__ is original
    assert len(records) == 1 and records[0]['sample_count'] == 1
    assert {record['role'] for record in records[0]['files'].values()} == {'raw', 'processed'}
    receipts = [{'native_datasets': records}]
    verified = verify_runtime_datasets({'datasets': [declaration]}, receipts, workspace)
    assert verified[0]['verified'] is True
    path = Path(next(iter(records[0]['files'])))
    path.write_bytes(b'changed input')
    with pytest.raises(ValueError, match='files changed'):
        verify_runtime_datasets({'datasets': [declaration]}, receipts, workspace)


def test_wrong_scope_and_data_created_after_training_are_not_complete(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    observer = DatasetObserver([declaration], workspace, lambda: 12)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'late_data'})
        with pytest.raises(ValueError, match='before actual training'):
            observer.finish()
    finally:
        observer.close()
    declaration['function'] = '<module>'
    observer = DatasetObserver([declaration], workspace, lambda: 0)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'wrong_scope'})
        # Only the immediate author caller establishes the constructor scope.
        assert observer.records == []
    finally:
        observer.close()


def test_repeated_native_constructor_after_training_retains_real_file_checks(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    updates = [0]
    observer = DatasetObserver([declaration], workspace, lambda: updates[0])
    try:
        for count in (0, 12):
            updates[0] = count
            runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'repeated_data'})
        records = observer.finish()
    finally:
        observer.close()
    assert [record['loaded_before_training'] for record in records] == [True, False]
    plan, receipts = {'datasets': [declaration]}, [{'native_datasets': records}]
    assert verify_runtime_datasets(plan, receipts, workspace)[0]['verified'] is True
    late_only = deepcopy(records[1:])
    with pytest.raises(ValueError, match='pre-training construction'):
        verify_runtime_datasets(plan, [{'native_datasets': late_only}], workspace)
    Path(next(iter(records[-1]['files']))).write_bytes(b'changed after second construction')
    with pytest.raises(ValueError, match='files changed'):
        verify_runtime_datasets(plan, receipts, workspace)


def test_changed_data_and_unobserved_loaders_cannot_finish(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    observer = DatasetObserver([declaration], workspace, lambda: 0)
    try:
        with pytest.raises(ValueError, match='not constructed'):
            observer.finish()
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'data'})
        Path(next(iter(observer.records[0]['files']))).write_bytes(b'changed')
        with pytest.raises(ValueError, match='changed during'):
            observer.finish()
    finally:
        observer.close()


@pytest.mark.parametrize('change', [
    lambda p, i: p.pop('author_execution_only'),
    lambda p, i: i.update(dataset_class='ArbitraryDataset'),
    lambda p, i: i.update(function='unknown'),
    lambda p, i: i.update(dataset_name='InventedOtherData'),
    lambda p, i: i.update(generated_code='return data'),
])
def test_native_source_and_identity_guards(tmp_path, change):
    workspace, packet, item, citations = setup(tmp_path)
    change(packet, item)
    with pytest.raises(ValueError):
        compile_runtime_dataset(item, packet, workspace, citations)


def test_receipt_without_observed_real_files_is_not_verified(tmp_path):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    with pytest.raises(ValueError, match='No original native dataset'):
        verify_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': []}], workspace)


def test_function_in_original_main_guard_is_a_real_native_scope(tmp_path):
    workspace, packet, item, citations = setup(tmp_path)
    source = "from torch_geometric.datasets import WikiCS\nif __name__ == '__main__':\n    def get_dataset(path):\n        return WikiCS(root=path)\n"
    (workspace / 'data_loader.py').write_bytes(source.encode())
    packet['repository']['files']['data_loader.py'].update(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    citations['loader']['quote'] = source
    compiled = compile_runtime_dataset(item, packet, workspace, citations)
    assert compiled['function'] == 'get_dataset' and compiled['constructor_quotes'] == ['WikiCS(root=path)']
