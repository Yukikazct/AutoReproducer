"""Invented framework objects test observation mechanics, never real-paper success."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

from src import discovered_repository_plan as planner
from src.runtime_dataset_evidence import audit_runtime_datasets, compile_runtime_dataset, verify_runtime_datasets
from src.experiments.observe_datasets import DatasetObserver, file_record
from discovered_fixtures import make_experiment
from test_discovered_repository_service import cpu_dependency_cache


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


def test_informational_missing_inputs_preserve_original_constructor_return(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    framework = fake_framework(monkeypatch)
    original = framework.WikiCS.__init__
    def cached(instance, root):
        original(instance, root)
        Path(instance.raw_paths[0]).unlink()
    framework.WikiCS.__init__ = cached
    observer = DatasetObserver([declaration], workspace, lambda: 0, strict=False)
    try:
        namespace = runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'cached_data'})
        assert isinstance(namespace['dataset'], framework.WikiCS)
        records = observer.finish()
    finally:
        observer.close()
    assert records and all(record['verified'] is False for record in records)
    audit = audit_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': records}], workspace)
    assert audit['verified'] is False and audit['independent_reference_verified'] is False
    assert audit['errors'][0]['observation_errors']


def test_informational_changed_inputs_keep_failure_evidence_without_replacing_return(tmp_path, monkeypatch):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    observer = DatasetObserver([declaration], workspace, lambda: 0, strict=False)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': tmp_path / 'changed_input'})
        Path(next(iter(observer.records[0]['files']))).write_bytes(b'changed during training')
        records = observer.finish()
    finally:
        observer.close()
    assert records[0]['verified'] is False
    assert 'changed during' in records[0]['verification_error']
    assert audit_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': records}], workspace)['verified'] is False


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


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_dataset_file_observation_rejects_filesystem_links(tmp_path, require_symlinks, kind):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'input.bin').write_bytes(b'original dataset bytes')
    link = tmp_path / 'link'
    if kind == 'file':
        link.symlink_to(original / 'input.bin')
        selected = link
    else:
        link.symlink_to(original, target_is_directory=True)
        selected = link / 'input.bin'
    with pytest.raises(ValueError, match='symlink|junction|链接'):
        file_record(selected, 'raw')


@pytest.mark.parametrize('during', ['observer_finish', 'parent_verification'])
def test_dataset_root_retargeting_cannot_keep_a_verified_receipt(
        tmp_path, monkeypatch, require_symlinks, during):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    fake_framework(monkeypatch)
    root = tmp_path / 'original_data'
    observer = DatasetObserver([declaration], workspace, lambda: 0)
    try:
        runpy.run_path(str(workspace / 'data_loader.py'), init_globals={'DATA_ROOT': root})
        records = observer.records
        original = tmp_path / 'same_bytes_at_different_location'
        assert root.resolve().is_relative_to(tmp_path.resolve())
        assert original.resolve().is_relative_to(tmp_path.resolve())
        root.rename(original)
        root.symlink_to(original, target_is_directory=True)
        with pytest.raises(ValueError, match='symlink|junction|链接'):
            if during == 'observer_finish':
                observer.finish()
            else:
                verify_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': records}], workspace)
    finally:
        observer.close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction semantics')
def test_windows_junction_is_rejected_without_symbolic_link_mode(tmp_path):
    target, junction = tmp_path / 'target', tmp_path / 'junction'
    target.mkdir()
    (target / 'input.bin').write_bytes(b'original bytes')
    assert target.resolve().is_relative_to(tmp_path.resolve())
    assert junction.absolute().is_relative_to(tmp_path.resolve())
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(target)],
                   capture_output=True, check=True)
    assert junction.is_dir() and not junction.is_symlink()
    with pytest.raises(ValueError, match='junction'):
        file_record(junction / 'input.bin', 'raw')


def test_real_pyg_dataset_files_still_verify_without_links(tmp_path, cpu_dependency_cache):
    workspace, packet, item, citations = setup(tmp_path)
    declaration = compile_runtime_dataset(item, packet, workspace, citations)
    root = tmp_path / 'native_wikics'
    (root / 'raw').mkdir(parents=True)
    # Invented tiny JSON exercises the actual fixed PyG loader, never paper data.
    payload = {'features': [[1., 0.], [0., 1.], [1., 1.]], 'labels': [0, 1, 0],
               'links': [[1], [0, 2], [1]], 'train_masks': [[True, False, False]],
               'val_masks': [[False, True, False]], 'stopping_masks': [[False, True, False]],
               'test_mask': [False, False, True]}
    raw = root / 'raw/data.json'
    raw.write_text(json.dumps(payload), encoding='utf-8')
    expected = hashlib.sha256(raw.read_bytes()).hexdigest()
    program = '''import json,runpy,sys
from pathlib import Path
from src.experiments.observe_datasets import DatasetObserver
workspace,root=Path(sys.argv[1]),Path(sys.argv[2])
declaration=json.loads(sys.argv[3])
observer=DatasetObserver([declaration],workspace,lambda:0)
try:
    runpy.run_path(str(workspace/'data_loader.py'),init_globals={'DATA_ROOT':root})
    print('NATIVE_DATASET_RECEIPT='+json.dumps(observer.finish()))
finally:
    observer.close()
'''
    environment = {**os.environ, 'PYTHONPATH': str(cpu_dependency_cache) + os.pathsep + str(Path.cwd())}
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', program,
        str(workspace), str(root), json.dumps(declaration)], env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert completed.returncode == 0, completed.stderr
    line = next(line for line in completed.stdout.splitlines() if line.startswith('NATIVE_DATASET_RECEIPT='))
    records = json.loads(line.split('=', 1)[1])
    record = records[0]
    assert record['sample_count'] == 1 and record['num_nodes'] == 3
    assert {value['role'] for value in record['files'].values()} == {'raw', 'processed'}
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == expected
    assert verify_runtime_datasets({'datasets': [declaration]}, [{'native_datasets': records}], workspace)[0]['verified']


def test_function_in_original_main_guard_is_a_real_native_scope(tmp_path):
    workspace, packet, item, citations = setup(tmp_path)
    source = "from torch_geometric.datasets import WikiCS\nif __name__ == '__main__':\n    def get_dataset(path):\n        return WikiCS(root=path)\n"
    (workspace / 'data_loader.py').write_bytes(source.encode())
    packet['repository']['files']['data_loader.py'].update(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    citations['loader']['quote'] = source
    compiled = compile_runtime_dataset(item, packet, workspace, citations)
    assert compiled['function'] == 'get_dataset' and compiled['constructor_quotes'] == ['WikiCS(root=path)']
