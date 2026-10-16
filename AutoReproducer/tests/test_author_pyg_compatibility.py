"""Real PyG legacy pickles, immutable inputs and bounded author retries."""
import hashlib
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from discovered_fixtures import configured_native_cache
from src.experiments.author_pyg_compatibility import LegacyPyGCompatibility, validate_repair


APP = Path(__file__).resolve().parents[1]
CREATE = r'''
import sys, torch
from torch_geometric.data import Data
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
y = torch.tensor([0, 1, 0])
graph = Data(x=x, edge_index=edges, y=y)
graph.train_mask = torch.tensor([True, False, False])
graph.val_mask = torch.tensor([False, True, False])
graph.test_mask = torch.tensor([False, False, True])
graph.train_percent = .6
if sys.argv[2] != 'modern':
    state = graph.to_dict()
    state.update(norm=None, pos=None, face=None, edge_attr=None)
    if sys.argv[2] == 'private': state['_unknown'] = 1
    Data.__getstate__ = lambda self: state
torch.save((graph, None), sys.argv[1])
'''
LOADER = '''from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset
class OldGraph(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def processed_paths(self):
        return [str(Path(__file__).with_name('graph.pt'))]
'''
DRIVER = r'''
import builtins, gc, hashlib, json, sys
from pathlib import Path
import torch
from src.experiments.author_pyg_compatibility import LegacyPyGCompatibility
root = Path(__file__).resolve().parent
config = json.loads((root/'config.json').read_text())
original_import = builtins.__import__
scope = LegacyPyGCompatibility(root, config['sources'], config.get('repairs', []),
    lambda: config.get('updates', 0), enabled=True)
from loader import OldGraph
from torch_geometric.data import InMemoryDataset
original_get = scope.original
def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
try:
    dataset = OldGraph()
    original = dataset._data
    state = original.to_dict() if '_store' in vars(original) else dict(vars(original))
    before = {key: digest(value) for key,value in state.items() if type(value) is torch.Tensor}
    data = dataset[0]
    gc.collect()
    assert dataset._data is original
    assert dataset._data._store._parent() is original
    assert data._store._parent() is data
    assert data.num_nodes == 3
    assert before == {key: digest(getattr(data,key)) for key in before}
    assert all(getattr(data,key) is None for key,value in state.items() if value is None)
    assert data.train_percent == .6
    result = {'success': True, 'applied': scope.applied, 'tensors_preserved': True}
except Exception as error:
    result = {'success': False, 'error': str(error), 'candidate': scope.diagnose(error), 'applied': scope.applied}
finally:
    scope.close()
assert builtins.__import__ is original_import
assert InMemoryDataset.get is original_get
Path(sys.argv[1]).write_text(json.dumps(result))
'''


@pytest.fixture
def native_env():
    cache = configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu')
    if cache is None:
        pytest.skip('Requires a dedicated verified CPU / PyG cache outside application data')
    return {**os.environ, 'PYTHONPATH': os.pathsep.join([str(APP), str(cache)]), 'CUDA_VISIBLE_DEVICES': ''}


def case(tmp_path, native_env, kind='legacy'):
    root = tmp_path / 'author'
    root.mkdir()
    (root / 'loader.py').write_text(LOADER, encoding='utf-8')
    (root / 'train.py').write_text(DRIVER, encoding='utf-8')
    created = subprocess.run([sys.executable, '-c', CREATE, str(root / 'graph.pt'), kind],
        env=native_env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert created.returncode == 0, created.stderr
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('loader.py', 'train.py')}
    return root, sources


def invoke(root, sources, native_env, repairs=(), updates=0):
    before = hashlib.sha256((root / 'graph.pt').read_bytes()).hexdigest()
    (root / 'config.json').write_text(json.dumps({'sources': sources, 'repairs': list(repairs), 'updates': updates}))
    output = root.parent / 'result.json'
    result = subprocess.run([sys.executable, str(root / 'train.py'), str(output)], cwd=root,
        env=native_env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256((root / 'graph.pt').read_bytes()).hexdigest() == before
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected for name, expected in sources.items())
    return json.loads(output.read_text())


def test_actual_failure_then_recorded_upgrade_preserves_tensor_bytes_masks_and_identity(tmp_path, native_env):
    root, sources = case(tmp_path, native_env)
    failed = invoke(root, sources, native_env)
    assert not failed['success'] and 'older version of PyG' in failed['error']
    fix = failed['candidate']
    assert fix['data_path'] == 'graph.pt'
    assert fix['dataset_path'] == 'loader.py'
    assert fix['caller_path'] == 'train.py'
    assert fix['state_schema']['train_mask']['dtype'] == 'torch.bool'
    validate_repair(root, sources, fix)
    completed = invoke(root, sources, native_env, [fix])
    assert completed['success'] and completed['tensors_preserved']
    assert completed['applied'] == [fix]


def test_modern_data_is_not_converted_or_recorded(tmp_path, native_env):
    root, sources = case(tmp_path, native_env, 'modern')
    result = invoke(root, sources, native_env)
    assert result['success'] and result['applied'] == []


def test_unknown_private_legacy_fields_remain_an_original_failure(tmp_path, native_env):
    root, sources = case(tmp_path, native_env, 'private')
    result = invoke(root, sources, native_env)
    assert not result['success'] and result['candidate'] is None and not result['applied']


def test_repair_is_not_applied_after_training_updates(tmp_path, native_env):
    root, sources = case(tmp_path, native_env)
    failed = invoke(root, sources, native_env)
    result = invoke(root, sources, native_env, [failed['candidate']], updates=1)
    assert not result['success'] and result['candidate'] is None and not result['applied']


def test_linked_processed_input_is_not_authorized(tmp_path, native_env, require_symlinks):
    root, sources = case(tmp_path, native_env)
    source = root / 'graph.pt'
    moved = root / 'original.pt'
    source.rename(moved)
    source.symlink_to(moved)
    result = invoke(root, sources, native_env)
    assert not result['success'] and result['candidate'] is None and not result['applied']


@pytest.mark.parametrize('changed', ['loader.py', 'train.py', 'graph.pt'])
def test_recorded_context_rejects_changed_source_or_data(tmp_path, native_env, changed):
    root, sources = case(tmp_path, native_env)
    failed = invoke(root, sources, native_env)
    with (root / changed).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        validate_repair(root, sources, failed['candidate'])


def test_passive_observer_does_not_import_optional_framework(tmp_path):
    loaded = {name for name in sys.modules if name.startswith('torch_geometric')}
    original = builtins.__import__
    scope = LegacyPyGCompatibility(tmp_path, {}, [], lambda: 0, enabled=True)
    try:
        builtins.__import__('math')
        assert {name for name in sys.modules if name.startswith('torch_geometric')} == loaded
    finally:
        scope.close()
    assert builtins.__import__ is original


def test_full_service_retries_real_legacy_data_error_and_preserves_original_files(
        tmp_path, monkeypatch, native_env):
    from test_author_execution_plan import AuthorLLM
    from test_discovered_repository_service import setup_service, use_preinstalled_cache
    from src.method_adapters import read_json

    fixture, service, request, _ = setup_service(tmp_path, monkeypatch)
    workspace = fixture['workspace']
    (workspace / 'loader.py').write_text(LOADER, encoding='utf-8')
    created = subprocess.run([sys.executable, '-c', CREATE, str(workspace / 'graph.pt'), 'legacy'],
        env=native_env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert created.returncode == 0, created.stderr
    script = workspace / 'train.py'
    prefix = "from loader import OldGraph\ngraph = OldGraph()\ndata = graph[0]\nassert data.num_nodes == 3\nassert data.norm is None\n"
    text = prefix + script.read_text(encoding='utf-8')
    script.write_text(text, encoding='utf-8')
    for name in ('loader.py', 'graph.pt', 'train.py'):
        fixture['snapshot']['files'][name] = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = text
    service.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    cache = configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu')
    use_preinstalled_cache(monkeypatch, service, cache)
    result = service.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert len(data['pyg_repair_attempts']) == 1
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert len(data['validation']['pyg_compatibility']) == 1
    assert data['validation']['paper_metrics_verified'] is False
    repaired = Path(data['run_dir']) / 'repo'
    for name, expected in fixture['snapshot']['files'].items():
        assert hashlib.sha256((repaired / name).read_bytes()).hexdigest() == expected
    fix = data['pyg_repair_attempts'][0]
    failure = Path(fix['failed_execution_dir']) / 'train.stderr.log'
    assert 'older version of PyG' in failure.read_text(encoding='utf-8')
    config = repaired / '_autorepro_capture/config.json'
    manifest = read_json(Path(data['run_dir']) / 'adapter_manifest.json')
    assert manifest['files']['_autorepro_capture/config.json'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert any('旧 PyG' in warning for warning in data['validation']['warnings'])
