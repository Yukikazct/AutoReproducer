"""Native PyG record and copied-cache mechanics, not a paper acceptance run."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

import pytest

from src.author_values import geometric_data_fields, resolve_author_value
from src.discovered_repository_runtime import verify_capture
from src.native_loop import resolve_module_loop
from test_discovered_module_loop import use_function_loop
from test_discovered_repository_series import run_series
from test_discovered_repository_runtime import run_fixture


@pytest.fixture
def native_support(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / 'src'))
    spec = importlib.util.spec_from_file_location('_graph_capture_mechanics',
                                                 project / 'src/experiments/discovered_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pyg_record_preserves_all_fields_and_raw_selectors(native_support):
    import torch
    from torch_geometric.data import Data
    data = Data(x=torch.arange(6.).reshape(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]),
                y=torch.tensor([1, 0, 1]), test_mask=torch.tensor([True, False, True]), num_nodes=3)
    mapping = geometric_data_fields(data)
    assert resolve_author_value({'data': data}, {'name': 'data', 'attributes': ['y']}) is mapping['y']
    tensors = []
    layout = native_support.flatten(data, tensors)
    restored = native_support.reconstruct(layout, tensors)
    assert type(restored) is Data and set(geometric_data_fields(restored)) == set(mapping)
    for name, value in mapping.items():
        copied = geometric_data_fields(restored)[name]
        assert torch.equal(value, copied) if isinstance(value, torch.Tensor) else value == copied
    data.x[0, 0] = 99
    assert restored.x[0, 0] == 0  # The author record remains separate from the export.


def test_pyg_subclasses_and_properties_cannot_replace_direct_storage(native_support):
    import torch
    from torch_geometric.data import Data
    class Derived(Data):
        @property
        def invented(self):
            raise AssertionError('Getter must not run')
    data = Derived(x=torch.ones(2, 1))
    assert geometric_data_fields(data) is None
    with pytest.raises(ValueError, match='directly stored'):
        resolve_author_value({'data': data}, {'name': 'data', 'attributes': ['invented']})
    with pytest.raises(ValueError, match='real tensor trees'):
        native_support.flatten(data, [])
    original = Data(x=torch.ones(2, 1))
    original._store.__dict__['_mapping'] = []
    with pytest.raises(ValueError, match='complete directly stored'):
        geometric_data_fields(original)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_copied_cache_migration_keeps_aliases_registered_tensors_and_author_instance(native_support, device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Requires native CUDA')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, device=device))
            self.register_buffer('offset', torch.full((2,), 2., device=device))
            cache = torch.full((2,), 3., device=device)
            self.cache = {'values': [cache, self.offset, self.weight]}
            self.cache_alias = self.cache
            self.child = torch.nn.Identity()
            self.child.shared_cache = self.cache
        def forward(self, x):
            assert self.cache is self.cache_alias is self.child.shared_cache
            assert self.cache['values'][1] is self.offset
            assert self.cache['values'][2] is self.weight
            return x * self.weight + self.offset + self.cache['values'][0]
    author = Model().train()
    x = torch.full((2,), 4., device=device)
    original = author(x).detach().cpu()
    copied = native_support.cpu_snapshot(deepcopy(author)).eval()
    assert torch.equal(copied(x.cpu()), original)
    assert all(value.device.type == 'cpu' for value in copied.cache['values'])
    assert author.training and author.weight.device.type == device
    copied.cache['values'][0].zero_()
    assert torch.equal(author.cache['values'][0], torch.full((2,), 3., device=device))


@pytest.mark.parametrize('kind', ['cycle', 'depth'])
def test_cache_cycles_and_excessive_nesting_fail_explicitly(native_support, kind):
    import torch
    model = torch.nn.Linear(1, 2)
    model.cache = []
    if kind == 'cycle':
        model.cache.append(model.cache)
    else:
        for _ in range(20):
            model.cache = [model.cache]
    with pytest.raises(ValueError, match='cycle|nesting bound'):
        native_support.cpu_snapshot(deepcopy(model))


def use_pyg_function_loop(fixture):
    use_function_loop(fixture)
    source = (fixture['workspace'] / 'train.py').read_text('utf-8')
    begin = source.index('class AuthorInputs:')
    end = source.index('nothing = None', begin)
    source = source[:begin] + '''from torch_geometric.data import Data
bundle = Data(x=features, y=labels, indices=idx_test,
    edge_index=torch.stack([torch.arange(len(features), device=device)] * 2))
''' + source[end:]
    source = re.sub(r'\bidx\b', '_', source)
    source = source.replace('model(features, nothing)', 'model(bundle, nothing)')
    source = source.replace('labels[:4]', 'bundle.y[:4]').replace('labels[idx_val]', 'bundle.y[idx_val]')
    source = source.replace('labels[idx_test]', 'bundle.y[bundle.indices]')
    source = source.replace('\n            if val_loss < best_val_loss:',
        "\n            eval_info = {'val_loss': val_loss}\n            if eval_info['val_loss'] < best_val_loss:")
    model = '''import torch
from torch_geometric.utils import scatter
class SignalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.cached = None
        self.cached_alias = None
    def forward(self, data, unused=None):
        assert unused is None
        if self.cached is None:
            self.cached = torch.ones_like(data.x)
            self.cached_alias = self.cached
        assert self.cached is self.cached_alias
        aggregate = scatter(data.x[data.edge_index[0]], data.edge_index[1], dim=0, dim_size=data.x.shape[0])
        return self.linear((data.x + aggregate) * .5 * self.cached)
'''
    for name, text in {'train.py': source, 'author_model.py': model}.items():
        path = fixture['workspace'] / name
        path.write_text(text, encoding='utf-8')
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        next(c for c in fixture['proposal']['citations'] if c.get('path') == name)['quote'] = text
    capture = fixture['proposal']['capture']
    capture['scope']['loop_counter'] = '_'
    capture['scope']['selection']['validation_metric'] = {'name': 'eval_info', 'keys': ['val_loss']}
    capture.update(forward_args=['bundle', 'nothing'],
                   labels={'name': 'bundle', 'attributes': ['y']})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pyg_native_repetitions_dictionary_validation_and_fresh_reload(tmp_path, device):
    fixture, plan, spec_hash, execution = run_series(tmp_path, device, use_pyg_function_loop)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['is_reproduced'] and result['independent_metrics_pass']
    assert [row['accuracy'] for row in result['independent_metrics']['runs']] == [1., 1., 1.]
    assert [row['optimizer_steps'] for row in result['capture']['runs']] == [8, 10, 12]
    assert all(row['validation_selection']['optimizer_steps'] < row['optimizer_steps']
               for row in result['capture']['runs'])


def test_single_native_entrypoint_exports_a_complete_pyg_argument(tmp_path):
    def use_record(fixture):
        workspace = fixture['workspace']
        model = (workspace / 'author_model.py').read_text('utf-8')
        model = model.replace('forward(self, features)', 'forward(self, data)').replace(
            'self.linear(features)', 'self.linear(data.x)')
        source = (workspace / 'train.py').read_text('utf-8')
        source = source.replace('model = SignalModel()',
            'from torch_geometric.data import Data\ndata = Data(x=features, y=labels)\nmodel = SignalModel()')
        source = source.replace('model(features[:8])', 'model(data)[:8]').replace('model(features)', 'model(data)')
        for name, text in {'train.py': source, 'author_model.py': model}.items():
            path = workspace / name
            path.write_text(text, encoding='utf-8')
            fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            next(c for c in fixture['proposal']['citations'] if c.get('path') == name)['quote'] = text
        fixture['proposal']['capture']['forward_args'] = ['data']
    fixture, plan, spec_hash, _, execution = run_fixture(tmp_path, mutate=use_record)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['is_reproduced'] and result['independent_metrics']['predictions_match']


@pytest.mark.parametrize('key', ['test_loss', True, -1, 'val_loss()'])
def test_validation_selector_cannot_change_or_execute_the_author_comparison(tmp_path, key):
    from test_discovered_repository_series import series_fixture
    fixture = series_fixture(tmp_path)
    use_pyg_function_loop(fixture)
    scope = deepcopy(fixture['proposal']['capture']['scope'])
    scope['selection']['validation_metric']['keys'] = [key]
    with pytest.raises(ValueError):
        resolve_module_loop((fixture['workspace'] / 'train.py').read_text('utf-8'), scope)
