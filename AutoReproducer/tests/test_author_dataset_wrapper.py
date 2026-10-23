"""Invented source-bound dispatch and plain graph wrappers, without training."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from discovered_fixtures import configured_native_cache
from src.author_dataset_loader import AuthorLoaderUnsupported, compile_author_loader


WRAPPER = '''import definitely_unavailable_unrelated_dependency
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

class GraphRecord(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def unrelated(self, fraction=.5):
        return unavailable_original_method(fraction)

def load(name, normalize_features=False):
    if name in ('Cora', 'cora', 'CiteSeer', 'PubMed'):
        dataset = inner('./data', name, normalize_features)
    else:
        raise ValueError('Another data family')
    return dataset

def inner(folder, name, normalize=False):
    if normalize:
        transform = T.NormalizeFeatures()
        original = Planetoid(root=f'{folder}/graphs', name=name, transform=transform)
    else:
        original = Planetoid(f'{folder}/graphs', name)
    data = original[0]
    nodes = data.num_nodes
    print(f'Nodes: {nodes}')
    result = GraphRecord(name)
    result.graph = {'features': data.x, 'edges': data.edge_index, 'nodes': nodes, 'optional': None}
    result.label = data.y
    result.train = torch.where(data.train_mask)[0]
    result.validation = torch.where(data.val_mask)[0]
    result.test = torch.where(data.test_mask)[0]
    return result
'''
ENTRY = '''import argparse
from datasets import load
parser = argparse.ArgumentParser()
args = parser.parse_args()
print(args)
dataset = load(args.dataset, args.normalize_features)
model = parse_method(args)
evaluate(model, args)
args.dataset = 'Later training metadata'
'''


def compile_case(tmp_path, loader=WRAPPER, entry=ENTRY, name='Cora', normalize=False, extra_files=None):
    files = {}
    for path, text in {'scripts/datasets.py': loader, 'scripts/train.py': entry, **(extra_files or {})}.items():
        full = tmp_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(text.encode())
        files[path] = {'text': text, 'sha256': hashlib.sha256(full.read_bytes()).hexdigest()}
    result = compile_author_loader({'path': 'scripts/datasets.py', 'function': 'load'}, files, tmp_path,
        {'argv': ['python', 'train.py', '--dataset', name], 'cwd': 'scripts'},
        {'dataset': {'value': name}, 'normalize_features': {'value': normalize}})
    return result, {path: item['sha256'] for path, item in files.items()}


@pytest.mark.parametrize('name', ['Cora', 'cora', 'CiteSeer', 'PubMed'])
@pytest.mark.parametrize('normalize', [False, True])
def test_unchanged_helper_dispatch_relative_cwd_and_record_contract(tmp_path, name, normalize):
    result, sources = compile_case(tmp_path, name=name, normalize=normalize)
    assert result['name'] == name
    assert result['root'] == 'scripts/data/graphs'
    assert result['transform'] == ('NormalizeFeatures' if normalize else None)
    assert result['arguments'] == {'name': name, 'normalize_features': normalize}
    sliced = result['source_slice']
    assert sliced['cwd'] == 'scripts'
    assert sliced['returned_record']['class'] == 'GraphRecord'
    assert sliced['returned_record']['fields']['test'] == {'mask_indices': 'test_mask'}
    assert [item['kind'] for item in sliced['statements']].count('FunctionDef') == 2
    assert not any(item['line'] == 1 for item in sliced['statements'])
    assert any(item['line'] == 3 for item in sliced['statements']) is normalize
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())


@pytest.mark.parametrize('change', [
    lambda text: text.replace('class GraphRecord(object):', 'class GraphRecord(object, metaclass=side_effect()):'),
    lambda text: text.replace('class GraphRecord(object):', '@side_effect()\nclass GraphRecord(object):'),
    lambda text: text.replace('    def unrelated', '    @property\n    def unrelated'),
    lambda text: text.replace('def unrelated', 'def __new__'),
    lambda text: text.replace('def unrelated', 'def __setattr__'),
    lambda text: text.replace('def unrelated', 'def __getattribute__'),
    lambda text: text.replace('fraction=.5', 'fraction=side_effect()'),
    lambda text: text.replace('self.graph = {}', 'self.graph = side_effect()'),
    lambda text: text.replace('self.graph = {}', 'self.graph = {"bad": 1}'),
    lambda text: text.replace('data = original[0]', 'data = original[1]'),
    lambda text: text.replace('nodes = data.num_nodes', 'nodes = data.unknown_property'),
    lambda text: text.replace('data = original[0]', "another = Planetoid('./data/other', name)\n    data = original[0]"),
    lambda text: text.replace("inner('./data', name, normalize_features)", "inner('../../../outside', name, normalize_features)"),
    lambda text: text.replace('    return result', '    side_effect()\n    return result'),
    lambda text: text.replace("dataset = inner('./data', name, normalize_features)", 'dataset = load(name, normalize_features)'),
    lambda text: text.replace("'features': data.x", "'features': 'x'"),
    lambda text: text.replace('data = original[0]', 'data = arbitrary_index(original)'),
])
def test_unreviewed_wrapper_operations_are_rejected(tmp_path, change):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, loader=change(WRAPPER))


@pytest.mark.parametrize('parameter', ['*items', '**items'])
@pytest.mark.parametrize('location', ['record_method', 'module_function'])
def test_variadic_annotations_cannot_execute_during_loading(tmp_path, monkeypatch, parameter, location):
    effects = []
    monkeypatch.setattr(builtins, '_autorepro_header_effect', lambda: effects.append('executed'), raising=False)
    annotation = parameter + ": __import__('builtins')._autorepro_header_effect()"
    if location == 'record_method':
        source = WRAPPER.replace('def unrelated(self, fraction=.5):',
                                 'def unrelated(self, fraction=.5, ' + annotation + '):')
    else:
        source = WRAPPER + '\ndef unused(' + annotation + '):\n    pass\n'
    with pytest.raises(AuthorLoaderUnsupported, match='headers.*annotations'):
        compile_case(tmp_path, loader=source)
    assert effects == []


@pytest.mark.parametrize('prefix', [
    'mutate(args)', "args.dataset = 'PubMed'", 'alias = args\nmutate(alias)',
    'print = mutate\nprint(args)',
])
def test_namespace_mutation_or_escape_before_load_is_rejected(tmp_path, prefix):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', prefix))


def test_second_original_loading_call_after_training_is_rejected(tmp_path):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY + '\nother = load(args.dataset, args.normalize_features)\n')


NATIVE_DRIVER = r'''
import copy, hashlib, json, os, pickle, shutil, sys, types
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.io import fs
from src.experiments import author_dataset_prepare as prep
root = Path(sys.argv[1]).resolve()
config = json.loads((root/'config.json').read_text())
prep.WORKSPACE = root
fixture = root/'invented_raw'
fixture.mkdir()
count, features = 1704, 3
allx = sp.csr_matrix(np.ones((704, features), dtype=np.float32))
tx = sp.csr_matrix(np.ones((1000, features), dtype=np.float32))
labels = np.eye(2, dtype=np.int64)[np.arange(count) % 2]
values = {'x': allx[:4], 'tx': tx, 'allx': allx, 'y': labels[:4],
    'ty': labels[704:], 'ally': labels[:704], 'graph': {i: [(i+1)%count] for i in range(count)}}
for key,value in values.items():
    (fixture/('ind.cora.'+key)).write_bytes(pickle.dumps(value))
(fixture/'ind.cora.test.index').write_text('\n'.join(str(i) for i in range(704,count))+'\n')
download_calls=[]
def copy_fixture(url, destination, *args, **kwargs):
    assert url.startswith(Planetoid.url+'/')
    Path(destination).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture/url.rsplit('/',1)[1], Path(destination)/url.rsplit('/',1)[1])
    download_calls.append(url)
fs.cp=copy_fixture
original_init=Planetoid.__init__
cwd=Path.cwd()
result=prep.prepare(config)
assert Planetoid.__init__ is original_init and fs.cp is copy_fixture and Path.cwd()==cwd
assert len(download_calls)==8 and result['verified']
assert result['sources'][0]['nodes']==count
assert result['sources'][0]['split_counts']=={'train':4,'validation':500,'test':1000}
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in config['source_files'].items())
# Exercise direct storage comparisons with real Tensor/Data objects, including negative cases.
module=types.ModuleType('direct_record_test')
source=root/config['loaders'][0]['path']
decl=config['loaders'][0]['source_slice']
prep.original_slice(source,decl,module)
os.chdir(root/'scripts')
record=module.load('Cora',True)
data=Planetoid(str(root/'scripts/data/graphs'),'Cora',transform=prep.NormalizeFeatures())[0]
prep.verify_record(record,decl['returned_record'],module,data)
for change in ('shape','dtype','edges','indices','literal','extra','getter'):
    target=module.load('Cora',True)
    if change=='shape': target.label=target.label.reshape(-1,1)
    elif change=='dtype': target.label=target.label.float()
    elif change=='edges': target.graph['edges']=target.graph['edges'].flip(0)
    elif change=='indices': target.test=target.test[:-1]
    elif change=='literal': target.name='Other'
    elif change=='extra': target.extra=1
    elif change=='getter':
        class Other:
            @property
            def graph(self): raise AssertionError('A getter was invoked')
        target=Other()
    try: prep.verify_record(target,decl['returned_record'],module,data)
    except ValueError: pass
    else: raise AssertionError('Accepted invalid record '+change)
# Slices must be original, ordered and hash-bound.
for change in ('hash','duplicate','order'):
    altered=copy.deepcopy(decl)
    if change=='hash': altered['statements'][0]['sha256']='0'*64
    elif change=='duplicate': altered['statements'].append(altered['statements'][0])
    else: altered['statements'].reverse()
    try: prep.original_slice(source,altered,types.ModuleType('bad'))
    except ValueError: pass
    else: raise AssertionError('Accepted invalid source slice '+change)
# A pre-construction source-validation failure restores all installed observers.
bad=copy.deepcopy(config)
bad['loaders'][0]['source_slice']['statements'][0]['sha256']='0'*64
before=Path.cwd()
try: prep.prepare(bad)
except ValueError: pass
else: raise AssertionError('Accepted a changed source slice')
assert Planetoid.__init__ is original_init and fs.cp is copy_fixture and Path.cwd()==before
print('NATIVE_WRAPPER_PREPARATION_VERIFIED')
'''


def test_original_sliced_loading_with_real_pyg_and_invented_download_fixture(tmp_path):
    cache = configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu')
    if cache is None:
        pytest.skip('Requires a dedicated verified CPU/PyG cache outside application data')
    result, sources = compile_case(tmp_path, normalize=True)
    (tmp_path/'config.json').write_text(json.dumps({'loaders': [result], 'source_files': sources}), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', NATIVE_DRIVER, str(tmp_path)],
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(cache)]),
             'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'NATIVE_WRAPPER_PREPARATION_VERIFIED' in completed.stdout


@pytest.mark.parametrize('extra', [
    'import definitely_unavailable_unrelated_dependency as Planetoid',
    'import definitely_unavailable_unrelated_dependency as torch',
    'from definitely_unavailable_unrelated_dependency import load as inner',
    'from definitely_unavailable_unrelated_dependency import *',
    'import definitely_unavailable_unrelated_dependency as object',
    'import definitely_unavailable_unrelated_dependency as print',
])
def test_unreviewed_import_bindings_cannot_replace_reviewed_symbols(tmp_path, extra):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, loader=WRAPPER.replace('class GraphRecord(object):', extra + '\n\nclass GraphRecord(object):'))


def test_plain_dotted_reviewed_import_retains_its_python_root_binding(tmp_path):
    result, _ = compile_case(tmp_path, normalize=True, loader=WRAPPER.replace(
        'import torch_geometric.transforms as T', 'import torch_geometric.transforms').replace(
        'T.NormalizeFeatures()', 'torch_geometric.transforms.NormalizeFeatures()'))
    assert result['transform'] == 'NormalizeFeatures'


@pytest.mark.parametrize('prefix', [
    'alias = [args]\nmutate(alias)',
    'alias = args.__dict__\nmutate(alias)',
    'from unavailable_dependency import load',
    'from unavailable_dependency import *',
])
def test_entry_binding_and_nested_namespace_escapes_are_rejected(tmp_path, prefix):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', prefix))


@pytest.mark.parametrize('prefix', [
    'if unknown_condition:\n    load = other_loader',
    'if unknown_condition:\n    from unavailable_dependency import load',
])
def test_unknown_entry_branch_cannot_replace_loader_binding(tmp_path, prefix):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', prefix))


def test_original_import_restores_a_prior_unrelated_binding(tmp_path):
    result, _ = compile_case(tmp_path, entry=ENTRY.replace('from datasets import load',
        'from unavailable_dependency import load\nfrom datasets import load'))
    assert result['name'] == 'Cora'


def test_known_local_wildcard_module_without_loader_export_is_accepted(tmp_path):
    result, _ = compile_case(tmp_path, entry=ENTRY.replace('from datasets import load',
        'from datasets import load\nfrom local_helpers import *'),
        extra_files={'scripts/local_helpers.py': 'import torch\ndef evaluate():\n    return None\n'})
    assert result['name'] == 'Cora'


@pytest.mark.parametrize('module', [
    'def load():\n    pass\n',
    "__all__ = dynamic_exports()\n",
    "__all__ = ['load']\ndef __getattr__(name):\n    return dynamic(name)\n",
    'from unavailable_dependency import *\n',
    'mutate_globals()\n',
])
def test_local_wildcard_exports_cannot_replace_loader_or_be_dynamic(tmp_path, module):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', 'from local_helpers import *'),
            extra_files={'scripts/local_helpers.py': module})


@pytest.mark.parametrize('prefix', [
    "globals()['load'] = other",
    "globals().update(load=other)",
    "exec('load = other')",
    "eval('globals()').update(load=other)",
    "namespace = globals()\nnamespace['load'] = other",
    "vars()['load'] = other",
    "namespace_getter = globals\nnamespace_getter()['load'] = other",
    "execute = exec\nexecute('load = other')",
])
def test_explicit_dynamic_entry_bindings_before_load_are_rejected(tmp_path, prefix):
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', prefix))


@pytest.mark.parametrize('body', [
    "    global load\n    load = other",
    "    globals()['load'] = other",
    "    globals().update(load=other)",
    "    exec('load = other')",
])
@pytest.mark.parametrize('invocation', ['replace_loader()', 'alias = replace_loader\nalias()'])
def test_referenced_local_helper_cannot_replace_loader_binding(tmp_path, body, invocation):
    helper = 'def replace_loader():\n' + body + '\n'
    with pytest.raises(AuthorLoaderUnsupported):
        compile_case(tmp_path, entry=ENTRY.replace('print(args)', helper + invocation))


def test_plain_local_seed_setup_and_unreferenced_helper_are_supported(tmp_path):
    prefix = """def unused():
    global load
    load = other

def fix_seed(seed):
    torch.manual_seed(seed)

fix_seed(args.seed)
"""
    result, _ = compile_case(tmp_path, entry=ENTRY.replace('print(args)', prefix))
    assert result['name'] == 'Cora'
