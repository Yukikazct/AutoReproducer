"""Source-bound real loading contracts, without running the author's training."""
import hashlib

import pytest

from src.author_dataset_loader import AuthorLoaderUnsupported, compile_author_loader


LOADER = '''import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
def load(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset
'''
ENTRY = '''import argparse
from datasets import load
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()
if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or args.dataset == 'PubMed':
    dataset = load(args.dataset, args.normalize_features)
else:
    raise RuntimeError('Other dataset not supported')
raise AssertionError('Never execute the training entrypoint during compilation')
'''


def compiled(tmp_path, *, loader=LOADER, entry=ENTRY, name='Cora', normalize=True):
    files = {}
    for path, text in {'scripts/datasets.py': loader, 'scripts/train.py': entry}.items():
        full = tmp_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(text.encode())
        files[path] = {'text': text, 'sha256': hashlib.sha256(full.read_bytes()).hexdigest()}
    declaration = {'kind': 'author_loader', 'path': 'scripts/datasets.py', 'function': 'load'}
    effective = {'dataset': {'value': name}, 'normalize_features': {'value': normalize}}
    train = {'argv': ['python', 'train.py', '--dataset', name], 'cwd': 'scripts'}
    return compile_author_loader(declaration, files, tmp_path, train, effective)


@pytest.mark.parametrize('name', ['Cora', 'CiteSeer', 'PubMed'])
@pytest.mark.parametrize('normalize', [True, False])
def test_compile_actual_cli_selected_call_root_and_original_transform(tmp_path, name, normalize):
    result = compiled(tmp_path, name=name, normalize=normalize)
    assert result['name'] == name and result['root'] == 'data/' + name
    assert result['arguments'] == {'name': name, 'normalize_features': normalize, 'transform': None}
    assert result['transform'] == ('NormalizeFeatures' if normalize else None)
    assert result['split'] == 'public'
    assert result['call']['quote'] == 'load(args.dataset, args.normalize_features)'
    assert result['definition'].startswith('def load(')
    assert 'url' not in result and 'sha256' not in result  # Actual downloaded bytes freeze later.


def test_literal_alias_and_original_defaults_are_resolved(tmp_path):
    result = compiled(tmp_path, entry=ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
        "selected_name = 'Cora'\n    dataset = load(selected_name)"))
    assert result['arguments']['normalize_features'] is False


@pytest.mark.parametrize('loader', [
    LOADER.replace("'..', 'data'", "'..', '..', '..', 'data'"),
    LOADER.replace('dataset = Planetoid(path, name)', "dataset = Planetoid(path, name, split='random')"),
    LOADER.replace('dataset = Planetoid(path, name)', 'dataset = download_custom_data(path, name)'),
    LOADER.replace('    return dataset', '    eval(name)\n    return dataset'),
    LOADER.replace('def load(', '@decorator\ndef load('),
    LOADER.replace('import os.path as osp', 'import os.path as osp\nprint("side effect")'),
    LOADER.replace('import os.path as osp', 'import os.path as osp\nimport requests'),
    LOADER + '\ndef unused(value=execute_side_effect()):\n    pass\n',
    LOADER + '\n@execute_side_effect()\ndef unused():\n    pass\n',
])
def test_escape_dynamic_loading_unreviewed_split_and_side_effects_are_rejected(tmp_path, loader):
    with pytest.raises(AuthorLoaderUnsupported):
        compiled(tmp_path, loader=loader)


@pytest.mark.parametrize('entry', [
    ENTRY.replace('if args.dataset ==', 'if unknown_condition and args.dataset =='),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
                  'dataset = load(args.dataset, args.normalize_features)\n    other = load(args.dataset)'),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)', 'dataset = load(*args.dataset)'),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
                  "args = unknown_namespace()\n    dataset = load(args.dataset, args.normalize_features)"),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
                  "load = unknown_function\n    dataset = load(args.dataset, args.normalize_features)"),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
                  "args.dataset = 'PubMed'\n    dataset = load(args.dataset, args.normalize_features)"),
    ENTRY.replace('dataset = load(args.dataset, args.normalize_features)',
                  "mutate(args)\n    dataset = load(args.dataset, args.normalize_features)"),
])
def test_selected_invocation_must_have_one_fixed_original_binding(tmp_path, entry):
    with pytest.raises(AuthorLoaderUnsupported):
        compiled(tmp_path, entry=entry)


def test_a_different_dataset_cannot_acquire_a_planetoid_contract(tmp_path):
    with pytest.raises(AuthorLoaderUnsupported):
        compiled(tmp_path, name='SyntheticReplacement')
