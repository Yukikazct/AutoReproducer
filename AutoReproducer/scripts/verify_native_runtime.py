"""Fail native CI on missing capabilities, wrong runtime or skipped selected tests."""
import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def verify_junit(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if not cases or any(case.find(tag) is not None for case in cases for tag in ('skipped', 'failure', 'error')):
        raise ValueError('Native selection must execute nonempty tests with no skips, failures or errors')
    return {'executed_tests': len(cases), 'skipped': 0}


def verify_runtime(family, deps):
    deps = Path(deps).resolve(strict=True)
    expected_python = (3, 11) if family == 'dgl' else (3, 12)
    if sys.version_info[:2] != expected_python:
        raise ValueError(f'{family} requires CPython {expected_python}')
    sys.path.insert(0, str(deps))
    modules = ('numpy', 'torch', 'dgl') if family == 'dgl' else ('numpy', 'torch', 'torch_geometric', 'torch_scatter', 'torch_sparse')
    loaded = {name: importlib.import_module(name) for name in modules}
    for name, module in loaded.items():
        if not Path(module.__file__).resolve().is_relative_to(deps):
            raise ValueError('Native module was borrowed from outside the dedicated environment: ' + name)
    expected = {'torch': '2.5.1+cpu', 'numpy': '1.26.4', 'dgl': '1.1.2'} if family == 'dgl' else {
        'torch': '2.5.1+cpu', 'numpy': '1.26.4', 'torch_geometric': '2.6.1',
        'torch_scatter': '2.1.2+pt25cpu', 'torch_sparse': '0.6.18+pt25cpu'}
    for name, value in expected.items():
        if loaded[name].__version__ != value:
            raise ValueError('Unexpected native version: ' + name)
    torch, np = loaded['torch'], loaded['numpy']
    assert np.array_equal((torch.ones(2, 2) @ torch.ones(2, 2)).numpy(), np.full((2, 2), 2))
    if family == 'dgl':
        dgl = loaded['dgl']
        graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
        graph.ndata['x'] = torch.ones(3, 1)
        graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'out'))
        assert graph.ndata['out'].flatten().tolist() == [0, 1, 1]
    else:
        from torch_geometric.nn import GCNConv
        from torch_sparse import SparseTensor
        from torch_scatter import scatter_add
        assert scatter_add(torch.ones(2), torch.tensor([0, 0])).tolist() == [2]
        sparse = SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]), sparse_sizes=(2, 2))
        assert sparse.matmul(torch.ones(2, 1)).tolist() == [[1], [1]]
        assert GCNConv(2, 2)(torch.ones(2, 2), torch.tensor([[0, 1], [1, 0]])).shape == (2, 2)
    versions = {d.metadata['Name']: d.version for d in metadata.distributions(path=[str(deps)])}
    (deps / '.ready').touch()
    return {'family': family, 'python': sys.version, 'deps': str(deps), 'versions': versions,
            'native_operations_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('pyg', 'dgl'))
    parser.add_argument('--deps')
    parser.add_argument('--junit')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.junit:
        result = verify_junit(args.junit)
    elif args.family and args.deps:
        result = verify_runtime(args.family, args.deps)
    else:
        parser.error('Supply --junit or both --family and --deps')
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
