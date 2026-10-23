"""Run the frozen original loader and record its actual framework downloads."""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.io import fs
from torch_geometric.transforms import NormalizeFeatures


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(relative, *, exists=False):
    target = (WORKSPACE / relative).resolve(strict=exists)
    if not target.is_relative_to(WORKSPACE) or '.git' in target.relative_to(WORKSPACE).parts:
        raise ValueError('Original dataset loader path escaped the frozen workspace')
    current = WORKSPACE
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise ValueError('Original dataset loader path uses a filesystem link')
    return target


def original_slice(source, declaration, module):
    """Compile only hash-bound, unchanged original definitions and imports."""
    if declaration.get('version') != 1:
        raise ValueError('Original loading slice has an unsupported version')
    text = source.read_text('utf-8')
    tree = ast.parse(text, filename=str(source))
    available = {(type(node).__name__, node.lineno): node for node in tree.body}
    retained, seen = [], set()
    for statement in declaration['statements']:
        identity = (statement['kind'], statement['line'])
        node = available.get(identity)
        if (identity in seen or node is None or not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                or hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest() != statement['sha256']):
            raise ValueError('Original loading slice changed or contains unsupported statements')
        seen.add(identity)
        retained.append(node)
    if not retained or [node.lineno for node in retained] != sorted(node.lineno for node in retained):
        raise ValueError('Original loading slice must retain its original statement order')
    exec(compile(ast.Module(body=retained, type_ignores=[]), str(source), 'exec'), module.__dict__)


def verify_record(result, declaration, module, data):
    """Read direct plain storage only, without author attribute/index hooks."""
    cls = module.__dict__.get(declaration['class'])
    if not isinstance(cls, type) or type(result) is not cls or cls.__bases__ != (object,):
        raise ValueError('Original loader returned another record class')
    fields = object.__getattribute__(result, '__dict__')
    if type(fields) is not dict or set(fields) != set(declaration['fields']):
        raise ValueError('Original record stored fields disagree with the frozen loading contract')

    def compare(actual, expected):
        if set(expected) == {'literal'}:
            value = expected['literal']
            return type(value) in (str, int, float, bool, type(None)) and type(actual) is type(value) and actual == value
        if set(expected) == {'dictionary'}:
            values = expected['dictionary']
            return (type(actual) is dict and set(actual) == set(values)
                    and all(compare(actual[key], value) for key, value in values.items()))
        if set(expected) == {'data_field'}:
            name = expected['data_field']
            if name not in {'x', 'y', 'edge_index', 'num_nodes', 'train_mask', 'val_mask', 'test_mask'}:
                return False
            value = getattr(data, name)
        elif set(expected) == {'mask_indices'}:
            name = expected['mask_indices']
            if name not in {'train_mask', 'val_mask', 'test_mask'}:
                return False
            value = torch.where(getattr(data, name))[0]
        else:
            return False
        if type(value) is torch.Tensor:
            return (type(actual) is torch.Tensor and actual.dtype == value.dtype and actual.shape == value.shape
                    and actual.device == value.device and torch.equal(actual, value))
        return type(actual) is type(value) and type(value) in (int, bool) and actual == value

    if not all(compare(fields[name], value) for name, value in declaration['fields'].items()):
        raise ValueError('Original record changed framework features, labels, edges, masks or metadata')


def prepare(config):
    if torch_geometric.__version__ != '2.6.1':
        raise ValueError('Author loading needs the frozen reviewed PyG version')
    def check_sources():
        for name, expected in config['source_files'].items():
            if sha(checked(name, exists=True)) != expected:
                raise ValueError('Frozen author source changed during dataset preparation')
    check_sources()
    records = []
    for index, loader in enumerate(config['loaders']):
        if loader['framework'] != 'torch_geometric.datasets.Planetoid' or loader['split'] != 'public':
            raise ValueError('Dataset preparation received an unreviewed framework or split')
        source = checked(loader['path'], exists=True)
        root = checked(loader['root'])
        raw_dir = root / loader['name'] / 'raw'
        names = [f"ind.{loader['name'].lower()}.{name}" for name in
                 ('x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index')]
        urls = {Planetoid.url + '/' + name: name for name in names}
        copied = []
        original_cp = fs.cp
        def observe(source_url, destination, *args, **kwargs):
            parsed = urlsplit(source_url)
            if (source_url not in urls or parsed.scheme != 'https' or parsed.username or parsed.password
                    or Path(destination).resolve() != raw_dir.resolve()):
                raise ValueError('Original framework attempted an unexpected dataset download')
            if config.get('offline'):
                raise ValueError('Offline preparation lacks the original real dataset bytes')
            value = original_cp(source_url, destination, *args, **kwargs)
            target = checked((raw_dir / urls[source_url]).relative_to(WORKSPACE).as_posix(), exists=True)
            size = target.stat().st_size
            if not 0 < size <= 512 * 1024**2:
                raise ValueError('Original real dataset download has invalid size')
            copied.append({'url': source_url, 'path': target.relative_to(WORKSPACE).as_posix(),
                           'sha256': sha(target), 'bytes': size})
            return value
        spec = importlib.util.spec_from_file_location('_autorepro_original_data_' + str(index), source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        original_init = Planetoid.__init__
        constructed = []
        def observe_constructor(instance, *args, **kwargs):
            if type(instance) is not Planetoid or constructed:
                raise ValueError('Original loader must construct exactly one native Planetoid dataset')
            constructed.append(instance)
            return original_init(instance, *args, **kwargs)
        previous_cwd = Path.cwd()
        try:
            fs.cp = observe
            Planetoid.__init__ = observe_constructor
            if loader.get('source_slice'):
                original_slice(source, loader['source_slice'], module)
                os.chdir(checked(loader['source_slice']['cwd'], exists=True))
            else:
                spec.loader.exec_module(module)
                if loader.get('cwd') is not None:
                    os.chdir(checked(loader['cwd'], exists=True))
            result = module.__dict__[loader['function']](**loader['arguments'])
            if len(constructed) == 1:
                native = constructed[0]
                observed_root = Path(native.root).resolve()
                observed_raw_dir = Path(native.raw_dir).resolve()
                observed_raw_paths = [Path(path).resolve() for path in native.raw_paths]
                observed_processed_paths = [Path(path).resolve() for path in native.processed_paths]
        finally:
            fs.cp = original_cp
            Planetoid.__init__ = original_init
            os.chdir(previous_cwd)
            sys.modules.pop(spec.name, None)
        if len(constructed) != 1:
            raise ValueError('Original loader did not construct one native framework dataset')
        dataset = constructed[0]
        record = loader.get('source_slice', {}).get('returned_record')
        if record is None and result is not dataset:
            raise ValueError('Original loader did not return its original framework dataset')
        if type(dataset) is not Planetoid or dataset.name != loader['name'] or dataset.split != 'public':
            raise ValueError('Original loader did not return the frozen framework dataset')
        if observed_root != root or observed_raw_dir != raw_dir:
            raise ValueError('Original loader used another dataset storage root')
        expected_transform = loader['transform']
        if not ((expected_transform is None and dataset.transform is None)
                or (expected_transform == 'NormalizeFeatures' and type(dataset.transform) is NormalizeFeatures)):
            raise ValueError('Original loader changed the frozen preprocessing transform')
        if dataset.pre_transform is not None or dataset.pre_filter is not None:
            raise ValueError('Original loader used an unreviewed preprocessing/filter')
        if len(dataset) != 1 or set(path.name for path in observed_raw_paths) != set(names):
            raise ValueError('Original loader does not expose the complete reviewed graph dataset')
        raw_records = []
        for path in observed_raw_paths:
            target = checked(Path(path).relative_to(WORKSPACE).as_posix(), exists=True)
            raw_records.append({'url': Planetoid.url + '/' + target.name,
                'path': target.relative_to(WORKSPACE).as_posix(), 'sha256': sha(target), 'bytes': target.stat().st_size})
        if {item['path'] for item in copied} and {item['path'] for item in copied} != {item['path'] for item in raw_records}:
            raise ValueError('Framework download observation omitted original raw dataset files')
        if not copied:
            # A fresh run cannot assign guessed provenance to pre-existing bytes.
            raise ValueError('Original real data has no observed download provenance in this fresh workspace')
        data = dataset[0]
        if record is not None:
            verify_record(result, record, module, data)
        masks = [data.train_mask, data.val_mask, data.test_mask]
        if (any(mask.dtype != torch.bool or mask.ndim != 1 or len(mask) != data.num_nodes for mask in masks)
                or any(torch.any(masks[a] & masks[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
                or int(data.test_mask.sum()) != 1000 or int(data.val_mask.sum()) != 500):
            raise ValueError('Original dataset masks disagree with the reviewed complete public split')
        files = {}
        total = 0
        for target in root.rglob('*'):
            if not target.is_file():
                continue
            name = target.relative_to(WORKSPACE).as_posix()
            target = checked(name, exists=True)
            size = target.stat().st_size
            total += size
            if size <= 0 or total > 1024**3:
                raise ValueError('Original dataset file tree exceeds the reviewed preparation size')
            files[name] = {'sha256': sha(target), 'bytes': size}
        if not files or any(path.relative_to(WORKSPACE).as_posix() not in files for path in observed_processed_paths):
            raise ValueError('Original loader did not create its complete processed data')
        records.append({'kind': 'author_loader', 'verified': True, 'loader': loader,
            'framework_version': torch_geometric.__version__,
            'framework_source_sha256': sha(Path(inspect.getfile(Planetoid))),
            'downloads': raw_records, 'files': files, 'nodes': data.num_nodes,
            'classes': dataset.num_classes, 'split_counts': {
                'train': int(data.train_mask.sum()), 'validation': int(data.val_mask.sum()), 'test': int(data.test_mask.sum())}})
    check_sources()
    return {'version': 1, 'verified': True, 'sources': records}


if __name__ == '__main__':
    configuration = json.loads((ROOT / 'dataset_loaders.json').read_text('utf-8'))
    result = prepare(configuration)
    (ROOT / 'prepared_datasets.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('AUTHOR_REAL_DATA_PREPARED ' + json.dumps({'sources': len(result['sources'])}), flush=True)
