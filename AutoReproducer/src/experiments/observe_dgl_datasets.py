"""Read complete native DGL citation inputs without calling __getitem__ or transforms."""
import hashlib
from pathlib import Path
import re


RAW_PARTS = ('x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index')
TENSOR_FIELDS = ('feat', 'label', 'train_mask', 'val_mask', 'test_mask')


def capture_dgl(dataset, declaration, before_training, file_record, real_path):
    root, raw = real_path(dataset.raw_dir), real_path(dataset.raw_path)
    native = str(dataset.name)
    if native != declaration['framework_dataset_name']:
        raise ValueError('Actual DGL dataset identity differs from the reviewed native class')
    url = f'https://data.dgl.ai/dataset/{native}.zip'
    if dataset.url != url:
        raise ValueError('DGL dataset source URL differs from the supported native origin')
    files = {}
    for path, role in [*((raw / f'ind.{native}.{part}', 'raw') for part in RAW_PARTS),
                       (dataset.graph_path, 'processed'), (dataset.info_path, 'processed')]:
        name, value = file_record(path, role)
        if not Path(name).is_relative_to(root):
            raise ValueError('Original DGL input escaped its native raw/save directory')
        files[name] = value
    graph = vars(dataset).get('_g')
    if type(graph).__module__ != 'dgl.heterograph':
        raise ValueError('DGL observation needs the original native homogeneous graph')
    nodes = int(graph.num_nodes())
    evidence, masks = {}, {}
    for name in TENSOR_FIELDS:
        tensor = graph.ndata[name]
        if not tensor.shape or tensor.shape[0] != nodes:
            raise ValueError('DGL native tensor does not cover every original node')
        evidence[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        if name.endswith('_mask'):
            if str(tensor.dtype) != 'torch.bool' or tensor.ndim != 1:
                raise ValueError('DGL citation split needs the original complete boolean mask')
            masks[{'train_mask': 'train', 'val_mask': 'validation', 'test_mask': 'test'}[name]] = int(tensor.sum().item())
    return {'declaration': declaration, 'dataset_name': declaration['dataset_name'],
        'framework_dataset_name': native, 'framework_version': declaration['framework_version'],
        'source_url': dataset.url, 'root': str(root), 'raw_path': str(raw),
        'graph_path': str(real_path(dataset.graph_path)), 'info_path': str(real_path(dataset.info_path)),
        'files': files, 'loaded_before_training': before_training, 'verified': True,
        'sample_count': len(dataset), 'num_nodes': nodes, 'num_edges': int(graph.num_edges()),
        'num_features': int(graph.ndata['feat'].shape[1]), 'num_classes': int(dataset.num_classes),
        'mask_counts': masks, 'tensor_evidence': evidence}


def validate_dgl_record(declaration, record, real_path):
    native = declaration['framework_dataset_name']
    if (record.get('framework_dataset_name') != native or record.get('framework_version') != '1.1.2'
            or record.get('source_url') != f'https://data.dgl.ai/dataset/{native}.zip'
            or record.get('sample_count') != 1):
        raise ValueError('DGL dataset lacks its supported native class/version/source identity')
    for field in ('num_nodes', 'num_edges', 'num_features', 'num_classes'):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError('DGL dataset lacks complete native graph metadata')
    nodes = record['num_nodes']
    masks = record.get('mask_counts') or {}
    tensors = record.get('tensor_evidence') or {}
    if (set(masks) != {'train', 'validation', 'test'}
            or any(type(value) is not int or not 0 <= value <= nodes for value in masks.values())
            or sum(masks.values()) > nodes or set(tensors) != set(TENSOR_FIELDS)):
        raise ValueError('DGL dataset lacks complete native split metadata')
    for name, value in tensors.items():
        expected = [nodes, record['num_features']] if name == 'feat' else [nodes]
        if (value.get('shape') != expected or not re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', '')))
                or value.get('dtype') != ('torch.float32' if name == 'feat' else
                                          'torch.int64' if name == 'label' else 'torch.bool')):
            raise ValueError('DGL dataset lacks complete native tensor evidence')
    raw = real_path(record['raw_path'])
    expected = {str(real_path(raw / f'ind.{native}.{part}')): 'raw' for part in RAW_PARTS}
    expected.update({str(real_path(record['graph_path'])): 'processed',
                     str(real_path(record['info_path'])): 'processed'})
    if (len(expected) != 10 or set(record['files']) != set(expected)
            or any(record['files'][name].get('role') != role for name, role in expected.items())):
        raise ValueError('DGL dataset evidence omits original raw or processed inputs')
