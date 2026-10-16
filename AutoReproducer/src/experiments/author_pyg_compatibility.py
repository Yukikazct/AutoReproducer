"""Recorded in-memory adaptation of old, single-graph PyG Data dictionaries.

This module is copied beside the author observer and has no application imports.
It never rewrites a dataset or changes tensors, splits, transforms or commands.
"""
import builtins
import hashlib
import inspect
import math
from pathlib import Path
import sys


KIND = 'pyg_legacy_data_state'
SUPPORTED_PYG = '2.6.1'


def _file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts or '.git' in relative.parts:
        raise ValueError('PyG compatibility path escaped the workspace')
    current = Path(root)
    if current.is_symlink() or getattr(current.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('PyG compatibility root contains a link or junction')
    for part in relative.parts:
        current /= part
        if current.is_symlink() or getattr(current.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('PyG compatibility path contains a link or junction')
    result = current.resolve(strict=True)
    if not result.is_relative_to(Path(root).resolve()) or not result.is_file():
        raise ValueError('PyG compatibility file escaped the workspace')
    return result


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root, value):
    raw = Path(value).absolute()
    for item in (raw, *raw.parents):
        if item.is_symlink() or getattr(item.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('PyG input or source contains a link or junction')
    return raw.resolve(strict=True).relative_to(root).as_posix()


def validate_repair(workspace, sources, repair):
    if (not isinstance(repair, dict) or repair.get('kind') != KIND
            or repair.get('pyg_version') != SUPPORTED_PYG
            or type(repair.get('caller_line')) is not int or repair['caller_line'] < 1
            or not isinstance(repair.get('state_schema'), dict) or not repair['state_schema']):
        raise ValueError('Unsupported legacy PyG compatibility evidence')
    for prefix in ('caller', 'dataset'):
        relative = repair.get(prefix + '_path')
        expected = repair.get(prefix + '_sha256')
        if not isinstance(relative, str) or expected != sources.get(relative):
            raise ValueError('Legacy PyG source is not pinned')
        if _digest(_file(workspace, relative)) != expected:
            raise ValueError('Legacy PyG source changed')
    relative = repair.get('data_path')
    if not isinstance(relative, str) or _digest(_file(workspace, relative)) != repair.get('data_sha256'):
        raise ValueError('Legacy PyG input changed')


def _schema(state, torch, Data):
    if '_store' in state or not {'x', 'edge_index', 'y'} <= state.keys():
        raise ValueError('Not a supported legacy homogeneous graph')
    result = {}
    for key, value in state.items():
        if (not isinstance(key, str) or not key.isidentifier() or key.startswith('_')
                or callable(getattr(Data, key, None))):
            raise ValueError('Unsupported legacy PyG field')
        if type(value) is torch.Tensor and value.layout == torch.strided:
            result[key] = {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype),
                           'device': str(value.device)}
        elif value is None:
            result[key] = {'type': 'none'}
        elif type(value) in (bool, int) or type(value) is float and math.isfinite(value):
            result[key] = {'type': type(value).__name__, 'value': value}
        else:
            raise ValueError('Unsupported legacy PyG field value')
    if any(result[key]['type'] != 'tensor' for key in ('x', 'edge_index', 'y')):
        raise ValueError('Legacy PyG graph lacks original tensors')
    return result


def upgrade_data(data, Data, torch):
    """Preserve identity and original values while supplying current storage."""
    if type(data) is not Data:
        raise ValueError('Only the native homogeneous Data class is supported')
    state = dict(object.__getattribute__(data, '__dict__'))
    _schema(state, torch, Data)
    fresh = Data.from_dict({key: value for key, value in state.items() if value is not None})
    replacement = dict(fresh.__dict__)
    replacement.update({key: None for key, value in state.items() if value is None})
    for key, value in state.items():
        if value is not None and getattr(fresh, key) is not value:
            raise ValueError('PyG container construction changed an original value')
    original = object.__getattribute__(data, '__dict__')
    original.clear()
    Data.__setstate__(data, replacement)
    # __setstate__ assigns attributes but does not repair this weak reference.
    replacement['_store']._parent = data
    return data


class LegacyPyGCompatibility:
    def __init__(self, workspace, sources, repairs, update_count, *, enabled=False):
        self.workspace = Path(workspace).resolve()
        self.sources, self.repairs, self.update_count = sources, repairs, update_count
        self.candidate, self.applied = None, []
        self.cls = self.original = self.wrapper = None
        self.original_import = self.import_wrapper = None
        if not enabled:
            return
        self.original_import = builtins.__import__

        def importing(*args, **kwargs):
            result = self.original_import(*args, **kwargs)
            self._install_if_loaded()
            return result

        self.import_wrapper = importing
        builtins.__import__ = importing
        self._install_if_loaded()

    def _install_if_loaded(self):
        if self.cls is not None:
            return
        framework = sys.modules.get('torch_geometric')
        module = sys.modules.get('torch_geometric.data.in_memory_dataset')
        data_module = sys.modules.get('torch_geometric.data.data')
        torch = sys.modules.get('torch')
        InMemoryDataset = getattr(module, 'InMemoryDataset', None)
        Data = getattr(data_module, 'Data', None)
        if (getattr(framework, '__version__', None) != SUPPORTED_PYG or torch is None
                or Data is None or InMemoryDataset is None
                or getattr(InMemoryDataset.get, '__module__', None) != 'torch_geometric.data.in_memory_dataset'):
            return
        self.cls, self.original = InMemoryDataset, InMemoryDataset.get

        def observed(dataset, index):
            self.candidate = None
            data = vars(dataset).get('_data')
            if type(data) is Data and '_store' not in vars(data) and self.update_count() == 0:
                try:
                    repair = self._evidence(dataset, data, torch, Data, inspect.currentframe().f_back)
                except (OSError, ValueError, TypeError, AttributeError):
                    repair = None
                self.candidate = repair
                if repair is not None and repair in self.repairs:
                    validate_repair(self.workspace, self.sources, repair)
                    upgrade_data(data, Data, torch)
                    self.applied.append(repair)
            return self.original(dataset, index)

        self.wrapper = observed
        InMemoryDataset.get = observed
        if builtins.__import__ is self.import_wrapper:
            builtins.__import__ = self.original_import

    def _evidence(self, dataset, data, torch, Data, frame):
        if vars(dataset).get('slices') is not None:
            raise ValueError('Only an original single-graph dataset is supported')
        caller = None
        while frame is not None:
            path = Path(frame.f_code.co_filename).resolve()
            if path.is_relative_to(self.workspace):
                relative = path.relative_to(self.workspace).as_posix()
                if relative in self.sources:
                    caller = (_relative_file(self.workspace, frame.f_code.co_filename), frame.f_lineno)
                break
            frame = frame.f_back
        if caller is None:
            raise ValueError('Legacy PyG access lacks pinned author context')
        dataset_path = _relative_file(self.workspace, inspect.getsourcefile(type(dataset)))
        if dataset_path not in self.sources:
            raise ValueError('Dataset class is not from the pinned author repository')
        inputs = dataset.processed_paths
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ValueError('Legacy PyG access requires one actual processed file')
        relative = _relative_file(self.workspace, inputs[0])
        repair = {'kind': KIND, 'pyg_version': SUPPORTED_PYG,
            'caller_path': caller[0], 'caller_line': caller[1], 'caller_sha256': self.sources[caller[0]],
            'dataset_path': dataset_path, 'dataset_sha256': self.sources[dataset_path],
            'data_path': relative, 'data_sha256': _digest(_file(self.workspace, relative)),
            'state_schema': _schema(dict(vars(data)), torch, Data)}
        validate_repair(self.workspace, self.sources, repair)
        return repair

    def diagnose(self, error):
        if (self.candidate is None or self.update_count() != 0
                or "created by an older version of PyG" not in str(error)):
            return None
        trace = error.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            if (frame.f_globals.get('__name__') == 'torch_geometric.data.in_memory_dataset'
                    and frame.f_code.co_name == 'get'):
                return self.candidate
            trace = trace.tb_next
        return None

    def close(self):
        if self.cls is not None and self.cls.get is self.wrapper:
            self.cls.get = self.original
        if builtins.__import__ is self.import_wrapper:
            builtins.__import__ = self.original_import
