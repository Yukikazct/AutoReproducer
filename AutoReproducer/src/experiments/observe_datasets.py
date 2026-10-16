"""Observe original PyG constructors, without changing data or author arguments."""
import hashlib
import importlib
import inspect
from pathlib import Path
import stat

if __package__:
    from .observe_dgl_datasets import capture_dgl
else:
    from observe_dgl_datasets import capture_dgl


def real_dataset_path(path):
    """Reject link indirection before resolving an observed input's identity."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        metadata = component.lstat()
        if (stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0)
                & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
            raise ValueError('Original dataset path contains a symlink or junction')
    return path.resolve(strict=True)


def file_record(path, role):
    path = real_dataset_path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Original framework dataset is missing a complete real file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return str(path), {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size, 'role': role}


class DatasetObserver:
    def __init__(self, declarations, workspace, update_count, *, strict=True):
        self.records, self.patches = [], []
        self.strict = strict
        self.workspace, self.update_count = Path(workspace).resolve(), update_count
        self.declarations = declarations
        if declarations:
            for module_name, name in dict.fromkeys((item.get('framework', 'torch_geometric.datasets'),
                                                   item['dataset_class']) for item in declarations):
                try:
                    framework = importlib.import_module(module_name)
                    if module_name == 'dgl.data' and importlib.import_module('dgl').__version__ != '1.1.2':
                        raise ValueError('DGL dataset observation requires the reviewed DGL 1.1.2 runtime')
                    cls = getattr(framework, name)
                    if not cls.__module__.startswith(module_name + '.'):
                        raise ValueError('Native dataset class differs from the fixed framework')
                except Exception as exc:
                    if self.strict:
                        raise
                    for declaration in declarations:
                        if declaration['dataset_class'] == name and declaration.get('framework', 'torch_geometric.datasets') == module_name:
                            self._unknown(declaration, exc, False)
                    continue
                original = cls.__init__
                def observed(instance, *args, __original=original, __name=name, __module=module_name,
                             __class=cls, **kwargs):
                    frame = inspect.currentframe().f_back
                    declarations = []
                    while frame is not None:
                        filename = Path(frame.f_code.co_filename).resolve()
                        if filename.is_relative_to(self.workspace):
                            declarations.extend(item for item in self.declarations
                                if item['dataset_class'] == __name and filename == self.workspace / item['path']
                                and item.get('framework', 'torch_geometric.datasets') == __module
                                and frame.f_code.co_name == item['function'])
                            break
                        frame = frame.f_back
                    before_training = self.update_count() == 0
                    __original(instance, *args, **kwargs)
                    for declaration in declarations:
                        try:
                            if __module == 'dgl.data' and type(instance) is not __class:
                                raise ValueError('DGL data observation does not accept an author subclass')
                            self.capture(instance, declaration, before_training)
                        except Exception as exc:
                            if self.strict:
                                raise
                            self._unknown(declaration, exc, before_training)
                self.patches.append((cls, original))
                cls.__init__ = observed

    def _unknown(self, declaration, error, before_training):
        self.records.append({'declaration': declaration, 'verified': False,
                             'loaded_before_training': before_training, 'verification_error': str(error)})

    def capture(self, dataset, declaration, before_training):
        if declaration.get('framework') == 'dgl.data':
            self.records.append(capture_dgl(dataset, declaration, before_training, file_record, real_dataset_path))
            return
        root = real_dataset_path(dataset.root)
        files = {}
        for role, paths in [('raw', dataset.raw_paths), ('processed', dataset.processed_paths)]:
            for path in paths:
                name, record = file_record(path, role)
                if not Path(name).is_relative_to(root):
                    raise ValueError('Original framework data file escaped its declared root')
                files[name] = record
        if not any(item['role'] == 'raw' for item in files.values()):
            raise ValueError('Original framework dataset has no complete raw files')
        metadata = {'sample_count': len(dataset)}
        data = getattr(dataset, '_data', None)
        if data is not None:
            for name in ('num_nodes', 'num_edges', 'num_features'):
                value = getattr(data, name, None)
                if value is not None:
                    metadata[name] = int(value)
        name = getattr(dataset, 'name', type(dataset).__name__)
        if str(name).casefold() != declaration['dataset_name'].casefold():
            raise ValueError('Actual original dataset identity differs from the reviewed plan')
        self.records.append({'declaration': declaration, 'dataset_name': str(name),
            'root': str(root), 'files': files, 'loaded_before_training': before_training,
            'verified': True, **metadata})

    def finish(self):
        for record in self.records:
            if record.get('verified') is not True:
                continue
            try:
                for name, expected in record['files'].items():
                    _, actual = file_record(name, expected['role'])
                    if actual != expected:
                        raise ValueError('Real framework dataset changed during author training')
            except Exception as exc:
                if self.strict:
                    raise
                record.update(verified=False, verification_error=str(exc))
        for declaration in self.declarations:
            if not any(record['declaration'] == declaration and record.get('loaded_before_training')
                       and record.get('verified') for record in self.records):
                error = ValueError('The reviewed native dataset was not constructed before actual training')
                if self.strict:
                    raise error
                if not any(record['declaration'] == declaration and record.get('verified') is False for record in self.records):
                    self._unknown(declaration, error, False)
        return self.records

    def close(self):
        for cls, original in reversed(self.patches):
            cls.__init__ = original
