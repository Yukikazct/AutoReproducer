"""Observe original PyG constructors, without changing data or author arguments."""
import hashlib
import importlib
import inspect
from pathlib import Path


def file_record(path, role):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Original framework dataset is missing a complete real file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return str(path), {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size, 'role': role}


class DatasetObserver:
    def __init__(self, declarations, workspace, update_count):
        self.records, self.patches = [], []
        self.workspace, self.update_count = Path(workspace).resolve(), update_count
        self.declarations = declarations
        if declarations:
            framework = importlib.import_module('torch_geometric.datasets')
            for name in dict.fromkeys(item['dataset_class'] for item in declarations):
                cls = getattr(framework, name)
                if not cls.__module__.startswith('torch_geometric.datasets.'):
                    raise ValueError('Native dataset class differs from the fixed framework')
                original = cls.__init__
                def observed(instance, *args, __original=original, __name=name, **kwargs):
                    frame = inspect.currentframe().f_back
                    declarations = []
                    while frame is not None:
                        filename = Path(frame.f_code.co_filename).resolve()
                        if filename.is_relative_to(self.workspace):
                            declarations.extend(item for item in self.declarations
                                if item['dataset_class'] == __name and filename == self.workspace / item['path']
                                and frame.f_code.co_name == item['function'])
                            break
                        frame = frame.f_back
                    before_training = self.update_count() == 0
                    __original(instance, *args, **kwargs)
                    for declaration in declarations:
                        self.capture(instance, declaration, before_training)
                self.patches.append((cls, original))
                cls.__init__ = observed

    def capture(self, dataset, declaration, before_training):
        root = Path(dataset.root).resolve(strict=True)
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
            for name, expected in record['files'].items():
                _, actual = file_record(name, expected['role'])
                if actual != expected:
                    raise ValueError('Real framework dataset changed during author training')
        for declaration in self.declarations:
            if not any(record['declaration'] == declaration and record['loaded_before_training'] for record in self.records):
                raise ValueError('The reviewed native dataset was not constructed before actual training')
        return self.records

    def close(self):
        for cls, original in reversed(self.patches):
            cls.__init__ = original
