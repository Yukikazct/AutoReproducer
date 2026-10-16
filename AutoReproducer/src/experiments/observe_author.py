"""Observe an unchanged original Python entrypoint; no model/metric substitution."""
import hashlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import runpy
import sys
import types

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook
from observe_datasets import DatasetObserver
from author_numeric_compatibility import diagnose_numpy_alias_error, diagnose_reciprocal_error, run_entrypoint
from author_path_compatibility import diagnose_temporary_root_error
from author_pyg_compatibility import LegacyPyGCompatibility


runtime = Path(__file__).resolve().parent
workspace = runtime.parent
config = json.loads((runtime / 'config.json').read_text('utf-8'))
identifier = sys.argv[1]
step = next(item for item in config['steps'] if item['id'] == identifier)
failure_path = runtime / ('numeric_failure_' + identifier + '.json')
failure_path.unlink(missing_ok=True)
path_failure = runtime / ('path_failure_' + identifier + '.json')
path_failure.unlink(missing_ok=True)
pyg_failure = runtime / ('pyg_failure_' + identifier + '.json')
pyg_failure.unlink(missing_ok=True)
(runtime / ('author_' + identifier + '.json')).unlink(missing_ok=True)
for relative, expected in config['source_files'].items():
    source = (workspace / relative).resolve(strict=True)
    if not source.is_relative_to(workspace) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned author source changed before execution: ' + relative)
for relative, record in config.get('data_files', {}).items():
    source = (workspace / relative).resolve(strict=True)
    if (not source.is_relative_to(workspace) or hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']
            or source.stat().st_size != record['bytes']):
        raise ValueError('Frozen original author data changed before execution: ' + relative)
argv = step['argv']
entrypoint = step.get('entrypoint') or {'kind': 'script', 'source_path': argv[2 if argv[1] == '-u' else 1]}
index = 2 if len(argv) > 1 and argv[1] == '-u' else 1
directory = (workspace / step['cwd']).resolve(strict=True)
script = (workspace / entrypoint['source_path']).resolve(strict=True)
if not directory.is_relative_to(workspace) or not script.is_relative_to(workspace):
    raise ValueError('Author command escaped its workspace')
for item in step.get('output_directories', []):
    if config['source_files'].get(item['source_path']) != item['source_sha256']:
        raise ValueError('Author output parent lacks pinned source evidence')
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
        raise ValueError('Author output parent escaped its workspace')
    target = (workspace / relative).resolve()
    if not target.is_relative_to(workspace):
        raise ValueError('Author output parent escaped its workspace')
    for parent in [workspace / relative, *(workspace / relative).parents]:
        if parent == workspace:
            break
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Author output parent contains a link or junction')
    target.mkdir(parents=True, exist_ok=True)
updates = 0


def after_step(optimizer, args, kwargs):
    global updates
    updates += 1


handle = register_optimizer_step_post_hook(after_step)
datasets = DatasetObserver(config.get('runtime_datasets', []) if step['kind'] == 'train' else [], workspace, lambda: updates)
legacy_pyg = LegacyPyGCompatibility(workspace, config['source_files'], config.get('pyg_compatibility', []),
    lambda: updates, enabled=config.get('observe_legacy_pyg', False) and entrypoint['kind'] != 'notebook')
os.chdir(directory)
if entrypoint['kind'] == 'module':
    sys.argv = [entrypoint['target'], *argv[index + 2:]]
    sys.path[0] = str(directory)
elif entrypoint['kind'] == 'notebook':
    sys.path[0] = str(directory)
else:
    sys.argv = argv[index:]
    sys.path[0] = str(script.parent)
for namespace in config.get('repository_namespaces', []):
    name = namespace['name']
    if (not name.isidentifier() or namespace['path'] != '.' or name in sys.modules
            or (workspace / name).exists() or (workspace / (name + '.py')).exists()
            or (workspace / '__init__.py').exists()):
        raise ValueError('Repository namespace conflicts with an existing Python module')
    package = types.ModuleType(name)
    package.__path__ = [str(workspace)]
    package.__package__ = name
    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package
try:
    try:
        if entrypoint['kind'] == 'notebook':
            from author_notebook import run_notebook
            updates = run_notebook(script, runtime, identifier, directory)
        elif entrypoint['kind'] == 'module':
            if config.get('numeric_compatibility') or config.get('path_compatibility'):
                raise ValueError('Module compatibility repair needs a separately reviewed entrypoint')
            runpy.run_module(entrypoint['target'], run_name='__main__', alter_sys=True)
        else:
            run_entrypoint(script, config.get('numeric_compatibility', []), workspace, config['source_files'],
                           config.get('path_compatibility', []))
    except SystemExit as exc:
        # absl.app.run and ordinary sys.exit(0) finish valid author programs
        # through SystemExit. A clean exit still needs real data and updates.
        if exc.code is not None and (type(exc.code) is not int or exc.code != 0):
            raise
    except Exception as exc:
        candidate = (diagnose_reciprocal_error(exc, script, workspace, config['source_files'])
                     or diagnose_numpy_alias_error(exc, script, workspace, config['source_files']))
        if candidate:
            failure_path.write_text(json.dumps({'spec_sha256': config['spec_sha256'],
                'step': identifier, 'optimizer_steps': updates, 'repair': candidate}), encoding='utf-8')
        candidate = diagnose_temporary_root_error(exc, script, workspace, config['source_files'])
        if candidate:
            path_failure.write_text(json.dumps({'spec_sha256': config['spec_sha256'],
                'step': identifier, 'optimizer_steps': updates, 'repair': candidate}), encoding='utf-8')
        candidate = legacy_pyg.diagnose(exc)
        if candidate:
            pyg_failure.write_text(json.dumps({'spec_sha256': config['spec_sha256'],
                'step': identifier, 'optimizer_steps': updates, 'repair': candidate}), encoding='utf-8')
        raise
    native_data = datasets.finish()
finally:
    handle.remove()
    datasets.close()
    legacy_pyg.close()
# Failed exits, exceptions and interrupted processes never create a receipt.
receipt = {'spec_sha256': config['spec_sha256'], 'step': step,
           'natural_return': True, 'optimizer_steps': updates,
           'torch_version': torch.__version__, 'native_datasets': native_data,
           'numeric_compatibility': config.get('numeric_compatibility', []),
           'path_compatibility': config.get('path_compatibility', []),
           'pyg_compatibility': config.get('pyg_compatibility', []),
           'pyg_conversions': legacy_pyg.applied}
(runtime / ('author_' + identifier + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'author_command_finished': identifier, 'optimizer_steps': updates}), flush=True)
