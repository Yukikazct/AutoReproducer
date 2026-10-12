"""Observe an unchanged original Python entrypoint; no model/metric substitution."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook


runtime = Path(__file__).resolve().parent
workspace = runtime.parent
config = json.loads((runtime / 'config.json').read_text('utf-8'))
identifier = sys.argv[1]
step = next(item for item in config['steps'] if item['id'] == identifier)
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
index = 2 if argv[1] == '-u' else 1
directory = (workspace / step['cwd']).resolve(strict=True)
script = (directory / argv[index]).resolve(strict=True)
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
os.chdir(directory)
sys.argv = argv[index:]
sys.path[0] = str(script.parent)
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    handle.remove()
# Exceptions, SystemExit and interrupted processes never create a receipt.
receipt = {'spec_sha256': config['spec_sha256'], 'step': step,
           'natural_return': True, 'optimizer_steps': updates,
           'torch_version': torch.__version__}
(runtime / ('author_' + identifier + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'author_command_finished': identifier, 'optimizer_steps': updates}), flush=True)
