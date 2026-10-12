"""Runtime placement is distinct from scientific training parameters."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from src import execution_host
from src import discovered_repository_plan as planner
from discovered_fixtures import make_experiment


CUDA_SOURCE = '''import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5000)
args = parser.parse_args()
torch.cuda.set_device(args.cuda_device)
'''


def packet(source=CUDA_SOURCE):
    return {'author_execution_only': True, 'execution_host': {'cuda_device_indices': [0]},
            'repository': {'files': {'train.py': {'text': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}}}}


def test_only_available_source_bound_placement_may_change():
    value = packet()
    options = execution_host.placement_options(value, 'train.py')
    assert set(options) == {'--cuda_device'}
    planner._verify_author_arguments(['--cuda_device', '0', '--epochs', '5000'], CUDA_SOURCE, value, script='train.py', placements=options)
    with pytest.raises(ValueError, match='unavailable device'):
        planner._verify_author_arguments(['--cuda_device', '2'], CUDA_SOURCE, value, script='train.py', placements=options)
    with pytest.raises(ValueError, match='differs'):
        planner._verify_author_arguments(['--epochs', '1'], CUDA_SOURCE, value, script='train.py', placements=options)


def test_gpu_index_override_cannot_change_a_method_parameter():
    value = packet(CUDA_SOURCE + 'epochs = args.cuda_device\n')
    assert execution_host.placement_options(value, 'train.py') == {}
    with pytest.raises(ValueError, match='differs'):
        planner._verify_author_arguments(['--cuda_device', '0'], value['repository']['files']['train.py']['text'], value, script='train.py')


def test_original_launcher_argv_literals_can_select_an_author_training_run():
    value = packet()
    quote = CUDA_SOURCE + "\ncommand = ['python', 'train.py', '--epochs', '100']\n"
    planner._verify_author_arguments(['--epochs', '100'], quote, value, script='train.py')
    with pytest.raises(ValueError, match='differs'):
        planner._verify_author_arguments(['--epochs', '1'], quote, value, script='train.py')


def test_readme_apostrophes_and_source_quotes_do_not_erase_real_commands():
    value = packet()
    quote = "Author's implementation.\n```python\npython train.py --epochs 20\n```\n" + CUDA_SOURCE
    planner._verify_author_arguments(['--epochs', '20'], quote, value, script='train.py')
    with pytest.raises(ValueError, match='differs'):
        planner._verify_author_arguments(['--epochs', '1'], quote, value, script='train.py')


def test_source_without_original_placement_api_is_not_an_override():
    assert execution_host.placement_options(packet(CUDA_SOURCE.replace('torch.cuda.set_device(args.cuda_device)', 'print(args.cuda_device)')), 'train.py') == {}


def test_hardware_probe_respects_visible_numeric_devices(monkeypatch):
    monkeypatch.setattr(execution_host.shutil, 'which', lambda _: 'nvidia-smi')
    monkeypatch.setattr(execution_host.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='0\n1\n'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert execution_host.probe_execution_host()['cuda_device_indices'] == [0]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert execution_host.probe_execution_host()['cuda_device_indices'] == []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '9')
    assert execution_host.probe_execution_host()['gpu_probe_confirmed'] is False


@pytest.mark.parametrize('operation', ["sys.path.append('models/')", "sys.path.insert(0, 'models/')"])
def test_literal_author_search_path_resolves_fixed_local_source(tmp_path, operation):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    (workspace / 'models').mkdir()
    sources = {'train.py': f'import sys\n{operation}\nfrom local_layer import Layer\n',
               'models/local_layer.py': 'import torch\nclass Layer: pass\n'}
    for name, text in sources.items():
        (workspace / name).write_bytes(text.encode('utf-8'))
        fixture['snapshot']['files'][name] = hashlib.sha256(text.encode()).hexdigest()
    value = planner.build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    if 'models/local_layer.py' not in value['repository']['files']:
        planner._add_requested_files(['models/local_layer.py'], value, workspace)
    imports = planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': '.'}, include_sources=True)
    assert 'models/local_layer.py' in imports['sources']
    assert [item['module'] for item in imports['imports']] == ['torch']


def test_search_path_after_import_does_not_rewrite_earlier_import_identity(tmp_path):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    (workspace / 'models').mkdir()
    sources = {'train.py': "import sys\nfrom local_layer import Layer\nsys.path.append('models/')\n",
               'models/local_layer.py': 'class Layer: pass\n'}
    for name, text in sources.items():
        (workspace / name).write_bytes(text.encode('utf-8'))
        fixture['snapshot']['files'][name] = hashlib.sha256(text.encode()).hexdigest()
    value = planner.build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    imports = planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': '.'}, include_sources=True)
    assert 'models/local_layer.py' not in imports['sources']
    assert any(item['module'] == 'local_layer' for item in imports['imports'])
