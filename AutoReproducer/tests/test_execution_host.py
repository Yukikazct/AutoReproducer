"""Runtime placement is distinct from scientific training parameters."""
from copy import deepcopy
import hashlib
from pathlib import Path
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


def import_packet(sources, *, relaxed=False):
    files = {name: {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'bytes': len(text.encode())} for name, text in sources.items()}
    return {'author_execution_only': relaxed,
            'repository': {'url': 'https://example.invalid/owner/project', 'files': files},
            'file_inventory': {name: {'sha256': source['sha256'], 'bytes': source['bytes']}
                               for name, source in files.items()}}


@pytest.mark.parametrize('operation', ['append', 'insert'])
@pytest.mark.parametrize('cwd,hint,target', [
    ('scripts', '..', 'shared.py'), ('scripts/deep', '../..', 'shared.py'),
    ('scripts', '../models', 'models/local_layer.py'),
    ('scripts', r'..\models', 'models/local_layer.py'),
    ('scripts', '../models/../models', 'models/local_layer.py'),
])
def test_parent_search_hint_resolves_only_canonical_inventory_paths(operation, cwd, hint, target):
    call = f'sys.path.append({hint!r})' if operation == 'append' else f'sys.path.insert(0, {hint!r})'
    entry = f'{cwd}/train.py'
    value = import_packet({entry: f'import sys\n{call}\nimport {Path(target).stem}\n',
                           target: 'import torch\n'})
    before = deepcopy(value)
    result = planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': cwd}, include_sources=True)
    assert set(result['sources']) == {entry, target}
    assert all('..' not in Path(name).parts for name in result['sources'])
    assert [item['module'] for item in result['imports']] == ['torch']
    assert result['ignored_import_paths'] == []
    assert value == before


def test_imported_module_search_hint_stays_relative_to_original_command_cwd():
    value = import_packet({'scripts/train.py': 'from nested import config\n',
        'scripts/nested/config.py': "import sys\nsys.path.append('../models')\nimport local_layer\n",
        'models/local_layer.py': 'import torch\n'})
    result = planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': 'scripts'}, include_sources=True)
    assert 'models/local_layer.py' in result['sources']
    assert [item['module'] for item in result['imports']] == ['torch']


@pytest.mark.parametrize('operation', ['append', 'insert'])
@pytest.mark.parametrize('hint', ['..', '../outside_modules', '../../outside', '../workspace/models'])
def test_external_search_hint_is_ignored_without_reading_or_claiming_external_source(tmp_path, monkeypatch, operation, hint):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    external = tmp_path / 'outside_layer.py'
    external.write_text("raise AssertionError('External module must not be read or executed')\n", encoding='utf-8')
    call = f'sys.path.append({hint!r})' if operation == 'append' else f'sys.path.insert(0, {hint!r})'
    value = import_packet({'train.py': f'import sys\n{call}\nimport outside_layer\nimport torch\n'}, relaxed=True)
    before = deepcopy(value)
    original_read = Path.read_bytes
    def read_inside_only(path):
        assert path.resolve().is_relative_to(workspace.resolve()), 'External source was read'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read_inside_only)
    result = planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': '.'},
                                     include_sources=True, workspace=workspace)
    assert result['sources'] == ['train.py']
    assert [item['module'] for item in result['imports']] == ['outside_layer', 'torch']
    assert result['repository_namespaces'] == []
    ignored, = result['ignored_import_paths']
    assert ignored['literal_path'] == hint and ignored['path'] == 'train.py' and ignored['line'] == 2
    assert ignored['sha256'] == value['repository']['files']['train.py']['sha256']
    assert ignored['reason'] == 'outside_workspace_not_a_frozen_local_import_root'
    assert ignored['evidence_scope'] == 'ignored_static_hint_only' and ignored['external_source_read'] is False
    assert value == before


def test_strict_discovery_still_rejects_external_search_path():
    value = import_packet({'train.py': "import sys\nsys.path.append('..')\nimport outside_layer\n"})
    with pytest.raises(planner.PlanEvidenceError, match='sys.path escapes'):
        planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': '.'}, include_sources=True)


@pytest.mark.parametrize('relaxed', [False, True])
@pytest.mark.parametrize('hint', [
    '/outside', r'C:\outside', r'C:..\outside', r'\outside', r'\\server\share\outside',
    r'\\?\UNC\server\share\outside', '.git/../models', '../.GiT', 'CON/..',
    'bad. /..', 'models:stream/..', 'models\x00/..', '',
])
def test_parent_hint_normalization_keeps_existing_portable_path_rejections(relaxed, hint):
    value = import_packet({'train.py': f'import sys\nsys.path.append({hint!r})\n'}, relaxed=relaxed)
    with pytest.raises(ValueError, match='relative path inside the workspace'):
        planner._author_imports(value, {'argv': ['python', 'train.py'], 'cwd': '.'}, include_sources=True)


def test_execution_plan_preserves_ignored_hints_across_original_stages(tmp_path):
    from src.author_execution_plan import validate_author_execution_plan
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    script = workspace / 'train.py'
    source = "import sys\nsys.path.append('..')\n" + script.read_text('utf-8')
    script.write_text(source, encoding='utf-8')
    stage_source = "import sys\nsys.path.insert(0, '../outside_stage')\nprint('stage')\n"
    (workspace / 'student.py').write_text(stage_source, encoding='utf-8')
    for name in ('train.py', 'student.py'):
        fixture['snapshot']['files'][name] = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
    proposal = deepcopy(fixture['proposal'])
    next(ref for ref in proposal['citations'] if ref['id'] == 'script')['quote'] = source
    proposal['citations'].append({'id': 'student', 'source': 'repository', 'path': 'student.py', 'quote': stage_source})
    proposal.update(scope='author_code_execution', device='cpu')
    proposal['steps'].append({'id': 'student', 'kind': 'train', 'argv': ['python', 'student.py'],
                              'cwd': '.', 'citations': ['student']})
    value = planner.build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    value.update(author_execution_only=True, execution_host={'gpu_probe_confirmed': False, 'cuda_device_indices': []})
    original_commands = [deepcopy(step['argv']) for step in proposal['steps']]
    plan = validate_author_execution_plan(proposal, value, workspace)
    assert [step['argv'] for step in plan['steps']] == original_commands
    ignored = plan['requirements']['ignored_import_paths']
    assert {(item['path'], item['literal_path']) for item in ignored} == {
        ('train.py', '..'), ('student.py', '../outside_stage')}
    assert all(item['external_source_read'] is False for item in ignored)
    assert plan['requirements']['repository_namespaces'] == []
    assert all(hashlib.sha256((workspace / name).read_bytes()).hexdigest() == expected
               for name, expected in fixture['snapshot']['files'].items())
