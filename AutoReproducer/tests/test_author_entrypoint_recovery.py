"""Original module/notebook selection and bounded linked-implementation discovery."""
import hashlib
import json
from pathlib import Path

import pytest

from discovered_fixtures import make_experiment
from src import author_python
from src.author_entrypoints import fixed_cuda_in_notebook, notebook_cells
from src.author_execution_plan import plan_from_author_commands, validate_author_execution_plan, SCOPE
from src.discovered_repository_plan import build_evidence_packet, PlanEvidenceError
from src.discovered_repository_reproduction import linked_training_repositories


def packet_for(fixture):
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    return packet


def add_file(fixture, name, content):
    path = fixture['workspace'] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()


def proposal(fixture, step):
    return {'version': 1, 'scope': SCOPE,
            'repository': {key: fixture['snapshot'][key] for key in ('url', 'revision')},
            'device': 'cpu', 'steps': [{'id': 'train', 'kind': 'train', 'cwd': '.', **step}]}


def test_module_entrypoint_resolves_relative_imports_from_pinned_package(tmp_path):
    fixture = make_experiment(tmp_path)
    add_file(fixture, 'package/__init__.py', '')
    add_file(fixture, 'package/train.py', 'from .helper import train\nif __name__ == "__main__": train()\n')
    add_file(fixture, 'package/helper.py', 'import torch\ndef train(): pass\n')
    packet = packet_for(fixture)
    selected = proposal(fixture, {'argv': ['python', '-m', 'package.train'],
                                  'entrypoint': {'kind': 'module', 'target': 'package.train'}})
    plan = validate_author_execution_plan(selected, packet, fixture['workspace'])
    assert plan['steps'][0]['entrypoint']['source_path'] == 'package/train.py'
    assert 'package/helper.py' in packet['repository']['files']
    assert plan['steps'][0]['argv'] == ['python', '-m', 'package.train']


def test_documented_notebook_is_planned_without_fake_python_script(tmp_path):
    fixture = make_experiment(tmp_path)
    add_file(fixture, 'README.md', 'Train the paper method with training.ipynb.\n')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'language': 'python'}},
                'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                           'source': ['import torch\n', 'x = torch.nn.Parameter(torch.tensor(1.0))\n',
                                      'optimizer = torch.optim.SGD([x], lr=0.1)\n',
                                      '(x*x).backward()\n', 'optimizer.step()\n']}]}
    add_file(fixture, 'training.ipynb', json.dumps(notebook))
    packet = packet_for(fixture)
    plan = plan_from_author_commands(packet, fixture['workspace'])
    assert plan['steps'][0]['entrypoint']['kind'] == 'notebook'
    assert plan['steps'][0]['argv'] == []
    assert 'nbclient==0.10.2' in plan['requirements']['compatibility']


def test_notebook_cuda_requirement_is_reported_before_training(tmp_path):
    fixture = make_experiment(tmp_path)
    notebook = {'nbformat': 4, 'metadata': {'kernelspec': {'language': 'python'}},
                'cells': [{'cell_type': 'code', 'source': "device = 'cuda'\n", 'metadata': {},
                           'execution_count': None, 'outputs': []}]}
    add_file(fixture, 'training.ipynb', json.dumps(notebook))
    packet = packet_for(fixture)
    assert fixed_cuda_in_notebook(notebook_cells(packet, fixture['workspace'], 'training.ipynb'))
    with pytest.raises(PlanEvidenceError, match='CUDA') as rejected:
        validate_author_execution_plan(proposal(fixture, {
            'entrypoint': {'kind': 'notebook', 'target': 'training.ipynb'}}),
            packet, fixture['workspace'])
    assert rejected.value.category == 'hardware_unavailable'


def test_linked_training_repo_excludes_improved_method_and_dependency_urls():
    text = '\n'.join([
        '* A [python version of R-FCN](https://github.com/Orpine/py-R-FCN) is available.',
        '* Recommended deformable R-FCN: https://github.com/msracver/Deformable-ConvNets',
        '* Dependency: https://github.com/example/caffe'])
    packet = {'repository': {'url': 'https://github.com/daijifeng001/r-fcn',
                             'files': {'README.md': {'text': text, 'sha256': 'a' * 64}}}}
    found = linked_training_repositories(packet)
    assert [item['url'] for item in found] == ['https://github.com/Orpine/py-R-FCN']
    assert 'python version' in found[0]['relationship']['quote']


def test_python_declaration_selects_an_existing_compatible_interpreter(tmp_path, monkeypatch):
    packet = {'repository': {'files': {'pyproject.toml': {'text': '[project]\nrequires-python = ">=3.10,<3.12"\n'}}}}
    fake = tmp_path / 'python3.11'
    fake.touch()
    monkeypatch.setattr(author_python.shutil, 'which',
                        lambda name: str(fake) if name == 'python3.11' else None)
    monkeypatch.setattr(author_python, '_probe',
                        lambda path: (3, 11) if Path(path) == fake else (3, 12))
    monkeypatch.setattr(author_python, '_managed_venv', lambda root, base, version: fake)
    monkeypatch.setattr(author_python, 'runtime_fingerprint', lambda executable: 'cpython-311-test')
    selected = author_python.select_author_python(packet, tmp_path / 'managed', offline=True)
    assert selected['python_version'] == '3.11'
    assert selected['source'] == 'isolated_existing'


def test_actual_requested_package_python_bound_skips_incompatible_interpreter(tmp_path, monkeypatch):
    packet = {'repository': {'files': {}}}
    existing = tmp_path / 'python3.9'
    existing.touch()
    monkeypatch.setattr(author_python.shutil, 'which',
                        lambda name: str(existing) if name == 'python3.9' else None)
    monkeypatch.setattr(author_python, '_probe',
                        lambda path: (3, 9) if Path(path) == existing else (3, 12))
    monkeypatch.setattr(author_python, '_managed_venv', lambda root, base, version: existing)
    monkeypatch.setattr(author_python, 'runtime_fingerprint', lambda executable: 'cpython-39-test')
    selected = author_python.select_author_python(packet, tmp_path / 'managed', offline=True,
                                                  required_python='<3.10')
    assert selected['python_version'] == '3.9'
    assert selected['requirements'] == [{'path': 'actual_requested_dependency', 'specifier': '<3.10'}]


def test_requested_python_conflict_switches_and_retries_original_preflight(tmp_path, monkeypatch):
    from src.author_python import retry_requested_python_conflict
    error = "ERROR: Package 'example' requires a different Python: 3.12.0 not in '<3.10'"
    initial = {'success': False, 'effective_env_config': {'requirements_txt': 'example==1.2.0\n',
        'dependency_failure_kind': 'python_incompatible',
        'dependency_install_attempts': [{'failure_kind': 'python_incompatible', 'diagnostic': error}]}}
    calls = []
    def choose(packet, root, **kwargs):
        calls.append(('choose', kwargs))
        return {'python_executable': '/managed/python3.9', 'python_version': '3.9'}
    monkeypatch.setattr(author_python, 'select_author_python', choose)
    def preflight(environment):
        calls.append(('preflight', dict(environment)))
        return {'success': True}
    environment = {'requirements_txt': 'example==1.2.0\n', 'python_executable': '/managed/python3.12'}
    result = retry_requested_python_conflict({}, tmp_path, environment, {'python_version': '3.12'},
                                              initial, preflight)
    assert result['success'] is True
    assert calls[0][1]['required_python'] == '<3.10'
    assert calls[0][1]['exclude_versions'] == {(3, 12)}
    assert calls[1][1]['python_executable'] == '/managed/python3.9'
    assert calls[1][1]['requirements_txt'] == 'example==1.2.0\n'
