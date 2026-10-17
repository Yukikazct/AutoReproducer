"""Original module/notebook selection and bounded linked-implementation discovery."""
import hashlib
import json
from pathlib import Path

import pytest

from discovered_fixtures import make_experiment
from src import author_python
from src.author_entrypoints import fixed_cuda_in_notebook, notebook_cells, resolve_entrypoint
from src.author_execution_plan import plan_from_author_commands, propose_author_execution_plan, validate_author_execution_plan, SCOPE
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


def test_model_can_correct_nested_entrypoint_without_a_readme_command(tmp_path):
    fixture = make_experiment(tmp_path)
    add_file(fixture, 'README.md', 'Experiment implementations are in the experiments directory.\n')
    add_file(fixture, 'experiments/sensors/run_model.py',
             'import argparse\nimport torch\n'
             'parser = argparse.ArgumentParser()\n'
             'parser.add_argument("--epochs", type=int, default=17)\n'
             'args = parser.parse_args()\n')
    add_file(fixture, 'experiments/other/run_model.py', 'import torch\n')
    packet = packet_for(fixture)
    packet['repository']['files'].pop('experiments/sensors/run_model.py', None)

    class NavigationLLM:
        mock_mode = False

        def __init__(self):
            self.calls = 0

        def chat(self, prompt, **kwargs):
            self.calls += 1
            assert kwargs['task'] == 'author_execution_plan'
            if self.calls == 1:
                return json.dumps(proposal(fixture, {'argv': ['python', 'run_model.py']}))
            assert 'Matching pinned inventory paths:' in prompt
            assert 'experiments/sensors/run_model.py' in prompt
            assert 'experiments/other/run_model.py' in prompt
            if self.calls == 2:
                return json.dumps({'request_files': ['experiments/sensors/run_model.py']})
            assert 'default=17' in prompt
            return json.dumps(proposal(fixture, {'argv': ['python', 'run_model.py'],
                'cwd': 'experiments/sensors'}))

    llm = NavigationLLM()
    plan = propose_author_execution_plan(llm, packet, fixture['workspace'])
    assert llm.calls == 3
    assert 'command_source' not in plan
    assert plan['steps'][0]['cwd'] == 'experiments/sensors'
    assert plan['steps'][0]['argv'] == ['python', 'run_model.py']
    assert plan['steps'][0]['entrypoint']['source_path'] == 'experiments/sensors/run_model.py'


def test_missing_entrypoint_diagnostics_do_not_allow_path_escape(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = packet_for(fixture)
    with pytest.raises(ValueError, match='relative path inside the workspace'):
        resolve_entrypoint({'argv': ['python', '../outside.py']}, packet, fixture['workspace'])


@pytest.mark.parametrize('script', ['train.py', './train.py', 'nested/train.py'])
def test_inventory_script_target_matches_original_argv_from_nested_cwd(tmp_path, script):
    fixture = make_experiment(tmp_path)
    source_path = 'experiments/' + Path(script).as_posix()
    add_file(fixture, source_path, (fixture['workspace'] / 'train.py').read_text('utf-8'))
    packet = packet_for(fixture)
    argv = ['python3', '-u', script, '--epochs', '17']
    selected = proposal(fixture, {'argv': argv, 'cwd': 'experiments',
                                  'entrypoint': {'kind': 'script', 'target': source_path}})
    plan = validate_author_execution_plan(selected, packet, fixture['workspace'])
    assert plan['steps'][0]['argv'] == argv
    assert plan['steps'][0]['cwd'] == 'experiments'
    assert plan['steps'][0]['entrypoint'] == {
        'kind': 'script', 'target': script, 'source_path': source_path}


@pytest.mark.parametrize('explicit', [
    {'kind': 'script', 'target': 'other/train.py'},
    {'kind': 'module', 'target': 'experiments.train'},
])
def test_explicit_entrypoint_cannot_redirect_the_original_nested_command(tmp_path, explicit):
    fixture = make_experiment(tmp_path)
    for path in ('experiments/train.py', 'other/train.py'):
        add_file(fixture, path, 'import torch\n')
    packet = packet_for(fixture)
    with pytest.raises(ValueError, match='differs from the original argv'):
        resolve_entrypoint({'argv': ['python', 'train.py'], 'cwd': 'experiments',
                            'entrypoint': explicit}, packet, fixture['workspace'])


def test_legacy_syntax_reaches_model_feedback_and_keeps_runtime_failure_kind(tmp_path):
    fixture = make_experiment(tmp_path)
    add_file(fixture, 'README.md', 'Legacy implementation: legacy/train.py.\n')
    add_file(fixture, 'legacy/train.py', "import caffe\nprint 'author training'\n")
    packet = packet_for(fixture)
    selected = proposal(fixture, {'argv': ['python', 'legacy/train.py']})

    class LegacyLLM:
        mock_mode = False

        def __init__(self):
            self.calls = 0

        def chat(self, prompt, **kwargs):
            self.calls += 1
            if self.calls > 1:
                assert 'legacy/train.py:2' in prompt
                assert 'Missing parentheses' in prompt
            return json.dumps(selected)

    llm = LegacyLLM()
    with pytest.raises(PlanEvidenceError, match='legacy/train.py:2') as rejected:
        propose_author_execution_plan(llm, packet, fixture['workspace'])
    assert llm.calls == 2
    assert rejected.value.category == 'runtime_unsupported'
    details = rejected.value.planning_evidence['failure_details']
    assert len(details) == 2
    assert all(row['category'] == 'runtime_unsupported' for row in details)
    assert details[0]['evidence']['path'] == 'legacy/train.py'
    assert (fixture['workspace'] / 'legacy/train.py').read_text('utf-8') == "import caffe\nprint 'author training'\n"


def test_runtime_failure_report_does_not_blame_missing_paper_evidence():
    from src.agents.report_generator import ReportGeneratorAgent
    report = ReportGeneratorAgent()._build_report({
        'execution': {'mode': 'repository', 'executed': False, 'success': False},
        'validation': {'status': 'insufficient_evidence', 'failure_kind': 'runtime_unsupported',
                       'is_reproduced': None, 'reason': 'Original source requires Python 2 and Caffe.'}})
    assert '运行环境不受支持，未执行论文实验' in report
    assert '论文证据不足，未执行论文实验' not in report


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
