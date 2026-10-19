"""Automatic recovery must verify real execution and preserve original inputs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.author_runtime_dependencies import repair_requirement, run_with_dependency_repair
from src.author_self_repair import repair_local_import
from src.repository_runner import RepositoryRunner


def original_repository(root, source_root='lib', module='author_support', entrypoint='train.py', package=False):
    module_path = root / source_root / module / '__init__.py' if package else root / source_root / (module + '.py')
    module_path.parent.mkdir(parents=True)
    module_path.write_text('VALUE = 17\n', encoding='utf-8')
    script = root / entrypoint
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f'from {module} import VALUE\nprint(VALUE)\n', encoding='utf-8')
    files = {p.relative_to(root).as_posix(): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
             'text_available': True} for p in root.rglob('*.py')}
    return {'file_inventory': files}


def dependency_plan():
    return {'requirements': {'author': [], 'compatibility': [], 'repository_namespaces': []},
            'capture': {'device': 'cpu'}}


@pytest.mark.parametrize('source_root,module,entrypoint,package', [
    ('lib', 'author_support', 'train.py', False),
    ('src', 'signal_processing', 'experiments/evaluate.py', True),
    ('vendor/internal', 'image_math', 'benchmarks/predict.py', False),
])
def test_real_nested_import_failure_is_repaired_without_rewriting_source_or_argv(
        tmp_path, source_root, module, entrypoint, package):
    packet = original_repository(tmp_path, source_root, module, entrypoint, package)
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(return_value=None)
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    runner = RepositoryRunner(executor=executor)
    steps = [{'id': 'train', 'argv': ['python', entrypoint], 'cwd': '.', 'timeout_s': 10}]
    original_steps = deepcopy(steps)
    data, events = {}, []
    analyze = Mock(return_value=None)
    result = run_with_dependency_repair(runner, tmp_path, steps, {}, dependency_plan(), packet,
        tmp_path, data, events.append, lambda: None, analyze_failure=analyze)
    assert result['success'], result['final']['stderr']
    assert result['final']['stdout'].strip() == '17'
    assert steps == original_steps
    assert result['effective_env_config']['repository_pythonpath'] == [source_root]
    assert not data['dependency_repair_attempts']
    attempt = data['self_repair']['attempts'][0]
    assert attempt['action'] == 'restore_repository_import' and attempt['status'] == 'resolved'
    assert Path(attempt['failed_execution_dir']).is_dir()
    assert json.loads((tmp_path / 'self_repair.json').read_text('utf-8'))['status'] == 'resolved'
    analyze.assert_not_called()
    assert any(e.get('log', {}).get('action') == 'self_repair' for e in events)
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == record['sha256']
               for name, record in packet['file_inventory'].items())


class ScriptedRunner:
    def __init__(self, errors):
        self.errors, self.calls = iter(errors), []

    def run(self, workspace, steps, environment, **kwargs):
        self.calls.append(deepcopy(environment))
        error = next(self.errors)
        return {'success': not error, 'run_dir': f'/run/{len(self.calls)}',
                'final': {'stderr': error}, 'effective_env_config': deepcopy(environment)}


def test_known_dependency_errors_do_not_consume_model_analysis_calls(tmp_path):
    runner = ScriptedRunner(["ModuleNotFoundError: No module named 'einops'", ''])
    analyze = Mock(return_value=None)
    data = {}
    result = run_with_dependency_repair(runner, tmp_path, [], {}, dependency_plan(), {'file_inventory': {}},
        tmp_path, data, lambda _: None, lambda: None, analyze_failure=analyze)
    assert result['success']
    analyze.assert_not_called()
    assert data['self_repair']['attempts'][0]['status'] == 'resolved'


def test_caught_or_quoted_missing_module_does_not_trigger_dependency_installation(tmp_path):
    diagnostic = ("ModuleNotFoundError: No module named 'einops'\nRecovered optional import\n"
                  'Traceback (most recent call last):\n  File "train.py", line 20, in train\n'
                  'ValueError: incompatible tensor shape')
    assert repair_requirement(diagnostic, dependency_plan(), {'file_inventory': {}}) is None
    runner = ScriptedRunner([diagnostic])
    data = {}
    result = run_with_dependency_repair(runner, tmp_path, [], {}, dependency_plan(), {'file_inventory': {}},
        tmp_path, data, lambda _: None, lambda: None)
    assert not result['success'] and len(runner.calls) == 1
    assert not data['dependency_repair_attempts']


def test_only_the_last_exception_can_identify_the_missing_dependency():
    diagnostic = ("ModuleNotFoundError: No module named 'obsolete'\n\n"
                  'During handling of the above exception, another exception occurred:\n\n'
                  'Traceback (most recent call last):\n'
                  "ModuleNotFoundError: No module named 'einops'")
    assert repair_requirement(diagnostic, dependency_plan(), {'file_inventory': {}})['module'] == 'einops'


@pytest.mark.parametrize('mode', ['repeated', 'cancel', 'timeout', 'unknown'])
def test_repair_stops_with_a_persisted_reason_instead_of_claiming_success(tmp_path, mode):
    error = "ModuleNotFoundError: No module named 'einops'" if mode == 'repeated' else 'RuntimeError: original failure'
    runner = ScriptedRunner([error, error])
    if mode in {'cancel', 'timeout'}:
        runner.run = Mock(return_value={'success': False, 'final': {
            'stderr': error, 'cancelled': mode == 'cancel', 'timed_out': mode == 'timeout'}})
    data = {}
    outcome = run_with_dependency_repair(runner, tmp_path, [], {}, dependency_plan(), {'file_inventory': {}},
        tmp_path, data, lambda _: None, lambda: None)
    assert not outcome['success']
    state = json.loads((tmp_path / 'self_repair.json').read_text('utf-8'))
    assert state['status'] == ('no_progress' if mode == 'repeated' else
                               'interrupted' if mode == 'cancel' else 'timed_out' if mode == 'timeout' else 'blocked')
    assert state['reason']


def test_ambiguous_or_changed_local_sources_are_never_guessed(tmp_path):
    packet = original_repository(tmp_path)
    error = {'final': {'stderr': f'Traceback (most recent call last):\n  File "{tmp_path / "train.py"}", line 1, in <module>\n'
             "ModuleNotFoundError: No module named 'author_support'"}}
    assert repair_local_import(error, tmp_path, packet)['root'] == 'lib'
    (tmp_path / 'other').mkdir()
    duplicate = tmp_path / 'other' / 'author_support.py'
    duplicate.write_text('VALUE = 99\n', encoding='utf-8')
    packet['file_inventory']['other/author_support.py'] = {'sha256': hashlib.sha256(duplicate.read_bytes()).hexdigest()}
    assert repair_local_import(error, tmp_path, packet) is None
    packet['file_inventory'].pop('other/author_support.py')
    (tmp_path / 'lib' / 'author_support.py').write_text('VALUE = 42\n', encoding='utf-8')
    assert repair_local_import(error, tmp_path, packet) is None


@pytest.mark.parametrize('module,requirement,root', [
    ('hydra', 'hydra-core==1.3.2', 'source/utils'),
    ('yaml', 'PyYAML==6.0.2', 'helpers'),
    ('PIL', 'Pillow==10.4.0', 'tools/image'),
    ('external_engine', 'external-engine==2.4', 'vendor/internal'),
])
def test_declared_distribution_wins_over_same_named_nested_helper(tmp_path, module, requirement, root):
    packet = original_repository(tmp_path, source_root=root, module=module)
    plan = dependency_plan()
    plan['requirements']['author'] = [requirement]
    diagnostic = (f'Traceback (most recent call last):\n  File "{tmp_path / "train.py"}", line 1, in <module>\n'
                  f"ModuleNotFoundError: No module named '{module}'")
    runner = ScriptedRunner([diagnostic, ''])
    data = {}
    steps = [{'argv': ['python', 'train.py', '--epochs', '200'], 'cwd': '.'}]
    result = run_with_dependency_repair(runner, tmp_path, steps, {}, plan, packet,
        tmp_path, data, lambda _: None, lambda: None)
    assert result['success'] and len(runner.calls) == 2
    assert not data['import_repair_attempts']
    assert data['dependency_repair_attempts'][0]['author_declaration'] == requirement
    assert 'repository_pythonpath' not in runner.calls[1]


@pytest.mark.parametrize('module', ['configuration', 'metrics'])
def test_unrelated_package_helper_is_not_promoted_to_a_top_level_module(tmp_path, module):
    packet = original_repository(tmp_path, source_root='source/utils', module=module)
    initializer = tmp_path / 'source/__init__.py'
    initializer.write_bytes(b'')
    packet['file_inventory']['source/__init__.py'] = {'sha256': hashlib.sha256(b'').hexdigest()}
    error = {'final': {'stderr': f'Traceback (most recent call last):\n  File "{tmp_path / "train.py"}", line 1, in <module>\n'
             f"ModuleNotFoundError: No module named '{module}'"}}
    assert repair_local_import(error, tmp_path, packet) is None


def test_legacy_sibling_import_can_still_restore_its_original_directory(tmp_path):
    packet = original_repository(tmp_path, source_root='source/utils', entrypoint='source/utils/train.py')
    initializer = tmp_path / 'source/__init__.py'
    initializer.write_bytes(b'')
    packet['file_inventory']['source/__init__.py'] = {'sha256': hashlib.sha256(b'').hexdigest()}
    error = {'final': {'stderr': f'Traceback (most recent call last):\n  File "{tmp_path / "source/utils/train.py"}", line 1, in <module>\n'
             "ModuleNotFoundError: No module named 'author_support'"}}
    assert repair_local_import(error, tmp_path, packet)['root'] == 'source/utils'
