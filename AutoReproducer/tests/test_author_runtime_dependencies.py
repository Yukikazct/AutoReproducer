"""Missing imports trigger bounded retries; unrelated failures remain truthful."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.author_runtime_dependencies import repair_requirement, run_with_dependency_repair
from src.discovered_repository_plan import _author_imports, _citations, _requirements, build_evidence_packet
from src.discovered_repository_reproduction import environment_for_plan
from discovered_fixtures import make_experiment


def plan():
    return {'scope': 'author_code_execution', 'capture': {'device': 'cpu'},
            'requirements': {'author': ['einops==0.8.0'], 'repository_namespaces': []}}


@pytest.mark.parametrize('error', [
    'ValueError: invalid dataset', 'ImportError: cannot import name removed from numpy',
    "ModuleNotFoundError: No module named 'author_model'", "ModuleNotFoundError: No module named 'os'",
])
def test_non_missing_dependency_and_original_local_modules_are_not_pypi_repairs(error):
    assert repair_requirement(error, plan(), {'file_inventory': {'author_model.py': {}}}) is None


def test_readme_pin_is_preferred_and_native_package_aliases_keep_matching_pair():
    packet = {'file_inventory': {}}
    assert repair_requirement("ModuleNotFoundError: No module named 'einops'", plan(), packet)['requirement'] == 'einops==0.8.0'
    fix = repair_requirement("ModuleNotFoundError: No module named 'torchvision'", plan(), packet)
    assert fix['requirement'] == 'torchvision==0.20.1+cpu'


def test_unknown_imports_use_author_declarations_or_wait_for_actual_error(tmp_path):
    fixture = make_experiment(tmp_path)
    path = fixture['workspace'] / 'train.py'
    text = 'import einops\nimport missing_dependency\n' + path.read_text('utf8')
    path.write_text(text, encoding='utf8')
    import hashlib
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(x for x in fixture['proposal']['citations'] if x['id'] == 'script')['quote'] = text
    deps = fixture['workspace'] / 'requirements.txt'
    deps.write_text('torch==1.13.1\neinops==0.8.0\n', encoding='utf8')
    fixture['snapshot']['files']['requirements.txt'] = hashlib.sha256(deps.read_bytes()).hexdigest()
    next(x for x in fixture['proposal']['citations'] if x['id'] == 'requirements')['quote'] = deps.read_text('utf8')
    fixture['proposal']['requirements']['author'].append('einops==0.8.0')
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    citations = _citations(fixture['proposal'], packet, fixture['workspace'])
    compiled = _requirements(fixture['proposal']['requirements'], citations, packet, step=fixture['proposal']['steps'][0])
    assert compiled['declared_runtime_pins'] == ['einops==0.8.0']
    assert compiled['deferred_runtime_imports'][0]['module'] == 'missing_dependency'
    environment = environment_for_plan({**plan(), 'requirements': compiled})
    assert 'einops==0.8.0' in environment['requirements_txt']
    packet['author_execution_only'] = False
    with pytest.raises(ValueError, match='outside the reviewed runtime'):
        _requirements(fixture['proposal']['requirements'], citations, packet, step=fixture['proposal']['steps'][0])


class FailingRunner:
    def __init__(self, errors):
        self.errors, self.calls = iter(errors), []

    def run(self, workspace, steps, env_config, on_event):
        self.calls.append((deepcopy(steps), deepcopy(env_config)))
        error = next(self.errors)
        on_event({'type': 'execution_run', 'execution_id': str(len(self.calls)), 'status': 'error' if error else 'success'})
        return {'success': not error, 'run_dir': '/retained/' + str(len(self.calls)), 'final': {'stderr': error}}


def test_two_missing_modules_install_and_retry_same_author_argv_without_ending_task(tmp_path):
    runner = FailingRunner(["ModuleNotFoundError: No module named 'einops'",
                            "ModuleNotFoundError: No module named 'yaml'", ''])
    steps = [{'argv': ['python', 'train.py', '--epochs', '200'], 'timeout_s': 300}]
    environment = {'requirements_txt': 'torch==2.5.1+cpu\n', 'auto_prepare': False, 'require_prepared': True}
    data, events, validations = {}, [], []
    result = run_with_dependency_repair(runner, tmp_path, steps, environment, plan(), {'file_inventory': {}},
                                       tmp_path, data, events.append, lambda: validations.append(True))
    assert result['success'] is True and len(runner.calls) == 3
    assert all(call[0] == steps for call in runner.calls)
    assert 'einops==0.8.0' in runner.calls[1][1]['requirements_txt']
    assert 'pyyaml==6.0.2' in runner.calls[2][1]['requirements_txt']
    assert runner.calls[1][1]['auto_prepare'] is True and runner.calls[1][1]['require_prepared'] is False
    assert len(data['dependency_repair_attempts']) == 2 and len(validations) == 3
    assert 'einops==0.8.0' in result['effective_env_config']['requirements_txt']
    assert 'pyyaml==6.0.2' in result['effective_env_config']['requirements_txt']
    assert environment['requirements_txt'] == 'torch==2.5.1+cpu\n'


def test_repeated_missing_module_does_not_retry_forever_or_report_success(tmp_path):
    runner = FailingRunner(["ModuleNotFoundError: No module named 'einops'"] * 2)
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
                                       tmp_path, {}, lambda e: None, lambda: None)
    assert result['success'] is False and len(runner.calls) == 2


def test_repair_budget_is_shared_by_preflight_and_training(tmp_path):
    data = {'dependency_repair_attempts': [{'package': 'one'}, {'package': 'two'}, {'package': 'three'}]}
    runner = FailingRunner(["ModuleNotFoundError: No module named 'einops'"])
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
                                       tmp_path, data, lambda e: None, lambda: None)
    assert result['success'] is False and len(runner.calls) == 1
    assert len(data['dependency_repair_attempts']) == 3
