"""Missing imports trigger bounded retries; unrelated failures remain truthful."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.author_runtime_dependencies import MAX_DEPENDENCY_REPAIRS, repair_requirement, run_with_dependency_repair
from src.discovered_repository_plan import _author_imports, _citations, _requirements, build_evidence_packet
from src.discovered_repository_reproduction import environment_for_plan
from src.dependency_versions import compatibility_pins
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
    assert fix['requirement'] == 'torchvision==' + compatibility_pins()['torchvision']


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
    steps = [{'argv': ['python', 'train.py', '--epochs', '200'], 'timeout_s': None}]
    frozen_requirement = 'torch==' + compatibility_pins()['torch'] + '\n'
    environment = {'requirements_txt': frozen_requirement, 'auto_prepare': False, 'require_prepared': True,
                   'deadline_monotonic': 0}
    data, events, validations = {}, [], []
    result = run_with_dependency_repair(runner, tmp_path, steps, environment, plan(), {'file_inventory': {}},
                                       tmp_path, data, events.append, lambda: validations.append(True))
    assert result['success'] is True and len(runner.calls) == 3
    assert all(call[0] == steps for call in runner.calls)
    assert all(call[1].get('deadline_monotonic') is None for call in runner.calls)
    assert 'einops==0.8.0' in runner.calls[1][1]['requirements_txt']
    assert 'pyyaml==6.0.2' in runner.calls[2][1]['requirements_txt']
    assert runner.calls[1][1]['auto_prepare'] is True and runner.calls[1][1]['require_prepared'] is False
    assert len(data['dependency_repair_attempts']) == 2 and len(validations) == 3
    assert 'einops==0.8.0' in result['effective_env_config']['requirements_txt']
    assert 'pyyaml==6.0.2' in result['effective_env_config']['requirements_txt']
    assert environment['requirements_txt'] == frozen_requirement


def test_repeated_missing_module_does_not_retry_forever_or_report_success(tmp_path):
    runner = FailingRunner(["ModuleNotFoundError: No module named 'einops'"] * 2)
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
                                       tmp_path, {}, lambda e: None, lambda: None)
    assert result['success'] is False and len(runner.calls) == 2


def test_installation_deadline_does_not_trigger_model_or_missing_import_repair(tmp_path):
    runner = FailingRunner(["Earlier output: ModuleNotFoundError: No module named 'einops'\n"
                            "Installer exceeded total time limit"])
    analyze = Mock(return_value={'action': 'replace_compatibility_pin',
                                 'previous': 'numpy==1.26.4', 'requirement': 'numpy==1.24.4'})
    environment = {'requirements_txt': 'numpy==1.26.4', 'dependency_failure_kind': 'installation_timeout'}
    data = {}
    result = run_with_dependency_repair(runner, tmp_path, [], environment, plan(), {'file_inventory': {}},
                                       tmp_path, data, lambda e: None, lambda: None, analyze_failure=analyze)
    assert result['success'] is False and len(runner.calls) == 1
    assert result['effective_env_config']['requirements_txt'] == 'numpy==1.26.4'
    assert not data['dependency_repair_attempts'] and not data['model_dependency_repairs']
    analyze.assert_not_called()


def test_repair_budget_is_shared_by_preflight_and_training(tmp_path):
    data = {'dependency_repair_attempts': [{'package': str(i)} for i in range(MAX_DEPENDENCY_REPAIRS)]}
    runner = FailingRunner(["ModuleNotFoundError: No module named 'einops'"])
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
                                       tmp_path, data, lambda e: None, lambda: None)
    assert result['success'] is False and len(runner.calls) == 1
    assert len(data['dependency_repair_attempts']) == MAX_DEPENDENCY_REPAIRS


def test_pyg_retries_share_the_same_eight_error_budget(tmp_path):
    data = {'pyg_repair_attempts': [{} for _ in range(MAX_DEPENDENCY_REPAIRS)]}
    runner = FailingRunner(['RuntimeError: older version of PyG'])
    repair = Mock(return_value={'kind': 'pyg_legacy_data_state'})
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
        tmp_path, data, lambda event: None, lambda: None, repair_pyg=repair)
    assert not result['success'] and len(runner.calls) == 1
    repair.assert_not_called()


def test_pyg_repair_retries_unchanged_author_command_and_keeps_failure(tmp_path):
    runner = FailingRunner(['RuntimeError: older version of PyG', ''])
    steps = [{'argv': ['python', 'train.py', '--epochs', '1000'], 'timeout_s': None}]
    repair = Mock(return_value={'kind': 'pyg_legacy_data_state'})
    data = {}
    result = run_with_dependency_repair(runner, tmp_path, steps, {}, plan(), {'file_inventory': {}},
        tmp_path, data, lambda event: None, lambda: None, repair_pyg=repair)
    assert result['success'] and len(runner.calls) == 2
    assert all(call[0] == steps for call in runner.calls)
    assert len(data['pyg_repair_attempts']) == 1
    assert data['pyg_repair_attempts'][0]['failed_execution_dir'] == '/retained/1'


def test_more_than_three_distinct_missing_modules_can_be_repaired(tmp_path):
    modules = ['einops', 'yaml', 'tqdm', 'pandas', 'seaborn']
    runner = FailingRunner(["ModuleNotFoundError: No module named '" + module + "'" for module in modules] + [''])
    result = run_with_dependency_repair(runner, tmp_path, [], {}, plan(), {'file_inventory': {}},
                                       tmp_path, {}, lambda e: None, lambda: None)
    assert result['success'] is True and len(runner.calls) == 6
