"""Real frozen-product missing-dependency recovery and independent install checks."""
from copy import deepcopy
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import Mock
import zipfile

import pytest

from test_recovery_adapter import frozen_products, sha
from src.benchmark.recovery import DEFAULT_BUDGET, DEFAULT_TOOLS
from src.benchmark.recovery_adapter import RecoveryAdapter
from src.benchmark.recovery_environment import (
    _BASELINE_KEY, _check_environment, _environment_pins, _runtime_snapshot, _unclean, prepare_environment_runtime,
    record_environment_execution,
    run_environment_recovery, validate_environment_evidence, validate_environment_receipt,
)


def fake_venv(environment):
    """Static snapshot fixture; never used as an executable by a subprocess."""
    venv = environment / 'author-venv'
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.write_bytes(b'trusted venv launcher')
    base = environment / 'base-python'
    base.write_bytes(b'trusted base interpreter')
    (venv / 'pyvenv.cfg').write_text('include-system-site-packages = false\nexecutable = ' + str(base) + '\n', encoding='utf-8')
    (venv / 'original.py').write_text('value = 7\n', encoding='utf-8')
    return python


def dependency_wheel(module='adapter_runtime_dep', distribution='adapter-runtime-dep', version='1.0', value=7):
    name = distribution.replace('-', '_')
    info = f'{name}-{version}.dist-info'
    files = {
        f'{module}/__init__.py': f'def value():\n    return {value}\n'.encode(),
        f'{info}/METADATA': f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n'.encode(),
        f'{info}/WHEEL': b'Wheel-Version: 1.0\nGenerator: frozen-development-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    rows = []
    for path, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
        rows.append([path, 'sha256=' + digest, str(len(content))])
    rows.append([f'{info}/RECORD', '', ''])
    record = io.StringIO(newline='')
    csv.writer(record).writerows(rows)
    files[f'{info}/RECORD'] = record.getvalue().encode()
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return f'{name}-{version}-py3-none-any.whl', raw.getvalue()


def environment_case(workspace, *, allowed_module='adapter_runtime_dep', source=None, second=False):
    workspace.mkdir()
    wheel, raw = dependency_wheel()
    sources = {
        'train.py': source or b"from adapter_runtime_dep import value\nfrom pathlib import Path\nassert Path('data/split.csv').read_text() == 'frozen split\\n'\nprint(value())\n",
        'data/split.csv': b'frozen split\n', 'fixtures/' + wheel: raw,
    }
    repairs = [{'module': allowed_module, 'requirement': 'adapter-runtime-dep==1.0',
                'additions': ['adapter-runtime-dep==1.0']}]
    if second:
        wheel2, raw2 = dependency_wheel('adapter_other_dep', 'adapter-other-dep', value=11)
        sources['fixtures/' + wheel2] = raw2
        repairs.append({'module': 'adapter_other_dep', 'requirement': 'adapter-other-dep==1.0',
                        'additions': ['adapter-other-dep==1.0']})
        sources['train.py'] = b'from adapter_runtime_dep import value\nfrom adapter_other_dep import value as other\nprint(value() + other())\n'
    for name, content in sources.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    inventory = {name: {'sha256': sha(content), 'text_available': not name.endswith('.whl')}
                 for name, content in sources.items()}
    repository = {'url': 'https://github.com/example/runtime-environment-development', 'revision': 'd' * 40}
    return {'id': 'runtime-environment-development', 'repository_url': repository['url'],
        'repository_revision': repository['revision'], 'source_files': {name: item['sha256'] for name, item in inventory.items()},
        'category': 'runtime_environment', 'split': 'dev', 'origin': 'injected',
        'initial_evidence': {'adapter_version': 1, 'plan': {'repository': repository,
            'steps': [{'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'], 'cwd': '.',
                       'timeout_s': None, 'entrypoint': {'kind': 'script', 'source_path': 'train.py'}}],
            'requirements': {'author': [item['requirement'] for item in repairs],
                             'compatibility': [], 'repository_namespaces': []}, 'capture': {'device': 'cpu'}},
            'packet': {'file_inventory': inventory}, 'environment': {}, 'editable': [],
            'initial_failure': {'error_type': 'ModuleNotFoundError',
                                'diagnostic_contains': "No module named 'adapter_runtime_dep'"},
            'runtime_environment': {'version': 1, 'allowed_repairs': repairs,
                'development_wheels': [{'path': name, 'sha256': record['sha256']} for name, record in inventory.items()
                                       if name.endswith('.whl')]}}}


def setup_environment(tmp_path, frozen_products, *, arm='C', **options):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace, **options)
    private = tmp_path / 'private'
    private.mkdir()
    assertion = b'from adapter_runtime_dep import value\nassert value() == 7\n'
    if options.get('second'):
        assertion += b'from adapter_other_dep import value as other\nassert other() == 11\n'
    (private / 'original_assertions.py').write_bytes(assertion)
    oracle = private / 'oracle.json'
    oracle.write_text(json.dumps({'version': 1, 'scripts': [
        {'path': 'original_assertions.py', 'sha256': sha(assertion)}]}), encoding='utf-8')
    case['oracle'] = {'kind': 'original_assertions_v1', 'reference': 'private independent assertion',
                      'sha256': sha(oracle.read_bytes())}
    products, identities = frozen_products
    adapter = RecoveryAdapter(product_roots=products, oracle_paths={case['id']: oracle},
        development_responses={}, worker_timeout_s=90)
    config = {'mode': {'A': 'baseline', 'B': 'single', 'C': 'collaborative'}[arm],
        'model': 'scripted-development-only', 'tools': [] if arm == 'A' else list(DEFAULT_TOOLS),
        'budget': None if arm == 'A' else deepcopy(DEFAULT_BUDGET), 'product': identities[arm]}
    return adapter, case, workspace, environment, config


def run(adapter, case, workspace, environment, configuration, arm='C'):
    return adapter.run_trial(case={key: deepcopy(value) for key, value in case.items() if key != 'oracle'},
        arm=arm, repeat=1, workspace=workspace, environment=environment, context={}, configuration=configuration)


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_frozen_three_arms_install_only_registered_dependency_and_rerun_original_command(tmp_path, frozen_products, arm):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, arm=arm)
    before = {name: (workspace / name).read_bytes() for name in case['source_files']}
    result = run(adapter, case, workspace, environment, config, arm)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is True, receipt
    assert result['cleanup_confirmed'] is True
    assert receipt['initial_failure_observed'] is True
    recorded = receipt['runtime_environment']
    assert recorded['status'] == 'recovered'
    assert recorded['preserved'] is True
    assert len(recorded['proofs']) == 1
    proof = recorded['proofs'][0]
    assert proof['repair']['requirement'] == 'adapter-runtime-dep==1.0'
    assert proof['importing_sources'] == ['train.py']
    assert recorded['initial_environment']['python_executable'] == recorded['final_environment']['python_executable']
    assert result['usage']['repair_actions'] == 1
    assert result['usage']['model_requests'] == result['usage']['probes'] == result['usage']['source_repair_rounds'] == 0
    assert all((workspace / name).read_bytes() == content for name, content in before.items())
    assert 'src.author_runtime_dependencies' in receipt['loaded_product_modules']
    assert ('src.intelligent_recovery' in receipt['loaded_product_modules']) is (arm != 'A')
    evaluated = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert evaluated['oracle_passed'] is evaluated['protocol_preserved'] is evaluated['cleanup_confirmed'] is True, evaluated
    assert evaluated['numerical_equivalence'] == 'unknown'
    # macOS's canonical temporary root itself contains /private/. Check the
    # actual private oracle directory and filename, not that unrelated word.
    serialized = json.dumps(recorded)
    assert json.dumps(str((tmp_path / 'private').resolve()))[1:-1] not in serialized
    assert 'original_assertions.py' not in serialized


def test_two_actual_missing_imports_share_recovery_budget_and_keep_first_dependency(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, second=True)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is True, receipt
    recorded = receipt['runtime_environment']
    assert [proof['repair']['module'] for proof in recorded['proofs']] == ['adapter_runtime_dep', 'adapter_other_dep']
    assert result['usage']['repair_actions'] == 2
    prefix = ['--no-index', '--find-links ' + (environment / 'wheelhouse').as_uri()]
    assert _environment_pins(recorded['final_environment'], prefix) == ['adapter-runtime-dep==1.0', 'adapter-other-dep==1.0']
    assert adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)['oracle_passed'] is True


def test_unregistered_product_proposal_is_not_installed_or_reported_as_recovered(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, allowed_module='different_module')
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is False
    assert result['blocker'] == 'unsupported'
    assert receipt['runtime_environment']['status'] == 'unsupported'
    assert receipt['runtime_environment']['proofs'] == []
    assert not receipt['dependency_paths']
    assert result['cleanup_confirmed'] is True


def test_successive_missing_modules_do_not_reset_shared_repair_budget(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, second=True)
    config['budget']['max_repair_actions'] = 1
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is False
    assert result['blocker'] == 'budget_exhausted'
    assert result['usage']['repair_actions'] == 1
    assert len(receipt['runtime_environment']['proofs']) == 1
    assert receipt['runtime_environment']['proofs'][0]['repair']['module'] == 'adapter_runtime_dep'
    assert result['cleanup_confirmed'] is True


def test_cancelled_dependency_recovery_preserves_meter_and_confirms_cleanup(tmp_path, frozen_products):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products)
    adapter.cancel_check = lambda: (environment / 'recovery/runtime_repair_environment.json').exists()
    result = run(adapter, case, workspace, environment, config)
    assert result['claimed_success'] is False
    assert result['cancelled'] is True
    assert result['cleanup_confirmed'] is True
    assert result['usage']['repair_actions'] == 1
    assert result['usage']['model_requests'] == result['usage']['source_repair_rounds'] == 0


@pytest.mark.parametrize('source', [b'print(7)\n', b"raise RuntimeError('a different actual failure')\n"])
def test_unobserved_registered_failure_is_not_a_protocol_violation(tmp_path, frozen_products, source):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, source=source)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is False
    assert receipt['initial_failure_observed'] is False
    assert receipt['runtime_environment']['status'] == 'not_attempted'
    assert receipt['runtime_environment']['proofs'] == []
    assert result['usage']['repair_actions'] == 0
    evaluated = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert evaluated['protocol_preserved'] is evaluated['cleanup_confirmed'] is True, evaluated
    assert evaluated['oracle_passed'] is False


@pytest.mark.parametrize('field,value', [('split', 'holdout'), ('origin', 'natural')])
def test_offline_development_transport_cannot_be_enabled_for_real_cases(tmp_path, field, value):
    case = environment_case(tmp_path / 'workspace')
    case[field] = value
    with pytest.raises(ValueError, match='injected development'):
        validate_environment_evidence(case, tmp_path / 'workspace', case['initial_evidence'])


@pytest.mark.parametrize('error', ['AttributeError', 'ImportError', 'OSError', 'TypeError'])
def test_version_api_and_native_errors_are_explicitly_outside_supported_envelope(tmp_path, error):
    case = environment_case(tmp_path / 'workspace')
    case['initial_evidence']['initial_failure']['error_type'] = error
    with pytest.raises(ValueError, match='Only missing-module'):
        validate_environment_evidence(case, tmp_path / 'workspace', case['initial_evidence'])


def test_frozen_wheel_bytes_and_transport_are_checked_before_install(tmp_path):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    base = {'requirements_txt': 'adapter-runtime-dep==1.0', 'python_executable': str(fake_venv(environment))}
    prepared = prepare_environment_runtime(contract, workspace, environment, base)
    assert prepared['requirements_txt'].splitlines()[:2] == ['--no-index', '--find-links ' + (environment / 'wheelhouse').as_uri()]
    assert base['requirements_txt'] == 'adapter-runtime-dep==1.0'
    (environment / 'wheelhouse' / 'extra.whl').write_bytes(b'unregistered')
    with pytest.raises(ValueError, match='unregistered inputs'):
        prepare_environment_runtime(contract, workspace, environment, base)


@pytest.mark.parametrize('change', [
    {'python_executable': 'another-python'}, {'repository_pythonpath': ['unregistered']},
    {'offline': True}, {'dependency_health_check': True}, {'device': 'cuda'},
])
def test_complete_runtime_controls_cannot_be_changed_by_model_or_executor(change):
    baseline = {'python_executable': 'fixed-python', 'auto_prepare': False, 'dependency_health_check': False}
    candidate = {**baseline, 'requirements_txt': 'adapter-runtime-dep==1.0\n', 'auto_prepare': True,
                 'require_prepared': False, 'exclude_environment_preparation_from_deadline': True, **change}
    with pytest.raises(ValueError, match='Protected runtime'):
        _check_environment(candidate, baseline, ['adapter-runtime-dep==1.0'], [], repaired=True)


@pytest.mark.parametrize('option', ['--extra-index-url https://example.invalid/simple', '--index-url https://example.invalid',
                                  '--find-links other', '--trusted-host example.invalid', '-r injected.txt', '--no-deps'])
def test_fixed_transport_prefix_does_not_hide_extra_install_options(tmp_path, option):
    prefix = ['--no-index', '--find-links ' + str(tmp_path / 'wheelhouse')]
    environment = {'requirements_txt': '\n'.join([*prefix, option, 'adapter-runtime-dep==1.0'])}
    with pytest.raises(ValueError, match='Unregistered runtime install options'):
        _environment_pins(environment, prefix)


@pytest.mark.parametrize('target', ['metadata', 'payload', 'final_environment', 'venv_configuration',
                                   'venv_file', 'interpreter', 'forged_snapshot'])
def test_independent_oracle_rejects_tampered_installed_environment(tmp_path, frozen_products, target):
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is True, receipt
    dependency = Path(receipt['dependency_paths'][0])
    if target == 'metadata':
        path = next(dependency.glob('*.dist-info/METADATA'))
        path.write_text(path.read_text().replace('Version: 1.0', 'Version: 2.0'), encoding='utf-8')
    elif target == 'payload':
        (dependency / 'adapter_runtime_dep/__init__.py').write_text('def value():\n    return 7 # changed payload\n', encoding='utf-8')
    elif target == 'final_environment':
        receipt['runtime_environment']['final_environment']['requirements_txt'] += '\n--extra-index-url https://example.invalid/simple\n'
    elif target == 'venv_configuration':
        (environment / 'author-venv/pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding='utf-8')
    elif target == 'venv_file':
        (environment / 'author-venv/undeclared.py').write_text('value = 11\n', encoding='utf-8')
    elif target == 'interpreter':
        # Replace only the disposable venv launcher; never mutate a POSIX
        # symlink's shared system target.
        launcher = Path(receipt['author_python'])
        launcher.unlink()
        launcher.write_bytes(b'replaced interpreter')
    else:
        receipt['runtime_environment']['final_environment'][_BASELINE_KEY]['inventory_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_environment_receipt(receipt, case, workspace, environment)
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored['oracle_passed'] is scored['protocol_preserved'] is False


_CLEANUP_MARKERS = [
    {'process_cleanup_confirmed': False}, {'cleanup_confirmed': False},
    {'cleanup_errors': ['descendant still alive']}, {'cleanup_error': 'job close failed'},
    {'cleanup_failed': True}, {'cleanup_unconfirmed': True}, {'status': 'cleanup_failed'},
    {'final': {'cleanup_errors': ['descendant still alive']}},
    {'steps': [{'cleanup_receipt': {'status': 'unconfirmed'}}]},
    {'attempts': [{'process_cleanup': {'confirmed': False}}]},
    {'stages': [{'cleanup_receipt': {'confirmed': False}}]},
    {'environment': {'cleanup_receipt': {'status': 'failed'}}},
    {'effective_env_config': {'cleanup_errors': ['installer still alive']}},
    {'reason': 'process-tree cleanup failed'}, {'final': {'stderr': 'process group cleanup was denied'}},
    {'final': {'stdout': 'cleanup_unconfirmed'}},
]


def test_shared_and_cyclic_receipt_wrappers_do_not_hide_cleanup_uncertainty():
    shared = {'status': 'failed'}
    record = {'environment': shared, 'cleanup_receipt': shared}
    record['attempts'] = [record]
    assert _unclean(record) is True
    clean = {}
    clean['attempts'] = [clean]
    assert _unclean(clean) is False


def _merge_marker(target, marker):
    for key, value in marker.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_marker(target[key], value)
        else:
            target[key] = deepcopy(value)


@pytest.mark.parametrize('marker', _CLEANUP_MARKERS)
def test_unconfirmed_cleanup_does_not_start_an_environment_retry(tmp_path, marker):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    evidence = case['initial_evidence']
    contract = validate_environment_evidence(case, workspace, evidence)
    python = fake_venv(environment)
    baseline = {'python_executable': str(python), 'auto_prepare': False, 'dependency_health_check': False}
    baseline = prepare_environment_runtime(contract, workspace, environment, baseline)
    execution = {'success': False, 'effective_env_config': baseline}
    _merge_marker(execution, marker)
    runner, analyze, data = Mock(), Mock(), {}
    result = run_environment_recovery(runner, execution, [], evidence['plan'], evidence['packet'], workspace,
        baseline, environment, data, lambda: None, analyze, lambda event: None,
        contract=contract, trial_environment=environment)
    assert result is execution
    assert data['_runtime_environment']['status'] == 'cleanup_unconfirmed'
    runner.run.assert_not_called()
    analyze.assert_not_called()
    assert data.get('self_repair', {}).get('attempts', []) == []
    assert 'dependency_repair_attempts' not in data


@pytest.mark.parametrize('marker', _CLEANUP_MARKERS)
def test_cleanup_uncertainty_after_retry_stops_before_more_analysis_or_repair_actions(tmp_path, marker):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace, second=True)
    evidence = case['initial_evidence']
    contract = validate_environment_evidence(case, workspace, evidence)
    python = fake_venv(environment)
    baseline = prepare_environment_runtime(contract, workspace, environment,
        {'python_executable': str(python), 'auto_prepare': False, 'dependency_health_check': False})
    runtime = {'executable': str(python), 'runtime': 'frozen-development-runtime'}
    def missing(module, line):
        return {'stderr': 'Traceback (most recent call last):\n'
            + f'  File "{workspace / "train.py"}", line {line}, in <module>\n'
            + f"ModuleNotFoundError: No module named '{module}'\n"}
    execution = {'success': False, 'executed': True, 'environment': runtime,
                 'effective_env_config': baseline, 'final': missing('adapter_runtime_dep', 1)}
    returned = []
    def retry(selected_workspace, selected_steps, selected_environment, **kwargs):
        result = {'success': False, 'executed': True, 'environment': deepcopy(runtime),
            'effective_env_config': deepcopy(selected_environment), 'final': missing('adapter_other_dep', 2)}
        _merge_marker(result, marker)
        returned.append(result)
        return result
    runner, analyze, data = Mock(), Mock(return_value=None), {}
    runner.run.side_effect = retry
    recovery = environment / 'recovery'
    recovery.mkdir()
    with pytest.raises(RuntimeError, match='cleanup was not confirmed') as captured:
        run_environment_recovery(runner, execution, evidence['plan']['steps'], evidence['plan'], evidence['packet'],
            workspace, baseline, recovery, data, lambda: None, analyze, lambda event: None,
            contract=contract, trial_environment=environment)
    assert captured.value.process_cleanup_confirmed is False
    assert runner.run.call_count == 1
    analyze.assert_not_called()
    assert len(data['self_repair']['attempts']) == len(data['dependency_repair_attempts']) == 1
    assert len(data['_runtime_environment']['proofs']) == 1
    assert data['_runtime_environment_last_execution'] is returned[0]


@pytest.mark.parametrize('mutation', ['add', 'change', 'delete', 'base_interpreter', 'launcher', 'cache'])
def test_runtime_byte_changes_cannot_reset_baseline_or_start_recovery(tmp_path, mutation):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    evidence = case['initial_evidence']
    contract = validate_environment_evidence(case, workspace, evidence)
    python = fake_venv(environment)
    baseline = prepare_environment_runtime(contract, workspace, environment,
        {'python_executable': str(python), 'auto_prepare': False, 'dependency_health_check': False})
    venv = environment / 'author-venv'
    if mutation == 'add':
        (venv / 'unregistered.pth').write_text('import arbitrary\n', encoding='utf-8')
    elif mutation == 'change':
        (venv / 'pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding='utf-8')
    elif mutation == 'delete':
        (venv / 'original.py').unlink()
    elif mutation == 'base_interpreter':
        (environment / 'base-python').write_bytes(b'changed shared executable')
    elif mutation == 'launcher':
        python.write_bytes(b'replaced launcher')
    else:
        cache = venv / '__pycache__'
        cache.mkdir()
        (cache / 'original.cpython-312.pyc').write_bytes(b'untrusted executable cache')
    with pytest.raises(ValueError):
        prepare_environment_runtime(contract, workspace, environment, baseline)
    execution = {'success': False, 'executed': True, 'effective_env_config': deepcopy(baseline)}
    runner, analyze, data = Mock(), Mock(), {}
    result = run_environment_recovery(runner, execution, evidence['plan']['steps'], evidence['plan'],
        evidence['packet'], workspace, baseline, environment, data, lambda: None, analyze, lambda event: None,
        contract=contract, trial_environment=environment)
    assert result is execution
    assert data['_runtime_environment']['status'] == 'environment_changed'
    assert data['_runtime_environment']['preserved'] is False
    assert data['_runtime_environment_last_execution'] is execution
    assert not data.get('dependency_repair_attempts')
    runner.run.assert_not_called()
    analyze.assert_not_called()


def test_executor_cannot_replace_the_trusted_initial_snapshot_with_a_fresh_one(tmp_path):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    baseline = prepare_environment_runtime(contract, workspace, environment,
        {'python_executable': str(fake_venv(environment)), 'auto_prepare': False})
    (environment / 'author-venv/undeclared.py').write_text('value = 99\n', encoding='utf-8')
    forged = deepcopy(baseline)
    forged[_BASELINE_KEY] = _runtime_snapshot(environment, forged)
    proof = record_environment_execution({'effective_env_config': forged}, workspace, environment,
                                         baseline, contract=contract)
    assert proof['preserved'] is False
    assert proof['status'] == 'environment_changed'
    assert proof['initial_environment'] == baseline
    assert proof['final_environment'] == forged


def test_initial_runner_observations_are_retained_after_trusted_controls_are_checked(tmp_path):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    baseline = prepare_environment_runtime(contract, workspace, environment,
        {'python_executable': str(fake_venv(environment)), 'auto_prepare': False})
    actual = {**deepcopy(baseline), 'dependency_cache_maintenance': [{'phase': 'scope_started'}]}
    proof = record_environment_execution({'effective_env_config': actual}, workspace, environment,
                                         baseline, contract=contract)
    assert proof['preserved'] is True
    assert proof['initial_environment'] == proof['final_environment'] == actual
    assert 'dependency_cache_maintenance' not in baseline


@pytest.mark.parametrize('target', ['configuration', 'actual_binary'])
def test_nested_base_venv_redirects_are_bound_to_the_actual_interpreter(tmp_path, target):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    python = fake_venv(environment)
    parent_venv = environment / 'parent-venv'
    parent_launcher = parent_venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    parent_launcher.parent.mkdir(parents=True)
    parent_launcher.write_bytes(b'intermediate launcher')
    actual_base = environment / 'base-python'
    parent_config = parent_venv / 'pyvenv.cfg'
    parent_config.write_text('executable = ' + str(actual_base) + '\n', encoding='utf-8')
    (environment / 'author-venv/pyvenv.cfg').write_text('include-system-site-packages = false\nexecutable = '
                                                       + str(parent_launcher) + '\n', encoding='utf-8')
    baseline = prepare_environment_runtime(contract, workspace, environment, {'python_executable': str(python)})
    chain = baseline[_BASELINE_KEY]['base_interpreters']
    assert [record['path'] for record in chain] == [str(parent_launcher), str(actual_base)]
    if target == 'configuration':
        parent_config.write_text(parent_config.read_text() + 'home = altered\n', encoding='utf-8')
    else:
        actual_base.write_bytes(b'changed actual interpreter')
    with pytest.raises(ValueError, match='bytes changed'):
        prepare_environment_runtime(contract, workspace, environment, baseline)


def test_snapshot_cleanup_uncertainty_takes_priority_over_missing_venv_files(tmp_path):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    python = fake_venv(environment)
    baseline = prepare_environment_runtime(contract, workspace, environment, {'python_executable': str(python)})
    python.unlink()
    proof = record_environment_execution({'cleanup_confirmed': False}, workspace, environment,
                                         baseline, contract=contract)
    assert proof['status'] == 'cleanup_unconfirmed'
    assert proof['preserved'] is False


@pytest.mark.skipif(os.name == 'nt', reason='POSIX venv launchers and lib64 links')
def test_posix_venv_links_preserve_launcher_and_bind_external_interpreter_bytes(tmp_path):
    workspace, environment = tmp_path / 'workspace', tmp_path / 'environment'
    environment.mkdir()
    case = environment_case(workspace)
    contract = validate_environment_evidence(case, workspace, case['initial_evidence'])
    python = fake_venv(environment)
    python.unlink()
    python.symlink_to(environment / 'base-python')
    (environment / 'author-venv/lib').mkdir()
    (environment / 'author-venv/lib64').symlink_to('lib', target_is_directory=True)
    baseline = prepare_environment_runtime(contract, workspace, environment, {'python_executable': str(python)})
    assert baseline['python_executable'] == str(python)
    assert prepare_environment_runtime(contract, workspace, environment, baseline) == baseline
    outside = tmp_path / 'outside'
    outside.mkdir()
    (environment / 'author-venv/escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='external link'):
        prepare_environment_runtime(contract, workspace, environment, baseline)


@pytest.mark.parametrize('phase', ['initial', 'retry'])
def test_real_author_venv_mutation_stops_recovery_and_cannot_earn_success(tmp_path, frozen_products, phase):
    mutation = b"from pathlib import Path\nimport sys\n(Path(sys.prefix) / 'undeclared-author-file.py').write_text('value = 99\\n')\n"
    missing = b'from adapter_runtime_dep import value\n'
    source = mutation + missing if phase == 'initial' else missing + mutation + b'print(value())\n'
    adapter, case, workspace, environment, config = setup_environment(tmp_path, frozen_products, source=source)
    result = run(adapter, case, workspace, environment, config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert result['claimed_success'] is False, receipt
    assert result['cleanup_confirmed'] is True
    assert receipt['runtime_environment']['preserved'] is False
    assert receipt['runtime_environment']['status'] == ('environment_changed' if phase == 'initial' else 'stopped')
    assert result['usage']['repair_actions'] == (0 if phase == 'initial' else 1)
    assert result['usage']['model_requests'] == 0
    evaluated = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert evaluated['oracle_passed'] is evaluated['protocol_preserved'] is False


def test_trusted_driver_loads_helper_without_importing_any_product_module():
    helper = Path(__file__).resolve().parents[1] / 'src/benchmark/recovery_environment.py'
    # AST makes this invariant independent of src modules already imported by pytest.
    parsed = __import__('ast').parse(helper.read_text(encoding='utf-8'))
    for node in parsed.body:
        if isinstance(node, __import__('ast').ImportFrom):
            assert not (node.module or '').startswith(('src', 'packaging'))
        elif isinstance(node, __import__('ast').Import):
            assert all(not alias.name.startswith(('src', 'packaging')) for alias in node.names)
    loaded = runpy.run_path(str(helper))
    assert callable(loaded['run_environment_recovery'])
