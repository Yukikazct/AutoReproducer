"""Native build selection stays consistent from planning through runtime repair."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from discovered_fixtures import make_experiment
from src import dependency_versions, discovered_repository_plan as planner
from src.author_execution_plan import validate_author_execution_plan
from src.author_runtime_dependencies import repair_requirement, run_with_dependency_repair
from src.discovered_repository_reproduction import environment_for_plan
from src.method_adapters import NeuralODEAdapter
from src.method_profiles import method_profile


HOST_BUILDS = [
    pytest.param('darwin', 'cpu', ('2.5.1', '0.20.1', '2.1.2', '0.6.18'), id='macos-cpu'),
    pytest.param('linux', 'cpu', ('2.5.1+cpu', '0.20.1+cpu', '2.1.2+pt25cpu', '0.6.18+pt25cpu'), id='linux-cpu'),
    pytest.param('win32', 'cpu', ('2.5.1+cpu', '0.20.1+cpu', '2.1.2+pt25cpu', '0.6.18+pt25cpu'), id='windows-cpu'),
    pytest.param('linux', 'cuda', ('2.5.1+cu121', '0.20.1+cu121', '2.1.2+pt25cu121', '0.6.18+pt25cu121'), id='linux-cuda'),
    pytest.param('win32', 'cuda', ('2.5.1+cu121', '0.20.1+cu121', '2.1.2+pt25cu121', '0.6.18+pt25cu121'), id='windows-cuda'),
]
NATIVE_PACKAGES = ('torch', 'torchvision', 'torch-scatter', 'torch-sparse')


def set_host(monkeypatch, platform_name):
    # Patch only version selection, leaving pytest and Python's filesystem
    # implementation on the real host while exercising the full caller chain.
    monkeypatch.setattr(dependency_versions, 'sys', SimpleNamespace(platform=platform_name))


def graph_experiment(tmp_path):
    fixture = make_experiment(tmp_path)
    script = fixture['workspace'] / 'train.py'
    source = 'import torchvision\nimport torch_sparse\n' + script.read_text(encoding='utf-8')
    script.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c['id'] == 'script')['quote'] = source
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['execution_host'] = {'cpu_available': True, 'cuda_device_indices': [], 'gpu_probe_confirmed': False}
    return fixture, packet


@pytest.mark.parametrize('platform_name,device,versions', HOST_BUILDS)
def test_author_imports_compile_to_installable_native_builds(tmp_path, monkeypatch, platform_name, device, versions):
    fixture, packet = graph_experiment(tmp_path)
    packet['author_execution_only'] = True
    proposal = deepcopy(fixture['proposal'])
    proposal.update(scope='author_code_execution', device=device)
    set_host(monkeypatch, platform_name)

    plan = validate_author_execution_plan(proposal, packet, fixture['workspace'])
    environment = environment_for_plan(plan)

    expected = {name + '==' + version for name, version in zip(NATIVE_PACKAGES, versions)}
    assert expected <= set(plan['requirements']['compatibility'])
    lines = environment['requirements_txt'].splitlines()
    assert expected <= set(lines)
    assert 'scipy==1.14.1' in lines  # torch-sparse's native runtime dependency
    assert f'--find-links https://data.pyg.org/whl/torch-2.5.0+{"cu121" if device == "cuda" else "cpu"}.html' in lines
    assert plan['steps'][0]['argv'] == proposal['steps'][0]['argv']
    assert plan['requirements']['author'] == ['torch==1.13.1']
    assert script_digest(fixture) == fixture['snapshot']['files']['train.py']
    if platform_name == 'darwin':
        assert all('+' not in line for line in lines if not line.startswith('--'))


def script_digest(fixture):
    return hashlib.sha256((fixture['workspace'] / 'train.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('platform_name,device,versions', HOST_BUILDS)
def test_missing_native_import_repairs_use_the_same_build_family(monkeypatch, platform_name, device, versions):
    set_host(monkeypatch, platform_name)
    plan = {'scope': 'author_code_execution', 'capture': {'device': device},
            'requirements': {'author': ['torchvision==0.20.1+cpu', 'torch-sparse==0.6.18+pt25cpu']}}
    packet = {'file_inventory': {}}
    for module, package, version in zip(('torch', 'torchvision', 'torch_scatter', 'torch_sparse'), NATIVE_PACKAGES, versions):
        repair = repair_requirement(f"ModuleNotFoundError: No module named '{module}'", plan, packet)
        assert repair['requirement'] == package + '==' + version
        if module == 'torch_sparse':
            assert set(repair['additions']) == {
                'torch-sparse==' + versions[3], 'torch-scatter==' + versions[2], 'scipy==1.14.1',
            }
    # DeepRobust adds torchvision indirectly; this path must select the same
    # platform build even though torchvision was not the missing import.
    repair = repair_requirement("ModuleNotFoundError: No module named 'deeprobust'", plan, packet)
    assert 'torchvision==' + versions[1] in repair['additions']


@pytest.mark.parametrize('platform_name,device,versions', HOST_BUILDS)
def test_first_dynamic_graph_import_adds_binary_wheel_source(tmp_path, monkeypatch, platform_name, device, versions):
    set_host(monkeypatch, platform_name)
    plan = {'scope': 'author_code_execution', 'capture': {'device': device},
            'requirements': {'author': [], 'compatibility': ['torch==' + versions[0], 'numpy==1.26.4']}}
    environment = environment_for_plan(plan)
    assert '--find-links' not in environment['requirements_txt']
    original_environment = deepcopy(environment)
    steps = [{'argv': ['python', 'train.py', '--epochs', '40'], 'timeout_s': 120}]

    class MissingGraphRunner:
        def __init__(self):
            self.calls = []

        def run(self, workspace, passed_steps, env_config, on_event):
            self.calls.append((deepcopy(passed_steps), deepcopy(env_config)))
            missing = len(self.calls) == 1
            return {'success': not missing, 'run_dir': str(tmp_path / str(len(self.calls))),
                    'final': {'stderr': "ModuleNotFoundError: No module named 'torch_sparse'" if missing else ''}}

    runner = MissingGraphRunner()
    data, events, validations = {}, [], []
    result = run_with_dependency_repair(runner, tmp_path, steps, environment, plan,
                                       {'file_inventory': {}}, tmp_path, data, events.append,
                                       lambda: validations.append(True))

    assert result['success'] is True and len(runner.calls) == 2
    assert all(call[0] == steps for call in runner.calls)
    lines = runner.calls[1][1]['requirements_txt'].splitlines()
    wheel_source = f'--find-links https://data.pyg.org/whl/torch-2.5.0+{"cu121" if device == "cuda" else "cpu"}.html'
    assert lines.count(wheel_source) == 1
    assert {'torch-sparse==' + versions[3], 'torch-scatter==' + versions[2], 'scipy==1.14.1'} <= set(lines)
    assert result['effective_env_config']['requirements_txt'].splitlines() == lines
    assert len(data['dependency_repair_attempts']) == 1 and len(validations) == 2
    assert environment == original_environment
    if platform_name == 'darwin':
        assert all('+' not in line for line in lines if not line.startswith('--'))


@pytest.mark.parametrize('platform_name,device,versions', HOST_BUILDS)
def test_strict_plan_and_environment_reject_unreviewed_versions(tmp_path, monkeypatch, platform_name, device, versions):
    fixture, packet = graph_experiment(tmp_path)
    set_host(monkeypatch, platform_name)
    citations = planner._citations(fixture['proposal'], packet, fixture['workspace'])
    requirements = deepcopy(fixture['proposal']['requirements'])
    requirements['compatibility'] = [name + '==' + version for name, version in zip(NATIVE_PACKAGES, versions)] + ['numpy==1.26.4']
    compiled = planner._requirements(requirements, citations, packet, device, fixture['proposal']['steps'][0])
    plan = {'scope': 'selected_paper_experiment', 'capture': {'device': device}, 'requirements': compiled}
    assert 'torch==' + versions[0] in environment_for_plan(plan)['requirements_txt'].splitlines()

    # The host fix must not loosen exact-version review into accepting any
    # available wheel, or allow a stale plan from another platform.
    foreign_torch = '2.5.1+cpu' if platform_name == 'darwin' else '2.5.1'
    for invalid in ('torch==99.0.0', 'torch-scatter==99.0.0', 'torch==' + foreign_torch):
        package = invalid.split('==')[0]
        proposed = deepcopy(requirements)
        proposed['compatibility'] = [pin for pin in proposed['compatibility'] if not pin.startswith(package + '==')] + [invalid]
        with pytest.raises(planner.PlanEvidenceError, match='version map'):
            planner._requirements(proposed, citations, packet, device, fixture['proposal']['steps'][0])
        altered = deepcopy(plan)
        altered['requirements']['compatibility'] = proposed['compatibility']
        with pytest.raises(ValueError, match='兼容环境'):
            environment_for_plan(altered)


def test_macos_cuda_is_rejected_before_preparing_or_repairing_dependencies(monkeypatch):
    set_host(monkeypatch, 'darwin')
    with pytest.raises(ValueError, match='(?i)(macos.*cuda|cuda.*macos)'):
        planner.compatibility_pins('cuda')
    plan = {'scope': 'author_code_execution', 'capture': {'device': 'cuda'},
            'requirements': {'compatibility': ['torch==2.5.1+cu121'], 'author': []}}
    with pytest.raises(ValueError, match='(?i)(macos.*cuda|cuda.*macos)'):
        environment_for_plan(plan)
    with pytest.raises(ValueError, match='(?i)(macos.*cuda|cuda.*macos)'):
        repair_requirement("ModuleNotFoundError: No module named 'torch'", plan, {'file_inventory': {}})


@pytest.mark.parametrize('platform_name,expected', [('darwin', '2.5.1'), ('linux', '2.5.1+cpu'), ('win32', '2.5.1+cpu')])
def test_planning_prompt_advertises_host_native_builds(tmp_path, monkeypatch, platform_name, expected):
    fixture, packet = graph_experiment(tmp_path)
    set_host(monkeypatch, platform_name)
    prompt = planner._prompt(packet, '')
    assert '"torch":"' + expected + '"' in prompt
    if platform_name == 'darwin':
        assert '"torch":"2.5.1+cpu"' not in prompt
        assert '"torch":"2.5.1+cu121"' not in prompt


@pytest.mark.parametrize('platform_name,expected', [('darwin', '2.5.1'), ('linux', '2.5.1+cpu'), ('win32', '2.5.1+cpu')])
def test_neural_ode_profile_and_import_provenance_agree(tmp_path, monkeypatch, platform_name, expected):
    set_host(monkeypatch, platform_name)
    profile = method_profile('neural_ode_spiral')
    assert 'torch==' + expected in profile['environment']['requirements_txt'].splitlines()
    deps = tmp_path / 'deps'
    imported = {'torch': expected, 'cuda': None,
                'modules': {'torch': str(deps / 'torch' / '__init__.py')},
                'author_package': str(tmp_path / 'torchdiffeq' / '__init__.py')}
    provenance = tmp_path / 'import_provenance.json'
    provenance.write_text(json.dumps(imported), encoding='utf-8')
    execution = {'environment': {'dependencies_path': str(deps)}}
    NeuralODEAdapter().verify_environment(profile, execution, tmp_path)
    imported['torch'] = '2.5.1+cpu' if platform_name == 'darwin' else '2.5.1'
    provenance.write_text(json.dumps(imported), encoding='utf-8')
    with pytest.raises(ValueError, match='Neural ODE'):
        NeuralODEAdapter().verify_environment(profile, execution, tmp_path)
