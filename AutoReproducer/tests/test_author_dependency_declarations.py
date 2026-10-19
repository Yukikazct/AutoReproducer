"""Project metadata and import identity must agree before any runtime repair."""
import hashlib
import json

import pytest

from discovered_fixtures import make_experiment
from src.author_runtime_dependencies import repair_requirement
from src.author_self_repair import repair_local_import
from src.dependency_versions import compatibility_pins
from src.discovered_repository_plan import (
    PlanEvidenceError, _requirements, build_evidence_packet,
)
from src.discovered_repository_reproduction import environment_for_plan


def project_fixture(tmp_path, module, declarations, *, nested=False):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    script = 'experiments/train.py' if nested else 'train.py'
    files = {script: f'import {module}\nimport torch\n',
             'pyproject.toml': '[project]\ndependencies = ' + json.dumps(declarations) + '\n',
             'vendor/other/pyproject.toml': '[project]\ndependencies = ["unrelated-package==9.9"]\n',
             f'helpers/{module}.py': 'VALUE = 42\n'}
    for name, text in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    packet['author_execution_only'] = True
    step = {'argv': ['python', script], 'cwd': '.'}
    proposal = {'author': [], 'compatibility': [], 'reason': 'Use pinned author sources', 'citations': []}
    return workspace, packet, step, proposal


@pytest.mark.parametrize('module,declaration,expected', [
    ('hydra', 'hydra-core==1.3.2', 'hydra-core==1.3.2'),
    ('external_engine', 'External_Engine == 2.4', 'External_Engine==2.4'),
    ('yaml', 'pyyaml==5.4.1', 'pyyaml==' + compatibility_pins()['pyyaml']),
    ('torch_geometric', 'torch-geometric==2.4.0', 'torch-geometric==' + compatibility_pins()['torch-geometric']),
])
def test_pep621_dependencies_survive_model_omission_and_keep_native_policy(tmp_path, module, declaration, expected):
    workspace, packet, step, proposal = project_fixture(tmp_path, module, [declaration, 'unused_tool==3.2'], nested=True)
    compiled = _requirements(proposal, {}, packet, step=step, workspace=workspace)
    assert declaration in compiled['author']
    assert 'unused_tool==3.2' in compiled['author']
    assert expected in compiled['compatibility']
    assert all('unused_tool' not in item and 'unrelated-package' not in item for item in compiled['compatibility'])
    assert all(item['path'] == 'pyproject.toml' and item['sha256'] == packet['file_inventory']['pyproject.toml']['sha256']
               for item in compiled['declaration_evidence'])
    plan = {'scope': 'author_code_execution', 'capture': {'device': 'cpu'}, 'requirements': compiled}
    assert expected in environment_for_plan(plan)['requirements_txt']
    diagnostic = (f'Traceback (most recent call last):\n  File "{workspace / "experiments/train.py"}", line 1, in <module>\n'
                  f"ModuleNotFoundError: No module named '{module}'")
    assert repair_local_import({'final': {'stderr': diagnostic}}, workspace, packet, compiled) is None
    assert repair_requirement(diagnostic, plan, packet)['requirement'] == expected


def test_closest_project_owns_its_dependency_declarations(tmp_path):
    workspace, packet, step, proposal = project_fixture(tmp_path, 'external_engine', ['external-engine==1.0'], nested=True)
    path = workspace / 'experiments/pyproject.toml'
    path.write_bytes(b'[project]\ndependencies = ["external-engine==2.0"]\n')
    packet['file_inventory']['experiments/pyproject.toml'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    compiled = _requirements(proposal, {}, packet, step=step, workspace=workspace)
    assert compiled['author'] == ['external-engine==2.0']
    assert 'external-engine==2.0' in compiled['compatibility']


@pytest.mark.parametrize('metadata', ['name = "inner-project"', 'name = "inner-project"\ndynamic = ["dependencies"]'])
def test_nested_project_without_static_dependencies_does_not_inherit_parent_pins(tmp_path, metadata):
    workspace, packet, step, proposal = project_fixture(tmp_path, 'external_engine', ['external-engine==1.0'], nested=True)
    path = workspace / 'experiments/pyproject.toml'
    path.write_bytes(('[project]\n' + metadata + '\n').encode())
    packet['file_inventory']['experiments/pyproject.toml'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    compiled = _requirements(proposal, {}, packet, step=step, workspace=workspace)
    assert not compiled['author'] and not compiled['declared_runtime_pins']


@pytest.mark.parametrize('corruption', ['workspace', 'packet', 'invalid_requirement', 'invalid_table'])
def test_unverified_or_invalid_project_metadata_cannot_become_installation_input(tmp_path, corruption):
    workspace, packet, step, proposal = project_fixture(tmp_path, 'external_engine', ['external-engine==1.0'])
    if corruption == 'workspace':
        (workspace / 'pyproject.toml').write_bytes(b'[project]\ndependencies = ["external-engine==9.9"]\n')
    elif corruption == 'packet':
        packet['repository']['files']['pyproject.toml']['text'] += '\n# altered\n'
    else:
        raw = (b'[project]\ndependencies = ["--extra-index-url https://example.invalid"]\n'
               if corruption == 'invalid_requirement' else b'project = "bad"\n')
        (workspace / 'pyproject.toml').write_bytes(raw)
        packet['file_inventory']['pyproject.toml']['sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PlanEvidenceError):
        _requirements(proposal, {}, packet, step=step, workspace=None if corruption == 'packet' else workspace)


def test_conditional_declarations_preserve_markers_for_the_target_interpreter(tmp_path):
    declaration = 'external-engine>=2; python_version < "3.12"'
    workspace, packet, step, proposal = project_fixture(tmp_path, 'external_engine', [declaration])
    compiled = _requirements(proposal, {}, packet, step=step, workspace=workspace)
    assert not compiled['declared_runtime_pins']
    assert compiled['author'] == [declaration]
    fix = repair_requirement("ModuleNotFoundError: No module named 'external_engine'",
                             {'requirements': compiled, 'capture': {'device': 'cpu'}}, packet)
    assert fix['requirement'] == declaration


@pytest.mark.parametrize('declarations', [
    ['external-engine==1.0', 'external-engine==2.0'],
    ['external-engine @ https://example.invalid/engine.whl'],
])
def test_ambiguous_or_direct_dependencies_do_not_fall_back_to_a_namesake(tmp_path, declarations):
    workspace, packet, step, proposal = project_fixture(tmp_path, 'external_engine', declarations)
    compiled = _requirements(proposal, {}, packet, step=step, workspace=workspace)
    assert not compiled['declared_runtime_pins']
    diagnostic = (f'Traceback (most recent call last):\n  File "{workspace / "train.py"}", line 1, in <module>\n'
                  "ModuleNotFoundError: No module named 'external_engine'")
    assert repair_local_import({'final': {'stderr': diagnostic}}, workspace, packet, compiled) is None
    assert repair_requirement(diagnostic, {'requirements': compiled, 'capture': {'device': 'cpu'}}, packet) is None
