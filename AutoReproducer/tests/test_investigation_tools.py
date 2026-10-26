import hashlib
from pathlib import Path
import sys

import pytest

from src.investigation_tools import InvestigationTools


@pytest.fixture
def tools(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'original_test.py').write_text('def square(x):\n    return x*x\ndef test_square():\n    assert square(3) == 9\n')
    (root / 'unsafe.py').write_text('def test_write():\n    open("unwanted", "w").write("bad")\n    assert True\n')
    (root / 'data.csv').write_text('id,target\na,secret-target\nb,another-target\n')
    inventory = {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text_available': True}
                 for path in root.iterdir()}
    return InvestigationTools(root, {'file_inventory': inventory, 'investigation_artifacts': {
        'data.csv': {'sha256': inventory['data.csv']['sha256'], 'public': True}}},
        {'python_executable': sys.executable}, tmp_path)


def test_source_is_hash_bound_and_no_path_escape(tools):
    result = tools.execute('source_context', {'paths': ['original_test.py']})
    assert result['status'] == 'completed'
    assert result['evidence'][0]['value']['definitions'][0]['name'] == 'square'
    assert tools.execute('source_context', {'paths': ['../outside.py']})['status'] == 'rejected'
    (tools.workspace / 'original_test.py').write_text('print("changed")')
    assert tools.execute('source_context', {'paths': ['original_test.py']})['status'] == 'rejected'


def test_investigator_can_request_frozen_paper_context_for_unknown_protocol_fields(tools):
    text = 'We report ten runs; the test metric is expressed as a percentage.'
    tools.packet['pdf'] = {'sha256': 'a' * 64, 'pages': [
        {'page': 3, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}]}
    result = tools.execute('source_context', {'paper_pages': [3]})
    assert result['status'] == 'completed'
    assert result['evidence'][0]['value']['text'] == text
    assert result['evidence'][0]['pdf_sha256'] == 'a' * 64
    assert tools.execute('source_context', {'paper_pages': [True]})['status'] == 'rejected'
    assert tools.execute('source_context', {'paper_pages': [3], 'paths': ['original_test.py']})['status'] == 'rejected'
    tools.packet['pdf']['pages'][0]['text'] = 'The input changed.'
    assert tools.execute('source_context', {'paper_pages': [3]})['status'] == 'rejected'


def test_metadata_observes_selected_python_without_importing_author(tools):
    (tools.workspace / 'trigger.py').write_text('raise RuntimeError("MUST NOT IMPORT")')
    content = (tools.workspace / 'trigger.py').read_bytes()
    tools.packet['file_inventory']['trigger.py'] = {'sha256': hashlib.sha256(content).hexdigest()}
    result = tools.execute('runtime_metadata', {'packages': ['pip'], 'modules': ['trigger']})
    assert result['status'] == 'completed', result
    value = result['evidence'][0]['value']
    assert Path(value['python_executable']).resolve() == Path(sys.executable).resolve()
    assert value['modules']['trigger']['import_executed'] is False
    assert value['local_modules']['trigger'][0]['path'] == 'trigger.py'


def test_missing_author_python_never_falls_back_to_application(tools):
    tools.environment = {}
    assert tools.execute('runtime_metadata', {'packages': ['pip']})['status'] == 'rejected'


def test_owned_helper_without_cleanup_confirmation_stops_investigation(tools, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('src.runtime_preparation.run_owned_process',
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{}'))
    result = tools.execute('runtime_metadata', {'packages': ['pip']})
    assert result['status'] == 'stopped'
    assert result['cleanup_confirmed'] is False


def test_metadata_uses_real_runner_dependency_roots(tools, tmp_path):
    packages = tmp_path / 'recorded_author_dependencies'
    metadata = packages / 'original_package-1.2.3.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: original-package\nVersion: 1.2.3\n')
    tools.environment['investigation_dependency_paths'] = [str(packages.resolve())]
    result = tools.execute('runtime_metadata', {'packages': ['original-package']})
    assert result['status'] == 'completed', result
    assert result['evidence'][0]['value']['packages']['original-package']['version'] == '1.2.3'


def test_contract_runs_only_original_pure_assertion_in_managed_child(tools):
    result = tools.execute('author_contract', {'path': 'original_test.py', 'function': 'test_square'})
    assert result['status'] == 'completed', result
    assert result['cleanup_confirmed'] is True
    assert result['evidence'][0]['value']['numerical_equivalence'] == 'unknown'
    unsafe = tools.execute('author_contract', {'path': 'unsafe.py', 'function': 'test_write'})
    assert unsafe['status'] == 'rejected'
    assert not (tools.workspace / 'unwanted').exists()


def test_original_assertions_can_refute_a_candidate_without_modifying_sources(tools):
    path = tools.workspace / 'original_test.py'
    original = path.read_text()
    sha = tools.packet['file_inventory'][path.name]['sha256']
    candidate = original.replace('return x*x', 'return x+x')
    proposal = {'patches': [{'path': path.name, 'base_sha256': sha, 'content': candidate}]}
    with tools.candidate_contracts(proposal):
        checked = tools.execute('author_contract', {'path': path.name, 'function': 'test_square'})
        assert checked['status'] == 'failed'
        assert checked['evidence'][0]['value']['scope'] == 'candidate_with_unchanged_original_assertions'
    assert path.read_text() == original
    assert tools.execute('author_contract', {'path': path.name, 'function': 'test_square'})['status'] == 'completed'
    proposal['patches'][0]['content'] = candidate.replace('== 9', '== 6')
    with tools.candidate_contracts(proposal):
        assert tools.execute('author_contract', {'path': path.name, 'function': 'test_square'})['status'] == 'rejected'


@pytest.mark.parametrize('stopped', [False, True])
def test_patch_review_probe_enforces_original_contract_and_propagates_cleanup_stop(tools, stopped, monkeypatch):
    import json
    from src.intelligent_recovery import RecoverySession, InvestigationStopped
    path = tools.workspace / 'original_test.py'
    original = path.read_text()
    calls = []
    class Reviewer:
        def chat(self, prompt, **kwargs):
            calls.append(kwargs['task'])
            return json.dumps({'action': 'probe', 'reason': 'Check the original assertion.',
                'evidence_ids': ['patch_sources'], 'probe': {'tool': 'author_contract',
                'arguments': {'path': path.name, 'function': 'test_square'}}})
    session = RecoverySession(Reviewer())
    session._toolset = tools
    session._active_task = {'status': 'repair_proposed', 'probes': [], 'decisions': [], 'hypotheses': []}
    proposal = {'patches': [{'path': path.name,
        'base_sha256': tools.packet['file_inventory'][path.name]['sha256'],
        'content': original.replace('return x*x', 'return x+x')}]}
    if stopped:
        monkeypatch.setattr(tools, 'execute', lambda *args: {
            'status': 'stopped', 'cleanup_confirmed': False, 'evidence': []})
    with pytest.raises(InvestigationStopped if stopped else ValueError) as error:
        session.review_patch({'sources': {path.name: original}}, proposal)
    if stopped:
        assert error.value.process_cleanup_confirmed is False
    else:
        assert 'failed the unchanged original assertion' in str(error.value)
    assert len(calls) == 1
    assert tools._candidate_sources == {}
    assert path.read_text() == original


def test_artifact_structure_does_not_disclose_targets(tools):
    result = tools.execute('artifact_contract', {'paths': ['data.csv']})
    assert result['status'] == 'completed'
    assert result['evidence'][0]['value']['rows'] == 2
    assert 'secret-target' not in str(result)
    assert tools.execute('source_context', {'paths': ['data.csv']})['status'] == 'rejected'
    assert tools.execute('artifact_contract', {'paths': ['private/labels.csv']})['status'] == 'rejected'


def test_private_or_unreadable_inventory_is_not_a_source_or_public_artifact(tools):
    tools.packet['file_inventory']['original_test.py']['text_available'] = False
    assert tools.execute('source_context', {'paths': ['original_test.py']})['status'] == 'rejected'
    assert tools.execute('artifact_contract', {'paths': ['original_test.py']})['status'] == 'rejected'


@pytest.mark.parametrize('expression', ['[0] * (10**10)', '2 ** (2 ** 40)', "'%1000000000s' % ''", 'round(100, -1000000000)'])
def test_original_pure_contract_cannot_allocate_unbounded_values(tools, expression):
    path = tools.workspace / 'allocation_test.py'
    path.write_text('def test_allocation():\n    value = ' + expression + '\n    assert value\n')
    tools.packet['file_inventory'][path.name] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text_available': True}
    result = tools.execute('author_contract', {'path': path.name, 'function': 'test_allocation'})
    assert result['status'] == 'rejected', result
    assert result['cleanup_confirmed'] is True


@pytest.mark.parametrize('source,expected', [
    ('def abs(x):\n    return 0\ndef test_original():\n    assert abs(-1) == 1\n', 'failed'),
    ('def test_original(x, /):\n    assert True\n', 'rejected'),
    ('def test_original():\n    return\n    assert False\n', 'rejected'),
    ('def test_original():\n    abs = 0\n    assert abs(-1) == 1\n', 'rejected'),
    ('def test_original():\n    assert abs(-1) == 1\n    abs = 0\n', 'rejected'),
    ('abs = 0\ndef test_original():\n    assert abs(-1) == 1\n', 'rejected'),
    ('from custom import abs\ndef test_original():\n    assert abs(-1) == 1\n', 'rejected'),
    ('def helper(x, x):\n    return x\ndef test_original():\n    assert helper(1, 2) == 2\n', 'rejected'),
    ('def test_original():\n    assert True\ndef test_original():\n    assert True\n', 'rejected'),
])
def test_contract_does_not_report_a_different_python_binding_or_unexecuted_assertion(tools, source, expected):
    path = tools.workspace / 'binding_test.py'
    path.write_text(source)
    tools.packet['file_inventory'][path.name] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text_available': True}
    result = tools.execute('author_contract', {'path': path.name, 'function': 'test_original'})
    assert result['status'] == expected, result
    assert result['cleanup_confirmed'] is True


@pytest.mark.parametrize('name,readable', [('private/oracle.py', True), ('hidden/oracle.py', True), ('public_test.py', False)])
def test_contract_cannot_inspect_private_or_non_readable_tests(tools, name, readable):
    path = tools.workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('def test_hidden():\n    assert 3 == 3\n')
    tools.packet['file_inventory'][name] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text_available': readable}
    result = tools.execute('author_contract', {'path': name, 'function': 'test_hidden'})
    assert result['status'] == 'rejected'
    assert not result['evidence']


@pytest.mark.parametrize('tool,args', [('shell', {'command': 'anything'}),
    ('source_context', {'paths': ['data.csv'], 'execute': True}),
    ('runtime_metadata', {'modules': ['os;evil']}),
    ('author_contract', {'path': 'original_test.py', 'function': 'test_square', 'expected': 9})])
def test_model_cannot_supply_executables_or_oracles(tools, tool, args):
    assert tools.execute(tool, args)['status'] == 'rejected'
