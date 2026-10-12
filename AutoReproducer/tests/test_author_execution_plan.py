"""Execution-first mechanics use invented sources, not paper success evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from discovered_fixtures import PlannedLLM, make_experiment
from src import author_execution_plan as author
from src.discovered_repository_plan import build_evidence_packet, PlanEvidenceError
from src.method_adapters import read_json
from test_discovered_repository_runtime import native_environment
from test_discovered_repository_service import setup_service, use_preinstalled_cache, cpu_dependency_cache


def execution_proposal(fixture):
    original = fixture['proposal']
    return {'version': 1, 'scope': author.SCOPE, 'repository': original['repository'], 'device': 'cpu',
        'citations': deepcopy(original['citations']), 'claims': {'method': original['claims']['protocol'],
            'protocol': original['claims']['protocol'], 'dataset': original['claims']['dataset']},
        'requirements': deepcopy(original['requirements']), 'datasets': deepcopy(original['datasets']),
        'steps': deepcopy(original['steps'])}


def accepted_execution_review():
    return {'accepted': True, 'checks': {name: {'passed': True,
        'reason': 'The complete invented author command uses its full original fixture data.',
        'citations': ['paper_protocol', 'readme', 'script', 'requirements']}
        for name in author.CHECKS}}


class AuthorLLM(PlannedLLM):
    def __init__(self, fixture, review=None):
        super().__init__(fixture['proposal'])
        self.execution = execution_proposal(fixture)
        self.execution_review = review or accepted_execution_review()

    def chat(self, prompt, **kwargs):
        if kwargs['task'] in ('author_execution_plan', 'author_execution_plan_review'):
            self.calls.append((prompt, kwargs))
            return json.dumps(self.execution if kwargs['task'] == 'author_execution_plan' else self.execution_review)
        return super().chat(prompt, **kwargs)


def fixture_packet(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    return fixture, packet


def test_explicit_execution_objective_does_not_need_metric_or_checkpoint_claims(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    plan = author.propose_author_execution_plan(AuthorLLM(fixture), packet, fixture['workspace'])
    assert plan['scope'] == author.SCOPE
    assert plan['metrics'] == [] and plan['capture'] == {'kind': 'author_process', 'device': 'cpu'}
    assert plan['steps'][0]['argv'] == fixture['proposal']['steps'][0]['argv']
    assert plan['semantic_review']['accepted'] is True


def test_no_silent_downgrade_of_paper_reference_objective(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    del packet['author_execution_only']
    with pytest.raises(PlanEvidenceError, match='explicit execution objective'):
        author.validate_author_execution_plan(execution_proposal(fixture), packet, fixture['workspace'])


@pytest.mark.parametrize('change', [
    lambda p: p.update(code='print(1)'),
    lambda p: p['repository'].update(revision='b' * 40),
    lambda p: p['steps'][0].update(argv=['python', '-c', 'print(1)']),
    lambda p: p['steps'][0]['argv'].append('--smoke'),
    lambda p: p['steps'][0].update(required=False),
    lambda p: p['datasets'][0].update(paths=['*.json']),
])
def test_execution_objective_keeps_original_source_data_and_command_guards(tmp_path, change):
    fixture, packet = fixture_packet(tmp_path)
    proposal = deepcopy(execution_proposal(fixture))
    change(proposal)
    with pytest.raises(PlanEvidenceError):
        author.validate_author_execution_plan(proposal, packet, fixture['workspace'])


@pytest.mark.parametrize('gate', list(author.CHECKS))
def test_execution_review_records_negative_findings_and_allows_actual_runtime_check(tmp_path, gate):
    fixture, packet = fixture_packet(tmp_path)
    review = accepted_execution_review()
    review['checks'][gate]['passed'] = False
    plan = author.propose_author_execution_plan(AuthorLLM(fixture, review), packet, fixture['workspace'])
    assert plan['semantic_review']['accepted'] is False
    assert plan['semantic_review']['advisory'] is True and plan['semantic_review']['execution_allowed'] is True
    assert plan['semantic_review']['deferred_checks'] == [gate]
    assert plan['semantic_review']['checks'][gate]['passed'] is False


def test_only_dependency_uncertainty_may_be_verified_by_running(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    review = accepted_execution_review()
    review['accepted'] = False
    review['checks']['dependency_compatibility'].update(passed=False, severity='warning')
    plan = author.propose_author_execution_plan(AuthorLLM(fixture, review), packet, fixture['workspace'])
    assert plan['semantic_review']['deferred_checks'] == ['dependency_compatibility']


@pytest.mark.parametrize('body,receipt', [
    ('print("import only")\n', True),
    ('raise RuntimeError("author failed")\n', False),
    ('raise SystemExit(0)\n', True),
    ('raise SystemExit(1)\n', False),
    ('raise SystemExit("error")\n', False),
])
def test_exit_without_training_or_natural_return_cannot_pass(tmp_path, body, receipt):
    fixture, packet = fixture_packet(tmp_path)
    plan = author.validate_author_execution_plan(execution_proposal(fixture), packet, fixture['workspace'])
    path = fixture['workspace'] / 'train.py'
    path.write_text(body, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    author.materialize_author_execution(fixture['workspace'], plan, 'a' * 64, fixture['snapshot'])
    completed = subprocess.run([sys.executable, '_autorepro_capture/observe_author.py', 'train'],
        cwd=fixture['workspace'], env=native_environment(), capture_output=True, text=True, timeout=60)
    path = fixture['workspace'] / '_autorepro_capture/author_train.json'
    assert path.exists() is receipt
    execution = {'success': completed.returncode == 0, 'steps': [{'id': 'train', 'executed': True,
        'success': completed.returncode == 0, 'exit_code': completed.returncode}]}
    with pytest.raises((ValueError, FileNotFoundError)):
        author.verify_author_execution(plan, fixture['workspace'], execution, 'a' * 64)


@pytest.mark.parametrize('clean_exit,advisory', [(False, False), (True, False), (False, True)])
def test_full_service_observes_original_training_and_never_claims_paper_reproduction(
        tmp_path, monkeypatch, cpu_dependency_cache, clean_exit, advisory):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    if clean_exit:
        script = fixture['workspace'] / 'train.py'
        source = script.read_text('utf-8') + '\nraise SystemExit(0)\n'
        script.write_text(source, encoding='utf-8')
        fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
        next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    model = fixture['workspace'] / 'author_model.py'
    model_source = model.read_text('utf-8').replace(
        '        super().__init__()\n',
        "        super().__init__()\n        with open('logs/native/model.txt', 'w') as stream:\n"
        "            stream.write('original author output')\n")
    model.write_text(model_source, encoding='utf-8')
    fixture['snapshot']['files']['author_model.py'] = hashlib.sha256(model.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'model')['quote'] = model_source
    review = accepted_execution_review()
    if advisory:
        review['accepted'] = False
        review['checks']['dependency_compatibility']['passed'] = False
    runner.llm = AuthorLLM(fixture, review)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    validation = data['validation']
    assert data['repository_executed'] is True
    assert validation['status'] == 'author_code_completed' and validation['is_reproduced'] is None
    assert validation['author_execution'][0]['optimizer_steps'] == 40
    assert all(validation['completion_checks'].values())
    assert validation['paper_metrics_verified'] is False
    if advisory:
        assert data['grounded_repository_plan']['semantic_review']['accepted'] is False
        assert data['grounded_repository_plan']['semantic_review']['advisory'] is True
    assert '作者仓库代码运行完成' in data['report']
    assert (Path(data['run_dir']) / 'repo/logs/native/model.txt').read_text('utf-8') == 'original author output'
    assert 'experiment_protocol' not in data
    assert not (Path(data['run_dir']) / 'repo/_autorepro_capture/independent_metrics.json').exists()
    status = read_json(Path(data['run_dir']) / 'run_status.json')
    assert status['status'] == 'completed'


@pytest.mark.parametrize('module_import', [
    'from signals.author_model import SignalModel',
    'from signals import author_model\nSignalModel = author_model.SignalModel',
])
def test_repository_root_namespace_executes_unchanged_local_author_modules(tmp_path, module_import):
    fixture, packet = fixture_packet(tmp_path)
    path = fixture['workspace'] / 'train.py'
    source = path.read_text('utf8').replace('from author_model import SignalModel', module_import)
    path.write_text(source, encoding='utf8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    plan = author.validate_author_execution_plan(execution_proposal(fixture), packet, fixture['workspace'])
    assert plan['requirements']['repository_namespaces'][0]['name'] == 'signals'
    assert all(item['module'] != 'signals' for item in plan['requirements']['import_evidence'])
    author.materialize_author_execution(fixture['workspace'], plan, 'a' * 64, fixture['snapshot'])
    completed = subprocess.run([sys.executable, '_autorepro_capture/observe_author.py', 'train'],
        cwd=fixture['workspace'], env=native_environment(), capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    receipt = read_json(fixture['workspace'] / '_autorepro_capture/author_train.json')
    assert receipt['optimizer_steps'] == 40 and receipt['natural_return'] is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['snapshot']['files']['train.py']


def test_unrelated_or_missing_namespace_members_still_require_real_dependencies(tmp_path):
    from src.discovered_repository_plan import _author_imports
    fixture, packet = fixture_packet(tmp_path)
    packet['repository']['files']['train.py']['text'] = 'from signals.missing_module import Something\nfrom unrelated.author_model import SignalModel'
    discovery = _author_imports(packet, fixture['proposal']['steps'][0], include_sources=True)
    assert discovery['repository_namespaces'] == []
    assert {item['module'] for item in discovery['imports']} == {'signals', 'unrelated'}


def test_code_execution_without_full_data_provenance_stays_explicitly_unverified(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    fixture['proposal']['datasets'] = []
    runner.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result.get('error')
    validation = result['data']['validation']
    assert validation['author_execution'][0]['optimizer_steps'] == 40
    assert validation['dataset_provenance_verified'] is False and validation['paper_metrics_verified'] is False
    assert validation['is_reproduced'] is None
    assert result['data']['dataset_provenance']['verified'] is False
    assert 'real_data_frozen' not in validation['completion_checks']
    assert validation['warnings']


def test_original_python3_entrypoint_and_absl_tensorboard_dependencies(tmp_path):
    fixture, _ = fixture_packet(tmp_path)
    script = fixture['workspace'] / 'train.py'
    source = 'from absl import app\nfrom torch.utils.tensorboard import SummaryWriter\n' + script.read_text('utf-8')
    script.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    config = fixture['workspace'] / 'original.cfg'
    config.write_text('--epochs=40\n', encoding='utf-8')
    fixture['snapshot']['files']['original.cfg'] = hashlib.sha256(config.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    proposal = execution_proposal(fixture)
    proposal['steps'][0]['argv'][0] = 'python3'
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert plan['steps'][0]['argv'][0] == 'python3'
    assert 'original.cfg' in packet['repository']['files']
    assert 'absl-py==2.1.0' in plan['requirements']['compatibility']
    assert 'tensorboard==2.18.0' in plan['requirements']['compatibility']


def test_full_service_extracts_original_data_before_actual_training(tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    workspace = fixture['workspace']
    original = (workspace / 'dataset.json').read_bytes()
    with zipfile.ZipFile(workspace / 'dataset.zip', 'w') as archive:
        archive.writestr('dataset.json', original)
    (workspace / 'dataset.json').unlink()
    del fixture['snapshot']['files']['dataset.json']
    fixture['snapshot']['files']['dataset.zip'] = hashlib.sha256((workspace / 'dataset.zip').read_bytes()).hexdigest()
    readme = workspace / 'README.md'
    readme.write_text(readme.read_text('utf-8') + '\nUnzip dataset.zip in the repository root before training.\n', 'utf-8')
    fixture['snapshot']['files']['README.md'] = hashlib.sha256(readme.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'readme')['quote'] = readme.read_text('utf-8')
    fixture['proposal']['datasets'] = [{'kind': 'bundled', 'paths': ['dataset.zip'], 'extract_to': '.',
                                      'citations': ['readme', 'script']}]
    runner.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['validation']['status'] == 'author_code_completed'
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert data['dataset_provenance']['verified'] is True
    assert (Path(data['run_dir']) / 'repo/dataset.json').read_bytes() == original
    assert data['dataset_provenance']['sources'][0]['extracted_files']['dataset.json']['sha256'] == hashlib.sha256(original).hexdigest()
