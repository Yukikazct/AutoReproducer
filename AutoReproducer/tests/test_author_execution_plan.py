"""Execution-first mechanics use invented sources, not paper success evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

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
def test_known_negative_source_review_still_blocks_execution(tmp_path, gate):
    fixture, packet = fixture_packet(tmp_path)
    review = accepted_execution_review()
    review['checks'][gate]['passed'] = False
    with pytest.raises(PlanEvidenceError, match='review rejected'):
        author.propose_author_execution_plan(AuthorLLM(fixture, review), packet, fixture['workspace'])


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
    ('raise SystemExit(0)\n', False),
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


def test_full_service_observes_original_training_and_never_claims_paper_reproduction(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    model = fixture['workspace'] / 'author_model.py'
    model_source = model.read_text('utf-8').replace(
        '        super().__init__()\n',
        "        super().__init__()\n        with open('logs/native/model.txt', 'w') as stream:\n"
        "            stream.write('original author output')\n")
    model.write_text(model_source, encoding='utf-8')
    fixture['snapshot']['files']['author_model.py'] = hashlib.sha256(model.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'model')['quote'] = model_source
    runner.llm = AuthorLLM(fixture)
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
    assert '作者仓库代码运行完成' in data['report']
    assert (Path(data['run_dir']) / 'repo/logs/native/model.txt').read_text('utf-8') == 'original author output'
    assert 'experiment_protocol' not in data
    assert not (Path(data['run_dir']) / 'repo/_autorepro_capture/independent_metrics.json').exists()
    status = read_json(Path(data['run_dir']) / 'run_status.json')
    assert status['status'] == 'completed'
