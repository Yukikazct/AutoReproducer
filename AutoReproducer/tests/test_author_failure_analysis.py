"""Model failure analysis is bounded and cannot rewrite author constraints."""
import json

from discovered_fixtures import make_experiment
from src.author_failure_analysis import analyze_author_failure
from src.author_runtime_dependencies import run_with_dependency_repair
from src.discovered_repository_plan import build_evidence_packet


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append((json.loads(prompt), kwargs))
        return json.dumps(next(self.responses))


class TwoRunExecutor:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def run(self, workspace, steps, environment, on_event):
        self.calls.append((steps, dict(environment)))
        return {'success': len(self.calls) == 2, 'run_dir': f'/run/{len(self.calls)}',
                'final': {'stderr': self.error if len(self.calls) == 1 else ''},
                'effective_env_config': dict(environment)}


def prepared(tmp_path, *, author=()):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    plan = {'steps': [{'id': 'train', 'entrypoint': {'kind': 'script', 'source_path': 'train.py'}}],
            'capture': {'device': 'cpu'},
            'requirements': {'author': list(author), 'compatibility': ['numpy==1.24.0'],
                             'repository_namespaces': []}}
    environment = {'requirements_txt': 'numpy==1.24.0\n',
                   'python_version': '3.11', 'python_executable': '/isolated/python',
                   'dependency_failure_kind': 'wheel_unavailable'}
    return fixture, packet, plan, environment


def test_model_pin_repair_uses_log_candidate_then_retries_same_author_step(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    error = 'No matching distribution found for numpy==1.24.0 (from versions: 1.24.4)'
    llm = FakeLLM([{'failure_kind': 'wheel_unavailable', 'reason': 'resolver lists 1.24.4',
                    'action': 'replace_compatibility_pin', 'requirement': 'numpy==1.24.4'}])
    data, runner = {}, TwoRunExecutor(error)
    def analyze(result, effective):
        return analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                      effective, tmp_path, data)
    steps = [{'id': 'train', 'argv': ['python', 'train.py']}]
    outcome = run_with_dependency_repair(runner, fixture['workspace'], steps, environment,
        plan, packet, tmp_path, data, lambda _: None, lambda: None, analyze_failure=analyze)
    assert outcome['success'] and len(runner.calls) == 2
    assert runner.calls[0][0] == runner.calls[1][0] == steps
    assert runner.calls[1][1]['requirements_txt'] == 'numpy==1.24.4\n'
    assert data['failure_analyses'][0]['action'] == 'replace_compatibility_pin'
    assert data['model_dependency_repairs'][0]['attempt'] == 1
    assert llm.calls[0][1]['task'] == 'author_failure_analysis'


def test_model_cannot_change_author_pin_or_invent_unlisted_version(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path, author=['numpy==1.24.0'])
    result = {'final': {'stderr': 'No matching distribution found for numpy==1.24.0 (from versions: 1.24.4)'},
              'effective_env_config': environment}
    llm = FakeLLM([{'action': 'replace_compatibility_pin', 'requirement': 'numpy==1.24.4'},
                   {'action': 'replace_compatibility_pin', 'requirement': 'numpy==9.9.9'}])
    data = {}
    assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert len(llm.calls) == 1  # identical failure without new evidence
    plan['requirements']['author'] = []
    changed = {'final': {'stderr': 'No matching distribution found for numpy==1.24.0 (from versions: 1.24.5)'},
               'effective_env_config': environment}
    assert analyze_author_failure(llm, changed, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert len(llm.calls) == 2


def test_model_can_request_two_bounded_source_reads(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    llm = FakeLLM([{'request_files': ['README.md']}, {'action': 'blocked',
        'reason': 'GPU is required by the original entrypoint'}])
    result = {'final': {'stderr': 'RuntimeError: CUDA unavailable'},
              'effective_env_config': environment}
    data = {}
    assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert len(llm.calls) == 2
    assert data['failure_analyses'][0]['source_reads'] == ['README.md']
    assert 'README.md' in llm.calls[1][0]['source_excerpts']


def test_other_packages_candidate_version_is_not_evidence(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    result = {'final': {'stderr': 'No matching distribution found for numpy==1.24.0\n'
                                  'No matching distribution found for pandas==2.0.0 (from versions: 1.24.4)'},
              'effective_env_config': environment}
    llm = FakeLLM([{'action': 'replace_compatibility_pin', 'requirement': 'numpy==1.24.4'}])
    data = {}
    assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert data['failure_analyses'][0]['action'] == 'blocked'


def test_failure_analysis_stops_after_four_distinct_failures_per_repository(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    plan['repository'] = {'url': 'https://github.com/example/method'}
    llm = FakeLLM([{'action': 'blocked', 'reason': 'needs external condition'}] * 5)
    data = {}
    for index in range(5):
        result = {'final': {'stderr': f'RuntimeError: distinct failure {index}'},
                  'effective_env_config': environment}
        assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                      environment, tmp_path, data) is None
    assert len(llm.calls) == 4
    assert len(data['failure_analyses']) == 4


def test_model_reads_actual_traceback_location_and_previous_repair_outcomes(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    source = fixture['workspace'] / 'nested.py'
    content = '# long original source\n' * 350 + "raise RuntimeError('specific failure here')\n"
    source.write_text(content, encoding='utf-8')
    import hashlib
    packet['file_inventory']['nested.py'] = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                            'text_available': True}
    data = {'self_repair': {'attempts': [{'action': 'install_missing_dependency',
             'status': 'failed_with_new_error', 'reason': 'original module now imports'}]}}
    result = {'final': {'stdout': f'Traceback (most recent call last):\n  File "{source}", line 351, in <module>\n'
                                  "RuntimeError: specific failure here", 'stderr': ''}}
    llm = FakeLLM([{'action': 'blocked', 'reason': 'needs a compatible API'}])
    analyze_author_failure(llm, result, plan, packet, fixture['workspace'], environment, tmp_path, data)
    context = llm.calls[0][0]
    assert 'specific failure here' in context['diagnostic']
    assert 'specific failure here' in context['source_excerpts']['nested.py']
    assert '351' in context['source_excerpts']['nested.py']
    assert context['previous_repairs'][0]['status'] == 'failed_with_new_error'
