"""Model failure analysis is bounded and cannot rewrite author constraints."""
import json
import pytest

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
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


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


@pytest.mark.parametrize('invalid', [
    '{"action":', [], {'action': []}, {'action': 'blocked', 'reason': {}},
    {'request_files': ['../../outside.py']}, {'request_files': ['README.md', 'README.md']},
    {'request_files': ['README.md'], 'command': 'untrusted command'},
])
@pytest.mark.parametrize('action', ['blocked', 'replace_compatibility_pin'])
def test_invalid_response_feedback_can_recover_within_same_failure(tmp_path, invalid, action):
    fixture, packet, plan, environment = prepared(tmp_path)
    result = {'final': {'stderr': 'No matching distribution found for numpy==1.24.0 (from versions: 1.24.4)'}}
    final = {'action': action, 'reason': 'resolver evidence checked'}
    if action == 'replace_compatibility_pin':
        final['requirement'] = 'numpy==1.24.4'
    llm = FakeLLM([invalid, final])
    data = {}
    repair = analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                    environment, tmp_path, data)
    assert (repair is not None) is (action == 'replace_compatibility_pin')
    assert len(llm.calls) == 2
    feedback = llm.calls[1][0]['analysis_feedback']
    assert len(feedback) == 1 and feedback[0]['call'] == 1
    assert 0 < len(feedback[0]['error']) <= 500
    record = data['failure_analyses'][0]
    assert record['source_reads'] == []
    assert 'analysis_error' not in record
    assert record['action'] == action
    assert 'response_errors' in json.loads((tmp_path / 'failure_analyses.json').read_text(encoding='utf-8'))[0]


def test_response_correction_and_source_read_share_three_call_budget(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    result = {'final': {'stderr': 'No matching distribution found for numpy==1.24.0 (from versions: 1.24.4)'}}
    llm = FakeLLM(['not json', {'request_files': ['README.md']},
                   {'action': 'replace_compatibility_pin', 'requirement': 'numpy==1.24.4'}])
    data = {}
    repair = analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                    environment, tmp_path, data)
    assert repair['requirement'] == 'numpy==1.24.4'
    assert len(llm.calls) == 3
    assert 'README.md' in llm.calls[2][0]['source_excerpts']
    assert data['failure_analyses'][0]['source_reads'] == ['README.md']
    assert len(data['failure_analyses'][0]['response_errors']) == 1


def test_invalid_responses_exhaust_shared_budget_and_do_not_bypass_fingerprint(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    result = {'final': {'stderr': 'RuntimeError: failure'}}
    llm = FakeLLM(['not json'] * 4)
    data = {}
    for _ in range(2):
        assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                      environment, tmp_path, data) is None
    assert len(llm.calls) == 3
    assert len(llm.calls[2][0]['analysis_feedback']) == 2
    assert len(data['failure_analyses']) == 1
    assert len(data['failure_analyses'][0]['response_errors']) == 3
    assert data['failure_analyses'][0]['analysis_error']
    assert data['failure_analyses'][0]['action'] == 'blocked'


@pytest.mark.parametrize('api_error', ['[LLM API Error: HTTP 401 api_key=',
                                     '[LLM API Error: HTTP 429 token=',
                                     '[LLM API Error: connection failed authorization='])
def test_api_failure_is_redacted_and_not_retried(tmp_path, api_error):
    fixture, packet, plan, environment = prepared(tmp_path)
    result = {'final': {'stderr': 'RuntimeError: failure'}}
    secret = 'sk-' + 'fakevalue' * 4
    llm = FakeLLM([api_error + secret + ' ' + 'error details ' * 100 + ']', {'action': 'blocked', 'reason': 'unused'}])
    data = {}
    assert analyze_author_failure(llm, result, plan, packet, fixture['workspace'],
                                  environment, tmp_path, data) is None
    assert len(llm.calls) == 1
    assert not data['failure_analyses'][0]['response_errors']
    assert len(data['failure_analyses'][0]['analysis_error']) <= 500
    assert secret not in json.dumps(data)
    assert secret not in (tmp_path / 'failure_analyses.json').read_text(encoding='utf-8')


def test_transport_exception_is_not_retried(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    llm = FakeLLM([ConnectionError('server unavailable'), {'action': 'blocked', 'reason': 'unused'}])
    data = {}
    assert analyze_author_failure(llm, {'final': {'stderr': 'failure'}}, plan, packet,
                                  fixture['workspace'], environment, tmp_path, data) is None
    assert len(llm.calls) == 1
    assert data['failure_analyses'][0]['analysis_error'] == 'server unavailable'


def test_api_error_json_credentials_are_redacted_before_recording(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    secret = 'invented-test-credential'
    llm = FakeLLM(['[LLM API Error: HTTP 401 ' + json.dumps({'api_key': secret}) + ']'])
    data = {}
    analyze_author_failure(llm, {'final': {'stderr': 'failure'}}, plan, packet,
                           fixture['workspace'], environment, tmp_path, data)
    assert secret not in json.dumps(data)
    assert '[REDACTED]' in data['failure_analyses'][0]['analysis_error']


def test_missing_model_configuration_does_not_request_analysis(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    llm = FakeLLM([])
    llm.base_url, llm.model = '', 'configured-model'
    data = {}
    assert analyze_author_failure(llm, {'final': {'stderr': 'failure'}}, plan, packet,
                                  fixture['workspace'], environment, tmp_path, data) is None
    assert not llm.calls
    assert not data


def test_valid_blocked_analysis_does_not_consume_correction_budget(tmp_path):
    fixture, packet, plan, environment = prepared(tmp_path)
    llm = FakeLLM([{'action': 'blocked', 'reason': 'original hardware unavailable'}, 'not used'])
    data = {}
    assert analyze_author_failure(llm, {'final': {'stderr': 'CUDA unavailable'}}, plan, packet,
                                  fixture['workspace'], environment, tmp_path, data) is None
    assert len(llm.calls) == 1
    assert data['failure_analyses'][0]['reason'] == 'original hardware unavailable'
