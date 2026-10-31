"""Execution-first mechanics use invented sources, not paper success evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from discovered_fixtures import PlannedLLM, make_experiment, write_pdf
from src import author_execution_plan as author
from src import dependency_versions
from src.dependency_versions import compatibility_pins
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
        for name in ['dependency_compatibility']}}


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


@pytest.mark.parametrize('relative', ['config.json', '__pycache__/payload.pyc', '.pytest_cache/payload'])
def test_retained_observer_accepts_no_files_outside_verified_author_data(tmp_path, relative):
    fixture, packet = fixture_packet(tmp_path)
    plan = author.validate_author_execution_plan(execution_proposal(fixture), packet, fixture['workspace'])
    path = fixture['workspace'] / '_autorepro_capture' / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'not verified author data')
    with pytest.raises(ValueError, match='unverified file'):
        author.materialize_author_execution(fixture['workspace'], plan, 'f' * 64,
                                            fixture['snapshot'], retained_files={})


def test_teacher_student_plan_keeps_both_commands_and_student_dependencies(tmp_path):
    fixture, _ = fixture_packet(tmp_path)
    student = fixture['workspace'] / 'student.py'
    student.write_text('import yaml\n' + (fixture['workspace'] / 'train.py').read_text('utf-8'), encoding='utf-8')
    fixture['snapshot']['files']['student.py'] = hashlib.sha256(student.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    proposal = execution_proposal(fixture)
    second = deepcopy(proposal['steps'][0])
    second.update(id='student', depends_on=[proposal['steps'][0]['id']])
    second['argv'] = [arg if arg != 'train.py' else 'student.py' for arg in second['argv']]
    second.pop('entrypoint', None)
    proposal['steps'].append(second)
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert [step['argv'] for step in plan['steps']] == [step['argv'] for step in proposal['steps']]
    assert plan['steps'][1]['depends_on'] == [plan['steps'][0]['id']]
    assert any('yaml' in str(item).lower() for item in plan['requirements']['import_evidence'])
    materialized = author.materialize_author_execution(fixture['workspace'], plan, 'f' * 64, fixture['snapshot'])
    assert len(materialized['steps']) == 2
    assert materialized['steps'][1]['depends_on'] == [plan['steps'][0]['id']]


@pytest.mark.parametrize('manual', [False, True])
def test_execution_mode_fetches_repository_without_identity_review(tmp_path, monkeypatch, manual):
    from src.agents.repository_verifier import RepositoryVerifierAgent
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    runner.llm = AuthorLLM(fixture)
    request.update(author_execution_only=True, prepare_only=True)
    if manual:
        request['preferred_repo_url'] = fixture['snapshot']['url']
        # No repository link or author declaration in this PDF: a supplied
        # runnable repository must still reach original-command planning.
        method_quote = next(c['quote'] for c in fixture['proposal']['citations'] if c['id'] == 'paper_protocol')
        metrics_quote = next(c['quote'] for c in fixture['proposal']['citations'] if c['id'] == 'paper_results')
        write_pdf(fixture['pdf'], [['Invented manual repository paper', 'Abstract', method_quote],
                                  ['Full methods on the next page.'], metrics_quote.splitlines()])
    def reject(*args, **kwargs):
        raise AssertionError('Removed identity gate must never be called for direct execution')
    monkeypatch.setattr(RepositoryVerifierAgent, 'run', reject)
    events = []
    result = runner.run(request, on_event=events.append)
    assert result['state'] == 'COMPLETED', result['error']
    identity = result['data']['resources']['repository_identity']
    assert identity['status'] == 'unreviewed_execution' and identity['author_identity_confirmed'] is False
    stages = next(item for item in events if item['type'] == 'pipeline_plan')['stages']
    assert stages[0]['title'] == '获取代码仓库'
    assert not any(task['task'] == 'repository_identity_review' for _, task in runner.llm.calls)


def test_legacy_request_cannot_reenable_removed_identity_review(tmp_path, monkeypatch):
    from src.agents.repository_verifier import RepositoryVerifierAgent
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    runner.llm = AuthorLLM(fixture)
    request.update(author_execution_only=False, prepare_only=True)
    monkeypatch.setattr(RepositoryVerifierAgent, 'run', lambda *a, **k: pytest.fail('Reviewer was called'))
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    assert result['data']['grounded_repository_plan']['scope'] == author.SCOPE


def test_explicit_execution_objective_does_not_need_metric_or_checkpoint_claims(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    llm = AuthorLLM(fixture)
    plan = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
    assert plan['scope'] == author.SCOPE
    assert plan['metrics'] == [] and plan['capture'] == {'kind': 'author_process', 'device': 'cpu'}
    assert plan['steps'][0]['argv'] == fixture['proposal']['steps'][0]['argv']
    assert all(step['timeout_s'] is None for step in plan['steps'])
    assert plan['reviews_removed'] is True
    assert 'semantic_review' not in plan
    assert all(kwargs['task'] != 'author_execution_plan_review' for _, kwargs in llm.calls)


def test_unread_entrypoint_and_import_graph_are_loaded_without_model_reading_rounds(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    for name in ('train.py', 'author_model.py'):
        packet['repository']['files'].pop(name)
    packet['limits'].update(source_files=1, source_bytes=1)
    plan = author.propose_author_execution_plan(AuthorLLM(fixture), packet, fixture['workspace'])
    assert {'train.py', 'author_model.py'} <= set(packet['repository']['files'])
    assert plan['steps'][0]['argv'] == fixture['proposal']['steps'][0]['argv']
    assert packet['repository']['files']['train.py']['sha256'] == fixture['snapshot']['files']['train.py']


def test_execution_can_try_original_command_without_claims_or_exact_argument_quotes(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    proposal = execution_proposal(fixture)
    proposal.update(citations=[], claims={}, datasets=[])
    proposal['steps'][0].update(citations=[], argv=['python', 'train.py', '--epochs', '17'])
    proposal['requirements'] = {}
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert plan['steps'][0]['argv'][-2:] == ['--epochs', '17']
    assert 'torch==' + compatibility_pins()['torch'] in plan['requirements']['compatibility']
    assert plan['pre_execution_review'] == 'disabled'


def test_uncertain_dataset_provenance_defers_to_author_program_without_preparing_guessed_paths(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    proposal = execution_proposal(fixture)
    proposal['datasets'][0]['paths'] = ['*.json']
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert plan['datasets'] == []
    assert plan['reviews_removed'] is True


def test_runtime_dataset_observer_is_not_an_execution_prerequisite(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    proposal = execution_proposal(fixture)
    proposal['datasets'] = [{'kind': 'author_runtime', 'path': 'train.py', 'function': 'unknown_dynamic_loader'}]
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert plan['datasets'] == []
    assert plan['reviews_removed'] is True


@pytest.mark.parametrize('call', ['model.cuda()', 'torch.cuda.set_device(0)'])
@pytest.mark.parametrize('gpu_available,cpu_option,expected', [(True, False, 'cuda'), (False, False, 'cpu'), (True, True, 'cpu')])
@pytest.mark.parametrize('platform_name', ['linux', 'win32'])
def test_original_cuda_calls_select_matching_wheels_on_available_host(
        tmp_path, monkeypatch, gpu_available, cpu_option, expected, call, platform_name):
    monkeypatch.setattr(dependency_versions, 'sys', SimpleNamespace(platform=platform_name))
    fixture, _ = fixture_packet(tmp_path)
    script = fixture['workspace'] / 'train.py'
    script.write_text(script.read_text('utf-8') + '\n' + call + '\n', encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    packet['execution_host'] = {'gpu_probe_confirmed': gpu_available, 'cuda_device_indices': [0] if gpu_available else []}
    proposal = execution_proposal(fixture)
    proposal['citations'] = []
    if not cpu_option:
        proposal['steps'][0]['argv'].remove('--cpu')
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert plan['capture']['device'] == expected
    assert 'torch==2.5.1+' + ('cu121' if expected == 'cuda' else 'cpu') in plan['requirements']['compatibility']


@pytest.mark.parametrize('model_available', [False, True])
def test_literal_launcher_survives_missing_model_or_model_error_and_crlf_continuations(tmp_path, model_available):
    fixture, _ = fixture_packet(tmp_path)
    launcher = fixture['workspace'] / 'demo.sh'
    launcher.write_bytes(b'python train.py \\\r\n    --epochs 17\r\n')
    fixture['snapshot']['files']['demo.sh'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = True
    if model_available:
        class OfflineLLM:
            mock_mode = False
            def chat(self, *args, **kwargs):
                raise RuntimeError('Model temporarily unavailable')
        llm = OfflineLLM()
    else:
        llm = None
    # Keep only the launcher command so the README's separate example does
    # not mask the Windows continuation and network-failure regression.
    packet['repository']['files']['README.md']['text'] = 'Run demo.sh from the repository root.'
    plan = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--epochs', '17']
    assert plan['command_source']['path'] == 'demo.sh'
    assert plan['reviews_removed'] is True


def test_no_silent_downgrade_of_paper_reference_objective(tmp_path):
    fixture, packet = fixture_packet(tmp_path)
    del packet['author_execution_only']
    with pytest.raises(PlanEvidenceError, match='explicit execution objective'):
        author.validate_author_execution_plan(execution_proposal(fixture), packet, fixture['workspace'])


@pytest.mark.parametrize('variable', ['$epochs', '${epochs}'])
def test_subdirectory_launcher_expands_first_literal_loop_value_and_preserves_command(tmp_path, variable):
    fixture, _ = fixture_packet(tmp_path)
    workspace = fixture['workspace']
    (workspace / 'src').mkdir()
    (workspace / 'src/train.py').write_bytes((workspace / 'train.py').read_bytes())
    launcher = workspace / 'src/demo.sh'
    launcher.write_text('for epochs in 17 19; do\npython train.py --epochs ' + variable + '\ndone\n', encoding='utf-8')
    for name in ('src/train.py', 'src/demo.sh'):
        fixture['snapshot']['files'][name] = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    packet['author_execution_only'] = True
    packet['repository']['files']['README.md']['text'] = 'Run src/demo.sh.'
    plan = author.plan_from_author_commands(packet, workspace)
    assert plan['steps'][0]['cwd'] == 'src'
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--epochs', '17']
    assert plan['command_source']['template_bindings'] == {'epochs': '17'}
    assert plan['command_source']['quote'].endswith(variable)


def prose_command_fixture(tmp_path, instruction, *, directories=('src',), root_script=False,
                         command='python train.py --epochs 40 --seed 7 --cpu'):
    fixture = make_experiment(tmp_path)
    workspace = fixture['workspace']
    source = (workspace / 'train.py').read_bytes()
    if not root_script:
        (workspace / 'train.py').unlink()
        fixture['snapshot']['files'].pop('train.py')
    for directory in directories:
        path = workspace / directory / 'train.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        fixture['snapshot']['files'][path.relative_to(workspace).as_posix()] = hashlib.sha256(source).hexdigest()
    readme = workspace / 'README.md'
    readme.write_text(instruction + '\n```text\n' + command + '\n```\n', encoding='utf-8')
    fixture['snapshot']['files']['README.md'] = hashlib.sha256(readme.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    packet['author_execution_only'] = True
    return fixture, packet


@pytest.mark.parametrize('instruction', [
    'To run each experiment, navigate into `src`. Then, run the following command:',
    'Change directory to `src` before running this experiment.',
    'From the repository root, navigate to `src`. Then run:',
])
@pytest.mark.parametrize('command', ['python train.py --epochs 40 --seed 7 --cpu',
                                    'python3 -u train.py --epochs 40 --seed 7 --cpu'])
def test_readme_prose_directory_preserves_the_original_command(tmp_path, instruction, command):
    fixture, packet = prose_command_fixture(tmp_path, instruction, command=command)
    plan = author.plan_from_author_commands(packet, fixture['workspace'])
    step = plan['steps'][0]
    assert step['cwd'] == 'src'
    assert step['argv'] == command.split()
    assert step['entrypoint']['source_path'] == 'src/train.py'
    assert plan['command_source']['quote'] == command
    assert plan['command_source']['cwd_quote'] == instruction
    assert step['timeout_s'] is None


def test_readme_prose_directory_does_not_select_a_root_level_decoy(tmp_path):
    fixture, packet = prose_command_fixture(tmp_path, 'Navigate into `src`. Then run:', root_script=True)
    plan = author.plan_from_author_commands(packet, fixture['workspace'])
    assert plan['steps'][0]['cwd'] == 'src'
    assert plan['steps'][0]['entrypoint']['source_path'] == 'src/train.py'


@pytest.mark.parametrize('directory', ['./src', 'src/nested'])
def test_readme_prose_directory_retains_a_safe_relative_path(tmp_path, directory):
    fixture, packet = prose_command_fixture(tmp_path, f'Navigate into `{directory}`. Then run:',
                                           directories=(directory,))
    plan = author.plan_from_author_commands(packet, fixture['workspace'])
    assert plan['steps'][0]['cwd'] == Path(directory).as_posix()
    assert plan['steps'][0]['entrypoint']['source_path'] == (Path(directory) / 'train.py').as_posix()


@pytest.mark.parametrize('kind', ['directory', 'script'])
def test_readme_prose_directory_rejects_links_without_reading_external_sources(tmp_path, require_symlinks, kind):
    fixture, packet = prose_command_fixture(tmp_path, 'Navigate into `src`. Then run:', root_script=True)
    original = fixture['workspace'] / 'src'
    if kind == 'script':
        original /= 'train.py'
    target = tmp_path / ('outside_' + kind)
    original.rename(target)
    original.symlink_to(target, target_is_directory=kind == 'directory')
    source = target / 'train.py' if kind == 'directory' else target
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(PlanEvidenceError, match='symlink or junction'):
        author.plan_from_author_commands(packet, fixture['workspace'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('directory', ['missing', '../outside', '/outside', 'C:/outside', '.git',
                                      'src/../../outside', 'src:stream'])
def test_readme_prose_directory_never_falls_back_across_an_invalid_boundary(tmp_path, directory):
    fixture, packet = prose_command_fixture(tmp_path, f'Navigate into `{directory}`. Then run:', root_script=True)
    with pytest.raises(PlanEvidenceError):
        author.plan_from_author_commands(packet, fixture['workspace'])


@pytest.mark.parametrize('directories', [('src',), ('src', 'other')])
def test_readme_missing_directory_evidence_does_not_guess_among_pinned_scripts(tmp_path, directories):
    fixture, packet = prose_command_fixture(tmp_path, 'Run the complete original experiment:', directories=directories)
    with pytest.raises(PlanEvidenceError, match='entrypoint'):
        author.plan_from_author_commands(packet, fixture['workspace'])


def test_readme_prose_directory_requires_a_pinned_script(tmp_path):
    fixture, packet = prose_command_fixture(tmp_path, 'Navigate into `src`. Then run:')
    packet['file_inventory'].pop('src/train.py')
    with pytest.raises(PlanEvidenceError, match='pinned repository inventory'):
        author.plan_from_author_commands(packet, fixture['workspace'])


@pytest.mark.parametrize('declared_in', [None, 'entrypoint', 'unrelated'])
def test_readme_kwargs_placeholder_requires_an_original_argument_declaration(tmp_path, declared_in):
    command = 'python train.py --epochs 40 --seed 7 --cpu'
    fixture, _ = prose_command_fixture(tmp_path, 'Navigate into `src`. Then run:', command=command)
    workspace = fixture['workspace']
    readme = workspace / 'README.md'
    readme.write_text('Navigate into `src`.\n```text\npython train.py --kwargs\n```\n'
                      'where `kwargs` are specified in the training script.\n'
                      f'The complete example is:\n```text\n{command}\n```\n', encoding='utf-8')
    if declared_in == 'entrypoint':
        script = workspace / 'src/train.py'
        script.write_text(script.read_text('utf-8').replace(
            'args = parser.parse_args()', "parser.add_argument('--kwargs', action='store_true')\nargs = parser.parse_args()"),
            encoding='utf-8')
        fixture['snapshot']['files']['src/train.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    elif declared_in == 'unrelated':
        unrelated = workspace / 'unrelated.py'
        unrelated.write_text('import argparse\nparser = argparse.ArgumentParser()\n'
                             "parser.add_argument('--kwargs', action='store_true')\n", encoding='utf-8')
        fixture['snapshot']['files']['unrelated.py'] = hashlib.sha256(unrelated.read_bytes()).hexdigest()
    fixture['snapshot']['files']['README.md'] = hashlib.sha256(readme.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    packet['author_execution_only'] = True
    plan = author.plan_from_author_commands(packet, workspace)
    assert plan['steps'][0]['argv'] == (['python', 'train.py', '--kwargs']
                                      if declared_in == 'entrypoint' else command.split())
    assert plan['steps'][0]['cwd'] == 'src'


def test_literal_cd_keeps_its_author_directory_when_the_script_is_missing(tmp_path):
    fixture, packet = prose_command_fixture(tmp_path, 'cd src', root_script=True)
    (fixture['workspace'] / 'src/train.py').unlink()
    with pytest.raises(PlanEvidenceError):
        author.plan_from_author_commands(packet, fixture['workspace'])


@pytest.mark.parametrize('directory,allowed', [('../logs', True), ('../../outside', False)])
def test_subdirectory_author_logs_are_resolved_against_repository_boundary(tmp_path, directory, allowed):
    fixture, _ = fixture_packet(tmp_path)
    workspace = fixture['workspace']
    (workspace / 'src').mkdir()
    source = (workspace / 'src/train.py')
    source.write_text("with open('" + directory + "/result.txt', 'w') as stream:\n    stream.write('result')\n"
                      + (workspace / 'train.py').read_text('utf-8'), encoding='utf-8')
    fixture['snapshot']['files']['src/train.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], workspace, fixture['snapshot'])
    packet['author_execution_only'] = True
    proposal = execution_proposal(fixture)
    proposal['citations'] = []
    proposal['steps'][0].update(cwd='src', citations=[])
    if not allowed:
        with pytest.raises(ValueError, match='escaped the workspace'):
            author.validate_author_execution_plan(proposal, packet, workspace)
        return
    plan = author.validate_author_execution_plan(proposal, packet, workspace)
    assert plan['steps'][0]['output_directories'][0]['path'] == 'logs'
    assert not (workspace / 'logs').exists()


@pytest.mark.parametrize('change', [
    lambda p: p.update(code='print(1)'),
    lambda p: p['repository'].update(revision='b' * 40),
    lambda p: p['steps'][0].update(argv=['python', '-c', 'print(1)']),
    lambda p: p['steps'][0]['argv'].append('--smoke'),
    lambda p: p['steps'][0].update(required=False),
])
def test_execution_objective_keeps_original_source_data_and_command_guards(tmp_path, change):
    fixture, packet = fixture_packet(tmp_path)
    proposal = deepcopy(execution_proposal(fixture))
    change(proposal)
    with pytest.raises(PlanEvidenceError):
        author.validate_author_execution_plan(proposal, packet, fixture['workspace'])


def test_author_review_function_was_removed():
    assert not hasattr(author, 'review_author_execution_plan')


@pytest.mark.parametrize('body,receipt', [
    ('print("import only")\n', True),
    ('raise RuntimeError("author failed")\n', False),
    ('raise SystemExit(0)\n', True),
    ('raise SystemExit(1)\n', False),
    ('raise SystemExit("error")\n', False),
])
def test_process_completion_keeps_unobserved_training_and_failed_exits_distinct(tmp_path, body, receipt):
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
    if completed.returncode == 0:
        validation = author.verify_author_execution(plan, fixture['workspace'], execution, 'a' * 64)
        assert validation['status'] == 'author_code_completed'
        assert validation['training_observed'] is False
        assert validation['completion_checks']['observed_training_updates'] is False
        assert validation['is_reproduced'] is None
    else:
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
    assert validation['completion_checks']['all_required_author_commands_returned'] is True
    assert validation['completion_checks']['owned_processes_finished'] is True
    assert validation.get('native_environment_verified') is not True
    assert validation['pre_execution_review'] == 'disabled'
    assert validation['paper_metrics_verified'] is False
    assert 'semantic_review' not in data['grounded_repository_plan']
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
    assert validation['reviews_removed'] is True


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
