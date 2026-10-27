"""Strict eligibility supplements execution; incomplete evidence is not a veto."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.discovered_repository_plan import PlanEvidenceError
from src.method_adapters import digest, read_json
from test_discovered_repository_service import (
    cpu_dependency_cache, setup_service, use_preinstalled_cache,
)


def test_strict_prepare_builds_independent_plan_without_running(tmp_path, monkeypatch):
    _, runner, request, llm = setup_service(tmp_path, monkeypatch)
    runner.runner.run = Mock(side_effect=AssertionError('prepare-only must not execute'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['strict_validation']['status'] == 'eligible'
    assert data['reproduction_scope'] == 'selected_paper_experiment'
    assert data['validation']['status'] == 'prepared'
    assert data['validation']['is_reproduced'] is None
    assert not data['repository_executed']
    steps = data['execution_plan']['steps']
    assert [step['id'] for step in steps] == ['train', 'independent_evaluation']
    assert steps[0]['timeout_s'] is None
    assert steps[1]['timeout_s'] == 300
    assert {options['task'] for _, options in llm.calls} == {
        'experiment_protocol', 'discovered_repository_plan', 'discovered_repository_plan_review'}
    runner.runner.run.assert_not_called()


@pytest.mark.parametrize('stage', ['protocol', 'capture'])
def test_missing_strict_support_preserves_author_execution_plan(tmp_path, monkeypatch, stage):
    from src.agents.experiment_protocol import ExperimentProtocolAgent
    from src import discovered_repository_plan
    _, runner, request, llm = setup_service(tmp_path, monkeypatch)
    target, name = ((ExperimentProtocolAgent, 'run') if stage == 'protocol'
                    else (discovered_repository_plan, 'propose_plan'))
    monkeypatch.setattr(target, name, Mock(side_effect=PlanEvidenceError(
        'Original evidence is insufficient for independent capture', category='missing_information')))
    runner.runner.run = Mock(side_effect=AssertionError('prepare-only must not execute'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['reproduction_scope'] == 'author_code_execution'
    assert data['strict_validation']['status'] == 'unavailable'
    assert data['strict_validation']['failure_kind'] == 'missing_information'
    assert data['validation']['is_reproduced'] is None
    assert [step['id'] for step in data['execution_plan']['steps']] == ['train']
    assert 'author_execution_plan' in {options['task'] for _, options in llm.calls}
    assert not read_json(Path(data['run_dir']) / 'source_packet.json').get('experiment_protocol')
    runner.runner.run.assert_not_called()


def test_strict_rejection_cannot_hide_changed_source(tmp_path, monkeypatch):
    from src.agents.experiment_protocol import ExperimentProtocolAgent
    _, runner, request, llm = setup_service(tmp_path, monkeypatch)

    def corrupt_source(_self, _packet, workspace):
        (workspace / 'train.py').write_text('raise SystemExit(0)', encoding='utf-8')
        raise PlanEvidenceError('Capture is unsupported', category='missing_information')

    monkeypatch.setattr(ExperimentProtocolAgent, 'run', corrupt_source)
    runner.runner.run = Mock(side_effect=AssertionError('changed sources must not execute'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'ERROR'
    assert not result['data']['repository_executed']
    assert result['data']['validation']['is_reproduced'] is None
    assert not any(options['task'] == 'author_execution_plan' for _, options in llm.calls)
    assert not (Path(result['data']['run_dir']) / 'strict_rejection_packet.json').exists()
    runner.runner.run.assert_not_called()


def test_strict_source_budget_is_restored_before_ordinary_author_planning(tmp_path, monkeypatch):
    import hashlib
    from src import discovered_repository_plan as planner
    from src.agents.experiment_protocol import ExperimentProtocolAgent
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch)
    strict_name, author_name = 'strict_only.txt', 'author_needed.txt'
    sources = {
        strict_name: 'Extra strict protocol navigation that cannot establish full model capture.\n',
        author_name: 'Original author execution navigation: python train.py --epochs 40 --seed 7 --cpu\n',
    }
    for name, text in sources.items():
        raw = text.encode('utf-8')
        (fixture['workspace'] / name).write_bytes(raw)
        fixture['snapshot']['files'][name] = hashlib.sha256(raw).hexdigest()
    build = planner.build_evidence_packet

    def bounded_initial_packet(*args, **kwargs):
        packet = build(*args, **kwargs)
        for name in sources:
            packet['repository']['files'].pop(name, None)
        packet['limits']['source_files'] = len(packet['repository']['files']) + 1
        packet['limits']['source_bytes'] = (sum(item['bytes'] for item in packet['repository']['files'].values())
                                           + max(len(text.encode('utf-8')) for text in sources.values()))
        return packet

    def consume_strict_budget(_self, packet, workspace):
        planner._add_requested_files([strict_name], packet, workspace)
        # Prove that reusing this source view would reject the ordinary read.
        with pytest.raises(PlanEvidenceError, match='bounded source budget'):
            planner._add_requested_files([author_name], packet, workspace)
        raise PlanEvidenceError('Independent capture remains unsupported after actual source reading',
                                category='missing_information')

    monkeypatch.setattr(planner, 'build_evidence_packet', bounded_initial_packet)
    monkeypatch.setattr(ExperimentProtocolAgent, 'run', consume_strict_budget)
    llm.prefixes = [{'request_files': [author_name]}]
    llm.proposal['citations'].append({'id': 'execution_navigation', 'source': 'repository',
                                    'path': author_name, 'quote': sources[author_name]})
    runner.runner.run = Mock(side_effect=AssertionError('Prepare-only cannot execute'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['strict_validation']['status'] == 'unavailable'
    assert data['validation']['status'] == 'prepared'
    assert data['validation']['is_reproduced'] is None
    run_dir = Path(data['run_dir'])
    rejected = read_json(run_dir / 'strict_rejection_packet.json')
    ordinary = read_json(run_dir / 'source_packet.json')
    assert strict_name in rejected['repository']['files']
    assert author_name not in rejected['repository']['files']
    assert author_name in ordinary['repository']['files']
    assert strict_name not in ordinary['repository']['files']
    assert ordinary['author_execution_only'] is True
    assert 'experiment_protocol' not in ordinary
    assert ordinary['limits']['source_files'] == rejected['limits']['source_files']
    assert ordinary['limits']['source_bytes'] == rejected['limits']['source_bytes']
    assert data['strict_validation']['rejection_packet'] == 'strict_rejection_packet.json'
    assert any(citation.get('path') == author_name for citation in data['grounded_repository_plan']['citations'])
    assert len(llm.calls) == 2 and all(options['task'] == 'author_execution_plan' for _, options in llm.calls)
    runner.runner.run.assert_not_called()


def test_discovered_repository_without_literal_pdf_link_can_attempt_strict_mode(tmp_path, monkeypatch):
    from discovered_fixtures import write_pdf
    from src.pdf_input import extract_pdf_input
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    document = extract_pdf_input(fixture['pdf'])
    url = fixture['snapshot']['url']
    write_pdf(fixture['pdf'], [[line for line in page.splitlines() if url not in line]
                               for page in document.page_texts])
    runner.runner.run = Mock(side_effect=AssertionError('prepare-only must not execute'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'COMPLETED', result['error']
    assert result['data']['strict_validation']['status'] == 'eligible'
    assert result['data']['resources']['repository_identity']['author_identity_confirmed'] is False
    runner.runner.run.assert_not_called()


def test_strict_runtime_rejection_is_terminal_without_replanning_another_repository(tmp_path, monkeypatch):
    from src import discovered_repository_reproduction as service
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    linked = Mock(side_effect=AssertionError('cannot replace repository after execution'))
    monkeypatch.setattr(service, 'linked_training_repositories', linked)

    def runtime_failure(_request, data, *_args):
        data['repository_executed'] = True  # Simulate the execution boundary only.
        failure = PlanEvidenceError('Independent predictions did not match')
        failure.reproduction_phase = 'finish_execution'
        raise failure

    monkeypatch.setattr(runner, '_execute_frozen', runtime_failure)
    result = runner.run({**request, 'strict_validation': True})
    assert result['state'] == 'ERROR'
    assert result['data']['validation']['failure_phase'] == 'finish_execution'
    assert result['data']['validation']['is_reproduced'] is None
    assert len(result['data']['repository_identity_attempts']) == 1
    linked.assert_not_called()


@pytest.mark.parametrize('family', ['signals', 'pairs'])
def test_actual_strict_service_reloads_weights_and_reports(tmp_path, monkeypatch, cpu_dependency_cache, family):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch, family)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run({**request, 'strict_validation': True})
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    validation = data['validation']
    assert data['strict_validation']['status'] == 'verified'
    assert validation['paper_metrics_verified'] is True
    assert validation['protocol_pass'] is True
    assert validation['independent_metrics_pass'] is True
    assert validation['is_reproduced'] is True
    assert validation['scope'] == 'selected_paper_experiment'
    assert validation['capture']['process_pid'] != validation['independent_metrics']['process_pid']
    assert validation['independent_metrics']['capture_process_pid'] == validation['capture']['process_pid']
    assert data['native_preflight']['native_probe_pass'] is True
    assert [step['id'] for step in data['execution']['steps']] == ['train', 'independent_evaluation']
    assert all(step['executed'] and step['exit_code'] == 0 for step in data['execution']['steps'])
    run_dir = Path(data['run_dir'])
    for name, sha in fixture['snapshot']['files'].items():
        assert digest(run_dir / 'repo' / name) == sha
    assert read_json(run_dir / 'strict_execution_verification.json') == validation
    assert read_json(run_dir / 'result.json')['data']['validation'] == validation
    assert '100.0' in (run_dir / 'report.md').read_text(encoding='utf-8')


def test_actual_unsupported_strict_request_still_runs_author_command(tmp_path, monkeypatch, cpu_dependency_cache):
    from src.agents.experiment_protocol import ExperimentProtocolAgent
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    monkeypatch.setattr(ExperimentProtocolAgent, 'run', Mock(side_effect=PlanEvidenceError(
        'No independent capture for this framework', category='missing_information')))
    result = runner.run({**request, 'strict_validation': True})
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['strict_validation']['status'] == 'unavailable'
    assert data['repository_executed']
    assert data['validation']['status'] == 'author_code_completed'
    assert data['validation']['is_reproduced'] is None
    assert data['validation']['paper_metrics_verified'] is False
    assert any('严格验收未完成' in warning for warning in data['validation']['warnings'])
    view = data['report_view_model']
    assert view['answers']['completion'] == '已完成运行'
    assert view['conclusion']['independent']['status'] == 'unavailable'
    assert '独立核验尚未完成' in data['report']
    assert '作者仓库代码运行完成' in data['report']
    assert 'No independent capture for this framework' in data['report']


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('target', ['experiment_spec.json', 'source_packet.json'])
def test_initial_frozen_record_tampering_is_terminal_before_environment_selection(
        tmp_path, monkeypatch, strict, target):
    from src import discovered_repository_reproduction as service
    import src.author_python as author_python
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    linked = Mock(return_value=[])
    selected = Mock(side_effect=AssertionError('Changed inputs cannot select an environment'))
    monkeypatch.setattr(service, 'linked_training_repositories', linked)
    monkeypatch.setattr(author_python, 'select_author_python', selected)
    runner.runner.run = Mock(side_effect=AssertionError('Changed inputs cannot execute'))

    def corrupt(event):
        if (event.get('phase_id') == 'prepare_environment' and event.get('status') == 'running'):
            run_dir, = (tmp_path / 'service' / 'runs').iterdir()
            (run_dir / target).write_text('{"changed":true}', encoding='utf-8')

    result = runner.run({**request, 'strict_validation': strict, 'prepare_only': True}, on_event=corrupt)
    assert result['state'] == 'ERROR'
    assert result['data']['validation']['failure_kind'] == 'source_changed'
    assert result['data']['validation']['is_reproduced'] is None
    assert len(result['data']['repository_identity_attempts']) == 1
    linked.assert_not_called()
    selected.assert_not_called()
    runner.runner.run.assert_not_called()


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('target', ['pdf', 'experiment_spec.json', 'execution_plan.json',
    'source_packet.json', 'source', 'bundled_data', 'generated_config', 'downloaded_data'])
def test_actual_frozen_input_checks_prevent_tampering_from_becoming_repository_fallback(
        tmp_path, monkeypatch, strict, target):
    import sys
    from src import discovered_repository_reproduction as service
    import src.author_python as author_python
    from src.discovered_repository_runtime import CAPTURE_DIR
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    linked = Mock(return_value=[])
    monkeypatch.setattr(service, 'linked_training_repositories', linked)
    monkeypatch.setattr(author_python, 'select_author_python',
                        Mock(return_value={'python_executable': sys.executable}))
    runner.runner.run = Mock(side_effect=AssertionError('Prepare-only and changed inputs cannot execute'))
    write = service.write_json
    prepare = service.prepare_plan_datasets

    def prepare_with_frozen_download(root, workspace, *args, **kwargs):
        records = prepare(root, workspace, *args, **kwargs)
        if target == 'downloaded_data':
            payload = workspace / 'downloaded.bin'
            payload.write_bytes(b'original downloaded dataset')
            records['sources'].append({'kind': 'https', 'target': 'downloaded.bin',
                'sha256': digest(payload), 'bytes': payload.stat().st_size, 'verified': True})
        return records

    def write_then_corrupt(path, value):
        write(path, value)
        if path.name != 'execution_plan.json':
            return
        workspace = path.parent / 'repo'
        if target == 'pdf':
            with fixture['pdf'].open('ab') as stream:
                stream.write(b'\n% changed after plan materialization\n')
        elif target in {'experiment_spec.json', 'execution_plan.json', 'source_packet.json'}:
            (path.parent / target).write_text('{"changed":true}', encoding='utf-8')
        elif target == 'source':
            (workspace / 'train.py').write_text('raise SystemExit(0)\n', encoding='utf-8')
        elif target == 'bundled_data':
            (workspace / 'dataset.json').write_text('{}', encoding='utf-8')
        elif target == 'generated_config':
            (workspace / CAPTURE_DIR / 'config.json').write_text('{}', encoding='utf-8')
        else:
            (workspace / 'downloaded.bin').write_bytes(b'changed downloaded dataset')

    monkeypatch.setattr(service, 'prepare_plan_datasets', prepare_with_frozen_download)
    monkeypatch.setattr(service, 'write_json', write_then_corrupt)
    result = runner.run({**request, 'strict_validation': strict, 'prepare_only': True})
    assert result['state'] == 'ERROR', result.get('error')
    assert result['data']['validation']['failure_kind'] == 'source_changed'
    assert result['data']['validation']['is_reproduced'] is None
    assert len(result['data']['repository_identity_attempts']) == 1
    linked.assert_not_called()
    runner.runner.run.assert_not_called()


def test_real_environment_selection_failure_is_not_classified_as_changed_input(tmp_path, monkeypatch):
    from src import discovered_repository_reproduction as service
    import src.author_python as author_python
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    linked = Mock(return_value=[])
    monkeypatch.setattr(service, 'linked_training_repositories', linked)
    monkeypatch.setattr(author_python, 'select_author_python',
                        Mock(side_effect=ValueError('Requested Python runtime is unavailable')))
    runner.runner.run = Mock(side_effect=AssertionError('Environment selection failed before execution'))
    result = runner.run({**request, 'strict_validation': True, 'prepare_only': True})
    assert result['state'] == 'ERROR'
    assert result['data']['validation']['failure_kind'] != 'source_changed'
    assert result['data']['validation']['reason'] == 'Requested Python runtime is unavailable'
    linked.assert_called_once()
    runner.runner.run.assert_not_called()


@pytest.mark.parametrize('strict', [False, True])
def test_initial_materialized_snapshot_check_is_terminal_on_source_change(tmp_path, monkeypatch, strict):
    import sys
    from src import discovered_repository_reproduction as service
    import src.author_python as author_python
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    linked = Mock(return_value=[])
    monkeypatch.setattr(service, 'linked_training_repositories', linked)

    def select_then_corrupt(_packet, *_args, **_kwargs):
        run_dir, = (tmp_path / 'service' / 'runs').iterdir()
        (run_dir / 'repo' / 'train.py').write_text('raise SystemExit(0)\n', encoding='utf-8')
        return {'python_executable': sys.executable}

    monkeypatch.setattr(author_python, 'select_author_python', select_then_corrupt)
    runner.runner.run = Mock(side_effect=AssertionError('Changed source cannot execute'))
    result = runner.run({**request, 'strict_validation': strict, 'prepare_only': True})
    assert result['state'] == 'ERROR'
    assert result['data']['validation']['failure_kind'] == 'source_changed'
    assert not (Path(result['data']['run_dir']) / 'execution_plan.json').exists()
    linked.assert_not_called()
    runner.runner.run.assert_not_called()
