"""Post-run comparison wiring with injected runtime/network boundaries.

The native test also runs an unchanged author fixture when its exact dependency
cache is available. Invented documents never constitute paper acceptance.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from discovered_fixtures import PlannedLLM
from src import discovered_repository_reproduction as service
from src.discovered_repository_runtime import CAPTURE_DIR
from src.method_adapters import digest, read_json
from test_discovered_repository_service import (
    cpu_dependency_cache, setup_service, use_preinstalled_cache,
)


class ComparisonLLM(PlannedLLM):
    def __init__(self, fixture, outcome='matched'):
        super().__init__(fixture['proposal'])
        self.outcome = outcome

    def chat(self, prompt, **kwargs):
        if kwargs['task'] == 'reproduction_assessment':
            self.calls.append((prompt, kwargs))
            raise RuntimeError('This comparison fixture does not supply an AI assessment')
        if kwargs['task'] == 'result_diagnosis':
            self.calls.append((prompt, kwargs))
            evidence = json.loads(prompt)['evidence']
            reference = next(item['id'] for item in evidence if item['kind'] == 'comparison')
            return json.dumps({'hypotheses': [{'category': 'metric',
                'message': '现有输出与参考存在差异，尚需核对指标对应关系。',
                'evidence_ids': [reference]}]})
        if kwargs['task'] != 'author_claim_comparison':
            return super().chat(prompt, **kwargs)
        self.calls.append((prompt, kwargs))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if self.outcome == 'invalid_json':
            return 'invalid comparison JSON'
        if self.outcome == 'no_claim':
            return json.dumps({'comparisons': [], 'reason': '论文没有对应指标'})
        records = json.loads(prompt.split('\nObserved metrics:\n', 1)[1].split('\nOriginal sources:\n', 1)[0])
        sources, _ = json.JSONDecoder().raw_decode(prompt.split('\nOriginal sources:\n', 1)[1])
        excerpts = sources['evidence_excerpts']
        citations = []
        for identifier, source, locator in [('paper_results', 'pdf', 3),
                                             ('readme', 'repository', 'README.md'),
                                             ('script', 'repository', 'train.py')]:
            key = next(key for key, value in excerpts.items() if value['source'] == source
                       and value.get('page' if source == 'pdf' else 'path') == locator)
            citations.append({'id': identifier, 'evidence_id': key})
        test = next(record for record in records if record['split'] == 'test')
        reference = {'name': 'accuracy', 'value': 100.0, 'unit': 'percent',
                     'method': 'Signal Linear', 'dataset': 'SignGrid', 'split': 'test',
                     'aggregation': 'single', 'runs': 1,
                     'value_quote': 'Signal Linear SignGrid test 100.0', 'citations': ['paper_results']}
        if self.outcome == 'baseline':
            reference.update(value=98.0, value_quote='Baseline SignGrid train 98.0')
        return json.dumps({'citations': citations, 'comparisons': [{
            'actual_id': test['id'], 'reference': reference,
            'actual_context': {'unit': 'fraction', 'method': 'Signal Linear', 'dataset': 'SignGrid',
                               'split': 'test', 'aggregation': 'single', 'runs': 1,
                               'citations': ['readme', 'script']},
            'direction': 'higher', 'settings': {
                'status': self.outcome if self.outcome in {'different', 'unknown'} else 'matched',
                'reason': '仅比较固定作者命令的输出与引文，不作独立复算。',
                'citations': ['paper_results', 'readme', 'script']}}]})


def comparison_service(tmp_path, monkeypatch, *, outcome='matched', stdout='Test Accuracy: 0.94',
                       runtime_outcome='success'):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    runner.llm = ComparisonLLM(fixture, outcome)

    def runtime(workspace, steps, environment, **kwargs):
        if not isinstance(steps, dict):
            service.write_json(Path(workspace) / CAPTURE_DIR / 'preflight.json', {'device': 'cpu'})
            return {'success': True, 'steps': []}
        if runtime_outcome == 'interrupted':
            raise KeyboardInterrupt('author command cancelled')
        if runtime_outcome == 'failed':
            return {'success': False, 'steps': [{'id': 'train', 'executed': True,
                    'success': False, 'exit_code': 1, 'stdout': stdout}], 'reason': 'author command failed'}
        config = read_json(Path(workspace) / CAPTURE_DIR / 'config.json')
        service.write_json(Path(workspace) / CAPTURE_DIR / 'author_train.json', {
            'spec_sha256': config['spec_sha256'], 'step': config['steps'][0],
            'natural_return': True, 'optimizer_steps': 40, 'numeric_compatibility': []})
        return {'success': True, 'steps': [{'id': 'train', 'executed': True,
                'success': True, 'exit_code': 0, 'stdout': stdout, 'stderr': ''}]}

    monkeypatch.setattr(runner.runner, 'run', runtime)
    return fixture, runner, request


def assert_code_completed(result):
    assert result['state'] == 'COMPLETED', result.get('error')
    validation = result['data']['validation']
    assert validation['status'] == 'author_code_completed'
    assert validation['paper_metrics_verified'] is False and validation['is_reproduced'] is None
    assert result['data']['execution']['success'] is True
    assert validation['reviews_removed'] is True
    assert '独立核验论文数值' in result['data']['report'] or '未进行论文数值审核' in result['data']['report']
    return validation['author_claim_comparison']


def test_completed_pipeline_compares_real_records_and_persists_report_evidence(tmp_path, monkeypatch):
    from frontend.backend_pipeline import ProgressStore
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    events = []
    result = runner.run(request, on_event=events.append)
    comparison = assert_code_completed(result)
    assert comparison['status'] == 'compared'
    row = comparison['comparisons'][0]
    assert row['actual']['value'] == 0.94 and row['reference']['value'] == 100
    assert row['difference'] == pytest.approx(-6.0) and row['meets_reference'] is False
    run_dir = Path(result['data']['run_dir'])
    assert read_json(run_dir / 'author_claim_comparison.json') == comparison
    validation = read_json(run_dir / 'author_execution_verification.json')
    assert validation == result['data']['validation']
    assert read_json(run_dir / 'execution.json')['author_execution_verification'] == validation
    assert read_json(run_dir / 'result.json')['data']['validation'] == validation
    report = (run_dir / 'report.md').read_text('utf-8')
    assert '作者输出与论文宣称对比' in report and 'PDF 第 3 页' in report
    assert '个百分点' in report and '尚未独立复算' in report and '未达到参考' in report
    for name, sha in fixture['snapshot']['files'].items():
        assert digest(run_dir / 'repo' / name) == sha
    sequence = [event['phase_id'] for event in events if event.get('type') == 'state'
                and event['status'] == 'running']
    assert sequence.index('finish_execution') < sequence.index('compare_author_claims') < sequence.index('generate_report')
    store = ProgressStore(str(tmp_path / 'progress.jsonl'))
    for event in events:
        store.emit(event)
    store.emit({'type': 'done', 'result': result})
    progress = store.read_snapshot(str(store.path))
    stage = next(stage for stage in progress['pipeline_stages'] if stage['id'] == 'compare_author_claims')
    assert stage['status'] == 'success' and stage['completion_confirmed'] is True
    assert progress['done'] is True and progress['running'] is False
    assert [options['task'] for _, options in runner.llm.calls] == [
        'author_execution_plan', 'author_claim_comparison', 'reproduction_assessment', 'result_diagnosis']
    assert result['data']['total_llm_calls'] == len(runner.llm.calls) == 4
    diagnosis = result['data']['outcome_diagnosis']
    assert diagnosis['model_analysis']['status'] == 'completed'
    assert diagnosis['actions'] == {'mutated': False, 'retrained': False}
    assert read_json(run_dir / 'outcome_diagnosis.json') == diagnosis
    assert diagnosis['hypotheses'][0]['evidence_ids'] == ['comparison_1']


@pytest.mark.parametrize('observed', [True, False])
def test_author_runtime_data_stage_finishes_without_claiming_independent_provenance(tmp_path, monkeypatch, observed):
    from frontend.backend_pipeline import ProgressStore
    from src import runtime_dataset_evidence
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    source = ('from torch_geometric.datasets import WikiCS\n'
              'def load_inputs(root):\n    return WikiCS(root=root)\n')
    path = fixture['workspace'] / 'data_loader.py'
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = digest(path)
    fixture['proposal']['citations'].append({'id': 'runtime_loader', 'source': 'repository',
        'path': path.name, 'quote': source})
    fixture['proposal']['datasets'] = [{'kind': 'author_runtime', 'path': path.name,
        'function': 'load_inputs', 'dataset_class': 'WikiCS', 'dataset_name': 'WikiCS',
        'citations': ['runtime_loader']}]
    runner.llm = ComparisonLLM(fixture)
    monkeypatch.setattr(runtime_dataset_evidence, 'audit_runtime_datasets', lambda *args: {
        'verified': observed, 'independent_reference_verified': False, 'sources': [],
        'errors': [] if observed else [{'reason': 'Original input observation is incomplete'}]})
    events = []
    result = runner.run(request, on_event=events.append)
    assert_code_completed(result)
    validation = result['data']['validation']
    assert validation['dataset_observation_verified'] is observed
    assert validation['dataset_provenance_verified'] is False
    store = ProgressStore(str(tmp_path / 'runtime_data_progress.jsonl'))
    for event in events:
        store.emit(event)
    store.emit({'type': 'done', 'result': result})
    progress = store.read_snapshot(str(store.path))
    stage = next(stage for stage in progress['pipeline_stages'] if stage['id'] == 'prepare_dataset')
    assert stage['status'] == ('success' if observed else 'skipped')
    assert stage['completion_confirmed'] is True


@pytest.mark.parametrize('outcome', ['different', 'unknown'])
def test_unconfirmed_settings_show_gap_without_acceptance(tmp_path, monkeypatch, outcome):
    _, runner, request = comparison_service(tmp_path, monkeypatch, outcome=outcome)
    comparison = assert_code_completed(runner.run(request))
    assert comparison['status'] == 'partial'
    row = comparison['comparisons'][0]
    assert row['status'] == 'reference_only' and row['difference'] == pytest.approx(-6.0)
    assert row['meets_reference'] is None and row['within_tolerance'] is None
    assert row['limitations']


@pytest.mark.parametrize('outcome', [RuntimeError('API unavailable'), 'invalid_json', 'no_claim', 'baseline'])
def test_unavailable_comparison_preserves_metrics_and_completed_execution(tmp_path, monkeypatch, outcome):
    _, runner, request = comparison_service(tmp_path, monkeypatch, outcome=outcome)
    result = runner.run(request)
    comparison = assert_code_completed(result)
    assert comparison['status'] == 'unavailable' and comparison['comparisons'] == []
    assert comparison['metric_records'][0]['value'] == 0.94
    assert result['data']['validation']['metric_records'][0]['value'] == 0.94
    expected_calls = 2 if outcome in ('invalid_json', 'baseline') else 1
    assert len([options for _, options in runner.llm.calls if options['task'] == 'author_claim_comparison']) == expected_calls
    assert comparison['reason'] in result['data']['report']


def test_no_metric_output_does_not_request_model_comparison(tmp_path, monkeypatch):
    _, runner, request = comparison_service(tmp_path, monkeypatch, stdout='author command finished')
    comparison = assert_code_completed(runner.run(request))
    assert comparison['status'] == 'unavailable' and comparison['metric_records'] == []
    assert all(options['task'] != 'author_claim_comparison' for _, options in runner.llm.calls)


@pytest.mark.parametrize('mode', ['absent', 'mock'])
def test_no_real_model_preserves_command_fallback_and_actual_metrics(tmp_path, monkeypatch, mode):
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    # Supply a literal launcher, rather than the prose command intended for
    # the planning model, so this test reaches the API-free execution path.
    launcher = fixture['workspace'] / 'run.sh'
    launcher.write_text('python train.py --epochs 40 --seed 7 --cpu\n', encoding='utf-8')
    fixture['snapshot']['files']['run.sh'] = digest(launcher)
    llm = runner.llm
    if mode == 'absent':
        runner.llm = None
    else:
        llm.mock_mode = True
    comparison = assert_code_completed(runner.run(request))
    assert comparison['status'] == 'unavailable' and comparison['metric_records'][0]['value'] == 0.94
    assert '真实模型' in comparison['reason']
    assert llm.calls == []


def test_unexpected_comparison_error_is_informational(tmp_path, monkeypatch):
    from src import author_claim_comparison
    _, runner, request = comparison_service(tmp_path, monkeypatch)

    def fail_comparison(*args, **kwargs):
        raise RuntimeError('comparison boundary unavailable')

    monkeypatch.setattr(author_claim_comparison, 'compare_author_claims', fail_comparison)
    comparison = assert_code_completed(runner.run(request))
    assert comparison['status'] == 'unavailable' and comparison['metric_records'][0]['value'] == 0.94
    assert comparison['diagnostic'] == 'comparison boundary unavailable'


@pytest.mark.parametrize('mode', ['prepare_only', 'prepare_environment'])
def test_preparation_never_requests_post_run_comparison(tmp_path, monkeypatch, mode):
    _, runner, request = comparison_service(tmp_path, monkeypatch)
    events = []
    result = runner.run({**request, mode: True}, on_event=events.append)
    assert result['state'] == 'COMPLETED'
    assert 'author_claim_comparison' not in result['data']['validation']
    assert all(options['task'] != 'author_claim_comparison' for _, options in runner.llm.calls)
    assert any(event.get('phase_id') == 'compare_author_claims' and event['status'] == 'skipped' for event in events)


@pytest.mark.parametrize('runtime_outcome', ['failed', 'interrupted'])
def test_unsuccessful_execution_never_requests_comparison(tmp_path, monkeypatch, runtime_outcome):
    _, runner, request = comparison_service(tmp_path, monkeypatch, runtime_outcome=runtime_outcome)
    if runtime_outcome == 'interrupted':
        with pytest.raises(KeyboardInterrupt, match='author command cancelled'):
            runner.run(request)
    else:
        result = runner.run(request)
        assert result['state'] == 'ERROR' and result['data']['validation']['status'] == 'execution_failed'
    assert all(options['task'] != 'author_claim_comparison' for _, options in runner.llm.calls)


def test_cancelling_comparison_preserves_author_completion_and_stops_run(tmp_path, monkeypatch):
    _, runner, request = comparison_service(tmp_path, monkeypatch, outcome=KeyboardInterrupt('comparison cancelled'))
    events = []
    with pytest.raises(KeyboardInterrupt, match='comparison cancelled'):
        runner.run(request, on_event=events.append)
    run_dir = next((runner.root / 'runs').iterdir())
    result = read_json(run_dir / 'result.json')
    assert result['data']['validation']['status'] == 'author_code_completed'
    assert result['data']['execution']['success'] is True
    assert result['data']['validation']['author_claim_comparison']['status'] == 'interrupted'
    assert read_json(run_dir / 'run_status.json')['status'] == 'interrupted'
    assert any(event.get('phase_id') == 'compare_author_claims' and event['status'] == 'interrupted' for event in events)
    assert not any(event.get('phase_id') == 'generate_report' for event in events)


def test_optional_comparison_storage_failure_does_not_replace_completion(tmp_path, monkeypatch):
    _, runner, request = comparison_service(tmp_path, monkeypatch)
    persist = service.write_json

    def fail_comparison_file(path, value):
        if Path(path).name == 'author_claim_comparison.json':
            raise OSError('comparison file unavailable')
        return persist(path, value)

    monkeypatch.setattr(service, 'write_json', fail_comparison_file)
    result = runner.run(request)
    comparison = assert_code_completed(result)
    assert comparison['status'] == 'compared' and 'file unavailable' in comparison['persistence_error']
    assert 'file unavailable' in result['data']['report']
    assert read_json(Path(result['data']['run_dir']) / 'result.json')['data']['validation'] == result['data']['validation']


def test_native_author_output_reaches_comparison_without_retraining(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    runner.llm = ComparisonLLM(fixture)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    comparison = assert_code_completed(result)
    assert comparison['status'] == 'compared'
    assert comparison['comparisons'][0]['actual']['value'] == 1.0
    assert result['data']['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert len(result['data']['execution']['steps']) == 1


@pytest.mark.parametrize('state', ['complete', 'unobserved', 'changed'])
def test_runtime_input_observation_is_persisted_without_independent_acceptance(tmp_path, monkeypatch, state):
    from src.method_adapters import digest
    fixture, runner, request = comparison_service(tmp_path, monkeypatch)
    script = fixture['workspace'] / 'train.py'
    source = ('from torch_geometric.datasets import WikiCS\n'
              'def original_loader(path):\n    return WikiCS(root=path)\n' + script.read_text('utf-8'))
    script.write_bytes(source.encode())
    fixture['snapshot']['files']['train.py'] = digest(script)
    next(c for c in fixture['proposal']['citations'] if c['id'] == 'script')['quote'] = source
    fixture['proposal']['datasets'] = [{'kind': 'author_runtime', 'path': 'train.py',
        'function': 'original_loader', 'dataset_class': 'WikiCS', 'dataset_name': 'WikiCS', 'citations': ['script']}]
    runner.llm.proposal = deepcopy(fixture['proposal'])
    original = runner.runner.run
    def runtime(workspace, steps, environment, **kwargs):
        result = original(workspace, steps, environment, **kwargs)
        if not isinstance(steps, dict):
            return result
        workspace = Path(workspace)
        config = read_json(workspace / CAPTURE_DIR / 'config.json')
        assert len(config['runtime_datasets']) == 1
        if state != 'unobserved':
            root = workspace / 'invented_native_data'
            root.mkdir()
            files = {}
            for name, role in [('raw.bin', 'raw'), ('processed.bin', 'processed')]:
                path = root / name
                path.write_bytes(b'invented service wiring fixture')
                files[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size, 'role': role}
            receipt = read_json(workspace / CAPTURE_DIR / 'author_train.json')
            receipt['native_datasets'] = [{'declaration': config['runtime_datasets'][0], 'verified': True,
                'dataset_name': 'WikiCS', 'root': str(root), 'files': files,
                'loaded_before_training': True, 'sample_count': 1}]
            service.write_json(workspace / CAPTURE_DIR / 'author_train.json', receipt)
            if state == 'changed':
                (root / 'raw.bin').write_bytes(b'changed after construction')
        return result
    monkeypatch.setattr(runner.runner, 'run', runtime)
    result = runner.run(request)
    assert_code_completed(result)
    data, validation = result['data'], result['data']['validation']
    evidence = validation['runtime_dataset_evidence']
    assert validation['dataset_observation_verified'] is (state == 'complete')
    assert validation['dataset_provenance_verified'] is False
    assert evidence['independent_reference_verified'] is False
    assert data['runtime_dataset_evidence'] == evidence
    run_dir = Path(data['run_dir'])
    assert read_json(run_dir / 'runtime_dataset_evidence.json') == evidence
    assert read_json(run_dir / 'dataset.json')['runtime_observation'] == evidence
    assert '原作者数据加载观测' in data['report'] and '尚未独立核对' in data['report']
    if state == 'complete':
        assert 'raw.bin' in data['report'] and next(iter(evidence['sources'][0]['observations'][0]['files'].values()))['sha256'] in data['report']
    else:
        assert evidence['errors'] and validation['warnings']
