"""Preparation ordering and frozen inputs; injected process results are not paper acceptance."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.method_adapters import read_json, write_json
from test_author_dataset_loader import LOADER
from test_discovered_repository_service import setup_service
from discovered_fixtures import PlannedLLM


def with_loader(fixture):
    root = fixture['workspace']
    loader = LOADER.replace("'..', 'data', name", "'data', name")
    script = (root / 'train.py').read_text('utf-8').replace('torch.manual_seed(args.seed)',
        "from author_data import load\nreal_dataset = load('Cora')\ntorch.manual_seed(args.seed)")
    for name, source in {'author_data.py': loader, 'train.py': script}.items():
        path = root / name
        path.write_bytes(source.encode())
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        citation = next((item for item in fixture['proposal']['citations'] if item.get('path') == name), None)
        if citation is None:
            fixture['proposal']['citations'].append({'id': 'loader', 'source': 'repository', 'path': name, 'quote': source})
        else:
            citation['quote'] = source
    fixture['proposal']['datasets'] = [{'kind': 'author_loader', 'path': 'author_data.py',
                                        'function': 'load', 'citations': ['loader', 'script']}]


def test_loader_planning_binds_the_actual_call_and_never_accepts_model_arguments(tmp_path):
    from discovered_fixtures import make_experiment
    fixture = make_experiment(tmp_path)
    with_loader(fixture)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    result = propose_plan(PlannedLLM(fixture['proposal']), packet, fixture['workspace'])
    loader = result['datasets'][0]
    assert loader['name'] == 'Cora' and loader['arguments']['normalize_features'] is False
    assert 'torch-geometric==2.6.1' in result['requirements']['compatibility']
    assert any(pin.startswith('scipy==') for pin in result['requirements']['compatibility'])
    assert any(item.get('evidence_scope') == 'reviewed_planetoid_sparse_pickle_loading'
               for item in result['requirements']['import_evidence'])
    fixture['proposal']['datasets'][0]['arguments'] = {'name': 'PubMed'}
    with pytest.raises(ValueError, match='selects only'):
        propose_plan(PlannedLLM(fixture['proposal']), packet, fixture['workspace'])


def configure_service(tmp_path, monkeypatch):
    fixture, service, request, llm = setup_service(tmp_path, monkeypatch)
    with_loader(fixture)
    llm.proposal = deepcopy(fixture['proposal'])
    # setup_service's export closure holds the same snapshot mapping.
    request['offline'] = False
    return fixture, service, request


def test_prepare_only_reports_pending_real_data_and_starts_no_process(tmp_path, monkeypatch):
    _, service, request = configure_service(tmp_path, monkeypatch)
    monkeypatch.setattr(service.runner, 'run', lambda *args, **kwargs: pytest.fail('prepare_only starts no process'))
    events = []
    result = service.run({**request, 'prepare_only': True}, on_event=events.append)
    assert result['state'] == 'COMPLETED', result['error']
    assert result['data']['dataset_provenance']['verified'] is False
    assert result['data']['validation']['is_reproduced'] is None
    assert '仍待' in result['data']['validation']['reason']
    assert not any(event.get('phase_id') == 'prepare_dataset' and event.get('status') == 'success' for event in events)


@pytest.mark.parametrize('mode', ['valid', 'failed_process', 'missing_record', 'tampered_data'])
def test_dependency_then_data_order_and_no_training_with_unfrozen_data(tmp_path, monkeypatch, mode):
    _, service, request = configure_service(tmp_path, monkeypatch)
    calls = []
    def run(workspace, steps, environment, **kwargs):
        identifier = steps[0]['id']
        calls.append(identifier)
        runtime = Path(workspace) / '_autorepro_capture'
        if identifier == 'native_preflight':
            write_json(runtime / 'preflight.json', {'native_probe_pass': True, 'device': 'cpu'})
        elif identifier == 'prepare_author_datasets':
            assert environment['auto_prepare'] is False and environment['require_prepared'] is True
            if mode == 'failed_process':
                return {'success': False, 'final': {'stderr': 'Dataset process failed'}}
            configuration = read_json(runtime / 'dataset_loaders.json')
            loader = configuration['loaders'][0]
            target = Path(workspace) / loader['root'] / 'raw.bin'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'fixture process data record')
            name = target.relative_to(workspace).as_posix()
            record = {'kind': 'author_loader', 'verified': True, 'loader': loader,
                      'files': {name: {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}}}
            write_json(runtime / 'prepared_datasets.json', {'verified': True, 'sources': [] if mode == 'missing_record' else [record]})
            if mode == 'tampered_data':
                target.write_bytes(b'changed after successful data process')
        else:
            pytest.fail('No author training may start in an environment preparation test')
        return {'success': True, 'steps': [{'id': identifier, 'success': True, 'executed': True, 'exit_code': 0}]}
    monkeypatch.setattr(service.runner, 'run', run)
    events = []
    result = service.run({**request, 'prepare_environment': True}, on_event=events.append)
    assert calls == ['native_preflight', 'prepare_author_datasets']
    assert result['data']['repository_executed'] is False
    assert result['data']['validation']['is_reproduced'] is None
    if mode == 'valid':
        assert result['state'] == 'COMPLETED', result['error']
        assert result['data']['dataset_provenance']['verified'] is True
        runtime = Path(result['data']['run_dir']) / 'repo/_autorepro_capture'
        assert read_json(runtime / 'config.json')['data_files']
        assert result['data']['execution_plan']['dataset']['verified'] is True
        data_success = next(index for index, event in enumerate(events)
                            if event.get('phase_id') == 'prepare_dataset' and event.get('status') == 'success')
        env_success = next(index for index, event in enumerate(events)
                           if event.get('phase_id') == 'prepare_environment' and event.get('status') == 'success')
        assert data_success > env_success
    else:
        assert result['state'] == 'ERROR'
        assert result['data']['validation']['failure_phase'] == 'prepare_dataset'
