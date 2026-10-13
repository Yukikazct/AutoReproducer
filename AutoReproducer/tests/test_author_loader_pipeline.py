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
    assert '尚未' in result['data']['validation']['reason']
    assert result['data']['grounded_repository_plan']['datasets'] == []


@pytest.mark.parametrize('mode', ['ready', 'failed_process', 'repaired_environment'])
def test_author_loader_runs_with_original_command_after_environment_preparation(tmp_path, monkeypatch, mode):
    _, service, request = configure_service(tmp_path, monkeypatch)
    calls = []
    def run(workspace, steps, environment, **kwargs):
        planned = steps['steps'] if isinstance(steps, dict) else steps
        identifier = planned[0]['id']
        calls.append(identifier)
        if identifier == 'native_preflight':
            if mode == 'failed_process':
                return {'success': False, 'final': {'stderr': 'Actual preparation failure'}}
            write_json(Path(workspace) / '_autorepro_capture/preflight.json', {'native_probe_pass': True, 'device': 'cpu'})
            return {'success': True, 'steps': [{'id': identifier, 'exit_code': 0, 'success': True}],
                    'effective_env_config': {**environment, 'requirements_txt': environment['requirements_txt'] + '\neinops==0.8.0\n'}}
        assert identifier == 'train'
        assert mode == 'repaired_environment'
        assert 'einops==0.8.0' in environment['requirements_txt']
        assert planned[0]['timeout_s'] is None
        return {'success': False, 'final': {'stderr': 'Actual author data loading error', 'exit_code': 1}}
    monkeypatch.setattr(service.runner, 'run', run)
    result = service.run({**request, 'prepare_environment': mode != 'repaired_environment'})
    assert calls == (['native_preflight', 'train'] if mode == 'repaired_environment' else ['native_preflight'])
    assert result['data']['validation']['is_reproduced'] is None
    if mode == 'ready':
        assert result['state'] == 'COMPLETED', result['error']
        assert result['data']['repository_executed'] is False
        assert result['data']['dataset_provenance']['verified'] is False
    else:
        assert result['state'] == 'ERROR'
        assert result['data']['validation']['failure_phase'] == ('execute_repository' if mode == 'repaired_environment' else 'prepare_environment')
