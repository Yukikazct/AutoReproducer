"""Real combined recovery and observation, using invented author programs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from src.method_adapters import digest, read_json
from test_discovered_repository_service import cpu_dependency_cache, setup_service, use_preinstalled_cache
from test_source_repair_service import legacy_api_fixture


def update_source(fixture, llm, name, source):
    (fixture['workspace'] / name).write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][name] = digest(fixture['workspace'] / name)
    for citation in llm.proposal['citations']:
        if citation.get('path') == name:
            citation['quote'] = source


def enable_source_repair(request, llm, monkeypatch, *, shift_lines=False):
    request.update(source_repair=True, source_repair_auto=True, source_repair_editable=[])
    original_chat = llm.chat
    def chat(prompt, **kwargs):
        if kwargs['task'] != 'repository_repair':
            return original_chat(prompt, **kwargs)
        context = json.loads(prompt)
        llm.calls.append((prompt, kwargs))
        return json.dumps({'reason': 'Correct the obsolete dictionary API after the actual failure.',
            'patches': [{'path': name, 'base_sha256': record['base_sha256'],
                         'content': ('\n\n' if shift_lines else '')
                                    + record['content'].replace('.iteritems()', '.items()')}
                        for name, record in context['sources'].items()]})
    monkeypatch.setattr(llm, 'chat', chat)


@pytest.mark.parametrize('shift_lines', [False, True])
def test_numeric_repair_survives_source_copy_with_training_and_new_log_evidence(
        tmp_path, monkeypatch, cpu_dependency_cache, shift_lines):
    fixture, service, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, service, cpu_dependency_cache)
    legacy_api_fixture(fixture, llm)
    source = (fixture['workspace'] / 'train.py').read_text('utf-8')
    source = ('import numpy as np\nassert np.float(1) == 1\n' + source
              + "\nwith open('results.log', 'a', encoding='utf-8') as stream:\n"
                "    stream.write('test_acc: 1.0\\n')\n")
    update_source(fixture, llm, 'train.py', source)
    (fixture['workspace'] / 'results.log').write_text('test_acc: 0.125\n', encoding='utf-8')
    fixture['snapshot']['files']['results.log'] = digest(fixture['workspace'] / 'results.log')
    enable_source_repair(request, llm, monkeypatch, shift_lines=shift_lines)

    result = service.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    state = data['source_repair']
    assert state['status'] == 'succeeded'
    assert data['self_repair']['status'] == 'resolved'
    assert len(data['numeric_repair_attempts']) == 1
    validation = data['validation']
    assert validation['source_modified'] is True
    assert validation['training_observed'] is True
    assert validation['paper_metrics_verified'] is False
    observed = validation['author_execution']
    assert len(observed) == 1 and observed[0]['optimizer_steps'] == 40
    repair = observed[0]['numeric_compatibility'][0]
    assert repair['kind'] == 'numpy_builtin_alias'
    assert repair['line'] == 2 + (2 if shift_lines else 0)
    repaired_workspace = Path(state['workspace'])
    assert repair['source_sha256'] == digest(repaired_workspace / 'train.py')
    assert repair['source_sha256'] != fixture['snapshot']['files']['train.py']
    assert (repaired_workspace / 'results.log').read_text('utf-8') == 'test_acc: 0.125\n'
    from src.author_output_metrics import read_output_metrics
    records, errors = read_output_metrics(observed, repaired_workspace, state['effective_source_files'])
    assert not errors
    assert len(records) == 1 and records[0]['value'] == 1.0
    for name, expected in fixture['snapshot']['files'].items():
        assert digest(Path(data['run_dir']) / 'repo' / name) == expected
    manifest = read_json(state['manifest_path'])
    assert manifest['observation']['generated_files']
    assert manifest['observation']['original_compatibility']['numeric_compatibility'][0]['line'] == 2
    with zipfile.ZipFile(state['evidence_archive']['path']) as archive:
        assert 'source_repair_observation.json' in archive.namelist()
        assert 'observation/_autorepro_capture/author_train.json' in archive.namelist()
        assert 'test_acc: 1.0' in archive.read('observation/' + observed[0]['output_files'][0]['captured_path']).decode()


@pytest.mark.parametrize('family', ['path', 'pyg'])
def test_recorded_data_and_in_memory_repair_survive_source_copy(
        tmp_path, monkeypatch, cpu_dependency_cache, family):
    fixture, service, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, service, cpu_dependency_cache)
    legacy_api_fixture(fixture, llm)
    original = fixture['workspace']
    source = (original / 'train.py').read_text('utf-8')
    if family == 'path':
        from test_author_path_compatibility import LOADER
        prefix = LOADER.replace('cora.npz', 'input_payload.bin')
        input_name = '_autorepro_capture/author_data/input_payload.bin'
    else:
        from test_author_pyg_compatibility import CREATE, LOADER
        update_source(fixture, llm, 'loader.py', LOADER)
        input_name = 'graph.pt'
        created = subprocess.run([sys.executable, '-c', CREATE, str(original / input_name), 'legacy'],
            env={**os.environ, 'PYTHONPATH': str(cpu_dependency_cache)},
            capture_output=True, text=True, encoding='utf-8', timeout=60)
        assert created.returncode == 0, created.stderr
        fixture['snapshot']['files'][input_name] = digest(original / input_name)
        prefix = 'from loader import OldGraph\nassert OldGraph()[0].num_nodes == 3\n'
    update_source(fixture, llm, 'train.py', prefix + source)
    enable_source_repair(request, llm, monkeypatch, shift_lines=True)
    result = service.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert len(data[family + '_repair_attempts']) == 1
    assert data['source_repair']['status'] == 'succeeded'
    assert data['self_repair']['status'] == 'resolved'
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    state = data['source_repair']
    candidate = Path(state['workspace'])
    original = Path(data['run_dir']) / 'repo'
    assert digest(candidate / input_name) == digest(original / input_name)
    assert read_json(state['manifest_path'])['frozen_input_files'][input_name]['sha256'] == digest(original / input_name)
    repairs = data['validation'][family + '_compatibility']
    assert len(repairs) == 1
    hash_key = 'caller_sha256' if family == 'pyg' else 'source_sha256'
    assert repairs[0][hash_key] == digest(candidate / 'train.py')
    if family == 'pyg':
        assert data['validation']['author_execution'][0]['pyg_conversions'] == repairs


def test_observed_failed_candidate_can_repair_next_error_without_stale_observer_cache(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, service, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, service, cpu_dependency_cache)
    legacy_api_fixture(fixture, llm)
    source = (fixture['workspace'] / 'train.py').read_text('utf-8').replace('.iteritems()', '.iterkeys()')
    update_source(fixture, llm, 'train.py', source)
    request.update(source_repair=True, source_repair_auto=True, source_repair_editable=[])
    original_chat = llm.chat
    rounds = []

    def chat(prompt, **kwargs):
        if kwargs['task'] != 'repository_repair':
            return original_chat(prompt, **kwargs)
        context = json.loads(prompt)
        rounds.append(context)
        patches = []
        for name, record in context['sources'].items():
            candidate = record['content'].replace('.iteritems()', '.items()')
            if len(rounds) > 1:
                candidate = candidate.replace('.iterkeys()', '.keys()')
            if candidate != record['content']:
                patches.append({'path': name, 'base_sha256': record['base_sha256'], 'content': candidate})
        return json.dumps({'reason': 'Repair only compatibility APIs demonstrated by actual failures.',
                           'patches': patches})

    monkeypatch.setattr(llm, 'chat', chat)
    result = service.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    attempts = data['source_repair']['attempts']
    assert [attempt['status'] for attempt in attempts] == ['failed', 'succeeded']
    assert all(attempt['execution']['executed'] for attempt in attempts)
    assert 'iterkeys' in attempts[0]['execution']['steps'][0]['stderr']
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert data['source_repair']['status'] == 'succeeded'
