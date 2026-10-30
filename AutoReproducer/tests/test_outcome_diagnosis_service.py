"""Auxiliary result diagnosis cannot replace an already completed experiment."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.discovered_repository_reproduction import _diagnose_outcome
from src import result_diagnosis
from src.agents.report_generator import _outcome_diagnosis_lines
from src.agent_collaboration import CollaborationSession


def test_auxiliary_failure_preserves_execution_and_validation(monkeypatch, tmp_path):
    data = {'execution': {'success': True}, 'validation': {
        'status': 'author_code_completed', 'is_reproduced': None}}
    before = deepcopy(data)
    monkeypatch.setattr(result_diagnosis, 'diagnose_experiment_result',
                        Mock(side_effect=ValueError('provider-sensitive-message')))
    result = _diagnose_outcome(None, {}, {}, tmp_path, data['execution'],
                               data['validation'], tmp_path, data, None)
    assert {key: data[key] for key in before} == before
    assert result['status'] == 'unavailable'
    assert 'provider-sensitive-message' not in str(result)
    assert '暂不可用' in '\n'.join(_outcome_diagnosis_lines(data))


def test_auxiliary_cancellation_still_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(result_diagnosis, 'diagnose_experiment_result',
                        Mock(side_effect=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _diagnose_outcome(None, {}, {}, tmp_path, {}, {}, tmp_path, {}, None)


def test_repaired_results_use_approved_source_copy_and_hashes(monkeypatch, tmp_path):
    packet = {'file_inventory': {'train.py': {'sha256': 'old', 'text_available': True}}}
    data = {'source_repair': {'status': 'succeeded', 'workspace': str(tmp_path / 'repaired'),
                             'effective_source_files': {'train.py': 'approved'}}}
    diagnose = Mock(return_value={'status': 'diagnosed'})
    monkeypatch.setattr(result_diagnosis, 'diagnose_experiment_result', diagnose)
    _diagnose_outcome(None, {}, packet, tmp_path / 'original', {}, {}, tmp_path, data, None)
    args = diagnose.call_args.args
    assert args[3] == tmp_path / 'repaired'
    assert args[2]['file_inventory']['train.py'] == {'sha256': 'approved', 'text_available': True}
    assert packet['file_inventory']['train.py']['sha256'] == 'old'


def test_real_diagnosis_service_reuses_evidence_and_preserves_author_completion(tmp_path):
    llm = Mock(mock_mode=False, api_key='', base_url='https://example.invalid', model='fixture')
    llm.chat.return_value = '{"hypotheses":[]}'
    execution = {'success': True, 'steps': [{'id': 'train', 'executed': True, 'success': True,
                                          'exit_code': 0, 'stdout': 'accuracy: 0.7'}]}
    validation = {'status': 'author_code_completed', 'is_reproduced': None,
        'paper_metrics_verified': False, 'author_claim_comparison': {
            'status': 'compared', 'comparisons': [{'metric': 'accuracy', 'status': 'reference_only',
                'reference': {'value': .9}, 'actual': {'value': .7}, 'relative_difference': -.22}]}}
    before = deepcopy((execution, validation))
    data, session = {'execution': execution, 'validation': validation}, CollaborationSession()
    with session.scope():
        first = _diagnose_outcome(llm, {'steps': []}, {'file_inventory': {}}, tmp_path,
                                  execution, validation, tmp_path / 'run', data, None)
        second = _diagnose_outcome(llm, {'steps': []}, {'file_inventory': {}}, tmp_path,
                                   execution, validation, tmp_path / 'run', data, None)
    assert first is second and llm.chat.call_count == 1
    assert second['efficiency']['cache_hits'] == 1
    assert second['assessment']['comparison_coverage']['reference_only'] == 1
    assert second['next_steps'][0]['priority'] == 1
    assert (execution, validation) == before
    assert any(message['recipient'] == 'ReportGenerator' and message['kind'] == 'outcome_diagnosis'
               for message in session.snapshot()['messages'])
