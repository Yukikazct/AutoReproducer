"""Aggregate observations require explicit public authorization and bounded data."""
import hashlib
import json
import sys

import pytest

from src.investigation_tools import InvestigationTools


def observe(tmp_path, payload, *, public=True, authorized=True, name='metrics.json'):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, allow_nan=True).encode()
    path.write_bytes(raw)
    packet = {'file_inventory': {}, 'investigation_artifacts': {name: {
        'sha256': hashlib.sha256(raw).hexdigest(), 'public': public, 'metric_summary': authorized}}}
    tools = InvestigationTools(tmp_path, packet, {'python_executable': sys.executable}, tmp_path / 'run')
    return tools.execute('artifact_contract', {'paths': [name]})


def test_explicit_aggregate_returns_metric_without_prediction_values(tmp_path):
    result = observe(tmp_path, {'metric': 'accuracy', 'value': 0.91, 'unit': 'fraction',
        'runs': 3, 'dataset': 'original', 'targets': ['PRIVATE-TARGET'], 'predictions': [42],
        'notes': 'PRIVATE-LITERAL'})
    assert result['status'] == 'completed'
    summary = result['evidence'][0]['value']['metric_summary']
    assert summary == {'metric': 'accuracy', 'value': 0.91, 'unit': 'fraction', 'runs': 3, 'dataset': 'original'}
    rendered = json.dumps(result)
    assert 'PRIVATE-TARGET' not in rendered and 'PRIVATE-LITERAL' not in rendered
    assert '42' not in json.dumps(summary)


def test_generic_public_json_does_not_authorize_value_projection(tmp_path):
    result = observe(tmp_path, {'metric': 'accuracy', 'value': 0.91}, authorized=False)
    assert result['status'] == 'completed'
    assert 'metric_summary' not in result['evidence'][0]['value']
    assert '0.91' not in json.dumps(result)


@pytest.mark.parametrize('payload', [
    {'metric': 'accuracy', 'value': True}, {'metric': 'accuracy', 'value': [0.91]},
    {'metric': 'accuracy', 'value': {'hidden_label': 1}}, {'metric': '', 'value': 0.91},
    {'metric': 'a' * 129, 'value': 0.91}, {'metric': 'accuracy'},
])
def test_invalid_aggregate_never_exposes_numeric_or_row_payload(tmp_path, payload):
    result = observe(tmp_path, payload)
    assert result['status'] == 'completed'
    value = result['evidence'][0]['value']
    assert 'metric_summary' not in value
    assert value['metric_summary_status'] == 'invalid_or_unavailable'
    assert 'hidden_label' not in json.dumps(value)


@pytest.mark.parametrize('public,name', [(False, 'metrics.json'), (True, 'private/metrics.json')])
def test_explicit_metric_flag_does_not_override_private_artifact_guard(tmp_path, public, name):
    result = observe(tmp_path, {'metric': 'accuracy', 'value': 0.91}, public=public, name=name)
    assert result['status'] == 'rejected'
    assert '0.91' not in json.dumps(result)


@pytest.mark.parametrize('raw', [
    b'{"metric":"accuracy","value":0.91,"value":0.99}',
    b'{"metric":"accuracy","value":NaN}', b'{"metric":"accuracy","value":Infinity}',
    b'{"metric":"accuracy","value":1e999}',
    b'{"metric":"accuracy","value":0.91,"unit":"fraction","unit":"percent"}',
])
def test_ambiguous_or_nonfinite_json_is_rejected_before_a_metric_claim(tmp_path, raw):
    (tmp_path / 'metrics.json').write_bytes(raw)
    tools = InvestigationTools(tmp_path, {'file_inventory': {}, 'investigation_artifacts': {
        'metrics.json': {'sha256': hashlib.sha256(raw).hexdigest(), 'public': True, 'metric_summary': True}}},
        {'python_executable': sys.executable}, tmp_path / 'run')
    result = tools.execute('artifact_contract', {'paths': ['metrics.json']})
    assert result['status'] == 'rejected'
    assert result['evidence'] == []
