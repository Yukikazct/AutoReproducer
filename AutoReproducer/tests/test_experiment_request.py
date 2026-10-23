"""User experiment selection is optional and cannot replace source evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.experiment_request import bind_experiment_request, normalize_experiment_request
from src.agents.experiment_protocol import validate_experiment_protocol
from src.discovered_repository_plan import PlanEvidenceError
from test_experiment_protocol import protocol_fixture
from test_verify_browser_pdf_holdout import Page
from scripts import verify_browser_pdf_holdout as browser


def test_selection_normalizes_empty_names_and_preserves_original_spelling():
    assert normalize_experiment_request() == {}
    assert normalize_experiment_request({'method': '  ', 'dataset': ' Grid '}) == {'dataset': 'Grid'}
    assert bind_experiment_request({'method': {'name': 'GCN*'}, 'dataset': {'name': 'Cora'}},
        {'method': 'gcn', 'dataset': 'cora'}) == {'method': 'gcn', 'dataset': 'cora'}


@pytest.mark.parametrize('value', [[], '', {'epochs': 5}, {'dataset': 4},
    {'dataset': 'a' * 129}, {'method': 'GCN\nother'}, {'dataset': 'Cora\x00'}])
def test_selection_rejects_malformed_requests(value):
    with pytest.raises(ValueError):
        normalize_experiment_request(value)


@pytest.mark.parametrize('kind', ['method', 'dataset'])
def test_protocol_cannot_switch_requested_experiment(tmp_path, kind):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    packet['experiment_request'] = {kind: raw[kind]['name']}
    assert validate_experiment_protocol(raw, packet, fixture['workspace'])
    packet['experiment_request'][kind] = 'Another experiment'
    with pytest.raises(PlanEvidenceError, match='differs from the user request'):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


def test_browser_sets_target_and_detects_changed_backend_selection():
    page = Page()
    requested = browser.configure_experiment_request(page,
        SimpleNamespace(experiment_method='Fixture', experiment_dataset='Grid'))
    assert requested == {'method': 'Fixture', 'dataset': 'Grid'}
    assert page.fields['实验数据集（可选）'].presses == ['Enter']
    result = {'data': {'experiment_request': requested,
        'grounded_repository_plan': {'experiment_protocol': {
            'method': {'name': 'Fixture'}, 'dataset': {'name': 'Grid'}}}}}
    browser.verify_experiment_request(result, requested, True)
    altered = deepcopy(result)
    altered['data']['grounded_repository_plan']['experiment_protocol']['dataset']['name'] = 'Other'
    with pytest.raises(ValueError, match='differs from the user request'):
        browser.verify_experiment_request(altered, requested, True)
    altered['data']['experiment_request'] = {}
    with pytest.raises(AssertionError, match='backend received'):
        browser.verify_experiment_request(altered, requested, False)
    assert browser.configure_experiment_request(page, SimpleNamespace()) == {}
    assert page.fields['实验数据集（可选）'].value == ''
