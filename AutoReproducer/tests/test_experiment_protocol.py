"""A protocol binds experiment selection before execution; unsupported stays failed."""
from copy import deepcopy
import hashlib

import pytest

from discovered_fixtures import PlannedLLM, fixture_protocol, make_experiment
from src.agents.experiment_protocol import ExperimentProtocolAgent, bind_plan_to_protocol, validate_experiment_protocol
from src import discovered_repository_plan as planner


def protocol_fixture(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    raw = fixture_protocol(fixture['proposal'], planner._planning_view(packet))
    protocol = validate_experiment_protocol(raw, packet, fixture['workspace'])
    return fixture, packet, raw, protocol


def test_selected_experiment_binds_command_numbers_repetitions_and_device(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    packet['experiment_protocol'] = protocol
    llm = PlannedLLM(fixture['proposal'])
    plan = planner.propose_plan(llm, packet, fixture['workspace'])
    assert plan['experiment_protocol']['protocol_sha256'] == protocol['protocol_sha256']
    assert plan['requirements']['import_evidence']
    assert plan['requirements']['compatibility'] == ['numpy==1.26.4','torch==2.5.1+cpu']
    assert 'experiment_protocol' in llm.calls[0][0]
    for mutation, match in [
        (lambda p:p['metrics'][0].update(reference=98), 'selected paper reference'),
        (lambda p:p['steps'][0]['argv'].append('--other'), 'selected author command'),
        (lambda p:p['capture'].update(expected_runs=2), 'all paper repetitions'),
        (lambda p:p['capture'].update(device='cuda'), 'Execution device'),
        (lambda p:p['parameters'].update(seed=8), 'frozen training parameter'),
    ]:
        changed = deepcopy(plan)
        mutation(changed)
        with pytest.raises(planner.PlanEvidenceError, match=match):
            bind_plan_to_protocol(changed, protocol)


def test_numeric_representation_is_normalized_only_when_original_source_supports_value(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    raw['repetitions']['count'] = '1'
    raw['metrics'][0]['reference'] = '1e2'
    raw['training']['parameters'][0].update(name='--epochs', value='4e1')
    accepted = validate_experiment_protocol(raw, packet, fixture['workspace'])
    assert accepted['training']['parameters'][0]['value'] == 40
    assert accepted['training']['parameters'][0]['name'] == 'epochs'
    assert accepted['metrics'][0]['reference'] == 100
    raw['metrics'][0]['reference'] = '97.123'
    with pytest.raises(planner.PlanEvidenceError, match='real paper number'):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


def test_protocol_conflict_is_a_distinct_non_retryable_rejection(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    raw['alignment'].update(status='conflict', reason='Paper repetitions and native author entrypoint disagree.')
    with pytest.raises(planner.PlanEvidenceError, match='experiment conflict') as exc:
        validate_experiment_protocol(raw, packet, fixture['workspace'])
    assert exc.value.planning_evidence['stage'] == 'experiment_protocol'


@pytest.mark.parametrize('status,category', [('conflict','protocol_conflict'), ('insufficient','missing_information')])
def test_negative_protocol_does_not_require_fabricated_positive_fields(tmp_path, status, category):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    rejection = {'version': 1, 'alignment': {**raw['alignment'], 'status': status,
                  'reason': 'Paper and repository do not establish the complete repetition protocol.'}}
    with pytest.raises(planner.PlanEvidenceError) as exc:
        validate_experiment_protocol(rejection, packet, fixture['workspace'])
    assert exc.value.category == category
    audit = exc.value.planning_evidence['protocol']
    assert audit['citations'] and audit['pdf_sha256'] == packet['pdf']['sha256']
    assert 'metrics' not in audit and 'training' not in audit


def test_native_package_imports_follow_initializers_children_and_parent_relative_imports():
    texts = {'train.py': 'from network import layers\n',
             'network/__init__.py': 'import numpy\n',
             'network/layers.py': 'from . import core\nfrom .helpers import tensors\n',
             'network/core.py': 'import torch\n',
             'network/helpers/__init__.py': '',
             'network/helpers/tensors.py': 'from .. import core\nimport scipy\n'}
    packet = {'file_inventory': dict.fromkeys(texts), 'repository': {'files': {
        name: {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
        for name, text in texts.items()}}}
    result = planner._author_imports(packet, {'argv': ['python','train.py']}, include_sources=True)
    assert set(result['sources']) == set(texts)
    assert {item['module'] for item in result['imports']} == {'torch','numpy','scipy'}


@pytest.mark.parametrize('declaration', ["type=parse_integer", "action='append'", "nargs='+'"])
def test_argparse_semantics_that_cannot_be_compiled_are_explicitly_rejected(tmp_path, declaration):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    packet['repository']['files']['train.py']['text'] += (
        "\nparser.add_argument('--dynamic', default=4, " + declaration + ")\n")
    with pytest.raises(planner.PlanEvidenceError, match='unsupported dynamic/action') as exc:
        planner._effective_author_parameters(packet, fixture['proposal']['steps'][0])
    assert exc.value.category == 'runtime_unsupported'


def test_required_options_and_short_aliases_use_actual_argparse_destinations(tmp_path):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    packet['repository']['files']['train.py']['text'] += (
        "\nparser.add_argument('-d', '--dataset', type=str, required=True)\n"
        "parser.add_argument('-t', '--type', type=int, required=True)\n")
    step = deepcopy(fixture['proposal']['steps'][0])
    step['argv'].extend(['-d=author_data', '-t', '1'])
    values = planner._effective_author_parameters(packet, step)
    assert values['dataset']['value'] == 'author_data' and values['type']['value'] == 1
    planner._verify_author_arguments(['-d=author_data', '-t', '1'],
                                    'python train.py -d=author_data -t 1', packet)
    step['argv'] = step['argv'][:-2]
    with pytest.raises(planner.PlanEvidenceError, match='required option: type'):
        planner._effective_author_parameters(packet, step)


def test_unused_reviewed_packages_are_omitted_and_required_imports_are_hash_bound(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    proposal = deepcopy(fixture['proposal'])
    proposal['requirements']['compatibility'].extend(['pandas==2.2.3','torchvision==0.20.1+cpu'])
    packet['experiment_protocol'] = protocol
    result = planner.validate_plan(proposal, packet, fixture['workspace'])
    requirements = result['requirements']
    assert requirements['omitted_unused_proposals'] == ['pandas==2.2.3','torchvision==0.20.1+cpu']
    assert requirements['compatibility'] == ['numpy==1.26.4','torch==2.5.1+cpu']
    assert all(item['sha256'] == fixture['snapshot']['files'][item['path']] for item in requirements['import_evidence'])


def test_boolean_action_and_omitted_defaults_are_compiled_from_the_actual_parser(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    packet['experiment_protocol'] = protocol
    proposal = deepcopy(fixture['proposal'])
    proposal['parameters']['cpu'] = True
    proposal['parameter_citations']['cpu'] = ['readme']
    plan = planner.validate_plan(proposal, packet, fixture['workspace'])
    assert plan['parameters']['cpu'] is True
    refs = {item['id']:item for item in plan['citations']}
    assert any(refs[key].get('evidence_scope')=='effective_author_argument' for key in plan['parameter_citations']['cpu'])
    proposal['parameters']['seed'] = 8
    with pytest.raises(planner.PlanEvidenceError, match='actual author CLI/default: seed'):
        planner.validate_plan(proposal, packet, fixture['workspace'])


def test_unsupported_real_import_is_not_hidden_by_model_omission(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    path = fixture['workspace'] / 'author_model.py'
    path.write_text('import unsupported_graph_library\n' + path.read_text('utf-8'), encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['experiment_protocol'] = protocol
    citations = planner._citations(fixture['proposal'], packet, fixture['workspace'])
    with pytest.raises(planner.PlanEvidenceError, match='outside the reviewed runtime'):
        planner._requirements(fixture['proposal']['requirements'], citations, packet,
                              step=fixture['proposal']['steps'][0])
