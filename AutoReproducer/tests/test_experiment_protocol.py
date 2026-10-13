"""A protocol binds experiment selection before execution; unsupported stays failed."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from discovered_fixtures import PlannedLLM, accepted_review, fixture_protocol, make_experiment
from src.agents.experiment_protocol import ExperimentProtocolAgent, bind_plan_to_protocol, validate_experiment_protocol
from src import discovered_repository_plan as planner
from src.dependency_versions import compatibility_pins


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
    assert plan['requirements']['compatibility'] == ['numpy==1.26.4', 'torch==' + compatibility_pins()['torch']]
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


def test_author_training_defaults_do_not_require_duplicate_paper_details(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    kinds = planner._planning_view(packet)['evidence_excerpts']
    raw['training']['citations'] = [key for key in raw['training']['citations']
                                   if kinds[key]['source'] == 'repository']
    protocol = validate_experiment_protocol(raw, packet, fixture['workspace'])
    assert protocol['training']['citations'] == raw['training']['citations']
    assert protocol['effective_author_parameters']['epochs']['value'] == 40
    raw['training']['parameters'][0].update(name='epochs', value=7)
    with pytest.raises(planner.PlanEvidenceError, match='actual author command/default: epochs'):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


def test_training_still_requires_actual_repository_evidence(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    kinds = planner._planning_view(packet)['evidence_excerpts']
    raw['training']['citations'] = [key for key in raw['training']['citations']
                                   if kinds[key]['source'] == 'pdf']
    with pytest.raises(planner.PlanEvidenceError, match='repository'):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


@pytest.mark.parametrize('unit', ['runs', 'epochs', 'seeds', 'nodes', 'samples', 'classes'])
def test_pdf_extracted_count_joined_to_known_unit_is_numerical_evidence(unit):
    assert planner._has_value(100, [{'source': 'pdf', 'quote': 'We use 100' + unit + '.'}])
    assert not planner._has_value(100, [{'source': 'repository', 'quote': 'name = 100' + unit}])
    assert not planner._has_value(100, [{'source': 'pdf', 'quote': 'We use 1000' + unit + '.'}])
    assert not planner._has_value(100, [{'source': 'pdf', 'quote': 'config_100' + unit}])
    assert not planner._has_value(100, [{'source': 'pdf', 'quote': '100' + unit + '_file'}])


def test_protocol_reads_missing_pinned_experiment_sources_before_deciding(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    source = packet['repository']['files'].pop('train.py')
    calls = []

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            calls.append(prompt)
            evidence = json.loads(prompt.split('Protocol evidence:\n', 1)[1])
            if len(calls) == 1:
                assert not any(ref.get('path') == 'train.py' and ref['source'] == 'repository'
                               for ref in evidence['evidence_excerpts'].values())
                assert any(ref.get('path') == 'train.py' and ref['source'] == 'inventory'
                           for ref in evidence['evidence_excerpts'].values())
                return json.dumps({'request_files': ['train.py']})
            assert any(ref.get('path') == 'train.py' and ref['quote'] == source['text']
                for ref in evidence['evidence_excerpts'].values())
            return json.dumps(raw)

    protocol = ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert len(calls) == 2
    assert packet['repository']['files']['train.py'] == source
    assert any(ref.get('path') == 'train.py' for ref in protocol['citations'])


def test_protocol_rejection_after_source_reading_is_final(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    packet['repository']['files'].pop('train.py')
    calls = []

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            calls.append(prompt)
            return json.dumps({'request_files': ['train.py']} if len(calls) == 1 else {
                'version': 1, 'alignment': {**raw['alignment'], 'status': 'conflict',
                    'reason': 'The newly read implementation contradicts the paper training protocol.'}})

    with pytest.raises(planner.PlanEvidenceError) as exc:
        ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert exc.value.category == 'protocol_conflict'
    assert len(calls) == 2


@pytest.mark.parametrize('names', [['missing.py'], [['unhashable']]])
def test_invalid_source_requests_cannot_loop(tmp_path, names):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    original = deepcopy(packet)
    calls = []

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            calls.append(prompt)
            return json.dumps({'request_files': names})

    with pytest.raises(planner.PlanEvidenceError) as exc:
        ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert len(calls) == 2
    assert exc.value.planning_evidence['source_requests'] == 0
    assert packet == original


def test_repeated_read_is_idempotent_but_bounded_and_catalog_is_explicit(tmp_path):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    original = deepcopy(packet)
    calls = []

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            calls.append(prompt)
            evidence = json.loads(prompt.split('Protocol evidence:\n', 1)[1])
            assert evidence['source_catalog']['train.py']['already_read'] is True
            assert 'dataset.json' not in evidence['source_catalog']
            return json.dumps({'request_files': ['train.py']})

    with pytest.raises(planner.PlanEvidenceError) as exc:
        ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert len(calls) == 4
    assert exc.value.planning_evidence['source_requests'] == 2
    assert packet == original


def test_two_source_rounds_do_not_exhaust_two_split_summary_rounds(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    names = []
    for index in range(10):
        path = fixture['workspace'] / f'split_{index}.npz'
        np.savez(path, test_mask=np.array([0, 1], dtype=np.uint8))
        packet['file_inventory'][path.name] = {'bytes': path.stat().st_size, 'text_available': False,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        names.append(path.name)
    requests = [{'request_files': ['train.py']}, {'request_files': ['author_model.py']},
                {'request_files': names[:8]}, {'request_files': names[8:]}]

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            if len(requests) == 1:
                assert 'Do not request them again' in prompt
                assert 'remaining_dataset_rounds":1' in prompt
                assert 'split_7.npz' in prompt and 'data_' in prompt
            return json.dumps(requests.pop(0) if requests else raw)

    protocol = ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert protocol['alignment']['status'] == 'matched'
    assert all(packet['file_inventory'][name].get('dataset_evidence') for name in names)
    llm = PlannedLLM(fixture['proposal'], prefixes=[{'request_files': ['train.py']},
        {'request_files': ['author_model.py']}, {'request_files': names[:8]}, {'request_files': names[8:]}])
    plan = planner.propose_plan(llm, packet, fixture['workspace'])
    assert plan['semantic_review']['accepted'] is True and len(llm.calls) == 6


def test_mixed_requests_consume_both_budgets_and_cannot_loop(tmp_path):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    path = fixture['workspace'] / 'split.npz'
    np.savez(path, test_mask=np.array([0, 1], dtype=np.uint8))
    packet['file_inventory'][path.name] = {'bytes': path.stat().st_size, 'text_available': False,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    calls = []

    class Reader:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            calls.append(prompt)
            return json.dumps({'request_files': ['train.py', 'split.npz']})

    with pytest.raises(planner.PlanEvidenceError) as failure:
        ExperimentProtocolAgent(Reader()).run(packet, fixture['workspace'])
    assert len(calls) == 4
    assert failure.value.planning_evidence['text_requests'] == 2
    assert failure.value.planning_evidence['dataset_requests'] == 2


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


@pytest.mark.parametrize('name', ['epochs', 'invented_option'])
def test_protocol_parameter_must_match_actual_native_command_not_an_unrelated_source_number(tmp_path, name):
    fixture, packet, raw, actual_protocol = protocol_fixture(tmp_path)
    parameter = raw['training']['parameters'][0]
    parameter.update(name=name, value=7)  # Source has seed=7, but this command uses epochs=40.
    assert planner._has_value(7, planner._references(parameter['citations'],
        {item['id']: item for item in actual_protocol['citations']}))
    with pytest.raises(planner.PlanEvidenceError, match='actual author command/default: ' + name):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


def test_protocol_conflict_is_a_distinct_non_retryable_rejection(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    raw['alignment'].update(status='conflict', reason='Paper repetitions and native author entrypoint disagree.')
    with pytest.raises(planner.PlanEvidenceError, match='experiment conflict') as exc:
        validate_experiment_protocol(raw, packet, fixture['workspace'])
    assert exc.value.planning_evidence['stage'] == 'experiment_protocol'


@pytest.mark.parametrize('change', [
    lambda raw: raw.pop('model_selection'),
    lambda raw: raw['model_selection'].update(policy='best_test_accuracy'),
    lambda raw: raw['model_selection'].update(evaluation_weights=''),
    lambda raw: raw['model_selection'].update(citations=raw['metrics'][0]['citations']),
])
def test_model_selection_cannot_be_omitted_or_lack_actual_source_policy(tmp_path, change):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    change(raw)
    with pytest.raises(planner.PlanEvidenceError):
        validate_experiment_protocol(raw, packet, fixture['workspace'])


def test_checkpoint_policy_may_come_from_author_source_without_inventing_pdf_detail(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    raw['model_selection']['citations'] = raw['implementation']['citations']
    accepted = validate_experiment_protocol(raw, packet, fixture['workspace'])
    assert accepted['model_selection']['policy'] == 'last_epoch'
    assert accepted['alignment']['citations']  # Paper/repository alignment remains independently mandatory.


def test_best_validation_scalar_does_not_authorize_final_model_capture(tmp_path):
    fixture, packet, raw, _ = protocol_fixture(tmp_path)
    raw['model_selection'].update(policy='best_validation', criterion='Minimum validation loss.',
        evaluation_weights='Author reports a selected score but retains only final-epoch weights.')
    protocol = validate_experiment_protocol(raw, packet, fixture['workspace'])
    packet['experiment_protocol'] = protocol
    review = accepted_review()
    review['checks']['model_selection_and_evaluation_weights'].update(passed=False,
        reason='The final-epoch captured model does not retain the best-validation weights for the reported score.')
    llm = PlannedLLM(fixture['proposal'], review)
    with pytest.raises(planner.PlanEvidenceError, match='model_selection_and_evaluation_weights') as exc:
        planner.propose_plan(llm, packet, fixture['workspace'])
    assert exc.value.category == 'protocol_mismatch'
    assert len(llm.calls) == 2  # A real negative judgment never triggers replanning/tuning.
    assert 'matching model' in llm.calls[1][0] or 'corresponding retained/restored model' in llm.calls[1][0]


def test_model_selection_audit_can_use_source_details_with_separate_paper_context(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    packet['experiment_protocol'] = protocol
    review = accepted_review()
    review['checks']['model_selection_and_evaluation_weights']['citations'] = ['script']
    plan = planner.propose_plan(PlannedLLM(fixture['proposal'], review), packet, fixture['workspace'])
    assert plan['semantic_review']['accepted'] is True
    review['checks']['model_selection_and_evaluation_weights']['citations'] = ['paper_protocol']
    with pytest.raises(planner.PlanEvidenceError, match='original source evidence: model_selection'):
        planner.propose_plan(PlannedLLM(fixture['proposal'], review), packet, fixture['workspace'])


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


def test_only_exact_cuda_guard_is_pruned_for_frozen_cpu_with_source_audit():
    source = ('import torch\nif torch.cuda.is_available():\n    import cuda_graph_dependency\n'
              'else:\n    import numpy\nif unknown_condition:\n    import unknown_dependency\n')
    packet = {'file_inventory': {'train.py': {}}, 'repository': {'files': {
        'train.py': {'text': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}}}}
    step = {'argv': ['python', 'train.py']}
    cpu = planner._author_imports(packet, step, include_sources=True, device='cpu')
    assert {item['module'] for item in cpu['imports']} == {'torch', 'numpy', 'unknown_dependency'}
    excluded, = cpu['excluded_device_branches']
    assert 'import cuda_graph_dependency' in excluded['statement']
    assert excluded['sha256'] == packet['repository']['files']['train.py']['sha256']
    assert 'cuda_graph_dependency' in {item['module'] for item in planner._author_imports(packet, step, device='cuda')}
    assert 'cuda_graph_dependency' in {item['module'] for item in planner._author_imports(packet, step)}


@pytest.mark.parametrize('binding', [
    'torch = other', 'from other import torch', 'def use(torch):\n    pass',
    'torch.cuda.is_available = lambda: False', 'del torch', "exec('dynamic code')",
])
def test_shadowed_or_mutated_torch_guard_cannot_hide_a_dependency(binding):
    source = 'import torch\n' + binding + '\nif torch.cuda.is_available():\n    import unsupported_graph_library\n'
    packet = {'file_inventory': {'train.py': {}}, 'repository': {'files': {
        'train.py': {'text': source, 'sha256': 'a' * 64}}}}
    actual = planner._author_imports(packet, {'argv': ['python', 'train.py']}, device='cpu')
    assert 'unsupported_graph_library' in {item['module'] for item in actual}


def test_downloader_import_compiles_its_real_missing_runtime_dependency(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    packet['repository']['files']['train.py']['text'] += '\nfrom google_drive_downloader import GoogleDriveDownloader\n'
    citations = planner._citations(fixture['proposal'], packet, fixture['workspace'])
    result = planner._requirements(fixture['proposal']['requirements'], citations, packet,
        step=fixture['proposal']['steps'][0])
    assert {'googledrivedownloader==0.4', 'requests==2.32.3'} <= set(result['compatibility'])
    assert any(item.get('dependency_of') == 'googledrivedownloader' and item['package'] == 'requests'
        for item in result['import_evidence'])


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


def test_only_unambiguous_native_long_option_abbreviations_are_compiled(tmp_path):
    fixture, packet, _, _ = protocol_fixture(tmp_path)
    step = deepcopy(fixture['proposal']['steps'][0])
    step['argv'] = [value if value != '--epochs' else '--epoch' for value in step['argv']]
    assert planner._effective_author_parameters(packet, step)['epochs']['value'] == 40
    packet['repository']['files']['train.py']['text'] += "\nparser.add_argument('--epoch-file', default='')\n"
    with pytest.raises(planner.PlanEvidenceError, match='unambiguously resolve'):
        planner._effective_author_parameters(packet, step)


def test_unused_reviewed_packages_are_omitted_and_required_imports_are_hash_bound(tmp_path):
    fixture, packet, raw, protocol = protocol_fixture(tmp_path)
    proposal = deepcopy(fixture['proposal'])
    unused = ['pandas==2.2.3', 'torchvision==' + compatibility_pins()['torchvision']]
    proposal['requirements']['compatibility'].extend(unused)
    packet['experiment_protocol'] = protocol
    result = planner.validate_plan(proposal, packet, fixture['workspace'])
    requirements = result['requirements']
    assert requirements['omitted_unused_proposals'] == unused
    assert requirements['compatibility'] == ['numpy==1.26.4', 'torch==' + compatibility_pins()['torch']]
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
