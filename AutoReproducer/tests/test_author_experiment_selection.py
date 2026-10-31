"""Requested datasets cannot silently turn into a different author CLI experiment."""
from copy import deepcopy
import json

import pytest

from src.author_execution_plan import SCOPE, plan_from_author_commands, propose_author_execution_plan, validate_author_execution_plan
from src.author_experiment_selection import dataset_cli_selection
from src.discovered_repository_plan import PlanEvidenceError
from test_author_command_inference import TRAIN, packet_for


def setup(tmp_path, *, source=TRAIN, readme='Training is implemented in train.py.', dataset='PubMed'):
    packet = packet_for(tmp_path, {'README.md': readme, 'train.py': source})
    packet['experiment_request'] = {'dataset': dataset}
    proposal = {'version': 1, 'scope': SCOPE,
        'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
        'device': 'cpu', 'steps': [{'id': 'train', 'kind': 'train',
                                  'argv': ['python', 'train.py'], 'cwd': '.', 'timeout_s': None}]}
    return packet, proposal


@pytest.mark.parametrize('arguments,actual', [([], 'Cora'), (['--dataset', 'Cora'], 'Cora'),
    (['--dataset=Citeseer'], 'Citeseer'), (['--dataset', 'PubMed', '--dataset', 'Cora'], 'Cora')])
def test_known_dataset_conflict_is_rejected_before_freezing(tmp_path, arguments, actual):
    packet, proposal = setup(tmp_path)
    proposal['steps'][0]['argv'].extend(arguments)
    with pytest.raises(PlanEvidenceError, match='dataset differs from the user request') as error:
        validate_author_execution_plan(proposal, packet, tmp_path)
    evidence = error.value.planning_evidence
    assert evidence['stage'] == 'experiment_request_binding'
    assert evidence['observation']['value'] == actual
    assert evidence['observation']['source_sha256'] == packet['file_inventory']['train.py']['sha256']


@pytest.mark.parametrize('arguments', [['--dataset', 'pubmed'], ['--dataset=PubMed'],
    ['--dataset', 'Cora', '--dataset', 'PubMed'], ['--dataset', 'PubMed', '--epochs', '1000', '--seed', '42']])
def test_matching_cli_choice_preserves_original_argv_and_limits_claims(tmp_path, arguments):
    packet, proposal = setup(tmp_path)
    proposal['steps'][0]['argv'].extend(arguments)
    frozen = deepcopy(proposal)
    plan = validate_author_execution_plan(proposal, packet, tmp_path)
    assert proposal == frozen and plan['steps'][0]['argv'] == frozen['steps'][0]['argv']
    selection = plan['experiment_selection']
    assert selection['dataset_status'] == 'matched_cli_selection'
    assert selection['runtime_dataset_identity_verified'] is False
    assert selection['method_status'] == 'unverified'


def test_declared_alias_and_set_defaults_are_used(tmp_path):
    source = TRAIN.replace("'--dataset', default='Cora'", "'-d', '--data', dest='dataset', default='Cora'")
    packet, proposal = setup(tmp_path, source=source)
    proposal['steps'][0]['argv'].extend(['-d', 'PubMed'])
    assert validate_author_execution_plan(proposal, packet, tmp_path)['experiment_selection']['dataset_status'] == 'matched_cli_selection'
    source += "\n"  # set_defaults must precede parse_args in the original source.
    source = source.replace('args = parser.parse_args()', "parser.set_defaults(dataset='PubMed')\nargs = parser.parse_args()")
    packet, proposal = setup(tmp_path, source=source)
    assert validate_author_execution_plan(proposal, packet, tmp_path)['experiment_selection']['dataset_status'] == 'matched_cli_selection'


@pytest.mark.parametrize('extra,arguments', [('', ['--data', 'PubMed']),
    ("parser.add_argument('--label', nargs=argparse.REMAINDER)\n", ['--label', '--dataset', 'PubMed']),
    ("parser.add_argument('remainder', nargs='*')\n", ['--', '--dataset', 'PubMed'])])
def test_abbreviations_and_complex_consumption_stay_unverified(extra, arguments):
    source = TRAIN.replace('args = parser.parse_args()', extra + 'args = parser.parse_args()')
    assert dataset_cli_selection(source, arguments)['status'] == 'unverified'


@pytest.mark.parametrize('declaration', ["'--dataset', action='append'", "'--dataset', nargs='+'",
    "'--dataset', type=lambda name: name.lower()", "'--dataset', default=choose_dataset()"])
def test_dynamic_or_nonscalar_selector_stays_unverified(declaration):
    source = TRAIN.replace("'--dataset', default='Cora'", declaration)
    result = dataset_cli_selection(source, [] if 'choose_dataset' in declaration else ['--dataset', 'PubMed'])
    # An explicit literal with a custom converter is not proof of the final name.
    assert result['status'] == 'unverified'


def test_double_dash_does_not_change_observed_default():
    assert dataset_cli_selection(TRAIN, ['--', '--dataset', 'PubMed'])['value'] == 'Cora'


def test_literal_fallback_selects_matching_documented_command(tmp_path):
    packet, _ = setup(tmp_path, readme='python train.py --dataset Cora --epochs 1000\n'
        'python train.py --dataset PubMed --epochs 1000 --seed 42')
    plan = plan_from_author_commands(packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--dataset', 'PubMed', '--epochs', '1000', '--seed', '42']


def test_literal_fallback_prefers_observed_match_to_earlier_unknown_entrypoint(tmp_path):
    packet, _ = setup(tmp_path, readme='python unknown.py\npython train.py --dataset PubMed')
    raw = "print('No parser dataset evidence')\n"
    import hashlib
    (tmp_path / 'unknown.py').write_text(raw, encoding='utf-8')
    checksum = hashlib.sha256(raw.encode()).hexdigest()
    packet['file_inventory']['unknown.py'] = {'sha256': checksum, 'bytes': len(raw), 'text_available': True}
    packet['repository']['files']['unknown.py'] = {'sha256': checksum, 'text': raw}
    plan = plan_from_author_commands(packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--dataset', 'PubMed']


@pytest.mark.parametrize('variable', ['$data', '${data}'])
def test_literal_dataset_loop_selects_only_author_enumerated_request(tmp_path, variable):
    packet, _ = setup(tmp_path, readme='for seed in 42 43; do\nfor data in Cora Citeseer PubMed; do\n'
        'python train.py --dataset=' + variable + ' --seed $seed --epochs 1000\ndone\ndone')
    plan = plan_from_author_commands(packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--dataset=PubMed', '--seed', '42', '--epochs', '1000']
    assert plan['command_source']['template_bindings'] == {'seed': '42', 'data': 'PubMed'}
    packet['experiment_request'] = {'dataset': 'Undocumented'}
    with pytest.raises(PlanEvidenceError, match='dataset differs from the user request'):
        plan_from_author_commands(packet, tmp_path)


def test_dataset_loop_never_changes_a_shared_hyperparameter_variable(tmp_path):
    packet, _ = setup(tmp_path, readme='for value in Cora PubMed; do\n'
        'python train.py --dataset $value --seed $value\ndone')
    with pytest.raises(PlanEvidenceError, match='dataset differs from the user request'):
        plan_from_author_commands(packet, tmp_path)


def test_loop_does_not_rewrite_remainder_payload(tmp_path):
    from src.author_experiment_selection import select_dataset_loop_bindings
    source = TRAIN.replace('args = parser.parse_args()',
        "parser.add_argument('--rest', nargs=argparse.REMAINDER)\nargs = parser.parse_args()")
    packet, _ = setup(tmp_path, source=source)
    assert select_dataset_loop_bindings(['python', 'train.py', '--rest', '--dataset', '$value'],
        {'value': ['Cora', 'PubMed']}, packet, tmp_path, '.') == {'value': 'Cora'}


def test_model_receives_selection_error_and_replans_original_invocation(tmp_path):
    packet, proposal = setup(tmp_path)
    class LLM:
        mock_mode = False
        calls = []
        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            result = deepcopy(proposal)
            result['steps'][0]['argv'].extend(['--dataset', 'Cora' if len(self.calls) == 1 else 'PubMed'])
            return json.dumps(result)
    llm = LLM()
    plan = propose_author_execution_plan(llm, packet, tmp_path)
    assert len(llm.calls) == 2 and 'dataset differs from the user request' in llm.calls[1]
    assert plan['steps'][0]['argv'][-1] == 'PubMed'


def test_staged_teacher_method_is_not_confused_with_requested_method(tmp_path):
    source = TRAIN.replace('args = parser.parse_args()', "parser.add_argument('--model', default='MLP')\nargs = parser.parse_args()")
    packet, proposal = setup(tmp_path, source=source)
    packet['experiment_request']['method'] = 'Distillation'
    teacher = proposal['steps'][0]
    teacher['argv'].extend(['--dataset', 'PubMed', '--model', 'SAGE'])
    student = deepcopy(teacher)
    student.update(id='student', depends_on=['train'])
    student['argv'][-1] = 'MLP'
    proposal['steps'].append(student)
    plan = validate_author_execution_plan(proposal, packet, tmp_path)
    assert plan['experiment_selection']['dataset_status'] == 'matched_cli_selection'
    assert plan['experiment_selection']['method_status'] == 'unverified'
    student['argv'][3] = 'Cora'
    with pytest.raises(PlanEvidenceError, match='dataset differs from the user request'):
        validate_author_execution_plan(proposal, packet, tmp_path)


def test_source_fallback_prefers_requested_dataset_without_changing_defaults(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training lives in cora/train.py and pubmed/train.py.',
        'cora/train.py': TRAIN, 'pubmed/train.py': TRAIN.replace("default='Cora'", "default='PubMed'")})
    packet['experiment_request'] = {'dataset': 'PubMed'}
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['entrypoint']['source_path'] == 'pubmed/train.py'
    assert plan['steps'][0]['argv'] == ['python', 'train.py']


def test_no_selection_request_keeps_existing_execution_behavior(tmp_path):
    packet, proposal = setup(tmp_path)
    del packet['experiment_request']
    assert 'experiment_selection' not in validate_author_execution_plan(proposal, packet, tmp_path)


def test_shadowed_str_converter_is_not_treated_as_builtin():
    source = "str = lambda value: 'Cora'\n" + TRAIN.replace("default='Cora'", "type=str, default='Cora'")
    assert dataset_cli_selection(source, ['--dataset', 'PubMed'])['status'] == 'unverified'


def test_direct_imported_parser_is_bound_to_both_original_sources(tmp_path):
    entry = 'from options import original_args as get_args\nargs = get_args()\n'
    helper = ('import argparse\ndef original_args():\n'
        '    parser = argparse.ArgumentParser()\n'
        "    parser.add_argument('--dataset', default='Cora')\n"
        '    args, _ = parser.parse_known_args()\n    return args\n')
    packet = packet_for(tmp_path, {'README.md': 'python train.py --dataset PubMed',
                                  'train.py': entry, 'options.py': helper})
    packet['experiment_request'] = {'dataset': 'PubMed'}
    plan = plan_from_author_commands(packet, tmp_path)
    observed = plan['experiment_selection']['steps'][0]
    assert observed['status'] == 'matched_cli_selection'
    assert observed['source_path'] == 'options.py'
    assert observed['source_sha256'] == packet['file_inventory']['options.py']['sha256']
    assert observed['parser_call']['source_path'] == 'train.py'
    assert observed['parser_call']['source_sha256'] == packet['file_inventory']['train.py']['sha256']
    assert observed['parser_call']['quote'] == 'get_args()'
    packet['experiment_request']['dataset'] = 'Cora'
    with pytest.raises(PlanEvidenceError, match='dataset differs from the user request'):
        plan_from_author_commands(packet, tmp_path)


def test_loop_keeps_original_template_entrypoint(tmp_path):
    packet, _ = setup(tmp_path, readme='for script in train.py other.py; do\nfor data in Cora PubMed; do\n'
        'python $script --dataset $data --seed 42\ndone\ndone')
    plan = plan_from_author_commands(packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--dataset', 'PubMed', '--seed', '42']


def test_imported_helper_main_demo_is_not_parser_evidence(tmp_path):
    entry = 'from options import get_args\nargs = get_args()\n'
    helper = ("from argparse import Namespace\ndef get_args():\n    return Namespace(dataset='PubMed')\n"
        "if __name__ == '__main__':\n    import argparse\n    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--dataset', default='Cora')\n    p.parse_args()\n")
    packet = packet_for(tmp_path, {'README.md': 'python train.py', 'train.py': entry, 'options.py': helper})
    packet['experiment_request'] = {'dataset': 'PubMed'}
    assert plan_from_author_commands(packet, tmp_path)['experiment_selection']['dataset_status'] == 'unverified'


def test_absolute_module_import_does_not_guess_sibling_helper(tmp_path):
    helper = ('import argparse\ndef get_args():\n    p = argparse.ArgumentParser()\n'
              "    p.add_argument('--dataset', default='PubMed')\n    return p.parse_args()\n")
    packet = packet_for(tmp_path, {'README.md': 'python -m package.train',
        'package/train.py': 'from options import get_args\nargs = get_args()\n', 'package/options.py': helper})
    packet['experiment_request'] = {'dataset': 'PubMed'}
    assert plan_from_author_commands(packet, tmp_path)['experiment_selection']['dataset_status'] == 'unverified'


def test_absolute_module_import_uses_author_cwd_instead_of_root_decoy(tmp_path):
    helper = ('import argparse\ndef get_args():\n    p = argparse.ArgumentParser()\n'
              "    p.add_argument('--dataset', default='PubMed')\n    return p.parse_args()\n")
    packet = packet_for(tmp_path, {'README.md': 'cd src\npython -m package.train',
        'src/package/train.py': 'from options import get_args\nargs = get_args()\n',
        'options.py': helper.replace('PubMed', 'Cora'), 'src/options.py': helper})
    packet['experiment_request'] = {'dataset': 'PubMed'}
    plan = plan_from_author_commands(packet, tmp_path)
    assert plan['experiment_selection']['dataset_status'] == 'matched_cli_selection'
    assert plan['experiment_selection']['steps'][0]['source_path'] == 'src/options.py'
