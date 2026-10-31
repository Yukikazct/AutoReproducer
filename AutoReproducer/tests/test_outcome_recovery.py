"""A numeric discrepancy can restore one independently documented omission."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from src.author_execution_plan import SCOPE, validate_author_execution_plan
from src.outcome_recovery import prepare_outcome_recovery
from test_author_command_inference import packet_for


SOURCE = ("import argparse\np=argparse.ArgumentParser()\n"
    "p.add_argument('--dataset',default='pubmed')\n"
    "p.add_argument('--lr',type=float,default=0.156)\n"
    "p.add_argument('--weight_decay',type=float,default=3.544e-5)\n"
    "p.add_argument('--T',type=float,default=5.27)\n"
    "p.add_argument('--K',type=int,default=250)\n"
    "p.add_argument('--seed',type=int,default=42)\n"
    "p.add_argument('--model',default='DGC')\n"
    "args=p.parse_args()\n")
ORIGINAL = ['python', 'main.py', '--dataset', 'pubmed', '--T', '6.0498', '--K', '900']
FULL = 'python main.py --dataset pubmed --T 6.0498 --K 900 --lr 0.627 --weight_decay 9.38e-6'
README = 'To reproduce the results above, run ```sh run_citation.sh```. Report 10 trials.\n'


def case(tmp_path, *, readme=README, launcher=FULL, argv=None, source=SOURCE, extra_sources=None):
    packet = packet_for(tmp_path, {**(extra_sources or {}), 'README.md': readme, 'run_citation.sh': launcher, 'main.py': source})
    packet['experiment_request'] = {'method': 'DGC', 'dataset': 'pubmed'}
    payload = {'version': 1, 'scope': SCOPE,
        'repository': {key: packet['repository'][key] for key in ('url', 'revision')}, 'device': 'cpu',
        'steps': [{'id': 'train', 'kind': 'train', 'argv': argv or list(ORIGINAL), 'cwd': '.', 'timeout_s': None}]}
    plan = validate_author_execution_plan(payload, packet, tmp_path)
    execution = {'success': True, 'executed': True, 'cleanup_confirmed': True,
        'steps': [{'id': 'train', 'success': True, 'executed': True, 'exit_code': 0}],
        'final': {'success': True, 'exit_code': 0}}
    validation = {'status': 'ai_not_reproduced', 'execution_status': 'author_code_completed',
        'completion_checks': {'owned_processes_finished': True, 'all_required_author_commands_returned': True},
        'author_execution': [{'step': deepcopy(plan['steps'][0]), 'natural_return': True}],
        'reproduction_assessment': {'status': 'completed', 'source': 'ai', 'verdict': 'not_reproduced',
            'metric_assessments': [{'actual_id': 'train:stdout:0', 'verdict': 'not_reproduced'}]},
        'author_claim_comparison': {'comparisons': [{'actual': {'id': 'train:stdout:0', 'value': .6,
            'method': 'DGC', 'dataset': 'pubmed', 'split': 'test'},
            'reference': {'value': .9, 'method': 'DGC', 'dataset': 'pubmed', 'split': 'test'},
            'status': 'comparable', 'difference': -.3,
            'meets_reference': False, 'within_tolerance': False}]}}
    return plan, packet, tmp_path, execution, validation, {'source_repair': True, 'source_repair_auto': True}


def test_exact_author_omission_proposes_new_frozen_plan_and_preserves_baseline(tmp_path):
    args = case(tmp_path)
    before = deepcopy(args)
    files = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'proposed', decision
    assert decision['plan']['steps'][0]['argv'] == FULL.split()
    assert decision['original_commands'] == args[0]['steps']
    assert decision['selection_basis'] == 'author_documentation_only'
    assert decision['tolerance_changed'] is False and decision['max_attempts'] == 1
    assert decision['evidence'][0]['added_options'] == ['lr', 'weight_decay']
    assert decision['evidence'][0]['command']['designation']['quote'] == README.strip()
    assert args == before
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == files


def test_all_documented_dataset_overrides_may_be_restored_as_one_fixed_command(tmp_path):
    initial = ['python', 'main.py', '--dataset', 'pubmed']
    decision = prepare_outcome_recovery(*case(tmp_path, argv=initial))
    assert decision['status'] == 'proposed', decision
    assert decision['original_commands'][0]['argv'] == initial
    assert decision['plan']['steps'][0]['argv'] == FULL.split()
    assert decision['evidence'][0]['added_options'] == ['K', 'T', 'lr', 'weight_decay']


@pytest.mark.parametrize('patch', [
    {'cancelled': True}, {'timed_out': True}, {'cleanup_confirmed': False},
    {'final': {'success': True, 'exit_code': 0, 'cleanup_receipt': {'status': 'unconfirmed'}}},
    {'effective_env_config': {'process_cleanup_confirmed': False}}, {'success': False}])
def test_interruption_or_nested_cleanup_uncertainty_never_retrains(tmp_path, patch):
    args = case(tmp_path)
    args[3].update(patch)
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


def test_missing_positive_completion_evidence_is_blocking(tmp_path):
    args = case(tmp_path)
    args[4]['completion_checks'].pop('owned_processes_finished')
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


@pytest.mark.parametrize('verdict', ['reproduced', 'insufficient_evidence'])
def test_existing_ai_judgement_is_not_overruled_by_fixed_five_percent_rule(tmp_path, verdict):
    args = case(tmp_path)
    args[4]['reproduction_assessment']['verdict'] = verdict
    assert prepare_outcome_recovery(*args)['status'] == 'not_needed'


@pytest.mark.parametrize('status', ['reference_only', 'not_comparable'])
def test_reference_or_pairing_uncertainty_alone_never_retrains(tmp_path, status):
    args = case(tmp_path)
    args[4]['author_claim_comparison']['comparisons'][0]['status'] = status
    args[4]['reproduction_assessment']['verdict'] = 'insufficient_evidence'
    assert prepare_outcome_recovery(*args)['status'] == 'not_needed'


def test_negative_paired_assessment_with_author_setting_difference_can_restore_omission(tmp_path):
    args = case(tmp_path)
    args[4]['author_claim_comparison']['comparisons'][0].update(status='reference_only',
        limitations=['The initial command omitted the author Pubmed optimizer settings.'],
        within_tolerance=None, meets_reference=None)
    assert prepare_outcome_recovery(*args)['status'] == 'proposed'


@pytest.mark.parametrize('direction,gap', [('higher', -.3), ('lower', .3)])
def test_partial_adverse_result_with_different_settings_can_restore_a_proven_omission(tmp_path, direction, gap):
    args = case(tmp_path)
    assessment = args[4]['reproduction_assessment']
    assessment.update(verdict='partially_reproduced')
    assessment['metric_assessments'][0]['verdict'] = 'partially_reproduced'
    row = args[4]['author_claim_comparison']['comparisons'][0]
    row.update(status='reference_only', direction=direction, difference=gap,
               settings={'status': 'different'}, within_tolerance=True, meets_reference=None)
    before = deepcopy(args[4])
    assert prepare_outcome_recovery(*args)['status'] == 'proposed'
    assert args[4] == before  # The model verdict/tolerance is never rewritten.


@pytest.mark.parametrize('gap,direction,settings,metric_verdict', [
    (0, 'higher', 'different', 'partially_reproduced'),
    (.3, 'higher', 'different', 'partially_reproduced'),
    (-.3, 'lower', 'different', 'partially_reproduced'),
    (-.3, 'unknown', 'different', 'partially_reproduced'),
    (-.3, 'higher', 'unknown', 'partially_reproduced'),
    (-.3, 'higher', 'matched', 'partially_reproduced'),
    (-.3, 'higher', 'different', 'reproduced'),
    (-.3, 'higher', 'different', 'insufficient_evidence')])
def test_partial_or_evidence_limit_alone_never_authorizes_retraining(tmp_path, gap, direction, settings, metric_verdict):
    args = case(tmp_path)
    args[4]['reproduction_assessment']['verdict'] = 'partially_reproduced'
    args[4]['reproduction_assessment']['metric_assessments'][0]['verdict'] = metric_verdict
    args[4]['author_claim_comparison']['comparisons'][0].update(
        direction=direction, difference=gap, settings={'status': settings})
    assert prepare_outcome_recovery(*args)['status'] == 'not_needed'


@pytest.mark.parametrize('field', ['method', 'dataset', 'split'])
def test_unpaired_reference_cannot_retrain_even_with_bad_ai_verdict(tmp_path, field):
    args = case(tmp_path)
    args[4]['author_claim_comparison']['comparisons'][0]['reference'][field] = 'unknown'
    assert prepare_outcome_recovery(*args)['status'] == 'not_needed'


@pytest.mark.parametrize('flag', ['source_repair', 'source_repair_auto'])
def test_disabled_auto_repair_never_discovers_candidates(tmp_path, monkeypatch, flag):
    args = case(tmp_path)
    args[5][flag] = False
    monkeypatch.setattr('src.outcome_recovery._designated_sources', lambda *a: pytest.fail('Disabled recovery read source'))
    assert prepare_outcome_recovery(*args)['status'] == 'not_requested'


def test_preexisting_fixed_comparison_is_used_only_without_completed_ai_judgement(tmp_path):
    args = case(tmp_path)
    args[4].pop('reproduction_assessment')
    assert prepare_outcome_recovery(*args)['status'] == 'proposed'


@pytest.mark.parametrize('kind', ['strict', 'patched', 'budget'])
def test_strict_patch_or_used_budget_stays_unresolved(tmp_path, kind):
    args = case(tmp_path)
    if kind == 'strict':
        args[5]['strict_validation'] = True
    elif kind == 'patched':
        args[4]['source_modified'] = True
    else:
        args[5]['_outcome_recovery_attempt'] = 1
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


@pytest.mark.parametrize('readme', ['## Usage\nsh run_citation.sh', 'To reproduce paper results.\n'])
def test_usage_command_or_missing_designated_command_is_not_a_recovery_contract(tmp_path, readme):
    args = case(tmp_path, readme=readme)
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


@pytest.mark.parametrize('launcher', [FULL+' --seed 7', FULL+' --model Different',
    FULL.replace('--K 900', '--K 901'), FULL+' --K 900', FULL.replace('pubmed', 'cora')])
def test_seed_method_explicit_setting_and_duplicate_override_are_rejected(tmp_path, launcher):
    assert prepare_outcome_recovery(*case(tmp_path, launcher=launcher))['status'] == 'unresolved'


def test_author_sweep_is_ambiguous_and_not_ranked_by_actual_metrics(tmp_path):
    args = case(tmp_path, launcher=FULL+'\n'+FULL.replace('0.627', '0.999'))
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'unresolved' and decision['candidate_count'] == 2


def test_already_full_command_or_redundant_defaults_do_not_cause_resampling(tmp_path):
    assert prepare_outcome_recovery(*case(tmp_path, argv=FULL.split()))['status'] == 'unresolved'
    assert prepare_outcome_recovery(*case(tmp_path, launcher=' '.join(ORIGINAL)+' --lr 0.156'))['status'] == 'unresolved'


@pytest.mark.parametrize('flag,dest', [('--no-cuda', 'no_cuda'), ('--cpu', 'cpu')])
def test_original_boolean_cpu_switch_survives_the_documented_full_command(tmp_path, flag, dest):
    source = SOURCE.replace('args=p.parse_args()',
        f"p.add_argument('{flag}', action='store_true', default=False)\nargs=p.parse_args()")
    args = case(tmp_path, source=source, argv=ORIGINAL + [flag])
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'proposed', decision
    assert decision['plan']['steps'][0]['argv'] == FULL.split() + [flag]
    assert decision['plan']['command_source']['quote'] == FULL
    assert decision['plan']['command_source']['retained_original_cpu_flags'] == [flag]
    assert decision['evidence'][0]['added_options'] == ['lr', 'weight_decay']


@pytest.mark.parametrize('default,device,flag', [(True, 'cpu', '--no-cuda'),
    (False, 'cuda', '--no-cuda'), (False, 'cpu', '--use-cpu')])
def test_device_extension_needs_the_original_false_default_cpu_boolean(tmp_path, default, device, flag):
    source = SOURCE.replace('args=p.parse_args()',
        f"p.add_argument('{flag}', action='store_true', default={default})\nargs=p.parse_args()")
    args = case(tmp_path, source=source, argv=ORIGINAL + [flag])
    args[0]['capture']['device'] = device
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


def test_conflicting_cpu_dest_declarations_never_extend_a_documented_command(tmp_path):
    source = SOURCE.replace('args=p.parse_args()',
        "p.add_argument('--no-cuda', action='store_true', default=False)\n"
        "p.add_argument('--disable-gpu', dest='no_cuda', action='store_true', default=False)\nargs=p.parse_args()")
    args = case(tmp_path, source=source, argv=ORIGINAL + ['--no-cuda'])
    assert prepare_outcome_recovery(*args)['status'] == 'unresolved'


@pytest.mark.parametrize('source', [SOURCE.replace("p.add_argument('--lr',type=float,default=0.156)",
    "rate=0.156\np.add_argument('--lr',type=float,default=rate)\nrate=0.627"),
    SOURCE.replace('args=p.parse_args()', "p.set_defaults(lr=0.627)\nargs=p.parse_args()")])
def test_ambiguous_parser_default_timing_never_authorizes_resampling(tmp_path, source):
    decision = prepare_outcome_recovery(*case(tmp_path, source=source))
    assert decision['status'] == 'unresolved'


def test_changed_source_cannot_authorize_a_candidate(tmp_path):
    args = case(tmp_path)
    (tmp_path/'run_citation.sh').write_text(FULL+' --seed 7', encoding='utf-8')
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'unresolved' and 'verified' in decision['reason']


@pytest.mark.parametrize('field', ['repository', 'pdf', 'evidence'])
def test_recovery_cannot_bind_to_a_different_frozen_input(tmp_path, field):
    args = case(tmp_path)
    if field == 'repository':
        args[1]['repository']['revision'] = 'f' * 40
    elif field == 'pdf':
        args[1]['pdf']['sha256'] = 'f' * 64
    else:
        args[1]['unfrozen_change'] = True
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'unresolved' and 'original frozen' in decision['reason']


def test_fixed_author_loop_is_selected_by_original_tokens_not_test_scores(tmp_path):
    launcher = ('for setting in tran ind; do\nfor data in cora pubmed; do\n'
        'python main.py --dataset $data --T 6.0498 --K 900 --setting $setting --lr 0.627\ndone\ndone')
    source = SOURCE.replace('args=p.parse_args()', "p.add_argument('--setting', default='tran')\nargs=p.parse_args()")
    args = case(tmp_path, source=source, launcher=launcher, argv=ORIGINAL+['--setting', 'tran'])
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'proposed', decision
    assert decision['plan']['steps'][0]['argv'][-3:] == ['tran', '--lr', '0.627']
    assert decision['plan']['command_source']['template_bindings'] == {'data': 'pubmed', 'setting': 'tran'}


@pytest.mark.parametrize('cpu_flags', [[], ['--no-cuda']])
def test_cached_real_dgc_command_has_an_independent_author_omission_contract(tmp_path, cpu_flags):
    cache = Path(__file__).parents[1]/'data/repository_cache/1d7ee9e7c310371a/f91b35a0d4388660e250d575d4c1b8fef41d7039'
    if not cache.exists():
        pytest.skip('Optional fixed author cache is not installed')
    # Parse actual author sources without importing or training their model.
    args = case(tmp_path, readme=(cache/'README.md').read_text('utf-8'),
                launcher=(cache/'run_citation.sh').read_text('utf-8'),
                source=(cache/'main.py').read_text('utf-8'),
                argv=ORIGINAL + cpu_flags,
                extra_sources={name: (cache/name).read_text('utf-8')
                               for name in ('utils.py', 'dgc.py', 'normalization.py')})
    decision = prepare_outcome_recovery(*args)
    assert decision['status'] == 'proposed', decision
    assert decision['plan']['steps'][0]['argv'] == FULL.split() + cpu_flags
    command = decision['evidence'][0]['command']
    assert command['path'] == 'run_citation.sh' and command['line'] == 3
    assert command['sha256'] == hashlib.sha256((tmp_path/'run_citation.sh').read_bytes()).hexdigest()
