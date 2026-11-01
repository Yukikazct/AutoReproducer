"""Freeze analysis inputs from each executed argv, never a documented launcher."""
from copy import deepcopy
import hashlib

import pytest

from src.author_comparison_context import frozen_cli_inputs


SOURCE = (
    "import argparse\n"
    "parser = argparse.ArgumentParser()\n"
    "parser.add_argument('--dataset', default='pubmed')\n"
    "parser.add_argument('--lr', '--learning-rate', type=float, default=0.156)\n"
    "parser.add_argument('--weight_decay', type=float, default=3.544e-5)\n"
    "parser.add_argument('--epochs', type=int, default=100)\n"
    "parser.add_argument('--trials', type=int, default=10)\n"
    "parser.add_argument('--no-cuda', action='store_true', default=False)\n"
    "parser.add_argument('--keep-cache', action='store_false', default=True)\n"
    "args = parser.parse_args()\n"
)
FULL = 'python main.py --dataset pubmed --lr 0.627 --weight_decay 9.38e-6 --no-cuda'


def case(directory, *, source=SOURCE, arguments=None, extra_sources=None, entry='main.py'):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {'README.md': 'To reproduce paper results, run run_citation.sh.\n',
               'run_citation.sh': FULL + '\n', entry: source, **(extra_sources or {})}
    inventory, selected = {}, {}
    for name, text in sources.items():
        raw = text.encode('utf-8')
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
        inventory[name] = {'sha256': checksum, 'bytes': len(raw), 'text_available': True}
        selected[name] = {'sha256': checksum, 'bytes': len(raw), 'text': text}
    packet = {'version': 1, 'repository': {'url': 'https://github.com/fixtures/frozen-cli',
              'revision': 'a' * 40, 'files': selected}, 'file_inventory': inventory,
              'limits': {'source_bytes': 1000000, 'source_files': 64}}
    argv = ['python', entry] + (['--dataset', 'pubmed', '--no-cuda'] if arguments is None else arguments)
    plan = {'scope': 'author_code_execution',
            'steps': [{'id': 'train', 'kind': 'train', 'argv': argv, 'cwd': '.',
                       'entrypoint': {'kind': 'script', 'target': entry, 'source_path': entry}}],
            'command_source': {'path': 'run_citation.sh', 'quote': FULL,
                               'sha256': inventory['run_citation.sh']['sha256']}}
    return packet, plan


def unavailable(result):
    assert result['status'] == 'unavailable'
    assert result['effective_runtime_verified'] is False
    assert len(result['steps']) == 1
    assert result['steps'][0]['status'] == 'unavailable'
    assert result['steps'][0]['values'] == {}


def test_omitted_optimizer_values_come_from_parser_instead_of_full_documented_command(tmp_path):
    source = "raise RuntimeError('Author code must never execute for this summary')\n" + SOURCE
    packet, plan = case(tmp_path, source=source)
    before = deepcopy((packet, plan))

    result = frozen_cli_inputs(packet, plan, tmp_path)

    assert result['version'] == 1 and result['status'] == 'observed'
    assert result['effective_runtime_verified'] is False
    step = result['steps'][0]
    assert step['step_id'] == 'train' and step['argv'] == plan['steps'][0]['argv']
    assert step['cwd'] == '.' and step['entrypoint'] == plan['steps'][0]['entrypoint']
    assert step['parser']['source_path'] == 'main.py'
    assert step['parser']['source_sha256'] == packet['file_inventory']['main.py']['sha256']
    assert step['values']['lr']['value'] == 0.156
    assert step['values']['weight_decay']['value'] == 3.544e-5
    assert step['values']['epochs']['value'] == 100 and step['values']['trials']['value'] == 10
    for name in ('lr', 'weight_decay', 'epochs', 'trials'):
        value = step['values'][name]
        assert value['origin'] == 'parser_default' and value['explicit_tokens'] == []
        assert value['declaration']['line'] > 0 and 'add_argument' in value['declaration']['quote']
        assert value['default_evidence']
    assert step['values']['dataset']['value'] == 'pubmed'
    assert step['values']['dataset']['origin'] == 'explicit'
    assert step['values']['no_cuda']['value'] is True
    assert step['values']['keep_cache']['value'] is True
    assert (packet, plan) == before
    assert not list(tmp_path.glob('*.pyc'))


def test_explicit_alias_equals_and_boolean_flags_preserve_original_input_values(tmp_path):
    arguments = ['--dataset', 'pubmed', '--learning-rate=0.627', '--weight_decay', '9.38e-6',
                 '--epochs', '200', '--no-cuda', '--keep-cache']
    packet, plan = case(tmp_path, arguments=arguments)

    result = frozen_cli_inputs(packet, plan, tmp_path)

    assert result['status'] == 'observed'
    values = result['steps'][0]['values']
    assert values['lr']['value'] == '0.627' and values['lr']['origin'] == 'explicit'
    assert values['lr']['explicit_tokens'] == ['--learning-rate=0.627']
    assert set(values['lr']['flags']) == {'--lr', '--learning-rate'}
    assert values['weight_decay']['value'] == '9.38e-6'
    assert values['weight_decay']['explicit_tokens'] == ['--weight_decay', '9.38e-6']
    assert values['epochs']['value'] == '200'
    assert values['no_cuda']['value'] is True and values['keep_cache']['value'] is False
    assert values['keep_cache']['origin'] == 'explicit'


@pytest.mark.parametrize('arguments', [
    ['--lr', '0.2', '--lr', '0.3'],
    ['--lr', '0.2', '--learning-rate', '0.3'],
    ['--lr=0.2', '--learning-rate=0.2'],
    ['--no-cuda', '--no-cuda'],
])
def test_repeated_destinations_including_aliases_never_produce_a_single_claimed_value(tmp_path, arguments):
    packet, plan = case(tmp_path, arguments=arguments)
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


def test_duplicate_declarations_for_one_destination_are_not_collapsed(tmp_path):
    source = SOURCE.replace('args = parser.parse_args()',
        "parser.add_argument('--optimizer-lr', dest='lr', default=0.9)\nargs = parser.parse_args()")
    packet, plan = case(tmp_path, source=source)
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


@pytest.mark.parametrize('source,arguments', [
    (SOURCE.replace("type=float, default=0.156", "type=float, nargs='+', default=0.156"), ['--lr', '0.2']),
    (SOURCE.replace('args = parser.parse_args()', 'configure(parser)\nargs = parser.parse_args()'), []),
    (SOURCE.replace('type=float, default=0.156', 'type=convert, default=0.156'), ['--lr', '0.2']),
    (SOURCE.replace('type=float, default=0.156', 'type=lambda value: float(value) * 2, default=0.156'), ['--lr', '0.2']),
    ("float = lambda value: 0.9\n" + SOURCE.replace('default=0.156', "default='0.156'"), []),
    (SOURCE, ['--undocumented', 'value']),
    (SOURCE, ['--learn', '0.2']),
    (SOURCE, ['--lr']),
    (SOURCE, ['--no-cuda=true']),
    (SOURCE + 'other = argparse.ArgumentParser()\nother.parse_args()\n', []),
])
def test_ambiguous_dynamic_or_unsupported_cli_semantics_clear_the_entire_step(tmp_path, source, arguments):
    packet, plan = case(tmp_path, source=source, arguments=arguments)
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


def test_nonliteral_default_is_unavailable_and_does_not_borrow_launcher_value(tmp_path):
    source = SOURCE.replace('default=0.156', 'default=choose_lr()')
    packet, plan = case(tmp_path, source=source)

    result = frozen_cli_inputs(packet, plan, tmp_path)

    unavailable(result)


def test_explicit_value_does_not_authorize_a_parser_with_nonliteral_default_expressions(tmp_path):
    source = SOURCE.replace('default=0.156', 'default=choose_lr()')
    packet, plan = case(tmp_path, source=source, arguments=['--lr', '0.31'])

    result = frozen_cli_inputs(packet, plan, tmp_path)

    unavailable(result)


@pytest.mark.parametrize('reassigned', [False, True])
def test_default_name_is_not_resolved_using_parser_bindings(tmp_path, reassigned):
    source = 'LR = 0.1\n' + SOURCE.replace('default=0.156', 'default=LR').replace(
        'args = parser.parse_args()', ('LR = 0.9\n' if reassigned else '') + 'args = parser.parse_args()')
    packet, plan = case(tmp_path, source=source)
    # This invented argparse program freezes LR at add_argument time, not parse time.
    namespace = {}
    exec(source.replace('parser.parse_args()', 'parser.parse_args([])'), namespace)
    assert namespace['args'].lr == 0.1
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


@pytest.mark.parametrize('position,expected', [('before', 0.1), ('after', 0.9)])
def test_set_defaults_order_is_not_guessed_from_a_final_parser_snapshot(tmp_path, position, expected):
    declaration = "parser.add_argument('--lr', '--learning-rate', type=float, default=0.1)"
    mutation = 'parser.set_defaults(lr=0.9)'
    replacement = mutation + '\n' + declaration if position == 'before' else declaration + '\n' + mutation
    source = SOURCE.replace("parser.add_argument('--lr', '--learning-rate', type=float, default=0.156)", replacement)
    packet, plan = case(tmp_path, source=source)
    namespace = {}
    exec(source.replace('parser.parse_args()', 'parser.parse_args([])'), namespace)
    assert namespace['args'].lr == expected
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


def test_changed_entrypoint_bytes_cannot_support_frozen_default_values(tmp_path):
    packet, plan = case(tmp_path)
    (tmp_path / 'main.py').write_text(SOURCE.replace('default=0.156', 'default=0.9'), encoding='utf-8')
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


def imported_case(directory):
    entry = 'from options import original_args as get_args\nargs = get_args()\n'
    helper = ('import argparse\ndef original_args():\n'
              '    parser = argparse.ArgumentParser()\n'
              "    parser.add_argument('--lr', type=float, default=0.156)\n"
              '    return parser.parse_args()\n')
    packet, plan = case(directory, source=entry, arguments=[], extra_sources={'options.py': helper})
    # Loading a helper must affect only the analyzer's private packet copy.
    packet['repository']['files'].pop('options.py')
    return packet, plan


def test_imported_zero_argument_parser_keeps_original_hash_and_line_evidence_without_mutation(tmp_path):
    packet, plan = imported_case(tmp_path)
    before = deepcopy((packet, plan))

    result = frozen_cli_inputs(packet, plan, tmp_path)

    assert result['status'] == 'observed'
    step = result['steps'][0]
    assert step['parser']['source_path'] == 'options.py'
    assert step['parser']['source_sha256'] == packet['file_inventory']['options.py']['sha256']
    assert step['values']['lr']['value'] == 0.156
    assert step['values']['lr']['declaration']['line'] == 4
    assert (packet, plan) == before
    assert 'options.py' not in packet['repository']['files']


@pytest.mark.parametrize('changed_source', ['main.py', 'options.py'])
def test_imported_parser_rejects_a_changed_entrypoint_or_helper(tmp_path, changed_source):
    packet, plan = imported_case(tmp_path)
    path = tmp_path / changed_source
    path.write_bytes(path.read_bytes() + b'\n# changed after freeze\n')
    unavailable(frozen_cli_inputs(packet, plan, tmp_path))


def test_recovery_child_uses_its_own_argv_and_leaves_original_attempt_unchanged(tmp_path):
    baseline, child = tmp_path / 'baseline', tmp_path / 'outcome_recovery' / 'attempt_1' / 'repo'
    packet, plan = case(baseline)
    child_packet, child_plan = case(child, arguments=FULL.split()[2:])
    before = deepcopy((packet, plan, child_packet, child_plan))

    original = frozen_cli_inputs(packet, plan, baseline)
    corrected = frozen_cli_inputs(child_packet, child_plan, child)

    assert original['steps'][0]['values']['lr']['value'] == 0.156
    assert original['steps'][0]['values']['lr']['origin'] == 'parser_default'
    assert corrected['steps'][0]['values']['lr']['value'] == '0.627'
    assert corrected['steps'][0]['values']['weight_decay']['value'] == '9.38e-6'
    assert corrected['steps'][0]['values']['lr']['origin'] == 'explicit'
    assert corrected['steps'][0]['argv'] != original['steps'][0]['argv']
    assert (packet, plan, child_packet, child_plan) == before
