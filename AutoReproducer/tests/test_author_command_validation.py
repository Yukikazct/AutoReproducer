"""Reject provably incomplete commands without guessing experiment parameters."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.author_command_validation import completes_required_options, missing_required_options
from src.author_execution_plan import propose_author_execution_plan, validate_author_execution_plan
from src.discovered_repository_plan import PlanEvidenceError, build_evidence_packet
from test_author_execution_plan import execution_proposal
from discovered_fixtures import make_experiment


@pytest.mark.parametrize('declaration', [
    'import argparse\np = argparse.ArgumentParser()',
    'import argparse as cli\np = cli.ArgumentParser()',
    'from argparse import ArgumentParser as Parser\np = Parser()',
])
@pytest.mark.parametrize('option', ['--corpus', '--configuration', '--input-file'])
def test_missing_required_option_is_detected_for_different_parser_and_flag_names(declaration, option):
    source = declaration + f'\np.add_argument("{option}", required=True)\nargs = p.parse_args()\n'
    assert missing_required_options(source, []) == [option]
    assert missing_required_options(source, [option, 'author_value']) == []
    assert missing_required_options(source, [option + '=author_value']) == []


def test_aliases_and_unambiguous_abbreviations_follow_argparse_semantics():
    source = ('import argparse\np = argparse.ArgumentParser()\n'
              'p.add_argument("-c", "--corpus", required=True)\n'
              'p.add_argument("--count", default=5)\nargs = p.parse_args()\n')
    assert missing_required_options(source, ['-c', 'toy']) == []
    assert missing_required_options(source, ['--corp=toy']) == []
    assert missing_required_options(source, ['--', '--corpus', 'toy']) == ['-c/--corpus']
    assert missing_required_options(source.replace('ArgumentParser()', 'ArgumentParser(allow_abbrev=False)'),
                                    ['--corp=toy']) == ['-c/--corpus']


@pytest.mark.parametrize('change', [
    lambda s: s.replace('p.parse_args()', 'p.parse_args(["--corpus", "original"])'),
    lambda s: s.replace('args = p.parse_args()', 'configure(p)\nargs = p.parse_args()'),
    lambda s: s.replace('args = p.parse_args()', 'config.configure(p)\nargs = p.parse_args()'),
    lambda s: s.replace('args = p.parse_args()', 'p.parse_args = custom_parse\nargs = p.parse_args()'),
    lambda s: s.replace('args = p.parse_args()', 'alias = p\nalias.set_defaults()\nargs = p.parse_args()'),
    lambda s: s.replace('args = p.parse_args()', 'p = custom_parser()\nargs = p.parse_args()'),
    lambda s: s.replace('import argparse', 'import unknown_library as argparse'),
    lambda s: s.replace('required=True', 'required=condition'),
    lambda s: 'def unrelated():\n' + '\n'.join('    ' + line for line in s.splitlines()),
])
def test_unknown_or_uninvoked_parsers_remain_runtime_responsibility(change):
    source = ('import argparse\np = argparse.ArgumentParser()\n'
              'p.add_argument("--corpus", required=True)\nargs = p.parse_args()\n')
    assert missing_required_options(change(source), []) == []


def test_main_guard_and_only_the_invoked_parser_are_checked():
    source = ('import argparse\nunused = argparse.ArgumentParser()\n'
              'unused.add_argument("--unrelated", required=True)\n'
              'if __name__ == "__main__":\n    selected = argparse.ArgumentParser()\n'
              '    selected.add_argument("--source", required=True)\n    args = selected.parse_known_args()\n')
    assert missing_required_options(source, []) == ['--source']
    assert missing_required_options(source, ['--source', 'author_input']) == []


@pytest.mark.parametrize('arguments,accepted', [
    (['--epochs', '40', '--corpus=original'], True),
    (['--corpus', 'original', '--epochs', '40'], True),
    (['--epochs', '4', '--corpus=original'], False),
    (['--epochs', '40', '--corpus=original', '--evaluate'], False),
])
def test_fallback_completion_preserves_other_arguments(arguments, accepted):
    assert completes_required_options(arguments, ['--epochs', '40'], ['--corpus']) is accepted


@pytest.mark.parametrize('repeated_model_error', [False, True])
def test_planning_repairs_a_readme_shorthand_before_any_environment_is_created(tmp_path, repeated_model_error):
    fixture = make_experiment(tmp_path)
    path = fixture['workspace'] / 'train.py'
    text = path.read_text('utf-8').replace('args = parser.parse_args()',
        'parser.add_argument("--corpus", required=True)\nargs = parser.parse_args()')
    path.write_bytes(text.encode())
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    launcher = fixture['workspace'] / 'run.sh'
    command = 'python train.py --epochs 40 --seed 7 --cpu --corpus SignGrid\n'
    launcher.write_bytes(command.encode())
    fixture['snapshot']['files']['run.sh'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    inference = fixture['workspace'] / 'inference.sh'
    inference.write_bytes((command.strip() + ' --evaluate\n').encode())
    fixture['snapshot']['files']['inference.sh'] = hashlib.sha256(inference.read_bytes()).hexdigest()
    for name, text in {'README_other.md': 'python other.py\n', 'other.py': 'print("unrelated example")\n'}.items():
        other = fixture['workspace'] / name
        other.write_bytes(text.encode())
        fixture['snapshot']['files'][name] = hashlib.sha256(other.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet.update(author_execution_only=True, execution_host={'gpu_probe_confirmed': False})
    proposal = execution_proposal(fixture)
    proposal['citations'] = []
    original_steps = deepcopy(proposal['steps'])
    with pytest.raises(PlanEvidenceError, match='required argparse options.*--corpus'):
        validate_author_execution_plan(proposal, packet, fixture['workspace'])
    assert proposal['steps'] == original_steps

    class Planner:
        mock_mode = False
        def __init__(self):
            self.calls = []
        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            answer = deepcopy(proposal)
            if len(self.calls) > 1 and not repeated_model_error:
                assert '--corpus' in prompt.split('Previous structural feedback')[-1]
                assert command.strip() in prompt
                answer['steps'][0]['argv'] += ['--corpus', 'SignGrid']
            return json.dumps(answer)

    planner = Planner()
    plan = propose_author_execution_plan(planner, packet, fixture['workspace'])
    assert len(planner.calls) == 2
    assert plan['steps'][0]['argv'] == command.split()
    assert plan['steps'][0]['timeout_s'] is None
    if repeated_model_error:
        assert plan['command_source']['path'] == 'run.sh'
    assert 'self_repair' not in plan  # Planning validation is not a claimed successful execution.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['snapshot']['files']['train.py']
