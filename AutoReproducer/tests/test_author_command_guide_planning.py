"""Following an author guide must preserve its full native invocation."""
import pytest

from src.author_execution_plan import propose_author_execution_plan
from src.discovered_repository_plan import PlanEvidenceError
from test_author_command_inference import TRAIN, packet_for


def guide_packet(tmp_path, guide):
    packet = packet_for(tmp_path, {
        'README.md': '[Original experiment instructions](docs/experiment.md)',
        'docs/experiment.md': guide,
        'experiments/node/classification.py': TRAIN,
    })
    packet['repository']['files'] = {'README.md': packet['repository']['files']['README.md']}
    return packet


def test_linked_guide_literal_settings_take_precedence_over_native_defaults(tmp_path):
    packet = guide_packet(tmp_path, 'Our experiment uses the complete Cora input.\n'
        '```sh\npython experiments/node/classification.py --epochs 500 --seed 9\n```')
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'experiments/node/classification.py', '--epochs', '500', '--seed', '9']
    assert plan['steps'][0]['cwd'] == '.'
    assert plan['command_source']['path'] == 'docs/experiment.md'
    assert plan['command_source']['sha256'] == packet['file_inventory']['docs/experiment.md']['sha256']


def test_linked_guide_prose_directory_keeps_original_invocation(tmp_path):
    packet = guide_packet(tmp_path, 'Navigate into `experiments/node` before training.\n'
        '```sh\npython classification.py --epochs 500 --seed 9\n```')
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == 'experiments/node'
    assert plan['steps'][0]['argv'] == ['python', 'classification.py', '--epochs', '500', '--seed', '9']


@pytest.mark.parametrize('invocation', ['classification.py', '-u classification.py', '-m classification'])
def test_nested_readme_resolves_document_relative_entrypoint(tmp_path, invocation):
    packet = packet_for(tmp_path, {
        'README.md': 'Each benchmark is run in its own task directory.',
        'experiments/node/README.md': f'Train with:\n```sh\npython {invocation} --epochs 500 --seed 9\n```',
        'experiments/node/classification.py': TRAIN,
    })
    plan = propose_author_execution_plan(None, packet, tmp_path)
    step = plan['steps'][0]
    assert step['cwd'] == 'experiments/node'
    assert step['argv'] == ['python', *invocation.split(), '--epochs', '500', '--seed', '9']
    assert plan['command_source']['path'] == 'experiments/node/README.md'
    assert plan['command_source']['cwd_evidence'] == 'document_relative_entrypoint'


@pytest.mark.parametrize('directory_hint', ['', 'cd .\n'])
def test_nested_readme_needs_explicit_cwd_when_both_entrypoints_exist(tmp_path, directory_hint):
    packet = packet_for(tmp_path, {
        'benchmark/README.md': f'```sh\n{directory_hint}python train.py --epochs 500\n```',
        'benchmark/train.py': TRAIN,
        'train.py': TRAIN,
    })
    if not directory_hint:
        with pytest.raises(PlanEvidenceError, match='explicit working directory is required'):
            propose_author_execution_plan(None, packet, tmp_path)
        return
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == '.'
    assert plan['steps'][0]['entrypoint']['source_path'] == 'train.py'
    assert plan['command_source']['cwd_quote'] == 'cd .'


def test_incomplete_guide_literal_must_not_silently_revert_to_defaults(tmp_path):
    packet = guide_packet(tmp_path, '```sh\npython experiments/node/classification.py --epochs <author-value>\n```')
    with pytest.raises(PlanEvidenceError):
        propose_author_execution_plan(None, packet, tmp_path)


def test_unlinked_document_is_not_a_literal_command_source(tmp_path):
    packet = guide_packet(tmp_path, 'Our experiment is implemented in `experiments/node/classification.py`.')
    extra = packet_for(tmp_path, {'docs/unrelated.md': 'python experiments/node/classification.py --seed 999'})
    packet['file_inventory'].update(extra['file_inventory'])
    packet['repository']['files'].update(extra['repository']['files'])
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'classification.py']
    assert plan['steps'][0]['cwd'] == 'experiments/node'
    assert plan['command_source']['kind'] == 'source_inference'


@pytest.mark.parametrize('installation', ['python setup.py install', 'python -u setup.py develop', 'python -m setup install'])
def test_guide_installation_never_replaces_the_following_training_command(tmp_path, installation):
    packet = guide_packet(tmp_path, 'First install the package:\n' + installation + '\n'
        'Then train:\npython experiments/node/classification.py --epochs 500 --seed 9\n')
    extra = packet_for(tmp_path, {'setup.py': "print('setup only')\n"})
    packet['file_inventory'].update(extra['file_inventory'])
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'experiments/node/classification.py', '--epochs', '500', '--seed', '9']


def test_installation_only_literal_does_not_block_descriptive_training_defaults(tmp_path):
    packet = guide_packet(tmp_path, 'First install:\npython setup.py install\n'
        'The native training defaults live in `experiments/node/classification.py`.')
    extra = packet_for(tmp_path, {'setup.py': "print('setup only')\n"})
    packet['file_inventory'].update(extra['file_inventory'])
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'classification.py']
    assert plan['command_source']['kind'] == 'source_inference'
