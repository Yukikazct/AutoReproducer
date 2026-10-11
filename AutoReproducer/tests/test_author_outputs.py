"""Source-bound output preparation must not evaluate author expressions."""
from copy import deepcopy
import hashlib

import pytest

from src.author_outputs import output_directories
from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.discovered_repository_runtime import verify_capture
from discovered_fixtures import PlannedLLM
from test_discovered_repository_series import run_series


def test_symbolic_checkpoint_and_joined_log_parents_are_source_bound():
    source = """import torch, os
checkpoint = 'pretrained/' + '{}_{}'.format(args.dataset, stamp) + '.pt'
torch.save(model.state_dict(), checkpoint)
with open(os.path.join('logs', f'{args.dataset}_results.txt'), 'w') as stream:
    stream.write('complete')
with open('data/input.json', 'r') as stream:
    pass
"""
    outputs = output_directories(source)
    assert [item['directory'] for item in outputs] == ['pretrained', 'logs']
    assert all(item['statement'] in source for item in outputs)


@pytest.mark.parametrize('source', [
    "torch.save(model.state_dict(), directory + '/weights.pt')",
    "checkpoint = 'pretrained/weights.pt'\ncheckpoint = user_path\ntorch.save(model.state_dict(), checkpoint)",
    "torch.save(model.state_dict(), make_path())",
])
def test_dynamic_or_ambiguous_output_directory_is_not_guessed(source):
    assert output_directories(source) == []


def use_checkpoint_directory(fixture):
    path = fixture['workspace'] / 'train.py'
    text = path.read_text('utf-8').replace('    model.train()\n    return accuracy, model',
        "    torch.save(model.state_dict(), 'pretrained/' + str(run) + '.pt')\n"
        '    model.train()\n    return accuracy, model')
    path.write_text(text, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item.get('path') == path.name)['quote'] = text


def test_native_checkpoint_parents_exist_before_complete_author_series(tmp_path):
    fixture, plan, digest, execution = run_series(tmp_path, 'cpu', use_checkpoint_directory)
    directories = plan['steps'][0]['output_directories']
    assert [item['path'] for item in directories] == ['pretrained']
    assert directories[0]['source_sha256'] == fixture['snapshot']['files']['train.py']
    assert len(list((fixture['workspace'] / 'pretrained').glob('*.pt'))) == 3
    result = verify_capture(plan, fixture['workspace'], execution, digest)
    assert result['protocol_pass'] and result['independent_metrics_pass']


@pytest.mark.parametrize('directory', ['../outside', '/absolute', '.git/objects'])
def test_output_directory_cannot_escape_workspace_or_write_git(tmp_path, directory):
    from test_discovered_repository_series import series_fixture
    fixture = series_fixture(tmp_path)
    use_checkpoint_directory(fixture)
    path = fixture['workspace'] / 'train.py'
    text = path.read_text('utf-8').replace("'pretrained/'", repr(directory + '/'))
    path.write_text(text, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    proposal = deepcopy(fixture['proposal'])
    next(item for item in proposal['citations'] if item.get('path') == path.name)['quote'] = text
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    with pytest.raises(ValueError):
        propose_plan(PlannedLLM(proposal), packet, fixture['workspace'])
