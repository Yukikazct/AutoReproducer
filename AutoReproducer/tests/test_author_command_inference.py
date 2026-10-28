"""Sparse README prose plus frozen author source can establish real argv."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.author_command_inference import discover_author_commands, select_author_command
from src.author_execution_plan import propose_author_execution_plan
from src.discovered_repository_plan import PlanEvidenceError


def packet_for(tmp_path, sources):
    inventory, selected = {}, {}
    for name, text in sources.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        raw = text.encode('utf-8')
        file.write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
        inventory[name] = {'sha256': checksum, 'bytes': len(raw), 'text_available': True}
        if Path(name).suffix != '.json':
            selected[name] = {'sha256': checksum, 'bytes': len(raw), 'text': text}
    return {'version': 1, 'author_execution_only': True,
        'repository': {'url': 'https://github.com/invented-fixtures/source-inference',
                       'revision': 'a' * 40, 'files': selected},
        'file_inventory': inventory,
        'pdf': {'sha256': 'b' * 64, 'pages': [], 'repository_links': []},
        'limits': {'source_bytes': 1000000, 'source_files': 64},
        'execution_host': {'cpu_available': True, 'gpu_probe_confirmed': True, 'cuda_device_indices': []}}


TRAIN = ("import argparse\n"
         "parser = argparse.ArgumentParser()\n"
         "parser.add_argument('--epochs', type=int, default=1000)\n"
         "parser.add_argument('--seed', type=int, default=42)\n"
         "parser.add_argument('--dataset', default='Cora')\n"
         "args = parser.parse_args()\n"
         "def train(args):\n    pass\n"
         "if __name__ == '__main__':\n    train(args)\n")


def test_sparse_readme_is_combined_with_real_parser_defaults_without_running_source(tmp_path):
    source = "raise RuntimeError('Author source must never be imported for planning')\n" + TRAIN
    packet = packet_for(tmp_path, {'README.md': 'Training lives in `train.py`; it uses Cora and the defaults in its parser.',
                                  'train.py': source})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py']
    assert plan['steps'][0]['timeout_s'] is None
    origin = plan['command_source']
    assert origin['kind'] == 'source_inference' and origin['sha256'] == packet['file_inventory']['train.py']['sha256']
    assert origin['readme_evidence'][0]['path'] == 'README.md'
    assert {item['dest']: item['value'] for item in origin['cli']['defaults']} == {
        'epochs': 1000, 'seed': 42, 'dataset': 'Cora'}
    assert 'train(args)' in origin['quote']
    assert not list(tmp_path.glob('*.pyc'))


def test_prose_selects_matching_entrypoint_and_original_config_among_alternatives(tmp_path):
    configured = TRAIN.replace("args = parser.parse_args()", "parser.add_argument('--config', required=True)\nargs = parser.parse_args()")
    packet = packet_for(tmp_path, {
        'README.md': 'Our Cora experiment is handled by `src/train.py`. The recommended setup is `configs/cora.yaml`.',
        'src/train.py': configured, 'other/train.py': TRAIN,
        'configs/cora.yaml': 'dataset: Cora\nepochs: 1000\nseed: 42\n',
        'configs/pubmed.yaml': 'dataset: PubMed\nepochs: 1000\nseed: 42\n'})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    step = plan['steps'][0]
    assert step['cwd'] == 'src'
    assert step['argv'] == ['python', 'train.py', '--config', '../configs/cora.yaml']
    assert plan['command_source']['configuration_sources'][0]['text'].startswith('dataset: Cora')
    assert plan['command_source']['readme_evidence'][0]['quote'].startswith('Our Cora')
    assert plan['command_source']['cli']['config_paths'] == ['configs/cora.yaml']


def test_actual_bundled_loader_paths_establish_repository_root_cwd(tmp_path):
    source = TRAIN + "\nwith open('data/full.csv') as stream:\n    rows = stream.read()\n"
    packet = packet_for(tmp_path, {'README.md': 'The full dataset is read by `src/train.py`.',
                                  'src/train.py': source, 'data/full.csv': 'features,label\n0,1\n'})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == '.'
    assert plan['steps'][0]['argv'] == ['python', 'src/train.py']
    assert plan['command_source']['input_paths'][0]['value'] == 'data/full.csv'
    assert plan['command_source']['cwd_evidence'] == 'original_input_paths'


def test_package_entrypoint_preserves_relative_imports_with_python_module(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training is implemented in `research/train.py`.',
        'research/__init__.py': '', 'research/model.py': 'class Model:\n    pass\n',
        'research/train.py': 'from .model import Model\n' + TRAIN})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', '-m', 'research.train']
    assert plan['steps'][0]['cwd'] == '.'
    assert plan['steps'][0]['entrypoint']['source_path'] == 'research/train.py'


def test_unknown_required_values_are_reported_without_inventing_dataset_or_seed(tmp_path):
    source = TRAIN.replace("parser.add_argument('--dataset', default='Cora')",
                           "parser.add_argument('--dataset', required=True, choices=['Cora', 'PubMed'])")
    packet = packet_for(tmp_path, {'README.md': 'Training is implemented in train.py.', 'train.py': source})
    with pytest.raises(PlanEvidenceError) as captured:
        propose_author_execution_plan(None, packet, tmp_path)
    evidence = captured.value.planning_evidence
    assert evidence['stage'] == 'author_command_inference'
    assert evidence['unresolved_arguments'] == [{'path': 'train.py', 'missing': ['--dataset']}]
    assert '--dataset' not in evidence['candidates'][0]['argv']


def test_ambiguous_entrypoints_are_given_to_model_with_actual_source_context(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'See the source for our two training experiments.',
                                  'cora/train.py': TRAIN, 'pubmed/train.py': TRAIN})
    class SourcePlanner:
        calls = []
        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            assert 'source-derived candidates' in prompt
            assert 'See the source for our two training experiments.' in prompt
            assert "parser.add_argument('--epochs', type=int, default=1000)" in prompt
            assert 'cora/train.py' in prompt and 'pubmed/train.py' in prompt
            return json.dumps({'version': 1, 'scope': 'author_code_execution',
                'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
                'device': 'cpu', 'requirements': {'author': []}, 'datasets': [],
                'steps': [{'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'],
                           'cwd': 'cora', 'timeout_s': None}]})
    planner = SourcePlanner()
    plan = propose_author_execution_plan(planner, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == 'cora' and len(planner.calls) == 1
    assert select_author_command(packet['program_detected_author_commands']) is None


def test_nontraining_scripts_and_uninvoked_training_helpers_are_not_commands(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'The package exports helpers.',
        'train.py': 'def train():\n    loss.backward()\n',
        'main.py': 'class Model:\n    def train(self):\n        loss.backward()\n',
        'run.py': "print('hello')\n"})
    assert discover_author_commands(packet, tmp_path)['candidates'] == []


def test_inference_bound_and_source_checks_remain_enforced(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Read the original sources.',
        'train_a.py': TRAIN, 'train_b.py': TRAIN})
    limited = discover_author_commands(packet, tmp_path, max_files=1)
    assert limited['files_examined'] == 1 and len(limited['candidates']) == 1
    (tmp_path / 'train_a.py').write_text(TRAIN + '\n# changed\n', encoding='utf-8')
    changed = discover_author_commands(packet, tmp_path)
    assert any(item['path'] == 'train_a.py' and 'changed after snapshot' in item['reason'] for item in changed['failures'])
    assert all(item['source_path'] != 'train_a.py' for item in changed['candidates'])


def test_existing_literal_command_retains_author_settings_over_inferred_defaults(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'python train.py --epochs 500 --seed 9', 'train.py': TRAIN})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--epochs', '500', '--seed', '9']
    assert plan['command_source']['path'] == 'README.md'
    assert plan['command_source'].get('kind') != 'source_inference'


def test_unique_basenames_in_prose_resolve_original_entrypoint_and_configuration(tmp_path):
    source = TRAIN.replace('args = parser.parse_args()',
                           "parser.add_argument('--config', required=True)\nargs = parser.parse_args()")
    packet = packet_for(tmp_path, {'README.md': 'Training uses `train.py` and the recommended `cora.yaml`.',
        'src/train.py': source, 'configs/cora.yaml': 'dataset: Cora\n',
        'configs/pubmed.yaml': 'dataset: PubMed\n'})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--config', '../configs/cora.yaml']
    assert plan['command_source']['readme_evidence'][0]['quote'].endswith('`cora.yaml`.')


def test_duplicate_basenames_in_prose_do_not_select_an_arbitrary_experiment(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training uses train.py.',
        'cora/train.py': TRAIN, 'pubmed/train.py': TRAIN})
    with pytest.raises(PlanEvidenceError, match='No unique complete source-derived'):
        propose_author_execution_plan(None, packet, tmp_path)


@pytest.mark.parametrize('condition', ['False', "args.mode == 'train'", 'args.train'])
def test_dead_or_disabled_training_branches_do_not_authorize_a_bare_command(tmp_path, condition):
    source = ("import argparse\np = argparse.ArgumentParser()\n"
              "p.add_argument('--mode', default='eval')\np.add_argument('--train', action='store_true')\n"
              "args = p.parse_args()\ndef train(args):\n    pass\n"
              f'if {condition}:\n    train(args)\n')
    packet = packet_for(tmp_path, {'README.md': 'See train.py for available modes.', 'train.py': source})
    assert discover_author_commands(packet, tmp_path)['candidates'] == []


def test_original_default_training_mode_is_analyzed_without_cli_override(tmp_path):
    source = ("import argparse\np = argparse.ArgumentParser()\n"
              "p.add_argument('--mode', default='train')\nargs = p.parse_args()\n"
              "def train(args):\n    pass\nif args.mode == 'train':\n    train(args)\n")
    packet = packet_for(tmp_path, {'README.md': 'See train.py for available modes.', 'train.py': source})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py']


def test_sparse_prose_working_directory_overrides_filename_parent_guess(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Run the training from the repository root; the entrypoint is src/train.py.',
                                  'src/train.py': TRAIN})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == '.' and plan['steps'][0]['argv'] == ['python', 'src/train.py']
    assert plan['command_source']['cwd_readme_evidence'][0]['path'] == 'README.md'


def test_failed_explicit_command_does_not_discard_its_flags_or_directory(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Navigate into `missing`.\npython train.py --epochs 500 --seed 9',
                                  'train.py': TRAIN})
    with pytest.raises(PlanEvidenceError):
        propose_author_execution_plan(None, packet, tmp_path)


def test_sparse_readme_dependency_pins_are_preserved_in_inferred_plan(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training lives in train.py.\nDependencies:\n- `numpy==1.26.4`\n',
                                  'train.py': TRAIN})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert any('numpy==1.26.4' in line for line in plan['requirements']['author'])


def test_completion_adds_only_missing_source_backed_flags_and_retains_model_selection(tmp_path):
    source = TRAIN.replace("parser.add_argument('--dataset', default='Cora')",
        "parser.add_argument('--dataset', required=True)\nparser.add_argument('--config', required=True)")
    packet = packet_for(tmp_path, {'README.md': 'Training uses train.py and configs/cora.yaml.',
                                  'train.py': source, 'configs/cora.yaml': 'dataset: Cora\n'})
    class IncompletePlanner:
        def chat(self, *args, **kwargs):
            return json.dumps({'version': 1, 'scope': 'author_code_execution',
                'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
                'device': 'cpu', 'requirements': {'author': []}, 'datasets': [],
                'steps': [{'id': 'train', 'kind': 'train',
                    'argv': ['python', 'train.py', '--dataset', 'Cora', '--epochs', '1000', '--seed', '42'],
                    'cwd': '.', 'timeout_s': None}]})
    plan = propose_author_execution_plan(IncompletePlanner(), packet, tmp_path)
    assert plan['steps'][0]['argv'] == ['python', 'train.py', '--dataset', 'Cora', '--epochs', '1000',
                                       '--seed', '42', '--config', 'configs/cora.yaml']


def test_readme_and_config_evidence_share_the_total_read_budget(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training is implemented in train.py.\n' * 100,
                                  'train.py': TRAIN})
    result = discover_author_commands(packet, tmp_path, max_bytes=100)
    assert result['bytes_examined'] <= 100 and result['candidates'] == []


def test_source_inferred_command_runs_original_stdlib_training_with_config_and_complete_data(tmp_path):
    source = ("import argparse, json, random\n"
              "p = argparse.ArgumentParser()\np.add_argument('--config', required=True)\n"
              "p.add_argument('--epochs', type=int, default=7)\n"
              "p.add_argument('--seed', type=int, default=11)\nargs = p.parse_args()\n"
              "def train(args):\n"
              "    with open(args.config, encoding='utf-8') as stream:\n        settings = json.load(stream)\n"
              "    with open('data/full.json', encoding='utf-8') as stream:\n        data = json.load(stream)\n"
              "    random.seed(args.seed)\n"
              "    updates = 0\n    for epoch in range(args.epochs):\n        for value in data:\n            updates += 1\n"
              "    print(json.dumps({'epochs': args.epochs, 'seed': args.seed, 'samples': len(data),\n"
              "                      'updates': updates, 'dataset': settings['dataset']}))\n"
              "if __name__ == '__main__':\n    train(args)\n")
    packet = packet_for(tmp_path, {'README.md': 'Our complete experiment uses train.py with cora.json. The source defines its seed and schedule.',
        'src/train.py': source, 'configs/cora.json': '{"dataset": "CompleteFixture"}',
        'data/full.json': '[1, 2, 3, 4]'})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    step = plan['steps'][0]
    assert step['cwd'] == '.'
    assert step['argv'] == ['python', 'src/train.py', '--config', 'configs/cora.json']
    completed = subprocess.run([sys.executable, *step['argv'][1:]], cwd=tmp_path / step['cwd'],
                               capture_output=True, text=True, encoding='utf-8', timeout=10, check=True)
    assert json.loads(completed.stdout) == {'epochs': 7, 'seed': 11, 'samples': 4,
                                            'updates': 28, 'dataset': 'CompleteFixture'}


@pytest.mark.parametrize('invocation', [
    ['python', 'src/train.py'], ['python3', '-u', 'src/train.py'],
    ['python', '-u', '-m', 'research.train'],
])
def test_required_completion_preserves_selected_root_cwd_entrypoint_prefix_and_data_arguments(tmp_path, invocation):
    module = '-m' in invocation
    path = 'research/train.py' if module else 'src/train.py'
    source = ("import argparse, json\n"
              "p = argparse.ArgumentParser()\np.add_argument('--data', required=True)\n"
              "p.add_argument('--config', required=True)\nargs = p.parse_args()\n"
              "def train(args):\n"
              "    with open(args.data, encoding='utf-8') as stream:\n        data = stream.read()\n"
              "    with open(args.config, encoding='utf-8') as stream:\n        config = stream.read()\n"
              "    print(json.dumps({'data': data, 'config': config}))\n"
              "if __name__ == '__main__':\n    train(args)\n")
    sources = {'README.md': f'Training is handled by {path}; use configs/author.yaml.',
               path: source, 'configs/author.yaml': 'seed: 42\nepochs: 1000\n',
               'data/full.csv': 'feature,label\n1,0\n'}
    if module:
        sources.update({'research/__init__.py': '', 'research/constants.py': "ID = 'original'\n"})
        sources[path] = 'from .constants import ID\n' + source
    packet = packet_for(tmp_path, sources)
    original = [*invocation, '--data', 'data/full.csv']
    class SelectedCommandPlanner:
        def chat(self, *args, **kwargs):
            return json.dumps({'version': 1, 'scope': 'author_code_execution',
                'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
                'device': 'cpu', 'requirements': {'author': []}, 'datasets': [],
                'steps': [{'id': 'train', 'kind': 'train', 'argv': original,
                           'cwd': '.', 'timeout_s': None}]})
    plan = propose_author_execution_plan(SelectedCommandPlanner(), packet, tmp_path)
    step = plan['steps'][0]
    assert step['cwd'] == '.'
    assert step['argv'] == [*original, '--config', 'configs/author.yaml']
    assert plan['command_source']['cwd_evidence'] == 'selected_author_invocation'
    assert plan['command_source']['cli']['config_paths'] == ['configs/author.yaml']
    result = subprocess.run([sys.executable, *step['argv'][1:]], cwd=tmp_path,
                            text=True, capture_output=True, encoding='utf-8', timeout=10, check=True)
    assert json.loads(result.stdout) == {'data': sources['data/full.csv'], 'config': sources['configs/author.yaml']}


def test_unrelated_reporting_directory_hint_does_not_block_training_source_inference(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'Training is implemented in src/train.py.\n'
        'For the optional reporting utility, navigate into `tools`.',
        'src/train.py': TRAIN, 'tools/report.py': "print('report')\n"})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == 'src'
    assert plan['steps'][0]['argv'] == ['python', 'train.py']
    assert plan['command_source']['cwd_readme_evidence'] == []


def test_cwd_hint_for_a_named_sibling_experiment_does_not_change_selected_training(tmp_path):
    packet = packet_for(tmp_path, {'README.md': 'The chosen training is src/train.py.\n'
        'For tools/report.py, change directory to `tools`.',
        'src/train.py': TRAIN, 'tools/report.py': "print('report')\n"})
    plan = propose_author_execution_plan(None, packet, tmp_path)
    assert plan['steps'][0]['cwd'] == 'src' and plan['steps'][0]['argv'] == ['python', 'train.py']
