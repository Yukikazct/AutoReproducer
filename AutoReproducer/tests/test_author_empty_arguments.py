"""Preserve original empty CLI values from model planning through author execution."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src import author_execution_plan as author
from src.author_entrypoints import resolve_entrypoint
from test_author_command_inference import packet_for


@pytest.fixture
def empty_value_plan(tmp_path, monkeypatch):
    # This is the shape of a real model proposal, with invented stdlib-only
    # sources. The original parser, not planning code, validates its values.
    argv = ['python', 'main.py', '--dataset', 'wisconsin', '--sub_dataset', '',
            '--method', 'linkx', '--hidden_channels', '64', '--num_layers', '1',
            '--lr', '0.05', '--dropout', '0', '--display_step', '25', '--runs', '10',
            '--link_init_layers_A', '1', '--link_init_layers_X', '1', '--cpu']
    parser = (
        "def parser_add_main_args(parser):\n"
        "    parser.add_argument('--dataset', default='fb100')\n"
        "    parser.add_argument('--sub_dataset', default='')\n"
        "    parser.add_argument('--method', default='link')\n"
        "    parser.add_argument('--epochs', type=int, default=500)\n"
        "    parser.add_argument('--runs', type=int, default=1)\n"
        "    parser.add_argument('--hidden_channels', type=int, default=32)\n"
        "    parser.add_argument('--num_layers', type=int, default=2)\n"
        "    parser.add_argument('--lr', type=float, default=.01)\n"
        "    parser.add_argument('--dropout', type=float, default=.5)\n"
        "    parser.add_argument('--display_step', type=int, default=1)\n"
        "    parser.add_argument('--link_init_layers_A', type=int, default=1)\n"
        "    parser.add_argument('--link_init_layers_X', type=int, default=1)\n"
        "    parser.add_argument('--cpu', action='store_true')\n")
    packet = packet_for(tmp_path, {
        'README.md': 'Run experiments from the repository root: bash experiments/model.sh DATASET\n',
        'experiments/model.sh': 'python main.py --dataset $dataset --sub_dataset "${2:-}" --runs 10\n',
        'main.py': 'import argparse, json, sys\nfrom parse import parser_add_main_args\n'
                   'parser = argparse.ArgumentParser()\nparser_add_main_args(parser)\n'
                   'args = parser.parse_args()\n'
                   'print(json.dumps({"argv": sys.argv, "parsed": vars(args)}, sort_keys=True))\n',
        'parse.py': parser,
    })
    selected = {'version': 1, 'scope': author.SCOPE, 'device': 'cpu',
                'repository': {key: packet['repository'][key] for key in ('url', 'revision')},
                'citations': [], 'requirements': {}, 'datasets': [],
                'steps': [{'id': 'original', 'kind': 'train', 'argv': argv,
                           'entrypoint': {'kind': 'script', 'target': 'main.py'},
                           'cwd': '.', 'timeout_s': None}]}

    class Planner:
        mock_mode = False

        def __init__(self):
            self.calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            assert kwargs['task'] == 'author_execution_plan'
            assert 'parser_add_main_args' in prompt
            return json.dumps(selected)

    def reject_fallback(*args, **kwargs):
        pytest.fail('A valid model plan must not be replaced with deterministic fallback')

    monkeypatch.setattr(author, '_plan_from_available_sources', reject_fallback)
    monkeypatch.setattr('src.execution_host.probe_execution_host',
                        lambda: {'cpu_available': True, 'gpu_probe_confirmed': False})
    planner = Planner()
    original = deepcopy(selected)
    plan = author.propose_author_execution_plan(planner, packet, tmp_path)
    assert len(planner.calls) == 1
    assert selected == original
    assert plan['steps'][0]['argv'] == argv
    assert plan['steps'][0]['cwd'] == '.'
    assert 'command_source' not in plan and 'planning_failures' not in plan
    snapshot = {'files': {name: item['sha256'] for name, item in packet['file_inventory'].items()}}
    materialized = author.materialize_author_execution(tmp_path, plan, 'c' * 64, snapshot)
    config = json.loads((tmp_path / '_autorepro_capture/config.json').read_text('utf-8'))
    assert config['steps'][0]['argv'] == argv
    return plan, snapshot, materialized


def test_model_plan_preserves_empty_value_and_native_defaults_through_real_popen(tmp_path, empty_value_plan):
    plan, snapshot, _ = empty_value_plan
    argv = plan['steps'][0]['argv']
    # Windows subprocess uses list2cmdline here; neither layer may discard ''.
    result = subprocess.run([sys.executable, *argv[1:]], cwd=tmp_path,
                            capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed['argv'] == argv[1:]
    assert observed['parsed']['sub_dataset'] == ''
    assert observed['parsed']['epochs'] == 500
    assert observed['parsed']['runs'] == 10
    for name, checksum in snapshot['files'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum


def test_materialized_observer_preserves_empty_value_in_real_author_process(tmp_path, empty_value_plan):
    pytest.importorskip('torch')
    from src.agents.code_executor import CodeExecutorAgent
    from src.repository_runner import RepositoryRunner

    plan, snapshot, materialized = empty_value_plan
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._ensure_local_deps = Mock(return_value=None)
    execution = RepositoryRunner(executor=executor).run(tmp_path, materialized['steps'], {})
    assert execution['success'], execution['final']['stderr']
    observed = next(json.loads(line) for line in execution['final']['stdout'].splitlines()
                    if '"argv"' in line)
    # runpy uses the absolute script filename as sys.argv[0]; every original
    # argument after the entrypoint must still match the frozen list exactly.
    assert observed['argv'][0] == str(tmp_path / 'main.py')
    assert observed['argv'][1:] == plan['steps'][0]['argv'][2:]
    assert observed['parsed']['sub_dataset'] == ''
    assert observed['parsed']['epochs'] == 500 and observed['parsed']['runs'] == 10
    receipt = json.loads((tmp_path / '_autorepro_capture/author_original.json').read_text('utf-8'))
    assert receipt['step'] == plan['steps'][0]
    assert receipt['natural_return'] is True
    assert receipt['optimizer_steps'] == 0  # This is an argv fixture, not paper training.
    for name, checksum in snapshot['files'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize('prefix', [
    ['python', 'main.py'], ['python3', '-u', 'main.py'],
    ['python', '-m', 'package.main'], ['python3', '-u', '-m', 'package.main'],
])
@pytest.mark.parametrize('arguments', [['--sub_dataset', ''], [''], ['--', '']])
def test_script_and_module_entrypoints_preserve_empty_values(tmp_path, prefix, arguments):
    packet = packet_for(tmp_path, {'main.py': '', 'package/main.py': ''})
    argv = prefix + arguments
    _, preserved, values = resolve_entrypoint({'argv': argv}, packet, tmp_path)
    assert preserved == argv
    assert values == arguments


@pytest.mark.parametrize('argv', [
    ['', 'main.py'], ['python', ''], ['python', '-u', ''],
    ['python', '-m', ''], ['python', '-u', '-m', ''],
    ['python', '-u'], ['python', '-m'], ['python', 'main.py', None],
])
def test_empty_python_prefix_or_nonstring_argument_remains_rejected(tmp_path, argv):
    packet = packet_for(tmp_path, {'main.py': ''})
    with pytest.raises(ValueError):
        resolve_entrypoint({'argv': argv}, packet, tmp_path)


@pytest.mark.parametrize('kind', ['script', 'module', 'notebook'])
def test_empty_explicit_entrypoint_remains_rejected(tmp_path, kind):
    packet = packet_for(tmp_path, {'main.py': ''})
    with pytest.raises(ValueError, match='needs a script, module or notebook target'):
        resolve_entrypoint({'argv': ['python', 'main.py'],
                            'entrypoint': {'kind': kind, 'target': ''}}, packet, tmp_path)


@pytest.mark.parametrize('control', ['\x00', '\r', '\n'])
@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_control_characters_in_any_argv_position_remain_rejected(tmp_path, control, position):
    packet = packet_for(tmp_path, {'main.py': ''})
    argv = ['python', 'main.py', '--sub_dataset', '']
    argv[position] += control
    with pytest.raises(ValueError, match='Only structured original Python argv'):
        resolve_entrypoint({'argv': argv}, packet, tmp_path)
