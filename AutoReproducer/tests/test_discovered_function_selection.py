"""One unchanged author function per native run, retaining its selected weights."""
from copy import deepcopy
import hashlib

import pytest

from src.native_loop import resolve_function_selection, discover_native_selections
from src.discovered_repository_runtime import verify_capture, validate_capture, materialize_capture
from src.discovered_repository_plan import build_evidence_packet
from src.method_adapters import read_json, write_json
from test_discovered_module_loop import SOURCE, SCOPE, use_module_loop
from test_discovered_repository_series import run_series, series_fixture
from test_discovered_repository_runtime import invoke, native_environment


def use_selected_function(fixture):
    use_module_loop(fixture)
    start, end = SOURCE.index('for idx in range(3):'), SOURCE.index('assert final_results')
    body = SOURCE[start:end].splitlines()[1:]
    source = (SOURCE[:start] + 'def train_one(idx):\n' + '\n'.join(body)
              + '\nfor idx in range(3):\n    train_one(idx)\n' + SOURCE[end:])
    path = fixture['workspace'] / 'train.py'
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(item for item in fixture['proposal']['citations'] if item.get('path') == 'train.py')['quote'] = source
    scope = {key: deepcopy(value) for key, value in SCOPE.items() if key != 'loop_counter'}
    scope['function'] = 'train_one'
    fixture['proposal']['capture']['scope'] = scope


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_complete_natural_runs_retain_original_best_weights_after_author_deletes_model(tmp_path, device):
    fixture, plan, spec_hash, execution = run_series(tmp_path, device, use_selected_function)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    runs = result['capture']['runs']
    assert result['is_reproduced'] is True
    assert [run['optimizer_steps'] for run in runs] == [8, 10, 12]
    assert all(run['natural_return'] is True and 'native_loop_completed' not in run for run in runs)
    assert [row['accuracy'] for row in result['independent_metrics']['runs']] == [1.0, 1.0, 1.0]
    assert all(row['validation_selection']['optimizer_steps'] < row['optimizer_steps'] for row in runs)
    receipt_path = fixture['workspace'] / '_autorepro_capture/capture.json'
    receipt = read_json(receipt_path)
    receipt['runs'][0].pop('validation_selection')
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='Validation-selected weights'):
        verify_capture(plan, fixture['workspace'], execution, spec_hash)


def test_function_selection_is_advertised_and_pinned(tmp_path):
    fixture = series_fixture(tmp_path)
    use_selected_function(fixture)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    source = packet['repository']['files']['train.py']['text']
    options = discover_native_selections(packet['repository']['files'])
    scope = deepcopy(fixture['proposal']['capture']['scope'])
    selected = next(item for item in options if item['scope'].get('function') == 'train_one'
                    and 'loop_counter' not in item['scope'])
    assert selected['completion_mechanism'] == 'natural_function_return_per_native_run'
    assert selected['source_sha256'] == fixture['snapshot']['files']['train.py']
    assert resolve_function_selection(source, scope)['scope_name'] == 'train_one'
    assert validate_capture(fixture['proposal']['capture'], packet, fixture['workspace'])['scope'] == scope


@pytest.mark.parametrize('change', [
    lambda s: s.replace('def train_one(idx):', 'async def train_one(idx):'),
    lambda s: s.replace('def train_one(idx):', '@unknown\ndef train_one(idx):'),
    lambda s: s.replace('if val_loss < best_val_loss:', 'if compute() < best_val_loss:'),
    lambda s: s.replace('            selected_test =', '            nested = lambda: 1\n            selected_test2 ='),
])
def test_selected_function_cannot_guess_dynamic_or_missing_original_branch(tmp_path, change):
    fixture = series_fixture(tmp_path)
    use_selected_function(fixture)
    with pytest.raises(ValueError):
        resolve_function_selection(change((fixture['workspace'] / 'train.py').read_text('utf-8')),
                                   fixture['proposal']['capture']['scope'])


@pytest.mark.parametrize('insertion', [
    '    if idx == 0:\n        train_one(1)\n',
    '    if idx == 2:\n        raise RuntimeError("incomplete final native run")\n',
])
def test_recursive_or_exceptional_training_is_never_complete(tmp_path, insertion):
    fixture = series_fixture(tmp_path)
    use_selected_function(fixture)
    path = fixture['workspace'] / 'train.py'
    source = path.read_text('utf-8').replace('def train_one(idx):\n', 'def train_one(idx):\n' + insertion)
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    materialize_capture(fixture['workspace'], fixture['proposal'], 'a' * 64, snapshot=fixture['snapshot'])
    result = invoke(fixture['workspace'], 'capture', native_environment())
    assert result.returncode != 0
    assert not (fixture['workspace'] / '_autorepro_capture/capture.json').exists()
