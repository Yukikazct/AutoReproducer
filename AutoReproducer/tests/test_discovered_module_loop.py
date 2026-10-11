"""Observe unchanged module loops; selected weights differ from final weights."""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import pytest

from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.discovered_repository_runtime import materialize_capture, verify_capture
from src.method_adapters import read_json, write_json
from src.native_loop import resolve_module_loop
from discovered_fixtures import PlannedLLM
from test_discovered_repository_series import run_series, series_fixture
from test_discovered_repository_runtime import invoke, native_environment


SOURCE = '''import argparse
import json
import torch
import numpy as np
from author_model import SignalModel
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
device = torch.device('cpu' if args.cpu else 'cuda')
with open('dataset.json') as stream:
    records = json.load(stream)
features = torch.tensor(records['features'], device=device)
labels = torch.tensor(records['labels'], dtype=torch.long, device=device)
idx_val = torch.tensor([4, 5, 6, 7], device=device)
idx_test = torch.tensor([8, 9, 10, 11], device=device)
nothing = None
results = []
final_results = []
for idx in range(3):
    torch.manual_seed(args.seed + idx)
    model = SignalModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features, nothing)[:4], labels[:4])
        loss.backward()
        optimizer.step()
        if epoch >= 5:
            with torch.no_grad():
                model.linear.weight.copy_(torch.tensor([[1.0], [-1.0]], device=device))
                model.linear.bias.zero_()
        model.eval()
        output = model(features, nothing)
        val_loss = torch.nn.functional.cross_entropy(output[idx_val], labels[idx_val])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            selected_test = (output[idx_test].argmax(1) == labels[idx_test]).double().mean().item()
        if epoch + 1 == 8 + idx * 2:
            break
    results.append(selected_test)
    final_results.append((output[idx_test].argmax(1) == labels[idx_test]).double().mean().item())
    del model, optimizer
assert final_results == [0.0, 0.0, 0.0]
print('Test Accuracy:', np.mean(results))
'''
SCOPE = {'path': 'train.py', 'loop_counter': 'idx', 'selection': {
    'validation_metric': 'val_loss', 'comparison': 'lt', 'best_metric': 'best_val_loss',
    'result': 'selected_test', 'result_unit': 'fraction'}}


def use_module_loop(fixture):
    workspace = fixture['workspace']
    model = '''import torch
class SignalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, features, unused=None):
        assert unused is None
        return self.linear(features)
'''
    for name, source in {'train.py': SOURCE, 'author_model.py': model}.items():
        path = workspace / name
        path.write_text(source, encoding='utf-8')
        fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        next(c for c in fixture['proposal']['citations'] if c.get('path') == name)['quote'] = source
    fixture['proposal']['capture'].update(scope=deepcopy(SCOPE), forward_args=['features', 'nothing'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_every_native_iteration_finishes_but_best_weights_are_evaluated(tmp_path, device):
    if device == 'cuda':
        from test_discovered_repository_series import cuda_dependency_cache
        cuda_dependency_cache()
    fixture, plan, spec_hash, execution = run_series(tmp_path, device, use_module_loop)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['is_reproduced'] is True
    assert [r['optimizer_steps'] for r in result['capture']['runs']] == [8, 10, 12]
    assert [r['accuracy'] for r in result['independent_metrics']['runs']] == [1.0, 1.0, 1.0]
    assert all(r['native_loop_completed'] and not r['natural_return'] for r in result['capture']['runs'])
    assert all(r['validation_selection']['optimizer_steps'] < r['optimizer_steps'] for r in result['capture']['runs'])
    assert (fixture['workspace'] / 'train.py').read_text('utf-8') == SOURCE


@pytest.mark.parametrize('mutate', [
    lambda scope: scope.update(loop_counter='idx+1'),
    lambda scope: scope['selection'].update(validation_metric='test_accuracy'),
    lambda scope: scope['selection'].update(comparison='eval'),
    lambda scope: scope['selection'].update(result='missing'),
    lambda scope: scope['selection'].update(result_unit='scalar'),
])
def test_selection_must_resolve_to_actual_source_not_an_expression(mutate):
    scope = deepcopy(SCOPE)
    mutate(scope)
    with pytest.raises(ValueError):
        resolve_module_loop(SOURCE, scope)


def test_final_weights_cannot_replace_selected_receipt(tmp_path):
    fixture, plan, spec_hash, execution = run_series(tmp_path, 'cpu', use_module_loop)
    path = fixture['workspace'] / '_autorepro_capture/capture.json'
    receipt = read_json(path)
    receipt['runs'][0]['validation_selection']['accuracy'] = 0.0
    write_json(path, receipt)
    # Rehashing a modified training receipt does not authorize new selections.
    metric_path = path.with_name('independent_metrics.json')
    metrics = read_json(metric_path)
    metrics['capture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(metric_path, metrics)
    with pytest.raises(ValueError, match='Validation-selected weights'):
        verify_capture(plan, fixture['workspace'], execution, spec_hash)


@pytest.mark.parametrize('mutation,reason', [
    (lambda text: text.replace('range(3)', 'range(3 - 1)'), 'every declared native repetition'),
    (lambda text: text.replace('range(3)', 'range(3 + 1)'), 'complete repetition order'),
    (lambda text: text.replace('torch.manual_seed(args.seed + idx)', 'torch.manual_seed(args.seed + idx)\n    idx += 1'), 'complete repetition order'),
    (lambda text: text.replace("print('Test Accuracy:', np.mean(results))", "raise RuntimeError('after training')\nprint('Test Accuracy:', np.mean(results))"), 'after training'),
])
def test_partial_extra_or_failed_author_execution_never_writes_success(tmp_path, mutation, reason):
    fixture = series_fixture(tmp_path)
    use_module_loop(fixture)
    path = fixture['workspace'] / 'train.py'
    source = mutation(SOURCE).replace('assert final_results == [0.0, 0.0, 0.0]\n', '')
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c.get('path') == 'train.py')['quote'] = source
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    plan = propose_plan(PlannedLLM(fixture['proposal']), packet, fixture['workspace'])
    materialize_capture(fixture['workspace'], plan, 'a' * 64, snapshot=fixture['snapshot'])
    result = invoke(fixture['workspace'], 'capture', native_environment())
    assert result.returncode != 0
    assert reason in result.stderr
    assert not (fixture['workspace'] / '_autorepro_capture/capture.json').exists()


def test_snapshot_export_preserves_the_original_native_random_sequence(tmp_path):
    baseline = []

    def mutate(fixture):
        use_module_loop(fixture)
        source = SOURCE.replace('import argparse', 'import argparse\nimport random')
        source = source.replace('results = []', 'random.seed(19)\nnp.random.seed(23)\ntorch.manual_seed(29)\ndraws = []\nresults = []', 1)
        source = source.replace('    torch.manual_seed(args.seed + idx)\n', '')
        source = source.replace('    del model, optimizer',
            '    draws.append([random.random(), float(np.random.random()), torch.rand(1).item()])\n    del model, optimizer')
        source += "\nwith open('rng.json', 'w') as stream:\n    json.dump(draws, stream)\n"
        model_path = fixture['workspace'] / 'author_model.py'
        model = model_path.read_text('utf-8').replace('import torch', 'import torch\nimport random\nimport numpy as np')
        # Evaluation consumes RNG but predictions remain deterministic. Export
        # must leave the following native run identical to direct author execution.
        model = model.replace('        assert unused is None',
            '        assert unused is None\n        if not self.training:\n            random.random()\n            np.random.random()\n            torch.rand(1)')
        for name, text in {'train.py': source, 'author_model.py': model}.items():
            path = fixture['workspace'] / name
            path.write_text(text, encoding='utf-8')
            fixture['snapshot']['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            next(c for c in fixture['proposal']['citations'] if c.get('path') == name)['quote'] = text
        completed = subprocess.run([sys.executable, *fixture['proposal']['steps'][0]['argv'][1:]],
            cwd=fixture['workspace'], env=native_environment(), capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stderr
        baseline.extend(json.loads((fixture['workspace'] / 'rng.json').read_text('utf-8')))

    fixture, plan, spec_hash, execution = run_series(tmp_path, 'cpu', mutate)
    assert json.loads((fixture['workspace'] / 'rng.json').read_text('utf-8')) == baseline
    assert verify_capture(plan, fixture['workspace'], execution, spec_hash)['is_reproduced'] is True


def test_multiline_selection_is_resolved_but_ambiguous_guards_stop():
    source = SOURCE.replace('selected_test = (output[idx_test].argmax(1) == labels[idx_test]).double().mean().item()',
        'selected_test = (\n                (output[idx_test].argmax(1) == labels[idx_test]).double().mean().item()\n            )')
    resolved = resolve_module_loop(source, SCOPE)
    assert resolved['selection_end'] == resolved['selection_start'] + 2
    source = source.replace('    results.append(selected_test)',
        '    if val_loss < best_val_loss:\n        selected_test = 0.0\n    results.append(selected_test)')
    with pytest.raises(ValueError, match='one original source comparison'):
        resolve_module_loop(source, SCOPE)
