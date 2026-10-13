"""Invented native series mechanics, never independent paper acceptance."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

from discovered_fixtures import PlannedLLM, make_experiment
from src.discovered_repository_plan import build_evidence_packet, propose_plan
from src.discovered_repository_reproduction import environment_for_plan
from src.discovered_repository_runtime import materialize_capture, validate_capture, verify_capture
from src.method_adapters import read_json, write_json
from src.repository_reproduction import spec_digest
from src.runtime_platform import runtime_fingerprint
from test_discovered_repository_runtime import invoke, native_environment


def series_fixture(root, device="cpu"):
    fixture = make_experiment(root)
    workspace = fixture["workspace"]
    model_source = '''import torch
class SignalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, inputs, normalize):
        output = self.linear(inputs[0] + inputs[1])
        return torch.nn.functional.normalize(output) if normalize else output
'''
    source = '''import argparse
import json
import torch
import numpy as np
from author_model import SignalModel
from common import NATIVE
assert NATIVE is True
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
device = torch.device('cpu' if args.cpu else 'cuda')
normalize = False
def train(run):
    torch.manual_seed(args.seed + run)
    with open('dataset.json') as stream:
        records = json.load(stream)
    features = torch.tensor(records['features'], device=device)
    labels = torch.tensor(records['labels'], dtype=torch.long, device=device)
    if run == 1:
        labels[-1] = 0
    idx_test = torch.tensor([8, 9, 10, 11], device=device)
    inputs = [features, features * 0.5]
    model = SignalModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs, normalize)[:8], labels[:8])
        loss.backward()
        optimizer.step()
        if epoch + 1 == 8 + run * 2:
            break
    model.eval()
    with torch.no_grad():
        output = model(inputs, normalize)
        accuracy = (output[idx_test].argmax(1) == labels[idx_test]).double().mean().item()
    model.train()
    return accuracy, model
results = [train(run) for run in range(3)]
assert all(model.training and next(model.parameters()).device.type == device.type for _, model in results)
accuracies = [accuracy for accuracy, _ in results]
print('Test Accuracy:', np.mean(accuracies))
'''
    for name, text in {"train.py": source, "author_model.py": model_source}.items():
        path = workspace / name
        path.write_text(text, encoding="utf-8")
        fixture["snapshot"]["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        next(citation for citation in fixture["proposal"]["citations"] if citation["source"] == "repository"
             and citation.get("path") == name)["quote"] = text
    (workspace / "common.py").write_text("NATIVE = True\n", encoding="utf-8")
    fixture["snapshot"]["files"]["common.py"] = hashlib.sha256((workspace / "common.py").read_bytes()).hexdigest()
    proposal = fixture["proposal"]
    proposal["capture"].update(
        kind="torch_classification_series", device=device, scope={"path": "train.py", "function": "train"},
        forward_args=["inputs", "normalize"], expected_runs=3, expected_test_samples=[4, 4, 4],
        completion={"kind": "author_epoch_counter", "epoch_counter": "epoch", "epoch_offset": 1,
                    "updates_per_epoch": 1}, aggregation="mean",
    )
    if device == "cuda":
        proposal["steps"][0]["argv"].remove("--cpu")
        proposal["requirements"]["compatibility"][0] = "torch==2.5.1+cu121"
    packet = build_evidence_packet(fixture["pdf"], workspace, fixture["snapshot"])
    fixture["packet"] = packet
    return fixture


def run_series(root, device="cpu", mutate=None):
    cache = cuda_dependency_cache() if device == "cuda" else None
    fixture = series_fixture(root, device)
    if mutate:
        mutate(fixture)
        fixture['packet'] = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    plan = propose_plan(PlannedLLM(fixture["proposal"]), fixture["packet"], fixture["workspace"])
    spec_hash = spec_digest(plan)
    manifest = materialize_capture(fixture["workspace"], plan, spec_hash, snapshot=fixture["snapshot"])
    environment = native_environment()
    if device == "cuda":
        extra = environment.get('AUTOREPRO_TEST_CUDA_DEPS', '')
        environment["PYTHONPATH"] = str(cache) + (os.pathsep + extra if extra else '')
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    log_root = root / "owned_logs"
    log_root.mkdir()
    records = []
    for step in manifest["steps"]:
        if step["kind"] == "eval":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = invoke(fixture["workspace"], "capture" if step["kind"] == "train" else "evaluate", environment)
        assert result.returncode == 0, result.stderr
        log = log_root / (step["id"] + ".stdout.log")
        log.write_text(result.stdout, encoding="utf-8")
        records.append({"id": step["id"], "success": True, "executed": True, "exit_code": 0,
                        "stdout_path": str(log), "timed_out": False, "cancelled": False})
    execution = {"success": True, "steps": records, "skipped": [], "run_dir": str(log_root)}
    return fixture, plan, spec_hash, execution


def use_selected_test_rows(fixture):
    """Author selects its own output rows; full NumPy labels stay in author locals."""
    path = fixture['workspace'] / 'author_model.py'
    source = path.read_text('utf-8').replace('def forward(self, inputs, normalize):',
                                           'def forward(self, inputs, normalize, selected):')
    source = source.replace('return torch.nn.functional.normalize(output) if normalize else output',
                            'return (torch.nn.functional.normalize(output) if normalize else output)[selected]')
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c.get('path') == path.name)['quote'] = source
    path = fixture['workspace'] / 'train.py'
    source = path.read_text('utf-8').replace('model(inputs, normalize)[:8]',
        'model(inputs, normalize, torch.arange(8, device=device))')
    source = source.replace('output = model(inputs, normalize)', 'output = model(inputs, normalize, idx_test)')
    source = source.replace('output[idx_test].argmax(1)', 'output.argmax(1)')
    source = source.replace('    model.train()\n    return accuracy, model',
        "    splits = {'test': idx_test}\n    labels_all = labels.cpu().numpy().astype(np.int32)\n"
        '    model.train()\n    return accuracy, model')
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c.get('path') == path.name)['quote'] = source
    selector = {'name': 'splits', 'keys': ['test']}
    fixture['proposal']['capture'].update(forward_args=['inputs','normalize',selector],
        labels='labels_all', test_indices=selector, output_alignment='selected_test')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dictionary_test_selector_numpy_labels_and_already_selected_outputs(tmp_path, device):
    if device == 'cuda':
        cuda_dependency_cache()
    fixture, plan, spec_hash, execution = run_series(tmp_path, device, use_selected_test_rows)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['protocol_pass'] and result['independent_metrics_pass']
    assert [row['accuracy'] for row in result['independent_metrics']['runs']] == [1.0,0.75,1.0]
    assert all(row['author_label_dtype'] == 'int32' and row['output_alignment'] == 'selected_test'
               for row in result['capture']['runs'])


def test_selected_outputs_require_the_same_complete_test_indices_as_a_forward_input(tmp_path):
    fixture = series_fixture(tmp_path)
    capture = deepcopy(fixture['proposal']['capture'])
    capture['output_alignment'] = 'selected_test'
    with pytest.raises(ValueError, match='same complete author test indices'):
        validate_capture(capture, fixture['packet'], fixture['workspace'])


def cuda_dependency_cache():
    root = Path(__file__).resolve().parents[1] / "data/deps/repository" / runtime_fingerprint()
    caches = [path for path in root.glob("*") if (path / ".ready").is_file()
              and (path / "torch-2.5.1+cu121.dist-info").is_dir()
              and (path / "numpy-1.26.4.dist-info").is_dir()]
    if not caches:
        pytest.skip("Native CUDA test requires an installed reviewed CUDA cache")
    return caches[0]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    return run_series(tmp_path_factory.mktemp("native_series"))


def test_all_native_function_returns_and_independent_mean(completed):
    fixture, plan, spec_hash, execution = completed
    result = verify_capture(plan, fixture["workspace"], execution, spec_hash)
    assert result["protocol_pass"] and result["independent_metrics_pass"]
    assert [run["optimizer_steps"] for run in result["capture"]["runs"]] == [8, 10, 12]
    values = [run["accuracy"] for run in result["independent_metrics"]["runs"]]
    assert values == [1.0, 0.75, 1.0]
    assert result["independent_metrics"]["accuracy"] == sum(values) / 3
    assert result["author_reported_metric"]["value"] == sum(values) / 3
    assert result["is_reproduced"] is False  # The frozen 100% reference remains unmet.


@pytest.mark.parametrize("mutation", [
    lambda capture: capture.update(expected_runs=2),
    lambda capture: capture["scope"].update(function="train()"),
    lambda capture: capture["scope"].update(path="../train.py"),
    lambda capture: capture.update(aggregation="best"),
    lambda capture: capture["completion"].update(epoch_counter="epoch+1"),
    lambda capture: capture["completion"].update(epoch_offset=True),
    lambda capture: capture["completion"].update(updates_per_epoch=0),
])
def test_series_scope_and_completion_schema_fail_closed(tmp_path, mutation):
    fixture = series_fixture(tmp_path)
    capture = deepcopy(fixture["proposal"]["capture"])
    mutation(capture)
    with pytest.raises(ValueError):
        validate_capture(capture, fixture["packet"], fixture["workspace"])


@pytest.mark.parametrize("runs,offset", [(2, 1), (4, 1), (3, 0)])
def test_actual_native_returns_and_updates_override_claimed_completion(tmp_path, runs, offset):
    fixture = series_fixture(tmp_path)
    plan = fixture["proposal"]
    plan["capture"].update(expected_runs=runs, expected_test_samples=[4] * runs)
    plan["capture"]["completion"]["epoch_offset"] = offset
    materialize_capture(fixture["workspace"], plan, "a" * 64, snapshot=fixture["snapshot"])
    result = invoke(fixture["workspace"], "capture", native_environment())
    assert result.returncode != 0
    assert not (fixture["workspace"] / "_autorepro_capture/capture.json").exists()


def test_fresh_series_metrics_cannot_omit_or_replace_a_repetition(completed):
    fixture, plan, spec_hash, execution = completed
    path = fixture["workspace"] / "_autorepro_capture/independent_metrics.json"
    saved = path.read_bytes()
    try:
        metrics = read_json(path)
        metrics["runs"].pop(1)
        write_json(path, metrics)
        with pytest.raises(ValueError, match="Every declared native repetition"):
            verify_capture(plan, fixture["workspace"], execution, spec_hash)
    finally:
        path.write_bytes(saved)


def test_native_cuda_series_exports_all_models_for_fresh_cpu_evaluation(tmp_path):
    fixture, plan, spec_hash, execution = run_series(tmp_path, "cuda")
    result = verify_capture(plan, fixture["workspace"], execution, spec_hash)
    assert result["capture"]["training_device"] == "cuda"
    assert result["independent_metrics"]["device"] == "cpu"
    assert len(result["independent_metrics"]["runs"]) == 3
    environment = environment_for_plan(plan)
    assert "https://download.pytorch.org/whl/cu121" in environment["requirements_txt"]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sparse_native_series_preserves_every_model_and_run(tmp_path, device):
    if device == 'cuda':
        cuda_dependency_cache()
    def sparse_model(fixture):
        root = fixture['workspace']
        model = (root / 'author_model.py').read_text('utf-8')
        model = model.replace('forward(self, inputs, normalize)', 'forward(self, inputs, normalize, adj)')
        model = model.replace('self.linear(inputs[0] + inputs[1])',
                              'self.linear(torch.sparse.mm(adj, inputs[0] + inputs[1]))')
        script = (root / 'train.py').read_text('utf-8')
        script = script.replace('    model = SignalModel().to(device)',
            '    adj = torch.eye(len(features), device=device).to_sparse()\n    model = SignalModel().to(device)')
        script = script.replace('model(inputs, normalize)', 'model(inputs, normalize, adj)')
        for name, text in {'train.py':script, 'author_model.py':model}.items():
            (root / name).write_text(text, encoding='utf-8')
            fixture['snapshot']['files'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            next(item for item in fixture['proposal']['citations'] if item.get('path') == name)['quote'] = text
        fixture['proposal']['capture']['forward_args'].append('adj')
    fixture, plan, spec_hash, execution = run_series(tmp_path, device, mutate=sparse_model)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['protocol_pass'] and result['independent_metrics_pass']
    assert [item['optimizer_steps'] for item in result['capture']['runs']] == [8, 10, 12]
    assert result['capture']['training_device'] == device


def test_bundled_npz_split_summary_is_from_actual_hash_bound_arrays(tmp_path):
    fixture = make_experiment(tmp_path)
    path = fixture["workspace"] / "split.npz"
    np.savez(path, test_mask=np.array([False, True, True, False]), train_indices=np.array([0, 3]))
    fixture["snapshot"]["files"]["split.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    evidence = packet["file_inventory"]["split.npz"]["dataset_evidence"]
    summary = json.loads(evidence["text"])
    assert summary["arrays"]["test_mask"]["selected_samples"] == 2
    assert summary["arrays"]["train_indices"]["unique_values"] == 2
    assert evidence["sha256"] == fixture["snapshot"]["files"]["split.npz"]


def test_npz_header_cannot_request_unbounded_allocation(tmp_path):
    fixture = make_experiment(tmp_path)
    path = fixture["workspace"] / "dangerous.npz"
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"shape": (10**12,), "fortran_order": False, "descr": "<f8"})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("test_mask.npy", stream.getvalue())
    fixture["snapshot"]["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"])
    assert "dataset_evidence" not in packet["file_inventory"][path.name]


def test_complete_cuda_series_service_runs_native_preflight_and_owned_evaluation(tmp_path, monkeypatch):
    from src import discovered_repository_reproduction as service
    from src.audit.audit_logger import AuditLogger
    from test_discovered_repository_service import use_preinstalled_cache
    cache = cuda_dependency_cache()
    fixture = series_fixture(tmp_path, "cuda")
    snapshot = {**fixture["snapshot"], "resolved_sha": fixture["snapshot"]["revision"]}

    def export(root, profile, workspace, **kwargs):
        shutil.copytree(fixture["workspace"], workspace)
        return deepcopy(snapshot)

    monkeypatch.setattr(service, "export_repository", export)
    llm = PlannedLLM(fixture["proposal"])
    logger = AuditLogger(log_dir=str(tmp_path / "logs"), ledger_dir=str(tmp_path / "ledger"))
    runner = service.DiscoveredRepositoryReproduction(tmp_path / "service", logger, llm=llm)
    use_preinstalled_cache(monkeypatch, runner, cache)
    result = runner.run({"pdf_path": str(fixture["pdf"]), "offline": True,
                         "resources": {"code_repo_url": snapshot["url"], "repo_discovery": {
                             "selected_repo": snapshot["url"], "pinned_revision": snapshot["revision"]}}})
    assert result["state"] == "COMPLETED", result.get("error")
    data = result["data"]
    assert data["native_preflight"]["device"] == "cuda"
    assert data["native_preflight"]["native_probe_pass"] is True
    assert len(data["validation"]["independent_metrics"]["runs"]) == 3
    assert data["validation"]["protocol_pass"] is True
    assert data["validation"]["is_reproduced"] is False
    assert data["execution"]["success"] is True
