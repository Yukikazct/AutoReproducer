import builtins
import hashlib

import pytest

from src.experiments.author_output_capture import AppendOutputCapture
from src.author_output_metrics import read_output_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe(tmp_path, monkeypatch, script, *, initial=b'test_acc\n0.99\n', tracked=True, inputs=()):
    source = tmp_path / 'train.py'
    source.write_text(script, encoding='utf-8')
    log = tmp_path / 'log.txt'
    if initial is not None:
        log.write_bytes(initial)
    sources = {'train.py': digest(source)}
    if tracked and log.exists():
        sources['log.txt'] = digest(log)
    monkeypatch.chdir(tmp_path)
    capture = AppendOutputCapture(tmp_path, tmp_path / '_autorepro_capture', sources, 'train', inputs)
    original_open = builtins.open
    try:
        capture.install()
        exec(compile(script, str(source), 'exec'), {'__name__': '__main__'})
        outputs = capture.finish()
    finally:
        capture.close()
    assert builtins.open is original_open
    return outputs, sources, log


def test_new_csv_metrics_are_captured_without_reading_historical_example(tmp_path, monkeypatch):
    original = b'tau,lr,test_acc\n2,0.01,0.99\n'
    outputs, sources, log = observe(tmp_path, monkeypatch,
        "with open('log.txt', encoding='utf-8', mode='a+') as out:\n"
        "    print('tau,lr,data,test_acc', file=out)\n"
        "    print('2,0.01,cora,0.81', file=out)\n", initial=original)
    assert log.read_bytes() == original
    assert len(outputs) == 1 and outputs[0]['redirected'] is True
    rows, errors = read_output_metrics([{'step': {'id': 'train'}, 'natural_return': True,
                                       'output_files': outputs}], tmp_path, sources)
    assert errors == [] and len(rows) == 1
    assert rows[0]['value'] == .81 and rows[0]['name'] == 'accuracy'
    assert rows[0]['split'] == 'test'
    assert rows[0]['source_quote'] == 'tau,lr,data,test_acc\n2,0.01,cora,0.81'
    assert ':file:log.txt@bytes' in rows[0]['source']


def test_untracked_outputs_keep_their_original_path_and_file_exists_behavior(tmp_path, monkeypatch):
    outputs, sources, log = observe(tmp_path, monkeypatch,
        "import os\nwith open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.82', file=out)\n"
        "assert os.path.exists('log.txt')\n", initial=None, tracked=False)
    assert log.read_text().strip() == 'Test Accuracy: 0.82'
    rows, errors = read_output_metrics([{'step': {'id': 'train'}, 'natural_return': True,
                                       'output_files': outputs}], tmp_path, sources)
    assert not errors and rows[0]['value'] == .82
    assert outputs[0]['redirected'] is False


def test_input_files_are_never_redirected_or_reported_as_metrics(tmp_path, monkeypatch):
    outputs, _, log = observe(tmp_path, monkeypatch,
        "with open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.81', file=out)\n",
        inputs=['log.txt'])
    assert outputs == [] and '.81' in log.read_text()


def test_non_author_writes_do_not_create_captures(tmp_path):
    capture = AppendOutputCapture(tmp_path, tmp_path / '_autorepro_capture', {}, 'train')
    try:
        capture.install()
        with open(tmp_path / 'log.txt', 'a') as out:
            out.write('Test Accuracy: 0.91')
        assert capture.finish() == []
    finally:
        capture.close()


def test_truncated_or_rewritten_prefix_is_not_accepted(tmp_path, monkeypatch):
    outputs, _, log = observe(tmp_path, monkeypatch,
        "with open('log.txt', 'a+') as out:\n    out.truncate(0)\n    out.write('Test Accuracy: 0.81')\n")
    assert outputs == [] and log.read_bytes() == b'test_acc\n0.99\n'


def test_separate_attempts_cannot_mix_failed_and_successful_log_suffixes(tmp_path, monkeypatch):
    code = "with open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.81', file=out)\n"
    first, _, _ = observe(tmp_path, monkeypatch, code)
    second, _, _ = observe(tmp_path, monkeypatch, code)
    assert first[0]['captured_path'] != second[0]['captured_path']
    for output in (first[0], second[0]):
        assert (tmp_path / output['captured_path']).read_text().count('Test Accuracy: 0.81') == 1


@pytest.mark.parametrize('mutate', [
    lambda row: row.update(step_id='other'),
    lambda row: row.update(captured_sha256='0' * 64),
    lambda row: row.update(original_bytes=-1),
    lambda row: row.update(source_sha256='0' * 64),
    lambda row: row.update(captured_path='../outside.txt'),
    lambda row: row.update(captured_path='_autorepro_capture/output_logs/other/log.txt'),
])
def test_changed_or_unbound_output_cannot_supply_metrics(tmp_path, monkeypatch, mutate):
    outputs, sources, _ = observe(tmp_path, monkeypatch,
        "with open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.81', file=out)\n")
    mutate(outputs[0])
    rows, errors = read_output_metrics([{'step': {'id': 'train'}, 'natural_return': True,
                                       'output_files': outputs}], tmp_path, sources)
    assert rows == [] and errors


def test_failed_run_and_changed_bytes_cannot_supply_metrics(tmp_path, monkeypatch):
    outputs, sources, _ = observe(tmp_path, monkeypatch,
        "with open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.81', file=out)\n")
    receipt = {'step': {'id': 'train'}, 'natural_return': False, 'output_files': outputs}
    assert read_output_metrics([receipt], tmp_path, sources)[0] == []
    receipt['natural_return'] = True
    (tmp_path / outputs[0]['captured_path']).write_text('Test Accuracy: 0.99')
    rows, errors = read_output_metrics([receipt], tmp_path, sources)
    assert rows == [] and errors


def test_stream_metrics_survive_a_malformed_optional_file_receipt(tmp_path):
    from src.author_claim_comparison import collect_author_metrics
    execution = {'success': True, 'steps': [{'id': 'train', 'success': True, 'exit_code': 0,
                  'stdout': 'Test Accuracy: 0.81'}], 'author_output_receipts': [None, {'step': {}, 'output_files': [None]}]}
    rows, errors = collect_author_metrics(execution, workspace=tmp_path, sources={})
    assert rows[0]['value'] == .81 and errors


def test_output_metrics_join_the_production_metric_collection(tmp_path, monkeypatch):
    from src.author_claim_comparison import collect_author_metrics
    outputs, sources, _ = observe(tmp_path, monkeypatch,
        "with open('log.txt', 'a') as out:\n    print('Test Accuracy: 0.81', file=out)\n")
    execution = {'success': True, 'steps': [{'id': 'train', 'success': True, 'exit_code': 0, 'stdout': ''}],
        'author_output_receipts': [{'step': {'id': 'train'}, 'natural_return': True, 'output_files': outputs}]}
    rows, errors = collect_author_metrics(execution, workspace=tmp_path, sources=sources)
    assert not errors and len(rows) == 1 and rows[0]['value'] == .81
    assert rows[0]['id'] == 'train:file:0'
