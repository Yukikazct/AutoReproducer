"""Actual failure -> model source selection -> data recovery -> original rerun."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import author_data_repair as repair
from src.author_runtime_dependencies import run_with_dependency_repair


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, 'source_directory', lambda *args: None)
    source = ('from pathlib import Path\n'
              'path = Path.cwd() / "data" / "cora.npz"\n'
              'if not path.exists():\n'
              '    raise ValueError(f"{path} doesn\'t exist.")\n'
              'print("test_accuracy: 0.81")\n')
    (tmp_path / 'train.py').write_text(source, encoding='utf-8')
    (tmp_path / 'README.md').write_text('Download Cora NPZ data from '
        'https://github.com/example/author-data/tree/master/data/npz and put it under data/.', encoding='utf-8')
    inventory = {name: {'sha256': hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
                        'text_available': True} for name in ('train.py', 'README.md')}
    step = {'id': 'train', 'kind': 'train', 'cwd': '.', 'argv': ['python', 'train.py']}
    plan = {'scope': 'author_code_execution', 'steps': [step],
            'capture': {'device': 'cpu'}, 'requirements': {'author': [], 'compatibility': []}}
    packet = {'file_inventory': inventory}
    calls = []
    def run(workspace, steps, environment, **kwargs):
        calls.append(steps)
        completed = subprocess.run([sys.executable, 'train.py'], cwd=workspace, capture_output=True, text=True)
        final = {'id': 'train', 'executed': True, 'exit_code': completed.returncode,
                 'stdout': completed.stdout, 'stderr': completed.stderr, 'success': completed.returncode == 0}
        return {'success': completed.returncode == 0, 'executed': True, 'process_cleanup_confirmed': True,
                'final': final, 'steps': [final], 'run_dir': str(tmp_path / f'run-{len(calls)}')}
    llm = SimpleNamespace(mock_mode=False, chat=Mock(return_value=json.dumps({
        'source_id': 'source_1', 'file': 'cora.npz', 'reason': 'README identifies this exact Cora input.'})))
    return tmp_path, plan, packet, SimpleNamespace(run=run), llm, calls


def test_missing_input_is_downloaded_then_original_command_actually_succeeds(context, monkeypatch):
    root, plan, packet, runner, llm, calls = context
    download = Mock(return_value=b'actual-test-dataset-bytes')
    monkeypatch.setattr(repair, 'download_data', download)
    data = {}
    def recover(result):
        return repair.recover_author_data(llm, result, plan, packet, root, root, data, lambda: None)
    result = run_with_dependency_repair(runner, root, plan['steps'], {}, plan, packet,
        root, data, lambda _: None, lambda: None, repair_data=recover)
    assert result['success'] and len(calls) == 2 and calls[0] == calls[1] == plan['steps']
    assert 'test_accuracy' in result['final']['stdout']
    assert data['self_repair']['status'] == 'resolved'
    receipt = data['data_repair_attempts'][0]
    assert receipt['sha256'] == hashlib.sha256(b'actual-test-dataset-bytes').hexdigest()
    assert receipt['independently_verified'] is False
    download.assert_called_once_with('https://raw.githubusercontent.com/example/author-data/master/data/npz/cora.npz')
    assert llm.chat.call_args.kwargs['task'] == 'author_data_recovery'
    assert hashlib.sha256((root / 'train.py').read_bytes()).hexdigest() == packet['file_inventory']['train.py']['sha256']


@pytest.mark.parametrize('field', ['cleanup_unconfirmed', 'cancelled', 'timed_out'])
def test_data_repair_never_runs_after_terminal_lifecycle(context, field):
    root, plan, packet, runner, _, _ = context
    result = runner.run(root, plan['steps'], {})
    result[field] = True
    callback = Mock()
    run_with_dependency_repair(SimpleNamespace(run=lambda *a, **k: result), root, plan['steps'], {},
        plan, packet, root, {}, lambda _: None, lambda: None, repair_data=callback)
    callback.assert_not_called()


@pytest.mark.parametrize('filename', ['../cora.npz', '/cora.npz', 'C:/cora.npz', '.git/config', 'a/../../cora.npz'])
def test_source_path_cannot_escape_cited_directory(filename):
    with pytest.raises(ValueError):
        repair.source_download_url({'url': 'https://github.com/owner/data/tree/main/files'}, filename)


@pytest.mark.parametrize('url', [
    'https://127.0.0.1/a/b/c/d', 'https://github.com@localhost/a/b/tree/main/data',
    'https://raw.githubusercontent.com:443/a/b/main/file.npz',
    'https://github.com/a/b/tree/main/%2e%2e',
])
def test_unapproved_source_urls_rejected(url):
    with pytest.raises(ValueError):
        repair.source_download_url({'url': url}, 'cora.npz')


def test_existing_input_is_never_overwritten(context, monkeypatch):
    root, plan, packet, runner, llm, _ = context
    failed = runner.run(root, [], {})
    (root / 'data').mkdir()
    (root / 'data/cora.npz').write_bytes(b'original')
    assert repair.recover_author_data(llm, failed, plan, packet, root, root, {}, lambda: None) is None
    llm.chat.assert_not_called()
    assert (root / 'data/cora.npz').read_bytes() == b'original'


def test_changed_source_cannot_authorize_download(context):
    root, plan, packet, runner, llm, _ = context
    failed = runner.run(root, [], {})
    (root / 'README.md').write_text('https://github.com/changed/data/tree/main/files')
    with pytest.raises(ValueError, match='改变'):
        repair.recover_author_data(llm, failed, plan, packet, root, root, {}, lambda: None)
    llm.chat.assert_not_called()


def test_model_gets_validation_feedback_and_can_correct_source(context, monkeypatch):
    root, plan, packet, runner, llm, _ = context
    valid = llm.chat.return_value
    llm.chat.side_effect = ['{"source_id":"invented","file":"cora.npz","reason":"data"}', valid]
    monkeypatch.setattr(repair, 'download_data', lambda _: b'dataset')
    data = {}
    receipt = repair.recover_author_data(llm, runner.run(root, [], {}), plan, packet, root, root, data, lambda: None)
    assert receipt and llm.chat.call_count == 2
    assert json.loads(llm.chat.call_args.args[0])['errors']


def test_nonterminal_missing_file_does_not_trigger_recovery(context):
    root, _, packet, runner, _, _ = context
    failed = runner.run(root, [], {})
    failed['final']['stderr'] += '\nRuntimeError: unrelated final failure\n'
    assert repair.missing_data_target(failed, root, packet) is None


@pytest.mark.parametrize('content', [b'', b'<html>login</html>', b'version https://git-lfs.github.com/spec/v1\n'])
def test_download_rejects_empty_webpage_and_lfs_pointer(monkeypatch, content):
    response = Mock(status_code=200, headers={})
    response.iter_content.return_value = [content]
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(repair.requests, 'get', Mock(return_value=response))
    with pytest.raises(ValueError, match='没有实际数据'):
        repair.download_data('https://raw.githubusercontent.com/author/data/main/input.npz')


def test_download_race_preserves_existing_file(context, monkeypatch):
    root, plan, packet, runner, llm, _ = context
    def download(_):
        (root / 'data').mkdir(exist_ok=True)
        (root / 'data/cora.npz').write_bytes(b'concurrent-original')
        return b'new-download'
    monkeypatch.setattr(repair, 'download_data', download)
    data = {}
    assert repair.recover_author_data(llm, runner.run(root, [], {}), plan, packet, root, root, data, lambda: None) is None
    assert (root / 'data/cora.npz').read_bytes() == b'concurrent-original'
    assert data['data_recovery_analyses'][0]['status'] == 'blocked'


def test_data_directory_exposes_only_actual_files_in_requested_directory():
    payload = {'payload': {'codeViewTreeRoute': {'path': 'data/npz',
        'refInfo': {'currentOid': 'a' * 40}, 'tree': {'items': [
            {'name': 'cora.npz', 'path': 'data/npz/cora.npz', 'contentType': 'file'},
            {'name': 'other.npz', 'path': 'elsewhere/other.npz', 'contentType': 'file'},
            {'name': 'folder', 'path': 'data/npz/folder', 'contentType': 'directory'}]}}}}
    html = '<script type="application/json" data-target="react-app.embeddedData">' + json.dumps(payload) + '</script>'
    listing = repair.parse_data_directory(html, 'data/npz')
    assert listing['files'] == ['cora.npz'] and listing['revision'] == 'a' * 40
    assert repair.parse_data_directory(html, 'unrelated') is None


def test_live_directory_feedback_corrects_wrong_file_and_pins_download_revision(context, monkeypatch):
    root, plan, packet, runner, llm, _ = context
    valid = llm.chat.return_value
    llm.chat.side_effect = [json.dumps({'source_id': 'source_1', 'file': 'wrong.npz', 'reason': 'data'}), valid]
    monkeypatch.setattr(repair, 'source_directory', lambda *a: {'files': ['cora.npz'], 'revision': 'a' * 40})
    download = Mock(return_value=b'data')
    monkeypatch.setattr(repair, 'download_data', download)
    receipt = repair.recover_author_data(llm, runner.run(root, [], {}), plan, packet, root, root, {}, lambda: None)
    assert receipt and llm.chat.call_count == 2
    feedback = json.loads(llm.chat.call_args.args[0])['previous_selections'][0]
    assert feedback['file'] == 'wrong.npz' and feedback['directory']['files'] == ['cora.npz']
    assert download.call_args.args[0] == 'https://raw.githubusercontent.com/example/author-data/' + 'a' * 40 + '/data/npz/cora.npz'
