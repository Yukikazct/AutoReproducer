"""Attribute cancellation without weakening owner generations or cleanup gates."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontend import backend_pipeline as backend
from frontend import task_control as tasks
from src import runtime_preparation as runtime


@pytest.fixture(autouse=True)
def isolated_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(tasks, '_blocked_roots', {})
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(tmp_path / 'dependencies'))


@pytest.mark.parametrize('control', [None, SimpleNamespace(), SimpleNamespace(cancellation=None)])
def test_legacy_owner_without_provenance_still_formats_cancellation(control):
    assert backend._cancellation_details(control) == {'source': 'unknown'}
    message = backend._cancellation_message(control, 'worker interrupted')
    assert '来源未记录' in message and '用户' not in message
    assert 'worker interrupted' in message


@pytest.mark.parametrize('source', ['user', 'time_budget', 'agent', 'system'])
def test_source_survives_finish_but_cannot_cancel_the_next_owner(tmp_path, source):
    path = tmp_path / 'progress.jsonl'
    control = tasks.register_task(path)
    try:
        assert tasks.request_cancel(path, source=source)
        assert control.cancelled()
        details = control.cancellation()
        assert details['source'] == source and details['requested_at']
        assert tasks.task_status(path)['cancellation'] == details
    finally:
        control.finish()
    saved = tasks.task_status(path)
    assert saved['record']['cancellation'] == details
    assert not saved['cancelling']
    current = tasks.register_task(path)
    try:
        assert not current.cancelled()
        assert current.cancellation() == {'source': 'unknown'}
        current.cancel_file.write_text(control.token, encoding='ascii')
        assert not current.cancelled()
    finally:
        current.finish()


def external_cancel(path, source):
    program = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from frontend.task_control import request_cancel;"
        "print(request_cancel(sys.argv[2],source=sys.argv[3]))")
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', program,
        str(Path(backend.__file__).resolve().parents[1]), str(path), source],
        capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'True'


@pytest.mark.parametrize('external_first', [False, True])
def test_first_cancellation_source_wins_across_processes(tmp_path, external_first):
    path = tmp_path / 'progress.jsonl'
    control = tasks.register_task(path)
    try:
        if external_first:
            external_cancel(path, 'time_budget')
        else:
            assert tasks.request_cancel(path, source='time_budget')
        first = control.cancellation()
        external_cancel(path, 'user')
        assert tasks.request_cancel(path, source='agent')
        assert control.cancelled()
        assert control.cancellation() == first
        assert first['source'] == 'time_budget'
        # An old cross-process writer must not erase the new owner's request.
        control.event.clear()
        control.cancel_file.write_text('f' * 32, encoding='ascii')
        assert control.cancelled()
    finally:
        control.finish()


def test_default_ui_cancel_retains_user_attribution(tmp_path):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    try:
        assert tasks.request_cancel(control.path)
        assert control.cancellation()['source'] == 'user'
    finally:
        control.finish()
    assert tasks.request_cancel(control.path, source='agent') is False


@pytest.mark.parametrize('source', ['', 'budget with arbitrary text', None, {}, 'USER'])
def test_invalid_source_does_not_signal_cancellation(tmp_path, source):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    try:
        with pytest.raises(ValueError, match='source'):
            tasks.request_cancel(control.path, source=source)
        assert not control.cancelled()
    finally:
        control.finish()


@pytest.mark.parametrize('raw', ['{', '{}', '{"source":"user"}'])
def test_malformed_provenance_is_not_an_authenticated_request(tmp_path, raw):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    try:
        tasks._cancellation_path(control.path, control.token).write_text(raw, encoding='utf-8')
        assert not control.cancelled()
        control.cancel_file.write_text(control.token, encoding='ascii')
        assert control.cancelled()  # Legacy token-only cancellation still works.
        assert control.cancellation() == {'source': 'unknown'}
    finally:
        control.finish()


def test_metadata_write_error_does_not_block_stopping(tmp_path, monkeypatch):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    original = Path.open
    def open_file(path, *args, **kwargs):
        if path == tasks._cancellation_path(control.path, control.token):
            raise PermissionError('metadata unavailable')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_file)
    try:
        assert tasks.request_cancel(control.path, source='time_budget')
        assert control.cancelled()
        assert control.cancellation() == {'source': 'unknown'}
    finally:
        control.finish()


def test_late_named_request_does_not_relabel_an_earlier_legacy_stop(tmp_path):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    try:
        control.cancel_file.write_text(control.token, encoding='ascii')
        assert tasks.request_cancel(control.path, source='user')
        assert control.cancelled()
        assert control.cancellation() == {'source': 'unknown'}
    finally:
        control.finish()


def test_token_inside_metadata_must_match_the_current_generation(tmp_path):
    control = tasks.register_task(tmp_path / 'progress.jsonl')
    try:
        record = {'version': 1, 'token': '0' * 32, 'source': 'user',
                  'requested_at': '2026-10-30T05:00:00+00:00'}
        tasks._cancellation_path(control.path, control.token).write_text(json.dumps(record), encoding='utf-8')
        assert not control.cancelled()
        assert control.cancellation() == {'source': 'unknown'}
    finally:
        control.finish()


def test_legacy_token_characters_cannot_change_metadata_directory(tmp_path):
    for token in ('../' * 10 + '..', '..\\' * 10 + '..'):
        path = tasks._cancellation_path(tmp_path / 'progress.jsonl', token)
        assert path.parent == tmp_path
        assert path.suffix == '.json'


@pytest.mark.parametrize('source,word', [
    ('user', '用户已停止'), ('time_budget', '时间预算'), ('agent', '代理'),
    ('system', '系统'), (None, '来源未记录'),
])
def test_bootstrap_cancellation_retains_source_in_result_and_installation(tmp_path, monkeypatch, source, word):
    path = tmp_path / 'progress.jsonl'
    ready = []
    def prepare(*args, cancel_check, progress_callback, **kwargs):
        progress_callback({'type': 'dependency_install_progress', 'installation_id': 'fixture',
                           'status': 'running', 'phase': 'download'})
        ready.append(True)
        while not cancel_check():
            time.sleep(.01)
        raise KeyboardInterrupt('运行环境准备已取消')
    monkeypatch.setattr(runtime, 'prepare_runtime', prepare)
    thread = backend.run_pipeline_background(str(path), paper_title='cancellation attribution')
    try:
        deadline = time.monotonic() + 10
        while not ready and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready
        if source is None:
            tasks._tasks[tasks._key(path)].event.set()
        else:
            assert tasks.request_cancel(path, source=source)
        thread.join(10)
        assert not thread.is_alive()
        view = backend.ProgressStore.read_snapshot(str(path))
        result = view['result']
        assert result['state'] == 'ERROR' and result['data']['run_status'] == 'interrupted'
        assert result['data']['cancellation']['source'] == (source or 'unknown')
        assert word in result['error']
        assert word in view['dependency_installations'][0]['message']
        if source != 'user':
            assert '用户' not in result['error']
        assert not tasks.task_status(path)['active']
    finally:
        if thread.is_alive():
            tasks.request_cancel(path)
            thread.join(10)


def test_worker_checkpoint_keeps_cancellation_source_in_saved_result(tmp_path, monkeypatch):
    path, run = tmp_path / 'progress.jsonl', tmp_path / 'run'
    run.mkdir()
    retained = {'state': 'ERROR', 'error': 'author interrupted', 'data': {
        'run_dir': str(run), 'interrupted': True, 'validation': {'status': 'interrupted'}}}
    (run / 'result.json').write_text(json.dumps(retained), encoding='utf-8')
    control = tasks.register_task(path)
    tasks.bind_task(control)
    store = backend.ProgressStore(str(path))
    monkeypatch.setattr(runtime, 'prepare_runtime', Mock(return_value=runtime.RuntimePreparation(
        sys.executable, True, False, 'fixture', 'fixture')))
    monkeypatch.setattr(runtime, 'release_worker_environment', Mock(return_value={'status': 'released', 'errors': []}))
    def execute(argv, **kwargs):
        store.emit({'type': 'result_checkpoint', 'result': retained})
        assert tasks.request_cancel(path, source='time_budget')
        kwargs['on_cleanup']()
        raise KeyboardInterrupt('worker shutdown')
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    try:
        with pytest.raises(KeyboardInterrupt):
            backend._run_prepared_pipeline(store, {}, 'fixture')
        view = store.read_snapshot(str(path))
        assert view['done'] and not view['active_workers']
        result = view['result']
        assert result['data']['cancellation']['source'] == 'time_budget'
        assert '用户' not in result['error']
        assert json.loads((run / 'result.json').read_text('utf-8'))['data'] == result['data']
    finally:
        control.finish()
        tasks.bind_task(None)
