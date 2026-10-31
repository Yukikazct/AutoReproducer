"""A restarted task crosses the real worker boundary without reusing old data."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock
from types import ModuleType

import pytest

from frontend import backend_pipeline as backend
from frontend import task_control
from src import runtime_preparation as runtime
from src.project_paths import get_data_root, get_deps_root


def reserve_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOREPRO_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(tmp_path / 'deps'))
    original = tmp_path / 'old' / 'progress.jsonl'
    original.parent.mkdir()
    original.write_text('old progress', encoding='utf-8')
    original.with_suffix('.control.json').write_bytes(b'\x00' * 327)
    directory, _, shared = task_control._root_paths(tmp_path / 'deps', original)
    directory.mkdir()
    shared.write_bytes(b'\x00' * 327)
    return original, shared, Path(task_control.create_isolated_restart(original))


def test_restarted_background_task_and_supervisor_use_only_the_reserved_root(tmp_path, monkeypatch):
    original, shared, selected = reserve_restart(tmp_path, monkeypatch)
    old_bytes = [path.read_bytes() for path in (original, original.with_suffix('.control.json'), shared)]
    for directory in (tmp_path / 'deps' / 'old-package', tmp_path / 'training_python' / 'envs'):
        directory.mkdir(parents=True)
        (directory / 'keep').write_bytes(b'unknown owner')
    environment = dict(os.environ)
    prepared = runtime.RuntimePreparation(sys.executable, True, False, 'test', 'test')
    monkeypatch.setattr(runtime, 'prepare_runtime', Mock(return_value=prepared))
    run_owned = runtime.run_owned_process
    script = tmp_path / 'worker_fixture.py'
    script.write_text('''import json,sys
from pathlib import Path
from frontend.backend_pipeline import ProgressStore
from src.project_paths import get_data_root,get_deps_root
from src.agents.code_executor import DEPS_CACHE_ROOT
request=json.loads(sys.stdin.read())
root=get_data_root()
(root/'deps'/'new-package').mkdir(parents=True)
(root/'deps'/'new-package'/'remove').write_text('owned')
(root/'training_python'/'envs').mkdir(parents=True)
(root/'training_python'/'envs'/'remove').write_text('owned')
ProgressStore(request['progress_path'],reset=False).emit({'type':'done','result':{
    'state':'COMPLETED','data':{'roots':[str(root),str(get_deps_root()),str(DEPS_CACHE_ROOT)]}}})
''', encoding='utf-8')

    def actual_worker(argv, **kwargs):
        program = 'import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name="__main__")'
        return run_owned([sys.executable, '-I', '-c', program, str(Path(backend.__file__).parents[1]), str(script)], **kwargs)

    monkeypatch.setattr(runtime, 'run_owned_process', actual_worker)
    callbacks = []
    progress = selected / 'runtime' / 'progress_1.jsonl'
    worker = backend.run_pipeline_background(str(progress), paper_title='owned fixture',
        isolated_data_root=str(selected), on_done=lambda result: callbacks.append((get_data_root(), get_deps_root())))
    worker.join(timeout=30)
    assert not worker.is_alive()
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert not snapshot['running'], snapshot
    assert snapshot['result']['state'] == 'COMPLETED', snapshot
    assert snapshot['result']['data']['roots'] == [str(selected), str(selected / 'deps'), str(selected / 'deps')]
    assert callbacks == [(selected, selected / 'deps')]
    assert snapshot['result']['data']['environment_cleanup']['status'] == 'released'
    assert not (selected / 'deps' / 'new-package').exists()
    assert not (selected / 'training_python').exists()
    assert [path.read_bytes() for path in (original, original.with_suffix('.control.json'), shared)] == old_bytes
    assert (tmp_path / 'deps' / 'old-package' / 'keep').read_bytes() == b'unknown owner'
    assert (tmp_path / 'training_python' / 'envs' / 'keep').read_bytes() == b'unknown owner'
    assert dict(os.environ) == environment
    assert not task_control.task_status(progress)['active']
    with pytest.raises(RuntimeError, match='无法确认'):
        task_control.register_task(tmp_path / 'still-blocked.jsonl')


@pytest.mark.parametrize('arguments', [{'mock_mode': True}, {'use_docker': True}, {'workspace_dir': 'outside'}])
def test_restart_rejects_modes_and_workspaces_that_cannot_be_isolated(tmp_path, monkeypatch, arguments):
    _, _, selected = reserve_restart(tmp_path, monkeypatch)
    progress = selected / 'runtime' / 'progress_2.jsonl'
    with pytest.raises((ValueError, RuntimeError)):
        backend.run_pipeline_background(str(progress), isolated_data_root=str(selected), **arguments)
    assert not progress.exists()
    assert not progress.with_suffix('.control.json').exists()


@pytest.mark.parametrize('old_version', [1, 2])
def test_restart_rejects_a_warm_canonical_cleanup_without_scope_handoff(tmp_path, monkeypatch, old_version):
    _, _, selected = reserve_restart(tmp_path, monkeypatch)
    legacy = ModuleType('src.runtime_preparation')
    legacy.OWNED_PROCESS_API_VERSION = old_version
    legacy._clean_environment = lambda: dict(os.environ)
    monkeypatch.setitem(sys.modules, 'src.runtime_preparation', legacy)
    progress = selected / 'runtime' / 'progress_3.jsonl'
    with pytest.raises(RuntimeError, match='重启网页服务'):
        backend.run_pipeline_background(str(progress), isolated_data_root=str(selected))
    assert not progress.exists()
    assert not progress.with_suffix('.control.json').exists()
