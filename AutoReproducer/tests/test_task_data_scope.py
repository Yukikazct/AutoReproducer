"""Task storage stays local to a context and its explicitly launched workers."""
import asyncio
import builtins
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import threading
from unittest.mock import Mock

import pytest

from src import runtime_preparation as runtime
from src.project_paths import (PROJECT_ROOT, get_data_root, get_deps_root,
                               get_task_data_root, task_data_scope)


def test_scope_overrides_process_roots_but_preserves_explicit_paths_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOREPRO_DATA_ROOT', str(tmp_path / 'shared'))
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(tmp_path / 'shared-packages'))
    before = dict(os.environ)
    task_root = tmp_path / 'task'
    with task_data_scope(task_root) as selected:
        assert selected == get_task_data_root() == get_data_root() == task_root
        assert get_deps_root() == task_root / 'deps'
        assert get_data_root(tmp_path / 'explicit') == tmp_path / 'explicit'
        assert get_deps_root(tmp_path / 'packages') == tmp_path / 'packages'
        with pytest.raises(KeyboardInterrupt):
            with task_data_scope(tmp_path / 'nested'):
                assert get_data_root() == tmp_path / 'nested'
                raise KeyboardInterrupt('cancelled')
        assert get_data_root() == task_root
        assert dict(os.environ) == before
    assert get_task_data_root() is None
    assert get_data_root() == tmp_path / 'shared'
    assert get_deps_root() == tmp_path / 'shared-packages'
    assert dict(os.environ) == before


@pytest.mark.parametrize('value', [None, '', '  '])
def test_scope_rejects_an_unspecified_root_without_changing_context(value):
    before = get_data_root()
    with pytest.raises(ValueError, match='explicit nonempty'):
        with task_data_scope(value):
            pytest.fail('invalid task root entered')
    assert get_task_data_root() is None and get_data_root() == before


def test_inherited_scope_marker_restores_after_an_explicit_nested_scope(tmp_path, monkeypatch):
    inherited = tmp_path / 'inherited-task'
    monkeypatch.setenv('AUTOREPRO_TASK_DATA_ROOT', str(inherited))
    monkeypatch.setenv('AUTOREPRO_DATA_ROOT', str(tmp_path / 'shared'))
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(tmp_path / 'shared-packages'))
    before = dict(os.environ)
    assert get_task_data_root() == get_data_root() == inherited
    assert get_deps_root() == inherited / 'deps'
    assert get_data_root(tmp_path / 'explicit') == tmp_path / 'explicit'
    assert get_deps_root(tmp_path / 'packages') == tmp_path / 'packages'
    with task_data_scope(tmp_path / 'nested-task'):
        assert get_task_data_root() == tmp_path / 'nested-task'
    assert get_task_data_root() == get_data_root() == inherited
    assert dict(os.environ) == before


def test_concurrent_threads_use_their_own_scope_without_changing_the_parent(tmp_path):
    before = dict(os.environ)
    parent_root = get_data_root()
    barrier = threading.Barrier(3)

    def selected(root):
        assert get_task_data_root() is None
        with task_data_scope(root):
            barrier.wait(timeout=5)
            result = (get_data_root(), get_deps_root())
            barrier.wait(timeout=5)
            return result

    with task_data_scope(tmp_path / 'parent'):
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(selected, tmp_path / name) for name in ('first', 'second')]
            barrier.wait(timeout=5)
            assert get_data_root() == tmp_path / 'parent'
            barrier.wait(timeout=5)
            assert [future.result(timeout=5) for future in futures] == [
                (tmp_path / name, tmp_path / name / 'deps') for name in ('first', 'second')]
    assert get_data_root() == parent_root and dict(os.environ) == before


def test_interleaved_async_tasks_do_not_share_their_scope(tmp_path):
    before = dict(os.environ)

    async def selected(root):
        with task_data_scope(root):
            await asyncio.sleep(0)
            return get_data_root(), get_deps_root()

    async def run():
        return await asyncio.gather(*(selected(tmp_path / name) for name in ('first', 'second')))

    assert asyncio.run(run()) == [(tmp_path / name, tmp_path / name / 'deps')
                                 for name in ('first', 'second')]
    assert get_task_data_root() is None and dict(os.environ) == before


def test_scoped_roots_reach_a_fresh_worker_and_its_import_time_constants(tmp_path):
    before = dict(os.environ)
    task_root = tmp_path / '任务 数据'
    provided = dict(os.environ, AUTOREPRO_DATA_ROOT='stale-data', AUTOREPRO_DEPS_ROOT='stale-deps',
                    AUTOREPRO_EXPERIENCE_DIR='stale-experience')
    program = ("import sys,json;sys.path.insert(0,sys.argv[1]);"
               "from src.project_paths import get_data_root,get_deps_root;"
               "from src.agents.code_executor import DEPS_CACHE_ROOT;"
               "from src.experience.experience_store import DEFAULT_EXPERIENCE_PATH;"
               "print(json.dumps([str(get_data_root()),str(get_deps_root()),str(DEPS_CACHE_ROOT),"
               "str(DEFAULT_EXPERIENCE_PATH)]))")
    with task_data_scope(task_root):
        completed = runtime.run_owned_process([sys.executable, '-I', '-c', program, str(PROJECT_ROOT)],
                                              env=provided, timeout_s=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [str(task_root), str(task_root / 'deps'), str(task_root / 'deps'),
                                          str(task_root / 'experience' / 'experience.jsonl')]
    assert provided['AUTOREPRO_DATA_ROOT'] == 'stale-data'
    assert provided['AUTOREPRO_DEPS_ROOT'] == 'stale-deps'
    assert provided['AUTOREPRO_EXPERIENCE_DIR'] == 'stale-experience'
    assert dict(os.environ) == before


@pytest.mark.parametrize('hops', [1, 2])
def test_task_scope_routes_runtime_cache_and_download_cleanup_in_fresh_processes(tmp_path, hops):
    project, task_root = tmp_path / 'project', tmp_path / 'task'
    shared = project / 'data' / 'runtime' / 'test-machine'
    isolated = task_root / 'runtime' / 'test-machine'
    for root in (shared, isolated):
        (root / 'wheels').mkdir(parents=True)
        (root / 'wheels' / 'cached.whl').write_bytes(b'owned')
    before = dict(os.environ)
    provided = dict(os.environ, AUTOREPRO_TASK_DATA_ROOT=str(tmp_path / 'stale-task'),
                    AUTOREPRO_DATA_ROOT='stale-data', AUTOREPRO_DEPS_ROOT='stale-deps')
    probe = ("import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
             "from src import runtime_preparation as runtime;"
             "from src.project_paths import get_task_data_root;"
             "runtime._machine_key=lambda root:'test-machine';"
             "project=Path(sys.argv[2]);cache=runtime._runtime_cache_root(project);"
             "receipt=runtime.release_runtime_downloads(project);"
             "print(json.dumps({'task_root':str(get_task_data_root()),'cache':str(cache),"
             "'status':receipt['status']}))")
    program = probe
    if hops == 2:
        program = ("import sys;sys.path.insert(0,sys.argv[1]);"
                   "from src.runtime_preparation import run_owned_process;"
                   f"result=run_owned_process([sys.executable,'-I','-c',{probe!r},*sys.argv[1:]],timeout_s=15);"
                   "sys.stdout.write(result.stdout);sys.stderr.write(result.stderr);sys.exit(result.returncode)")
    with task_data_scope(task_root):
        completed = runtime.run_owned_process([sys.executable, '-I', '-c', program,
                                              str(PROJECT_ROOT), str(project)], env=provided, timeout_s=25)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {'task_root': str(task_root), 'cache': str(isolated), 'status': 'released'}
    assert not (isolated / 'wheels').exists()
    assert (shared / 'wheels' / 'cached.whl').read_bytes() == b'owned'
    assert provided['AUTOREPRO_TASK_DATA_ROOT'] == str(tmp_path / 'stale-task')
    assert dict(os.environ) == before


def test_clean_environment_preserves_caller_roots_without_a_scope():
    provided = {'AUTOREPRO_DATA_ROOT': 'caller-data', 'AUTOREPRO_DEPS_ROOT': 'caller-deps'}
    cleaned = runtime._clean_environment(provided)
    assert cleaned['AUTOREPRO_DATA_ROOT'] == 'caller-data'
    assert cleaned['AUTOREPRO_DEPS_ROOT'] == 'caller-deps'
    assert provided == {'AUTOREPRO_DATA_ROOT': 'caller-data', 'AUTOREPRO_DEPS_ROOT': 'caller-deps'}


@pytest.mark.parametrize('delegated', [False, True])
def test_supervisor_cleanup_keeps_shared_dependencies_and_training_python(tmp_path, monkeypatch, delegated):
    from src.agents import code_executor

    shared, task_root = tmp_path / 'shared', tmp_path / 'task'
    monkeypatch.setenv('AUTOREPRO_DATA_ROOT', str(shared))
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(shared / 'deps'))
    monkeypatch.setattr(code_executor, 'DEPS_CACHE_ROOT', shared / 'deps')
    for root in (shared, task_root):
        (root / 'deps' / 'package').mkdir(parents=True)
        (root / 'deps' / 'package' / 'installed.py').write_text('owned', encoding='utf-8')
        (root / 'training_python' / 'envs').mkdir(parents=True)
        (root / 'training_python' / 'envs' / 'owned.bin').write_bytes(b'owned')
    before = dict(os.environ)
    if delegated:
        original_import = builtins.__import__

        def without_filelock(name, *args, **kwargs):
            if name == 'src.dependency_cache':
                raise ModuleNotFoundError('No module named filelock', name='filelock')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', without_filelock)
    prepared = runtime.RuntimePreparation(sys.executable, True, False, 'test', 'test')
    with task_data_scope(task_root):
        receipt = runtime.release_worker_environment(prepared)
    assert receipt['status'] == 'released', receipt
    assert not (task_root / 'deps' / 'package').exists()
    assert not (task_root / 'training_python').exists()
    assert (shared / 'deps' / 'package' / 'installed.py').read_text('utf-8') == 'owned'
    assert (shared / 'training_python' / 'envs' / 'owned.bin').read_bytes() == b'owned'
    assert dict(os.environ) == before


def test_host_cleanup_explicitly_receives_the_scoped_dependency_root(tmp_path, monkeypatch):
    from src import environment_lifecycle

    release = Mock(return_value={'status': 'released'})
    monkeypatch.setattr(environment_lifecycle, 'release_abandoned_environment', release)
    with task_data_scope(tmp_path / 'task'):
        runtime.release_worker_environment(None)
    release.assert_called_once_with(tmp_path / 'task' / 'deps')
