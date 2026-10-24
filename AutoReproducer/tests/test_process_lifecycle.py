"""Real Windows owner/session termination must not leave training descendants."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest
from filelock import FileLock

from src.optimization_recovery import recover_interrupted_optimizations
from src.process_lifecycle import termination_signals


@pytest.mark.skipif(os.name!='nt',reason='Windows venv configuration')
def test_ordinary_venv_keeps_supported_process_tree_path(tmp_path,monkeypatch):
    import src.process_lifecycle as lifecycle
    from types import SimpleNamespace
    executable=tmp_path/'venv'/'Scripts'/'python.exe'
    executable.parent.mkdir(parents=True)
    (executable.parent.parent/'pyvenv.cfg').write_text(f'home = {tmp_path / "ordinary-python"}\n',encoding='utf-8')
    monkeypatch.setattr(lifecycle,'sys',SimpleNamespace(executable=str(executable)))
    monkeypatch.setattr(lifecycle,'_is_app_execution_alias',lambda path:False)
    lifecycle.validate_windows_venv()


def test_app_execution_alias_uses_windows_reparse_tag(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.process_lifecycle import _is_app_execution_alias
    monkeypatch.setattr(Path,'lstat',lambda p:SimpleNamespace(st_reparse_tag=0x8000001B))
    assert _is_app_execution_alias(tmp_path/'python.exe')
    monkeypatch.setattr(Path,'lstat',lambda p:SimpleNamespace(st_reparse_tag=0xA000000C))
    assert not _is_app_execution_alias(tmp_path/'python.exe')


def test_catchable_termination_restores_original_handler():
    before=signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with termination_signals():
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    assert signal.getsignal(signal.SIGTERM)==before


def test_failed_native_job_close_retains_handle_until_successful_retry(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import src.process_lifecycle as lifecycle

    failure = OSError(6, "job handle close failed")
    native_error = Mock(return_value=failure)
    monkeypatch.setattr(lifecycle.ctypes, "get_last_error", lambda: 6, raising=False)
    monkeypatch.setattr(lifecycle.ctypes, "WinError", native_error, raising=False)
    job = lifecycle.ProcessJob.__new__(lifecycle.ProcessJob)
    job.handle = 123
    job.api = SimpleNamespace(CloseHandle=Mock(side_effect=[0, 1]))
    job._terminate_and_wait = Mock()

    with pytest.raises(OSError) as raised:
        job.close()
    assert raised.value is failure
    native_error.assert_called_once_with(6)
    assert job.handle == 123, "a failed close must retain the owned job"

    job.close()
    assert job.handle is None
    job.close()
    assert job.api.CloseHandle.call_args_list == [((123,), {}), ((123,), {})]


@pytest.mark.parametrize('failure', [None, 'terminate', 'query', 'timeout'])
def test_job_close_confirms_all_descendants_before_releasing_handle(monkeypatch, failure):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import src.process_lifecycle as lifecycle

    active_counts = iter([2, 1, 0])
    def query(handle, kind, pointer, size, returned):
        pointer._obj.ActiveProcesses = next(active_counts)
        return failure != 'query'
    job = lifecycle.ProcessJob.__new__(lifecycle.ProcessJob)
    job.handle = 123
    job.api = SimpleNamespace(TerminateJobObject=Mock(return_value=failure != 'terminate'),
                              QueryInformationJobObject=Mock(side_effect=query),
                              CloseHandle=Mock(return_value=1))
    job._process_ids = Mock(return_value=[])
    monkeypatch.setattr(lifecycle.ctypes, 'get_last_error', lambda: 6, raising=False)
    monkeypatch.setattr(lifecycle.ctypes, 'WinError', lambda code: OSError(code, 'native failure'), raising=False)
    clock = iter([0, 6]) if failure == 'timeout' else iter([0, 1, 2])
    monkeypatch.setattr(lifecycle.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(lifecycle.time, 'sleep', Mock())
    if failure:
        with pytest.raises(TimeoutError if failure == 'timeout' else OSError):
            job.close()
        assert job.handle == 123
        job.api.CloseHandle.assert_not_called()
    else:
        job.close()
        assert job.handle is None
        assert job.api.QueryInformationJobObject.call_count == 3
        assert lifecycle.time.sleep.call_count == 2
        job.api.CloseHandle.assert_called_once_with(123)


def wait_file(path,timeout=12):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except ValueError:
                pass
        time.sleep(.05)
    pytest.fail(f'timed out waiting for {path.name}')


def wait_stopped(pid,timeout=8):
    from src.process_lifecycle import _windows_api
    api=_windows_api()
    handle=api.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            assert api.WaitForSingleObject(handle,int(timeout*1000))==0, f'process {pid} survived cancellation'
        finally:
            api.CloseHandle(handle)


@pytest.mark.skipif(os.name != 'nt', reason='Windows SIGBREAK and native process jobs')
def test_windows_break_preserves_legacy_output_and_stops_training_tree(tmp_path):
    """A real Ctrl+Break must unwind despite non-UTF-8 native diagnostics."""
    workspace = tmp_path / '中文 实验'
    workspace.mkdir()
    cache = tmp_path / 'dependency cache'
    ready = workspace / 'children.json'
    record = tmp_path / 'cancelled.json'
    child = (
        "import json,os,subprocess,sys,time;from pathlib import Path;"
        "descendant=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "sys.stdout.buffer.write(b'CANCEL_READY \\xb4\\xed\\xce\\xf3\\n');sys.stdout.buffer.flush();"
        "sys.stderr.buffer.write(b'native diagnostic \\xb4\\xed\\xce\\xf3\\n');sys.stderr.buffer.flush();"
        "Path('children.json').write_text(json.dumps([os.getpid(),descendant.pid]),encoding='utf-8');"
        "time.sleep(60)"
    )
    owner = tmp_path / 'owner.py'
    owner.write_text(
        "import json,os,sys\nfrom pathlib import Path\nfrom unittest.mock import Mock\n"
        "from filelock import FileLock\n"
        "from src.agents.code_executor import CodeExecutorAgent\n"
        "from src.repository_runner import RepositoryRunner\n"
        "from src.process_lifecycle import termination_signals\n"
        "executor=CodeExecutorAgent(None,logger=Mock())\n"
        "executor._ensure_local_deps=Mock(return_value=None)\n"
        "executor._exec_env=Mock(return_value={**os.environ,'PYTHONIOENCODING':'utf-8'})\n"
        "runner=RepositoryRunner(executor=executor)\n"
        "try:\n"
        f"    with FileLock({str(cache / '.owner.lock')!r}),termination_signals():\n"
        f"        runner.run({str(workspace)!r},[{{'id':'train','argv':['python','-c',{child!r}],'timeout_s':None}}],{{}})\n"
        "except KeyboardInterrupt as exc:\n"
        f"    Path({str(record)!r}).write_text(json.dumps(exc.execution),encoding='utf-8')\n"
        "    sys.exit(130)\n", encoding='utf-8')
    cache.mkdir()
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
                   'AUTOREPRO_DEPS_ROOT': str(cache), 'PYTHONUTF8': '0', 'PYTHONIOENCODING': 'cp1252'}
    # Hosted Windows workers need not have a console. A separate console driver
    # shares its console with the owner, so GenerateConsoleCtrlEvent exercises a
    # real Ctrl+Break without changing pytest's own console or signal handlers.
    cancel = tmp_path / 'send-break'
    driver = tmp_path / 'console_driver.py'
    driver.write_text(
        "import signal,subprocess,sys,time\nfrom pathlib import Path\n"
        f"owner=subprocess.Popen([sys.executable,{str(owner)!r}],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)\n"
        "try:\n"
        "    deadline=time.monotonic()+30\n"
        f"    while not Path({str(cancel)!r}).exists():\n"
        "        if owner.poll() is not None:\n"
        "            raise RuntimeError('owner exited before cancellation: '+str(owner.returncode))\n"
        "        if time.monotonic()>deadline:\n"
        "            raise TimeoutError('cancellation was not requested')\n"
        "        time.sleep(.05)\n"
        "    owner.send_signal(signal.CTRL_BREAK_EVENT)\n"
        "    sys.exit(owner.wait(timeout=15))\n"
        "finally:\n"
        "    if owner.poll() is None:\n"
        "        owner.kill()\n"
        "    owner.wait(timeout=5)\n", encoding='utf-8')
    with (tmp_path / 'owner.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, str(driver)], env=environment,
                                   stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
        descendants = []
        try:
            descendants = wait_file(ready)
            cancel.touch()
            assert process.wait(timeout=15) == 130
            saved = wait_file(record)
            assert saved['status'] == 'interrupted' and saved['cancelled']
            assert saved['final']['exit_code'] == 130 and not saved['final']['timed_out']
            assert 'CANCEL_READY' in saved['final']['stdout']
            assert b'\xb4\xed\xce\xf3' in Path(saved['final']['stdout_path']).read_bytes()
            assert b'\xb4\xed\xce\xf3' in Path(saved['final']['stderr_path']).read_bytes()
            for pid in descendants:
                wait_stopped(pid)
            with FileLock(str(cache / '.owner.lock'), timeout=0):
                pass
            with FileLock(str(cache / '.autorepro-cache.lock'), timeout=0):
                pass
        finally:
            for pid in [*descendants, process.pid]:
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    assert 'UnicodeDecodeError' not in (tmp_path / 'owner.log').read_text('utf-8', errors='replace')


@pytest.mark.skipif(os.name!='nt',reason='Windows owner and launching-session semantics')
@pytest.mark.parametrize('mode',['close_session','close_session_py_launcher','kill_owner'])
def test_windows_session_exit_and_force_kill_stop_tree_and_release_locks(tmp_path,mode):
    run=tmp_path/'runs'/'repository_terminated'
    workspace=run/'repo'
    workspace.mkdir(parents=True)
    cache=tmp_path/'cache'
    child_code=("import json,os,subprocess,sys,time;from pathlib import Path;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                "Path('children.json').write_text(json.dumps([os.getpid(),child.pid]),encoding='utf-8');"
                "time.sleep(60)")
    owner=tmp_path/'owner.py'
    owner.write_text(
        "import json,os\nfrom pathlib import Path\nfrom unittest.mock import Mock\n"
        "from filelock import FileLock\n"
        "from src.agents.code_executor import CodeExecutorAgent\n"
        "from src.repository_runner import RepositoryRunner\n"
        "from src.method_adapters import write_json\n"
        "from src.process_lifecycle import watch_parent_session\n"
        f"run=Path({str(run)!r})\n"
        "executor=CodeExecutorAgent(None,logger=Mock())\n"
        "executor._ensure_local_deps=Mock(return_value=None)\n"
        "executor._exec_env=Mock(return_value={**os.environ,'PYTHONIOENCODING':'utf-8'})\n"
        "runner=RepositoryRunner(executor=executor)\n"
        "with FileLock(str(run/'.optimization.lock')):\n"
        "    write_json(run/'optimization.json',{'status':'running','optimized':False})\n"
        "    write_json(run/'owner.json',{'pid':os.getpid()})\n"
        "    try:\n"
        "        with watch_parent_session():\n"
        f"            runner.run(run/'repo',[{{'id':'train','argv':['python','-c',{child_code!r}],'timeout_s':None}}],{{}})\n"
        "    except KeyboardInterrupt:\n"
        "        write_json(run/'optimization.json',{'status':'interrupted','optimized':False})\n"
        "        write_json(run/'cancelled.json',{'received':True})\n",
        encoding='utf-8')
    launcher=tmp_path/'launcher.py'
    if mode=='close_session_py_launcher':
        py=shutil.which('py')
        shell=shutil.which('pwsh') or shutil.which('powershell')
        if not py or not shell:
            pytest.skip('Python launcher or PowerShell is unavailable')
        # Exercise the launcher without selecting a different global interpreter
        # that may not contain the current test environment's dependencies.
        launch_command=([py,str(owner)] if sys.prefix != sys.base_prefix else
                        [py,f'-{sys.version_info.major}.{sys.version_info.minor}',str(owner)])
        quoted=' '.join("'"+part.replace("'","''")+"'" for part in launch_command)
        session_command=[shell,'-NoProfile','-Command','& '+quoted]
    else:
        launch_command=[sys.executable,str(owner)]
        session_command=[sys.executable,str(launcher)]
    launcher.write_text(f"import subprocess\nsubprocess.run({launch_command!r},check=True)\n",encoding='utf-8')
    project=Path(__file__).resolve().parents[1]
    environment={**os.environ,'PYTHONPATH':str(project),'AUTOREPRO_DEPS_ROOT':str(cache)}
    if sys.prefix != sys.base_prefix:
        environment['VIRTUAL_ENV']=sys.prefix
    log=(tmp_path/'owner.log').open('wb')
    process=subprocess.Popen(session_command,stdout=log,stderr=log,env=environment,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    owner_pid=None
    try:
        owner_pid=wait_file(run/'owner.json')['pid']
        descendants=wait_file(workspace/'children.json')
        assert recover_interrupted_optimizations(tmp_path/'runs')==[]
        if mode.startswith('close_session'):
            process.kill()  # Only the launching shell/session exits.
            assert wait_file(run/'cancelled.json')['received']
        else:
            # Deliberately omit /T: the job, not taskkill, must remove children.
            subprocess.run(['taskkill','/PID',str(owner_pid),'/F'],capture_output=True,check=True)
            recovered=recover_interrupted_optimizations(tmp_path/'runs')
            assert recovered==[str(run/'optimization.json')]
        wait_stopped(owner_pid)
        for pid in descendants:
            wait_stopped(pid)
        with FileLock(str(cache/'.autorepro-cache.lock'),timeout=0):
            pass
        saved=json.loads((run/'optimization.json').read_text(encoding='utf-8'))
        assert saved['status']=='interrupted' and not saved['optimized']
    finally:
        if owner_pid:
            subprocess.run(['taskkill','/PID',str(owner_pid),'/T','/F'],capture_output=True)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        log.close()


@pytest.mark.skipif(os.name!='posix',reason='POSIX catchable SIGTERM semantics')
def test_posix_sigterm_persists_interruption_and_stops_descendants(tmp_path):
    project=Path(__file__).resolve().parents[1]
    descendant="import time;from pathlib import Path;time.sleep(1);Path('escaped').touch()"
    child=("import subprocess,sys,time;from pathlib import Path;"
           f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
           "Path('ready.json').write_text('{}');time.sleep(30)")
    owner=("from pathlib import Path\nfrom unittest.mock import Mock\nimport os\n"
           "from src.agents.code_executor import CodeExecutorAgent\n"
           "from src.repository_runner import RepositoryRunner\n"
           "executor=CodeExecutorAgent(None,logger=Mock())\n"
           "executor._ensure_local_deps=Mock(return_value=None)\n"
           "executor._exec_env=Mock(return_value=dict(os.environ))\n"
           "try:\n"
           f"    RepositoryRunner(executor=executor).run(Path.cwd(),[{{'id':'train','argv':['python','-c',{child!r}],'timeout_s':30}}],{{}})\n"
           "except KeyboardInterrupt:\n"
           "    Path('cancelled.json').write_text('{}')\n"
           "    raise SystemExit(130)\n")
    environment={**os.environ,'PYTHONPATH':str(project),'AUTOREPRO_DEPS_ROOT':str(tmp_path/'cache')}
    process=subprocess.Popen([sys.executable,'-c',owner],cwd=tmp_path,env=environment,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        wait_file(tmp_path/'ready.json')
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8)==130
        wait_file(tmp_path/'cancelled.json')
        time.sleep(1.1)
        assert not (tmp_path/'escaped').exists()
        with FileLock(str(tmp_path/'cache'/'.autorepro-cache.lock'),timeout=0):
            pass
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)
