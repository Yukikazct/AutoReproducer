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
        f"            runner.run(run/'repo',[{{'id':'train','argv':['python','-c',{child_code!r}],'timeout_s':60}}],{{}})\n"
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
        launch_command=[py,f'-{sys.version_info.major}.{sys.version_info.minor}',str(owner)]
        quoted=' '.join("'"+part.replace("'","''")+"'" for part in launch_command)
        session_command=[shell,'-NoProfile','-Command','& '+quoted]
    else:
        launch_command=[sys.executable,str(owner)]
        session_command=[sys.executable,str(launcher)]
    launcher.write_text(f"import subprocess\nsubprocess.run({launch_command!r},check=True)\n",encoding='utf-8')
    project=Path(__file__).resolve().parents[1]
    environment={**os.environ,'PYTHONPATH':str(project),'AUTOREPRO_DEPS_ROOT':str(cache)}
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
