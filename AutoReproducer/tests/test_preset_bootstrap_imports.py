"""A Python host without site-packages can reach preset runtime recovery."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("background", [False, True])
def test_missing_application_dependencies_do_not_block_bootstrap_handoff(tmp_path, background):
    project = Path(__file__).resolve().parents[1]
    progress = tmp_path / "bootstrap.jsonl"
    program = """
import importlib.util,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
dependencies=Path(sys.argv[2]).parent/'dependencies'
os.environ['AUTOREPRO_DEPS_ROOT']=str(dependencies)
os.environ['AUTOREPRO_DATA_ROOT']=str(Path(sys.argv[2]).parent/'data')
orphan=dependencies/'orphaned-environment'
orphan.mkdir(parents=True)
(orphan/'installed.py').write_text('VALUE = 42',encoding='utf-8')
assert importlib.util.find_spec('filelock') is None
assert importlib.util.find_spec('streamlit') is None
from frontend.backend_pipeline import ProgressStore,run_pipeline_core,run_pipeline_background
from frontend.task_control import task_status
from src import runtime_preparation as runtime
assert 'src.orchestrator' not in sys.modules
assert runtime.runtime_requirement()
observed=[]
owned_process=runtime.run_owned_process
def prepare(*args,**kwargs):
    observed.append('prepare')
    return runtime.RuntimePreparation(sys.executable,True,False,'missing app dependencies','cpython-safe')
def worker(argv,**kwargs):
    if 'input' not in kwargs:
        assert argv[0]==sys.executable and '-I' in argv
        observed.append('safe cleanup')
        return owned_process(argv,**kwargs)
    request=json.loads(kwargs['input'])
    assert request['_managed_runtime'] and request['_append_progress']
    assert 'src.orchestrator' not in sys.modules
    observed.append('safe worker')
    store=ProgressStore(request['progress_path'],reset=False)
    store.emit({'type':'done','result':{'state':'COMPLETED','error':None,'data':{}}})
    return subprocess.CompletedProcess(argv,0,'','')
runtime.prepare_runtime=prepare
runtime.run_owned_process=worker
if sys.argv[3]=='background':
    results=[]
    thread=run_pipeline_background(sys.argv[2],experiment_profile='dlinear_etth1_reference',
                                   prepare_only=True,offline=True,on_done=results.append)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(results)==1 and results[0] is not None
    result=results[0]
    status=task_status(sys.argv[2])
    assert status['managed'] and not status['active']
    assert status['record']['state']=='finished'
else:
    result=run_pipeline_core(sys.argv[2],experiment_profile='dlinear_etth1_reference',prepare_only=True,offline=True)
assert result['state']=='COMPLETED'
assert result['data']['environment_cleanup']['status']=='released'
assert not orphan.exists()
assert observed==['prepare','safe worker','safe cleanup']
print(json.dumps({'state':result['state'],'handoff':observed}))
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(project), str(progress),
                             "background" if background else "core"],
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["handoff"] == ["prepare", "safe worker", "safe cleanup"]
