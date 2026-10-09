import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.method_adapters import write_json
from src.method_optimization import Study
from src.method_profiles import method_profile
from src.method_reproduction import MethodReproduction


@pytest.mark.parametrize('phase',['train','holdout'])
def test_real_study_persists_interrupted_execution_record(tmp_path,monkeypatch,phase):
    profile=method_profile('siren_camera_quick')
    interrupted=KeyboardInterrupt('cancelled')
    interrupted.execution={'success':False,'cancelled':True,'final':{'exit_code':130,'cancelled':True}}
    service=SimpleNamespace(root=tmp_path,runner=SimpleNamespace(run=Mock(side_effect=interrupted)))
    study=Study(service,profile,tmp_path,time.monotonic(),7200,lambda e:None)
    monkeypatch.setattr('src.repository_reproduction.export_repository',lambda *a,**k:{'files':{}})
    study.adapter.prepare_dataset=Mock(return_value={})
    study.adapter.materialize=Mock(return_value={'files':{}})
    if phase=='holdout':
        directory=tmp_path/'trials'/'baseline_2021'
        directory.mkdir(parents=True)
        study.selection_closed=True
        study.runs['baseline_2021']={'status':'completed','profile':profile,'workspace':str(directory/'repo'),
            'snapshot':{'files':{}},'manifest':{'files':{}},'spec_sha256':'fixed'}
    with pytest.raises(KeyboardInterrupt):
        getattr(study,phase)('baseline_2021')
    name='trial.json' if phase=='train' else 'holdout.json'
    saved=json.loads((tmp_path/'trials'/'baseline_2021'/name).read_text(encoding='utf-8'))
    assert saved['status']=='interrupted'
    assert saved['execution']['final']['exit_code']==130


@pytest.mark.parametrize('mode',['validate','suggest'])
def test_method_interruption_preserves_verified_baseline_and_final_run_state(tmp_path,monkeypatch,mode):
    baseline={'status':'method_experiment_completed','result_level':'method_experiment_completed','is_reproduced':None}
    adapter=SimpleNamespace(prepare_dataset=Mock(return_value={}),materialize=Mock(return_value={}),
        public_sources=Mock(return_value=[]),steps=Mock(return_value=[]),verify=Mock(return_value=baseline))
    monkeypatch.setattr('src.method_reproduction.get_adapter',lambda p:adapter)
    def export(root,profile,workspace,**kwargs):
        workspace.mkdir()
        return {'url':'https://example.org/repo','path':str(workspace)}
    monkeypatch.setattr('src.repository_reproduction.export_repository',export)
    monkeypatch.setattr('src.agents.report_generator.ReportGeneratorAgent.run',lambda *a,**k:{'report':'baseline report'})
    def cancel(service,profile,data,*args):
        write_json(Path(data['run_dir'])/'optimization.json',
                   {'status':'interrupted','optimized':False,'confirmation':[{'seed':2021,'pass':True}]})
        raise KeyboardInterrupt('cancelled')
    monkeypatch.setattr('src.method_optimization.validate_candidates',cancel)
    def cancel_advice(*args,**kwargs):
        raise KeyboardInterrupt('cancelled advice')
    monkeypatch.setattr('src.method_advice.suggest',cancel_advice)
    runner=SimpleNamespace(run=Mock(return_value={'success':True,'artifacts':[],'final':{}}))
    service=MethodReproduction(tmp_path,Mock(),runner=runner)
    with pytest.raises(KeyboardInterrupt):
        service.run({'experiment_profile':'siren_camera_quick','optimization_mode':mode,'offline':True})
    result_path=next((tmp_path/'runs').glob('*/result.json'))
    result=json.loads(result_path.read_text(encoding='utf-8'))
    assert result['state']=='INTERRUPTED'
    assert result['data']['validation']==baseline
    if mode=='validate':
        assert result['data']['optimization']['confirmation'][0]['pass']
    assert result['data']['optimization']['status']=='interrupted'
    report=(result_path.parent/'report.md').read_text(encoding='utf-8')
    assert report.startswith('> **实验已中断（interrupted）**')
    assert 'baseline report' in report
