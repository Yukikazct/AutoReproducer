"""Protect selection/confirmation boundaries and metric direction with tiny fixtures."""
import json
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.method_profiles import method_profile
from src.method_adapters import digest, write_json, SirenAdapter
from src.method_optimization import confirmed_gain, validate_candidates, Study


@pytest.mark.parametrize("method,baseline,candidate,expected", [
    ("siren",30.,30.2,True),("siren",30.,30.05,False),("siren",30.,29.,False),
    ("neural_ode",1.,.98,True),("neural_ode",1.,.995,False),("neural_ode",1.,1.1,False),
    ("neural_ode",0.,0.,False),("siren",30.,float('nan'),False),
])
def test_confirmation_uses_correct_units_direction_and_finite_values(method,baseline,candidate,expected):
    assert confirmed_gain(method,baseline,candidate) is expected


@pytest.fixture
def study_harness(tmp_path,monkeypatch):
    events=[]
    state={"candidate_value":31.,"holdout_candidate":30.2,"remaining":7200}
    class FakeStudy:
        def __init__(self,*args):
            self.profile=deepcopy(args[1]); self.profile["parameters"]["protocol"]="pixel_holdout"
            self.contract={"pixel_split_seed":1729,"pixel_fractions":[.8,.1,.1]}
            self.metric="psnr"; self.direction=1; self.contract_hash="frozen";self.selection_closed=False
            self.profile=deepcopy(args[1]);self.contract={"base_profile":self.profile}
        def remaining(self): return state["remaining"]
        def train(self,label,seed=2021,candidate=None,reserve_s=0):
            saved=json.loads((tmp_path/'optimization.json').read_text(encoding='utf-8'))
            assert saved['status']=='running'
            events.append(("train",label,reserve_s))
            if state.get('interrupt_at')==label:
                raise state.get('exception',KeyboardInterrupt)('stopped')
            metrics={"psnr":state["candidate_value"] if candidate else 30.}
            return {"label":label,"candidate":candidate,"status":"completed","metrics":metrics,
                    "validation":{"metrics_comparison":{"actual":metrics}},"elapsed_s":.1}
        def holdout(self,label):
            assert self.selection_closed
            assert any(e[1]=="candidate_2022" for e in events)
            events.append(("holdout",label))
            if state.get('interrupt_at')=='holdout_'+label:
                raise state.get('exception',KeyboardInterrupt)('stopped')
            return state["holdout_candidate"] if label.startswith("candidate") else 30.
    monkeypatch.setattr("src.method_optimization.Study",FakeStudy)
    def advice(*args,**kwargs):
        if "remaining_after_advice" in state:
            state["remaining"] = state["remaining_after_advice"]
        return {"status":"suggested","suggestions":[{"parameter":"learning_rate","value":.0002,"status":"untested"}],"calls":1}
    monkeypatch.setattr("src.method_optimization.suggest",advice)
    def run():
        return validate_candidates(SimpleNamespace(llm=object()),method_profile("siren_camera_quick"),
                                   {"run_dir":str(tmp_path),"method_sources":[]},{},time.monotonic(),lambda e:None)
    state['run_dir']=tmp_path
    return run,state,events


def test_holdout_waits_for_selection_and_both_confirmation_trainings(study_harness):
    run,state,events=study_harness
    result=run()
    assert result["optimized"] and result["status"]=="validated_gain"
    assert [r["seed"] for r in result["confirmation"]]==[2021,2022]
    assert len([e for e in events if e[0]=="holdout"])==4
    assert ("train","candidate_1_2021",2400) in events


def test_winning_validation_score_is_not_enough(study_harness):
    run,state,events=study_harness
    state["holdout_candidate"]=30.01
    result=run()
    assert not result["optimized"]
    assert result["status"]=="tested_no_gain"


def test_no_validation_improvement_never_opens_holdout(study_harness):
    run,state,events=study_harness
    state["candidate_value"]=29.
    result=run()
    assert not result["optimized"] and not result["confirmation"]
    assert not any(e[0]=="holdout" for e in events)


def test_reserve_is_enforced_before_starting(study_harness):
    run,state,events=study_harness
    state["remaining"]=2349.4
    result=run()
    assert result["status"]=="budget_insufficient" and not events
    assert "39.2 分钟" in result["reason"] and "预留 40 分钟" in result["reason"]
    assert "基线结果已保留" in result["reason"] and "候选尚未完成全部验证" in result["reason"]


def test_advice_consuming_candidate_budget_preserves_baseline_and_explains_stop(study_harness):
    run,state,events=study_harness
    state["remaining_after_advice"] = 2350
    result=run()
    assert result["status"] == "budget_insufficient" and not result["optimized"]
    assert result["baseline"] == {"psnr": 30.}
    assert events == [("train", "baseline_2021", 0)]
    assert not result["trials"] and not result["confirmation"]
    assert "39.2 分钟" in result["reason"] and "预留 40 分钟" in result["reason"]
    assert "基线结果已保留" in result["reason"] and "候选尚未完成全部验证" in result["reason"]
    saved=json.loads((state['run_dir']/'optimization.json').read_text(encoding='utf-8'))
    assert saved["status"] == "budget_insufficient" and saved["reason"] == result["reason"]


@pytest.mark.parametrize('label', ['baseline_2021','candidate_1_2021','baseline_2022','candidate_2022',
                                  'holdout_baseline_2022','holdout_candidate_1_2021','holdout_candidate_2022'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_cancellation_persists_running_phase_and_completed_evidence(study_harness,label,exception):
    run,state,events=study_harness
    state.update(interrupt_at=label,exception=exception)
    with pytest.raises((KeyboardInterrupt,SystemExit)):
        run()
    saved=json.loads((state['run_dir']/'optimization.json').read_text(encoding='utf-8'))
    assert saved['status']=='interrupted' and not saved['optimized']
    assert not any(row.get('status')=='running' for group in ('trials','confirmation_training','confirmation')
                   for row in saved[group])
    if label=='candidate_2022':
        assert saved['trials'][0]['status']=='completed'
        assert saved['confirmation_training'][0]['status']=='completed'
    if label=='holdout_baseline_2022':
        assert [row['seed'] for row in saved['confirmation']]==[2021,2022]
        assert saved['confirmation'][0]['status']=='completed'
        assert saved['confirmation'][1]['status']=='interrupted'
        assert 'candidate' not in saved['confirmation'][1]
    if label=='holdout_candidate_2022':
        assert saved['confirmation'][0]['status']=='completed'
        assert saved['confirmation'][0]['pass']
        assert saved['confirmation'][1]['baseline']==30.
        assert 'candidate' not in saved['confirmation'][1]


@pytest.mark.parametrize('exception,status',[(TimeoutError,'budget_exhausted'),(ValueError,'failed')])
def test_only_real_timeout_reports_budget_exhausted(study_harness,exception,status):
    run,state,events=study_harness
    state.update(interrupt_at='candidate_1_2021',exception=exception)
    result=run()
    assert result['status']==status and not result['optimized']
    assert result['trials'][0]['status']==('timed_out' if exception is TimeoutError else 'failed')


def test_elapsed_total_budget_is_a_real_timeout(study_harness):
    run,state,events=study_harness
    state['remaining']=-1
    assert run()['status']=='budget_exhausted'
    assert not events


def test_real_study_refuses_early_holdout(tmp_path):
    study=Study(SimpleNamespace(),method_profile("siren_camera_quick"),tmp_path,time.monotonic(),7200,lambda e:None)
    with pytest.raises(ValueError,match="禁止留出"):
        study.holdout("anything")


def test_runtime_change_invalidates_frozen_study(tmp_path):
    study=Study(SimpleNamespace(),method_profile("siren_camera_quick"),tmp_path,time.monotonic(),7200,lambda e:None)
    study.contract["metric"]="fake_accuracy"
    with pytest.raises(ValueError,match="协议"):
        study.check_contract()


def test_edited_metric_file_is_not_independent_evidence(tmp_path):
    p=method_profile("siren_camera_quick")
    out=tmp_path/'artifacts';out.mkdir()
    (out/'prediction.npy').write_bytes(b'fixed prediction')
    (out/'checkpoint.pt').write_bytes(b'fixed checkpoint')
    deps=tmp_path/'deps';deps.mkdir()
    write_json(tmp_path/'import_provenance.json',{'torch':'2.5.1+cu121','cuda':'12.1','torchvision':'0.20.1+cu121',
        'modules':{'torch':str(deps/'torch.py')}})
    write_json(out/'training.json',{'parameters':p['parameters'],'steps_completed':500,'losses':[.1]*500,
        'spec_sha256':'spec','prediction_sha256':digest(out/'prediction.npy'),
        'checkpoint_sha256':digest(out/'checkpoint.pt'),'seed':2021,'training_elapsed_s':1})
    metrics={'pass':True,'split':'fit','metrics':{'mse':.001,'psnr':30.},'spec_sha256':'spec',
        'prediction_sha256':digest(out/'prediction.npy')}
    execution={'success':True,'final':{'exit_code':0,'stdout':json.dumps(metrics)},
               'environment':{'dependencies_path':str(deps)}}
    metrics['metrics']['psnr']=99.
    write_json(out/'metrics_fit.json',metrics)
    with pytest.raises(ValueError,match="输出不一致"):
        SirenAdapter().verify(p,execution,tmp_path,{'files':{}},{'files':{}},'spec')


def test_neural_ode_preserves_author_protocol_and_has_no_table_claim():
    p=method_profile('neural_ode_spiral')
    assert p['parameters']['steps']==2000 and p['parameters']['batch_time']==10
    assert p['parameters']['solver']=='dopri5' and p['parameters']['learning_rate']==.001
    assert p['paper']['metrics']=={} and p['dataset']['kind']=='analytic'
