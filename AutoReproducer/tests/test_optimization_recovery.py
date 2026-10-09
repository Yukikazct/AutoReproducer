import json
from filelock import FileLock

from src.method_adapters import write_json
from src.optimization_recovery import recover_interrupted_optimizations


def test_abandoned_study_recovers_running_records_and_preserves_completed_metrics(tmp_path):
    run=tmp_path/'repository_stopped'
    trial=run/'trials'/'candidate_2022'
    trial.mkdir(parents=True)
    write_json(trial/'trial.json',{'status':'completed','metrics':{'psnr':31.}})
    write_json(trial/'holdout.json',{'status':'running'})
    write_json(run/'optimization.json',{'status':'running','optimized':False,
        'trials':[{'status':'completed','metrics':{'psnr':31.}}],
        'confirmation':[{'seed':2021,'status':'completed','pass':True}, {'seed':2022,'status':'running'}]})
    assert recover_interrupted_optimizations(tmp_path)==[str(run/'optimization.json')]
    result=json.loads((run/'optimization.json').read_text(encoding='utf-8'))
    assert result['status']=='interrupted' and result['recovered'] and not result['optimized']
    assert result['confirmation'][0]['pass'] is True
    assert result['confirmation'][1]['status']=='interrupted'
    assert json.loads((trial/'trial.json').read_text(encoding='utf-8'))['metrics']=={'psnr':31.}
    assert json.loads((trial/'holdout.json').read_text(encoding='utf-8'))['status']=='interrupted'
    final=json.loads((run/'result.json').read_text(encoding='utf-8'))
    assert final['state']=='INTERRUPTED' and final['data']['run_status']=='interrupted'
    assert (run/'report.md').read_text(encoding='utf-8').startswith('> **实验已中断（interrupted）**')
    assert recover_interrupted_optimizations(tmp_path)==[]


def test_recovery_never_relabels_an_active_owner_or_completed_study(tmp_path):
    for name,status in [('active','running'),('done','validated_gain')]:
        run=tmp_path/f'repository_{name}'
        run.mkdir()
        write_json(run/'optimization.json',{'status':status,'optimized':status=='validated_gain'})
    with FileLock(str(tmp_path/'repository_active'/'.optimization.lock'),timeout=0):
        assert recover_interrupted_optimizations(tmp_path)==[]
    assert json.loads((tmp_path/'repository_done'/'optimization.json').read_text(encoding='utf-8'))['optimized']
