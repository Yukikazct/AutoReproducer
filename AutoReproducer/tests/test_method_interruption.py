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


@pytest.mark.parametrize('mode', ['validate', 'suggest', 'off'])
def test_author_execution_interruption_preserves_logs_and_run_state(tmp_path, monkeypatch, mode):
    adapter = SimpleNamespace(prepare_dataset=Mock(return_value={}), materialize=Mock(return_value={}),
        public_sources=Mock(return_value=[]),
        steps=Mock(return_value=[{'id': 'train', 'kind': 'train', 'argv': ['python', '-c', 'print(1)']}]),
        verify_environment=Mock(), verify=Mock())
    monkeypatch.setattr('src.method_reproduction.get_adapter', lambda p: adapter)
    def export(root, profile, workspace, **kwargs):
        workspace.mkdir()
        return {'url': 'https://example.org/repo', 'path': str(workspace)}
    monkeypatch.setattr('src.repository_reproduction.export_repository', export)
    advice = Mock(side_effect=AssertionError('Reserved advice must not run'))
    optimize = Mock(side_effect=AssertionError('Reserved optimization must not run'))
    monkeypatch.setattr('src.method_optimization.validate_candidates', optimize)
    monkeypatch.setattr('src.method_advice.suggest', advice)
    interrupted = KeyboardInterrupt('cancelled author command')
    interrupted.execution = {'success': False, 'executed': True, 'cancelled': True,
        'final': {'success': False, 'exit_code': 130, 'cancelled': True, 'stdout': 'retained author output'}}
    prepared = {'success': True, 'executed': True, 'artifacts': [], 'final': {'success': True, 'exit_code': 0}}
    runner = SimpleNamespace(run=Mock(side_effect=[prepared, interrupted]))
    logger = Mock()
    logger.get_stats.return_value = {}
    service = MethodReproduction(tmp_path, logger, runner=runner)
    with pytest.raises(KeyboardInterrupt):
        service.run({'experiment_profile': 'siren_camera_quick', 'optimization_mode': mode, 'offline': True})
    result_path = next((tmp_path / 'runs').glob('*/result.json'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert result['state'] == 'INTERRUPTED'
    assert result['data']['validation']['status'] == 'interrupted'
    assert result['data']['validation']['is_reproduced'] is None
    assert result['data']['execution'] == interrupted.execution
    assert result['data']['optimization']['status'] == 'disabled'
    assert runner.run.call_count == 2
    assert runner.run.call_args.args[1][0]['id'] == 'train'
    status = json.loads((result_path.parent / 'run_status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'interrupted'
    advice.assert_not_called()
    optimize.assert_not_called()
    adapter.verify.assert_not_called()
