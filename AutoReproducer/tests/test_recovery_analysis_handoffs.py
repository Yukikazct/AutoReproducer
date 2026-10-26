"""Both enhanced modes give the production analyst the actual new findings."""
import hashlib
import json

import pytest

from src.author_failure_analysis import analyze_author_failure
from src.intelligent_recovery import RecoverySession


@pytest.mark.parametrize('mode', ['single', 'collaborative'])
def test_production_failure_analyst_receives_observed_findings_in_both_modes(tmp_path, mode):
    source = tmp_path / 'train.py'
    source.write_text('print(unknown_api())\n')
    packet = {'file_inventory': {'train.py': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                             'text_available': True}}}
    plan = {'repository': {'url': 'https://github.com/example/handoff'}, 'steps': [],
            'requirements': {'author': [], 'compatibility': []}}
    execution = {'success': False, 'final': {'stderr': 'NameError: unknown_api'}}
    observations = []
    class Analyst:
        mock_mode = False
        def chat(self, prompt, **kwargs):
            context = json.loads(prompt)
            observations.append(context)
            assert kwargs['task'] == 'author_failure_analysis'
            assert context['investigation_findings'][0]['evidence'][0]['id'] == 'runtime_observation'
            assert context['review_feedback']['experiment_understanding'] == {'unit': 'percent'}
            return json.dumps({'action': 'blocked', 'reason': 'The observed runtime metadata rules out the original hypothesis.'})
    llm = Analyst()
    session = RecoverySession(llm, mode=mode)
    (tmp_path / 'run').mkdir()
    session.attach({}, tmp_path / 'run')
    task = {'id': 'task1', 'status': 'repair_proposed', 'probes': [],
            'evidence': [{'id': 'runtime_observation', 'source': 'author_runtime', 'value': {'version': '1.2'}}],
            'experiment_understanding': {'unit': 'percent'}}
    session.state['tasks'].append(task)
    session._active_task = task
    with session.scope():
        analyze_author_failure(session.wrap_llm(), execution, plan, packet, tmp_path, {}, tmp_path / 'run', {})
    assert len(observations) == 1
    assert session.state['usage']['roles']['author_failure_analysis']['model_requests'] == 1
