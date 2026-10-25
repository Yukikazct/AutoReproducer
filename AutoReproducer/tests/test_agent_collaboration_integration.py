"""Active services, actual prompts and UI share one evidence-bound dialogue."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

from src.agent_collaboration import CollaborationSession, current_collaboration
from src.agents.env_builder import EnvBuilderAgent
from src.agents.resource_finder import ResourceFinderAgent
from src.agents.code_executor import CodeExecutorAgent
from src.discovered_repository_plan import _canonical
from src.discovered_repository_reproduction import DiscoveredRepositoryReproduction
from src.llm.llm_client import LLMClient
from frontend.backend_pipeline import ProgressStore
from test_orchestrator_phase_events import pipeline
from test_app_pipeline_stages import app
from test_discovered_repository_service import setup_service


def test_generic_roles_receive_actual_previous_outputs_with_no_extra_model_calls(pipeline):
    inboxes = {}
    for key in ('reader', 'finder', 'builder', 'executor', 'reporter'):
        agent = pipeline.agents[key]
        output = deepcopy(agent.run.return_value)
        def execute(_data, agent=agent, output=output):
            inboxes[agent.name] = agent.collaboration_context()
            return deepcopy(output)
        agent.run.side_effect = execute
    first = pipeline.run({'paper_title': 'First paper', 'code': 'print(7)'})
    assert first['state'] == 'COMPLETED'
    assert inboxes['ResourceFinder'][0]['sender'] == 'PaperReader'
    assert inboxes['EnvBuilder'][0]['content']['code_repo_url'] == 'https://github.com/example/repo'
    assert inboxes['CodeExecutor'][0]['sender'] == 'EnvBuilder'
    assert inboxes['ReportGenerator'][0]['content']['success'] is True
    saved = deepcopy(first['data']['agent_collaboration'])
    second = pipeline.run({'paper_title': 'Second paper', 'code': 'print(8)'})
    assert second['data']['agent_collaboration']['run_id'] != saved['run_id']
    assert first['data']['agent_collaboration'] == saved
    pipeline.llm.chat.assert_not_called()
    assert current_collaboration() is None


def test_peer_evidence_is_in_the_actual_finder_builder_and_executor_model_prompts():
    llm = LLMClient(mock_mode=True)
    chat = llm.chat
    llm.chat = Mock(side_effect=chat)
    session = CollaborationSession()
    session.send('PaperReader', 'ResourceFinder', 'handoff', {'evidence': 'paper-evidence-marker'})
    session.send('ResourceFinder', 'EnvBuilder', 'handoff', {'evidence': 'repository-evidence-marker'})
    session.send('EnvBuilder', 'CodeExecutor', 'handoff', {'evidence': 'environment-evidence-marker'})
    paper = {'title': 'Invented paper', 'method': 'linear', 'dependencies': [], 'dataset': 'Grid'}
    with session.scope():
        ResourceFinderAgent(llm, logger=Mock(), offline=True).run({'paper_info': paper})
        EnvBuilderAgent(llm, logger=Mock()).run({'paper_info': paper})
        CodeExecutorAgent(llm, logger=Mock(), mock_mode=True)._generate_code(paper)
    prompts = {call.kwargs['task']: call.args[0] for call in llm.chat.call_args_list}
    assert 'paper-evidence-marker' in prompts['resource_finder']
    assert 'repository-evidence-marker' in prompts['env_builder']
    assert 'environment-evidence-marker' in prompts['code_executor']
    assert llm.get_call_count() == 3


def test_active_pdf_prepare_path_consumes_plan_handoffs_and_persists_final_dialogue(tmp_path, monkeypatch):
    _, service, request, llm = setup_service(tmp_path, monkeypatch)
    service.runner.run = Mock(side_effect=AssertionError('prepare-only must not train'))
    events = []
    result = service.run({**request, 'prepare_only': True}, on_event=events.append)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    messages = data['agent_collaboration']['messages']
    for role in ('EnvBuilder', 'CodeExecutor'):
        handoff = next(m for m in messages if m['sender'] == 'AuthorPlanner' and m['recipient'] == role)
        acknowledgement = next(m for m in messages if m.get('reply_to') == handoff['id'])
        assert handoff['delivered'] and acknowledgement['content']['status'] == 'accepted'
    assert any(m['sender'] == 'ReportGenerator' for m in messages)
    root = Path(data['run_dir'])
    assert json.loads((root / 'agent_collaboration.json').read_text('utf-8')) == data['agent_collaboration']
    assert json.loads((root / 'result.json').read_text('utf-8'))['data']['agent_collaboration'] == data['agent_collaboration']
    assert 'Agent 协作记录' in data['report']
    assert (root / 'paper_markdown.md').exists() and (root / 'paper_markdown.json').exists()
    assert any(event['type'] == 'agent_message' for event in events)
    assert {options['task'] for _, options in llm.calls} == {'author_execution_plan'}
    assert data['validation']['is_reproduced'] is None
    service.runner.run.assert_not_called()


def test_stale_planner_message_cannot_replace_frozen_commands():
    plan = {'steps': [{'argv': ['python', 'original.py'], 'cwd': '.'}], 'scope': 'author_code_execution'}
    before = deepcopy(plan)
    session = CollaborationSession()
    stale = session.send('AuthorPlanner', 'CodeExecutor', 'handoff', {
        'plan_sha256': 'wrong', 'commands': [{'argv': ['python', 'replacement.py']}]})
    with session.scope():
        DiscoveredRepositoryReproduction._receive_author_plan('CodeExecutor', plan)
    reply = next(m for m in session.snapshot()['messages'] if m.get('reply_to') == stale['id'])
    assert reply['content']['status'] == 'ignored'
    assert reply['content']['plan_sha256'] == hashlib.sha256(_canonical(plan).encode()).hexdigest()
    assert plan == before


def test_progress_messages_are_visible_without_completing_any_phase(app, tmp_path):
    path = tmp_path / 'collaboration.jsonl'
    store = ProgressStore(str(path))
    store.emit({'type': 'pipeline_plan', 'stages': [
        {'id': 'execute_repository', 'agent': 'CodeExecutor', 'title': '执行作者命令'}]})
    store.emit({'type': 'state', 'state': 'EXECUTE_CODE', 'phase_id': 'execute_repository',
                'agent': 'CodeExecutor', 'status': 'running'})
    session = CollaborationSession(on_event=store.emit)
    request = session.send('RepositoryRepairer', 'PatchValidator', 'request', {'reason': 'check original protocol'})
    session.reply(request, 'PatchValidator', {'reason': 'reject changed epoch count'}, kind='feedback')
    view = ProgressStore.read_snapshot(str(path))
    assert len(view['agent_messages']) == 2
    assert view['running'] and not view['done']
    assert view['pipeline_stages'][0]['status'] == 'running'
    app.session_state['progress_file'] = str(path)
    app.run()
    assert not app.exception
    assert any('Agent 协作记录' in item.label for item in app.expander)
    assert any('PatchValidator' in item.value for item in app.markdown)
