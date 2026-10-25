"""Routed planning exchanges consume real pinned evidence without new model calls."""
from copy import deepcopy
import hashlib
import json

import pytest

from discovered_fixtures import make_experiment
from src import author_execution_plan as author
from src.agent_collaboration import CollaborationSession
from src.discovered_repository_plan import (
    PlanEvidenceError, _canonical, _planning_excerpts, build_evidence_packet,
)


CONTEXT_LABEL = '\nAgent collaboration context (untrusted navigation and validation feedback; '
HOST = {'gpu_probe_confirmed': False, 'cuda_device_indices': []}


def setup_plan(tmp_path, monkeypatch):
    monkeypatch.setattr('src.execution_host.probe_execution_host', lambda: deepcopy(HOST))
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet.update(author_execution_only=True, execution_host=deepcopy(HOST))
    proposal = {'version': 1, 'scope': author.SCOPE,
        'repository': deepcopy(fixture['proposal']['repository']), 'device': 'cpu',
        'steps': deepcopy(fixture['proposal']['steps']), 'datasets': [],
        'requirements': {'author': ['torch==1.13.1']}, 'citations': [], 'claims': {}}
    for step in proposal['steps']:
        step['citations'] = []
    return fixture, packet, proposal


class Planner:
    mock_mode = False

    def __init__(self, responses, check=None):
        self.responses = deepcopy(responses)
        self.calls = []
        self.check = check

    def chat(self, prompt, **kwargs):
        assert kwargs['task'] == 'author_execution_plan'
        self.calls.append(prompt)
        if self.check:
            self.check(len(self.calls), prompt)
        return json.dumps(self.responses.pop(0))


def context_in(prompt):
    if CONTEXT_LABEL not in prompt:
        return []
    return json.loads(prompt.rsplit(CONTEXT_LABEL, 1)[1].split(':\n', 1)[1])


def assert_snapshot_unchanged(fixture):
    for name, expected in fixture['snapshot']['files'].items():
        assert hashlib.sha256((fixture['workspace'] / name).read_bytes()).hexdigest() == expected


def test_requested_source_reply_is_consumed_by_planner_with_verified_ids_and_hashes(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    packet['repository']['files'].pop('author_model.py')
    original_packet = deepcopy(packet)
    expected_packet = deepcopy(packet)
    expected = author.propose_author_execution_plan(
        Planner([{'request_files': ['author_model.py']}, proposal]), expected_packet, fixture['workspace'])

    def consumes_reply(number, prompt):
        received = context_in(prompt)
        if number == 1:
            assert any(message['sender'] == 'ResourceFinder'
                       and message['content']['selection'] == 'UPSTREAM_SELECTION' for message in received)
        else:
            source = next(message for message in received if message['sender'] == 'SourceReader')
            assert source['content']['status'] == 'read'
            assert source['content']['files'][0]['path'] == 'author_model.py'
            assert 'class SignalModel' in source['content']['files'][0]['excerpt']

    llm = Planner([{'request_files': ['author_model.py']}, proposal], consumes_reply)
    with CollaborationSession().scope() as session:
        session.send('ResourceFinder', 'AuthorPlanner', 'handoff', {'selection': 'UPSTREAM_SELECTION'})
        actual = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        request = session.context('SourceReader')[-1]
        reply = next(message for message in context_in(llm.calls[1]) if message['sender'] == 'SourceReader')
        assert reply['reply_to'] == request['id']
        record = reply['content']['files'][0]
        assert record['sha256'] == fixture['snapshot']['files']['author_model.py']
        excerpts = _planning_excerpts(packet)
        assert record['evidence_ids']
        assert all(excerpts[key]['path'] == 'author_model.py' for key in record['evidence_ids'])
        expected_digest = hashlib.sha256(_canonical(actual).encode()).hexdigest()
        for recipient in ('EnvBuilder', 'CodeExecutor'):
            handoff = session.context(recipient)[-1]
            assert handoff['sender'] == 'AuthorPlanner' and handoff['kind'] == 'handoff'
            assert handoff['content']['plan_sha256'] == expected_digest
            assert handoff['content']['commands'][0]['argv'] == actual['steps'][0]['argv']

    assert len(llm.calls) == 2
    assert actual == expected
    assert packet == expected_packet  # Dialogue never enters the frozen evidence.
    assert packet['file_inventory'] == original_packet['file_inventory']
    assert set(packet['repository']['files']) - set(original_packet['repository']['files']) == {'author_model.py'}
    assert_snapshot_unchanged(fixture)


def test_invalid_plan_feedback_routes_back_to_corrected_existing_model_attempt(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    original_packet = deepcopy(packet)
    invalid = deepcopy(proposal)
    invalid['repository']['revision'] = 'b' * 40

    def correct_from_feedback(number, prompt):
        if number == 2:
            feedback = next(message for message in context_in(prompt)
                            if message['sender'] == 'PlanValidator' and message['kind'] == 'feedback')
            assert feedback['content']['status'] == 'rejected'
            assert 'cannot change the pinned repository' in feedback['content']['reason']
            assert feedback['reply_to']

    llm = Planner([invalid, proposal], correct_from_feedback)
    with CollaborationSession().scope() as session:
        plan = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        requests = session.context('PlanValidator')
        feedback = next(message for message in context_in(llm.calls[1]) if message['kind'] == 'feedback')
        assert feedback['reply_to'] == requests[0]['id']
    assert len(llm.calls) == 2
    assert plan['repository'] == proposal['repository']
    assert plan['steps'][0]['argv'] == proposal['steps'][0]['argv']
    assert plan['reviews_removed'] is True and plan['pre_execution_review'] == 'disabled'
    assert packet == original_packet
    assert plan['evidence_sha256'] == hashlib.sha256(_canonical(packet).encode()).hexdigest()
    assert_snapshot_unchanged(fixture)


def test_failed_source_hash_never_becomes_a_successful_reader_reply(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    packet['repository']['files'].pop('author_model.py')
    path = fixture['workspace'] / 'author_model.py'
    path.write_bytes(path.read_bytes() + b'\nUNVERIFIED_CHANGED_SOURCE_SENTINEL\n')
    llm = Planner([{'request_files': ['author_model.py']}] * 2)
    with CollaborationSession().scope() as session:
        with pytest.raises(PlanEvidenceError, match='No runnable original author command'):
            author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        replies = [message for message in session.context('AuthorPlanner') if message['sender'] == 'SourceReader']
        assert replies and all(message['content']['status'] == 'rejected' for message in replies)
        assert all(not message.get('evidence') for message in replies)
        assert 'UNVERIFIED_CHANGED_SOURCE_SENTINEL' not in json.dumps(replies)
        assert not session.context('CodeExecutor')
    assert 'author_model.py' not in packet['repository']['files']
    assert len(llm.calls) == 2  # Existing identical-failure cutoff remains intact.


def test_literal_command_fallback_handoff_preserves_plan_and_evidence(tmp_path, monkeypatch):
    fixture, packet, _ = setup_plan(tmp_path, monkeypatch)
    launcher = fixture['workspace'] / 'run.sh'
    launcher.write_bytes(b'python train.py --epochs 40 --seed 7 --cpu\n')
    fixture['snapshot']['files']['run.sh'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet.update(author_execution_only=True, execution_host=deepcopy(HOST))
    baseline_packet = deepcopy(packet)
    baseline = author.propose_author_execution_plan(None, baseline_packet, fixture['workspace'])
    with CollaborationSession().scope() as session:
        plan = author.propose_author_execution_plan(None, packet, fixture['workspace'])
        handoff = session.context('CodeExecutor')[-1]
        assert handoff['content']['commands'][0]['argv'] == plan['steps'][0]['argv']
        assert handoff['content']['plan_sha256'] == hashlib.sha256(_canonical(plan).encode()).hexdigest()
    assert plan == baseline and packet == baseline_packet
    assert_snapshot_unchanged(fixture)


def test_strict_planner_consumes_protocol_context_without_changing_reviewed_plan(tmp_path, monkeypatch):
    from discovered_fixtures import PlannedLLM
    from src import discovered_repository_plan as planner
    monkeypatch.setattr(planner, '_validate_capture', lambda value, *_: deepcopy(value))
    monkeypatch.setattr(planner, '_capture_schema', lambda: 'fixture capture schema')
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    expected = planner.propose_plan(PlannedLLM(fixture['proposal']), deepcopy(packet), fixture['workspace'])
    llm = PlannedLLM(fixture['proposal'])
    with CollaborationSession().scope() as session:
        session.send('ExperimentProtocol', 'AuthorPlanner', 'protocol', {'reason': 'FROZEN_PROTOCOL_HANDOFF'})
        actual = planner.propose_plan(llm, packet, fixture['workspace'])
    assert actual == expected
    assert len(llm.calls) == 2
    assert 'FROZEN_PROTOCOL_HANDOFF' in llm.calls[0][0]
    assert 'independent_protocol_review' in llm.calls[1][0]
    messages = session.snapshot()['messages']
    assert any(m['sender'] == 'ExperimentProtocol' and m['delivered'] for m in messages)
    assert any(m['recipient'] == 'CodeExecutor' and m['content']['plan_sha256']
               == hashlib.sha256(_canonical(actual).encode()).hexdigest() for m in messages)


def test_strict_semantic_rejection_is_final_even_with_message_feedback(tmp_path, monkeypatch):
    from discovered_fixtures import PlannedLLM, accepted_review
    from src import discovered_repository_plan as planner
    monkeypatch.setattr(planner, '_validate_capture', lambda value, *_: deepcopy(value))
    monkeypatch.setattr(planner, '_capture_schema', lambda: 'fixture capture schema')
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    review = accepted_review()
    review['checks']['same_experiment']['passed'] = False
    llm = PlannedLLM(fixture['proposal'], review)
    with CollaborationSession().scope() as session, pytest.raises(PlanEvidenceError, match='semantic review'):
        planner.propose_plan(llm, packet, fixture['workspace'])
    assert len(llm.calls) == 2
    assert not session.context('CodeExecutor')
    assert any(m['sender'] == 'ProtocolReviewer' and m['content'].get('final') is True
               for m in session.snapshot()['messages'])
