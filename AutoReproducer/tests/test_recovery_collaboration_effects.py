"""Review findings affect candidates and remain attributable to their scope."""
from copy import deepcopy
import json

import pytest

from src.intelligent_recovery import InvestigationStopped, RecoveryBudgetExceeded, RecoverySession


class Model:
    mock_mode = False

    def __init__(self, respond):
        self.respond = respond
        self.calls = []
        self.last_usage = {}

    def chat(self, prompt, **kwargs):
        context = json.loads(prompt)
        self.calls.append((context, kwargs))
        self.last_usage = {'total_tokens': 19}
        return json.dumps(self.respond(context, kwargs))


class Tools:
    def __init__(self):
        self.calls = []

    def execute(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {'status': 'completed', 'cleanup_confirmed': True, 'tool': tool,
                'evidence': [{'id': 'observation_' + str(len(self.calls)),
                              'source': 'frozen_source', 'value': arguments}]}


def active_session(tmp_path, respond, *, mode='collaborative', budget=None):
    session = RecoverySession(Model(respond), mode=mode, budget=budget)
    session.attach({}, tmp_path)
    session.state['identity'].update(source_sha256='a' * 64, plan_sha256='b' * 64)
    task = {'id': 'investigation_1', 'trigger': 'execution_failure', 'status': 'repair_proposed',
            'evidence': [{'id': 'checked_api', 'source': 'frozen_source', 'value': 'observed API'}],
            'experiment_understanding': {'repeats': 10}, 'probes': [], 'decisions': [], 'hypotheses': []}
    session.state['tasks'].append(task)
    session._active_task = task
    session._toolset = Tools()
    return session


def accept(ref='patch_sources'):
    return {'action': 'accept', 'reason': 'No counterexample found in the original evidence.', 'evidence_ids': [ref]}


def test_patch_reviewer_receives_investigator_observations_and_experiment_identity(tmp_path):
    def respond(context, kwargs):
        assert kwargs['task'] == 'recovery_counter_review'
        assert context['experiment_understanding'] == {'repeats': 10}
        assert any(item['id'] == 'checked_api' for item in context['evidence'])
        assert context['identity']['source_sha256'] == 'a' * 64
        return accept('checked_api')
    session = active_session(tmp_path, respond)
    session.review_patch({'sources': {'model.py': 'original'}}, {'patches': []})
    assert not session._toolset.calls
    review = session.state['reviews'][0]
    assert review['stage'] == 'patch' and review['reviewer'] == 'independent'
    assert review['candidate_sha256'] and review['identity_sha256']
    assert review['evidence_ids'] == ['checked_api']
    assert session.state['usage']['roles']['recovery_counter_review']['total_tokens'] == 19


def test_reviewer_can_follow_two_different_checks_before_accepting(tmp_path):
    def respond(context, kwargs):
        count = len(context['probe_results'])
        if count < 2:
            return {'action': 'probe', 'reason': 'Need another independently observed condition.',
                    'evidence_ids': ['patch_sources'],
                    'probe': {'tool': 'source_context', 'arguments': {'paths': ['api' + str(count) + '.py']}}}
        assert {item['id'] for item in context['evidence']} >= {'observation_1', 'observation_2'}
        return accept('observation_2')
    session = active_session(tmp_path, respond)
    session.review_patch({'sources': {}}, {'patches': []})
    assert len(session._toolset.calls) == 2
    assert session.state['usage']['probes'] == 2
    assert [row['action'] for row in session.state['reviews']] == ['probe', 'probe', 'accept']
    assert len(session._active_task['patch_reviews'][0]['probes']) == 2


def test_repeated_review_check_is_rejected_without_spending_shared_budget_again(tmp_path):
    def respond(context, kwargs):
        return {'action': 'probe', 'reason': 'Check the implementation.', 'evidence_ids': ['patch_sources'],
                'probe': {'tool': 'source_context', 'arguments': {'paths': ['api.py']}}}
    session = active_session(tmp_path, respond)
    with pytest.raises(ValueError, match='repeated'):
        session.review_patch({'sources': {}}, {'patches': []})
    assert len(session._toolset.calls) == 1
    assert session.state['reviews'][-1]['reviewer'] == 'controller'
    with pytest.raises(ValueError, match='already rejected; no progress'):
        session.review_patch({'sources': {}}, {'patches': []})
    assert len(session.llm.calls) == 2
    assert session.state['usage']['probes'] == 1


def test_review_and_direct_probe_share_the_same_allowance(tmp_path):
    def respond(context, kwargs):
        return {'action': 'probe', 'reason': 'Need an observation.', 'evidence_ids': ['patch_sources'],
                'probe': {'tool': 'source_context', 'arguments': {'paths': ['api.py']}}}
    session = active_session(tmp_path, respond, budget={'max_probes': 1})
    session.probe(session._toolset, 'source_context', {'paths': ['other.py']}, session._active_task)
    with pytest.raises(ValueError, match='shared probe budget'):
        session.review_patch({'sources': {}}, {'patches': []})
    assert len(session._toolset.calls) == 1
    assert session.state['reviews'][-1]['action'] == 'reject'
    session.llm.respond = lambda *_: accept()
    with pytest.raises(ValueError, match='already rejected; no progress'):
        session.review_patch({'sources': {}}, {'patches': []})
    # A direct check cannot reset the reviewer allowance. Exhausting the
    # task-wide budget is terminal even when a later model would accept.
    with pytest.raises(RecoveryBudgetExceeded):
        session.probe(session._toolset, 'source_context', {'paths': ['third.py']}, session._active_task)
    with pytest.raises(InvestigationStopped, match='probe_budget_exhausted'):
        session.review_patch({'sources': {}}, {'patches': []})
    assert session.state['usage']['probes'] == 1
    assert len(session._toolset.calls) == 1
    assert len(session.llm.calls) == 1


def test_rejected_candidate_evidence_is_not_relabelled_as_next_candidate_evidence(tmp_path):
    first = True
    def respond(context, kwargs):
        nonlocal first
        if first:
            first = False
            return {'action': 'probe', 'reason': 'Check original assertion.', 'evidence_ids': ['patch_sources'],
                    'probe': {'tool': 'author_contract', 'arguments': {'path': 'test.py', 'function': 'test_api'}}}
        assert not context['probe_results']
        assert 'candidate_failure' not in {item['id'] for item in context['evidence']}
        return accept()
    session = active_session(tmp_path, respond)
    def failed(tool, arguments):
        return {'status': 'failed', 'cleanup_confirmed': True, 'tool': tool,
                'evidence': [{'id': 'candidate_failure', 'source': 'original_assertions',
                              'value': {'candidate_sha256': 'first-candidate'}}]}
    session._toolset.execute = failed
    with pytest.raises(ValueError, match='failed the unchanged original assertion'):
        session.review_patch({'sources': {}}, {'patches': [{'content': 'wrong'}]})
    feedback = session.repair_feedback()
    assert feedback['reviews'][-1]['reviewer'] == 'original_contract'
    assert feedback['candidate_observations'][0]['scope'] == 'that_candidate_only'
    assert feedback['candidate_observations'][0]['probes'][0]['status'] == 'failed'
    session.review_patch({'sources': {}}, {'patches': [{'content': 'fixed'}]})
    assert len(session.state['reviews']) == 3
    assert session.state['reviews'][-1]['candidate_sha256'] != session.state['reviews'][-2]['candidate_sha256']
    assert session._active_task['evidence'][0]['id'] == 'checked_api'
    assert session.repair_findings()[0]['probes'] == []
    assert 'patch_reviews' not in session.repair_findings()[0]


@pytest.mark.parametrize('mode,role,reviewer', [('single', 'recovery_self_check', 'self_check'),
    ('collaborative', 'recovery_counter_review', 'independent')])
def test_role_usage_and_review_receipts_distinguish_self_check_from_collaboration(tmp_path, mode, role, reviewer):
    session = active_session(tmp_path, lambda *_: accept(), mode=mode)
    before = deepcopy(session.state['identity'])
    session.review_patch({'sources': {}}, {'patches': []})
    assert session.state['usage']['roles'][role]['model_requests'] == 1
    assert session.state['usage']['roles'][role]['total_tokens'] == 19
    assert session.state['reviews'][0]['reviewer'] == reviewer
    assert session.state['identity'] == before
    assert 'execution_recovered' not in session._active_task


def test_failed_model_call_keeps_role_request_and_unknown_tokens(tmp_path):
    session = active_session(tmp_path, lambda *_: accept())
    def unavailable(*args, **kwargs):
        raise RuntimeError('provider unavailable')
    session.llm.chat = unavailable
    with pytest.raises(RuntimeError, match='provider unavailable'):
        session.wrap_llm().chat('{}', task='recovery_counter_review')
    usage = session.state['usage']['roles']['recovery_counter_review']
    assert usage['model_requests'] == 1 and usage['total_tokens'] is None
    assert usage['missing_usage_requests'] == 1
    assert session.state['reviews'] == []


def test_rewording_and_reordering_rejected_patch_does_not_buy_another_review(tmp_path):
    session = active_session(tmp_path, lambda *_: {'action': 'reject', 'reason': 'Counterexample.',
                                                 'evidence_ids': ['patch_sources']})
    patches = [{'path': 'a.py', 'base_sha256': 'a', 'content': 'wrong a'},
               {'path': 'b.py', 'base_sha256': 'b', 'content': 'wrong b'}]
    with pytest.raises(ValueError, match='Counterexample'):
        session.review_patch({'sources': {}}, {'reason': 'first attempt', 'patches': patches})
    with pytest.raises(ValueError, match='already rejected; no progress'):
        session.review_patch({'sources': {}}, {'reason': 'different justification', 'patches': list(reversed(patches))})
    assert len(session.llm.calls) == 1
    assert len(session.state['reviews']) == 1


def test_unused_probe_metadata_cannot_disguise_the_same_observation(tmp_path):
    def respond(context, kwargs):
        return {'action': 'probe', 'reason': 'Observe API.', 'evidence_ids': ['patch_sources'],
                'probe': {'tool': 'source_context', 'arguments': {'paths': ['api.py']},
                          'explanation': 'try ' + str(len(context['probe_results']))}}
    session = active_session(tmp_path, respond)
    with pytest.raises(ValueError, match='repeated'):
        session.review_patch({'sources': {}}, {'patches': []})
    assert len(session._toolset.calls) == 1
    assert session.state['usage']['probes'] == 1
