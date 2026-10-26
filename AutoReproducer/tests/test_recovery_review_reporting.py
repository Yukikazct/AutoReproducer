"""Count observed review interventions without claiming a recovery benefit."""
from copy import deepcopy

from src.report_details import intelligent_recovery_summary


def test_short_review_summary_counts_candidates_once_and_retains_saved_data():
    recovery = {'enabled': True, 'mode': 'collaborative', 'status': 'investigated', 'tasks': [],
        'usage': {'total_tokens': 95}, 'reviews': [
            {'stage': 'investigation', 'reviewer': 'independent', 'action': 'accept'},
            {'stage': 'patch', 'reviewer': 'independent', 'action': 'probe'},
            {'stage': 'patch', 'reviewer': 'original_contract', 'action': 'reject', 'candidate_sha256': 'a'},
            {'stage': 'patch', 'reviewer': 'independent', 'action': 'reject', 'candidate_sha256': 'a'},
            {'stage': 'patch', 'reviewer': 'independent', 'action': 'accept', 'candidate_sha256': 'b'}]}
    original = deepcopy(recovery)
    summary = intelligent_recovery_summary(recovery)
    assert '独立反证审查 4 次；请求补查 1 次；否决候选补丁 1 个' in summary
    assert '收益' not in summary and '成功率' not in summary
    assert len(summary) < 500
    assert recovery == original


def test_old_report_without_review_receipts_does_not_invent_review_counts():
    summary = intelligent_recovery_summary({'enabled': True, 'mode': 'collaborative', 'tasks': []})
    assert '审查作用' not in summary


def test_single_mode_does_not_describe_self_checks_as_independent_agents():
    summary = intelligent_recovery_summary({'enabled': True, 'mode': 'single', 'tasks': [],
        'reviews': [{'reviewer': 'self_check', 'action': 'accept', 'stage': 'investigation'}]})
    assert '自检 1 次' in summary
    assert '独立反证审查' not in summary
