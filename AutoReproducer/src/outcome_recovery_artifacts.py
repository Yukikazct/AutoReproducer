"""Locate the one owned outcome attempt under a stable canonical run root."""
from pathlib import Path

from src.safety.paths import workspace_path


def outcome_attempt_directory(data):
    attempts = (data.get('outcome_recovery') or {}).get('attempts') or []
    if not attempts or not data.get('run_dir'):
        return None
    if len(attempts) != 1 or attempts[0].get('id') != 1:
        raise ValueError('Outcome recovery has no unique owned attempt directory')
    expected = workspace_path(data['run_dir'], 'outcome_recovery/attempt_1', 'outcome recovery artifacts')
    if Path(attempts[0].get('run_dir', '')).absolute() != expected.absolute():
        raise ValueError('Outcome attempt escaped its canonical run directory')
    if data.get('active_attempt_dir') and Path(data['active_attempt_dir']).absolute() != expected.absolute():
        raise ValueError('Active outcome attempt differs from its owned artifact directory')
    if not expected.exists():
        return None
    if not expected.is_dir():
        raise ValueError('Outcome attempt is not an artifact directory')
    return expected
