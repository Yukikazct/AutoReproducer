"""Dependency-free recovery controls for the host-to-managed-runtime handoff.

The host can validate requests before application dependencies are installed.
Keep investigation, repository and model imports out of this module.
"""


def normalize_intelligent_recovery_request(request):
    enabled = request.get('intelligent_recovery', False)
    if type(enabled) is not bool:
        raise ValueError('intelligent_recovery must be an explicit boolean')
    mode = request.get('intelligent_recovery_mode', 'collaborative')
    if not isinstance(mode, str) or mode not in {'single', 'collaborative'}:
        raise ValueError('intelligent_recovery_mode must be single or collaborative')
    return {**request, 'intelligent_recovery': enabled,
            'intelligent_recovery_mode': mode if enabled else 'collaborative'}
