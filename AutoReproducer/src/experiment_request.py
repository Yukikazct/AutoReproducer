"""Optional user selection, kept separate from paper/source evidence."""
import re


def normalize_experiment_request(value=None):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {'method', 'dataset'}:
        raise ValueError('experiment_request must contain only method and dataset names')
    selected = {}
    for name, text in value.items():
        if (not isinstance(text, str) or len(text) > 128
                or any(ord(char) < 32 or ord(char) == 127 for char in text)):
            raise ValueError('Experiment ' + name + ' must be a single name of at most 128 characters')
        if text.strip():
            selected[name] = text.strip()
    return selected


def bind_experiment_request(protocol, value=None):
    """A strict protocol cannot quietly select a different requested experiment.

    Names ignore punctuation/case (e.g. a paper's marked model row). This is
    selection only: citations and the frozen command still establish identity.
    """
    requested = normalize_experiment_request(value)
    for kind, expected in requested.items():
        actual = (protocol.get(kind) or {}).get('name', '')
        canonical = lambda text: re.sub(r'[\W_]+', '', text.casefold())
        if not isinstance(actual, str) or not canonical(expected) or canonical(actual) != canonical(expected):
            raise ValueError('Selected experiment ' + kind + ' differs from the user request: ' + expected)
    return requested
