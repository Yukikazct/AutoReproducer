"""Resolve exact supported paper titles before loading a runtime or pipeline.

This registry deliberately uses complete titles. A search result, similar title,
or generated repository guess cannot select a reviewed experiment.
"""
import re
import unicodedata


_PAPERS = {
    "neural ordinary differential equations": (
        "neural_ode_spiral", "official_method_experiment"),
    "are transformers effective for time series forecasting?": (
        "dlinear_etth1_reference", "selected_paper_experiment"),
    "implicit neural representations with periodic activation functions": (
        "siren_camera_quick", "official_method_experiment"),
}


def _normalized_title(title):
    if not isinstance(title, str):
        return ""
    normalized = unicodedata.normalize("NFKC", title)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \"'`“”‘’").casefold()


def match_paper_title(title):
    """Return a reviewed profile for a complete canonical title, or None."""
    match = _PAPERS.get(_normalized_title(title))
    return match[0] if match is not None else None


def resolve_title_request(request):
    """Copy a request and resolve an unambiguous title without changing options.

    Explicit profiles, mock runs, source files/code, repository preferences and
    corpus selections retain their existing routes. Review, result sharing and
    optimization preferences are left to the caller, including explicit False.
    """
    resolved = dict(request)
    if request.get("mock_mode") or any(request.get(key) for key in (
        "experiment_profile", "pdf_path", "code", "preferred_repo_url",
        "code_repo_url", "corpus_paper",
    )):
        return resolved

    requested_title = request.get("paper_title")
    match = _PAPERS.get(_normalized_title(requested_title))
    if match is None:
        return resolved

    profile_id, scope = match
    resolved["experiment_profile"] = profile_id
    resolved["title_resolution"] = {
        "requested_title": requested_title,
        "profile": profile_id,
        "source": "exact_paper_title",
        "scope": scope,
    }
    return resolved
