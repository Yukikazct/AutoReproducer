"""Known title routing must preserve explicit user choices and exact identity."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.title_routing import match_paper_title, resolve_title_request


NEURAL_ODE = "Neural Ordinary Differential Equations"


@pytest.mark.parametrize("title,profile,scope", [
    (NEURAL_ODE, "neural_ode_spiral", "official_method_experiment"),
    ("Are Transformers Effective for Time Series Forecasting?",
     "dlinear_etth1_reference", "selected_paper_experiment"),
    ("Implicit Neural Representations with Periodic Activation Functions",
     "siren_camera_quick", "official_method_experiment"),
])
def test_canonical_titles_select_reviewed_experiment_with_provenance(title, profile, scope):
    request = {"paper_title": title}
    result = resolve_title_request(request)
    assert match_paper_title(title) == profile
    assert result["experiment_profile"] == profile
    assert result["title_resolution"] == {
        "requested_title": title, "profile": profile,
        "source": "exact_paper_title", "scope": scope,
    }
    assert request == {"paper_title": title}


@pytest.mark.parametrize("title", [
    "  neural ordinary differential equations  ",
    "NEURAL ORDINARY DIFFERENTIAL EQUATIONS",
    '"Neural Ordinary Differential Equations"',
    "‘Neural Ordinary Differential Equations’",
    "Neural\nOrdinary\tDifferential   Equations",
    "Ｎｅｕｒａｌ Ordinary Differential Equations",
])
def test_formatting_normalization_does_not_require_search_or_runtime(title):
    assert match_paper_title(title) == "neural_ode_spiral"
    assert resolve_title_request({"paper_title": title})["title_resolution"]["requested_title"] == title


@pytest.mark.parametrize("title", [
    "Neural ODE", "Neural Ordinary Differential Equation",
    "Neural Ordinary Differential Equations for Time Series",
    "A Review of Neural Ordinary Differential Equations",
    "Neural Ordinary Differential Equations?", "", None, 42,
])
def test_similar_or_unknown_titles_do_not_override_generic_route(title):
    request = {"paper_title": title, "use_llm_review": True}
    assert match_paper_title(title) is None
    result = resolve_title_request(request)
    assert result == request and result is not request
    assert "experiment_profile" not in result and "title_resolution" not in result


@pytest.mark.parametrize("explicit", [
    {"experiment_profile": "dlinear_etth1_smoke"},
    {"mock_mode": True},
    {"pdf_path": "chosen.pdf"},
    {"code": "print('user supplied code')"},
    {"preferred_repo_url": "https://github.com/example/chosen"},
    {"code_repo_url": "https://github.com/example/chosen"},
    {"corpus_paper": "chosen_corpus_paper"},
])
def test_explicit_inputs_keep_their_existing_route(explicit):
    request = {"paper_title": NEURAL_ODE, **explicit}
    result = resolve_title_request(request)
    assert result == request and result is not request
    assert "title_resolution" not in result


@pytest.mark.parametrize("review,sharing,optimization", [
    (False, False, "off"), (True, True, "validate"), (False, True, "suggest"),
])
def test_resolution_preserves_review_sharing_optimization_and_other_settings(review, sharing, optimization):
    options = {
        "use_llm_review": review, "allow_result_summary_review": sharing,
        "optimization_mode": optimization, "budget_seconds": 7200,
        "max_candidates": 2, "prepare_only": True, "use_docker": True,
    }
    request = {"paper_title": NEURAL_ODE, **options}
    result = resolve_title_request(request)
    assert all(result[key] == value for key, value in options.items())
    assert result["experiment_profile"] == "neural_ode_spiral"
    assert request == {"paper_title": NEURAL_ODE, **options}


def test_empty_optional_inputs_do_not_block_title_resolution_and_resolution_is_idempotent():
    request = {"paper_title": NEURAL_ODE, "mock_mode": False,
               "experiment_profile": None, "pdf_path": "", "code": "",
               "preferred_repo_url": "", "code_repo_url": "", "corpus_paper": None}
    first = resolve_title_request(request)
    second = resolve_title_request(first)
    assert first["experiment_profile"] == "neural_ode_spiral"
    assert second == first and second is not first


def test_title_resolution_imports_without_site_packages():
    project = Path(__file__).resolve().parents[1]
    script = (
        "import json,sys;sys.path.insert(0,sys.argv[1]);"
        "from src.title_routing import resolve_title_request;"
        "print(json.dumps(resolve_title_request({'paper_title':"
        "'Neural Ordinary Differential Equations'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(project)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert json.loads(completed.stdout)["experiment_profile"] == "neural_ode_spiral"
