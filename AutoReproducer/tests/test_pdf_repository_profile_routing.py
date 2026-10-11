"""A reviewed repository experiment requires both visible identity and code evidence."""
import hashlib
from types import ModuleType
import sys
from unittest.mock import Mock

import pytest

import src.pdf_input as pdf_input
import src.repository_routing as routing
import src.repository_reproduction as repository
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.repository_evidence import extract_repository_links
from test_pdf_input_routing import write_pdf
from test_pdf_repository_evidence import add_uri


TITLE = routing.REZERO_TITLE
AUTHOR = routing.REZERO_DISCOVERY_REPOSITORY
TRAINING = routing.REZERO_TRAINING_REPOSITORY
PROFILE = routing.REZERO_PROFILE_ID
CLAIM = "2Code for ReZero applied to various neural architectures: "


@pytest.mark.parametrize("statement", [
    CLAIM, "Code for ReZero applied to various neural architectures: ",
    "2Our code is available at ", "2Code is available at ",
])
def test_numeric_footnote_marker_does_not_hide_original_code_declaration(statement):
    evidence = extract_repository_links([statement + AUTHOR])[0]
    assert evidence["is_author_code"] and evidence["evidence_type"] == "author_code_statement"


@pytest.mark.parametrize("statement", [CLAIM, "2Code is available at "])
def test_numeric_declaration_on_reference_pages_stays_a_reference(statement):
    evidence = extract_repository_links(["References\n[1] Prior work.", statement + AUTHOR])[0]
    assert not evidence["is_author_code"] and evidence["evidence_type"] == "reference"
    assert routing.match_repository_profile(TITLE, [evidence]) is None


def test_binary_pdf_routes_original_author_footnote_and_preserves_discovery_training_distinction(tmp_path):
    path = write_pdf(tmp_path / "arbitrary-filename.pdf", [
        "ReZero is All You Need:", "Fast Convergence at Large Depth", "Authors", "Abstract",
        "Paper content.", CLAIM + "https://github.", "com/majumderb/rezero",
    ], later_pages=(["Prior implementation: https://github.com/fastai/imagenet-fast"],))
    request = {"pdf_path": str(path), "paper_title": "stale unrelated textbox",
               "use_llm_review": False, "optimization_mode": "off"}
    resolved = pdf_input.resolve_pdf_request(request)
    assert resolved["experiment_profile"] == PROFILE
    resolution = resolved["pdf_resolution"]
    assert resolution["title"] == TITLE and resolution["source"] == "pdf_author_code_repository"
    assert resolution["scope"] == "selected_paper_experiment"
    assert resolution["discovery_repository_url"] == AUTHOR
    assert resolution["training_repository_url"] == TRAINING and TRAINING != AUTHOR
    assert resolution["repository_relationship"]["source"] == "reviewed_author_readme"
    hops = resolution["repository_relationship"]["hops"]
    assert len(hops) == 2
    assert hops[0]["repository_url"] == AUTHOR
    assert hops[0]["target_repository_url"] == hops[1]["repository_url"] == routing.REZERO_EXAMPLES_REPOSITORY
    assert hops[1]["target_repository_url"] == TRAINING
    assert all(hop["revision"] in hop["source_url"] and len(hop["sha256"]) == 64 for hop in hops)
    assert resolution["training_repository_revision"] == routing.REZERO_TRAINING_REVISION
    assert resolution["evidence"]["page"] == 1
    assert resolution["evidence"]["source"] == "pdf_text"
    assert "https://github.\ncom/majumderb/rezero" == resolution["evidence"]["raw_url"]
    assert CLAIM in resolution["evidence"]["context"]
    assert resolution["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolution["pages"] == 2 and resolution["bytes"] == path.stat().st_size
    assert len(resolution["repository_links"]) == 2
    assert all(resolved[key] == value for key, value in request.items())
    assert "pdf_resolution" not in request


def test_original_annotation_can_select_reviewed_profile_with_original_visible_declaration(tmp_path):
    path = write_pdf(tmp_path / "annotation.pdf", [TITLE, "Authors", "Abstract",
                                                     "Our source code is available here."])
    add_uri(path, 0, AUTHOR, rect=(72, 655, 400, 668))
    resolved = pdf_input.resolve_pdf_request({"pdf_path": str(path)})
    assert resolved["experiment_profile"] == PROFILE
    assert resolved["pdf_resolution"]["evidence"]["source"] == "pdf_annotation"


@pytest.mark.parametrize("header,body,metadata", [
    ([TITLE, "Abstract"], [AUTHOR], None),
    ([TITLE, "Abstract"], [CLAIM + "https://github.com/some-fork/rezero"], None),
    ([TITLE, "Abstract"], [CLAIM + TRAINING], None),
    ([TITLE, "Abstract", "References"], [CLAIM + AUTHOR], None),
    (["An unrelated paper", "Abstract", TITLE], [CLAIM + AUTHOR], TITLE),
    (["ReZero is All You Need for Images", "Abstract"], [CLAIM + AUTHOR], TITLE),
    (["Abstract", TITLE], [CLAIM + AUTHOR], TITLE),
    ([TITLE, "Abstract"], [CLAIM + AUTHOR], "Neural Ordinary Differential Equations"),
    ([TITLE, "Abstract"], ["Code for the baseline: " + AUTHOR], None),
])
def test_repo_link_metadata_title_and_reference_cannot_replace_joint_original_evidence(tmp_path, header, body, metadata):
    path = write_pdf(tmp_path / "paper.pdf", [*header, *body], metadata_title=metadata)
    request = {"pdf_path": str(path)}
    assert pdf_input.resolve_pdf_request(request) == request


@pytest.mark.parametrize("explicit", [
    {"mock_mode": True}, {"experiment_profile": "dlinear_etth1_smoke"},
    {"preferred_repo_url": AUTHOR}, {"code_repo_url": AUTHOR},
    {"code": "print('user code')"}, {"corpus_paper": "external-corpus"},
])
def test_rezero_pdf_respects_explicit_modes_and_presets(tmp_path, explicit):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    request = {"pdf_path": str(path), **explicit}
    assert pdf_input.resolve_pdf_request(request) == request


def test_worker_request_with_matching_profile_rebuilds_hash_and_does_not_trust_preview(tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    request = {"pdf_path": str(path), "experiment_profile": PROFILE,
               "pdf_resolution": {"sha256": "forged", "evidence": {"url": TRAINING}}}
    resolved = pdf_input.resolve_pdf_request(request)
    assert resolved["pdf_resolution"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolved["pdf_resolution"]["evidence"]["url"] == AUTHOR
    assert request["pdf_resolution"]["sha256"] == "forged"


@pytest.mark.parametrize("header,claim,metadata", [
    ("An unrelated paper", CLAIM + AUTHOR, None),
    (TITLE, AUTHOR, None),
    (TITLE, CLAIM + "https://github.com/fork/rezero", None),
    (TITLE, CLAIM + AUTHOR, "Neural Ordinary Differential Equations"),
])
def test_worker_rejects_stale_or_forged_rezero_profile_after_pdf_evidence_changes(tmp_path, header, claim, metadata):
    path = write_pdf(tmp_path / "paper.pdf", [header, "Abstract", claim], metadata_title=metadata)
    request = {"pdf_path": str(path), "experiment_profile": PROFILE,
               "pdf_resolution": {"sha256": "stale", "profile": PROFILE,
                                  "source": "pdf_author_code_repository"}}
    with pytest.raises(pdf_input.PDFInputError, match="未共同确认"):
        pdf_input.resolve_pdf_request(request)


@pytest.mark.parametrize("mutation", [
    {"raw_url": "https://github.com/fork/rezero"}, {"page": None}, {"page": True},
    {"source": "model_metadata"}, {"context": "Only a package link."},
    {"is_author_code": False}, {"evidence_type": "reference"},
])
def test_route_rejects_inconsistent_or_guessed_evidence_records(mutation):
    evidence = extract_repository_links([CLAIM + AUTHOR])[0]
    evidence.update(mutation)
    assert routing.match_repository_profile(TITLE, [evidence]) is None


def test_route_returns_independent_evidence_and_accepts_case_insensitive_canonical_github_identity():
    evidence = extract_repository_links([CLAIM + AUTHOR.upper() + ".git"])[0]
    result = routing.match_repository_profile("REZERO IS ALL YOU NEED:  FAST CONVERGENCE AT LARGE DEPTH", [evidence])
    assert result["profile"] == PROFILE
    result["evidence"]["context"] = "modified copy"
    assert evidence["context"] != "modified copy"
    result["repository_relationship"]["hops"][0]["quote"] = "modified copy"
    again = routing.match_repository_profile(TITLE, [evidence])
    assert again["repository_relationship"]["hops"][0]["quote"] != "modified copy"


@pytest.mark.parametrize("api_version", [None, 1])
def test_stale_extractor_is_upgraded_without_mutating_active_reader_globals(monkeypatch, api_version):
    legacy = ModuleType("src.repository_evidence")
    legacy.REPOSITORY_EVIDENCE_API_VERSION = api_version
    legacy.owner = {"running": True}
    exec("def extract_repository_links(*args):\n    return owner\n", vars(legacy))
    old_extract = legacy.extract_repository_links
    monkeypatch.setitem(sys.modules, "src.repository_evidence", legacy)
    current = routing.current_repository_evidence()
    assert current is not legacy and current.REPOSITORY_EVIDENCE_API_VERSION == 1
    assert sys.modules["src.repository_evidence"] is legacy
    assert old_extract.__globals__ is vars(legacy) and old_extract() is legacy.owner
    assert current.extract_repository_links([CLAIM + AUTHOR])[0]["is_author_code"]


def test_pdf_route_selects_registered_full_experiment_and_author_training_revision(tmp_path):
    from src.repository_adapters import get_adapter
    from src.repository_profiles import get_profile

    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    resolved = pdf_input.resolve_pdf_request({"pdf_path": str(path)})
    profile = get_profile(resolved["experiment_profile"])
    assert profile["adapter_id"] == "rezero"
    assert profile["repository"] == {"url": TRAINING, "revision": routing.REZERO_TRAINING_REVISION}
    assert profile["paper"]["title"] == TITLE
    assert get_adapter(profile) is not None


def test_direct_orchestrator_pdf_route_bypasses_single_file_generation_and_search(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    llm = LLMClient(mock_mode=False)
    llm.chat = Mock(side_effect=AssertionError("Grounded routing must not call a model"))
    logger = Mock()
    logger.get_summary.return_value = []
    logger.get_stats.return_value = {}
    orchestrator = Orchestrator(llm_client=llm, mock_mode=False, logger=logger,
                                resource_manager=Mock(data_root=tmp_path))
    for agent in orchestrator.agents.values():
        agent.run = Mock(side_effect=AssertionError("Do not generate or search after reviewed PDF routing"))
    reviewed = Mock()
    reviewed.run.return_value = {"state": "COMPLETED", "error": None, "data": {}}
    monkeypatch.setattr(repository, "RepositoryReproduction", Mock(return_value=reviewed))
    result = orchestrator.run({"pdf_path": str(path), "use_llm_review": False})
    request = reviewed.run.call_args.args[0]
    assert request["experiment_profile"] == PROFILE and request["use_llm_review"] is False
    assert result["data"]["pdf_resolution"] == request["pdf_resolution"]
    assert request["pdf_resolution"]["evidence"]["url"] == AUTHOR
    assert request["pdf_resolution"]["training_repository_url"] == TRAINING
    llm.chat.assert_not_called()
    for agent in orchestrator.agents.values():
        agent.run.assert_not_called()
