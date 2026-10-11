"""Official source review and project adaptation use distinct evidence domains."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src import method_advice
from src.method_adapters import digest
from src.method_profiles import method_profile
from src.method_reproduction import method_review_project_sources


LLM = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test-model")
AUTHOR = {"source_id": "author_ode_demo", "origin": "official_repository",
          "url": "https://example.org/pinned/examples/ode_demo.py", "locator": "examples/ode_demo.py",
          "content": "optimizer = optim.RMSprop(func.parameters(), lr=1e-3)"}
PROJECT = {"source_id": "project_independent_evaluator", "origin": "project_adapter",
           "url": "project://evaluate.py", "locator": "evaluate.py",
           "content": "metrics = {'mae': measured_mae, 'rmse': measured_rmse}"}


def response(context):
    evidence = [{"source_id": AUTHOR["source_id"], "quote": AUTHOR["content"]},
                {"source_id": AUTHOR["source_id"], "quote": "optim.RMSprop"}]
    if context["review_responsibility"] == "project_adaptation_readiness":
        evidence.append({"source_id": PROJECT["source_id"], "quote": PROJECT["content"]})
    return {"status": "accepted", "summary": "Official defaults and project adaptations have distinct origins.",
            "evidence": evidence}


@pytest.mark.parametrize("profile_id", ["neural_ode_spiral", "siren_camera_quick"])
def test_author_roles_do_not_receive_project_parameters_or_independent_metrics(monkeypatch, profile_id):
    profile, contexts, prompts = method_profile(profile_id), [], []
    original = deepcopy(profile)

    def request(llm, prompt, timeout):
        context = json.loads(prompt.split("\n", 1)[1])
        contexts.append(context)
        prompts.append(prompt)
        return {"response": json.dumps(response(context))}

    monkeypatch.setattr(method_advice, "request_text", request)
    analysis = method_advice.review_sources(LLM, profile, [AUTHOR], lambda *args: None,
                                          project_sources=[PROJECT])
    assert analysis["status"] == "accepted" and profile == original
    for context in contexts[:2]:
        assert context["paper"] == {key: profile["paper"][key] for key in ("title", "url")}
        assert "required_metrics" not in context["paper"]
        assert "experiment_contract" not in context and "project_adaptations" not in context
        assert context["sources"] == [AUTHOR]
        assert context["source_mapping_scope"]["paper_access"] == "bibliography_only"
        assert context["experiment_scope"]["paper_full_text_provided"] is False
    for context in contexts[2:]:
        assert context["experiment_contract"]["parameters"] == profile["parameters"]
        assert context["experiment_contract"]["required_metrics"] == profile["paper"]["required_metrics"]
        assert context["project_adaptations"]["author_paper_claim"] is False
        assert context["source_origins"][PROJECT["source_id"]] == "project_adapter"
        assert context["source_origins"]["project_frozen_contract"] == "project_configuration"
        assert context["experiment_scope"]["execution_completed"] is False
    assert "用户PDF正文" not in "".join(prompts)
    assert analysis["input_scope"]["user_pdf_shared"] is False


@pytest.mark.parametrize("role", ["reader", "finder"])
def test_author_only_role_cannot_cite_project_configuration(monkeypatch, role):
    calls = []

    def request(llm, prompt, timeout):
        context = json.loads(prompt.split("\n", 1)[1])
        calls.append(context)
        parsed = response(context)
        current_role = ("reader", "finder", "builder", "verifier")[len(context["previous_reviews"])]
        if current_role == role:
            parsed["evidence"] = [{"source_id": "project_frozen_contract", "quote": '"seed": 2021'}]
        return {"response": json.dumps(parsed)}

    monkeypatch.setattr(method_advice, "request_text", request)
    with pytest.raises(method_advice.SourceReviewError, match="不符合公开原文") as caught:
        method_advice.review_sources(LLM, method_profile("neural_ode_spiral"), [AUTHOR], lambda *args: None,
                                     project_sources=[PROJECT])
    assert caught.value.analysis["failed_role"] == role


@pytest.mark.parametrize("role", ["builder", "verifier"])
@pytest.mark.parametrize("missing", ["official", "project"])
def test_project_review_requires_both_evidence_domains(monkeypatch, role, missing):
    calls = []

    def request(llm, prompt, timeout):
        context = json.loads(prompt.split("\n", 1)[1])
        calls.append(context)
        parsed = response(context)
        current_role = ("reader", "finder", "builder", "verifier")[len(context["previous_reviews"])]
        if current_role == role:
            keep = PROJECT["source_id"] if missing == "official" else AUTHOR["source_id"]
            parsed["evidence"] = [item for item in parsed["evidence"] if item["source_id"] == keep]
            if len(parsed["evidence"]) == 1:
                parsed["evidence"].append(deepcopy(parsed["evidence"][0]))
        return {"response": json.dumps(parsed)}

    monkeypatch.setattr(method_advice, "request_text", request)
    with pytest.raises(method_advice.SourceReviewError, match="分别引用官方来源与项目协议依据") as caught:
        method_advice.review_sources(LLM, method_profile("neural_ode_spiral"), [AUTHOR], lambda *args: None,
                                     project_sources=[PROJECT])
    assert caught.value.analysis["failed_role"] == role


def test_project_source_quote_is_still_checked_verbatim(monkeypatch):
    calls = []

    def request(llm, prompt, timeout):
        context = json.loads(prompt.split("\n", 1)[1])
        calls.append(context)
        parsed = response(context)
        if len(context["previous_reviews"]) == 2:
            for item in parsed["evidence"]:
                if item["source_id"] == PROJECT["source_id"]:
                    item["quote"] = "independent verification passed"
        return {"response": json.dumps(parsed)}

    monkeypatch.setattr(method_advice, "request_text", request)
    with pytest.raises(method_advice.SourceReviewError, match="不符合公开原文"):
        method_advice.review_sources(LLM, method_profile("neural_ode_spiral"), [AUTHOR], lambda *args: None,
                                     project_sources=[PROJECT])


def test_project_packet_reads_only_manifest_verified_adapter_files_not_private_pdf(tmp_path):
    files = {"run_experiment.py": "train_with_seed(cfg['seed'])\n",
             "evaluate.py": "measure_independent_mae_rmse()\n", "experiment.json": '{"seed":2021}'}
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    private = tmp_path / "user_upload.pdf"
    private.write_bytes(b"private PDF content must never be part of API review")
    manifest = {"files": {name: digest(tmp_path / name) for name in [*files, private.name]}}
    sources = method_review_project_sources(tmp_path, manifest)
    assert {source["locator"] for source in sources} == set(files)
    assert all(source["origin"] == "project_adapter" for source in sources)
    assert all(source["content"] == files[source["locator"]] for source in sources)
    assert "private PDF" not in json.dumps(sources)


@pytest.mark.parametrize("name", ["run_experiment.py", "evaluate.py", "experiment.json"])
def test_changed_adapter_evidence_is_rejected_before_online_review(tmp_path, name):
    path = tmp_path / name
    path.write_text("original implementation", encoding="utf-8")
    manifest = {"files": {name: digest(path)}}
    path.write_text("changed implementation", encoding="utf-8")
    with pytest.raises(ValueError, match="适配证据校验失败"):
        method_review_project_sources(tmp_path, manifest)


def test_verified_preflight_exposes_versions_without_local_paths_or_private_fields(tmp_path):
    imported = {"torch": "2.5.1+cpu", "cuda": None,
                "modules": {"torch": "C:/private/cache/torch/__init__.py"},
                "author_package": "C:/private/workspace/torchdiffeq/__init__.py", "api_key": "never-share"}
    (tmp_path / "import_provenance.json").write_text(json.dumps(imported), encoding="utf-8")
    assert method_review_project_sources(tmp_path, {}) == []
    sources = method_review_project_sources(tmp_path, {}, environment_verified=True)
    evidence = json.loads(sources[0]["content"])
    assert evidence["torch"] == "2.5.1+cpu" and evidence["module_names"] == ["torch"]
    assert evidence["frozen_import_paths_verified"] and not evidence["training_started"]
    assert evidence["author_package_matches_pinned_workspace"]
    assert "private" not in json.dumps(sources) and "never-share" not in json.dumps(sources)
