"""Advice must describe its actual measurements without observing confirmation data."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import pytest

from src.method_advice import suggest
from src.method_optimization import Study, validate_candidates
from src.method_profiles import method_profile
from src.repository_reproduction import spec_digest


SOURCES = [{"source_id": "author", "content": "optimizer learning rate",
            "url": "https://example.org/pinned-author", "locator": "training.py:L10"}]
LLM = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test")


def measurement(profile, split="validation"):
    actual = {"mse": .001, "psnr": 30.} if profile["adapter_id"] == "siren" else {"mae": .34, "rmse": .42}
    return {"metrics_comparison": {"actual": actual},
            "metric_records": [{"name": name, "value": value, "split": split,
                                "spec_sha256": spec_digest(profile)} for name, value in actual.items()],
            "training_summary": {"initial_loss": .2, "final_loss": .004, "steps_completed": profile["parameters"]["steps"],
                                 "holdout_metrics": {"secret_result": 987654.321}},
            "holdout_metrics": {"secret_result": 987654.321},
            "protocol_pass": True, "independent_metrics_pass": True}


@pytest.fixture
def captured_advice(monkeypatch):
    calls = []
    def request(llm, prompt, timeout_s):
        context = json.loads(prompt.split("\n", 1)[1])
        calls.append(context)
        candidate = {"parameter": "learning_rate", "value": context["search_space"]["learning_rate"][0],
                     "hypothesis": "改动学习率可能降低固定验证误差", "expected_effect": "待验证",
                     "cost": "可能收敛较慢", "validation_plan": "遵循冻结验证划分",
                     "evidence": [{"source_id": "author", "quote": "optimizer learning rate"}]}
        return {"response": json.dumps({"suggestions": [candidate]}, ensure_ascii=False), "calls": 1}
    monkeypatch.setattr("src.method_advice.request_text", request)
    return calls


@pytest.mark.parametrize("profile_id,protocol", [
    ("siren_camera_quick", "pixel_holdout"), ("neural_ode_spiral", "initial_condition_holdout"),
])
def test_prompt_uses_actual_baseline_and_whitelisted_frozen_protocol(tmp_path, captured_advice, profile_id, protocol):
    study = Study(SimpleNamespace(), method_profile(profile_id), tmp_path, time.monotonic(), 7200, lambda e: None)
    original_contract = deepcopy(study.contract)
    result = suggest(LLM, study.profile, measurement(study.profile), SOURCES,
                     study_contract=study.contract,
                     baseline_provenance={"baseline_run_id": "parent-run", "trial_label": "baseline_2021",
                                          "spec_sha256": spec_digest(study.profile), "study_sha256": study.contract_hash})
    assert result["status"] == "suggested"
    payload, = captured_advice
    assert payload["parameters"] == study.profile["parameters"]
    assert payload["parameters"]["protocol"] == protocol
    assert payload["measured_summary"]["split"] == "validation"
    assert {record["split"] for record in payload["measured_summary"]["metric_records"]} == {"validation"}
    assert payload["baseline_provenance"] == {"baseline_run_id": "parent-run", "trial_label": "baseline_2021",
                                               "spec_sha256": spec_digest(study.profile), "study_sha256": study.contract_hash}
    definition = payload["evaluation_protocol"]
    assert definition["scope"] == "optimization_extension"
    assert definition["selection_split"] == "validation" and definition["confirmation_split"] == "holdout"
    if profile_id.startswith("siren"):
        assert definition["pixel_fractions"] == {"train": .8, "validation": .1, "holdout": .1}
        assert definition["pixel_split_seed"] == study.contract["pixel_split_seed"]
        assert definition["training_loss"]["target_range"] == [-1, 1]
        assert definition["evaluation_scale"]["target_range"] == [0, 1]
        assert "data_range=1" in definition["evaluation_scale"]["psnr"]
    else:
        assert definition["validation_initial_states"] == [[1.5, 0.], [0., 1.5]]
        assert definition["training_loss"]["initial_state"] == [2., 0.]
        assert "新增初值验证" in definition["validation_definition"]
        assert "original ODE state coordinates" in definition["evaluation_scale"]["mae"]
    serialized = json.dumps(payload)
    assert "987654.321" not in serialized and "holdout_metrics" not in serialized
    assert "ode_holdout_initials" not in serialized
    assert study.contract == original_contract
    assert result["advice_history"][0]["advice_context"] == result["advice_context"]
    assert json.loads(result["advice_history"][0]["raw_response"])["suggestions"][0]["value"] == result["suggestions"][0]["value"]
    result["suggestions"][0]["status"] = "validated_gain"
    assert result["advice_history"][0]["suggestions"][0]["status"] == "untested"


@pytest.mark.parametrize("profile_id", ["siren_camera_quick", "neural_ode_spiral"])
def test_fit_advice_is_explicitly_distinct_from_validation(captured_advice, profile_id):
    profile = method_profile(profile_id)
    result = suggest(LLM, profile, measurement(profile, "fit"), SOURCES)
    assert result["status"] == "suggested"
    payload, = captured_advice
    assert payload["parameters"]["protocol"] == "official_fit"
    assert payload["measured_summary"]["split"] == "fit"
    assert payload["evaluation_protocol"]["scope"] == "official_method_experiment"
    assert "selection_split" not in payload["evaluation_protocol"]


@pytest.mark.parametrize("mismatch", ["holdout", "official_parameters", "contract", "missing_records", "baseline_hash"])
def test_mismatched_or_holdout_evidence_never_calls_llm(tmp_path, captured_advice, mismatch):
    study = Study(SimpleNamespace(), method_profile("siren_camera_quick"), tmp_path, time.monotonic(), 7200, lambda e: None)
    profile, validation = study.profile, measurement(study.profile)
    provenance = None
    if mismatch == "holdout":
        for record in validation["metric_records"]:
            record["split"] = "holdout"
    elif mismatch == "official_parameters":
        profile = method_profile("siren_camera_quick")
    elif mismatch == "contract":
        study.contract["selection_split"] = "fit"
    elif mismatch == "missing_records":
        validation.pop("metric_records")
    else:
        provenance = {"spec_sha256": "wrong-baseline"}
    result = suggest(LLM, profile, validation, SOURCES, study_contract=study.contract, baseline_provenance=provenance)
    assert result["status"] == "advice_unavailable"
    assert not captured_advice and not result["suggestions"]


@pytest.mark.parametrize("raw", ['{"suggestions": [{"parameter": "steps", "value": 2}]}',
                                 '["wrong-shape"]', '{"suggestions": [null]}'])
def test_invalid_original_response_is_preserved_without_becoming_advice(monkeypatch, raw):
    monkeypatch.setattr("src.method_advice.request_text", lambda *args: {"response": raw, "calls": 1})
    profile = method_profile("siren_camera_quick")
    result = suggest(LLM, profile, measurement(profile, "fit"), SOURCES)
    assert result["status"] == "advice_unavailable" and result["suggestions"] == []
    assert result["advice_history"][0]["raw_response"] == raw
    assert result["advice_history"][0]["status"] == "rejected"


def test_optimizer_wires_real_study_context_and_preserves_raw_advice(tmp_path, monkeypatch, captured_advice):
    def train(study, label, *, seed=2021, candidate=None, reserve_s=0):
        validation = measurement(study.profile)
        return {"label": label, "candidate": candidate, "status": "completed", "elapsed_s": .1,
                "metrics": validation["metrics_comparison"]["actual"], "validation": validation,
                "spec_sha256": spec_digest(study.profile)}
    monkeypatch.setattr(Study, "train", train)
    result = validate_candidates(SimpleNamespace(llm=LLM), method_profile("siren_camera_quick"),
                                 {"run_dir": str(tmp_path), "method_sources": SOURCES}, {}, time.monotonic(), lambda e: None)
    assert result["status"] == "tested_no_gain"
    payload, = captured_advice
    assert payload["parameters"]["protocol"] == "pixel_holdout"
    assert payload["baseline_provenance"]["trial_label"] == "baseline_2021"
    assert result["advice_history"][0]["suggestions"][0]["status"] == "untested"
    assert result["suggestions"][0]["status"] == "tested"
    saved = json.loads((tmp_path / "optimization.json").read_text(encoding="utf-8"))
    assert saved["advice_context"] == result["advice_context"]
    assert saved["advice_history"] == result["advice_history"]
