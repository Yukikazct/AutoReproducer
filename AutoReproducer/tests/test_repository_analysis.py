"""Grounded model analysis tests; no network, training or credentials required."""
from copy import deepcopy
import json

import pytest

from src.repository_analysis import (
    RepositoryAnalysis, RepositoryAnalysisError, STAGES, public_packet,
)
from src.repository_profiles import get_profile


@pytest.fixture
def profile():
    return get_profile("dlinear_etth1_reference")


@pytest.fixture
def packet(profile):
    sources = [
        ("paper_table2", "Table 2", "ETTh1 DLinear forecast horizon 96 MSE 0.375 MAE 0.399"),
        ("repo_author_command", "scripts/EXP-LongForecasting/Linear/etth1.sh",
         "python run_longExp.py --model DLinear --data ETTh1 --features M --seq_len 336 "
         "--pred_len 96 --batch_size 32 --learning_rate 0.005"),
        ("repo_entrypoint", "run_longExp.py", "fix_seed = 2021; train_epochs default=10; patience default=3"),
        ("repo_requirements", "requirements.txt", "numpy\nmatplotlib\npandas\nscikit-learn\ntorch==1.9.0"),
        ("repo_model", "models/DLinear.py", "class Model: DLinear"),
        ("repo_data_split", "data_provider/data_loader.py", "class Dataset_ETT_hour: ETTh1"),
        ("repo_training", "exp/exp_main.py", "from utils.metrics import metric; class Exp_Main: pass"),
        ("repo_readme", "README.md", "LTSF-Linear author repository DLinear"),
    ]
    return {"version": 1, "repository": deepcopy(profile["repository"]), "sources": [
        {"source_id": source_id, "url": "https://example.org/public/" + source_id,
         "locator": locator, "text": text} for source_id, locator, text in sources
    ]}


def quote(packet, source_id):
    source = next(source for source in packet["sources"] if source["source_id"] == source_id)
    return [{key: source[key] for key in ("source_id", "locator")} | {"quote": source["text"]}]


@pytest.fixture
def responses(profile, packet):
    protocol = {"method": "DLinear", "dataset": "ETTh1", **{
        key: profile["parameters"][key] for key in (
            "seq_len", "pred_len", "features", "batch_size", "learning_rate", "seed", "train_epochs", "patience",
        )}}
    reader = {"status": "accepted", "protocol": protocol,
              "reference_metrics": deepcopy(profile["paper"]["metrics"]), "evidence": {}}
    for key in protocol:
        source_id = "repo_entrypoint" if key in {"seed", "train_epochs", "patience"} else "repo_author_command"
        reader["evidence"]["protocol." + key] = quote(packet, source_id)
    for key in ("mse", "mae"):
        reader["evidence"]["reference_metrics." + key] = quote(packet, "paper_table2")
    source_map = {"model": "repo_model", "data_split": "repo_data_split", "training_and_test": "repo_training",
                  "metrics": "repo_training", "author_command": "repo_author_command"}
    finder = {"status": "accepted", "repository": deepcopy(profile["repository"]),
              "entrypoints": deepcopy(profile["repository_map"]), "evidence": {"repository": quote(packet, "repo_readme")}}
    finder["evidence"].update({"entrypoints." + key: quote(packet, source_id) for key, source_id in source_map.items()})
    builder = {"status": "accepted", "original_requirements": ["numpy", "matplotlib", "pandas", "scikit-learn", "torch==1.9.0"],
               "changes_algorithm": False,
               "compatibility_note": "Modern Python needs a separately verified compatibility environment; the author used torch 1.9.0.",
               "compatibility_proposals": [{"package": "torch", "suggested_constraint": ">=2.0", "reason": "Check modern Python wheel availability separately."}],
               "evidence": {"original_requirements": quote(packet, "repo_requirements")}}
    verifier = {"status": "accepted", "pass": True, "issues": [], "reviewed_stages": ["reader", "finder", "builder"],
                "checks": {"protocol_alignment": True, "source_mapping": True, "dependency_provenance": True},
                "evidence": {"checks.protocol_alignment": quote(packet, "paper_table2"),
                             "checks.source_mapping": quote(packet, "repo_author_command"),
                             "checks.dependency_provenance": quote(packet, "repo_requirements")}}
    return [reader, finder, builder, verifier]


class FakeClient:
    """Model replies and counters without any request transport."""

    def __init__(self, responses, *, existing_calls=23, multiplier=1):
        self.responses = deepcopy(responses)
        self.mock_mode = False
        self.model = "fixture-real-client"
        self.call_count = existing_calls
        self.requests = []
        self.last_usage = {}
        self.multiplier = multiplier

    def get_call_count(self):
        return self.call_count

    def chat(self, prompt, **kwargs):
        self.requests.append({"prompt": prompt, **kwargs})
        self.call_count += self.multiplier
        self.last_usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
        value = self.responses[len(self.requests) - 1]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


def test_four_distinct_grounded_reviews_open_readiness_gate(profile, packet, responses):
    client = FakeClient(responses)
    events = []
    frozen_profile = deepcopy(profile)
    result = RepositoryAnalysis(client).run(packet, profile, lambda name, payload: events.append((name, payload)))
    assert result["status"] == "accepted"
    assert result["gate"] == {"pass": True, "scope": "readiness_for_selected_experiment", "determines_reproduction": False}
    assert list(result["analyses"]) == list(STAGES)
    assert result["calls"] == 4 and client.call_count == 27
    assert all(stage["attempted"] and stage["completed"] and stage["accepted"] for stage in result["stages"])
    assert [stage["calls"] for stage in result["stages"]] == [1, 1, 1, 1]
    assert all(stage["usage"]["total_tokens"] == 20 for stage in result["stages"])
    assert [request["task"] for request in client.requests] == ["repository_" + name for name in STAGES]
    assert [event[1]["status"] for event in events] == ["started", "accepted"] * 4
    assert profile == frozen_profile
    assert result["analyses"]["builder"]["proposals_executed"] is False
    assert "previous_analyses" in client.requests[-1]["prompt"]
    assert '"reader"' in client.requests[-1]["prompt"]


def test_client_counter_deltas_include_transport_retry_attempts(profile, packet, responses):
    result = RepositoryAnalysis(FakeClient(responses, multiplier=2)).run(packet, profile)
    assert result["calls"] == 8
    assert [stage["calls"] for stage in result["stages"]] == [2, 2, 2, 2]


@pytest.mark.parametrize("field,value", [
    ("seq_len", 96), ("pred_len", 192), ("features", "S"),
    ("seed", 7), ("train_epochs", 1), ("patience", 1),
    ("learning_rate", 0.001), ("dataset", "ETTh2"), ("batch_size", True),
])
def test_conflicting_protocol_stops_before_more_calls(profile, packet, responses, field, value):
    responses[0]["protocol"][field] = value
    client = FakeClient(responses)
    with pytest.raises(RepositoryAnalysisError, match="conflicts") as rejected:
        RepositoryAnalysis(client).run(packet, profile)
    assert rejected.value.result["calls"] == 1
    assert rejected.value.result["gate"]["pass"] is False
    assert rejected.value.result["gate"]["determines_reproduction"] is False
    assert rejected.value.result["analyses"] == {}
    stage = rejected.value.result["stages"][0]
    assert {key: stage[key] for key in ("name", "attempted", "completed", "accepted", "calls", "usage")} == {
        "name": "reader", "attempted": True, "completed": True, "accepted": False,
        "calls": 1, "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    assert stage["rejection_diagnostics"]["status"] == "accepted"


@pytest.mark.parametrize("metric,value", [("mse", 0.338), ("mae", 0.388), ("mse", float("nan"))])
def test_other_table_rows_or_nonfinite_reference_metrics_are_rejected(profile, packet, responses, metric, value):
    responses[0]["reference_metrics"][metric] = value
    with pytest.raises(RepositoryAnalysisError, match="metric conflicts"):
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)


@pytest.mark.parametrize("edit,match", [
    ("missing", "missing quoted evidence"), ("unknown_id", "unknown public source"),
    ("wrong_locator", "unknown public source"), ("fabricated_quote", "does not exist"),
    ("irrelevant_quote", "does not contain claimed value"),
])
def test_quotes_are_real_and_identify_the_claimed_value(profile, packet, responses, edit, match):
    field = "protocol.seq_len"
    ref = responses[0]["evidence"][field][0]
    if edit == "missing":
        del responses[0]["evidence"][field]
    elif edit == "unknown_id":
        ref["source_id"] = "invented_source"
    elif edit == "wrong_locator":
        ref["locator"] = "other section"
    elif edit == "fabricated_quote":
        ref["quote"] = "The author used seq_len 336."  # absent from authentic fixture
    else:
        ref["quote"] = "--batch_size 32"
    with pytest.raises(RepositoryAnalysisError, match=match):
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)


def test_whitespace_normalized_quotes_remain_verbatim_evidence(profile, packet, responses):
    responses[0]["evidence"]["protocol.seq_len"][0]["quote"] = "--seq_len\n  336"
    assert RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["gate"]["pass"] is True


@pytest.mark.parametrize("edit,match", [
    ("revision", "repository differs"), ("path", "entrypoints conflict"),
    ("unrelated_citation", "does not identify entrypoint"),
])
def test_resource_analysis_cannot_replace_author_source(profile, packet, responses, edit, match):
    if edit == "revision":
        responses[1]["repository"]["revision"] = "a" * 40
    elif edit == "path":
        responses[1]["entrypoints"]["model"] = "generated_model.py"
    else:
        responses[1]["evidence"]["entrypoints.model"] = quote(packet, "repo_readme")
    with pytest.raises(RepositoryAnalysisError, match=match) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert rejected.value.result["calls"] == 2
    assert list(rejected.value.result["analyses"]) == ["reader"]


@pytest.mark.parametrize("edit,match", [
    ("modern_pin_as_original", "original requirements differ"),
    ("missing_original", "original requirements differ"),
    ("change_algorithm", "preserve the frozen algorithm"),
    ("new_package", "unsupported package"),
    ("wrong_source", "must cite repo_requirements"),
])
def test_environment_facts_stay_separate_from_proposals(profile, packet, responses, edit, match):
    if edit == "modern_pin_as_original":
        responses[2]["original_requirements"][-1] = "torch==2.5.1"
    elif edit == "missing_original":
        responses[2]["original_requirements"].pop()
    elif edit == "change_algorithm":
        responses[2]["changes_algorithm"] = True
    elif edit == "new_package":
        responses[2]["compatibility_proposals"][0]["package"] = "tensorflow"
    else:
        responses[2]["evidence"]["original_requirements"] = quote(packet, "repo_readme")
    with pytest.raises(RepositoryAnalysisError, match=match) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert rejected.value.result["calls"] == 3
    assert list(rejected.value.result["analyses"]) == ["reader", "finder"]


@pytest.mark.parametrize("edit", ["rejected", "issues", "missing_stage", "failed_check", "missing_evidence"])
def test_verifier_is_a_real_failure_gate(profile, packet, responses, edit):
    if edit == "rejected":
        responses[3]["pass"] = False
    elif edit == "issues":
        responses[3]["issues"] = ["The table row was not supported."]
    elif edit == "missing_stage":
        responses[3]["reviewed_stages"].pop()
    elif edit == "failed_check":
        responses[3]["checks"]["source_mapping"] = False
    else:
        del responses[3]["evidence"]["checks.protocol_alignment"]
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    partial = rejected.value.partial_result
    assert partial["calls"] == 4 and partial["gate"]["pass"] is False
    assert list(partial["analyses"]) == ["reader", "finder", "builder"]
    assert partial["stages"][-1]["completed"] is True
    assert partial["stages"][-1]["accepted"] is False


def test_packet_and_model_extra_fields_cannot_disclose_local_information(profile, packet, responses):
    secret = "sk-test-secret-fixture-12345678"
    packet.update({"api_key": secret, "execution": {"stdout": "PRIVATE_RESULTS_123"},
                   "local_path": "/Users/secret/private-model", "dataset_bytes": "PRIVATE_DATA_123"})
    packet["sources"][0]["local_path"] = "/Users/secret/cache/paper.html"
    responses[0]["execution"] = {"stdout": "PRIVATE_RESPONSE_123"}
    client = FakeClient(responses)
    result = RepositoryAnalysis(client).run(packet, profile)
    serialized = json.dumps(client.requests) + json.dumps(result)
    for private in (secret, "PRIVATE_RESULTS_123", "PRIVATE_DATA_123", "PRIVATE_RESPONSE_123", "/Users/secret"):
        assert private not in serialized
    assert "requirements_txt" not in client.requests[2]["prompt"]
    assert "torch==2.5.1" not in client.requests[2]["prompt"]


@pytest.mark.parametrize("edit", ["secret_in_text", "private_path_in_text", "bad_sha", "private_url", "empty_text", "duplicate"])
def test_bad_source_packet_is_rejected_before_api(profile, packet, responses, edit):
    if edit == "secret_in_text":
        packet["sources"][0]["text"] += " sk-test-secret-fixture-12345678"
    elif edit == "private_path_in_text":
        packet["sources"][0]["text"] += " /Users/local/private-dataset"
    elif edit == "bad_sha":
        packet["repository"]["revision"] = "main"
    elif edit == "private_url":
        packet["sources"][0]["url"] = "file:///tmp/cache.html"
    elif edit == "empty_text":
        packet["sources"][0]["text"] = ""
    else:
        packet["sources"].append(deepcopy(packet["sources"][0]))
    client = FakeClient(responses)
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client).run(packet, profile)
    assert rejected.value.result["calls"] == 0 and client.requests == []


def test_mock_client_cannot_masquerade_as_real_analysis(profile, packet, responses):
    client = FakeClient(responses)
    client.mock_mode = True
    with pytest.raises(RepositoryAnalysisError, match="real API client") as rejected:
        RepositoryAnalysis(client).run(packet, profile)
    assert rejected.value.result["calls"] == 0


def test_response_exception_is_counted_without_retaining_secrets(profile, packet, responses):
    responses[1] = RuntimeError("Transport echoed sk-test-secret-fixture-12345678")
    client = FakeClient(responses)
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client).run(packet, profile)
    assert "sk-test" not in str(rejected.value)
    assert "sk-test" not in json.dumps(rejected.value.result)
    stage = rejected.value.result["stages"][-1]
    assert stage["attempted"] and not stage["completed"] and not stage["accepted"]
    assert stage["calls"] == 1 and stage["usage"] == {}
    assert rejected.value.result["calls"] == 2


@pytest.mark.parametrize("response", ["[LLM API Error: timeout]", "not JSON", "[]", "{}", '{"status":"insufficient_evidence"}'])
def test_api_or_schema_error_preserves_failure_gate(profile, packet, responses, response):
    responses[0] = response
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert rejected.value.result["calls"] == 1
    assert rejected.value.result["gate"]["pass"] is False
    if response.startswith("[LLM API Error:"):
        assert rejected.value.result["stages"][0]["completed"] is False
        assert rejected.value.result["stages"][0]["usage"] == {}


def test_failed_api_does_not_reuse_previous_stage_usage(profile, packet, responses):
    responses[1] = "[LLM API Error: timeout]"
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert rejected.value.result["stages"][0]["usage"]["total_tokens"] == 20
    assert rejected.value.result["stages"][1]["usage"] == {}
    assert rejected.value.result["calls"] == 2


def test_json_fence_is_supported_but_prose_is_not(profile, packet, responses):
    responses[0] = "```json\n" + json.dumps(responses[0]) + "\n```"
    assert RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["status"] == "accepted"


def test_author_requirements_allow_supplementary_public_context(profile, packet, responses):
    responses[2]["evidence"]["original_requirements"].extend(quote(packet, "repo_readme"))
    assert RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["status"] == "accepted"


def test_public_python_newlines_are_not_mistaken_for_windows_paths(packet):
    packet["sources"][0]["text"] += "\nclass Model:\n    if True:\n        pass\nelse:\n    pass\n"
    assert public_packet(packet)["sources"][0]["text"].endswith("    pass\n")


def test_public_packet_whitelist_leaves_only_public_source_material(packet):
    packet["foo"] = "private"
    sanitized = public_packet(packet)
    assert set(sanitized) == {"version", "repository", "sources"}
    assert all(set(source) == {"source_id", "url", "locator", "text"} for source in sanitized["sources"])


@pytest.fixture
def result_summary():
    return {"metrics": {"mse": 0.3841443955898285, "mae": 0.40471312403678894},
            "epochs_completed": 7, "protocol_pass": True, "independent_metrics_pass": True}


@pytest.fixture
def result_response(result_summary, profile, packet):
    differences = []
    for metric in ("mse", "mae"):
        actual = result_summary["metrics"][metric]
        reference = profile["paper"]["metrics"][metric]
        delta = abs(actual - reference)
        differences.append({"metric": metric, "paper_value": reference, "actual_value": actual,
                            "absolute_difference": delta, "relative_difference": delta / reference,
                            "explanation": "The submitted metric is slightly higher than the published reference."})
    return {"status": "accepted", "summary": "The selected experiment has small numerical differences from the Table 2 references.",
            "differences": differences,
            "limitations": ["This comparison concerns only the selected ETTh1 forecast horizon."],
            "evidence": {key: quote(packet, "paper_table2") for key in (
                "summary", "differences.mse", "differences.mae", "limitations.0",
            )}}


def test_authorized_result_summary_is_one_additional_call(profile, packet, result_summary, result_response):
    client = FakeClient([result_response], existing_calls=4)
    events = []
    result = RepositoryAnalysis(client).review_result_summary(
        result_summary, packet, profile, lambda name, payload: events.append((name, payload)))
    assert result["status"] == "accepted" and result["calls"] == 1
    assert client.call_count == 5
    assert result["stages"][0]["name"] == "result_validator"
    assert result["stages"][0]["accepted"] is True
    assert result["gate"] == {"pass": True, "scope": "explanation_only", "determines_reproduction": False}
    assert result["determines_reproduction"] is False
    assert result["analyses"]["result_validator"]["summary"] == result["summary"]
    assert events[0][0] == events[1][0] == "result_validator"
    assert client.requests[0]["task"] == "repository_result_validator"
    assert result["differences"][0]["relative_difference"] == abs(0.3841443955898285 - 0.375) / 0.375


def test_result_summary_whitelist_excludes_logs_data_paths_and_other_metrics(profile, packet, result_summary, result_response):
    result_summary.update({"logs": "PRIVATE_LOG_123", "paths": "/Users/private/results.npy", "api_key": "sk-test-secret-fixture-12345678"})
    result_summary["metrics"]["accuracy"] = "PRIVATE_ACCURACY_123"
    client = FakeClient([result_response])
    RepositoryAnalysis(client).review_result_summary(result_summary, packet, profile)
    prompt = client.requests[0]["prompt"]
    for private in ("PRIVATE_LOG_123", "/Users/private", "sk-test-secret", "PRIVATE_ACCURACY_123",
                    '"repo_training"', '"repo_author_command"', '"repo_requirements"'):
        assert private not in prompt
    assert '"submitted_summary"' in prompt and '"epochs_completed": 7' in prompt
    assert '"computed_differences"' in prompt and '"response_schema"' in prompt
    assert '"protocol_pass": true' in prompt


@pytest.mark.parametrize("edit,match", [
    ("metric_value", "changed the supplied"), ("derived_difference", "changed the supplied"),
    ("other_metric", "changed the metric set"), ("missing_limitations", "state its limitations"),
    ("unknown_source", "unknown public source"), ("fabricated_quote", "does not exist"),
    ("unrelated_quote", "lacks the paper reference"),
])
def test_result_review_cannot_change_metrics_or_invent_evidence(profile, packet, result_summary, result_response, edit, match):
    if edit == "metric_value":
        result_response["differences"][0]["actual_value"] = 0.375
    elif edit == "derived_difference":
        result_response["differences"][0]["relative_difference"] = 0
    elif edit == "other_metric":
        result_response["differences"][0]["metric"] = "accuracy"
    elif edit == "missing_limitations":
        result_response["limitations"] = []
    elif edit == "unknown_source":
        result_response["evidence"]["summary"] = quote(packet, "repo_training")
    elif edit == "fabricated_quote":
        result_response["evidence"]["differences.mse"][0]["quote"] = "The actual run has MSE 0.3841444."
    else:
        result_response["evidence"]["differences.mse"][0]["quote"] = "ETTh1"
    with pytest.raises(RepositoryAnalysisError, match=match) as rejected:
        RepositoryAnalysis(FakeClient([result_response])).review_result_summary(result_summary, packet, profile)
    assert rejected.value.result["calls"] == 1
    assert rejected.value.result["gate"]["determines_reproduction"] is False
    assert rejected.value.result["stages"][0]["accepted"] is False


@pytest.mark.parametrize("edit", ["nonfinite", "negative", "boolean_metric", "boolean_epochs", "string_status", "too_many_epochs"])
def test_invalid_authorized_summary_never_reaches_api(profile, packet, result_summary, result_response, edit):
    if edit == "nonfinite":
        result_summary["metrics"]["mse"] = float("inf")
    elif edit == "negative":
        result_summary["metrics"]["mae"] = -0.1
    elif edit == "boolean_metric":
        result_summary["metrics"]["mse"] = True
    elif edit == "boolean_epochs":
        result_summary["epochs_completed"] = True
    elif edit == "string_status":
        result_summary["protocol_pass"] = "passed"
    else:
        result_summary["epochs_completed"] = 11
    client = FakeClient([result_response])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client).review_result_summary(result_summary, packet, profile)
    assert rejected.value.result["calls"] == 0 and client.requests == []


def test_result_review_api_error_preserves_local_verification_status(profile, packet, result_summary):
    original = deepcopy(result_summary)
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(["[LLM API Error: timeout]"])).review_result_summary(result_summary, packet, profile)
    assert result_summary == original
    assert rejected.value.result["calls"] == 1
    assert rejected.value.result["gate"]["determines_reproduction"] is False
    assert rejected.value.result["stages"][0]["completed"] is False
    assert "training" not in str(rejected.value).lower()


def test_repair_budget_does_not_add_calls_when_all_four_reviews_pass(profile, packet, responses):
    result = RepositoryAnalysis(FakeClient(responses), max_repairs=1).run(packet, profile)
    assert result["calls"] == 4
    assert len(result["stages"]) == 4
    assert [stage["attempt"] for stage in result["stages"]] == [1, 1, 1, 1]


def test_finder_schema_repair_reuses_sources_and_prior_accepted_analysis(profile, packet, responses):
    rejected_finder = deepcopy(responses[1])
    rejected_finder["entrypoints"]["training_and_test"] = "run_longExp.py"
    rejected_finder["raw_response_secret"] = "DO_NOT_RETAIN_OR_SEND_RAW_123"
    client = FakeClient([responses[0], rejected_finder, *responses[1:]])
    events = []
    result = RepositoryAnalysis(client, max_repairs=1).run(
        packet, profile, lambda name, payload: events.append((name, payload)))
    assert result["status"] == "accepted" and result["calls"] == 5
    assert [stage["name"] for stage in result["stages"]] == ["reader", "finder", "finder", "builder", "verifier"]
    first, second = result["stages"][1:3]
    assert first["attempt"] == 1 and first["completed"] and not first["accepted"]
    assert second["attempt"] == 2 and second["completed"] and second["accepted"]
    assert first["calls"] == second["calls"] == 1
    assert first["candidate_entrypoints"]["training_and_test"] == "run_longExp.py"
    assert "training_and_test" in first["reason"]
    assert [payload["status"] for name, payload in events if name == "finder"] == ["started", "failed", "started", "accepted"]
    repair_prompt = client.requests[2]["prompt"]
    assert '"correction_request"' in repair_prompt
    assert '"reader"' in repair_prompt and '"public_sources"' in repair_prompt
    assert "Exp_Main.train and Exp_Main.test" in repair_prompt
    assert "not the run_longExp.py CLI launcher" in repair_prompt
    assert "DO_NOT_RETAIN_OR_SEND_RAW_123" not in repair_prompt
    assert "DO_NOT_RETAIN_OR_SEND_RAW_123" not in json.dumps(result)


def test_finder_diagnostics_retain_only_known_public_relative_candidates(profile, packet, responses):
    responses[1]["entrypoints"].update({"training_and_test": "run_longExp.py", "model": "/Users/private/checkpoint.py",
                                       "metrics": "invented_generated_metrics.py"})
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    stage = rejected.value.result["stages"][-1]
    assert stage["candidate_entrypoints"]["training_and_test"] == "run_longExp.py"
    assert "model" not in stage["candidate_entrypoints"] and "metrics" not in stage["candidate_entrypoints"]
    assert "model, training_and_test, metrics" in str(rejected.value)
    assert "/Users/private" not in json.dumps(rejected.value.result)
    assert "invented_generated_metrics.py" not in json.dumps(rejected.value.result)


def test_second_bad_finder_response_keeps_readiness_gate_closed(profile, packet, responses):
    bad_finder = deepcopy(responses[1])
    bad_finder["entrypoints"]["training_and_test"] = "run_longExp.py"
    client = FakeClient([responses[0], bad_finder, bad_finder, *responses[2:]])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client, max_repairs=1).run(packet, profile)
    assert rejected.value.result["calls"] == 3
    assert rejected.value.result["gate"]["pass"] is False
    assert list(rejected.value.result["analyses"]) == ["reader"]
    assert [stage["accepted"] for stage in rejected.value.result["stages"]] == [True, False, False]
    assert client.requests[-1]["task"] == "repository_finder"


@pytest.mark.parametrize("edit", ["invalid_json", "missing_schema", "fabricated_quote", "wrong_protocol"])
def test_reader_can_correct_one_schema_or_evidence_error(profile, packet, responses, edit):
    first_reader = deepcopy(responses[0])
    if edit == "invalid_json":
        first_reader = "not JSON"
    elif edit == "missing_schema":
        first_reader = {"protocol": {}}
    elif edit == "fabricated_quote":
        first_reader["evidence"]["protocol.seq_len"][0]["quote"] = "Fabricated input window 336"
    else:
        first_reader["protocol"]["seq_len"] = 96
    result = RepositoryAnalysis(FakeClient([first_reader, *responses]), max_repairs=1).run(packet, profile)
    assert result["calls"] == 5 and result["gate"]["pass"] is True
    assert result["stages"][0]["accepted"] is False and result["stages"][1]["attempt"] == 2
    assert result["analyses"]["reader"]["protocol"]["seq_len"] == 336


@pytest.mark.parametrize("response", [RuntimeError("transport failed"), "[LLM API Error: HTTP 429]",
                                     '{"status":"insufficient_evidence"}'])
def test_repair_never_retries_transport_or_explicit_evidence_refusal(profile, packet, responses, response):
    client = FakeClient([response, *responses])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client, max_repairs=1).run(packet, profile)
    assert rejected.value.result["calls"] == 1
    assert len(rejected.value.result["stages"]) == len(client.requests) == 1
    assert rejected.value.result["gate"]["pass"] is False


def test_verifier_evidence_refusal_is_not_coaxed_into_a_pass(profile, packet, responses):
    bad_verifier = deepcopy(responses[3])
    bad_verifier["pass"] = False
    bad_verifier["issues"] = ["The selected table evidence was not sufficient."]
    client = FakeClient([*responses[:3], bad_verifier, responses[3]])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client, max_repairs=1).run(packet, profile)
    assert rejected.value.result["calls"] == 4 and len(client.requests) == 4
    assert rejected.value.result["gate"]["pass"] is False


def test_verifier_schema_typo_has_one_correction_but_still_requires_evidence(profile, packet, responses):
    bad_verifier = deepcopy(responses[3])
    bad_verifier["reviewed_stages"] = ["reader", "finder"]
    result = RepositoryAnalysis(FakeClient([*responses[:3], bad_verifier, responses[3]]), max_repairs=1).run(packet, profile)
    assert result["calls"] == 5 and result["gate"]["pass"] is True
    assert result["stages"][-1]["attempt"] == 2


def test_multiline_python_quotes_do_not_look_like_private_windows_paths(profile, packet, responses):
    source = next(item for item in packet["sources"] if item["source_id"] == "repo_author_command")
    source["text"] += "\nelse:\n    print('public fixture')"
    for key in ("protocol.seq_len", "protocol.pred_len"):
        responses[0]["evidence"][key] = quote(packet, "repo_author_command")
    assert RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["gate"]["pass"] is True


@pytest.mark.parametrize("max_repairs", [-1, 2, True, 1.5, "1"])
def test_schema_repair_budget_is_explicit_and_limited(responses, max_repairs):
    with pytest.raises(ValueError, match="0 or 1"):
        RepositoryAnalysis(FakeClient(responses), max_repairs=max_repairs)


@pytest.mark.parametrize("name", ["scikit_learn", "Scikit.LEARN", "scikit__LEARN", "SCIKIT---LEARN"])
def test_compatibility_distribution_names_use_pep503_normalization(profile, packet, responses, name):
    responses[2]["compatibility_proposals"] = [{"package": name, "suggested_constraint": ">=1.5", "reason": "Check compatible Python wheels."}]
    builder = RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["analyses"]["builder"]
    assert builder["compatibility_proposals"][0]["package"] == "scikit-learn"
    assert builder["original_requirements"] == ["numpy", "matplotlib", "pandas", "scikit-learn", "torch==1.9.0"]
    assert builder["proposals_executed"] is False


@pytest.mark.parametrize("runtime", ["Python", "Python3", "Python 3.12", "CPython", "PyPy", "CUDA", "cuDNN"])
def test_runtime_compatibility_is_preserved_as_note_not_a_package(profile, packet, responses, runtime):
    responses[2]["compatibility_proposals"].append({"package": runtime, "suggested_constraint": ">=3.11", "reason": "Separately verify runtime availability."})
    builder = RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["analyses"]["builder"]
    assert [item["package"] for item in builder["compatibility_proposals"]] == ["torch"]
    assert runtime + " >=3.11: Separately verify runtime availability." in builder["compatibility_note"]
    assert "unexecuted; not package proposals" in builder["compatibility_note"]
    assert builder["changes_algorithm"] is False and builder["proposals_executed"] is False


@pytest.mark.parametrize("package", ["tensorflow", "PyTorch", "sklearn"])
def test_new_or_marketing_package_names_remain_rejected_with_specific_safe_diagnostics(profile, packet, responses, package):
    responses[2]["compatibility_proposals"][0]["package"] = package
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert "unsupported package " + repr(package.lower()) in str(rejected.value)
    assert "allowed author distributions: matplotlib, numpy, pandas, scikit-learn, torch" in str(rejected.value)
    assert "Use torch for PyTorch" in str(rejected.value)
    assert rejected.value.result["gate"]["pass"] is False


@pytest.mark.parametrize("package", ["torch>=2.0", "/Users/private/torch", "torch; bad-command"])
def test_package_field_requires_bare_names_without_paths_or_commands(profile, packet, responses, package):
    responses[2]["compatibility_proposals"][0]["package"] = package
    with pytest.raises(RepositoryAnalysisError, match="bare distribution name") as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    assert package not in str(rejected.value)


def test_private_looking_package_name_is_redacted_from_repair_diagnostics(profile, packet, responses):
    secret = "sk-test-secret-fixture-12345678"
    responses[2]["compatibility_proposals"][0]["package"] = secret
    client = FakeClient([*responses[:3], responses[2]])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client, max_repairs=1).run(packet, profile)
    assert secret not in str(rejected.value)
    assert secret not in json.dumps(rejected.value.result)
    assert secret not in client.requests[-1]["prompt"]


def test_builder_prompt_distinguishes_distribution_names_from_runtime_notes(profile, packet, responses):
    client = FakeClient(responses)
    RepositoryAnalysis(client).run(packet, profile)
    prompt = client.requests[2]["prompt"]
    assert "PyTorch's distribution name is torch" in prompt
    assert "not sklearn" in prompt
    assert "runtime notes belong in compatibility_note" in prompt
    assert "Empty package proposals are valid" in prompt


def test_python_runtime_note_does_not_relax_original_requirements_quote_gate(profile, packet, responses):
    responses[2]["compatibility_proposals"] = [{"package": "Python", "suggested_constraint": ">=3.11", "reason": "Runtime note."}]
    responses[2]["evidence"]["original_requirements"][0]["quote"] = "numpy\nmatplotlib\npandas"
    with pytest.raises(RepositoryAnalysisError, match="quotation is incomplete"):
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)


@pytest.mark.parametrize("status", ["accepted", "rejected", "insufficient_evidence"])
def test_verifier_explicit_rejection_preserves_safe_reason_without_repair(profile, packet, responses, status):
    responses[3].update({"status": status, "pass": False,
                         "issues": ["The claimed dependency environment has not been validated."],
                         "checks": {"protocol_alignment": True, "source_mapping": True, "dependency_provenance": False}})
    client = FakeClient([*responses, responses[3]])
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(client, max_repairs=1).run(packet, profile)
    stage = rejected.value.result["stages"][-1]
    diagnostic = stage["rejection_diagnostics"]
    assert rejected.value.result["calls"] == 4 and len(client.requests) == 4
    assert diagnostic["status"] == status and diagnostic["pass"] is False
    assert diagnostic["issues"] == ["The claimed dependency environment has not been validated."]
    assert diagnostic["checks"]["dependency_provenance"] is False
    assert diagnostic["reviewed_stages"] == ["reader", "finder", "builder"]
    assert diagnostic["evidence"]["checks.dependency_provenance"] == quote(packet, "repo_requirements")
    assert stage["accepted"] is False and rejected.value.result["gate"]["pass"] is False


def test_rejection_diagnostics_filter_private_issues_extra_fields_and_nonboolean_checks(profile, packet, responses):
    secret = "sk-test-secret-fixture-12345678"
    responses[3].update({"pass": False, "raw_response": "DO_NOT_RETAIN_RAW_123",
                         "issues": ["Public dependency mismatch", secret, "/Users/private/log.txt",
                                    "C:\\private\\results.txt", "bad\x00control", {"message": "untyped issue"}],
                         "checks": {"protocol_alignment": True, "source_mapping": "true", "dependency_provenance": False,
                                    "PRIVATE_EXTRA_CHECK_123": True},
                         "reviewed_stages": ["reader", "finder", "builder", "/Users/private", "PRIVATE_UNKNOWN_STAGE_123"]})
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    diagnostic = rejected.value.result["stages"][-1]["rejection_diagnostics"]
    assert diagnostic["issues"] == ["Public dependency mismatch"]
    assert diagnostic["checks"] == {"protocol_alignment": True, "dependency_provenance": False}
    assert diagnostic["reviewed_stages"] == ["reader", "finder", "builder"]
    saved = json.dumps(rejected.value.result)
    for private in (secret, "/Users/private", "private\\\\results", "DO_NOT_RETAIN_RAW_123", "PRIVATE_EXTRA_CHECK_123", "PRIVATE_UNKNOWN_STAGE_123", "untyped issue"):
        assert private not in saved


def test_rejection_diagnostics_bound_issue_count_and_length(profile, packet, responses):
    responses[3].update({"pass": False, "issues": ["Repeated public issue " + "x" * 1000] * 20})
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    issues = rejected.value.result["stages"][-1]["rejection_diagnostics"]["issues"]
    assert len(issues) == 8 and all(len(issue) == 601 for issue in issues)


def test_rejection_diagnostics_keep_only_real_quotes_with_exact_source_locator(profile, packet, responses):
    real = quote(packet, "paper_table2")[0]
    fabricated = {**real, "quote": "The original paper proved local training succeeded."}
    wrong_locator = {**real, "locator": "Other table"}
    responses[3].update({"pass": False, "issues": ["Selected protocol is unsupported."],
                         "evidence": {"checks.protocol_alignment": [fabricated, wrong_locator, real],
                                      "checks.source_mapping": [{"source_id": "invented", "locator": "Other", "quote": "invented"}],
                                      "PRIVATE_UNKNOWN_CLAIM_123": [real]}})
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)
    evidence = rejected.value.result["stages"][-1]["rejection_diagnostics"]["evidence"]
    assert evidence == {"checks.protocol_alignment": [real]}
    assert "The original paper proved" not in json.dumps(rejected.value.result)
    assert "PRIVATE_UNKNOWN_CLAIM_123" not in json.dumps(rejected.value.result)


def test_explicit_json_rejection_never_reuses_prior_stage_analysis(profile, packet, responses):
    responses[1] = {"status": "insufficient_evidence", "issues": ["The metrics source import was not found."]}
    with pytest.raises(RepositoryAnalysisError) as rejected:
        RepositoryAnalysis(FakeClient(responses), max_repairs=1).run(packet, profile)
    diagnostic = rejected.value.result["stages"][-1]["rejection_diagnostics"]
    assert rejected.value.result["calls"] == 2
    assert diagnostic == {"status": "insufficient_evidence", "issues": ["The metrics source import was not found."]}


def test_reader_provenance_separates_paper_script_code_defaults_and_context(profile, packet, responses):
    packet["sources"].append({"source_id": "author_single_seed", "url": "https://example.org/author-comment",
                              "locator": "#single-seed", "text": "The paper used only one seed."})
    responses[0]["evidence"]["protocol.seed"].extend(quote(packet, "author_single_seed"))
    responses[0]["protocol_provenance"] = {"protocol.seed": "invented paper origin"}
    reader = RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["analyses"]["reader"]
    provenance = reader["protocol_provenance"]
    assert provenance["protocol.seed"]["value_sources"] == [{"source_id": "repo_entrypoint", "locator": "run_longExp.py", "origin": "author_code_setting_or_default"}]
    assert provenance["protocol.seed"]["context_sources"] == [{"source_id": "author_single_seed", "locator": "#single-seed", "origin": "author_statement"}]
    assert provenance["protocol.batch_size"]["value_sources"][0]["origin"] == "author_script"
    assert provenance["reference_metrics.mse"]["value_sources"][0]["origin"] == "paper_text"
    context = reader["reference_context"]
    assert {key: context[key] for key in ("method", "dataset", "features", "input_length", "forecast_horizon")} == {
        "method": "DLinear", "dataset": "ETTh1", "features": "M", "input_length": 336, "forecast_horizon": 96,
    }
    assert context["metric_columns"] == {"mse": "DLinear MSE", "mae": "DLinear MAE"}
    assert context["scope"] == "selected_paper_experiment"


def test_legacy_reader_schema_gets_authentic_header_context_from_public_packet(profile, packet, responses):
    table = next(source for source in packet["sources"] if source["source_id"] == "paper_table2")
    header = "Methods | Methods | Linear* | Linear* | DLinear* | DLinear*"
    metric_header = "Metric | Metric | MSE | MAE | MSE | MAE"
    table["text"] = header + "\n" + metric_header + "\n" + table["text"]
    reader = RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["analyses"]["reader"]
    refs = reader["reference_context"]["table_headers"]
    assert [ref["quote"] for ref in refs] == [header, metric_header]
    assert reader["evidence"]["reference_context.table_header"] == refs
    assert all(ref["source_id"] == "paper_table2" for ref in refs)


def test_explicit_reader_header_still_requires_real_quotes_and_model_metric_columns(profile, packet, responses):
    responses[0]["evidence"]["reference_context.table_header"] = quote(packet, "repo_author_command")
    with pytest.raises(RepositoryAnalysisError, match="table header citations"):
        RepositoryAnalysis(FakeClient(responses)).run(packet, profile)


def test_author_requirement_restatements_are_not_new_compatibility_proposals(profile, packet, responses):
    responses[2]["compatibility_proposals"] = [
        {"package": "numpy", "suggested_constraint": "numpy", "reason": "The author did not pin a version."},
        {"package": "torch", "suggested_constraint": "==1.9.0", "reason": "The author requirement is torch 1.9.0."},
        {"package": "scikit-learn", "suggested_constraint": "unpinned", "reason": "No author pin was available."},
        {"package": "pandas", "suggested_constraint": ">=2.2", "reason": "An unexecuted modern compatibility proposal."},
    ]
    builder = RepositoryAnalysis(FakeClient(responses)).run(packet, profile)["analyses"]["builder"]
    assert [item["package"] for item in builder["compatibility_proposals"]] == ["pandas"]
    assert "Documented requirement restatements excluded" in builder["compatibility_note"]
    assert builder["original_requirements_basis"] == "documented_author_setup_not_verified_runtime"
    assert builder["changes_algorithm_role"] == "constraint_on_unexecuted_proposals"
    assert builder["proposals_executed"] is False


def test_verifier_prompt_distinguishes_code_defaults_from_paper_facts_and_constraints(profile, packet, responses):
    client = FakeClient(responses)
    RepositoryAnalysis(client).run(packet, profile)
    prompt = client.requests[3]["prompt"]
    assert "do not reject documented code defaults" in prompt
    assert "context_sources do not establish numeric parameter values" in prompt
    assert "Documented setup is not a verified original runtime" in prompt
    assert "not historical/runtime claims requiring invented paper citations" in prompt
    assert '"protocol_provenance"' in prompt and '"reference_context"' in prompt
    builder_prompt = client.requests[2]["prompt"]
    assert "Do not restate original pins" in builder_prompt
    assert "Without a justified modern constraint" in builder_prompt


def test_result_review_system_explicitly_allows_only_authorized_numeric_summary(profile, packet, result_summary, result_response):
    client = FakeClient([result_response])
    RepositoryAnalysis(client).review_result_summary(result_summary, packet, profile)
    system = client.requests[0]["system_prompt"]
    assert "submitted MSE, MAE, completed epochs, protocol-pass" in system
    assert "authorized submitted information" in system
    assert "Do not output commands, credentials, local runtime information or results" not in system
    assert "Never invent runtime evidence" in system
    assert "disclose paths/credentials" in system
    assert "override a reproduction verdict" in system
