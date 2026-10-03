"""Issue #18：Astra 五个已确认反例及最终执行/单位/指标证据门控。"""
import json
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.llm.llm_client import LLMClient
from src.runtime_metrics import METRICS_MARKER, parse_runtime_metrics


def execution(stdout="", exit_code=0, stage="full"):
    final = {"stage": stage, "success": exit_code == 0, "exit_code": exit_code,
             "stdout": stdout, "stderr": "RuntimeError: interrupted" if exit_code else ""}
    return {"success": final["success"], "final": final, "stages": [final]}


@pytest.fixture
def validator():
    llm = Mock(spec=LLMClient)
    llm.get_call_count.return_value = 0
    llm.chat.return_value = json.dumps({"match": True, "analysis": "解释"})
    return ResultValidatorAgent(llm, logger=Mock())


def validate(agent, run, metrics=None, **paper_fields):
    return agent.run({"paper_info": {"metrics": metrics or {}, **paper_fields}, "execution": run})


@pytest.mark.parametrize("text,expected", [
    ("loss: 1.2e-4", 0.00012), ("loss=-1.2E+3", -1200),
    ("loss: +.25", 0.25), ("loss = -0.25", -0.25),
])
def test_scientific_notation_and_signed_values(validator, text, expected):
    assert validator._extract_metrics(text)["loss"] == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "Epoch 1 loss: 1.0\nFinal loss: 0.01", "loss=1.0\nloss:0.01",
    "loss:1.0\nloss=0.01", "Final loss=0.01\npaper reference loss:1.0",
])
def test_final_metric_replaces_first_and_ignores_paper_reference(validator, text):
    assert validator._extract_metrics(text)["loss"] == 0.01


def test_missing_required_metric_rejects_partial_match(validator):
    result = validate(validator, execution("accuracy:0.90"), {"accuracy": 0.9, "f1": 0.85})
    assert result["is_reproduced"] is False
    assert result["validation"]["missing_metrics"]
    assert result["result_level"] == "failed"
    validator.llm.chat.assert_not_called()


@pytest.mark.parametrize("name", ["mse", "mae", "rmse", "loss"])
def test_error_metrics_never_use_implicit_percent_conversion(validator, name):
    result = validate(validator, execution(f"{name}:50"), {name: 0.5})
    assert result["is_reproduced"] is False
    assert "9900.0%" in result["validation"]["differences"][0]


@pytest.mark.parametrize("shape", ["final", "flat", "nested", "stages"])
def test_nonzero_exit_cannot_be_overridden_by_matching_metrics_or_llm(validator, shape):
    run = execution("accuracy:0.9", exit_code=1)
    if shape == "flat":
        run = run["final"]
    elif shape == "nested":
        run = {"execution": run["final"]}
    elif shape == "stages":
        run = {"stages": run["stages"]}
    result = validate(validator, run, {"accuracy": 0.9})
    assert result["status"] == "execution_failed"
    assert result["is_reproduced"] is False
    assert result["metrics_comparison"]["actual"] == {"accuracy": 0.9}
    assert "退出码 1" in result["reason"]
    validator.llm.chat.assert_not_called()


def test_required_step_failure_stops_success_even_if_final_evaluation_matches(validator):
    run = execution("mse:0.5")
    run["steps"] = [{"id": "train", "exit_code": 1, "success": False}, run["final"]]
    result = validate(validator, run, {"mse": 0.5})
    assert result["is_reproduced"] is False
    assert "train" in result["reason"]


def test_optional_step_failure_does_not_veto_required_success(validator):
    run = execution("mse:0.5")
    run["steps"] = [{"id": "optional_plot", "required": False, "exit_code": 1}, run["final"]]
    assert validate(validator, run, {"mse": 0.5})["is_reproduced"] is True


def test_repair_history_failure_is_replaced_by_successful_rerun(validator):
    run = execution("mse:0.5")
    run["stages"] = [
        {"stage": "smoke", "repair_round": 0, "exit_code": 1, "success": False},
        {"stage": "full", "repair_round": 0, "exit_code": 1, "success": False},
        {"stage": "smoke", "repair_round": 1, "exit_code": 0, "success": True},
        {"repair_round": 1, **run["final"]},
    ]
    # 真实 CodeExecutor 的 final 不含 repair_round，历史记录含该字段。
    assert validate(validator, run, {"mse": 0.5})["is_reproduced"] is True


def test_missing_execution_evidence_cannot_be_inferred_from_stdout(validator):
    result = validate(validator, {"stdout": "mse:0.5"}, {"mse": 0.5})
    assert result["status"] == "execution_incomplete"
    assert result["is_reproduced"] is None
    validator.llm.chat.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-Infinity"])
def test_nonfinite_final_text_metric_cannot_fall_back_to_earlier_valid_value(validator, value):
    result = validate(validator, execution(f"loss:0.1\nFinal loss:{value}"), {"loss": 0.1})
    assert result["status"] == "invalid_metrics"
    assert result["is_reproduced"] is False
    validator.llm.chat.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None, "not-a-number"])
def test_invalid_reference_values_never_pass(validator, value):
    assert validator._local_compare({"mse": value}, {"mse": 0.5})["match"] is False


def test_explicit_percent_and_fraction_units_allow_conversion(validator):
    result = validate(validator, execution("Test accuracy:85.2%"), {"accuracy": 0.85},
                      metric_units={"accuracy": "fraction"})
    assert result["is_reproduced"] is True
    assert result["metric_records"][0]["unit"] == "percent"


def test_unmarked_units_do_not_authorize_percent_conversion(validator):
    result = validate(validator, execution("accuracy:85.2%"), {"accuracy": 0.85})
    assert result["is_reproduced"] is False
    assert "单位不一致" in result["validation"]["differences"][0]


def test_zero_reference_requires_zero_measurement(validator):
    assert validator._local_compare({"loss": 0}, {"loss": 0})["match"] is True
    assert validator._local_compare({"loss": 0}, {"loss": 1e-12})["match"] is False


def test_no_reference_metrics_stays_inconclusive_despite_corpus_score(validator, monkeypatch):
    monkeypatch.setattr("src.corpus.get_declared_score", lambda _: 0.85)
    result = validator.run({"paper_info": {"metrics": {}}, "corpus_paper": "bbox",
                            "execution": execution("reproduction_score:0.85")})
    assert result["status"] == "no_reference_metrics"
    assert result["result_level"] == "inconclusive"
    assert result["execution_status"] == "experiment_completed"
    validator.llm.chat.assert_not_called()


def test_smoke_success_cannot_claim_numeric_reproduction(validator):
    result = validate(validator, execution("mse:0.5", stage="smoke"), {"mse": 0.5})
    assert result["status"] == "smoke_passed"
    assert result["is_reproduced"] is None
    validator.llm.chat.assert_not_called()


def test_llm_cannot_veto_a_deterministic_match(validator):
    validator.llm.chat.return_value = json.dumps({"match": False, "analysis": "错误解释"})
    result = validate(validator, execution("mse:0.5"), {"mse": 0.5})
    assert result["is_reproduced"] is True
    assert "错误解释" not in result["validation"]["analysis"]
    assert result["validation"]["verdict_source"] == "deterministic"


def test_unavailable_llm_preserves_deterministic_verdict(validator):
    validator.llm.chat.side_effect = RuntimeError("API unavailable")
    assert validate(validator, execution("mse:0.5"), {"mse": 0.5})["is_reproduced"] is True


def structured_line(records):
    return METRICS_MARKER + json.dumps({"version": 1, "metrics": records})


def test_structured_final_metrics_override_training_logs_and_preserve_evidence(validator):
    text = "Training loss:1.0\n" + structured_line([
        {"name": "loss", "value": 1.2e-4, "unit": "", "stage": "final", "split": "test"}])
    result = validate(validator, execution(text), {"loss": {"value": 0.00012, "split": "test"}})
    assert result["is_reproduced"] is True
    assert result["metrics_comparison"]["actual"] == {"loss": 0.00012}
    assert result["metric_records"][0]["source"] == "stdout:2"


@pytest.mark.parametrize("payload", ["not json", '{"version":2,"metrics":{"mse":0.5}}',
                                     '{"version":1,"metrics":[{"name":"mse","value":true}]}'])
def test_invalid_final_structured_record_blocks_text_fallback(validator, payload):
    result = validate(validator, execution("mse:0.5\n" + METRICS_MARKER + payload), {"mse": 0.5})
    assert result["is_reproduced"] is False
    assert result["status"] == "invalid_metrics"


def test_last_structured_message_is_authoritative(validator):
    text = structured_line({"mse": 0.5}) + "\n" + structured_line({"mse": 0.1})
    assert validator._extract_metrics(text) == {"mse": 0.1}


def test_direct_structured_execution_metrics_override_stdout(validator):
    run = execution("mse:50")
    run["final"]["metrics"] = {"mse": {"value": 0.5, "unit": "", "stage": "final", "split": "test"}}
    result = validate(validator, run, {"mse": 0.5})
    assert result["is_reproduced"] is True
    assert result["metric_records"][0]["source"] == "execution.metrics"


@pytest.mark.parametrize("record", [
    {"name": "mse", "value": 0.5, "stage": "train", "split": "train"},
    {"name": "mse", "value": 0.5, "stage": "final", "split": "train"},
])
def test_training_evidence_cannot_satisfy_final_test_contract(validator, record):
    result = validate(validator, execution(structured_line([record])),
                      {"mse": {"value": 0.5, "stage": "final", "split": "test"}})
    assert result["is_reproduced"] is False


def test_training_text_is_not_final_evaluation(validator):
    result = validate(validator, execution("Training loss:0.1"), {"loss": 0.1})
    assert result["is_reproduced"] is False
    assert "最终评估" in result["validation"]["differences"][-1]


def test_final_stdout_takes_precedence_over_top_level_history(validator):
    run = execution("mse:0.1")
    run["stdout"] = "mse:0.5"
    assert validate(validator, run, {"mse": 0.5})["is_reproduced"] is False


def test_precision_is_distinct_from_accuracy():
    records, errors = parse_runtime_metrics("精确率:0.9\n准确率:0.8")
    assert not errors
    assert {r["name"]: r["value"] for r in records} == {"precision": 0.9, "accuracy": 0.8}


@pytest.mark.parametrize("stage,code,state", [("smoke", 0, "仅冒烟通过"),
                                               ("full", 0, "论文数值核验通过"),
                                               ("full", 1, "执行失败")])
def test_report_separates_execution_level_and_numeric_verdict(validator, stage, code, state):
    run = execution("mse:0.5", exit_code=code, stage=stage)
    result = validate(validator, run, {"mse": 0.5})
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": run, "validation": result})
    assert state in report
    assert "实际指标证据" in report
    assert "stdout:1" in report
    if code or stage == "smoke":
        assert "**复现状态**: ✅" not in report


def test_web_pipeline_keeps_failure_report_and_skips_optimization(tmp_path, monkeypatch):
    from frontend.backend_pipeline import run_pipeline_core
    from src.agents.optimizer import OptimizerAgent

    monkeypatch.setattr(CodeExecutorAgent, "run", lambda *args: execution("accuracy:0.85\nf1:0.82", exit_code=1))
    optimize = Mock(side_effect=AssertionError("失败执行不允许优化"))
    monkeypatch.setattr(OptimizerAgent, "run", optimize)
    result = run_pipeline_core(str(tmp_path / "progress.jsonl"), paper_title="Gate regression", mock_mode=True)
    assert result["state"] == "COMPLETED"
    assert result["data"]["validation"]["status"] == "execution_failed"
    assert "执行失败" in result["data"]["report"]
    optimize.assert_not_called()


@pytest.mark.parametrize("code,reference,expected", [
    ("print('accuracy:0.9')\nraise RuntimeError('after metrics')\n", {"accuracy": 0.9}, False),
    ("print('loss:1.0')\nprint('Final loss:1.2e-4')\n", {"loss": 0.00012}, True),
    ("print('mse:50')\n", {"mse": 0.5}, False),
])
def test_actual_subprocess_results_obey_gate(tmp_path, validator, code, reference, expected):
    runner = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock())
    final = runner.execute_in_workspace(code, str(tmp_path), stage="full")
    run = {"success": final["success"], "final": {"stage": "full", **final}}
    assert validate(validator, run, reference)["is_reproduced"] is expected
