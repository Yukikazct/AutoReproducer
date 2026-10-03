"""真实运行回归：pip 提示不等于执行失败，无参考指标不判复现失败。"""
import json
from unittest.mock import Mock

import pytest

from src.agents.report_generator import ReportGeneratorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator

PIP_NOTICE = ("[notice] A new release of pip is available: 24.0 -> 26.2.1\n"
              "[notice] To update, run: pip install --upgrade pip\n")


def _execution(stdout="MSE = 0.945672\nRMSE = 0.972457\n", stderr=PIP_NOTICE,
               exit_code=0):
    final = {"stage": "full", "stdout": stdout, "stderr": stderr,
             "exit_code": exit_code, "success": exit_code == 0}
    return {"code": "print('MSE = 0.945672')", "stages": [final], "final": final}


@pytest.mark.parametrize("stdout", ["MSE = 0.945672\nRMSE = 0.972457", ""])
def test_missing_reference_has_no_verdict_and_does_not_call_model(stdout):
    llm = Mock(spec=LLMClient)
    llm.get_call_count.return_value = 0
    validator = ResultValidatorAgent(llm, logger=Mock())
    result = validator.run({"paper_info": {"metrics": {}},
                            "execution": _execution(stdout=stdout)})
    assert result["status"] == "no_reference_metrics"
    assert result["is_reproduced"] is None
    assert result["validation"]["match"] is None
    assert result["confidence"] == 0
    llm.chat.assert_not_called()
    if stdout:
        assert result["metrics_comparison"]["actual"]["mse"] == 0.945672

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.data = {"validation": result}
    assert "参考指标" in orchestrator._optimization_skip_reason()


def test_corpus_reference_still_allows_comparison(monkeypatch):
    monkeypatch.setattr("src.corpus.get_declared_score", lambda _: 0.85)
    llm = Mock(spec=LLMClient)
    llm.get_call_count.return_value = 1
    llm.chat.return_value = json.dumps({"match": True, "confidence": 0.8})
    result = ResultValidatorAgent(llm, logger=Mock()).run({
        "paper_info": {"metrics": {}}, "corpus_paper": "corpus-anchor",
        "execution": _execution(stdout="reproduction_score = 0.85\n"),
    })
    assert result["is_reproduced"] is True
    assert result["status"] == "reproduced"
    llm.chat.assert_called_once()


def test_report_keeps_pip_notice_as_diagnostic_and_preserves_actual_metrics():
    execution = _execution()
    validator = ResultValidatorAgent(LLMClient(mock_mode=True), logger=Mock())
    validation = validator.run({"paper_info": {"metrics": {}},
                               "execution": execution})
    report = ReportGeneratorAgent(logger=Mock())._build_report({
        "paper_info": {"metrics": {}}, "execution": execution,
        "validation": validation,
    })
    assert "**执行状态**: ✅ 成功" in report
    assert "### 诊断输出（stderr）" in report
    assert "本阶段退出码为 0，执行成功" in report
    assert PIP_NOTICE in report
    assert "无法核验（论文未声明参考指标数值）" in report
    assert "**复现状态**: ❌ 失败" not in report
    assert "| mse | N/A | 0.945672 |" in report


def test_report_keeps_real_error_and_failed_exit_code():
    traceback = "Traceback (most recent call last):\nValueError: bad input\n"
    report = ReportGeneratorAgent(logger=Mock())._build_report({
        "execution": _execution(stderr=traceback, exit_code=1),
    })
    assert "**执行状态**: ❌ 失败" in report
    assert "退出码 1" in report
    assert traceback in report
    assert "本阶段退出码为 0，执行成功" not in report
