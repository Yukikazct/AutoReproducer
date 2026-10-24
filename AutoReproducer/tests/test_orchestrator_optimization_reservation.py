"""Production orchestration reserves optimization without running any optimizer."""
from unittest.mock import Mock

import pytest

from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager


def pipeline(tmp_path, *, requested=False, workspace=False):
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("This reservation test must not call any model"))
    orch = Orchestrator(llm_client=llm, mock_mode=True,
                        logger=AuditLogger(log_dir=str(tmp_path / "logs")),
                        resource_manager=ResourceManager(data_root=str(tmp_path / "data")),
                        workspace_dir=str(tmp_path / "workspace") if workspace else None,
                        enable_optimization=requested)
    outputs = {
        "reader": {"paper_info": {"title": "Public paper", "method": "DLinear", "metrics": {"mse": 0.375}}},
        "finder": {"resources": {}}, "builder": {"env_config": {}},
        "executor": {"code": "print('baseline')", "executed": True, "final": {"success": True, "exit_code": 0}},
        "reporter": {"report": "Reserved optimization interface"},
    }
    for name, output in outputs.items():
        orch.agents[name].run = Mock(return_value=output)
    orch.agents["optimizer"].run = Mock(side_effect=AssertionError("Reserved optimizer must never run"))
    orch.agents["optimizer"].simulator = Mock(side_effect=AssertionError("Neither hash simulation nor retraining may run"))
    orch._fetch_resources = Mock()
    orch._finalize_storage = Mock()
    orch._materialize_workspace = Mock(side_effect=AssertionError("Do not create an optimization workspace"))
    return orch


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
def test_reserved_flag_never_runs_optimization_or_creates_workspace(tmp_path, requested, workspace):
    orch = pipeline(tmp_path, requested=requested, workspace=workspace)
    events = []
    result = orch.run({"paper_title": "Public paper"}, on_event=events.append)
    assert result["state"] == "COMPLETED"
    assert result["data"]["validation"]["is_reproduced"] is None
    assert result["data"]["validation"]["status"] == "mock_demo"
    assert "validator" not in orch.agents and "verifier" not in orch.agents
    assert not any(event.get("state") == "VALIDATE" for event in events)
    optimization = result["data"]["optimization"]
    assert optimization == {
        "optimized": False, "requested": requested, "available": False,
        "status": "not_implemented" if requested else "disabled",
        "reason": "智能优化接口已预留，当前版本尚未开放。" if requested else "智能优化未启用；当前版本仅保留未来接口。",
    }
    report = result["data"]["report"]
    original, marker, summary = report.partition("\n\n<!-- autoreproducer-run-finalization:start -->\n")
    assert original == "Reserved optimization interface"
    assert marker and report.count("<!-- autoreproducer-run-finalization:start -->") == 1
    assert report.count("<!-- autoreproducer-run-finalization:end -->") == 1
    assert "**总任务状态**: 已完成" in summary
    assert "**环境释放**: 已释放" in summary
    assert result["data"]["run_finalization"] == {"state": result["state"], "error": result["error"]}
    assert result["data"]["environment_cleanup"]["status"] == "released"
    assert result["data"]["report_length"] == len(report)
    orch.agents["optimizer"].run.assert_not_called()
    orch.agents["optimizer"].simulator.assert_not_called()
    orch._materialize_workspace.assert_not_called()
    orch.llm.chat.assert_not_called()
    assert not (tmp_path / "workspace").exists()
    assert [event["status"] for event in events if event.get("agent") == "Optimizer"] == ["skipped"]
    assert not any(event.get("state") in {"OPTIMIZING", "OPTIMIZED"} for event in events)
    assert not any(log.get("action") == "enter_OPTIMIZING" for log in result["audit_logs"])


@pytest.mark.parametrize("constructed,input_flag,expected", [(True, False, False), (False, True, True)])
def test_run_flag_overrides_constructor_reservation(tmp_path, constructed, input_flag, expected):
    orch = pipeline(tmp_path, requested=constructed)
    # This test concerns reservation flags, not PDF parsing. A nonexistent file
    # is now deliberately rejected before any generic pipeline work.
    result = orch.run({"paper_title": "Public paper", "enable_optimization": input_flag})
    assert result["data"]["optimization"]["requested"] is expected
    assert result["data"]["optimization"]["available"] is False
    orch.agents["optimizer"].run.assert_not_called()


def test_current_state_graph_goes_from_execution_to_report(tmp_path):
    orch = pipeline(tmp_path)
    assert orch._get_transitions("EXECUTE_CODE") == ["GENERATE_REPORT", "ERROR"]
    assert orch._get_transitions("VALIDATE") == []
