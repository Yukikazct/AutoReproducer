"""Repository mode must never degrade into single-file generation.

The guarantee has three independent layers, so one accidental change cannot
silently reintroduce a run.py fallback: the Orchestrator routes the request to
RepositoryReproduction and returns, CodeExecutorAgent refuses a repository
profile outright, and a blocked step is rendered as a failure in the report.
"""
import json
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.execution_plan import RepositoryModeFallbackRejected
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator


def test_repository_request_never_reaches_the_single_file_executor(monkeypatch, tmp_path):
    touched = Mock(side_effect=AssertionError("repository mode must not generate a run.py"))
    monkeypatch.setattr(CodeExecutorAgent, "run", touched)
    outcome = {"state": "COMPLETED", "error": None,
               "data": {"execution": {"mode": "repository", "executed": True}}}
    reproduction = Mock(name="RepositoryReproduction")
    reproduction.return_value.run.return_value = outcome
    monkeypatch.setattr("src.repository_reproduction.RepositoryReproduction", reproduction)
    events = []
    orchestrator = Orchestrator(llm_client=LLMClient(mock_mode=True), mock_mode=False,
                                logger=Mock(), workspace_dir=str(tmp_path / "ws"),
                                resource_manager=Mock(data_root=str(tmp_path / "data")))
    result = orchestrator.run({"experiment_profile": "dlinear_etth1_smoke"},
                              on_event=events.append)
    assert result["state"] == "COMPLETED"
    reproduction.return_value.run.assert_called_once()
    touched.assert_not_called()
    assert not any(event.get("state") == "EXECUTE_CODE" for event in events)


def test_repository_request_in_mock_mode_refuses_instead_of_faking(monkeypatch, tmp_path):
    # A mock result must not stand in for a real repository run either: refusing here
    # is what keeps "we ran the authors' code" from ever being simulated.
    touched = Mock(side_effect=AssertionError("repository mode must not generate a run.py"))
    monkeypatch.setattr(CodeExecutorAgent, "run", touched)
    logger = Mock(session_id="20261003_110111")
    result = Orchestrator(mock_mode=True, logger=logger,
                         resource_manager=Mock(data_root=str(tmp_path / "data"))).run(
                             {"experiment_profile": "dlinear_etth1_smoke"})
    assert result["state"] == "ERROR"
    assert "Mock" in (result.get("error") or "")
    touched.assert_not_called()


def test_executor_refuses_a_repository_profile_before_writing_anything(tmp_path):
    agent = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    with pytest.raises(RepositoryModeFallbackRejected, match="不得改走单文件生成"):
        agent.run({"experiment_profile": "dlinear_etth1_smoke",
                   "paper_info": {"title": "x"}, "env_config": {}})
    assert list(tmp_path.glob("run.py")) == []
    assert agent.llm is not None  # refused without generating, not after generating


def test_a_plain_paper_request_still_uses_the_single_file_path(tmp_path, monkeypatch):
    # The guard keys off experiment_profile only, so the ordinary paper pipeline that
    # legitimately generates one script keeps working.
    agent = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    final = {"stage": "full", "success": True, "executed": True, "exit_code": 0,
             "stdout": "ok\n", "stderr": "", "artifacts": [], "artifact_warnings": []}
    monkeypatch.setattr(agent, "_execute_with_repair",
                        Mock(return_value=("print('ok')", [final], [final], final)))
    result = agent.run({"paper_info": {"title": "A paper"}, "env_config": {},
                        "code": "print('ok')\n"})
    assert result["success"] is True
    assert result["final"]["stdout"] == "ok\n"


def report(execution):
    return ReportGeneratorAgent(logger=Mock())._build_report({"execution": execution},
                                                             report_path=None)


def test_report_shows_blocked_steps_as_未执行_with_their_cause():
    execution = {"mode": "repository", "not_runnable": False, "executed": True,
                 "stages": [{"id": "train", "stage": "full", "cwd": ".", "success": False,
                             "exit_code": 3, "argv": ["python", "train.py"],
                             "stdout_path": "/run/train.stdout.log"}],
                 "skipped": [{"id": "eval", "not_run": "prerequisite_failed",
                              "blocked_by": "train", "required": True}],
                 "final": {"stage": "full", "success": False, "exit_code": 3,
                           "stdout": "", "stderr": "boom", "artifacts": []}}
    rendered = report(execution)
    assert "### 未执行的步骤" in rendered
    assert "未执行" in rendered and "train 未通过" in rendered
    assert "**eval**" in rendered


def test_report_names_the_missing_product_when_a_requirement_is_absent():
    execution = {"mode": "repository", "not_runnable": False, "executed": True,
                 "stages": [{"id": "train", "stage": "full", "cwd": ".", "success": True,
                             "exit_code": 0, "argv": ["python", "train.py"]}],
                 "skipped": [{"id": "eval", "not_run": "missing_requirement",
                              "requirement": "artifacts/checkpoint.pt", "required": True}],
                 "final": {"stage": "full", "success": True, "exit_code": 0,
                           "stdout": "", "stderr": "", "artifacts": []}}
    rendered = report(execution)
    assert "缺少前置产物 `artifacts/checkpoint.pt`" in rendered
    assert "被挡住的步骤按失败计入结论" in rendered


def test_report_omits_the_section_when_every_step_ran():
    execution = {"mode": "repository", "not_runnable": False, "executed": True,
                 "stages": [{"id": "train", "stage": "full", "cwd": ".", "success": True,
                             "exit_code": 0, "argv": ["python", "train.py"]}],
                 "skipped": [],
                 "final": {"stage": "full", "success": True, "exit_code": 0,
                           "stdout": "", "stderr": "", "artifacts": []}}
    assert "### 未执行的步骤" not in report(execution)


def test_runner_reports_blocked_steps_in_the_result_a_report_reads(tmp_path):
    # Guards the field name the report depends on: a rename in the runner would make
    # the blocked-step section silently disappear from every future report.
    from src.repository_runner import RepositoryRunner
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={})
    executor._ensure_local_deps = Mock(return_value=None)
    result = RepositoryRunner(executor=executor).run(tmp_path, [
        {"id": "train", "argv": ["python", "-c", "raise SystemExit(3)"], "cwd": ".",
         "timeout_s": 5, "env": {}},
        {"id": "eval", "argv": ["python", "-c", "print('X')"], "cwd": ".", "timeout_s": 5,
         "env": {}, "depends_on": ["train"]}], {})
    assert json.dumps(result["skipped"])  # present and serializable
    assert result["skipped"][0]["id"] == "eval"
    assert "### 未执行的步骤" in report(result)
