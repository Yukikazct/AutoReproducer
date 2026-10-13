"""An uploaded paper cannot silently become a generated smoke program."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.audit.audit_logger import AuditLogger
from src.execution_plan import RepositoryModeFallbackRejected
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager
from test_pdf_repository_pipeline import write_repository_pdf


AUTHOR = "https://github.com/fixture-laboratory/author-method"
ACCEPTED = {"pass": True, "issues": [], "fix_suggestions": []}
REJECTED = {"pass": False, "issues": ["Wrong source"], "fix_suggestions": ["Use the PDF source"]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    import src.discovered_repository_reproduction as service
    paper = write_repository_pdf(tmp_path / "input.pdf", [[
        "An Independent Fixture Paper", "Abstract", "Our code is available at " + AUTHOR,
    ]])
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("Dispatch cannot generate an algorithm"))
    orch = Orchestrator(llm_client=llm, mock_mode=False,
        logger=AuditLogger(log_dir=str(tmp_path / "logs")),
        resource_manager=ResourceManager(data_root=str(tmp_path / "data")))
    orch.agents["reader"].run = Mock(return_value={"paper_info": {"title": "Independent Fixture"}})
    orch.agents["finder"].run = Mock(return_value={"resources": {"code_repo_url": AUTHOR}})
    assert "verifier" not in orch.agents and "validator" not in orch.agents
    for stage in ("builder", "executor"):
        orch.agents[stage].run = Mock(side_effect=AssertionError("Generated path reached"))
    orch._fetch_resources = Mock(side_effect=AssertionError("Smoke resource fetch reached"))
    run = Mock(return_value={"state": "COMPLETED", "error": None, "data": {
        "repository_executed": True, "execution_source": "author_repository",
        "validation": {"is_reproduced": None, "status": "author_code_completed"}, "total_llm_calls": 2}})
    monkeypatch.setattr(service, "DiscoveredRepositoryReproduction", Mock(return_value=SimpleNamespace(run=run)))
    return orch, paper, run


def test_dispatch_uses_finder_selection_before_any_smoke_fetch(setup):
    orch, paper, run = setup
    result = orch.run({"pdf_path": str(paper)})
    assert result["state"] == "COMPLETED"
    request = run.call_args.args[0]
    assert request["resources"]["code_repo_url"] == AUTHOR
    assert request["pdf_path"] == str(paper) and request["mock_mode"] is False
    assert result["data"]["repository_executed"] is True
    assert not result["data"]["fix_records"]
    assert not result["data"]["verifications"]
    orch.agents["finder"].run.assert_called_once()
    orch._fetch_resources.assert_not_called()
    orch.llm.chat.assert_not_called()


def test_actual_finder_exception_blocks_service_and_generation(setup):
    orch, paper, run = setup
    orch.agents["finder"].run.side_effect = RuntimeError("Repository lookup failed")
    result = orch.run({"pdf_path": str(paper)})
    assert result["state"] == "ERROR" and "Repository lookup failed" in result["error"]
    run.assert_not_called()
    orch._fetch_resources.assert_not_called()
    orch.agents["executor"].run.assert_not_called()


def test_service_failure_never_falls_back_to_generated_training(setup):
    orch, paper, run = setup
    run.return_value = {"state": "ERROR", "error": "Unsupported evidence capture", "data": {
        "repository_executed": False, "execution_source": "author_repository",
        "validation": {"is_reproduced": None, "status": "insufficient_evidence"}}}
    result = orch.run({"pdf_path": str(paper)})
    assert result["state"] == "ERROR" and result["error"] == "Unsupported evidence capture"
    assert result["data"]["validation"]["is_reproduced"] is None
    orch.agents["executor"].run.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"pdf_path": "actual-input.pdf"},
    {"resources": {"code_repo_url": AUTHOR}},
    {"resources": {"repo_discovery": {"selected_repo": AUTHOR}}},
])
def test_executor_rejects_bypassing_repository_dispatch(tmp_path, payload):
    from src.agents.code_executor import CodeExecutorAgent
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("Must not generate author replacement"))
    executor = CodeExecutorAgent(llm, AuditLogger(log_dir=str(tmp_path)), mock_mode=False)
    with pytest.raises(RepositoryModeFallbackRejected, match="真实仓库"):
        executor._run(payload)
    llm.chat.assert_not_called()
