"""Title discovery must retain its evidence and stop before generating code."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from frontend.backend_pipeline import ProgressStore
from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager


@pytest.mark.parametrize("selected", ["https://github.com/example/author-code", "未找到"])
def test_title_keeps_discovery_and_requests_pdf_before_download_or_execution(tmp_path, selected):
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("No generated implementation"))
    orch = Orchestrator(llm_client=llm, mock_mode=False,
        logger=AuditLogger(log_dir=str(tmp_path / "logs")),
        resource_manager=ResourceManager(data_root=str(tmp_path / "data")))
    orch.agents["reader"].run = Mock(return_value={
        "paper_info": {"title": "An unseen paper", "insufficient_info": True},
        "reading_coverage": {"scope": "provided_title", "complete": True}, "llm_calls": 1})
    resources = {"code_repo_url": selected, "repo_discovery": {"selected_repo": selected},
                 "repository_identity": {"status": "candidate_unverified"}}
    orch.agents["finder"].run = Mock(return_value={"resources": deepcopy(resources), "llm_calls": 1})
    for name in ("builder", "executor", "optimizer"):
        orch.agents[name].run = Mock(side_effect=AssertionError("Title evidence cannot authorize execution"))
    orch._fetch_resources = Mock(side_effect=AssertionError("No author source downloaded yet"))
    progress = ProgressStore(str(tmp_path / "progress.jsonl"))
    request = {"paper_title": "An unseen paper"}

    result = orch.run(request, on_event=progress.emit)

    assert request == {"paper_title": "An unseen paper"}
    assert result["state"] == "ERROR" and "上传PDF" in result["error"]
    data = result["data"]
    assert data["resources"] == resources
    assert data["source_request"]["status"] == "needs_pdf"
    assert data["source_request"]["selected_repository"] == (selected if selected.startswith("https:") else "")
    assert data["total_llm_calls"] == 2
    assert data["execution"]["executed"] is False
    assert data["repository_executed"] is False
    assert data["execution_source"] == "author_repository"
    assert data["validation"]["is_reproduced"] is None
    assert data["validation"]["paper_metrics_verified"] is False
    assert "待补充论文 PDF" in data["report"] and "下一步" in data["report"]
    assert "❌ 失败" not in data["report"]
    assert "仓库实验尚未执行" in data["report"]
    assert "尚未准备执行命令" in data["report"]
    assert "固定版本作者仓库" not in data["report"]
    assert "官方仓库执行命令" not in data["report"]
    if selected.startswith("https:"):
        assert "候选仓库（作者归属未核验）" in data["report"]
        assert selected in data["report"]
    else:
        assert "候选仓库（作者归属未核验）" not in data["report"]
    assert orch.agents["reader"].run.call_args.args[0]["reading_purpose"] == "repository_discovery"
    for name in ("builder", "executor", "optimizer"):
        orch.agents[name].run.assert_not_called()
    orch._fetch_resources.assert_not_called()
    llm.chat.assert_not_called()
    stages = {row["id"]: row for row in ProgressStore.read_snapshot(str(progress.path))["pipeline_stages"]}
    assert stages["require_paper_source"]["status"] == "error"
    assert stages["generate_report"]["status"] == "success"
    assert "execute_code" not in stages and "build_environment" not in stages


def test_explicit_supplied_code_keeps_its_execution_route(tmp_path):
    llm = LLMClient(mock_mode=True)
    orch = Orchestrator(llm_client=llm, mock_mode=False,
        logger=AuditLogger(log_dir=str(tmp_path / "logs")),
        resource_manager=ResourceManager(data_root=str(tmp_path / "data")))
    orch.agents["reader"].run = Mock(return_value={"paper_info": {"title": "An unseen paper"}})
    orch.agents["finder"].run = Mock(return_value={"resources": {}})
    orch.agents["builder"].run = Mock(return_value={"env_config": {}})
    orch.agents["executor"].run = Mock(return_value={
        "executed": True, "success": True, "final": {"exit_code": 0, "success": True}})
    orch._fetch_resources = Mock()
    orch._finalize_storage = Mock()

    result = orch.run({"paper_title": "An unseen paper", "code": "print('user code')"})

    assert result["state"] == "COMPLETED"
    assert "source_request" not in result["data"]
    orch.agents["executor"].run.assert_called_once()
