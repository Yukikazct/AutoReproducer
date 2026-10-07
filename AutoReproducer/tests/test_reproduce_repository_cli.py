"""CLI preparation must not request unused credentials or run training."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_repository.py"
    spec = importlib.util.spec_from_file_location("reproduce_repository_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_with_review_does_not_request_credentials(cli, monkeypatch):
    orchestrator = Mock()
    orchestrator.run.return_value = {
        "state": "COMPLETED", "data": {"validation": {"result_level": "prepared"}},
    }
    factory = Mock(return_value=orchestrator)
    prompt = Mock(side_effect=AssertionError("Preparation does not need API credentials"))
    client = Mock(side_effect=AssertionError("Preparation does not need an API client"))
    monkeypatch.setattr(cli, "Orchestrator", factory)
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    monkeypatch.setattr(cli, "LLMClient", client)

    assert cli.main(["--prepare-only", "--llm-review", "--offline"]) == 0
    request = orchestrator.run.call_args.args[0]
    assert request == {
        "experiment_profile": "dlinear_etth1_reference", "prepare_only": True,
        "offline": True, "use_llm_review": False, "analysis_mode": "multi_agent",
        "allow_result_summary_review": False,
    }
    factory.assert_called_once_with(mock_mode=False, llm_client=None)
    prompt.assert_not_called()
    client.assert_not_called()
