"""CLI preparation must not request unused credentials or run training."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import subprocess

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


def test_delegated_review_reads_hidden_key_before_noninteractive_child(cli, monkeypatch, capsys):
    from src import runtime_preparation as runtime
    from src import local_llm_settings
    monkeypatch.setattr(local_llm_settings, "load_local_llm_settings", lambda: None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    secret = "private-hidden-fixture"
    prompt = Mock(return_value=secret)
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    monkeypatch.setattr(runtime, "runtime_requirement", lambda: "Store host")
    monkeypatch.setattr(runtime, "prepare_runtime", Mock(return_value=runtime.RuntimePreparation(
        "safe-python.exe", True, False, "Store host", "cpython-312")))
    def execute(argv, **kwargs):
        assert secret not in " ".join(argv)
        assert kwargs["env"]["LLM_API_KEY"] == secret
        return subprocess.CompletedProcess(argv, 1, "completed " + secret, "failure " + secret)
    monkeypatch.setattr(runtime, "run_owned_process", execute)
    assert cli.main(["--llm-review"]) == 1
    prompt.assert_called_once()
    output = capsys.readouterr()
    assert secret not in output.out + output.err


def test_delegated_prepare_never_prompts_for_unused_credentials(cli, monkeypatch):
    from src import runtime_preparation as runtime
    monkeypatch.setattr(runtime, "runtime_requirement", lambda: "Store host")
    monkeypatch.setattr(runtime, "prepare_runtime", Mock(return_value=runtime.RuntimePreparation(
        "safe-python.exe", True, False, "Store host", "cpython-312")))
    monkeypatch.setattr(runtime, "run_owned_process", Mock(return_value=subprocess.CompletedProcess([], 0, "", "")))
    prompt = Mock(side_effect=AssertionError("unused credentials"))
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    assert cli.main(["--prepare-only", "--llm-review", "--offline"]) == 0
    prompt.assert_not_called()
