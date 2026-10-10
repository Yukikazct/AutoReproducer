"""Exact title input must reach the reviewed route through real entrypoints."""
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import frontend.backend_pipeline as backend
import frontend.history_manager as history
import src.local_llm_settings as local_settings
import src.repository_reproduction as repository
import src.runtime_preparation as runtime
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator


TITLE = "Neural Ordinary Differential Equations"
PROFILE = "neural_ode_spiral"
RESOLUTION = {
    "requested_title": TITLE, "profile": PROFILE,
    "source": "exact_paper_title", "scope": "official_method_experiment",
}
APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("review", [True, False])
def test_title_resolves_before_store_host_handoff_and_preserves_private_options(monkeypatch, tmp_path, review):
    path = tmp_path / "title-progress.jsonl"
    prepared = runtime.RuntimePreparation(str(tmp_path / "safe-python.exe"),
                                          True, False, "Store host", "cpython-312")
    requirement = Mock(return_value="Switch Store host to safe Python")
    prepare = Mock(return_value=prepared)
    host_factory = Mock(side_effect=AssertionError("Unsafe host constructed an orchestrator"))
    monkeypatch.setattr(runtime, "runtime_requirement", requirement)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    monkeypatch.setattr(backend, "_create_orchestrator", host_factory)
    secret = "title-fixture-private-key"
    monkeypatch.setenv("LLM_API_KEY", "host-fixture-private-key")
    requests = []

    def safe_worker(command, **kwargs):
        payload = json.loads(kwargs["input"])
        requests.append(payload)
        assert command[0] == prepared.executable
        assert payload["api_key"] == secret
        assert secret not in " ".join(command)
        assert "LLM_API_KEY" not in kwargs["env"]
        child = backend.ProgressStore(payload["progress_path"], reset=False)
        child.emit({"type": "done", "result": {
            "state": "COMPLETED", "error": None,
            "data": {"title_resolution": payload["title_resolution"]},
        }})
        return subprocess.CompletedProcess(command, 0, "", "")

    worker = Mock(side_effect=safe_worker)
    monkeypatch.setattr(runtime, "run_owned_process", worker)
    result = backend.run_pipeline_core(
        str(path), paper_title=TITLE, api_key=secret,
        use_llm_review=review, allow_result_summary_review=False,
        optimization_mode="off", prepare_only=False, prepare_environment=False,
    )

    requirement.assert_called_once()
    prepare.assert_called_once()
    worker.assert_called_once()
    host_factory.assert_not_called()
    request, = requests
    assert request["experiment_profile"] == PROFILE
    assert request["title_resolution"] == RESOLUTION
    assert request["paper_title"] == TITLE
    assert request["use_llm_review"] is review
    assert request["allow_result_summary_review"] is False
    assert request["optimization_mode"] == "off"
    assert request["prepare_only"] is False and request["prepare_environment"] is False
    assert request["_managed_runtime"] is True and request["_append_progress"] is True
    assert result["data"]["title_resolution"] == RESOLUTION
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "title_resolution"
    assert {key: events[0][key] for key in RESOLUTION} == RESOLUTION
    assert secret not in path.read_text(encoding="utf-8")
    assert "host-fixture-private-key" not in path.read_text(encoding="utf-8")
    assert backend.ProgressStore.read_snapshot(str(path))["done"] is True


@pytest.mark.parametrize("review", [True, False])
def test_direct_orchestrator_title_dispatches_reviewed_pipeline_without_generic_agents(monkeypatch, tmp_path, review):
    llm = LLMClient(mock_mode=False, api_key="fixture-key", base_url="https://api.invalid", model="fixture-model")
    llm.chat = Mock(side_effect=AssertionError("Generic title agents must not run"))
    manager = Mock(data_root=tmp_path)
    logger = Mock()
    logger.get_summary.return_value = []
    logger.get_stats.return_value = {}
    orchestrator = Orchestrator(llm_client=llm, mock_mode=False,
                                logger=logger, resource_manager=manager)
    generic_runs = []
    for agent in orchestrator.agents.values():
        agent.run = Mock(side_effect=AssertionError("Generic agent invoked"))
        generic_runs.append(agent.run)

    reviewed = Mock()
    reviewed.run.return_value = {
        "state": "COMPLETED", "error": None,
        "data": {"experiment_spec": {"id": PROFILE}},
    }
    factory = Mock(return_value=reviewed)
    monkeypatch.setattr(repository, "RepositoryReproduction", factory)
    request = {"paper_title": TITLE, "use_llm_review": review,
               "allow_result_summary_review": False, "optimization_mode": "off"}
    events = []
    result = orchestrator.run(request, on_event=events.append)

    factory.assert_called_once_with(tmp_path, logger, False, llm=llm)
    reviewed.run.assert_called_once()
    routed = reviewed.run.call_args.args[0]
    assert routed["experiment_profile"] == PROFILE
    assert routed["title_resolution"] == RESOLUTION
    assert routed["use_llm_review"] is review
    assert routed["allow_result_summary_review"] is False
    assert routed["optimization_mode"] == "off"
    assert reviewed.run.call_args.kwargs["on_event"] == events.append
    assert result["data"]["title_resolution"] == RESOLUTION
    assert result["data"]["experiment_spec"]["id"] == PROFILE
    assert request == {"paper_title": TITLE, "use_llm_review": review,
                       "allow_result_summary_review": False, "optimization_mode": "off"}
    llm.chat.assert_not_called()
    for run in generic_runs:
        run.assert_not_called()


@pytest.fixture
def title_app(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.invalid")
    monkeypatch.setenv("LLM_MODEL", "fixture-model")
    monkeypatch.setenv("LLM_API_KEY", "fixture-key")
    monkeypatch.setattr(local_settings, "load_local_llm_settings", Mock())
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    start = Mock()
    monkeypatch.setattr(backend, "run_pipeline_background", start)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["input_mode"] = "论文标题"
    app.session_state["mock_mode"] = False
    app.session_state["docker_probe"] = (False, "test uses local execution")
    app.run()
    assert not app.exception
    app.sidebar.text_input(key="paper_title_input").set_value(TITLE).run()
    assert not app.exception
    return app, start


@pytest.mark.parametrize("stale_action", ["仅准备源码和命令", "准备实验环境"])
@pytest.mark.parametrize("review", [None, False])
def test_actual_title_start_preserves_title_review_and_ignores_stale_preset_options(title_app, stale_action, review):
    app, start = title_app
    review_widget = app.sidebar.checkbox(key="title_llm_review")
    assert review_widget.value is True
    assert any("rtqichen/torchdiffeq" in notice.value for notice in app.success)
    if review is not None:
        review_widget.set_value(review).run()
        assert not app.exception

    # These are selections from a different input mode, not title-mode consent.
    app.session_state["experiment_profile"] = "siren_camera_quick"
    app.session_state["method_action"] = stale_action
    app.session_state["method_optimization"] = "validate"
    app.session_state["method_llm_review"] = False
    app.session_state["repository_prepare_only"] = True
    app.session_state["repository_result_review"] = True
    next(button for button in app.sidebar.button if "开始复现" in button.label).click().run()
    assert not app.exception

    start.assert_called_once()
    request = start.call_args.kwargs
    assert request["paper_title"] == TITLE
    assert request["experiment_profile"] == PROFILE
    assert request["title_resolution"] == RESOLUTION
    assert request["use_llm_review"] is (True if review is None else review)
    assert request["optimization_mode"] == "off"
    assert request["enable_optimization"] is False
    assert request["prepare_only"] is False
    assert request["prepare_environment"] is False
    assert request["allow_result_summary_review"] is False
    assert request["mock_mode"] is False and request["use_docker"] is False
    assert app.session_state["running"] is True
