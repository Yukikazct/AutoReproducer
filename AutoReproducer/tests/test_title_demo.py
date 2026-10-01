"""Title -> GitHub discovery -> official plan -> real-data verdict wiring.

Docker is replaced only at the process boundary. These tests do not claim to
validate the paper's numerical results; a real Docker run is user acceptance.
"""
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src import etth1
from src.agents.code_executor import CodeExecutorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.experiment_profiles import (
    PROFILES, PROFILE_ID, PARAMETERS, resolve_profile, build_plan,
)
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager


@pytest.fixture
def csv_bytes(monkeypatch):
    start = datetime(2016, 7, 1)
    content = (",".join(etth1.HEADER) + "\n" + "\n".join(
        f"{start + timedelta(hours=i)},1,2,3,4,5,6,7" for i in range(17420)) + "\n").encode()
    monkeypatch.setattr(etth1, "GIT_BLOB_SHA", hashlib.sha1(
        f"blob {len(content)}\0".encode() + content).hexdigest())
    return content


@pytest.mark.parametrize("title,expected", [
    ("iTransformer", PROFILE_ID),
    (PROFILES[PROFILE_ID]["title"], PROFILE_ID),
    ("Are Transformers Effective for Time Series Forecasting?", "dlinear_etth1_cpu_smoke"),
    ("DLinear", "dlinear_etth1_cpu_smoke"),
    ("NLinear", "nlinear_etth1_cpu_smoke"),
    ("Attention Is All You Need", ""),
])
def test_title_selects_adapter_without_preset(title, expected):
    assert resolve_profile(title) == expected


def test_data_download_fallback_validation_and_cache(tmp_path, monkeypatch, csv_bytes):
    calls = []

    def download(cmd, **kwargs):
        calls.append(cmd[-1])
        if cmd[-1] == etth1.DATA_URL:
            return subprocess.CompletedProcess(cmd, 1, "", "network unavailable")
        Path(cmd[cmd.index("--output") + 1]).write_bytes(csv_bytes)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(etth1.subprocess, "run", download)
    result = etth1.fetch_etth1(tmp_path)
    assert result["state"] == "downloaded"
    assert result["rows"] == 17420
    assert result["sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert calls == [etth1.DATA_URL, etth1.API_URL]
    assert etth1.fetch_etth1(tmp_path)["state"] == "cached"
    assert len(calls) == 2
    Path(result["path"]).write_text("corrupted")
    assert etth1.fetch_etth1(tmp_path)["state"] == "downloaded"
    assert len(calls) == 4
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_is_not_promoted(tmp_path, monkeypatch):
    def download(cmd, **kwargs):
        Path(cmd[cmd.index("--output") + 1]).write_text("<html>not CSV</html>")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(etth1.subprocess, "run", download)
    result = etth1.fetch_etth1(tmp_path)
    assert result["state"] == "unavailable" and not result["path"]
    assert not (tmp_path / "ETTh1.csv").exists()
    assert not list(tmp_path.glob("*.part"))


def test_real_dataset_ignores_synthetic_cache(tmp_path, monkeypatch, csv_bytes):
    manager = ResourceManager(str(tmp_path))
    old = tmp_path / "datasets" / "p" / "dataset_smoke"
    old.mkdir(parents=True)
    (old / "samples.csv").write_text("synthetic")
    def download(cmd, **kwargs):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(csv_bytes)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(etth1.subprocess, "run", download)
    result = manager.fetch_dataset("p", "ETTh1", level="smoke")
    assert result["data_kind"] == "real" and result["rows"] == 17420
    assert manager.fetch_dataset("p", "ETTh1", level="full")["state"] == "cached"
    assert [e["state"] for e in manager.resource_events.list()] == [
        "running", "succeeded", "running", "cached"]


def test_mock_skips_even_cached_official_resources(tmp_path, monkeypatch):
    manager = ResourceManager(str(tmp_path), mock_mode=True)
    repo = tmp_path / "repos" / "p"
    repo.mkdir(parents=True)
    (repo / "run.py").write_text("raise RuntimeError('never execute')")
    monkeypatch.setattr("src.resource_manager.subprocess.run",
                        lambda *a, **kw: pytest.fail("Mock attempted an external command"))
    assert manager.fetch_code("p", PROFILES[PROFILE_ID]["repo"])["state"] == "mock-skipped"
    assert manager.fetch_dataset("p", "ETTh1")["state"] == "mock-skipped"
    assert manager.fetch_weights("p", "https://example.net/model.bin")["state"] == "mock-skipped"


def fixture_inputs(tmp_path, profile_id=PROFILE_ID):
    profile = PROFILES[profile_id]
    root = tmp_path / "repo"
    (root / profile["entry"]).parent.mkdir(parents=True, exist_ok=True)
    (root / profile["entry"]).write_text(
        f"model_name={profile['model']}\npython -u {profile['runner']} "
        "--model $model_name --seq_len 336 --pred_len 96 --enc_in 7 "
        "--root_path ./dataset/ >logs/output.log\n"
        f"python -u {profile['runner']} --pred_len 192\n")
    (root / profile["runner"]).write_text("# official code fixture\n")
    (root / "requirements.txt").write_text("numpy==1.23.5\n")
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("test fixture; not used for real training")
    code = {"unit_id": "main", "path": str(root), "state": "cached",
            "commit": profile["revision"]}
    data = {"experiment_profile": profile_id, "paper_id": profile_id,
            "storage": {"fetched": {"code": code, "units": [code], "dataset": {
                "path": str(dataset), "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "data_kind": "real", "state": "cached", "rows": 17420}}}}
    return data


@pytest.mark.parametrize("profile_id", list(PROFILES))
def test_each_demo_has_one_cpu_training_command(tmp_path, profile_id):
    data = fixture_inputs(tmp_path, profile_id)
    plan = build_plan(data)
    run = [s for s in plan["steps"] if s["kind"] == "run"]
    assert len(run) == 1
    tokens = shlex.split(run[0]["cmd"])
    assert tokens.count("--pred_len") == 1
    assert "192" not in tokens and not any(t.startswith(">") for t in tokens)
    assert tokens[tokens.index("--model") + 1] == PROFILES[profile_id]["model"]
    for key, value in PARAMETERS.items():
        assert tokens[tokens.index("--" + key) + 1] == str(value)
    assert run[0]["timeout_s"] == 600
    assert plan["steps"][0]["timeout_s"] == 1200
    assert plan["datasets"][0]["target"] == "dataset/ETT-small/ETTh1.csv"


def test_dataset_binding_checks_hash_and_target(tmp_path):
    plan = build_plan(fixture_inputs(tmp_path))
    workspace = tmp_path / "workspace"
    (workspace / "main").mkdir(parents=True)
    CodeExecutorAgent._bind_plan_datasets(plan, workspace)
    assert (workspace / "main/dataset/ETT-small/ETTh1.csv").is_file()
    plan["datasets"][0]["target"] = "../../escaped.csv"
    with pytest.raises(ValueError, match="非法"):
        CodeExecutorAgent._bind_plan_datasets(plan, workspace)
    plan["datasets"][0]["target"] = "dataset/ETTh1.csv"
    plan["datasets"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA-256"):
        CodeExecutorAgent._bind_plan_datasets(plan, workspace)


def test_unprepared_profile_preserves_official_plan_failure():
    executor = CodeExecutorAgent(LLMClient(mock_mode=False), use_docker=True)
    result = executor.run({"experiment_profile": PROFILE_ID,
                           "execution_plan": {"steps": [],
                                              "notes": ["真实 ETTh1 不可用"]}})
    assert result["execution_mode"] == "plan"
    assert result["success"] is False
    assert result["fallback_used"] is False
    assert "真实 ETTh1 不可用" in result["final"]["stderr"]


@pytest.mark.parametrize("stdout,success,status", [
    ("mse:0.5, mae:0.4", True, "smoke_verified"),
    ("mse:0.5, mae:0.4", False, "smoke_failed"),
    ("mse:0.5", True, "smoke_failed"),
    ("mse:1e309, mae:0.4", True, "smoke_failed"),
    ("mse:0.5, mae:0.4\nmse:nan, mae:inf", True, "smoke_failed"),
])
def test_smoke_verdict_requires_actual_finite_metrics(stdout, success, status):
    validator = ResultValidatorAgent(LLMClient(mock_mode=False))
    result = validator.run({"experiment_profile": PROFILE_ID, "execution": {
        "execution_mode": "plan", "success": success,
        "datasets": [{"data_kind": "real", "sha256": "verified by executor"}],
        "final": {"stdout": stdout, "exit_code": 0 if success else 1}}})
    assert result["status"] == status
    assert result["is_reproduced"] is None


def test_title_pipeline_searches_github_and_records_execution(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path, "dlinear_etth1_cpu_smoke")
    fetched = inputs["storage"]["fetched"]
    searches = []
    def discover(query, **kwargs):
        searches.append((query, kwargs))
        return {"query": query, "discovery_chain": ["github_search"], "candidates": [
            {"repo_urls": [PROFILES["dlinear_etth1_cpu_smoke"]["repo"]], "source": "github_search"}]}
    monkeypatch.setattr("src.agents.repo_discovery.discover_repositories", discover)
    manager = ResourceManager(str(tmp_path / "data"))
    monkeypatch.setattr(manager, "fetch_units", lambda *a: fetched["units"])
    monkeypatch.setattr(manager, "fetch_dataset", lambda *a, **kw: fetched["dataset"])
    llm = LLMClient(mock_mode=False)
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: pytest.fail("Known demo must not need LLM"))
    orch = Orchestrator(llm, mock_mode=False, use_docker=True, resource_manager=manager)
    commands = []
    def docker_boundary(step, workspace, units):
        commands.append(step.cmd)
        assert (Path(workspace) / "main/dataset/ETT-small/ETTh1.csv").is_file()
        return {"step_id": step.step_id, "kind": step.kind, "cmd": step.cmd,
                "success": True, "exit_code": 0,
                "stdout": "mse:0.5, mae:0.4" if step.kind == "run" else "", "stderr": ""}
    monkeypatch.setattr(orch.agents["executor"], "_execute_plan_step", docker_boundary)
    result = orch.run({"paper_title": "Are Transformers Effective for Time Series Forecasting?"})
    assert result["state"] == "COMPLETED", result.get("error")
    data = result["data"]
    assert searches[0][1]["github_first"] is True
    assert data["validation"]["status"] == "smoke_verified"
    assert data["optimization"]["optimized"] is False
    assert len(commands) == 3  # install + environment + one training, no LLM retry
    record = Path(data["execution"]["evidence_dir"]) / "execution.json"
    assert json.loads(record.read_text())["execution_mode"] == "plan"
    assert "github_search" in data["report"] and "尚未核对论文数值" in data["report"]


def test_demo_failure_does_not_generate_replacement_code(tmp_path, monkeypatch):
    data = fixture_inputs(tmp_path)
    data["execution_plan"] = build_plan(data)
    llm = LLMClient(mock_mode=False)
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: pytest.fail("Must not fabricate replacement"))
    executor = CodeExecutorAgent(llm, use_docker=True)
    monkeypatch.setattr(executor, "_execute_plan", lambda *a: {
        "execution_mode": "plan", "plan_failed_irreparably": True,
        "success": False, "plan_fail_reason": "Docker unavailable"})
    result = executor.run(data)
    assert result["execution_mode"] == "plan" and result["success"] is False


def test_demo_selector_fills_title_and_respects_edits(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import frontend.backend_pipeline as backend
    import frontend.history_manager as history
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    submissions = []
    monkeypatch.setattr(backend, "run_pipeline_background",
                        lambda *args, **kw: submissions.append(kw))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.session_state["docker_probe"] = (True, None)
    app.run()
    app.sidebar.toggle[0].set_value(False).run()
    app.selectbox(key="experiment_profile").set_value("dlinear_etth1_cpu_smoke").run()
    assert not app.exception
    assert app.text_input(key="paper_title_input").value == PROFILES["dlinear_etth1_cpu_smoke"]["title"]
    # Manually changing the title supersedes a previously selected preset.
    app.text_input(key="paper_title_input").set_value("iTransformer").run()
    start = next(b for b in app.button if "开始" in b.label)
    start.click().run()
    assert not app.exception
    assert submissions[0]["paper_title"] == "iTransformer"
    assert submissions[0]["experiment_profile"] == PROFILE_ID
    assert submissions[0]["use_docker"] is True


def test_github_first_search_does_not_skip_network_discovery(monkeypatch):
    from src.agents import repo_discovery as discovery
    calls = []
    repo = PROFILES[PROFILE_ID]["repo"]
    def github(query, **kwargs):
        calls.append(query)
        return [discovery.RepoCandidate(title="iTransformer", repo_urls=[repo],
                                       source="github_search", score_hint=100)], ""
    monkeypatch.setattr(discovery, "_search_github", github)
    monkeypatch.setattr(discovery, "_search_pwc", lambda *a, **kw: pytest.fail("GitHub already found it"))
    result = discovery.discover_repositories("iTransformer", github_first=True)
    assert calls == ["iTransformer"]
    assert result["selected_repo"] == repo
    assert result["discovery_chain"] == ["github_search"]


def test_title_cli_checks_docker_without_requiring_api_key(monkeypatch):
    from scripts.real_e2e import main
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("src.base_agent.BaseAgent.docker_engine_available",
                        lambda *a: (False, "daemon stopped"))
    with pytest.raises(SystemExit, match="Docker Desktop"):
        main(["--paper-title", "DLinear"])


def test_llm_pipeline_bypasses_known_title_adapter(tmp_path, monkeypatch):
    calls = []
    llm = LLMClient(mock_mode=False)
    def chat(prompt, **kwargs):
        calls.append(kwargs.get("task"))
        return json.dumps({"title": "DLinear", "method": "linear",
                           "metrics": {}, "insufficient_info": True})
    monkeypatch.setattr(llm, "chat", chat)
    orch = Orchestrator(llm, mock_mode=False,
                        resource_manager=ResourceManager(str(tmp_path)))
    monkeypatch.setattr(orch, "_verify_step", lambda *a: None)
    def stop_after_reader(data):
        raise RuntimeError("stop after API reader")
    monkeypatch.setattr(orch.agents["finder"], "run", stop_after_reader)
    result = orch.run({"paper_title": "DLinear", "use_llm_pipeline": True})
    assert calls == ["paper_reader"]
    assert result["data"]["experiment_profile"] == ""
    assert result["data"]["paper_info"]["method"] == "linear"


def test_llm_cli_requires_key_even_for_known_title(monkeypatch):
    from scripts.real_e2e import main
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        main(["--paper-title", "DLinear", "--llm-pipeline"])


def test_llm_cli_does_not_report_placeholder_success(tmp_path, monkeypatch):
    from scripts import real_e2e
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-used")
    monkeypatch.setattr("src.base_agent.BaseAgent.docker_engine_available",
                        lambda *a: (True, None))
    class Pipeline:
        def __init__(self, **kwargs):
            pass
        def run(self, payload):
            assert payload["use_llm_pipeline"] is True
            assert "experiment_profile" not in payload
            return {"state": "COMPLETED", "data": {
                "execution": {"success": True, "best_effort": True},
                "validation": {"is_reproduced": None}, "report": "placeholder"}}
    monkeypatch.setattr(real_e2e, "Orchestrator", Pipeline)
    assert real_e2e.main(["--paper-title", "DLinear", "--llm-pipeline",
                          "--workspace", str(tmp_path / "ws"),
                          "--report-out", str(tmp_path / "report.md")]) == 1


def test_demo_timeout_stops_only_its_container(monkeypatch):
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), use_docker=True)
    executor.experiment_profile = PROFILE_ID
    commands = []
    def timeout(base, image, runner, budget):
        commands.append(base)
        raise subprocess.TimeoutExpired(base, budget)
    monkeypatch.setattr(executor, "_run_docker_cmd_with_sandbox_impl", timeout)
    monkeypatch.setattr("src.agents.code_executor.subprocess.run",
                        lambda command, **kw: commands.append(command))
    with pytest.raises(subprocess.TimeoutExpired):
        executor._run_docker_cmd_with_sandbox(["docker", "run", "--rm"],
                                             "python:3.11-slim", ["python", "run.py"], 600)
    name = commands[0][commands[0].index("--name") + 1]
    assert name.startswith("autorepro-demo-")
    assert commands[1] == ["docker", "rm", "--force", name]


def test_interrupted_github_response_returns_diagnostic(monkeypatch):
    import http.client
    from src.agents.repo_discovery import _http_json
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            raise http.client.IncompleteRead(b"partial", 10)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response())
    value, error = _http_json("https://api.github.com/search/repositories?q=demo", 1)
    assert value is None and "IncompleteRead" in error
