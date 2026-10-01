"""API official smoke: retain LLM provenance and stop on unusable evidence."""
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.agents.execution_planner import ExecutionPlannerAgent
from src.agents.result_validator import ResultValidatorAgent
from src.audit.audit_logger import AuditLogger
from src.experiment_profiles import PARAMETERS
from src.llm.llm_client import LLMClient
from src.official_smoke import (
    DATA_TARGET, ENTRY, INTENT, REPO_URL, RUNNER,
    build_llm_plan, cpu_environment, repository_context,
)
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager


TITLE = "Are Transformers Effective for Time Series Forecasting?"
SELECTION = {"model": "DLinear", "dataset": "ETTh1", "entry_script": ENTRY,
             "runner": RUNNER, "evidence_files": [ENTRY, RUNNER]}
PROPOSAL = {**SELECTION, "command": (
    "python -u run_longExp.py --is_training 1 --model DLinear --data ETTh1 "
    "--seq_len 336 --pred_len 96 --train_epochs 10 --num_workers 10 "
    "--learning_rate 0.005 --des Exp")}


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    keys = set(PARAMETERS) | {"is_training", "model", "data", "features", "data_path",
                              "root_path", "model_id", "learning_rate", "des"}
    runner = "import argparse\nparser = argparse.ArgumentParser()\n" + "\n".join(
        f"parser.add_argument('--{key}')" for key in sorted(keys))
    runner += ("\nparser.add_argument('--label_len', type=int, default=48)"
               "\nparser.add_argument('--train_only', type=bool, default=False)"
               "\nparser.add_argument('--do_predict', action='store_true')"
               "\nparser.add_argument('--individual', action='store_true', default=False)\n")
    documents = {"README.md": "DLinear uses ETTh1.", "requirements.txt": "torch==1.9.0\n",
                 RUNNER: runner, ENTRY: PROPOSAL["command"], "models/DLinear.py": "# fixture"}
    for name, content in documents.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("fixture; Docker boundary is replaced in these tests")
    code = {"unit_id": "main", "role": "main", "url": REPO_URL, "path": str(root),
            "commit": "a" * 40, "state": "cached"}
    data = {"paper_title": TITLE, "paper_id": "api-test", "execution_intent": INTENT,
            "experiment_profile": "", "mock_mode": False,
            "repository_selection": copy.deepcopy(SELECTION),
            "repository_context": repository_context(code),
            "storage": {"fetched": {"code": code, "units": [code], "dataset": {
                "path": str(dataset), "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "data_kind": "real", "state": "cached", "rows": 17420}}}}
    data["env_config"] = cpu_environment(data["repository_context"])
    return data


def test_llm_command_is_bounded_before_first_run_and_keeps_provenance(inputs, tmp_path):
    plan = build_llm_plan(inputs, copy.deepcopy(PROPOSAL))
    assert plan["source"] == "llm" and plan["execution_intent"] == INTENT
    assert plan["experiment_profile"] == ""
    run = next(step for step in plan["steps"] if step["kind"] == "run")
    command = shlex.split(run["cmd"])
    for key, value in PARAMETERS.items():
        assert command[command.index("--" + key) + 1] == str(value)
    assert command[command.index("--learning_rate") + 1] == "0.005"
    assert run["timeout_s"] == 600 and plan["steps"][0]["timeout_s"] == 1200
    assert plan["parameters"]["label_len"] == 48
    assert plan["parameters"]["use_gpu"] is False
    assert plan["llm_proposal"] == PROPOSAL
    assert plan["policy_overrides"]["--seq_len"] == {"requested": "336", "effective": "96"}
    assert plan["datasets"][0]["target"] == DATA_TARGET
    workspace = tmp_path / "work"
    (workspace / "main").mkdir(parents=True)
    CodeExecutorAgent._bind_plan_datasets(plan, workspace)
    assert (workspace / "main" / DATA_TARGET).read_bytes() == Path(
        inputs["storage"]["fetched"]["dataset"]["path"]).read_bytes()


def test_environment_check_keeps_import_path_for_cpu_assertion(inputs, tmp_path):
    # The Docker executor prefixes the command with PYTHONPATH. It applies to
    # one process, so the version check and torch import must share that process.
    (tmp_path / "torch.py").write_text(
        "__version__ = 'cpu-fixture'\nclass version:\n    cuda = None\n")
    plan = build_llm_plan(inputs, PROPOSAL)
    command = next(step["cmd"] for step in plan["steps"] if step["step_id"] == "environment")
    command = command.replace("python", shlex.quote(sys.executable), 1)
    result = subprocess.run("PYTHONPATH=" + shlex.quote(str(tmp_path)) + " " + command,
                            shell=True, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Torch cpu-fixture CUDA None" in result.stdout


@pytest.mark.parametrize("change", [
    {"model": "NLinear"}, {"evidence_files": ["invented.sh"]},
    {"command": "python -u made_up.py --model DLinear --data ETTh1"},
    {"command": PROPOSAL["command"] + " && curl https://example.com"},
    {"command": PROPOSAL["command"] + " --seq_len 96"},
    {"command": PROPOSAL["command"] + " --unknown 1"},
    {"command": PROPOSAL["command"] + " --individual False"},
    {"command": PROPOSAL["command"] + " --train_only True"},
])
def test_invalid_llm_proposal_cannot_become_preset(inputs, change):
    with pytest.raises(ValueError):
        build_llm_plan(inputs, {**PROPOSAL, **change})


@pytest.mark.parametrize("part,change", [
    ("dataset", {"data_kind": "synthetic"}), ("dataset", {"state": "unavailable"}),
    ("code", {"commit": ""}), ("code", {"state": "clone-failed"}),
])
def test_plan_rejects_unprepared_resources(inputs, part, change):
    inputs["storage"]["fetched"][part].update(change)
    with pytest.raises(ValueError):
        build_llm_plan(inputs, PROPOSAL)


@pytest.mark.parametrize("architecture,torch", [
    ("arm64", "torch==2.0.0"), ("aarch64", "torch==2.0.0"),
    ("x86_64", "torch==2.0.0+cpu"), ("amd64", "torch==2.0.0+cpu"),
])
def test_environment_uses_cpu_build(inputs, monkeypatch, architecture, torch):
    monkeypatch.setattr("src.official_smoke.platform.machine", lambda: architecture)
    env = cpu_environment(inputs["repository_context"])
    assert torch in env["requirements_txt"].splitlines()
    assert "scipy==1.15.3" in env["requirements_txt"]
    assert env["upstream_requirements_txt"] == "torch==1.9.0\n"


def test_repository_context_rejects_missing_or_external_evidence(inputs, tmp_path):
    code = inputs["storage"]["fetched"]["code"]
    entry = Path(code["path"]) / ENTRY
    entry.unlink()
    with pytest.raises(ValueError, match="证据文件"):
        repository_context(code)
    other = tmp_path / "outside.sh"
    other.write_text("not repository evidence")
    entry.symlink_to(other)
    with pytest.raises(ValueError, match="证据文件"):
        repository_context(code)


class ScriptedLLM:
    """Only API responses are substituted; real reader/finder/planner still run."""
    mock_mode = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_call_count(self):
        return len(self.calls)

    def chat(self, prompt, task=None, **kwargs):
        self.calls.append(task)
        response = self.responses[task]
        if isinstance(response, list):
            response = response.pop(0)
        return json.dumps(response)


def test_planner_corrects_once_then_stops_without_fallback(inputs):
    llm = ScriptedLLM({"execution_planner": [{"error": "cannot plan"}, {"error": "still invalid"}]})
    with pytest.raises(ValueError, match="不可执行"):
        ExecutionPlannerAgent(llm).run(inputs)
    assert llm.calls == ["execution_planner", "execution_planner"]


def make_orchestrator(inputs, tmp_path, monkeypatch, clone_failure=False):
    fetched = inputs["storage"]["fetched"]
    llm = ScriptedLLM({"paper_reader": {"title": TITLE, "insufficient_info": True},
                      "resource_finder": {"code_repo_url": REPO_URL, "confidence": 0.9},
                      "repository_reader": SELECTION, "execution_planner": PROPOSAL})
    monkeypatch.setattr("src.agents.resource_finder.discover_repositories", lambda *a, **kw: {
        "selected_repo": REPO_URL, "discovery_chain": ["github_search"],
        "code_units": [{"unit_id": "main", "role": "main", "url": REPO_URL}]})
    manager = ResourceManager(str(tmp_path / "data"))
    def fetch_units(*args, **kwargs):
        assert kwargs["verify_repository"] is True
        return [{**fetched["code"], "state": "clone-failed", "commit": "", "detail": "HTTP2"}] \
            if clone_failure else fetched["units"]
    monkeypatch.setattr(manager, "fetch_units", fetch_units)
    monkeypatch.setattr(manager, "fetch_dataset", lambda *a, **kw: fetched["dataset"])
    logger = AuditLogger(str(tmp_path / "logs"), str(tmp_path / "ledger"))
    orch = Orchestrator(llm, mock_mode=False, use_docker=True, resource_manager=manager, logger=logger)
    monkeypatch.setattr(orch.agents["verifier"], "run", lambda *a: pytest.fail("No generic verifier retry"))
    monkeypatch.setattr(orch.agents["executor"], "_execute_plan_step_with_repair",
                        lambda *a: pytest.fail("Infrastructure cannot spend repair budget"))
    return orch, llm


def test_api_pipeline_reads_repo_and_executes_llm_plan(inputs, tmp_path, monkeypatch):
    orch, llm = make_orchestrator(inputs, tmp_path, monkeypatch)
    commands = []
    def docker_boundary(step, workspace, units):
        commands.append(step.cmd)
        assert (Path(workspace) / "main" / DATA_TARGET).is_file()
        return {"step_id": step.step_id, "kind": step.kind, "cmd": step.cmd,
                "success": True, "exit_code": 0,
                "stdout": "mse:0.409, mae:0.417" if step.kind == "run" else "Python 3.11", "stderr": ""}
    monkeypatch.setattr(orch.agents["executor"], "_execute_plan_step", docker_boundary)
    result = orch.run({"paper_title": TITLE, "use_llm_pipeline": True})
    assert result["state"] == "COMPLETED", result.get("error")
    data = result["data"]
    assert llm.calls == ["paper_reader", "resource_finder", "repository_reader", "execution_planner"]
    assert data["total_llm_calls"] == 4 and result["audit_stats"]["llm_calls"] == 4
    assert data["experiment_profile"] == "" and data["execution_plan"]["source"] == "llm"
    assert data["validation"]["status"] == "smoke_verified"
    assert data["validation"]["is_reproduced"] is None
    assert data["optimization"]["optimized"] is False
    assert len(commands) == 3
    evidence = Path(data["execution"]["evidence_dir"])
    assert json.loads((evidence / "context.json").read_text())["repository_selection"] == SELECTION
    assert "LLM 提出的命令" in data["report"] and "尚未核对论文数值" in data["report"]


def test_clone_failure_stops_api_pipeline_and_leaves_failed_report(inputs, tmp_path, monkeypatch):
    orch, llm = make_orchestrator(inputs, tmp_path, monkeypatch, clone_failure=True)
    monkeypatch.setattr(orch.agents["builder"], "run", lambda *a: pytest.fail("Clone failed"))
    result = orch.run({"paper_title": TITLE, "use_llm_pipeline": True})
    assert result["state"] == "ERROR"
    data = result["data"]
    assert llm.calls == ["paper_reader", "resource_finder"]
    assert data["validation"]["status"] == "smoke_failed"
    assert data["execution"]["failure_stage"] == "FIND_RESOURCES"
    assert data["execution"]["fallback_used"] is False
    assert "HTTP2" in data["report"] and data["report"]
    assert (Path(data["execution"]["evidence_dir"]) / "execution.json").is_file()


@pytest.mark.parametrize("mutation", ["no_run", "synthetic", "fallback", "heuristic", "nonfinite"])
def test_api_verdict_requires_real_llm_execution(mutation):
    execution = {"execution_mode": "plan", "success": True,
                 "plan": {"source": "llm"}, "fallback_used": False,
                 "datasets": [{"data_kind": "real", "sha256": "verified"}],
                 "stages": [{"kind": "run", "success": True, "exit_code": 0}],
                 "final": {"stdout": "mse:0.4, mae:0.41", "exit_code": 0}}
    if mutation == "no_run":
        execution["stages"] = []
    elif mutation == "synthetic":
        execution["datasets"][0]["data_kind"] = "synthetic"
    elif mutation == "fallback":
        execution["fallback_used"] = True
    elif mutation == "heuristic":
        execution["plan"]["source"] = "heuristic"
    else:
        execution["final"]["stdout"] = "mse:nan, mae:inf"
    result = ResultValidatorAgent(LLMClient(mock_mode=False)).run({
        "execution_intent": INTENT, "execution": execution})
    assert result["status"] == "smoke_failed" and result["is_reproduced"] is None


def test_install_timeout_never_repairs_or_generates_replacement(inputs, monkeypatch):
    executor = CodeExecutorAgent(LLMClient(mock_mode=False), use_docker=True)
    monkeypatch.setattr(executor, "_execute_plan_step_with_repair", lambda *a: pytest.fail("No repair"))
    monkeypatch.setattr(executor.llm, "chat", lambda *a, **kw: pytest.fail("No generated fallback"))
    calls = []
    def boundary(step, *args):
        calls.append(step.kind)
        return {"step_id": step.step_id, "kind": step.kind, "success": False,
                "exit_code": -1, "timed_out": True, "stdout": "", "stderr": "install timed out"}
    monkeypatch.setattr(executor, "_execute_plan_step", boundary)
    inputs["execution_plan"] = build_llm_plan(inputs, PROPOSAL)
    result = executor.run(inputs)
    assert calls == ["install"]
    assert result["success"] is False and result["fallback_used"] is False
    assert result["failure_stage"] == "install_main"
    assert result["stages"][0]["timed_out"] is True
    assert "duration_sec" in result["stages"][0]


def init_repo(path, url=REPO_URL):
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    path.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("remote", "add", "origin", url)
    (path / "source.py").write_text("# tracked official fixture\n")
    git("add", "source.py")
    git("commit", "-qm", "fixture")
    return git("rev-parse", "HEAD")


def test_cache_requires_own_head_origin_revision_and_clean_source(tmp_path):
    repo = tmp_path / "repo"
    commit = init_repo(repo)
    verify = ResourceManager._verified_repo_commit
    assert verify(repo, REPO_URL + ".git", commit) == commit
    nested = repo / "nested"
    nested.mkdir()
    assert verify(nested, REPO_URL) == ""
    assert verify(repo, "https://github.com/other/repo") == ""
    assert verify(repo, REPO_URL, "b" * 40) == ""
    (repo / "source.py").write_text("# user modified\n")
    assert verify(repo, REPO_URL) == ""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    subprocess.run(["git", "-C", str(incomplete), "init", "-q"], check=True)
    assert verify(incomplete, REPO_URL) == ""


def test_failed_clone_preserves_invalid_cache(tmp_path, monkeypatch):
    repo = tmp_path / "cache"
    repo.mkdir()
    (repo / "user_file").write_text("must survive")
    calls = []
    def clone(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "HTTP/2 stream reset")
    monkeypatch.setattr("src.resource_manager.subprocess.run", clone)
    result = ResourceManager(str(tmp_path / "data")).fetch_code(
        "p", REPO_URL, target=str(repo), verify_repository=True)
    assert result["state"] == "clone-failed" and result["attempts"] == 2
    assert (repo / "user_file").read_text() == "must survive"
    assert "http.version=HTTP/1.1" in [cmd for cmd in calls if cmd[:2] == ["git", "clone"]][-1]
    assert not list(tmp_path.glob(".cache-clone-*"))


def test_transient_clone_retry_promotes_only_verified_repo(tmp_path, monkeypatch):
    repo = tmp_path / "cache"
    repo.mkdir()
    (repo / "user_file").write_text("preserve invalid prior cache")
    real_run = subprocess.run
    calls = []
    def boundary(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            calls.append(cmd)
            if len(calls) == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "HTTP2 early EOF")
            path = Path(cmd[-1])
            (path / ".git").mkdir()
            (path / "official.py").write_text("verified clone fixture")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, **kwargs)
    monkeypatch.setattr("src.resource_manager.subprocess.run", boundary)
    manager = ResourceManager(str(tmp_path / "data"))
    monkeypatch.setattr(manager, "_verified_repo_commit", lambda path, *a: "a" * 40
                        if (path / "official.py").exists() else "")
    monkeypatch.setattr(manager, "_pin_and_record", lambda *a: None)
    result = manager.fetch_code("p", REPO_URL, target=str(repo), verify_repository=True)
    assert result["state"] == "cloned" and result["attempts"] == 2
    assert (repo / "official.py").is_file()
    assert (Path(result["previous_cache"]) / "user_file").is_file()
    assert "http.version=HTTP/1.1" in calls[-1]
    assert not list(tmp_path.glob(".cache-clone-*"))
