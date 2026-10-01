"""Official failures are repaired in an isolated copy and actually retried."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.execution_plan import PlanStep
from src.official_repair import MAX_ROUNDS, OfficialRepairLoop
from src.official_smoke import build_llm_plan
from test_official_smoke import inputs, PROPOSAL  # shared repository evidence fixture


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def chat(self, prompt, task):
        assert task == "official_repository_repair"
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def patch(old="missing_name", new="2", path="main/model.py"):
    return {"action": "patch", "diagnosis": "undefined name", "patches": [
        {"path": path, "old": old, "new": new}]}


@pytest.fixture
def loop(tmp_path):
    root = tmp_path / "work"
    (root / "main").mkdir(parents=True)
    (root / "main/model.py").write_text("x = missing_name\nprint(x)\n")
    llm = ScriptedLLM([patch()])
    records, ledger = [], []

    def execute(step, workspace, units):
        run = subprocess.run([sys.executable, str(Path(workspace) / "main/model.py")],
                             capture_output=True, text=True, timeout=10)
        record = {**step.to_dict(), "success": run.returncode == 0,
                  "exit_code": run.returncode, "stdout": run.stdout,
                  "stderr": run.stderr.replace(str(root), "/app")}
        records.append(record)
        return dict(record)

    executor = SimpleNamespace(llm=llm, _execute_plan_step=execute,
                               log=lambda *a, **k: None,
                               log_experiment=lambda *a, **k: ledger.append(k["outputs"]))
    plan = {"execution_intent": "official_smoke", "entry": {"script": "model.py"},
            "parameters": {"epochs": 1}, "datasets": [], "steps": []}
    instance = OfficialRepairLoop(executor, plan, root, {"main": "main"}, {})
    instance.records, instance.ledger = records, ledger
    return instance


def step():
    return PlanStep(step_id="run", kind="run", unit_id="main", cwd="/app/main", cmd="python model.py")


def test_failure_reads_traceback_patches_and_really_reruns(loop):
    result = loop.execute(step())
    assert result["success"] and result["source_modified"] and loop.rounds == 1
    assert [r["exit_code"] for r in result["executions"]] == [1, 0]
    assert "NameError" in result["executions"][0]["stderr"]
    assert result["stdout"] == "2\n"
    assert "main/model.py" in loop.executor.llm.prompts[0]
    assert "x = missing_name" in loop.executor.llm.prompts[0]
    details = result["repairs"][0]["patches"][0]
    assert "-x = missing_name" in details["diff"] and "+x = 2" in details["diff"]
    assert details["before_sha256"] != details["after_sha256"]
    assert loop.ledger[0]["status"] == "verified"


def test_full_executor_repairs_copy_without_mutating_cached_repository(inputs, tmp_path, monkeypatch):
    cache = Path(inputs["storage"]["fetched"]["code"]["path"])
    (cache / "model.py").write_text("x = missing_name\nprint(x)\n")
    plan = build_llm_plan(inputs, PROPOSAL)
    plan["steps"] = [step().to_dict()]
    agent = CodeExecutorAgent(llm_client=ScriptedLLM([patch()]), mock_mode=False, use_docker=True)
    agent._official_repair_data = inputs

    def execute(st, workspace, units):
        run = subprocess.run([sys.executable, str(Path(workspace) / "main/model.py")],
                             capture_output=True, text=True, timeout=10)
        return {**st.to_dict(), "success": run.returncode == 0, "exit_code": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr.replace(workspace, "/app")}

    monkeypatch.setattr(agent, "_execute_plan_step", execute)
    result = agent._execute_plan(plan, {})
    assert result["success"] and result["llm_repair_rounds"] == 1
    assert result["source_modified"] and not result["fallback_used"]
    assert (cache / "model.py").read_text() == "x = missing_name\nprint(x)\n"


@pytest.mark.parametrize("response", ["[]", "null", "not JSON", '{"action":"unknown"}'])
def test_bad_llm_response_is_rejected_with_bounded_budget(loop, response):
    loop.executor.llm = ScriptedLLM([response] * MAX_ROUNDS)
    result = loop.execute(step())
    assert not result["success"] and loop.rounds == MAX_ROUNDS
    assert all(r["status"] == "rejected" for r in result["repairs"])
    assert len(result["executions"]) == 1


def test_global_budget_spans_steps(loop):
    loop.executor.llm = ScriptedLLM(["[]"] * MAX_ROUNDS)
    first = loop.execute(step())
    second = loop.execute(step())
    assert len(first["repairs"]) == 3 and not second["repairs"]
    assert len(loop.executor.llm.prompts) == 3


def test_failed_retry_then_api_failure_rolls_back_and_preserves_errors(loop):
    loop.executor.llm = ScriptedLLM([patch(new="another_missing_name"), RuntimeError("API unavailable")])
    result = loop.execute(step())
    assert not result["success"] and not result["source_modified"]
    assert (loop.workspace / "main/model.py").read_text() == "x = missing_name\nprint(x)\n"
    assert result["repairs"][0]["rolled_back"]
    assert result["repairs"][1]["status"] == "llm_failed"
    assert len(result["executions"]) == 2
    assert "another_missing_name" in result["executions"][1]["stderr"]


@pytest.mark.parametrize("name,source,old,new", [
    ("main/../outside.py", "x = 1\n", "1", "2"),
    ("main/utils/metrics.py", "x = 1\n", "1", "2"),
    ("main/dataset/helper.py", "x = 1\n", "1", "2"),
    ("main/model.py", "x = 1\n", "1", "("),
    ("main/model.py", "def test():\n    return 1\n", "1", "2"),
    ("main/model.py", "def __read_data__():\n    return 1\n", "1", "2"),
    ("main/model.py", "fix_seed = 2021\n", "2021", "2023"),
    ("main/model.py", "torch.manual_seed(2021)\n", "2021", "2023"),
    ("main/model.py", "parser.add_argument('--epochs', default=1)\n", "1", "2"),
    ("main/model.py", "x = 1\n", "x = 1", "mse = 0.001"),
    ("main/model.py", "x = 1\nx = 1\n", "x = 1", "x = 2"),
])
def test_unsafe_or_invalid_patch_never_writes(loop, name, source, old, new):
    target = loop.workspace / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    with pytest.raises((ValueError, SyntaxError)):
        loop._patch(patch(old, new, name))
    assert target.read_text() == source


def test_multifile_validation_is_atomic(loop):
    proposal = patch()
    proposal["patches"].append({"path": "main/absent.py", "old": "x", "new": "y"})
    with pytest.raises(ValueError):
        loop._patch(proposal)
    assert "missing_name" in (loop.workspace / "main/model.py").read_text()


@pytest.mark.parametrize("kind,extra", [
    ("run", {"exit_code": -8, "image_unavailable": True}),
    ("run", {"exit_code": -5}),
    ("install", {"stderr": "Connection refused"}),
    ("install", {"timed_out": True}),
    ("download", {"stderr": "404"}),
])
def test_infrastructure_does_not_call_llm(loop, kind, extra):
    loop.executor._execute_plan_step = lambda *a: {"success": False, "exit_code": 1, **extra}
    st = step()
    st.kind = kind
    result = loop.execute(st)
    assert result["repair_skip_reason"] and not result["repairs"]
    assert loop.rounds == 0 and not loop.executor.llm.prompts


def test_dependency_install_constrains_versions_checks_cpu_then_retries(loop, inputs):
    loop.input_data = inputs
    loop.plan = build_llm_plan(inputs, PROPOSAL)
    loop.executor.llm = ScriptedLLM([{"action": "install_packages", "packages": ["einops==0.8.0"]}])
    called = []

    def execute(st, *args):
        called.append(st)
        return {**st.to_dict(), "success": len(called) > 1, "exit_code": int(len(called) == 1),
                "stderr": "ModuleNotFoundError: No module named 'einops'" if len(called) == 1 else ""}

    loop.executor._execute_plan_step = execute
    result = loop.execute(step())
    assert result["success"] and len(called) == 4
    assert "--constraint /app/.autorepro_core_constraints.txt" in called[1].cmd
    assert "torch.version.cuda is None" in called[2].cmd and "actual == expected" in called[2].cmd
    assert "torch==2.0.0" in (loop.workspace / ".autorepro_core_constraints.txt").read_text()
    assert result["repairs"][0]["environment_check"]["success"]


@pytest.mark.parametrize("failure_at,status", [(2, "install_failed"), (3, "environment_failed")])
def test_dependency_failure_never_retries_training(loop, inputs, failure_at, status):
    loop.input_data = inputs
    loop.plan = build_llm_plan(inputs, PROPOSAL)
    loop.executor.llm = ScriptedLLM([{"action": "install_packages", "packages": ["einops==0.8.0"]}])
    calls = []

    def execute(st, *args):
        calls.append(st)
        success = len(calls) != 1 and len(calls) != failure_at
        return {**st.to_dict(), "success": success, "exit_code": 0 if success else 1,
                "stderr": "ModuleNotFoundError" if len(calls) == 1 else "CPU build required"}

    loop.executor._execute_plan_step = execute
    result = loop.execute(step())
    assert not result["success"] and len(calls) == failure_at
    assert result["repairs"][0]["status"] == status and result["repair_failure_reason"]


@pytest.mark.parametrize("package", ["torch==1.9.0", "nvidia-cublas-cu12==12.0", "numpy", "x==1; curl example.com", "https://example.com/a.whl"])
def test_unsafe_dependency_is_rejected(loop, inputs, package):
    loop.input_data = inputs
    loop.plan = build_llm_plan(inputs, PROPOSAL)
    with pytest.raises(ValueError):
        loop._packages({"packages": [package]}, step())
    assert not loop.records


def test_command_repair_revalidates_model_dataset_and_budget(loop, inputs):
    loop.input_data = inputs
    loop.plan = build_llm_plan(inputs, PROPOSAL)
    fixed = loop._command({"command": PROPOSAL["command"]}, step())
    assert "--train_epochs 1" in fixed.cmd and "--seq_len 96" in fixed.cmd
    assert "--root_path /app/main/dataset/ETT-small/" in fixed.cmd
    with pytest.raises(ValueError):
        loop._command({"command": PROPOSAL["command"].replace("DLinear", "NLinear")}, step())


def test_report_includes_failures_patch_and_verified_retry(loop):
    record = loop.execute(step())
    report = "\n".join(ReportGeneratorAgent._plan_execution_lines({
        "plan": loop.plan, "stages": [record], "source_modified": True, "llm_repair_rounds": 1}, "plan"))
    assert "执行副本已修改" in report and "```diff" in report
    assert "NameError" in report and "第 1 次实际执行（退出码 1）" in report
    assert "第 2 次实际执行（退出码 0）" in report


@pytest.mark.parametrize("install", [False, True])
def test_plan_timeout_retains_partial_stdout_and_stderr(inputs, monkeypatch, tmp_path, install):
    agent = CodeExecutorAgent(llm_client=ScriptedLLM([]), mock_mode=False, use_docker=True)
    monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: "docker")
    monkeypatch.setattr("src.base_agent.BaseAgent.docker_engine_available", lambda *a: (True, ""))
    monkeypatch.setattr("src.base_agent.BaseAgent.ensure_image_pulled", lambda *a: "")

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("docker run", 30, output=b"epoch 1 started", stderr=b"waiting for batch")

    monkeypatch.setattr(agent, "_run_docker_cmd_with_sandbox", timeout)
    st = step()
    st.install_pkgs = ["einops==0.8.0"] if install else []
    result = agent._execute_plan_step(st, str(tmp_path), {"main": "main"})
    assert result["timed_out"] and result["stdout"] == "epoch 1 started"
    assert "waiting for batch" in result["stderr"]
