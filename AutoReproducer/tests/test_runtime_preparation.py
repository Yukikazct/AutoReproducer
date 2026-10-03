"""Regression for guessed heavy dependencies consuming the smoke deadline."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
from src.agents.dependency_resolver import align_runtime_requirements
from src.agents.code_executor import CodeExecutorAgent
from src.llm.llm_client import LLMClient
from src.base_agent import BaseAgent
from src.sandbox_timeout import MARKER


SCRIPT = "import numpy as np\nimport matplotlib.pyplot as plt\nprint(np.mean([1,2]))\n"
GUESSES = "torch>=2.0\ntransformers>=4.38\nnumpy>=1.24\nopenai>=1.12"


def test_final_script_replaces_unused_guesses_and_adds_missing_matplotlib():
    env = {"requirements_txt": GUESSES, "static_source": "none"}
    result = align_runtime_requirements(env, SCRIPT, ["numpy"])
    assert result["requirements_txt"] == "numpy>=1.24\nmatplotlib"
    assert result["runtime_dependency_selection"]["removed"] == [
        "torch>=2.0", "transformers>=4.38", "openai>=1.12"]
    assert result["runtime_dependency_selection"]["added"] == ["matplotlib"]
    assert env["requirements_txt"] == GUESSES


@pytest.mark.parametrize("env,preserve,code", [
    ({"requirements_txt": GUESSES}, False, SCRIPT),
    ({"requirements_txt": GUESSES, "dependency_source": "corpus"}, False, SCRIPT),
    ({"requirements_txt": GUESSES, "static_source": "none"}, True, SCRIPT),
    ({"requirements_txt": GUESSES, "static_source": "repo"}, False, SCRIPT),
    ({"requirements_txt": GUESSES, "static_source": "none"}, False,
     SCRIPT + "import importlib\nimportlib.import_module('torch')\n"),
])
def test_authoritative_or_dynamic_environment_keeps_declared_packages(env, preserve, code):
    result = align_runtime_requirements(env, code, preserve_all=preserve)
    assert "torch>=2.0" in result["requirements_txt"]
    assert "transformers>=4.38" in result["requirements_txt"]
    assert "matplotlib" in result["requirements_txt"]


def supervised(phase, timeout, code):
    runner = Path(ce.__file__).resolve().parents[1] / "sandbox_timeout.py"
    return subprocess.run([sys.executable, str(runner), phase, str(timeout),
                           sys.executable, "-u", "-c", code],
                          capture_output=True, text=True, timeout=5)


def test_installation_can_take_longer_than_execution_deadline():
    prepared = supervised("dependencies", 2, "import time; time.sleep(0.3)")
    assert prepared.returncode == 0
    executed = supervised("execution", 0.2, "print('SCRIPT_RAN')")
    assert executed.returncode == 0
    assert "SCRIPT_RAN" in executed.stdout


@pytest.mark.parametrize("phase", ["dependencies", "execution"])
def test_timeout_reports_phase_and_preserves_partial_output(phase):
    result = supervised(phase, 0.2, "import time; print('PARTIAL'); time.sleep(1)")
    assert result.returncode == 124
    assert "PARTIAL" in result.stdout
    stderr, info = CodeExecutorAgent._decode_docker_phase(result.stderr)
    assert info["phase"] == phase
    assert info["timeout"] is True
    assert MARKER not in stderr


def test_docker_dependency_timeout_is_not_mislabeled_as_code_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseAgent, "docker_engine_available",
                        staticmethod(lambda *a, **k: (True, None)))
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), use_docker=True)
    executor.env_config = {"requirements_txt": "numpy"}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    calls = []
    def timed_out(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 124, stdout="Installing numpy\n",
            stderr=MARKER + '{"phase":"dependencies","timeout":true,"seconds":300,"returncode":124}\n')
    monkeypatch.setattr(ce.subprocess, "run", timed_out)
    result = executor._execute_code_docker(SCRIPT, "smoke", str(tmp_path))
    assert result["exit_code"] == ce.EXIT_DEPENDENCY_FAILED
    assert result["execution_phase"] == "dependencies"
    assert "代码尚未执行" in result["stderr"]
    assert "Installing numpy" in result["stdout"]
    assert calls[0][1]["timeout"] > ce.DOCKER_TIMEOUT_SMOKE + ce.DOCKER_PIP_TIMEOUT
    assert MARKER not in result["stderr"]
