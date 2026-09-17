"""危险代码静态门测试：本地无沙箱执行前拦截命令执行/动态执行/网络/递归删除。

覆盖：
1. `_dangerous_constructs` 对危险片段返回原因，对良性训练代码不误报；
2. `run()` 遇危险代码短路为 not_runnable，且不真正调用 subprocess；
3. `_execute_code_local` 遇危险代码返回 danger_blocked 且不执行。

运行: python -m pytest tests/test_code_executor_safety.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent,
    EXIT_DANGER_BLOCKED,
    EXIT_NOT_RUNNABLE,
)
from src.llm.llm_client import LLMClient  # noqa: E402


# ---------------- 1. _dangerous_constructs 命中 / 不误报 ----------------

DANGEROUS_SNIPPETS = [
    "import subprocess\nsubprocess.run(['id'])",
    "import os\nos.system('rm -rf /')",
    "os.popen('whoami')",
    "os.spawnv(os.P_WAIT, 'cmd')",
    "import pty",
    "eval('1+1')",
    "exec('import os')",
    "__import__('os').system('id')",
    "import socket\ns = socket.socket()",
    "import requests\nrequests.get('http://evil')",
    "import urllib.request",
    "import http.client",
    "import ftplib",
    "import smtplib",
    "import paramiko",
    "import httpx",
    "import aiohttp",
    "import shutil\nshutil.rmtree('/tmp')",
]


@pytest.mark.parametrize("snippet", DANGEROUS_SNIPPETS)
def test_dangerous_constructs_detected(snippet):
    assert CodeExecutorAgent._dangerous_constructs(snippet) is not None


BENIGN_SNIPPETS = [
    "print('Training complete. Test accuracy: 85.2%')",
    "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.sum())",
    "data = open('data.csv').read()",
    "import os\nprint(os.getcwd())\nprint(os.path.join('a', 'b'))",
    "import torch\nmodel = torch.compile(model)",      # compile 不误报
    "def train(epochs=3):\n    return epochs\n\n"
    "if __name__ == '__main__':\n    train()",
    "import shutil\nshutil.copy('a', 'b')",          # copy 非 rmtree
]


@pytest.mark.parametrize("snippet", BENIGN_SNIPPETS)
def test_dangerous_constructs_not_false_positive(snippet):
    assert CodeExecutorAgent._dangerous_constructs(snippet) is None


# ---------------- 2. run() 短路 + 不真正执行 ----------------

def test_run_blocks_dangerous_code(monkeypatch):
    """语法合法但含 os.system 的代码在 run() 即被拦下，subprocess 不被调用。"""
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    code = "import os\nos.system('echo pwned')\n"

    def _boom(*a, **kw):
        raise AssertionError("危险代码不得进入执行")

    monkeypatch.setattr(ce_mod.subprocess, "run", _boom)
    result = agent.run({"code": code, "paper_info": {}})

    assert result["not_runnable"] is True
    assert "危险" in result["reason"]
    assert result["final"]["exit_code"] == EXIT_NOT_RUNNABLE


# ---------------- 3. _execute_code_local 兜底拦截 ----------------

def test_execute_local_blocks_danger(monkeypatch):
    """Optimizer 真实执行绕过 run() 直入 _execute_code_local，仍需被拦。"""
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    code = "import subprocess\nsubprocess.run(['echo', 'hi'])\n"

    def _boom(*a, **kw):
        raise AssertionError("危险代码不得执行")

    monkeypatch.setattr(ce_mod.subprocess, "run", _boom)
    result = agent._execute_code_local(code, stage="smoke")

    assert result["success"] is False
    assert result["danger_blocked"] is True
    assert result["exit_code"] == EXIT_DANGER_BLOCKED


def test_execute_local_benign_code_still_runs(tmp_path):
    """良性代码不受静态门影响，正常执行（防止误伤 happy path）。"""
    agent = CodeExecutorAgent(LLMClient(mock_mode=True))
    result = agent._execute_code_local(
        "print('hello-from-sandbox')\n", stage="smoke")
    assert result["success"] is True
    assert "hello-from-sandbox" in result["stdout"]
