"""本地执行输出编码回归测试（真实模式实测发现的 bug）。

Windows 中文环境下 `subprocess.run(..., text=True)` 不指定 encoding 时，
父进程按系统 locale（GBK）解码捕获到的输出；而子进程的标准流编码受
PYTHONIOENCODING 影响。两者不一致时 reader 线程抛 UnicodeDecodeError，
`CompletedProcess.stdout` 变成 **None** —— 随后 `full.get("stdout", "")[-300:]`
直接崩：`TypeError: 'NoneType' object is not subscriptable`
（key 存在且值为 None 时，dict.get 的默认值不生效）。

实测触发场景：真实 LLM 生成的脚本打印中文（"训练集 MSE: ..."），
第一次真实模式运行即崩溃。

运行: python -m pytest tests/test_exec_output_encoding.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.code_executor import CodeExecutorAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_deps_cache():
    import src.agents.code_executor as ce_mod
    ce_mod._INSTALLED_DEPS.clear()
    yield
    ce_mod._INSTALLED_DEPS.clear()


def _executor() -> CodeExecutorAgent:
    return CodeExecutorAgent(LLMClient(mock_mode=True))


def test_chinese_output_is_captured_and_is_str():
    """打印中文的脚本，stdout 必须是 str 且内容完整（不能是 None）。"""
    code = "print('训练集 MSE: 0.2447')\nprint('准确率: 85.2%')\n"
    result = _executor()._execute_code_local(code, stage="smoke")

    assert result["success"] is True, result.get("stderr")
    assert isinstance(result["stdout"], str)
    assert "训练集 MSE: 0.2447" in result["stdout"]
    assert "准确率: 85.2%" in result["stdout"]


def test_emoji_and_non_gbk_output_is_captured():
    """含 emoji 等非 GBK 可表示字符时也不能崩、不能丢输出。"""
    code = "print('✅ 复现完成 🎉')\n"
    result = _executor()._execute_code_local(code, stage="smoke")

    assert isinstance(result["stdout"], str)
    assert "复现完成" in result["stdout"]


def test_stdout_and_stderr_are_never_none():
    """即便脚本报错，stdout/stderr 也必须是字符串——下游会直接切片。"""
    result = _executor()._execute_code_local(
        "import sys\nprint('中文输出')\nsys.exit(3)\n", stage="smoke")

    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert result["exit_code"] == 3


def test_child_gets_utf8_stdio():
    """_exec_env 必须把子进程标准流钉成 UTF-8，与父进程解码一致。"""
    env = _executor()._exec_env()
    assert env.get("PYTHONIOENCODING") == "utf-8"


def test_docker_run_capture_declares_utf8(monkeypatch):
    """docker run 捕获输出必须显式 UTF-8 解码，与本地执行同一条规矩。

    docker 的输出是 UTF-8；不指定 encoding 就按系统 locale（Windows 中文 =
    GBK）解码，非 GBK 字节让 reader 线程抛 UnicodeDecodeError、`stdout` 变
    None（实测 docker build 输出即触发："'gbk' codec can't decode byte 0xaf"）。
    """
    import src.agents.code_executor as ce_mod

    seen = []

    def fake_run(cmd, **kw):
        seen.append((list(cmd), kw))
        return type("P", (), {"returncode": 0, "stdout": "容器输出 ok",
                              "stderr": ""})()

    monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
    _executor()._run_docker_cmd_with_sandbox_impl(
        ["docker", "run"], "python:3.11-slim", ["python", "run.py"], 30)

    assert seen, "应发起 docker run"
    for _cmd, kw in seen:
        assert kw.get("encoding") == "utf-8"
        assert kw.get("errors") == "replace"


def test_run_pipeline_survives_chinese_stdout():
    """整条 run() 链路不因中文输出崩溃（曾在此处 TypeError）。"""
    agent = _executor()
    result = agent.run({"code": "print('训练集 MSE: 0.05')\nprint('完成')\n"})
    assert result["success"] is True, result.get("reason")
    assert "训练集 MSE: 0.05" in (result["final"].get("stdout") or "")
