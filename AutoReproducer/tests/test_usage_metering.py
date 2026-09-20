"""用量计量测试（P1-⑫）。

覆盖：
1. LLMClient.extract_token_usage：standard（prompt/completion/total）形态、
   input/output 别名形态、total 缺失按 p+c 推算、无 usage 返回 (0,0,0)；
2. LLMClient 真实模式（monkeypatch urllib.request.urlopen）：
   usage_hook 触发与 token/耗时累计，usage 缺失不触发 hook；
3. AuditLogger plan 级计量：begin/end_plan 界定、嵌套栈、plan 快照、
   unattributed 桶、get_stats() 的 plans/usage 汇总、models 聚合；
4. CodeExecutorAgent + fake docker：容器执行耗时归入当前 plan
   （record_sandbox_exec），无 plan 时归入 unattributed。

运行: python -m pytest tests/test_usage_metering.py -v
"""
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import CodeExecutorAgent  # noqa: E402
from src.audit.audit_logger import AuditLogger  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402


# ---------------- 1. extract_token_usage ----------------

class TestExtractTokenUsage:
    def test_standard_form(self):
        result = {"usage": {"prompt_tokens": 120,
                            "completion_tokens": 30,
                            "total_tokens": 150}}
        assert LLMClient.extract_token_usage(result) == (120, 30, 150)

    def test_alias_form_without_total(self):
        """部分网关用 input/output 别名且不返回 total：按 p+c 推算。"""
        result = {"usage": {"input_tokens": 10, "output_tokens": 5}}
        assert LLMClient.extract_token_usage(result) == (10, 5, 15)

    def test_missing_usage_returns_zero(self):
        assert LLMClient.extract_token_usage({}) == (0, 0, 0)
        assert LLMClient.extract_token_usage(None) == (0, 0, 0)

    def test_non_dict_usage_returns_zero(self):
        assert LLMClient.extract_token_usage({"usage": "n/a"}) == (0, 0, 0)

    def test_string_tokens_are_coerced(self):
        result = {"usage": {"prompt_tokens": "100", "completion_tokens": "20"}}
        assert LLMClient.extract_token_usage(result) == (100, 20, 120)

    def test_total_explicit_wins_over_sum(self):
        """网关返回的 total_tokens 与 p+c 不一致时以网关为准。"""
        result = {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                            "total_tokens": 130}}
        assert LLMClient.extract_token_usage(result) == (100, 20, 130)


# ---------------- 2. LLMClient 真实模式 hook ----------------

class TestLLMClientUsageHook:
    @staticmethod
    def _client():
        return LLMClient(base_url="http://fake.local/v1", model="gpt-fake")

    def _patch_urlopen(self, monkeypatch, payload):
        """monkeypatch urlopen 返回给定 JSON payload（支持 context manager）。"""
        body = json.dumps(payload).encode("utf-8")

        class _FakeResp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            assert timeout > 0
            return _FakeResp()

        import src.llm.llm_client as lc
        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        return fake_urlopen

    def test_hook_fires_with_tokens_on_success(self, monkeypatch):
        client = self._client()
        seen: list = []
        client.usage_hook = lambda **kw: seen.append(kw)
        payload = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                      "total_tokens": 150},
        }
        # monkeypatch httplib 层：直接替换 urlopen
        self._patch_urlopen(monkeypatch, payload)
        out = client.chat("hello", system_prompt="sys")
        assert out == "hi"
        assert len(seen) == 1
        kw = seen[0]
        assert kw["model"] == "gpt-fake"
        assert kw["prompt_tokens"] == 120
        assert kw["completion_tokens"] == 30
        assert kw["duration_seconds"] >= 0
        assert client.total_prompt_tokens == 120
        assert client.total_completion_tokens == 30
        assert client.last_usage == {"prompt_tokens": 120,
                                     "completion_tokens": 30,
                                     "total_tokens": 150}

    def test_hook_not_fired_without_usage(self, monkeypatch):
        client = self._client()
        fired = []
        client.usage_hook = lambda **kw: fired.append(kw)
        payload = {"choices": [{"message": {"content": "ok"},
                                "finish_reason": "stop"}]}
        self._patch_urlopen(monkeypatch, payload)
        out = client.chat("hello")
        assert out == "ok"
        assert fired == []
        assert client.total_prompt_tokens == 0
        assert client.total_completion_tokens == 0
        assert client.last_usage == {"prompt_tokens": 0,
                                     "completion_tokens": 0,
                                     "total_tokens": 0}

    def test_hook_exception_does_not_break_call(self, monkeypatch):
        client = self._client()

        def boom(**kw):
            raise RuntimeError("metering broken")

        client.usage_hook = boom
        payload = {
            "choices": [{"message": {"content": "still works"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }
        self._patch_urlopen(monkeypatch, payload)
        assert client.chat("hi") == "still works"

    def test_alias_usage_via_real_chat(self, monkeypatch):
        """真实响应用 input/output 别名：hook 收到 p+c 推算的 total。"""
        client = self._client()
        seen: list = []
        client.usage_hook = lambda **kw: seen.append(kw)
        payload = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        self._patch_urlopen(monkeypatch, payload)
        client.chat("ask")
        assert seen[0]["prompt_tokens"] == 7
        assert seen[0]["completion_tokens"] == 3


def _ce_mod():
    """返回 LLMClient 所在模块（便于 urlopen 的 monkeypatch 定位）。"""
    import src.llm.llm_client as lc
    return lc.urllib.request


# ---------------- 3. AuditLogger plan 级计量 ----------------

class TestAuditPlanMetering:
    @staticmethod
    def _logger(tmp_path):
        return AuditLogger(log_dir=str(tmp_path / "logs"),
                           ledger_dir=str(tmp_path / "ledger"))

    def test_begin_end_plan_snapshot(self, tmp_path):
        logger = self._logger(tmp_path)
        assert logger.current_plan == ""
        logger.begin_plan("READ_PAPER")
        assert logger.current_plan == "READ_PAPER"
        logger.record_llm_usage(model="m1", prompt_tokens=120,
                                completion_tokens=30, duration_seconds=1.5)
        snap = logger.end_plan("READ_PAPER")
        assert snap["plan_id"] == "READ_PAPER"
        assert snap["calls"] == 1
        assert snap["prompt_tokens"] == 120
        assert snap["completion_tokens"] == 30
        assert snap["total_tokens"] == 150
        assert snap["llm_seconds"] == 1.5
        assert snap["models"] == {"m1": 1}
        # 出栈后 current_plan 回空
        assert logger.current_plan == ""

    def test_end_plan_id_mismatch_is_noop(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.begin_plan("A")
        assert logger.end_plan("B") == {}
        assert logger.current_plan == "A"
        logger.end_plan()          # 默认出栈 A
        assert logger.current_plan == ""

    def test_end_plan_empty_stack_is_noop(self, tmp_path):
        logger = self._logger(tmp_path)
        assert logger.end_plan() == {}
        assert logger.current_plan == ""

    def test_nested_plans_charge_current(self, tmp_path):
        """嵌套 plan：用量归入栈顶 plan，互不污染。"""
        logger = self._logger(tmp_path)
        logger.begin_plan("EXECUTE_CODE")
        logger.record_llm_usage(model="m", prompt_tokens=10,
                                completion_tokens=2)
        logger.begin_plan("VALIDATE")
        logger.record_llm_usage(model="m", prompt_tokens=50,
                                completion_tokens=10)
        assert logger.current_plan == "VALIDATE"
        logger.end_plan("VALIDATE")
        logger.record_llm_usage(model="m", prompt_tokens=1,
                                completion_tokens=1)
        logger.end_plan()
        snap = logger.plan_snapshot("EXECUTE_CODE")
        assert snap["prompt_tokens"] == 11
        assert snap["completion_tokens"] == 3
        vsnap = logger.plan_snapshot("VALIDATE")
        assert vsnap["prompt_tokens"] == 50
        assert vsnap["completion_tokens"] == 10

    def test_unattributed_bucket(self, tmp_path):
        """无 plan 上下文归入 unattributed。"""
        logger = self._logger(tmp_path)
        logger.record_llm_usage(model="m", prompt_tokens=5,
                                completion_tokens=5)
        logger.record_sandbox_exec(duration_seconds=2.0)
        snap = logger.plan_snapshot("unattributed")
        assert snap["calls"] == 1
        assert snap["prompt_tokens"] == 5
        assert snap["exec_calls"] == 1
        assert snap["exec_seconds"] == 2.0

    def test_get_stats_summary(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.add_llm_calls(3)          # 预算口径（独立于 plan 计量）
        logger.begin_plan("READ_PAPER")
        logger.record_llm_usage(model="deepseek", prompt_tokens=100,
                                completion_tokens=20, duration_seconds=2.0)
        logger.record_sandbox_exec(duration_seconds=3.5)
        logger.end_plan()
        # 无 plan 的调用也计入汇总
        logger.record_llm_usage(model="deepseek", prompt_tokens=30,
                                completion_tokens=10)
        stats = logger.get_stats()
        assert stats["llm_calls"] == 3
        assert "READ_PAPER" in stats["plans"]
        usage = stats["usage"]
        assert usage["llm_calls"] == 2
        assert usage["prompt_tokens"] == 130
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 160
        assert usage["llm_seconds"] == 2.0
        assert usage["container_exec_calls"] == 1
        assert usage["container_exec_seconds"] == 3.5
        assert usage["models"] == {"deepseek": 2}
        # plans 明细也带容器维度字段
        assert stats["plans"]["READ_PAPER"]["exec_calls"] == 1

    def test_usage_hook_alias(self, tmp_path):
        """logger.usage_hook 与 record_llm_usage 等价（可直接绑定 LLMClient）。"""
        logger = self._logger(tmp_path)
        assert logger.usage_hook == logger.record_llm_usage
        logger.begin_plan("P")
        logger.usage_hook(model="m", prompt_tokens=9, completion_tokens=1)
        assert logger.plan_snapshot("P")["total_tokens"] == 10


# ---------------- 4. CodeExecutor 容器耗时计量 ----------------

class TestSandboxMetering:
    @staticmethod
    def _executor(logger):
        executor = CodeExecutorAgent(LLMClient(mock_mode=True),
                                     logger=logger, mock_mode=True)
        executor.use_docker = True
        return executor

    def _fake_docker(self, monkeypatch, calls, stderr="", code=0):
        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, code, stdout="ok", stderr=stderr)

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        return fake_run

    def test_docker_exec_charged_to_current_plan(self, monkeypatch, tmp_path):
        """容器执行耗时归入当前 plan（EXECUTE_CODE）。"""
        logger = AuditLogger(log_dir=str(tmp_path / "logs"),
                             ledger_dir=str(tmp_path / "ledger"))
        calls: list = []
        self._fake_docker(monkeypatch, calls)
        executor = self._executor(logger)
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        logger.begin_plan("EXECUTE_CODE")
        result = executor._execute_code_docker("print(1)\n", "smoke",
                                               workdir=str(tmp_path))
        logger.end_plan("EXECUTE_CODE")
        assert result["success"] is True
        assert len(calls) == 1
        snap = logger.plan_snapshot("EXECUTE_CODE")
        assert snap["exec_calls"] == 1
        assert snap["exec_seconds"] >= 0

    def test_docker_exec_charged_each_attempt(self, monkeypatch, tmp_path):
        """缺模块自愈的多次容器执行分别计量（exec_calls 次数对应执行次数）。"""
        logger = AuditLogger(log_dir=str(tmp_path / "logs"),
                             ledger_dir=str(tmp_path / "ledger"))
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            joined = " ".join(str(c) for c in cmd)
            if "pip install" in joined:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="ok", stderr="")
            return subprocess.CompletedProcess(
                cmd, 1, stdout="",
                stderr="ModuleNotFoundError: No module named 'cv2'")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = self._executor(logger)
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        logger.begin_plan("EXECUTE_CODE")
        result = executor._execute_code_docker("import cv2\n", "smoke",
                                               workdir=str(tmp_path))
        logger.end_plan()
        assert result["success"] is True
        assert len(calls) == 2
        snap = logger.plan_snapshot("EXECUTE_CODE")
        assert snap["exec_calls"] == 2
        assert snap["exec_seconds"] >= 0

    def test_docker_exec_without_plan_goes_unattributed(self, monkeypatch,
                                                        tmp_path):
        """CodeExecutor 独立使用（无 plan 上下文）归入 unattributed。"""
        logger = AuditLogger(log_dir=str(tmp_path / "logs"),
                             ledger_dir=str(tmp_path / "ledger"))
        calls: list = []
        self._fake_docker(monkeypatch, calls)
        executor = self._executor(logger)
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        executor._execute_code_docker("print(1)\n", "smoke",
                                      workdir=str(tmp_path))
        snap = logger.plan_snapshot("unattributed")
        assert snap["exec_calls"] == 1

    def test_wrapper_measures_wall_time(self, monkeypatch):
        """wrapper 记录真实墙钟时长（sleep 模拟慢容器，时长 >= sleep）。"""
        logger = AuditLogger()
        calls: list = []
        sleep_sec = 0.05

        def fake_run(cmd, **kw):
            calls.append(cmd)
            time.sleep(sleep_sec)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(ce_mod.subprocess, "run", fake_run)
        executor = self._executor(logger)
        monkeypatch.setattr(executor, "_resolve_docker_cmd",
                            lambda: "docker")
        executor.env_config = {"image_tag": "python:3.11-slim",
                               "requirements_txt": ""}
        import tempfile
        with tempfile.TemporaryDirectory() as wd:
            res = executor._execute_code_docker("print(1)\n", "smoke",
                                                workdir=wd)
        assert res["success"] is True
        snap = logger.plan_snapshot("unattributed")
        assert snap["exec_calls"] == 1
        assert snap["exec_seconds"] >= sleep_sec


# ---------------- 5. Orchestrator 接线 ----------------

class TestOrchestratorMeteringWire:
    def test_llm_hook_bound_to_logger(self, tmp_path):
        """Orchestrator 把 LLMClient.usage_hook 绑定到 logger.record_llm_usage，
        且各 plan 阶段产生可核算的 plan 级用量（mock 模式不触发 hook）。"""
        logger = AuditLogger(log_dir=str(tmp_path / "logs"),
                             ledger_dir=str(tmp_path / "ledger"))
        orch = Orchestrator(mock_mode=True, logger=logger,
                            max_trials=2)
        assert orch.llm.usage_hook == logger.record_llm_usage
        # mock 模式 LLM 不产生 usage，plans 至少存在（阶段界定已生效）
        stats = logger.get_stats()
        assert "usage" in stats
        assert stats["usage"]["llm_calls"] == 0

    def test_external_llm_client_hook_attached(self, tmp_path):
        """外部注入 LLMClient 也能被接上 hook（兼容外部传入）。"""
        logger = AuditLogger(log_dir=str(tmp_path / "logs"),
                             ledger_dir=str(tmp_path / "ledger"))
        client = LLMClient(mock_mode=True)
        assert client.usage_hook is None
        Orch = Orchestrator(llm_client=client, mock_mode=True, logger=logger)
        assert Orch.llm is client
        assert client.usage_hook == logger.record_llm_usage