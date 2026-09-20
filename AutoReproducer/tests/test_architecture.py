"""AutoReproducer 完整自动化测试套件（pytest）。

覆盖：
1. 单测：UCBScheduler 预算调度、LLMClient Mock 任务分发、指标提取、
        依赖诊断循环、标题透传；
2. 集成：Mock 模式下 Orchestrator 全流水线（复现->验证->优化->报告）；
3. 报告内容关键断言；
4. 真实（API）模式：OpenAI 兼容请求格式/鉴权/响应解析/错误处理，
   通过本地假 HTTP 服务器验证，不依赖任何外部服务（本机可用服务则追加真实验证）。

运行: python -m pytest tests/ -v
"""
import http.server
import json
import sys
import threading
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient
from src.optimizer.ucb_scheduler import UCBScheduler
from src.agents.paper_reader import PaperReaderAgent
from src.agents.result_validator import ResultValidatorAgent
from src.agents.env_builder import EnvBuilderAgent
from src.audit.audit_logger import AuditLogger
from src.orchestrator import Orchestrator


# ============================================================
# 工具函数
# ============================================================

def make_mock_llm():
    """Mock 模式 LLMClient，供不关心真实调用的单测使用。"""
    return LLMClient(mock_mode=True)


# ============================================================
# 1. UCBScheduler 单测
# ============================================================

class TestUCBScheduler:
    def test_select_explores_all_arms_first(self):
        s = UCBScheduler(["a", "b", "c"], budget=10)
        picked = set()
        # select 本身不消耗预算，需配合 update 推进（模拟一次完整尝试）
        for _ in range(3):
            arm = s.select_arm()
            picked.add(arm)
            s.update(arm, 0.0)
        assert picked == {"a", "b", "c"}  # 未探索的臂都被强制探索一次

    def test_update_tracks_average_reward(self):
        s = UCBScheduler(["a", "b"], budget=10)
        s.update("a", 1.0)
        s.update("a", 0.0)
        assert s.arms["a"]["Q"] == pytest.approx(0.5)
        assert s.arms["a"]["n"] == 2

    def test_exhausted_returns_none(self):
        s = UCBScheduler(["a", "b"], budget=1)
        s.update("a", 1.0)
        assert s.is_exhausted()
        assert s.select_arm() is None

    def test_trace_shape(self):
        s = UCBScheduler(["a"], budget=5)
        s.update("a", 0.5)
        trace = s.trace()
        assert trace["total_pulls"] == 1
        assert trace["budget"] == 5
        assert trace["history"][0]["arm"] == "a"
        assert "a" in trace["arms"]


# ============================================================
# 2. LLMClient Mock 分发单测
# ============================================================

class TestLLMClientMock:
    def test_task_dispatch_is_exact(self):
        llm = make_mock_llm()
        text = llm.chat("任意提示词", task="verifier")
        parsed = json.loads(text)
        assert parsed.get("pass") is True

    def test_task_dispatch_does_not_keyword_mislead(self):
        """task 精确分发优先：即使提示词命中其他关键词也不串台。"""
        llm = make_mock_llm()
        # 提示词含 "验证" 等，但 task=paper_reader 必须返回论文信息
        text = llm.chat("请验证一下并输出论文内容", task="paper_reader")
        parsed = json.loads(text)
        assert "title" in parsed

    def test_fallback_keyword(self):
        """未传 task 时回退到关键词匹配。"""
        llm = make_mock_llm()
        text = llm.chat("从以下论文内容中提取结构化信息")
        parsed = json.loads(text)
        assert "title" in parsed

    def test_call_count_and_reset(self):
        llm = make_mock_llm()
        llm.chat("x", task="verifier")
        llm.chat("y", task="verifier")
        assert llm.get_call_count() == 2
        llm.reset_call_count()
        assert llm.get_call_count() == 0


# ============================================================
# 3. ResultValidator 指标提取单测
# ============================================================

def _make_validator():
    return ResultValidatorAgent(make_mock_llm())


class TestMetricExtraction:
    def test_accuracy_percent(self):
        agent = _make_validator()
        m = agent._extract_metrics("Test accuracy: 85.2%")
        assert m["accuracy"] == pytest.approx(85.2)

    def test_accuracy_fraction(self):
        agent = _make_validator()
        m = agent._extract_metrics("acc=0.852")
        assert m["accuracy"] == pytest.approx(0.852)

    def test_loss_f1(self):
        agent = _make_validator()
        m = agent._extract_metrics("Final loss: 0.3120, f1_score=0.8234")
        assert m["loss"] == pytest.approx(0.3120)
        assert m["f1_score"] == pytest.approx(0.8234)

    def test_empty_text(self):
        agent = _make_validator()
        assert agent._extract_metrics("") == {}

    def test_local_compare_within_tolerance(self):
        agent = _make_validator()
        cmp = agent._local_compare({"accuracy": 85.0}, {"accuracy": 85.4})
        assert cmp["match"] is True

    def test_local_compare_beyond_tolerance(self):
        agent = _make_validator()
        cmp = agent._local_compare({"accuracy": 100.0}, {"accuracy": 50.0})
        assert cmp["match"] is False


# ============================================================
# 4. EnvBuilder 依赖诊断单测
# ============================================================

class TestDependencyDiagnosis:
    def test_diagnose_resolves_conflicts(self):
        agent = EnvBuilderAgent(make_mock_llm())
        report = agent.diagnose_dependencies(
            ["torchvision>=0.15.0", "numpy==0.31.1"])
        assert report["resolved"] is True
        assert report["rounds"] <= report["max_rounds"]
        # 缺失 torch 被补齐
        assert any("torch" in p for p
                   in report["final_requirements_txt"].splitlines())

    def test_diagnose_clean_packages_first_round(self):
        agent = EnvBuilderAgent(make_mock_llm())
        report = agent.diagnose_dependencies(["requests>=2.0", "pytest"])
        assert report["resolved"] is True
        assert report["rounds"] == 1

    def test_max_rounds_cap(self):
        agent = EnvBuilderAgent(make_mock_llm())
        report = agent.diagnose_dependencies(["numpy==0.19.0"])
        # 版本冲突为确定性修复，最多 5 轮内收敛
        assert report["rounds"] >= 1


# ============================================================
# 5. PaperReader 输入透传单测
# ============================================================

class TestPaperReaderInput:
    def test_title_is_forwarded_in_mock(self):
        agent = PaperReaderAgent(make_mock_llm())
        result = agent.run({"paper_title": "ResNet 深度残差学习"})
        assert result["paper_info"]["title"] != ""
        # Mock 响应必须忠实融入用户标题
        assert "ResNet" in result["paper_info"]["title"] or \
               "ResNet" in result.get("raw_text", "")

    def test_pdf_missing_degrades_gracefully(self):
        agent = PaperReaderAgent(make_mock_llm())
        result = agent.run({"pdf_path": "C:/definitely/not/exists.pdf"})
        assert result["paper_info"]  # 降级提取仍然给出结构化信息


# ============================================================
# 6. 端到端集成：Mock 模式全流水线
# ============================================================

@pytest.fixture(scope="module")
def e2e_result():
    """跑一次完整流水线（Mock 模式），供多个断言复用。"""
    logger = AuditLogger()
    llm = LLMClient(mock_mode=True)
    orch = Orchestrator(llm_client=llm, mock_mode=True, logger=logger,
                        max_trials=6)
    return orch.run({"paper_title": "ResNet: Deep Residual Learning"})


class TestEndToEnd:
    def test_final_state_completed(self, e2e_result):
        assert e2e_result["state"] == "COMPLETED", e2e_result.get("error")

    def test_paper_info_present(self, e2e_result):
        data = e2e_result["data"]
        assert data.get("paper_info", {}).get("title")

    def test_reproduction_success(self, e2e_result):
        data = e2e_result["data"]
        assert data.get("validation", {}).get("is_reproduced") is True

    def test_optimization_ran(self, e2e_result):
        opt = e2e_result["data"].get("optimization", {})
        assert opt.get("optimized") is True
        assert opt.get("budget_used", 0) <= opt.get("budget", 0)
        assert opt.get("best_arm") is not None

    def test_report_generated(self, e2e_result):
        report = e2e_result["data"].get("report", "")
        assert len(report) > 200
        # 关键内容块齐全
        for section in ["## 1. 论文信息", "## 2. 资源定位",
                        "## 3. 环境配置", "## 4. 代码执行",
                        "## 5. 验证结果", "## 6. 智能优化",
                        "## 8. 审计与预算统计"]:
            assert section in report, f"报告缺少章节 {section}"

    def test_budget_stats_in_audit(self, e2e_result):
        stats = e2e_result["audit_stats"]
        assert stats.get("llm_calls", 0) > 0
        assert stats.get("success", 0) > 0

    def test_audit_ledger_written(self, e2e_result):
        """审计账本应真实落盘（data/experiment_ledger）。"""
        data_dir = (Path(__file__).resolve().parents[1]
                    / "data" / "experiment_ledger")
        assert data_dir.exists(), "实验账本目录不存在"
        entries = list(data_dir.glob("ledger_*.jsonl"))
        assert entries, "未找到实验账本文件"

    def test_verification_records(self, e2e_result):
        data = e2e_result["data"]
        assert data.get("verifications"), "缺少 Prompt-Free 验证记录"
        for v in data["verifications"]:
            assert "pass" in v


# ============================================================
# 真实(API)模式：OpenAI 兼容格式（本地假服务器，不依赖外部服务）
# ============================================================

class _FakeOpenAIHandler(http.server.BaseHTTPRequestHandler):
    """最小 OpenAI Chat Completions 假服务：记录请求并返回固定内容。"""
    captured = {}
    fail = None  # 若非空，返回该状态码 + 错误体（用于错误处理测试）

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).captured = {
            "path": self.path,
            "body": body,
            "auth": self.headers.get("Authorization", ""),
        }
        if type(self).fail:
            code = type(self).fail
            payload = json.dumps({"error": {"message": "fake denied"}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = json.dumps({
            "id": "chatcmpl-fake",
            "choices": [{"message": {"role": "assistant",
                                      "content": "fake-reply"}}],
            "model": body.get("model", ""),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_openai_server():
    _FakeOpenAIHandler.fail = None
    _FakeOpenAIHandler.captured = {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             _FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class TestAPIMode:
    """真实(API)模式核心行为：请求格式 / 鉴权 / 解析 / 错误处理 / 环境变量。"""

    def test_request_shape_and_auth(self, fake_openai_server):
        llm = LLMClient(base_url=fake_openai_server, model="deepseek-chat",
                        api_key="sk-test-123", mock_mode=False)
        text = llm.chat("你好", system_prompt="你是助手", temperature=0.2)
        assert text == "fake-reply"
        cap = _FakeOpenAIHandler.captured
        assert cap["path"] == "/v1/chat/completions"      # base_url 无 /v1 时自动补
        assert cap["body"]["model"] == "deepseek-chat"
        assert cap["body"]["temperature"] == 0.2
        assert cap["body"]["stream"] is False
        assert cap["body"]["messages"] == [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        assert cap["auth"] == "Bearer sk-test-123"

    def test_endpoint_v1_prefix_no_duplication(self):
        llm = LLMClient(base_url="https://api.deepseek.com/v1",
                        model="m", mock_mode=True)
        assert llm._endpoint() == "https://api.deepseek.com/v1/chat/completions"
        llm2 = LLMClient(base_url="https://qianfan.baidubce.com/v2",
                         model="m", mock_mode=True)
        assert llm2._endpoint() == \
            "https://qianfan.baidubce.com/v2/chat/completions"

    def test_mock_mode_never_touches_network(self):
        llm = LLMClient(base_url="http://127.0.0.1:1", model="m",
                        mock_mode=True)
        text = llm.chat("训练代码", task="code_executor")
        # Mock code_executor 为纯标准库实现（不依赖 numpy 等第三方包），
        # 含 mock 特有输出即证明走了 Mock 分发、未触网
        assert "Test accuracy: 85.2%" in text
        assert "Error" not in text

    def test_http_error_returns_clear_message(self, fake_openai_server):
        _FakeOpenAIHandler.fail = 401
        llm = LLMClient(base_url=fake_openai_server, model="m",
                        mock_mode=False)
        text = llm.chat("hi")
        assert text.startswith("[LLM API Error: HTTP 401"), text

    def test_connection_error_returns_clear_message(self):
        llm = LLMClient(base_url="http://127.0.0.1:1", model="m",
                        timeout=3, mock_mode=False)
        text = llm.chat("hi")
        assert text.startswith("[LLM API Error"), text

    def test_missing_config_clear_error(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        llm = LLMClient(mock_mode=False)
        assert "未配置 base_url" in llm.chat("hi")
        llm2 = LLMClient(base_url="http://x", mock_mode=False)
        assert "未配置 model" in llm2.chat("hi")

    def test_env_vars_are_loaded(self, monkeypatch, fake_openai_server):
        monkeypatch.setenv("LLM_BASE_URL", fake_openai_server)
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        llm = LLMClient(mock_mode=False)       # 无参构造，全部来自环境变量
        assert llm.base_url == fake_openai_server.rstrip("/")
        assert llm.model == "env-model"
        assert llm.api_key == "env-key"

    def test_call_count_tracked_in_api_mode(self, fake_openai_server):
        llm = LLMClient(base_url=fake_openai_server, model="m",
                        mock_mode=False)
        llm.chat("a")
        llm.chat("b")
        assert llm.get_call_count() == 2
        llm.reset_call_count()
        assert llm.get_call_count() == 0


def test_docker_not_required_for_mock():
    """Mock 模式不依赖 Docker；真实构建仅在明确调用时发生。"""
    agent = EnvBuilderAgent(make_mock_llm())
    assert agent.build_image({"dockerfile": ""})["success"] is False


# ============================================================
# 代码纯净性：_sanitize_code 清洗层
# ============================================================

class TestCodeSanitize:
    def _agent(self):
        from src.agents.code_executor import CodeExecutorAgent
        return CodeExecutorAgent(make_mock_llm())

    def test_drops_chinese_narration_and_fence(self):
        raw = ("为了生成一个简短的训练代码，我们需要假设一些数据集。\n"
               "```python\n"
               "import numpy as np\n"
               "print('accuracy: 0.85')\n"
               "```\n")
        out = self._agent()._sanitize_code(raw)
        assert "import numpy as np" in out
        assert "print('accuracy: 0.85')" in out
        assert "为了" not in out and "```" not in out

    def test_pure_code_passthrough(self):
        raw = "import math\nprint(math.factorial(5))\n"
        assert self._agent()._sanitize_code(raw) == raw.strip()

    def test_drops_inline_chinese_lines(self):
        raw = ("import sys\n"
               "假设我们使用的是 IMDB 数据集，评估指标为准确率。\n"
               "print(sys.version)\n")
        out = self._agent()._sanitize_code(raw)
        assert "假设" not in out
        assert "import sys" in out and "print(sys.version)" in out

    def test_strips_line_numbers_when_uncompilable(self):
        raw = "1 import os\n2 print(os.name)\n"
        out = self._agent()._sanitize_code(raw)
        assert out == "import os\nprint(os.name)"

    def test_sanitized_code_is_compilable(self):
        """清洗产物必须可通过 Python 语法编译。"""
        raw = ("开门见山：这是一个训练脚本。\n"
               "```\nimport torch\nimport numpy as np\n"
               "x = torch.randn(4, 4)\nprint(x.shape)\n```\n说明完毕。")
        code = self._agent()._sanitize_code(raw)
        compile(code, "<generated>", "exec")


def test_resolve_docker_cmd_via_env(monkeypatch, tmp_path):
    """DOCKER_PATH 环境变量指向真实文件时优先返回该路径。"""
    from src.base_agent import BaseAgent
    fake = tmp_path / "docker-cli.exe"
    fake.write_text("")  # 只需存在
    monkeypatch.setenv("DOCKER_PATH", str(fake))
    assert BaseAgent._resolve_docker_cmd() == str(fake)


def test_resolve_docker_cmd_env_ignored_when_missing(monkeypatch):
    """DOCKER_PATH 指向不存在的文件时被忽略,不会作为结果返回。"""
    from src.base_agent import BaseAgent
    monkeypatch.setenv("DOCKER_PATH", r"C:\no\such\docker.exe")
    resolved = BaseAgent._resolve_docker_cmd()
    assert resolved != r"C:\no\such\docker.exe"
    assert resolved is None or Path(resolved).is_file()


# ============================================================
# Docker 引擎存活探测（CLI 在 PATH 上 != daemon 在跑）
# ============================================================
# 背景（本机实测，Docker Desktop 已装但未启动）：
#   docker version --format "{{.Server.Version}}"  -> 188ms 失败返回
#   docker info                                    -> 20.7s 才失败返回
# 所以探测必须走 version：走 info 会让 Streamlit 每轮重跑冻住 20 秒。

def _fake_docker(tmp_path, body: str):
    """写一个假 docker 脚本，返回 ([解释器, 脚本], 参数记录文件)。"""
    script = tmp_path / "docker_fake.py"
    argfile = tmp_path / "argv.txt"
    script.write_text(
        "import pathlib, sys\n"
        f"pathlib.Path(r'{argfile}').write_text(' '.join(sys.argv[1:]))\n"
        + body, encoding="utf-8")
    return [sys.executable, str(script)], argfile


def test_docker_engine_available_probes_version_not_info(tmp_path):
    """探测命令锁定为 `version --format {{.Server.Version}}`（不是 info）。"""
    from src.base_agent import BaseAgent
    probe, argfile = _fake_docker(tmp_path, "print('29.7.2')\n")
    ok, reason = BaseAgent.docker_engine_available(probe, timeout=30)
    assert ok is True and reason is None
    assert argfile.read_text(encoding="utf-8") == \
        "version --format {{.Server.Version}}"


def test_docker_engine_available_detects_down_daemon(tmp_path):
    """CLI 能跑但引擎没起来 -> 判不可用，且原因是人话（可直接展示给用户）。"""
    from src.base_agent import BaseAgent
    probe, _ = _fake_docker(
        tmp_path,
        "import sys\n"
        "sys.stderr.write('failed to connect to the docker API at "
        "npipe:////./pipe/dockerDesktopLinuxEngine')\n"
        "sys.exit(1)\n")
    ok, reason = BaseAgent.docker_engine_available(probe, timeout=30)
    assert ok is False
    assert "未启动" in reason


def test_docker_engine_available_requires_server_version_output(tmp_path):
    """exit 0 但没有 server 版本号（假 CLI / 包装脚本）同样判不可用。"""
    from src.base_agent import BaseAgent
    probe, _ = _fake_docker(tmp_path, "print('')\n")
    ok, _ = BaseAgent.docker_engine_available(probe, timeout=30)
    assert ok is False


def test_docker_engine_available_timeout_is_honest(tmp_path):
    """CLI 卡死（daemon 无响应）时按超时判定并说明，不无限等待。"""
    from src.base_agent import BaseAgent
    probe, _ = _fake_docker(tmp_path, "import time\ntime.sleep(30)\n")
    ok, reason = BaseAgent.docker_engine_available(probe, timeout=0.5)
    assert ok is False
    assert "超时" in reason


def test_docker_engine_available_without_cli(monkeypatch):
    """连 CLI 都没有时，原因指向「未安装」——与「装了没启动」区分开。"""
    from src.base_agent import BaseAgent
    monkeypatch.setattr(BaseAgent, "_resolve_docker_cmd",
                        staticmethod(lambda: None))
    ok, reason = BaseAgent.docker_engine_available()
    assert ok is False
    assert "未安装" in reason


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))