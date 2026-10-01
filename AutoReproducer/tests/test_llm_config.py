"""frontend.llm_config 的单元测试：配置解析 + 连接测试（本地假 OpenAI 服务）。

覆盖（对应需求"前端输入 -> 后端接口 -> 调用 AI 复现"的可验证链路）：
- 生效配置解析：界面输入 > 环境变量 > 内置默认值；
- 真实模式缺配置检查：缺失项提示正确，配置就绪返回空；
- 连接测试：200 成功、HTTP 错误（401）、服务不可达三类真实路径。
"""
import http.server
import json
import threading

import pytest

from frontend.llm_config import (
    resolve_llm_config,
    config_missing,
    test_llm_connection,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """最小 OpenAI Chat Completions 假服务。"""
    captured = {}
    fail = None

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
                                      "content": "连接成功，一切正常。"}}],
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
    _FakeHandler.fail = None
    _FakeHandler.captured = {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def clean_env(monkeypatch):
    """清空 LLM 相关环境变量，保证用例独立。"""
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)


class TestResolveConfig:
    def test_defaults_when_nothing_provided(self, clean_env):
        cfg = resolve_llm_config()
        assert cfg["base_url"] == DEFAULT_BASE_URL
        assert cfg["model"] == DEFAULT_MODEL
        assert cfg["api_key"] == ""

    def test_input_overrides_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        cfg = resolve_llm_config(base_url="https://ui.example.com/v1",
                                 model="ui-model")
        assert cfg["base_url"] == "https://ui.example.com/v1"
        assert cfg["model"] == "ui-model"

    def test_env_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "sk-env")
        cfg = resolve_llm_config()
        assert cfg["base_url"] == "https://env.example.com/v1"
        assert cfg["model"] == "env-model"
        assert cfg["api_key"] == "sk-env"


class TestConfigMissing:
    def test_missing_reports_both(self, clean_env):
        missing = config_missing()
        assert "API 地址 (LLM_BASE_URL)" in missing
        assert "模型名称 (LLM_MODEL)" in missing

    def test_input_satisfies(self, clean_env):
        assert config_missing(base_url="https://x/v1", model="m") == []

    def test_env_satisfies(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://x/v1")
        monkeypatch.setenv("LLM_MODEL", "m")
        assert config_missing() == []


class TestConnection:
    def test_success(self, fake_openai_server, clean_env):
        ok, msg = test_llm_connection(
            base_url=fake_openai_server, api_key="sk-test-123", model="m1",
            timeout=10)
        assert ok is True
        assert "连接成功" in msg
        cap = _FakeHandler.captured
        assert cap["path"] == "/v1/chat/completions"
        assert cap["body"]["model"] == "m1"
        assert cap["auth"] == "Bearer sk-test-123"

    def test_http_error(self, fake_openai_server, clean_env):
        _FakeHandler.fail = 401
        ok, msg = test_llm_connection(
            base_url=fake_openai_server, api_key="sk-bad", model="m1",
            timeout=10)
        assert ok is False
        assert "HTTP 401" in msg

    def test_unreachable(self, clean_env):
        ok, msg = test_llm_connection(
            base_url="http://127.0.0.1:1",  # 端口 1 通常无服务监听，必然连接失败
            model="m1", timeout=5)
        assert ok is False
        assert "LLM API Error" in msg