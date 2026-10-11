"""Owned API request boundaries, using only fake responses and stdlib children."""
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.method_advice as advice
import src.runtime_preparation as runtime


def client(*, call_count=4):
    return SimpleNamespace(base_url="https://example.invalid/v1", model="test-model",
                           api_key="synthetic-private-test-token", call_count=call_count,
                           _record_usage=Mock())


def test_response_accounting_happens_after_owned_cleanup(monkeypatch):
    llm = client()
    events = []
    response = {"response": "accepted", "calls": 2, "usage": {"total_tokens": 17},
                "elapsed_s": .25, "error": False}

    def owned(argv, **kwargs):
        events.append("running")
        assert llm.call_count == 4
        llm._record_usage.assert_not_called()
        events.append("cleanup_confirmed")
        return subprocess.CompletedProcess(argv, 0, json.dumps(response), "")

    def record_usage(payload, elapsed):
        assert events[-1] == "cleanup_confirmed"
        assert llm.call_count == 6
        events.append("usage_recorded")

    llm._record_usage.side_effect = record_usage
    monkeypatch.setattr(runtime, "run_owned_process", owned)

    result = advice.request_text(llm, "Review exact public sources.", 45)

    assert result == response
    assert llm.call_count == 6
    llm._record_usage.assert_called_once_with({"usage": response["usage"]}, .25)
    assert events == ["running", "cleanup_confirmed", "usage_recorded"]


def test_empty_usage_does_not_add_an_extra_call(monkeypatch):
    llm = client(call_count=7)
    response = {"response": "API rejected the request", "calls": 1, "usage": {}, "error": True}
    monkeypatch.setattr(runtime, "run_owned_process", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, json.dumps(response), ""))

    assert advice.request_text(llm, "Review public sources.", 45) == response
    assert llm.call_count == 8
    llm._record_usage.assert_not_called()


def test_credentials_travel_only_in_private_stdin(monkeypatch, tmp_path):
    llm = client()
    sensitive_names = ("LLM_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN",
                       "AUTOREPRO_GITHUB_TOKEN")
    for name in sensitive_names:
        monkeypatch.setenv(name, "synthetic-env-secret-" + name)
    monkeypatch.setenv("AUTOREPRO_TEST_PUBLIC_SETTING", "public-setting")
    # Give the request an isolated project directory, so persistence is visible.
    monkeypatch.setattr(advice, "__file__", str(tmp_path / "src" / "method_advice.py"))
    captured = {}

    def owned(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        request = json.loads(kwargs["input"])
        assert request == {"base_url": llm.base_url, "model": llm.model,
                           "api_key": llm.api_key, "timeout": 45,
                           "prompt": "Public source review."}
        return subprocess.CompletedProcess(argv, 0,
                                           json.dumps({"response": "accepted", "calls": 1}), "")

    monkeypatch.setattr(runtime, "run_owned_process", owned)
    result = advice.request_text(llm, "Public source review.", 45)

    assert captured["argv"] == [sys.executable, "-X", "utf8",
                                str(tmp_path / "src" / "method_advice.py"), "--request"]
    assert captured["timeout_s"] == 45
    assert captured["cwd"] == tmp_path
    assert all(name not in captured["env"] for name in sensitive_names)
    assert captured["env"]["AUTOREPRO_TEST_PUBLIC_SETTING"] == "public-setting"
    assert llm.api_key not in json.dumps({key: value for key, value in captured.items()
                                         if key not in {"input", "cwd"}})
    assert llm.api_key not in json.dumps(result)
    assert list(tmp_path.rglob("*")) == [], "request must not persist credential files or logs"


def test_timeout_maps_only_after_owned_cleanup_and_keeps_streams(monkeypatch):
    llm = client()
    cleaned = []
    owned_error = runtime.RuntimePreparationTimeout("local deadline", stdout="partial response",
                                                    stderr="partial diagnostic")
    captured = {}

    def owned(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        assert llm.call_count == 4
        cleaned.append(True)
        raise owned_error

    monkeypatch.setattr(runtime, "run_owned_process", owned)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        advice.request_text(llm, "Review public sources.", 2)

    error = raised.value
    assert cleaned == [True]
    assert error.cmd == captured["argv"]
    assert error.timeout == captured["timeout_s"] == 2
    assert error.output == error.stdout == "partial response"
    assert error.stderr == "partial diagnostic"
    assert error.__cause__ is owned_error
    assert llm.call_count == 5
    llm._record_usage.assert_not_called()


@pytest.mark.parametrize("terminal", ["response", "timeout"])
def test_actual_descendant_is_stopped_before_request_finishes(monkeypatch, tmp_path, terminal):
    """Substitute a local API stub; descendants cannot write into the next attempt."""
    llm = client(call_count=0)
    started = tmp_path / "descendant_started.txt"
    sentinel = tmp_path / "next_attempt.txt"
    descendant = ("import time;from pathlib import Path;"
                  f"Path({str(started)!r}).write_text('started');"
                  f"time.sleep(1.8);Path({str(sentinel)!r}).write_text('orphan')")
    response = {"response": "accepted", "calls": 1, "usage": {"total_tokens": 3},
                "elapsed_s": .1}
    owner = ("import json,subprocess,sys,time;from pathlib import Path;"
             "request=json.load(sys.stdin);"
             f"subprocess.Popen([sys.executable,'-I','-c',{descendant!r}]);"
             f"started=Path({str(started)!r});"
             "deadline=time.monotonic()+5\n"
             "while not started.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
             "assert started.exists()\n")
    if terminal == "response":
        owner += f"print({json.dumps(response)!r},flush=True)\n"
    else:
        owner += ("print('partial API response',flush=True)\n"
                  "print('partial API diagnostic',file=sys.stderr,flush=True)\n"
                  "time.sleep(10)\n")
    real_owned = runtime.run_owned_process
    cleanup = []

    def local_api_stub(argv, **kwargs):
        assert argv[-1] == "--request"
        assert llm.api_key not in repr(argv)
        return real_owned([sys.executable, "-I", "-c", owner], **kwargs,
                          on_cleanup=lambda: cleanup.append("confirmed"))

    monkeypatch.setattr(runtime, "run_owned_process", local_api_stub)
    if terminal == "response":
        assert advice.request_text(llm, "Local stub only.", 5) == response
        llm._record_usage.assert_called_once_with({"usage": response["usage"]}, .1)
    else:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            advice.request_text(llm, "Local stub only.", 1.2)
        assert "partial API response" in raised.value.stdout
        assert "partial API diagnostic" in raised.value.stderr
        assert isinstance(raised.value.__cause__, runtime.RuntimePreparationTimeout)
        llm._record_usage.assert_not_called()

    assert cleanup == ["confirmed"]
    assert started.exists(), "the child must have started before the owner finished"
    assert llm.call_count == 1
    sentinel.write_text("next attempt", encoding="utf-8")
    time.sleep(2.1)
    assert sentinel.read_text(encoding="utf-8") == "next attempt"
    for artifact in tmp_path.iterdir():
        assert llm.api_key not in artifact.read_text(encoding="utf-8")
