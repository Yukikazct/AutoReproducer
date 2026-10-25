"""Run isolation, bounded delivery and safe evidence handling for agent handoffs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src import agent_collaboration as collaboration
from src.agent_collaboration import (
    CollaborationSession, collaborative_run, current_collaboration,
)


def test_early_cancellation_of_reused_service_preserves_previous_result(tmp_path):
    class Service:
        data = None

        @collaborative_run
        def run(self, request, on_event=None):
            if request.get('cancel'):
                raise KeyboardInterrupt()
            self.data = {'run_dir': str(tmp_path)}
            current_collaboration().send('Reader', 'Planner', 'evidence', {'paper': 'previous'})
            return {'data': self.data}

    service = Service()
    result = service.run({})
    before = deepcopy(result)
    artifact = (tmp_path / 'agent_collaboration.json').read_bytes()
    with pytest.raises(KeyboardInterrupt):
        service.run({'cancel': True})
    assert result == before and service.data is result['data']
    assert (tmp_path / 'agent_collaboration.json').read_bytes() == artifact
    assert current_collaboration() is None


def test_linked_request_reply_and_explicit_delivery():
    session = CollaborationSession(run_id="run")
    request = session.send("AuthorPlanner", "SourceReader", "request", {"files": ["train.py"]})
    assert request["id"] == "run:1" and not request["delivered"]
    assert session.context("Unrelated") == []
    incoming = session.context("SourceReader")
    assert incoming[0]["delivered"]
    incoming[0]["content"]["files"].append("injected.py")
    assert session.context("SourceReader")[0]["content"] == {"files": ["train.py"]}
    response = session.reply(request, "SourceReader", {"train.py": {"sha256": "f" * 64}})
    assert response["recipient"] == "AuthorPlanner" and response["reply_to"] == request["id"]
    assert response["correlation_id"] == request["correlation_id"]
    assert session.context("AuthorPlanner")[0]["id"] == response["id"]
    assert session.snapshot()["participants"] == ["AuthorPlanner", "SourceReader"]


def test_reply_cannot_spoof_routing_or_cross_runs():
    session = CollaborationSession()
    request = session.send("A", "B", "request", "source")
    with pytest.raises(ValueError, match="requested recipient"):
        session.reply(request, "C", "unrequested response")
    with pytest.raises(ValueError, match="existing message"):
        CollaborationSession().reply(request, "B", "other run")
    tampered = {**request, "sender": "Injected"}
    reply = session.reply(tampered, "B", "response")
    assert reply["recipient"] == "A"


def test_messages_events_logger_and_artifact_redact_credentials(tmp_path):
    secret = "unusual-user-secret-xyz"
    events, logs = [], []
    logger = SimpleNamespace(log=lambda *args: logs.append(args))
    session = CollaborationSession(credential=secret, on_event=events.append, logger=logger)
    request = session.send("A", "B", "handoff", {
        "LLM_API_KEY": "a-private-key", "nested": {"password": "password-value"},
        "text": f"{secret} sk-1234567890abcdefgh Bearer abc.def.ghi\napi_key=embedded_secret",
        "raw_json": '{"LLM_API_KEY": "json-secret-value", "app.api_key": "dotted-secret-value"}',
        "app.api_key": "structured-dotted-secret",
        "source": "https://user:private@example.invalid/repo",
    }, evidence={"access_token": "another-private-token"})
    session.persist(tmp_path)
    combined = json.dumps([session.snapshot(), events, logs, request]) + (
        tmp_path / "agent_collaboration.json").read_text(encoding="utf-8")
    for value in (secret, "a-private-key", "password-value", "sk-1234567890abcdefgh",
                  "abc.def.ghi", "embedded_secret", "user:private", "another-private-token"):
        assert value not in combined
    for value in ("json-secret-value", "dotted-secret-value", "structured-dotted-secret"):
        assert value not in combined
    assert "[REDACTED]" in combined


def test_fixed_message_budget_drops_without_unbounded_growth(monkeypatch):
    monkeypatch.setattr(collaboration, "MAX_MESSAGES", 2)
    session = CollaborationSession()
    assert session.send("A", "B", "handoff", "first")
    assert session.send("B", "C", "handoff", "second")
    assert session.send("C", "A", "handoff", "third") is None
    assert session.reply(None, "A", "budget-limited response") is None
    snapshot = session.snapshot()
    assert len(snapshot["messages"]) == 2 and snapshot["dropped_messages"] == 1


def test_total_and_payload_budgets_are_explicit(monkeypatch):
    monkeypatch.setattr(collaboration, "MAX_TOTAL_CHARS", 300)
    session = CollaborationSession(run_id="run")
    assert session.send("A", "B", "handoff", "small")
    assert session.send("A", "B", "handoff", "another") is None
    assert session.snapshot()["dropped_messages"] == 1
    monkeypatch.setattr(collaboration, "MAX_TOTAL_CHARS", 128 * 1024)
    large = session.send("A", "B", "handoff", {"source": "a" * 10000, "more": "b" * 10000})
    assert large["content"]["truncated"] is True
    assert len(json.dumps(large, ensure_ascii=False)) <= collaboration.MAX_MESSAGE_CHARS


def test_context_limit_does_not_deliver_unselected_messages():
    session = CollaborationSession()
    for index in range(12):
        session.send("A", "B", "handoff", index)
    assert session.context("B", limit=0) == []
    selected = session.context("B", limit=100)
    assert len(selected) == collaboration.MAX_CONTEXT_MESSAGES
    assert selected[0]["content"] == 4 and selected[-1]["content"] == 11
    assert not session.snapshot()["messages"][0]["delivered"]


def test_unknown_objects_cycles_and_nonfinite_values_remain_json_safe():
    class PrivateObject:
        def __repr__(self):
            raise AssertionError("Never render unknown objects")
    content = {"number": float("nan"), "object": PrivateObject()}
    content["cycle"] = content
    session = CollaborationSession()
    message = session.send("A", "B", "handoff", content)
    assert message["content"]["number"] == "[nonfinite]"
    assert message["content"]["cycle"] == "[cycle]"
    json.dumps(session.snapshot(), allow_nan=False)


def test_container_subclasses_are_not_evaluated_as_source_data():
    class UnsafeDictionary(dict):
        def items(self):
            raise AssertionError("Do not call a custom mapping")
    class UnsafeString(str):
        def replace(self, *args):
            raise AssertionError("Do not call a custom string")
    class UnsafeList(list):
        def __getitem__(self, index):
            raise AssertionError("Do not index a custom list")
    session = CollaborationSession()
    result = session.send("A", "B", "handoff", {
        "mapping": UnsafeDictionary(secret="private"),
        "string": UnsafeString("private"), "list": UnsafeList(["private"]),
    })
    assert all(value == "[unsupported value]" for value in result["content"].values())


def test_observer_failures_do_not_fail_a_handoff():
    def broken(*args):
        raise OSError("display unavailable")
    session = CollaborationSession(on_event=broken, logger=SimpleNamespace(log=broken))
    assert session.send("A", "B", "handoff", "evidence")
    assert len(session.snapshot()["messages"]) == 1


def test_observer_does_not_swallow_cancellation():
    def cancelled(*args):
        raise KeyboardInterrupt()
    session = CollaborationSession(on_event=cancelled)
    with pytest.raises(KeyboardInterrupt):
        session.send("A", "B", "handoff", "evidence")
    assert len(session.snapshot()["messages"]) == 1


def test_scope_restores_parent_after_base_exception():
    parent, child = CollaborationSession(), CollaborationSession()
    assert current_collaboration() is None
    with parent.scope():
        with pytest.raises(KeyboardInterrupt):
            with child.scope():
                assert current_collaboration() is child
                raise KeyboardInterrupt()
        assert current_collaboration() is parent
    assert current_collaboration() is None


def test_concurrent_thread_runs_have_distinct_contexts():
    def execute(index):
        session = CollaborationSession()
        with session.scope():
            session.send("A", "B", "handoff", index)
            assert current_collaboration() is session
            return session.snapshot()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, range(4)))
    assert len({item["run_id"] for item in results}) == 4
    assert [item["messages"][0]["content"] for item in results] == list(range(4))
    assert current_collaboration() is None


def test_attach_returns_detached_serializable_snapshot():
    session, data = CollaborationSession(), {}
    session.send("A", "B", "handoff", {"value": 1})
    session.attach(data)
    data["agent_collaboration"]["messages"][0]["content"]["value"] = 2
    assert session.snapshot()["messages"][0]["content"]["value"] == 1
    json.dumps(data, allow_nan=False)


def test_nested_decorated_services_share_one_run_and_sequential_calls_do_not(tmp_path):
    class Service:
        @collaborative_run
        def inner(self, request, on_event=None):
            current_collaboration().send("B", "C", "handoff", "inner")
            return {"data": {}}

        @collaborative_run
        def run(self, request, on_event=None):
            current_collaboration().send("A", "B", "handoff", "outer")
            nested = self.inner(request)
            return {"data": {"run_dir": str(request["path"]), "nested_run": nested["data"]["agent_collaboration"]["run_id"]}}
    service, events = Service(), []
    first = service.run({"path": tmp_path}, events.append)["data"]
    assert first["nested_run"] == first["agent_collaboration"]["run_id"]
    assert len(first["agent_collaboration"]["messages"]) == 2 and len(events) == 2
    assert (tmp_path / "agent_collaboration.json").is_file()
    another = tmp_path / "another"
    another.mkdir()
    second = service.run({"path": another})["data"]
    assert first["agent_collaboration"]["run_id"] != second["agent_collaboration"]["run_id"]
    assert current_collaboration() is None


def test_decorator_retains_interrupted_handoffs_and_restores_context(tmp_path):
    class Service:
        def __init__(self):
            self.data = {"run_dir": str(tmp_path)}

        @collaborative_run
        def run(self, request, on_event=None):
            current_collaboration().send("A", "B", "request", "before cancellation")
            raise KeyboardInterrupt()
    service = Service()
    with pytest.raises(KeyboardInterrupt):
        service.run({})
    assert current_collaboration() is None
    assert len(service.data["agent_collaboration"]["messages"]) == 1
    saved = json.loads((tmp_path / "agent_collaboration.json").read_text(encoding="utf-8"))
    assert saved["messages"][0]["content"] == "before cancellation"


def test_persist_can_update_only_its_own_artifact(tmp_path):
    session = CollaborationSession()
    session.send("A", "B", "handoff", "one")
    assert session.persist(tmp_path)
    session.send("B", "C", "handoff", "two")
    assert session.persist(tmp_path)
    path = tmp_path / "agent_collaboration.json"
    saved = path.read_bytes()
    assert len(json.loads(saved)["messages"]) == 2
    assert CollaborationSession().persist(tmp_path) is None
    assert path.read_bytes() == saved


def test_persist_refuses_unrelated_files_and_reports_storage_failure(tmp_path):
    path = tmp_path / "agent_collaboration.json"
    path.write_text("user file", encoding="utf-8")
    session = CollaborationSession()
    assert session.persist(tmp_path) is None
    assert path.read_text(encoding="utf-8") == "user file"
    assert session.snapshot()["storage_error"]
    assert session.persist(tmp_path / "missing") is None


def test_persist_refuses_symlink_artifact_when_supported(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("user file", encoding="utf-8")
    target = tmp_path / "agent_collaboration.json"
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    assert CollaborationSession().persist(tmp_path) is None
    assert outside.read_text(encoding="utf-8") == "user file"
