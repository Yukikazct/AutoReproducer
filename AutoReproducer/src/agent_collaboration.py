"""Bounded, run-scoped evidence handoffs between active agent roles.

Messages are data and proposals, never executable commands or verification
authority. The transport does not call a model, schedule agents, or change an
experiment. Callers explicitly consume messages and validate their results.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import inspect
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
import uuid

from src.safety.paths import filesystem_path, workspace_path


MAX_MESSAGES = 128
MAX_MESSAGE_CHARS = 8192
MAX_TOTAL_CHARS = 128 * 1024
MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARS = 16 * 1024
MAX_CONTENT_CHARS = 6000
MAX_DEPTH = 6
MAX_ITEMS = 32

_CURRENT: ContextVar[CollaborationSession | None] = ContextVar(
    "autoreproducer_agent_collaboration", default=None)
_SECRET_NAME = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:api[_-]?key|password|passwd|secret|(?:access|auth|refresh)[_-]?token|"
    r"authorization|credentials?|cookies?|private[_-]?key)(?:$|[^A-Za-z0-9])", re.I)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|password|passwd|secret|(?:access|auth|refresh)[_-]?token|"
    r"authorization|credential)\b[\"']?\s*[:=]\s*)([\"']?)([^\s,;}\"'\r\n]+)")


def current_collaboration():
    """Return this execution context's session, or None outside a run."""
    return _CURRENT.get()


class CollaborationSession:
    """An in-process mailbox and audit trail with fixed, small budgets."""

    def __init__(self, *, on_event=None, logger=None, credential="", run_id=None):
        self.on_event, self.logger = on_event, logger
        self._credential = credential if type(credential) is str else ""
        self.run_id = run_id if type(run_id) is str and run_id else uuid.uuid4().hex
        self.run_id = self._text(self.run_id)[:96]
        self._messages = []
        self._by_id = {}
        self._participants = set()
        self._total_chars = 0
        self._dropped_messages = 0
        self._storage_error = ""
        self._lock = RLock()

    def _text(self, value):
        if self._credential:
            value = value.replace(self._credential, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        value = _SECRET_ASSIGNMENT.sub(lambda match: match[1] + match[2] + "[REDACTED]", value)
        return re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)

    def _safe(self, value, depth=0, ancestors=None):
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is str:
            text = self._text(value)
            return text if len(text) <= MAX_CONTENT_CHARS else text[:MAX_CONTENT_CHARS] + " [truncated]"
        if type(value) is float:
            return value if math.isfinite(value) else "[nonfinite]"
        if depth >= MAX_DEPTH:
            return "[depth limit]"
        if type(value) not in (dict, list, tuple):
            return "[unsupported value]"
        ancestors = set() if ancestors is None else ancestors
        if id(value) in ancestors:
            return "[cycle]"
        ancestors.add(id(value))
        try:
            if type(value) is dict:
                result = {}
                for index, (key, item) in enumerate(value.items()):
                    if index >= MAX_ITEMS:
                        result["_truncated"] = True
                        break
                    if type(key) is not str:
                        continue
                    clean_key = self._text(key)[:160]
                    result[clean_key] = ("[REDACTED]" if _SECRET_NAME.search(key)
                                         else self._safe(item, depth + 1, ancestors))
                return result
            result = [self._safe(item, depth + 1, ancestors) for item in value[:MAX_ITEMS]]
            if len(value) > MAX_ITEMS:
                result.append("[item limit]")
            return result
        finally:
            ancestors.remove(id(value))

    def _payload(self, value, limit):
        result = self._safe(value)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(encoded) <= limit:
            return result
        # Keep truncation explicit and JSON-serializable, rather than cutting a
        # serialized object into a malformed partial object.
        return {"excerpt": encoded[:max(0, limit - 80)], "truncated": True}

    def _role(self, value, label):
        if type(value) is not str or not value.strip() or len(value) > 96:
            raise ValueError(f"{label} must be a nonempty role or kind of at most 96 characters")
        return self._text(value.strip())

    @contextmanager
    def scope(self):
        """Bind only this thread/async context and always restore its parent."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def send(self, sender, recipient, kind, content, *, reply_to=None, evidence=None):
        """Store one sanitized message, returning None if its budget is full."""
        sender, recipient = self._role(sender, "sender"), self._role(recipient, "recipient")
        kind = self._role(kind, "kind")
        with self._lock:
            parent = None
            if reply_to is not None:
                if type(reply_to) is not str or reply_to not in self._by_id:
                    raise ValueError("reply_to must identify an existing message in this run")
                parent = self._by_id[reply_to]
                if sender != parent["recipient"] or recipient != parent["sender"]:
                    raise ValueError("a reply must return from the requested recipient to its sender")
            sequence = len(self._messages) + 1
            identifier = f"{self.run_id}:{sequence}"
            message = {"id": identifier, "sequence": sequence,
                       "sender": sender, "recipient": recipient, "kind": kind,
                       "content": self._payload(content, MAX_CONTENT_CHARS),
                       "reply_to": reply_to,
                       "correlation_id": parent["correlation_id"] if parent else identifier,
                       "evidence": self._payload([] if evidence is None else evidence, 1200),
                       "delivered": False}
            length = len(json.dumps(message, ensure_ascii=False, allow_nan=False))
            if (len(self._messages) >= MAX_MESSAGES or length > MAX_MESSAGE_CHARS
                    or self._total_chars + length > MAX_TOTAL_CHARS):
                self._dropped_messages += 1
                return None
            self._messages.append(message)
            self._by_id[identifier] = message
            self._participants.update((sender, recipient))
            self._total_chars += length
            if parent is not None:
                parent["delivered"] = True
            result = deepcopy(message)
        # Observers receive detached data. A UI/logger failure cannot change
        # command execution; actual cancellation (BaseException) propagates.
        if self.on_event is not None:
            try:
                self.on_event({"type": "agent_message", "message": deepcopy(result)})
            except Exception:
                pass
        if self.logger is not None:
            try:
                self.logger.log(sender, "agent_message", "INFO",
                                f"{sender} → {recipient}: {kind}", {"message": deepcopy(result)})
            except Exception:
                pass
        return result

    def reply(self, request, sender, content, *, kind="response", evidence=None):
        """Reply to a known request without accepting caller-edited routing."""
        if request is None:
            return None  # A request omitted by the budget has no response route.
        identifier = request.get("id") if type(request) is dict else request
        with self._lock:
            if type(identifier) is not str or identifier not in self._by_id:
                raise ValueError("request must identify an existing message in this run")
            recipient = self._by_id[identifier]["sender"]
        return self.send(sender, recipient, kind, content, reply_to=identifier, evidence=evidence)

    def context(self, recipient, limit=6):
        """Consume a bounded copy of this role's newest received messages."""
        recipient = self._role(recipient, "recipient")
        if type(limit) is not int or limit < 0:
            raise ValueError("context limit must be a nonnegative integer")
        limit = min(limit, MAX_CONTEXT_MESSAGES)
        with self._lock:
            selected, length = [], 0
            if limit:
                for message in reversed(self._messages):
                    if message["recipient"] != recipient:
                        continue
                    size = len(json.dumps(message, ensure_ascii=False))
                    if len(selected) >= limit or length + size > MAX_CONTEXT_CHARS:
                        break
                    message["delivered"] = True
                    selected.append(message)
                    length += size
            return deepcopy(list(reversed(selected)))

    def snapshot(self):
        with self._lock:
            result = {"version": 1, "run_id": self.run_id,
                      "authority": "messages_are_proposals",
                      "participants": sorted(self._participants),
                      "messages": deepcopy(self._messages),
                      "dropped_messages": self._dropped_messages,
                      "budgets": {"max_messages": MAX_MESSAGES,
                                  "max_message_chars": MAX_MESSAGE_CHARS,
                                  "max_total_chars": MAX_TOTAL_CHARS}}
            if self._storage_error:
                result["storage_error"] = self._storage_error
            return result

    def attach(self, data):
        """Attach a detached, JSON-safe trail to a result data dictionary."""
        if not isinstance(data, dict):
            raise TypeError("collaboration result data must be a dictionary")
        data["agent_collaboration"] = self.snapshot()
        return data["agent_collaboration"]

    def persist(self, run_dir):
        """Write only this run's named artifact, refusing links or other runs."""
        temporary = None
        try:
            root = workspace_path(run_dir, ".", "collaboration run", must_exist=True)
            target = workspace_path(root, "agent_collaboration.json", "collaboration artifact")
            target = Path(filesystem_path(target))

            def owned_existing():
                checked = workspace_path(root, "agent_collaboration.json", "collaboration artifact")
                checked = Path(filesystem_path(checked))
                if not checked.exists():
                    return False
                if not checked.is_file() or checked.stat().st_size > 1024 * 1024:
                    raise ValueError("existing collaboration artifact is not owned by this run")
                record = json.loads(checked.read_text(encoding="utf-8"))
                if (not isinstance(record, dict) or record.get("version") != 1
                        or record.get("run_id") != self.run_id or not isinstance(record.get("messages"), list)):
                    raise ValueError("existing collaboration artifact belongs to another run")
                return True

            exists = owned_existing()
            self._storage_error = ""
            encoded = json.dumps(self.snapshot(), ensure_ascii=False, indent=2, allow_nan=False)
            if not exists:
                with target.open("x", encoding="utf-8") as stream:
                    stream.write(encoded)
            else:
                temporary = target.with_name(f".agent_collaboration-{uuid.uuid4().hex}.tmp")
                with temporary.open("x", encoding="utf-8") as stream:
                    stream.write(encoded)
                owned_existing()
                os.replace(temporary, target)
                temporary = None
            self._storage_error = ""
            return str(target)
        except (OSError, ValueError, TypeError) as exc:
            self._storage_error = self._text(str(exc))[:500]
            return None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def collaborative_run(function):
    """Give nested services one session without serializing live session objects."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        previous_data = getattr(self, 'data', None)
        previous_receipt = previous_data.get('agent_collaboration') if isinstance(previous_data, dict) else None
        previous_run = previous_receipt.get('run_id') if isinstance(previous_receipt, dict) else None
        session = current_collaboration()
        if session is None:
            bound = signature.bind(self, *args, **kwargs)
            llm = getattr(self, "llm", None)
            session = CollaborationSession(on_event=bound.arguments.get("on_event"),
                logger=getattr(self, "logger", None), credential=getattr(llm, "api_key", ""))
        with session.scope():
            try:
                result = function(self, *args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get("data"), dict):
                    data = result["data"]
                    session.attach(data)
                    if data.get("run_dir"):
                        session.persist(data["run_dir"])
                        session.attach(data)
                return result
            except BaseException:
                # Earlier handoffs are still reviewable when a run is cancelled.
                # An early cancellation can precede a reused service's data
                # reset. That dictionary still belongs to the previous result.
                data = getattr(self, "data", None)
                receipt = data.get('agent_collaboration') if isinstance(data, dict) else None
                bound_run = receipt.get('run_id') if isinstance(receipt, dict) else None
                if isinstance(data, dict) and (data is not previous_data or not previous_run
                                              or bound_run == session.run_id):
                    session.attach(data)
                    if data.get("run_dir"):
                        session.persist(data["run_dir"])
                        session.attach(data)
                raise

    return wrapped
