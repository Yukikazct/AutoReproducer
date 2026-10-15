"""Dependency progress observes installer work without touching capture handles."""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from src.dependency_install_progress import DependencyInstallProgress, parse_pip_line
from src.runtime_preparation import run_owned_process


WHEEL = "demo-1.0-py3-none-any.whl"
HTTP_LINE = (
    '2026-10-15T12:00:00,000 https://files.pythonhosted.org:443 '
    f'"GET /packages/{WHEEL} HTTP/1.1" 200 20480'
)


def test_parser_extracts_exact_http_wheel_length():
    parsed = parse_pip_line(HTTP_LINE)

    assert parsed is not None
    assert parsed["phase"] == "downloading"
    assert parsed["package"] == "demo"
    assert parsed["file_name"] == WHEEL
    assert parsed["total_bytes"] == 20480
    assert parsed["source_domain"] == "files.pythonhosted.org"


@pytest.mark.parametrize("line", [
    f"Downloading {WHEEL} (20.5 kB)",
    f"Downloading https://files.pythonhosted.org/packages/{WHEEL} (1.2 MB)",
])
def test_parser_does_not_present_rounded_display_sizes_as_exact_bytes(line):
    parsed = parse_pip_line(line)

    assert parsed is not None
    assert parsed["phase"] == "downloading"
    assert parsed["file_name"] == WHEEL
    assert parsed.get("total_bytes") is None


@pytest.mark.parametrize("line", [
    HTTP_LINE.replace(".whl HTTP", ".whl.metadata HTTP"),
    HTTP_LINE.replace("200 20480", "404 20480"),
    HTTP_LINE.replace("200 20480", "200 None"),
    "  Progress: 20.5/20.5 kB 3.0 MB/s eta 0:00:00",
])
def test_parser_does_not_invent_total_from_metadata_error_or_progress_text(line):
    parsed = parse_pip_line(line)

    assert parsed is None or parsed.get("total_bytes") is None


def test_parser_reports_installation_phase_without_raw_log_text():
    parsed = parse_pip_line("Installing collected packages: demo, other-demo")

    assert parsed is not None
    assert parsed["phase"] == "installing"
    assert "Installing collected packages:" not in json.dumps(parsed)


@pytest.mark.parametrize("line", [
    f"Downloading https://alice:password-secret@example.invalid/{WHEEL}?token=token-secret (20.5 kB)",
    'https://alice:password-secret@example.invalid:443 '
    f'"GET /packages/{WHEEL}?token=token-secret HTTP/1.1" 200 20480',
    f"Downloading https://credential_demo-9.0-py3-none-any.whl:password-secret@example.invalid/{WHEEL}?token=token-secret (20.5 kB)",
    "Looking in indexes: https://alice:password-secret@example.invalid/simple?token=token-secret",
])
def test_parser_never_returns_repository_credentials_or_urls(line):
    parsed = parse_pip_line(line)

    encoded = json.dumps(parsed)
    for secret in ("alice", "credential_demo", "password-secret", "token-secret", "https://", "token=", "@"):
        assert secret not in encoded
    if parsed is not None:
        assert parsed["source_domain"] == "example.invalid"
        assert parsed["file_name"] == WHEEL


def test_parser_does_not_mistake_filename_shaped_userinfo_for_download():
    line = (
        "Downloading https://credential_demo-9.0-py3-none-any.whl:"
        "password-secret@example.invalid?token=token-secret (20.5 kB)"
    )

    assert parse_pip_line(line) is None


def test_live_open_wheel_growth_is_visible_before_child_exits(tmp_path):
    events = []
    live_events = []
    current_process = [None]

    def observe(event):
        events.append(dict(event))
        process = current_process[0]
        if process is not None and process.poll() is None:
            live_events.append(dict(event))

    monitor = DependencyInstallProgress(tmp_path, observe, timeout_s=10, interval_s=.1)
    assert monitor.log_path == tmp_path / "pip-progress.log"

    def poll(process):
        current_process[0] = process
        monitor(process)

    # A flushed wheel remains open through each poll. This reproduces pip's
    # Windows write sharing behavior instead of substituting a completed file.
    program = """
import pathlib, sys, time
work = pathlib.Path(sys.argv[1])
wheel_name = sys.argv[2]
print('stdout-before', flush=True)
print('stderr-before', file=sys.stderr, flush=True)
wheel_dir = work / 'pip-unpack-fixture'
wheel_dir.mkdir()
with (work / 'pip-progress.log').open('w', encoding='utf-8') as log:
    log.write(sys.argv[3] + '\\n')
    log.write('Downloading ' + wheel_name + ' (20.5 kB)\\n')
    log.flush()
    with (wheel_dir / wheel_name).open('wb') as wheel:
        for index in range(5):
            wheel.write(b'x' * 4096)
            wheel.flush()
            time.sleep(.3)
    log.write('Installing collected packages: demo\\n')
    log.flush()
    time.sleep(.3)
print('stdout-after', flush=True)
print('stderr-after', file=sys.stderr, flush=True)
"""
    completed = run_owned_process(
        [sys.executable, "-I", "-u", "-c", program, str(tmp_path), WHEEL, HTTP_LINE],
        timeout_s=10, on_poll=poll,
    )
    monitor.finish("success")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["stdout-before", "stdout-after"]
    assert completed.stderr.splitlines() == ["stderr-before", "stderr-after"]
    downloading = [event for event in live_events if event["phase"] == "downloading"]
    observed_bytes = [event["downloaded_bytes"] for event in downloading
                      if event.get("downloaded_bytes") is not None]
    assert len(set(observed_bytes)) >= 3
    assert any(0 < value < 20480 for value in observed_bytes)
    assert observed_bytes == sorted(observed_bytes)
    assert any(event["total_bytes"] == 20480 for event in downloading)
    assert any((event.get("download_speed_bps") or 0) > 0 for event in downloading)
    assert any(event["phase"] == "installing" for event in live_events)
    assert events[-1]["status"] == "success"
    assert len({event["installation_id"] for event in events}) == 1
    assert all(event["type"] == "dependency_install_progress" for event in events)
    assert all(event["timeout_s"] == 10 for event in events)
    assert all(event["elapsed_s"] >= 0 for event in events)
    assert all(isinstance(event["last_activity_at"], (int, float)) for event in events)
    assert all(1_500_000_000 < event["last_activity_at"] <= time.time() for event in events)
    required = {"package", "downloaded_bytes", "total_bytes", "download_speed_bps", "message"}
    assert all(required <= event.keys() for event in events)


def test_stalled_install_emits_heartbeats_without_faking_activity(tmp_path):
    events = []
    monitor = DependencyInstallProgress(tmp_path, events.append, timeout_s=30, interval_s=.1)
    process = SimpleNamespace(poll=lambda: None)
    monitor(process)
    for _ in range(3):
        time.sleep(.13)
        monitor(process)

    assert len(events) >= 3
    assert all(event["status"] == "running" for event in events)
    assert len({event["last_activity_at"] for event in events}) == 1
    assert events[-1]["elapsed_s"] > events[0]["elapsed_s"]
    assert all(event.get("downloaded_bytes") is None for event in events)
    assert all(event.get("total_bytes") is None for event in events)


@pytest.mark.parametrize("failure_mode", ["callback", "unreadable_log"])
def test_monitor_failures_cannot_abort_owned_child(tmp_path, failure_mode):
    callbacks = []

    def observe(event):
        callbacks.append(event)
        if failure_mode == "callback":
            raise RuntimeError("progress consumer unavailable")

    monitor = DependencyInstallProgress(tmp_path, observe, timeout_s=10, interval_s=.1)
    if failure_mode == "unreadable_log":
        # Opening a directory as a file fails portably without chmod/root quirks.
        monitor.log_path.mkdir()
    result = run_owned_process(
        [sys.executable, "-I", "-c", "import time;time.sleep(.4);print('child completed')"],
        timeout_s=10, on_poll=monitor,
    )
    monitor.finish("success")

    assert result.returncode == 0
    assert result.stdout.strip() == "child completed"
    assert callbacks


@pytest.mark.parametrize("status", ["success", "error", "cancelled"])
def test_final_event_identifies_attempt_and_terminal_status(tmp_path, status):
    first_events, second_events = [], []
    first = DependencyInstallProgress(tmp_path / "first", first_events.append, timeout_s=30)
    second = DependencyInstallProgress(tmp_path / "second", second_events.append, timeout_s=30)

    first.finish(status)
    second.finish(status)

    assert first_events[-1]["status"] == status
    assert second_events[-1]["status"] == status
    assert first_events[-1]["installation_id"] != second_events[-1]["installation_id"]


def test_monitor_events_do_not_leak_log_credentials(tmp_path):
    events = []
    monitor = DependencyInstallProgress(tmp_path, events.append, timeout_s=30, interval_s=.1)
    monitor.log_path.write_text(
        'Looking in indexes: https://alice:password-secret@example.invalid/simple\n'
        f'Downloading https://alice:password-secret@example.invalid/{WHEEL}?token=token-secret (20.5 kB)\n',
        encoding="utf-8",
    )
    monitor(SimpleNamespace(poll=lambda: None))
    monitor.finish("error")

    assert events
    assert all(event["source_domain"] == "example.invalid" for event in events)
    encoded = json.dumps(events)
    for secret in ("alice", "password-secret", "token-secret", "https://", "token=", "@"):
        assert secret not in encoded
