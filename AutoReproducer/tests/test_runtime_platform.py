import pytest

from src import runtime_platform as runtime


@pytest.mark.parametrize("build,machine", [
    ("win-amd64", "AMD64"), ("win32", "x86"), ("win-arm64", "ARM64"),
])
def test_windows_fingerprint_uses_python_build_without_host_environment(monkeypatch, build, machine):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.sysconfig, "get_platform", lambda: build)
    monkeypatch.setattr(runtime.platform, "machine", lambda: "")
    assert runtime.runtime_fingerprint().endswith(f"-win32-{machine}")


def test_windows_32bit_python_does_not_share_64bit_dependencies(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.sysconfig, "get_platform", lambda: "win32")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    assert runtime.runtime_fingerprint().endswith("-win32-x86")


def test_unknown_windows_build_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.sysconfig, "get_platform", lambda: "win-unknown")
    with pytest.raises(RuntimeError, match="Unsupported Windows"):
        runtime.runtime_fingerprint()
