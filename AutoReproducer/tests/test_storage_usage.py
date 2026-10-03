"""磁盘数据必须来自实际文件或 Docker；未知、部分统计和共享空间分别标注。"""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
import src.storage_usage as usage
from src.agents.code_executor import CodeExecutorAgent
from src.agents.env_builder import EnvBuilderAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.llm.llm_client import LLMClient


@pytest.fixture(autouse=True)
def isolated_dependency_state(tmp_path, monkeypatch):
    ce._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce, "DEPS_CACHE_ROOT", tmp_path / "cache")
    yield
    ce._INSTALLED_DEPS.clear()


@pytest.mark.parametrize("estimate", [3.0, 500, "unknown", None])
def test_model_disk_estimate_is_ignored_even_when_returned(estimate):
    llm = Mock(spec=LLMClient)
    llm.get_call_count.return_value = 1
    llm.chat.return_value = json.dumps({"required_packages": [], "estimated_disk_gb": estimate})
    result = EnvBuilderAgent(llm, logger=Mock()).run({"paper_info": {}})["env_config"]
    assert result["estimated_disk_gb"] is None
    assert result["disk_usage"]["status"] == "not_measured"
    assert result["disk_usage"]["components"] == []


def test_mock_environment_no_longer_claims_fixed_three_gigabytes():
    env = EnvBuilderAgent(LLMClient(mock_mode=True), logger=Mock()).run({})["env_config"]
    assert env["estimated_disk_gb"] is None
    report = ReportGeneratorAgent(logger=Mock())._build_report({"env_config": env})
    assert "未测量" in report
    assert "3.0 GB" not in report
    assert "预估磁盘" not in report


def test_actual_file_lengths_count_nested_files_once_and_skip_links(tmp_path):
    root = tmp_path / "workspace"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a" * 11)
    (nested / "b.bin").write_bytes(b"b" * 29)
    (root / "hardlink.bin").hardlink_to(root / "a.bin")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large.bin").write_bytes(b"x" * 1000)
    (root / "linked-file").symlink_to(outside / "large.bin")
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    stat = usage.directory_usage(root, "workspace", "工作区")
    assert stat["status"] == "measured"
    assert stat["bytes"] == 40
    assert stat["files"] == 2


def test_empty_directory_and_missing_directory_are_not_confused(tmp_path):
    empty = usage.directory_usage(tmp_path, "workspace", "工作区")
    missing = usage.directory_usage(tmp_path / "missing", "workspace", "工作区")
    assert empty["status"] == "measured" and empty["bytes"] == 0
    assert missing["status"] == "not_measured" and missing["bytes"] is None


def test_directory_symlink_is_not_followed(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "payload").write_bytes(b"x" * 999)
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    result = usage.directory_usage(tmp_path / "link", "workspace", "工作区")
    assert result["status"] == "not_measured"
    assert result["bytes"] is None


def test_dependency_directory_is_excluded_from_workspace_measurement(tmp_path):
    (tmp_path / "run.py").write_bytes(b"code")
    (tmp_path / ".autorepro_deps").mkdir()
    (tmp_path / ".autorepro_deps" / "package.bin").write_bytes(b"x" * 1000)
    result = usage.directory_usage(tmp_path, "workspace", "工作区", exclude=(".autorepro_deps",))
    assert result["bytes"] == 4


def test_sparse_file_logical_length_is_not_claimed_as_allocated_disk_space(tmp_path):
    path = tmp_path / "sparse.bin"
    with path.open("wb") as stream:
        stream.seek(2 * 1024 * 1024)
        stream.write(b"x")
    result = usage.directory_usage(tmp_path, "workspace", "工作区")
    assert result["bytes"] == 2 * 1024 * 1024 + 1
    if hasattr(path.stat(), "st_blocks"):
        assert result["allocated_bytes"] == path.stat().st_blocks * 512


def test_scan_limit_is_marked_partial(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "b").write_bytes(b"x" * 20)
    result = usage.directory_usage(tmp_path, "workspace", "工作区", max_entries=1)
    assert result["status"] == "partial"
    assert result["bytes"] < 30
    assert "不完整" in result["note"]


def test_unreadable_subdirectory_does_not_disappear_from_completeness(tmp_path, monkeypatch):
    (tmp_path / "visible.bin").write_bytes(b"x" * 10)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.bin").write_bytes(b"x" * 20)
    original = usage.os.scandir
    def guarded(path):
        if Path(path) == blocked:
            raise PermissionError("blocked")
        return original(path)
    monkeypatch.setattr(usage.os, "scandir", guarded)
    result = usage.directory_usage(tmp_path, "workspace", "工作区")
    assert result["status"] == "partial"
    assert result["bytes"] == 10


def test_docker_size_comes_from_inspected_image_metadata(monkeypatch):
    captured = {}
    def inspect(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "sha256:" + "a" * 64 + " 1234567\n", "")
    monkeypatch.setattr(usage.subprocess, "run", inspect)
    result = usage.docker_image_usage("docker", "python:3.11-slim")
    assert result["status"] == "measured"
    assert result["bytes"] == 1234567
    assert captured["cmd"][1:3] == ["image", "inspect"]
    assert result["shared"] is True
    assert "不等于本次新增" in result["note"]


@pytest.mark.parametrize("code,stdout", [(1, ""), (0, "ok"), (0, "sha256:abc -1"), (0, "sha256:abc unknown")])
def test_failed_or_invalid_image_query_never_falls_back_to_default_size(monkeypatch, code, stdout):
    monkeypatch.setattr(usage.subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, code, stdout, ""))
    result = usage.docker_image_usage("docker", "python:3.11-slim")
    assert result["status"] == "not_measured"
    assert result["bytes"] is None


def test_image_query_timeout_does_not_break_execution_result(monkeypatch):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)
    monkeypatch.setattr(usage.subprocess, "run", timeout)
    assert usage.docker_image_usage("docker", "python:3.11-slim")["bytes"] is None


def components(snapshot):
    return {item["key"]: item for item in snapshot["components"]}


def test_real_execution_is_measured_before_temporary_workspace_cleanup():
    code = "from pathlib import Path\nPath('payload.bin').write_bytes(b'x' * 4096)\nprint('done')\n"
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    result = executor.run({"code": code, "env_config": {"estimated_disk_gb": 99}})
    assert result["success"] is True
    measured = result["final"]["disk_usage"]
    workspace = components(measured)["workspace"]
    assert workspace["bytes"] == len(result["code"].encode()) + 4096
    assert workspace["retained"] is False
    assert not Path(workspace["source"]).exists()
    assert result["effective_env_config"]["disk_usage"] == measured
    assert result["effective_env_config"]["estimated_disk_gb"] is None
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": result})
    assert "文件体积实测" in report
    assert "清理前快照" in report
    assert "99 GB" not in report


def test_real_persistent_workspace_and_shared_dependency_cache_are_labeled(tmp_path):
    cache = tmp_path / "deps"
    cache.mkdir()
    (cache / "package.bin").write_bytes(b"x" * 1024)
    workspace = tmp_path / "workspace"
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock())
    executor._deps_dir = str(cache)
    result = executor.execute_in_workspace("print('ok')\n", str(workspace), stage="full")
    measured = components(result["disk_usage"])
    assert measured["workspace"]["retained"] is True
    assert workspace.exists()
    assert measured["dependency_cache_1"]["bytes"] == 1024
    assert measured["dependency_cache_1"]["shared"] is True


def test_mock_skipped_installation_is_not_reported_as_real_dependency_size(tmp_path):
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    executor.env_config = {"requirements_txt": "demo-package"}
    result = executor.execute_in_workspace("print('ok')\n", str(tmp_path), stage="full")
    dependency = components(result["disk_usage"])["dependencies"]
    assert dependency["status"] == "not_measured"
    assert dependency["bytes"] is None
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": {"final": result}})
    assert "跳过依赖安装" in report
    assert "**隔离依赖文件体积**: 未测量" in report


def test_failed_execution_still_keeps_actual_workspace_measurement(tmp_path):
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    code = "from pathlib import Path\nPath('partial.bin').write_bytes(b'x' * 99)\nraise RuntimeError('bad')\n"
    result = executor.execute_in_workspace(code, str(tmp_path), stage="full")
    assert result["success"] is False
    assert components(result["disk_usage"])["workspace"]["bytes"] == len(code.encode()) + 99


def test_old_model_estimate_is_not_displayed_as_measured_size():
    report = ReportGeneratorAgent(logger=Mock())._build_report({"env_config": {"estimated_disk_gb": 3.0}})
    assert "磁盘统计**: 未测量" in report
    assert "3.0 GB" not in report


def test_unreadable_directories_are_not_labeled_partially_measured(tmp_path):
    snapshot = usage.disk_usage_snapshot([
        usage.directory_usage(tmp_path / "missing", "workspace", "工作区")])
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": {"final": {"disk_usage": snapshot}}})
    assert "磁盘统计**: 未测量" in report
    assert "部分已测量" not in report
    assert "文件体积实测" not in report


def test_rejected_new_run_does_not_reuse_previous_disk_snapshot():
    old = usage.disk_usage_snapshot([
        {"key": "workspace", "label": "工作区", "status": "measured", "bytes": 999999}])
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), logger=Mock(), mock_mode=True)
    result = executor.run({"code": "def broken(:\n", "env_config": {"disk_usage": old}})
    assert result["not_runnable"] is True
    assert result["effective_env_config"]["disk_usage"]["status"] == "not_measured"
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": result, "env_config": {"disk_usage": old}})
    assert "磁盘统计**: 未测量" in report


def test_partial_statistic_and_image_shared_layers_are_explicit_in_report():
    snapshot = usage.disk_usage_snapshot([
        {"key": "workspace", "label": "执行工作区文件体积", "basis": "file_stat",
         "status": "partial", "bytes": 1024, "retained": False, "note": "统计不完整"},
        {"key": "docker_image", "label": "Docker 镜像内容体积", "basis": "docker_image_inspect",
         "status": "measured", "bytes": 2 * 1024 ** 3, "shared": True},
    ])
    report = ReportGeneratorAgent(logger=Mock())._build_report({"execution": {"final": {"disk_usage": snapshot}}})
    assert "已读取部分 1.00 KiB，统计不完整" in report
    assert "2.00 GiB" in report
    assert "含共享层" in report
    assert "不等于本次新增磁盘占用" in report


def test_measurements_have_readable_small_sizes_and_no_nan_defaults():
    assert usage.format_bytes(0) == "0 B"
    assert usage.format_bytes(1536) == "1.50 KiB"
    assert usage.format_bytes(None) == "未测量"
    assert usage.format_bytes(-1) == "未测量"
