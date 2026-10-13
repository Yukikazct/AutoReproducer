"""Historical report updates must be independent of model/execution work."""
from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile
from unittest.mock import Mock

import pytest

from src.agents.report_generator import ReportGeneratorAgent
from src.report_revisions import (REPORT_FORMAT_VERSION, can_regenerate_report,
                                  load_report_revision_source, needs_report_upgrade,
                                  regenerate_report, report_format_version,
                                  resolve_report_revision, sanitize_report_result)
from frontend.report_renderer import report_downloads, render_report


@pytest.fixture
def legacy_result(tmp_path):
    report = "# 旧报告\n| 指标 | 单位 | 数据划分 | 阶段 | 来源 |\n"
    path = tmp_path / "report.md"
    path.write_text(report, encoding="utf-8")
    result = {"state": "COMPLETED", "error": None, "report_path": str(path), "data": {
        "paper_info": {"title": "Saved experiment"}, "report": report, "report_path": str(path),
        "execution": {"mode": "repository", "executed": True,
                      "final": {"success": True, "exit_code": 0, "stdout": "UNCHANGED LOG TAIL"}},
        "validation": {"status": "author_code_completed", "is_reproduced": None,
                       "reviews_removed": True, "metric_records": [
                           {"name": "accuracy", "value": 0.845, "unit": "fraction", "split": "test",
                            "stage": "author", "source": "stdout"}]},
        "audit_stats": {"llm_calls": 4}}}
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "stdout.log").write_text("UNCHANGED LOG TAIL", encoding="utf-8")
    return result


@pytest.fixture
def forbid_execution_and_api(monkeypatch):
    from src.llm.llm_client import LLMClient
    from src.agents.code_executor import CodeExecutorAgent
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Report rendering must not create agents, call models or run processes")

    monkeypatch.setattr(ReportGeneratorAgent, "__init__", forbidden)
    monkeypatch.setattr(ReportGeneratorAgent, "run", forbidden)
    monkeypatch.setattr(LLMClient, "__init__", forbidden)
    monkeypatch.setattr(CodeExecutorAgent, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def test_upgrade_is_local_and_keeps_original_evidence(legacy_result, tmp_path, forbid_execution_and_api):
    before = deepcopy(legacy_result)
    files = {name: (tmp_path / name).read_bytes() for name in ("report.md", "result.json", "stdout.log")}
    upgraded = regenerate_report(legacy_result)
    assert legacy_result == before
    for name, raw in files.items():
        assert (tmp_path / name).read_bytes() == raw
    assert report_format_version(upgraded["data"]["report"]) == REPORT_FORMAT_VERSION
    assert "| 指标 | 实测值 |" in upgraded["data"]["report"]
    assert "| accuracy | 0.845 | fraction | test | author | stdout |" in upgraded["data"]["report"]
    assert "UNCHANGED LOG TAIL" in upgraded["data"]["report"]
    assert upgraded["state"] == before["state"]
    assert upgraded["data"]["validation"] == before["data"]["validation"]
    assert upgraded["data"]["execution"] == before["data"]["execution"]
    assert upgraded["data"]["audit_stats"] == before["data"]["audit_stats"]
    assert load_report_revision_source(tmp_path / "report.md") == before
    assert regenerate_report(legacy_result) == upgraded  # idempotent, including timestamp


def test_reopened_result_display_markdown_and_zip_use_same_revision(legacy_result, forbid_execution_and_api):
    upgraded = regenerate_report(legacy_result)
    # Simulate rereading the original result.json after a process restart.
    reopened = resolve_report_revision(deepcopy(legacy_result))
    assert reopened == upgraded
    ui = Mock()
    render_report(reopened["data"]["report"], reopened["report_path"], st_module=ui)
    report_downloads(legacy_result, key="history", st_module=ui)
    markdown, bundle = [call.kwargs["data"] for call in ui.download_button.call_args_list]
    assert markdown == reopened["data"]["report"]
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.read(Path(reopened["report_path"]).name).decode() == markdown
    assert "0.845" in "".join(call.args[0] for call in ui.markdown.call_args_list)


def test_revision_does_not_replace_changed_result_or_tampered_report(legacy_result):
    upgraded = regenerate_report(legacy_result)
    changed = deepcopy(legacy_result)
    changed["data"]["validation"]["metric_records"][0]["value"] = 0.1
    assert resolve_report_revision(changed) == changed
    Path(upgraded["report_path"]).write_text("incorrect replacement", encoding="utf-8")
    assert resolve_report_revision(legacy_result) == legacy_result


def test_report_only_and_running_results_cannot_be_regenerated(legacy_result):
    assert needs_report_upgrade({"data": {"report": "old text"}})
    assert not can_regenerate_report({"data": {"report": "old text"}})
    with pytest.raises(ValueError, match="结构化"):
        regenerate_report({"data": {"report": "old text"}})
    legacy_result["state"] = "EXECUTE_CODE"
    with pytest.raises(ValueError, match="尚未结束"):
        regenerate_report(legacy_result)


def test_progress_only_report_revision_survives_restart(legacy_result, tmp_path):
    legacy_result.pop("report_path")
    legacy_result["data"].pop("report_path")
    upgraded = regenerate_report(legacy_result, data_root=tmp_path)
    assert resolve_report_revision(deepcopy(legacy_result), data_root=tmp_path) == upgraded


def test_current_old_tab_upgrade_and_server_restart(legacy_result, tmp_path, monkeypatch,
                                                 forbid_execution_and_api):
    from streamlit.testing.v1 import AppTest
    import frontend.history_manager as hm
    from frontend.backend_pipeline import ProgressStore

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    app_path = str(Path(__file__).parents[1] / "app.py")
    app = AppTest.from_file(app_path, default_timeout=120)
    app.session_state["result"] = deepcopy(legacy_result)
    app.run()
    assert not app.exception
    assert any("旧格式" in item.value for item in app.info)
    app.button(key="upgrade_current_report").click().run()
    assert not app.exception
    upgraded = app.session_state["result"]
    assert "| accuracy | 0.845 |" in "\n".join(item.value for item in app.markdown)
    assert not any(item.key == "upgrade_current_report" for item in app.button)
    assert len(app.get("download_button")) == 2

    # The original persisted done event still holds the old report. A new server
    # session resolves the saved rendering automatically, without rewriting it.
    progress = tmp_path / "progress_finished.jsonl"
    ProgressStore(str(progress)).emit({"type": "done", "result": legacy_result})
    before = progress.read_bytes()
    monkeypatch.setenv("AUTOREPRO_RESUME_PROGRESS", str(progress))
    restarted = AppTest.from_file(app_path, default_timeout=120).run()
    assert not restarted.exception
    assert restarted.session_state["result"] == upgraded
    assert "| accuracy | 0.845 |" in "\n".join(item.value for item in restarted.markdown)
    assert restarted.session_state["running"] is False
    assert progress.read_bytes() == before


def test_history_uses_structured_archive_and_preserved_revision_source(legacy_result, tmp_path, monkeypatch):
    import frontend.history_manager as hm

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    path = tmp_path / "report.md"
    archive = path.with_suffix(".result.json")
    archive.write_text(json.dumps(legacy_result), encoding="utf-8")
    session = {"state": "COMPLETED", "report_path": str(path)}
    loaded = hm.get_session_report_result(session)
    assert loaded == legacy_result
    updated = regenerate_report(loaded)
    archive.unlink()
    # Both runtime progress and the archive can be gone after cleanup. The
    # saved revision keeps its exact input, original report and newest rendering.
    assert hm.get_session_report_result(session) == updated


def test_history_done_session_id_wins_over_later_log_timestamp(tmp_path):
    from frontend.history_manager import _extract_session_from_progress

    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"type": "log", "log": {"timestamp": "2026-10-13T01:02:09"}})
                    + "\n" + json.dumps({"type": "done", "result": {"session_id": "20261013_010203"}}),
                    encoding="utf-8")
    assert _extract_session_from_progress(path) == "20261013_010203"


def test_upgrade_preserves_verified_figures_after_original_collection_cleanup(legacy_result, tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from src import execution_artifacts
    from src.report_revisions import REPORT_FORMAT_MARKER

    collected = tmp_path / "collected"
    collected.mkdir()
    monkeypatch.setattr(execution_artifacts, "ARTIFACT_ROOT", collected)
    figure = collected / "actual.png"
    Image.new("RGB", (20, 20), "blue").save(figure)
    raw = figure.read_bytes()
    legacy_result["data"]["execution"]["final"]["artifacts"] = [
        {"name": "actual.png", "path": str(figure), "sha256": hashlib.sha256(raw).hexdigest()}]
    legacy_result["data"]["report"] = ReportGeneratorAgent._build_report(
        legacy_result["data"], report_path=legacy_result["report_path"]).replace(REPORT_FORMAT_MARKER, "")
    Path(legacy_result["report_path"]).write_text(legacy_result["data"]["report"], encoding="utf-8")
    figure.unlink()
    updated = regenerate_report(legacy_result)
    ui = Mock()
    render_report(updated["data"]["report"], updated["report_path"], st_module=ui)
    ui.image.assert_called_once_with(raw, caption="运行结果图 1")
    ui.warning.assert_not_called()
    report_downloads(updated, key="figures", st_module=ui)
    with zipfile.ZipFile(io.BytesIO(ui.download_button.call_args_list[1].kwargs["data"])) as archive:
        images = [name for name in archive.namelist() if name.endswith(".png")]
        assert len(images) == 1
        assert archive.read(images[0]) == raw


@pytest.mark.parametrize("state, execution_update", [
    ("RUNNING", {}), (None, {"running": True}), ("COMPLETED", {"running": True}),
    (None, {"stages": [{"status": "running"}]}), ("ERROR", {"final": {"running": True}}),
])
def test_running_result_cannot_regenerate_or_resolve_finished_revision(legacy_result, state, execution_update):
    regenerate_report(legacy_result)
    legacy_result["state"] = state
    legacy_result["data"]["execution"].update(execution_update)
    assert not can_regenerate_report(legacy_result)
    assert resolve_report_revision(legacy_result) is legacy_result
    with pytest.raises(ValueError, match="尚未结束"):
        regenerate_report(legacy_result)


def test_unversioned_cli_result_needs_terminal_execution_evidence(legacy_result):
    legacy_result.pop("state")
    assert can_regenerate_report(legacy_result)  # confirmed exit code from old CLI
    legacy_result["data"]["execution"] = {}
    assert not can_regenerate_report(legacy_result)  # a paper/config is insufficient


@pytest.mark.parametrize("corrupt_report", [False, True])
def test_revision_source_rejects_modified_snapshot(legacy_result, tmp_path, corrupt_report):
    updated = regenerate_report(legacy_result)
    manifest_path = Path(updated["report_path"]).parent / "current.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corrupt_report:
        manifest["source_result"]["data"]["report"] = "modified old report"
    else:
        manifest["source_result"]["data"]["validation"]["metric_records"][0]["value"] = 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_report_revision_source(tmp_path / "report.md") is None


def test_new_archives_redact_nested_credentials_without_changing_metrics(legacy_result):
    legacy_result["data"]["llm_config"] = {
        "api_key": "test-api-credential", "headers": {"Authorization": "test-bearer-credential"},
        "providers": [{"OPENAI_API_KEY": "test-provider-credential", "access_token": "test-access-credential"}],
        "clientSecret": "test-secret-credential", "password": "test-password-credential",
        "model": "model-name", "total_tokens": 321}
    original = deepcopy(legacy_result)
    safe = sanitize_report_result(legacy_result)
    assert legacy_result == original
    assert safe["data"]["execution"] == original["data"]["execution"]
    assert safe["data"]["validation"] == original["data"]["validation"]
    assert safe["data"]["llm_config"]["total_tokens"] == 321
    assert safe["data"]["llm_config"]["model"] == "model-name"
    assert "credential" not in json.dumps(safe)
    updated = regenerate_report(legacy_result)
    manifest_path = Path(updated["report_path"]).parent / "current.json"
    assert "credential" not in manifest_path.read_text(encoding="utf-8")
    assert load_report_revision_source(legacy_result["report_path"]) == safe
    # Credentials do not affect report identity; reconfiguration must not prevent
    # reopening the saved report for exactly the same experiment evidence.
    legacy_result["data"]["llm_config"]["api_key"] = "changed-credential"
    assert resolve_report_revision(legacy_result)["data"]["report"] == updated["data"]["report"]


def test_history_supplies_terminal_state_for_old_config_only_failure(legacy_result, monkeypatch, tmp_path):
    import frontend.history_manager as hm

    monkeypatch.setattr(hm, "get_project_data_dir", lambda: tmp_path)
    legacy_result.pop("state")
    legacy_result["data"].pop("execution")
    path = tmp_path / "report.md"
    path.with_suffix(".result.json").write_text(json.dumps(legacy_result), encoding="utf-8")
    assert not can_regenerate_report(legacy_result)
    historical = hm.get_session_report_result({"state": "ERROR", "report_path": str(path)})
    assert historical["state"] == "ERROR"
    assert can_regenerate_report(historical)
