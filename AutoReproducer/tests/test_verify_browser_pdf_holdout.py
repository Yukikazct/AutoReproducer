"""Local browser-audit regressions; no browser, network, or training required."""
from contextlib import contextmanager
import getpass
import hashlib
import io
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import warnings

import pytest

from scripts import verify_browser_pdf_holdout as audit


def test_stdin_credential_never_opens_secrets_or_prompts(monkeypatch):
    monkeypatch.setattr(audit.sys, "stdin", io.StringIO('{"api_key": "fixture-key-only"}'))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Must not read any credential file"))
    monkeypatch.setattr(audit.getpass, "getpass", lambda *a: pytest.fail("Must not prompt in stdin mode"))
    assert audit.read_credential(SimpleNamespace(credential_stdin=True)) == "fixture-key-only"


def test_default_credential_uses_hidden_prompt_and_refuses_echo_fallback(monkeypatch):
    monkeypatch.setattr(audit.getpass, "getpass", lambda prompt: "fixture-key-only")
    assert audit.read_credential(SimpleNamespace(credential_stdin=False)) == "fixture-key-only"

    def unsafe_fallback(prompt):
        warnings.warn("Cannot control echo", getpass.GetPassWarning)
        pytest.fail("Echo fallback must stop before reading input")

    monkeypatch.setattr(audit.getpass, "getpass", unsafe_fallback)
    with pytest.raises(getpass.GetPassWarning):
        audit.read_credential(SimpleNamespace(credential_stdin=False))


class Field:
    def __init__(self, value="", checked=False):
        self.value, self.checked, self.presses = value, checked, []

    def fill(self, value):
        self.value = value

    def press(self, value):
        self.presses.append(value)

    def input_value(self):
        return self.value

    def is_checked(self):
        return self.checked

    def set_checked(self, checked):
        self.checked = checked


class Page:
    def __init__(self):
        self.fields = {
            "API 地址（OpenAI 兼容）": Field("https://provider.example/v1/"),
            "模型名称": Field("custom.model+version"),
            "API Key": Field(),
            "Mock 演示模式（无需 API）": Field(checked=True),
            "Docker 沙箱执行（真实模式）": Field(checked=True),
        }
        self.expanded, self.tested, self.waited = [], False, False

    def get_by_label(self, name, *, exact):
        return self.fields[name]

    def get_by_role(self, role, *, name, exact):
        assert (role, name) == ("button", "测试 AI 连接")
        return SimpleNamespace(click=self.connection_test)

    def connection_test(self):
        self.tested = True

    def get_by_text(self, text, **kwargs):
        if isinstance(text, str):
            assert text in {"运行环境", "LLM API 配置"}
            return SimpleNamespace(click=lambda: self.expanded.append(text))
        expected = ("连接成功 (" + self.fields["模型名称"].value + " @ "
                    + self.fields["API 地址（OpenAI 兼容）"].value + "): OK")
        assert text.search(expected), "Success must use escaped page provider and model"
        return SimpleNamespace(wait_for=self.wait_for_connection)

    def wait_for_connection(self, *, timeout):
        assert self.tested
        self.waited = True


@pytest.mark.parametrize("overrides", [
    {},
    {"base_url": "https://other.example/api/", "model": "other/model(v2)"},
])
def test_configuration_uses_current_labels_and_page_provider(overrides):
    page = Page()
    config = audit.configure_page(
        page, SimpleNamespace(base_url=overrides.get("base_url"), model=overrides.get("model")),
        "fixture-key-only")
    assert page.waited and page.expanded == ["运行环境", "LLM API 配置"]
    assert config == {
        "base_url": overrides.get("base_url", "https://provider.example/v1/"),
        "model": overrides.get("model", "custom.model+version"),
    }
    assert page.fields["API Key"].presses == ["Enter"]
    assert not page.fields["Mock 演示模式（无需 API）"].checked
    assert not page.fields["Docker 沙箱执行（真实模式）"].checked


def test_browser_is_portable_and_accepts_explicit_configuration():
    args = SimpleNamespace(headed=False, browser_executable=None, browser_channel=None)
    assert audit.browser_launch_options(args) == {"headless": True}
    args.browser_channel = "msedge"
    assert audit.browser_launch_options(args) == {"headless": True, "channel": "msedge"}
    args.browser_channel, args.browser_executable, args.headed = None, "custom-browser.exe", True
    assert audit.browser_launch_options(args) == {"headless": False, "executable_path": "custom-browser.exe"}
    parser = audit.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--pdf", "a", "--repo", "b", "--freeze", "c", "--tag", "d",
                           "--browser-executable", "a", "--browser-channel", "chrome"])


@pytest.fixture
def completed_run(tmp_path):
    data_root = tmp_path / "data"
    deps = data_root / "deps"
    deps.mkdir(parents=True)
    (deps / ".autorepro-cache.lock").touch()
    run_dir = data_root / "runs" / "example"
    run_dir.mkdir(parents=True)
    workspace = run_dir / "repository"
    workspace.mkdir()
    plan = {"scope": "author_code_execution", "steps": [
        {"id": "train", "kind": "train", "command": ["python", "main.py"]}]}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
    (run_dir / "experiment_spec.json").write_text(json.dumps({"spec": plan, "sha256": digest}), "utf-8")
    cleanup = {
        "status": "released", "errors": [], "remaining": [],
        "removed": [{"path": "repository/runtime", "bytes": 32}], "freed_bytes": 96,
        "training_python": {"status": "released", "errors": [],
                            "removed": [{"path": str(data_root / "training_python"), "bytes": 64}]},
        "docker": {"status": "released", "errors": []},
        "runtime_downloads": {"status": "released", "errors": []},
    }
    report = data_root / "reports" / "example.md"
    report.parent.mkdir()
    report.write_text("# Saved report", "utf-8")
    (run_dir / "report.md").write_text("# Saved run report", "utf-8")
    validation = {
        "status": "author_code_completed", "is_reproduced": None,
        "paper_metrics_verified": False, "protocol_pass": None, "independent_metrics_pass": None,
        "training_observed": False,
        "completion_checks": {"all_required_author_commands_returned": True,
                              "observed_training_updates": False, "owned_processes_finished": True},
        "author_execution": [{"step": plan["steps"][0], "spec_sha256": digest,
                              "natural_return": True, "optimizer_steps": 0}],
    }
    result = {
        "state": "COMPLETED", "report_path": str(report), "error": None,
        "data": {
            "run_dir": str(run_dir), "report_path": str(report), "environment_cleanup": cleanup,
            "validation": validation, "spec_sha256": digest, "repository": {"path": str(workspace)},
            "reproduction_scope": "author_code_execution", "execution_source": "author_repository",
            "repository_executed": True, "pdf_input": {"sha256": "pdf-digest"},
            "resources": {"code_repo_url": "https://example.org/author/repo",
                          "repository_identity": {"status": "unreviewed_execution", "author_identity_confirmed": False}},
            "execution": {"success": True, "steps": [
                {"id": "train", "success": True, "executed": True, "exit_code": 0}]},
        },
    }
    for path, value in [(run_dir / "result.json", result),
                        (report.with_suffix(".result.json"), result),
                        (run_dir / "environment_cleanup.json", cleanup)]:
        path.write_text(json.dumps(value), "utf-8")
    return result, data_root, deps


def test_current_completion_allows_no_observed_optimizer_updates(completed_run):
    result, _, _ = completed_run
    validation = audit.verify_author_completion(result, "pdf-digest", "https://example.org/author/repo/")
    assert validation["training_observed"] is False
    # Informational native/dataset/review results are deliberately absent.
    evidence = audit.cleanup_evidence(*completed_run)
    assert evidence["passed"], evidence["checks"]
    assert evidence["receipt"]["training_python"]["removed"]


@pytest.mark.parametrize("failure", ["exit", "identity", "receipt", "duplicate", "training"])
def test_execution_audit_rejects_inconsistent_completion(completed_run, failure):
    result, _, _ = completed_run
    data = result["data"]
    if failure == "exit":
        data["execution"]["steps"][0]["exit_code"] = 1
    elif failure == "identity":
        data["resources"]["repository_identity"]["author_identity_confirmed"] = True
    elif failure == "receipt":
        data["validation"]["author_execution"][0]["spec_sha256"] = "different"
    elif failure == "duplicate":
        data["execution"]["steps"].append(data["execution"]["steps"][0].copy())
    elif failure == "training":
        data["validation"]["training_observed"] = True
    with pytest.raises(AssertionError):
        audit.verify_author_completion(result, "pdf-digest", "https://example.org/author/repo")


@pytest.mark.parametrize(("leftover", "check"), [
    ("package", "dependency_files_removed"),
    ("training", "training_environments_removed"),
    ("custom-training", "training_environments_removed"),
    ("report", "final_report_retained"),
    ("receipt", "cleanup_receipt_matches"),
    ("archive", "report_result_archive_cleanup_matches"),
])
def test_cleanup_audit_checks_actual_files_and_persisted_receipts(completed_run, leftover, check):
    result, data_root, deps = completed_run
    if leftover == "package":
        (deps / "installed-package").mkdir()
    elif leftover == "training":
        (data_root / "training_python").mkdir()
    elif leftover == "custom-training":
        custom = data_root.parent / "custom" / "training_python"
        custom.mkdir(parents=True)
        result["data"]["environment_cleanup"]["before_run"] = {
            "training_python": {"removed": [{"path": str(custom), "bytes": 12}]}}
    elif leftover == "report":
        Path(result["report_path"]).unlink()
    elif leftover == "receipt":
        (Path(result["data"]["run_dir"]) / "environment_cleanup.json").write_text("{}", "utf-8")
    elif leftover == "archive":
        Path(result["report_path"]).with_suffix(".result.json").write_text("{}", "utf-8")
    evidence = audit.cleanup_evidence(result, data_root, deps)
    assert evidence["checks"][check] is False
    assert evidence["passed"] is False


def test_failed_run_missing_paths_still_produces_cleanup_evidence(tmp_path):
    evidence = audit.cleanup_evidence({"state": "ERROR", "data": {}}, tmp_path, tmp_path / "deps")
    assert evidence["passed"] is False
    assert evidence["checks"]["final_report_retained"] is False


def test_evidence_redacts_credentials_in_nested_values_and_keys():
    redacted = audit.redact({"fixture-key-only": ["failure: fixture-key-only", {"body": "fixture-key-only"}]},
                            "fixture-key-only")
    assert "fixture-key-only" not in json.dumps(redacted)
    assert redacted == {"[REDACTED]": ["failure: [REDACTED]", {"body": "[REDACTED]"}]}


def test_browser_closes_before_playwright_stops_and_preserves_redacted_failure(tmp_path, monkeypatch):
    events = []
    running = False
    page = SimpleNamespace(
        goto=lambda *args, **kwargs: None,
        get_by_role=lambda *args, **kwargs: SimpleNamespace(wait_for=lambda **kwargs: None),
    )

    class Browser:
        def new_page(self, **kwargs):
            return page

        def is_connected(self):
            return True

        def close(self):
            if not running:
                raise RuntimeError("Event loop is closed! Is Playwright already stopped?")
            events.append("browser.close")

    @contextmanager
    def playwright_context():
        nonlocal running
        running = True
        try:
            yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kwargs: Browser()))
        finally:
            running = False
            events.append("playwright.stop")

    playwright_api = ModuleType("playwright.sync_api")
    playwright_api.sync_playwright = playwright_context
    monkeypatch.setitem(audit.sys.modules, "playwright", ModuleType("playwright"))
    monkeypatch.setitem(audit.sys.modules, "playwright.sync_api", playwright_api)
    monkeypatch.setattr(audit, "read_credential", lambda args: "fixture-key-only")

    def failed_configuration(*args):
        raise ValueError("connection failed: fixture-key-only")

    monkeypatch.setattr(audit, "configure_page", failed_configuration)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"test PDF bytes")
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"files": {}}', "utf-8")
    data = tmp_path / "data"
    args = audit.build_parser().parse_args([
        "--pdf", str(pdf), "--repo", "https://example.org/author/repo",
        "--freeze", str(freeze), "--tag", "browser-error", "--data-root", str(data),
        "--deps-root", str(data / "deps"),
    ])
    with pytest.raises(RuntimeError, match=r"connection failed: \[REDACTED\]"):
        audit.main(args)
    assert events == ["browser.close", "playwright.stop"]
    failure = json.loads((data / "runtime" / "browser-error_failure.json").read_text("utf-8"))
    assert failure == {"error": "connection failed: [REDACTED]", "type": "ValueError"}
