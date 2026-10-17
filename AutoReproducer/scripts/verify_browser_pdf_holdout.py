"""Upload a PDF through the browser and audit author execution and release.

Credentials are supplied by a hidden prompt or JSON on stdin, never a secrets file.
Use data/dependency roots that match the running application.
Independent acceptance additionally requires a paper-bound freeze and shared registry.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import warnings
import io
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.holdout_registry import repository_key, verify_freeze


def read_credential(args):
    if args.credential_stdin:
        key = json.loads(sys.stdin.read()).get("api_key")
    else:
        # Refuse getpass's echoed-input fallback on terminals without hidden input.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("API Key (hidden): ")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Supply a nonempty API key through hidden input or stdin JSON")
    return key.strip()


def redact(value, credential):
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]") if credential else value
    if isinstance(value, dict):
        return {redact(key, credential): redact(item, credential) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, credential) for item in value]
    return value


def browser_launch_options(args):
    options = {"headless": not args.headed}
    if args.browser_executable:
        options["executable_path"] = str(Path(args.browser_executable).expanduser())
    elif args.browser_channel:
        options["channel"] = args.browser_channel
    return options


def configure_page(page, args, credential):
    page.get_by_text("运行环境", exact=True).click()
    mock = page.get_by_label("Mock 演示模式（无需 API）", exact=True)
    if mock.is_checked():
        mock.set_checked(False)
    docker = page.get_by_label("Docker 沙箱执行（真实模式）", exact=True)
    if docker.is_checked():
        docker.set_checked(False)
    page.get_by_text("LLM API 配置", exact=True).click()
    fields = {"base_url": "API 地址（OpenAI 兼容）", "model": "模型名称"}
    config = {}
    for name, label in fields.items():
        field = page.get_by_label(label, exact=True)
        override = getattr(args, name)
        if override is not None:
            if not override.strip():
                raise ValueError(f"--{name.replace('_', '-')} must be nonempty")
            field.fill(override.strip())
            field.press("Enter")
        config[name] = field.input_value().strip()
        assert config[name], f"The page must have an effective {name}"
    password = page.get_by_label("API Key", exact=True)
    password.fill(credential)
    password.press("Enter")
    page.get_by_role("button", name="测试 AI 连接", exact=True).click()
    message = "连接成功 (" + config["model"] + " @ " + config["base_url"] + "):"
    page.get_by_text(re.compile(re.escape(message))).wait_for(timeout=45000)
    assert password.input_value() == credential, "Credential changed after connection test"
    return config


def cleanup_evidence(result, data_root, deps_root):
    """Compare the final cleanup receipt with remaining files and retained reports."""
    data = result.get("data") or {}
    receipt = data.get("environment_cleanup") or {}
    training = receipt.get("training_python") or {}
    checks = {
        "cleanup_released": receipt.get("status") == "released",
        "cleanup_has_no_errors": receipt.get("errors") == [],
        "cleanup_has_no_remaining": receipt.get("remaining") == [],
        "training_python_released": training.get("status") == "released" and training.get("errors") == [],
    }
    for name in ("docker", "runtime_downloads"):
        section = receipt.get(name) or {}
        checks[name + "_released"] = section.get("status") == "released" and section.get("errors") == []

    residues = []
    if deps_root.exists():
        if not deps_root.is_dir():
            residues.append(str(deps_root))
        else:
            residues = [str(path) for path in deps_root.iterdir()
                        if path.name != ".autorepro-cache.lock" or not path.is_file() or path.is_symlink()]
    checks["dependency_files_removed"] = not residues

    training_roots = {data_root / "training_python"}

    def collect_training_roots(item):
        if not isinstance(item, dict):
            return
        for removed in (item.get("training_python") or {}).get("removed", []):
            if isinstance(removed, dict) and removed.get("path"):
                training_roots.add(Path(removed["path"]))
        collect_training_roots(item.get("before_run"))
        for replacement in item.get("replacements") or []:
            collect_training_roots(replacement)

    collect_training_roots(receipt)
    remaining_training = sorted(str(path) for path in training_roots if path.exists() or path.is_symlink())
    checks["training_environments_removed"] = not remaining_training

    run_dir = Path(data["run_dir"]) if data.get("run_dir") else None
    report = result.get("report_path") or data.get("report_path")
    report = Path(report) if report else None
    retained = {
        "final_report": report,
        "report_result_archive": report.with_suffix(".result.json") if report else None,
        "run_report": run_dir / "report.md" if run_dir else None,
        "run_result": run_dir / "result.json" if run_dir else None,
        "cleanup_receipt": run_dir / "environment_cleanup.json" if run_dir else None,
        "source_workspace": Path(data["repository"]["path"])
            if (data.get("repository") or {}).get("path") else None,
        "application_python": Path(sys.executable),
    }
    for name, path in retained.items():
        checks[name + "_retained"] = path.is_dir() if name == "source_workspace" and path else bool(path and path.is_file())
    checks["cleanup_receipt_matches"] = False
    receipt_path = retained["cleanup_receipt"]
    if receipt_path and receipt_path.is_file():
        try:
            checks["cleanup_receipt_matches"] = json.loads(receipt_path.read_text("utf-8")) == receipt
        except (OSError, ValueError):
            pass
    for name in ("run_result", "report_result_archive"):
        checks[name + "_cleanup_matches"] = False
        path = retained[name]
        if path and path.is_file():
            try:
                saved = json.loads(path.read_text("utf-8"))
                checks[name + "_cleanup_matches"] = saved.get("data", {}).get("environment_cleanup") == receipt
            except (OSError, ValueError, AttributeError):
                pass
    return {"passed": all(checks.values()), "checks": checks, "receipt": receipt,
            "data_root": str(data_root), "deps_root": str(deps_root),
            "remaining_dependencies": sorted(residues), "remaining_training_roots": remaining_training,
            "retained": {name: str(path) if path else None for name, path in retained.items()}}


def verify_author_completion(result, pdf_hash, expected_repo):
    data = result.get("data") or {}
    validation = data.get("validation") or {}
    assert result.get("state") == "COMPLETED", result.get("error")
    assert validation.get("status") == "author_code_completed", validation
    assert validation.get("is_reproduced") is None and validation.get("paper_metrics_verified") is False
    assert validation.get("protocol_pass") is None and validation.get("independent_metrics_pass") is None
    assert data.get("reproduction_scope") == "author_code_execution"
    assert data.get("repository_executed") is True and data.get("execution_source") == "author_repository"
    assert data["pdf_input"]["sha256"] == pdf_hash
    assert data["resources"]["code_repo_url"].rstrip("/").casefold() == expected_repo.rstrip("/").casefold()
    identity = data["resources"]["repository_identity"]
    assert identity.get("status") == "unreviewed_execution" and identity.get("author_identity_confirmed") is False

    frozen = json.loads((Path(data["run_dir"]) / "experiment_spec.json").read_text("utf-8"))
    plan = frozen["spec"]
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
    assert frozen["sha256"] == data["spec_sha256"] == digest
    assert plan.get("scope") == "author_code_execution"
    steps = plan["steps"]
    assert steps and len({step["id"] for step in steps}) == len(steps)
    receipts = validation["author_execution"]
    assert len(receipts) == len(steps)
    assert {receipt["step"]["id"] for receipt in receipts} == {step["id"] for step in steps}
    execution = data["execution"]
    assert execution.get("success") is True
    records = execution["steps"]
    for step in steps:
        matches = [record for record in records if record.get("id") == step["id"]]
        assert len(matches) == 1
        record = matches[0]
        receipt = next(item for item in receipts if item["step"]["id"] == step["id"])
        assert receipt.get("step") == step and receipt.get("natural_return") is True
        assert receipt.get("spec_sha256") == digest
        assert record.get("success") is True and record.get("exit_code") == 0 and record.get("executed") is True
        assert not record.get("timed_out") and not record.get("cancelled")
    observed = any(type(receipt.get("optimizer_steps")) is int and receipt["optimizer_steps"] > 0
                   for receipt in receipts)
    checks = validation["completion_checks"]
    assert checks.get("all_required_author_commands_returned") is True
    assert checks.get("owned_processes_finished") is True
    assert checks.get("observed_training_updates") is observed
    assert validation.get("training_observed") is observed
    return validation


def verify_trial_repository(result, freeze):
    """Bind the real export and executed plan to the pre-registered author commit."""
    paper = freeze['paper']
    expected = {'url': repository_key(paper['repository_url']), 'revision': paper['repository_revision']}
    data = result['data']
    frozen_plan = json.loads((Path(data['run_dir']) / 'experiment_spec.json').read_text('utf-8'))['spec']
    for repository in (data.get('repository') or {}, frozen_plan.get('repository') or {}):
        assert repository.get('url'), 'Independent trial requires an observed repository URL'
        assert repository_key(repository['url']) == expected['url'], 'Executed repository differs from registered candidate'
        assert repository.get('revision') == expected['revision'], 'Executed author commit differs from registered candidate'
    return expected


def verify_claim_comparison(result):
    validation = (result.get('data') or {}).get('validation') or {}
    comparison = validation.get('author_claim_comparison') or {}
    assert comparison.get('status') in {'compared', 'partial'}, comparison.get('reason')
    records = {row['id']: row for row in comparison.get('metric_records', [])}
    citations = {ref['id']: ref for ref in comparison.get('citations', [])}
    eligible = []
    for row in comparison.get('comparisons') or []:
        if row.get('status') not in {'comparable', 'reference_only'}:
            continue
        difference = row.get('difference')
        assert type(difference) in (int, float) and math.isfinite(difference)
        actual, reference = row['actual'], row['reference']
        observed = records[actual['id']]
        assert actual['value'] == observed['value'] and actual['source'] == observed['source']
        for field in ('aggregation', 'runs', 'std'):
            if observed.get(field) is not None:
                assert actual.get(field) == observed[field]
        assert any(citations[key]['source'] == 'pdf' for key in reference['citations'])
        assert any(citations[key]['source'] == 'repository' for key in actual['citations'])
        assert reference['value_quote'] and actual.get('source_quote') == observed.get('source_quote')
        assert actual.get('source_quote')
        if row['status'] == 'reference_only':
            settings = row.get('settings') or {}
            assert settings.get('status') in {'matched', 'different', 'unknown'}
            assert isinstance(settings.get('reason'), str) and settings['reason'].strip()
            limitations = row.get('limitations') or []
            assert limitations and all(isinstance(note, str) and note.strip() for note in limitations)
        eligible.append(row)
    assert eligible, 'A verified, nonempty paper comparison is required for acceptance'
    assert validation.get('paper_metrics_verified') is False and validation.get('is_reproduced') is None
    return comparison


def verify_saved_report(result, downloaded):
    """Audit downloaded content against both persisted reports and result records."""
    report = Path(result['report_path'])
    run_dir = Path(result['data']['run_dir'])
    text = downloaded.decode('utf-8').replace('\r\n', '\n')
    assert text == report.read_text('utf-8'), 'Downloaded report differs from saved report'
    assert text == (run_dir / 'report.md').read_text('utf-8'), 'Run report differs from downloaded report'
    for path in (run_dir / 'result.json', report.with_suffix('.result.json')):
        saved = json.loads(path.read_text('utf-8'))
        assert saved.get('state') == result.get('state'), 'Saved terminal state differs'
        for key in ('validation', 'execution', 'pdf_input', 'spec_sha256',
                    'report', 'report_length', 'runtime_preparation'):
            assert saved.get('data', {}).get(key) == result['data'].get(key), 'Saved result differs: ' + key
    return {'saved_reports_identical': True, 'structured_records_identical': True}


def verify_report_reconnect(page, args, progress, runtime, result):
    """Real browser downloads and a reload must retain the saved comparison."""
    def show_report():
        page.get_by_role('tab', name=re.compile('复现报告')).click()
        table = page.get_by_role('table').filter(has_text='论文宣称').filter(has_text='作者运行输出').first
        table.wait_for(timeout=30000)
        return table.inner_text()
    displayed = show_report()
    content = []
    for label, suffix in [('下载报告', '.md'), ('下载报告和图片（ZIP）', '.zip')]:
        with page.expect_download(timeout=30000) as received:
            page.get_by_role('button', name=label, exact=True).click()
        path = runtime / (args.tag + '_downloaded_report' + suffix)
        received.value.save_as(str(path))
        content.append(path.read_bytes())
    with zipfile.ZipFile(io.BytesIO(content[1])) as archive:
        names = [name for name in archive.namelist() if name.endswith('.md')]
        assert len(names) == 1 and archive.read(names[0]) == content[0]
        report_path = Path(result['report_path'])
        attachment = report_path.with_name(report_path.stem + '_execution.txt')
        assert archive.read(attachment.name) == attachment.read_bytes()
        assert (Path(result['data']['run_dir']) / attachment.name).read_bytes() == attachment.read_bytes()
    assert '作者输出与论文宣称对比' in content[0].decode('utf-8')
    saved = verify_saved_report(result, content[0])
    before = set(runtime.glob('progress_*.jsonl'))
    page.goto(f'http://localhost:{args.port}/?run={progress.name}', wait_until='domcontentloaded')
    assert show_report() == displayed
    assert set(runtime.glob('progress_*.jsonl')) == before
    return {'comparison_table': displayed, 'markdown_zip_identical': True,
            'full_output_attachment_identical': True,
            'report_sha256': hashlib.sha256(content[0]).hexdigest(), 'reconnect_preserved_comparison': True,
            **saved}


def main(args):
    # Browser/runtime imports stay optional for CLI help and local audit tests.
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(ROOT))
    from frontend.backend_pipeline import ProgressStore

    data_root = Path(args.data_root or os.environ.get("AUTOREPRO_DATA_ROOT") or ROOT / "data").expanduser().resolve()
    deps_root = Path(args.deps_root or os.environ.get("AUTOREPRO_DEPS_ROOT") or data_root / "deps").expanduser().resolve()
    runtime = data_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    pdf = Path(args.pdf).resolve(strict=True)
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    credential = ""

    def verify_frozen():
        return verify_acceptance_source(args, freeze, pdf_hash)

    def save(suffix, value):
        (runtime / f"{args.tag}_{suffix}.json").write_text(
            json.dumps(redact(value, credential), ensure_ascii=False, indent=2), encoding="utf-8")

    def emit(value):
        print(json.dumps(redact(value, credential), ensure_ascii=False), flush=True)

    eligibility = verify_frozen()
    credential = read_credential(args)
    try:
        # The browser must close while Playwright's event loop still exists.
        # Contexts unwind right to left on success, failure, and interruption.
        with sync_playwright() as playwright, ExitStack() as browser_lifetime:
            browser = playwright.chromium.launch(**browser_launch_options(args))
            browser_lifetime.callback(browser.close)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://localhost:{args.port}", wait_until="domcontentloaded")
            page.get_by_role("heading", name="论文复现工作台", exact=True).wait_for(timeout=60000)
            config = configure_page(page, args, credential)
            emit({"browser_api_connection": "passed", "credential_matches": True, **config})
            page.get_by_text("上传PDF", exact=True).click()
            page.get_by_test_id("stFileUploader").locator('input[type="file"]').set_input_files(str(pdf))
            try:
                if args.allow_no_preview_link:
                    page.get_by_text(re.compile(re.escape("已上传：" + pdf.name))).wait_for(timeout=60000)
                else:
                    page.get_by_text(re.compile(
                        r"PDF (?:原文代码声明|仓库候选)链接：" + re.escape(args.repo), re.I)).wait_for(timeout=60000)
            except Exception:
                page.screenshot(path=str(runtime / f"{args.tag}_preview_failed.png"), full_page=True)
                save("preview_failed", {"body_text": page.locator("body").inner_text(), "task_submitted": False})
                raise
            page.screenshot(path=str(runtime / f"{args.tag}_preview.png"), full_page=True)
            assert page.get_by_label("API Key", exact=True).input_value() == credential, "Credential changed during PDF upload"
            before = set(runtime.glob("progress_*.jsonl"))
            page.get_by_role("button", name="开始复现", exact=True).click()
            deadline, created = time.monotonic() + 30, set()
            while time.monotonic() < deadline:
                created = set(runtime.glob("progress_*.jsonl")) - before
                if created:
                    break
                page.wait_for_timeout(250)
            assert len(created) == 1, "Exactly one browser-submitted task must be created; check --data-root"
            progress = created.pop()
            save("request", {"progress_file": str(progress), "input": "real_browser_pdf_upload",
                             "author_execution_only": True, "llm": config,
                             "repository_preview_required": not args.allow_no_preview_link,
                             "data_root": str(data_root), "deps_root": str(deps_root),
                             "pdf_sha256": pdf_hash, "freeze": freeze, "eligibility": eligibility,
                             "submitted_at": datetime.now(timezone.utc).isoformat()})
            emit({"progress_file": str(progress), "input": "real_browser_pdf_upload"})
            previous, observations = None, []
            installation_checks = {'heartbeat_seen': False, 'measured_download_seen': False,
                                   'live_panel_seen': False}
            deadline = time.monotonic() + args.observation_timeout
            while time.monotonic() < deadline:
                snapshot = ProgressStore.read_snapshot(str(progress))
                installations = snapshot.get('dependency_installations') or []
                if args.require_install_progress and installations:
                    installation_checks['heartbeat_seen'] = True
                    current = installations[-1]
                    if (current.get('downloaded_bytes') or 0) > 0 and not installation_checks['live_panel_seen']:
                        installation_checks['measured_download_seen'] = True
                        page.get_by_role('heading', name='依赖安装进度', exact=True).wait_for(timeout=30000)
                        page.wait_for_timeout(2500)
                        page.screenshot(path=str(runtime / f'{args.tag}_installation.png'), full_page=True)
                        installation_checks['live_panel_seen'] = True
                        installation_checks['sample'] = current
                observation = {"state": snapshot["state"], "done": snapshot["done"],
                               "active_runs": len(snapshot["active_runs"]), "active_workers": len(snapshot["active_workers"]),
                               "active_executions": len(snapshot["active_executions"]),
                               "stages": [{"id": row["id"], "status": row["status"], "confirmed": row["completion_confirmed"]}
                                          for row in snapshot["pipeline_stages"]]}
                if observation != previous:
                    emit(observation)
                    observations.append(observation)
                    previous = observation
                if snapshot["done"]:
                    result = snapshot["result"] or {}
                    save("full_result", result)
                    cleanup = cleanup_evidence(result, data_root, deps_root)
                    save("cleanup", cleanup)
                    eligibility = verify_frozen()
                    validation = verify_author_completion(result, pdf_hash, args.repo)
                    if getattr(args, 'require_independent', False):
                        assert validation.get('training_observed') is True, 'Independent acceptance requires actual training updates'
                        eligibility['repository'] = verify_trial_repository(result, freeze)
                    comparison = verify_claim_comparison(result) if args.require_comparison else None
                    if args.require_install_progress:
                        assert all(installation_checks[key] for key in
                                   ('heartbeat_seen', 'measured_download_seen', 'live_panel_seen')), installation_checks
                    assert cleanup["passed"], cleanup["checks"]
                    assert not snapshot["active_runs"] and not snapshot["active_workers"] and not snapshot["active_executions"]
                    assert all(row["status"] not in {"running", "waiting", "error", "blocked"}
                               for row in snapshot["pipeline_stages"])
                    events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
                    starts = [event for event in events if event.get("type") == "execution_run" and event.get("status") == "running"]
                    ends = [event for event in events if event.get("type") == "execution_run" and event.get("status") != "running"]
                    assert starts and len(starts) == len(ends)
                    assert all(event["status"] in {"success", "error"} for event in ends)
                    assert ends[-1]["status"] == "success"
                    assert sorted(event["execution_id"] for event in starts) == sorted(event["execution_id"] for event in ends)
                    page.get_by_text("所选仓库代码已运行完成，命令已正常退出。", exact=True).wait_for(timeout=30000)
                    caption = ("已对比作者输出与论文宣称，详见报告中的差值和实验设置；本次未进行论文数值审核。"
                               if (validation.get("author_claim_comparison") or {}).get("comparisons") else
                               "本次完成的是代码运行，论文数值未做验收。")
                    page.get_by_text(caption, exact=True).wait_for(timeout=30000)
                    if validation["training_observed"] is False:
                        page.get_by_text("本次未记录到优化器更新；详见作者程序输出。", exact=True).wait_for(timeout=30000)
                    page.get_by_text("任务完成", exact=True).wait_for(timeout=30000)
                    report_check = verify_report_reconnect(page, args, progress, runtime, result) if comparison else None
                    page.screenshot(path=str(runtime / f"{args.tag}_final.png"), full_page=True)
                    record = {"input": "real_browser_pdf_upload", "state": result["state"], "progress_file": str(progress),
                              "freeze": freeze, "pdf_sha256": pdf_hash, "validation": validation,
                              "eligibility": eligibility,
                              "run_dir": result["data"]["run_dir"], "execution_run_starts": len(starts),
                              "execution_run_ends": len(ends), "observations": observations,
                              "cleanup": cleanup, "live_page_completed": True,
                              "claim_comparison_required": args.require_comparison, "report_checks": report_check,
                              "installation_checks": installation_checks,
                              "live_page_paper_accepted": False, "live_page_author_code_completed": True}
                    save("acceptance", record)
                    emit({"state": result["state"], "validation": validation["status"],
                          "run_dir": result["data"]["run_dir"], "environment_cleanup": cleanup["passed"]})
                    return
                page.wait_for_timeout(3000)
            raise TimeoutError("Preserve the live run handle; observation deadline does not stop training")
    except Exception as exc:
        save("failure", {"error": str(exc), "type": type(exc).__name__})
        # Playwright call logs may contain fill arguments; never print the raw exception.
        raise RuntimeError(redact(str(exc), credential)) from None


def verify_acceptance_source(args, freeze, pdf_hash):
    independent = getattr(args, 'require_independent', False)
    if independent and not args.require_comparison:
        raise ValueError('--require-independent also requires --require-comparison')
    return verify_freeze(ROOT, freeze, paper_id=getattr(args, 'paper_id', None),
                         pdf_sha256=pdf_hash, repository=args.repo, require_independent=independent)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pdf", "repo", "freeze", "tag"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--paper-id", help="Paper identity in the shared acceptance registry")
    parser.add_argument("--require-independent", action="store_true",
                        help="Require an unused registry paper, matching PDF and unchanged source/dependency freeze")
    parser.add_argument("--execution-only", action="store_true", help="Compatibility flag; author execution is always used")
    parser.add_argument("--allow-no-preview-link", action="store_true")
    parser.add_argument("--require-comparison", action="store_true",
                        help="Require verified paper comparisons, matching downloads and browser reconnect")
    parser.add_argument("--require-install-progress", action="store_true",
                        help="Require measured installer events and the live browser progress panel")
    parser.add_argument("--credential-stdin", action="store_true", help='Read {"api_key": "..."} from stdin instead of hidden input')
    parser.add_argument("--base-url", help="Override the API URL shown on the page")
    parser.add_argument("--model", help="Override the model shown on the page")
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--browser-executable", help="Browser executable; defaults to Playwright Chromium")
    browser.add_argument("--browser-channel", help="Installed Playwright browser channel, e.g. chrome or msedge")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--data-root", help="Running application's data root (or AUTOREPRO_DATA_ROOT)")
    parser.add_argument("--deps-root", help="Running application's dependency root (or AUTOREPRO_DEPS_ROOT)")
    parser.add_argument("--observation-timeout", type=float, default=10000,
                        help="Observation deadline in seconds; does not cancel the application run")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
