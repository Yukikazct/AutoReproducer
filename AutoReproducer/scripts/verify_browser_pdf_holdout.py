"""Upload a previously unused PDF through the real browser and audit its run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import time
import tomllib

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontend.backend_pipeline import ProgressStore


def main(args):
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    pdf = Path(args.pdf).resolve(strict=True)
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()

    def verify_frozen():
        for name, expected in freeze["files"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

    def save(suffix, value):
        (ROOT / "data/runtime" / f"{args.tag}_{suffix}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    verify_frozen()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(f"http://localhost:{args.port}", wait_until="domcontentloaded")
        page.get_by_role("heading", name="论文复现工作台", exact=True).wait_for(timeout=60000)
        # Supply the tested local credential through the normal password field;
        # do not record its value or alter the paper's experimental settings.
        with (ROOT / ".streamlit/secrets.toml").open("rb") as handle:
            settings = tomllib.load(handle)
        key = json.loads(sys.stdin.read())['api_key'] if args.credential_stdin else settings['LLM_API_KEY']
        page.get_by_label("API Key", exact=True).fill(key)
        page.get_by_label("API Key", exact=True).press("Enter")
        page.get_by_role("button", name="测试 AI 连接", exact=True).click()
        page.get_by_text(re.compile(r"连接成功 \(deepseek-chat @")).wait_for(timeout=45000)
        assert page.get_by_label("API Key", exact=True).input_value() == key, "Credential changed after connection test"
        print(json.dumps({"browser_api_connection": "passed", "credential_matches": True}), flush=True)
        page.get_by_text("上传PDF", exact=True).click()
        page.get_by_test_id("stFileUploader").locator('input[type="file"]').set_input_files(str(pdf))
        try:
            if args.allow_no_preview_link:
                page.get_by_role('checkbox', name=re.compile('先运行作者仓库代码')).wait_for(timeout=60000)
            else:
                page.get_by_text(re.compile(r"PDF (?:原文代码声明|仓库候选)链接：" + re.escape(args.repo), re.I)).wait_for(timeout=60000)
        except Exception:
            page.screenshot(path=str(ROOT / f'data/runtime/{args.tag}_preview_failed.png'), full_page=True)
            save('preview_failed', {'body_text': page.locator('body').inner_text(),
                                    'task_submitted': False})
            raise
        page.screenshot(path=str(ROOT / f'data/runtime/{args.tag}_preview.png'), full_page=True)
        objective = page.get_by_role('checkbox', name=re.compile('先运行作者仓库代码'))
        objective.set_checked(args.execution_only)
        assert objective.is_checked() is args.execution_only
        assert page.get_by_label("API Key", exact=True).input_value() == key, "Credential changed during PDF upload"
        page.screenshot(path=str(ROOT / f"data/runtime/{args.tag}_upload.png"), full_page=True)
        before = set((ROOT / "data/runtime").glob("progress_*.jsonl"))
        page.get_by_role("button", name="开始复现", exact=True).click()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            created = set((ROOT / "data/runtime").glob("progress_*.jsonl")) - before
            if created:
                break
            page.wait_for_timeout(250)
        assert len(created) == 1, "Exactly one browser-submitted task must be created"
        progress = created.pop()
        save("request", {"progress_file": str(progress), "input": "real_browser_pdf_upload",
                         "author_execution_only": args.execution_only,
                         "repository_preview_required": not args.allow_no_preview_link,
                         "pdf_sha256": pdf_hash, "freeze": freeze,
                         "submitted_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"progress_file": str(progress), "input": "real_browser_pdf_upload"}), flush=True)
        previous, observations = None, []
        deadline = time.monotonic() + 10000
        while time.monotonic() < deadline:
            snapshot = ProgressStore.read_snapshot(str(progress))
            observation = {"state": snapshot["state"], "done": snapshot["done"],
                           "active_runs": len(snapshot["active_runs"]), "active_workers": len(snapshot["active_workers"]),
                           "active_executions": len(snapshot["active_executions"]),
                           "stages": [{"id": row["id"], "status": row["status"], "confirmed": row["completion_confirmed"]}
                                      for row in snapshot["pipeline_stages"]]}
            if observation != previous:
                print(json.dumps(observation, ensure_ascii=False), flush=True)
                observations.append(observation)
                previous = observation
            if snapshot["done"]:
                result = snapshot["result"] or {}
                save("full_result", result)
                page.wait_for_timeout(2500)
                page.screenshot(path=str(ROOT / f"data/runtime/{args.tag}_final.png"), full_page=True)
                verify_frozen()
                data, validation = result.get("data", {}), result.get("data", {}).get("validation", {})
                assert result.get("state") == "COMPLETED", result.get("error")
                if args.execution_only:
                    assert validation.get('status') == 'author_code_completed', validation
                    assert validation.get('is_reproduced') is None and validation.get('paper_metrics_verified') is False
                    assert validation.get('protocol_pass') is None and validation.get('independent_metrics_pass') is None
                    assert data['reproduction_scope'] == 'author_code_execution'
                    assert all(validation['completion_checks'].values())
                    assert data['native_preflight']['native_probe_pass'] is True
                    if data['dataset_provenance']['sources']:
                        assert data['dataset_provenance']['verified'] is True
                    else:
                        assert data['dataset_provenance']['verified'] is False
                        assert validation.get('dataset_provenance_verified') is False
                    plan = json.loads((Path(data['run_dir']) / 'experiment_spec.json').read_text('utf-8'))['spec']
                    observations_by_id = {receipt['step']['id']: receipt for receipt in validation['author_execution']}
                    records_by_id = {step['id']: step for step in data['execution']['steps']}
                    assert set(observations_by_id) == {step['id'] for step in plan['steps']}
                    assert (plan['semantic_review'].get('accepted') is True or
                            plan['semantic_review'].get('advisory') is True and
                            plan['semantic_review'].get('execution_allowed') is True)
                    for step in plan['steps']:
                        receipt, record = observations_by_id[step['id']], records_by_id[step['id']]
                        assert receipt['step'] == step and receipt['natural_return'] is True
                        assert record['success'] is True and record['exit_code'] == 0 and record['executed'] is True
                        assert not record.get('timed_out') and not record.get('cancelled')
                        if step['kind'] == 'train':
                            assert receipt['optimizer_steps'] > 0
                else:
                    assert validation.get("status") == "reproduced" and validation.get("is_reproduced") is True, validation
                    assert validation.get("protocol_pass") is True and validation.get("independent_metrics_pass") is True
                assert data.get("repository_executed") is True and data.get("execution_source") == "author_repository"
                assert data["pdf_input"]["sha256"] == pdf_hash
                assert data["resources"]["code_repo_url"].casefold() == args.repo.casefold()
                identity = data["resources"]["repository_identity"]
                if args.execution_only:
                    assert identity['status'] == 'unreviewed_execution'
                    assert identity['author_identity_confirmed'] is False
                    assert identity['review']['status'] == 'skipped'
                    assert identity['review']['execution_allowed'] is True
                else:
                    assert identity['status'] in {'author_verified', 'matching_implementation_verified'}
                    assert identity['review']['accepted'] is True
                    assert identity["review"]["repository_revision"] == data["repository"]["revision"]
                assert not snapshot["active_runs"] and not snapshot["active_workers"] and not snapshot["active_executions"]
                assert all(row["status"] not in {"running", "waiting", "error", "blocked"}
                           for row in snapshot["pipeline_stages"])
                events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
                starts = [event for event in events if event.get("type") == "execution_run" and event.get("status") == "running"]
                ends = [event for event in events if event.get("type") == "execution_run" and event.get("status") != "running"]
                assert starts and len(starts) == len(ends)
                # A repaired missing dependency may have an earlier failed run;
                # the final successful run and all process exits remain required.
                assert all(event['status'] in {'success', 'error'} for event in ends)
                assert ends[-1]['status'] == 'success'
                assert {event['execution_id'] for event in starts} == {event['execution_id'] for event in ends}
                if args.execution_only:
                    page.get_by_text('所选仓库代码已运行完成，实际训练更新和所属进程结束已核验。', exact=True).wait_for(timeout=30000)
                    caption = ('已对比作者输出与论文宣称，详见报告中的差值和实验设置；尚未独立核验论文数值。'
                               if (validation.get('author_claim_comparison') or {}).get('comparisons') else
                               '论文数值尚未核验；此结果不判定论文复现成功。')
                    page.get_by_text(caption, exact=True).wait_for(timeout=30000)
                else:
                    page.get_by_text("完整实验已完成，论文数值验收通过。", exact=True).wait_for(timeout=30000)
                page.get_by_text("任务完成", exact=True).wait_for(timeout=30000)
                record = {"input": "real_browser_pdf_upload", "state": result["state"], "progress_file": str(progress),
                          "freeze": freeze, "pdf_sha256": pdf_hash, "validation": validation,
                          "run_dir": data["run_dir"], "execution_run_starts": len(starts), "execution_run_ends": len(ends),
                          "observations": observations, "live_page_completed": True,
                          "live_page_paper_accepted": not args.execution_only,
                          "live_page_author_code_completed": args.execution_only}
                save("acceptance", record)
                print(json.dumps({"state": result["state"], "validation": validation["status"],
                                  "run_dir": data["run_dir"], "metrics": validation.get("metrics_comparison")}), flush=True)
                browser.close()
                return
            page.wait_for_timeout(3000)
        raise TimeoutError("Preserve the live run handle; observation deadline does not stop training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("pdf", "repo", "freeze", "tag"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument('--execution-only', action='store_true')
    parser.add_argument('--allow-no-preview-link', action='store_true')
    parser.add_argument('--credential-stdin', action='store_true')
    main(parser.parse_args())
