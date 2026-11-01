"""Upload a PDF through the browser and audit execution, metrics and release.

Credentials are supplied by a hidden prompt or JSON on stdin, never a secrets file.
Use data/dependency roots that match the running application.
Strict validation checks independent numerical evidence; unused-paper eligibility
is a separate opt-in requiring a paper-bound freeze and shared registry.
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
from types import SimpleNamespace

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


def visible_panel_control(page, panel, label):
    """Keep open panels open; recover a control lost during a Streamlit rerender."""
    control = page.get_by_label(label, exact=True)
    for attempt in range(3):
        try:
            if control.is_visible():
                return control
            page.get_by_text(panel, exact=True).click(timeout=10000)
            control.wait_for(state="visible", timeout=10000)
            return control
        except Exception:
            # A rerender may replace the heading or close the newly opened
            # panel. Recheck actual visibility before another bounded attempt.
            if attempt == 2:
                raise


def configure_page(page, args, credential):
    mock = visible_panel_control(page, "运行环境", "Mock 演示模式（无需 API）")
    if mock.is_checked():
        mock.set_checked(False)
    docker = visible_panel_control(page, "运行环境", "Docker 沙箱执行（暂未开放）")
    if docker.is_checked():
        docker.set_checked(False)
    fields = {"base_url": "API 地址（OpenAI 兼容）", "model": "模型名称"}
    config = {}
    for name, label in fields.items():
        field = visible_panel_control(page, "LLM API 配置", label)
        override = getattr(args, name)
        if override is not None:
            if not override.strip():
                raise ValueError(f"--{name.replace('_', '-')} must be nonempty")
            field.fill(override.strip())
            field.press("Enter")
        config[name] = field.input_value().strip()
        assert config[name], f"The page must have an effective {name}"
    password = visible_panel_control(page, "LLM API 配置", "API Key")
    password.fill(credential)
    password.press("Enter")
    visible_panel_control(page, "LLM API 配置", "API Key")
    page.get_by_role("button", name="测试 AI 连接", exact=True).click()
    message = "连接成功 (" + config["model"] + " @ " + config["base_url"] + "):"
    page.get_by_text(re.compile(re.escape(message))).wait_for(timeout=45000)
    assert password.input_value() == credential, "Credential changed after connection test"
    return config


def execution_settings(args):
    mode = getattr(args, 'intelligent_recovery_mode', 'off')
    if mode not in {'off', 'single', 'collaborative'}:
        raise ValueError('Unknown intelligent recovery mode')
    repair = getattr(args, 'allow_source_repair', False)
    strict = getattr(args, 'strict_validation', False)
    if strict and repair:
        raise ValueError('--allow-source-repair cannot be combined with --strict-validation')
    return {'intelligent_recovery_mode': mode, 'source_repair': repair,
            'strict_validation': strict}


def configure_execution_mode(page, strict_validation, *, recovery_mode='off', source_repair=False):
    """Set the real PDF controls, including when a browser session has old state."""
    execution_settings(SimpleNamespace(strict_validation=strict_validation,
        allow_source_repair=source_repair, intelligent_recovery_mode=recovery_mode))
    repair = page.get_by_label("AI 自动修复：失败后修复并重试", exact=True)
    if repair.is_checked() is not source_repair:
        repair.press("Space")
    strict = page.get_by_label("严格核验：重载模型并独立复算指标", exact=True)
    if strict.is_checked() is not strict_validation:
        # Streamlit's accessible React-Aria checkbox is covered by its label.
        # Use its supported keyboard interaction instead of clicking the input.
        strict.press("Space")
    assert strict.is_checked() is strict_validation, "The browser strict-validation checkbox differs from the requested mode"
    assert repair.is_checked() is source_repair, "The browser source repair checkbox differs from the requested mode"
    recovery = page.get_by_label('智能调查与排障', exact=True)
    if recovery.is_checked() is not (recovery_mode != 'off'):
        recovery.press('Space')
    assert recovery.is_checked() is (recovery_mode != 'off')
    if recovery_mode != 'off':
        label = {'single': '单智能体（自检）',
                 'collaborative': '多智能体协作（独立反证审查）'}[recovery_mode]
        page.get_by_label('调查方式', exact=True).click()
        page.get_by_role('option', name=label, exact=True).click()
    if source_repair:
        editable = page.get_by_label('限定修复文件（可选，每行一个仓库相对路径）', exact=True)
        editable.fill('')
        editable.press('Tab')


def verify_recovery_request(result, requested):
    data = result.get('data') or {}
    recovery = data.get('intelligent_recovery') or {}
    enabled = requested['intelligent_recovery_mode'] != 'off'
    assert (recovery.get('enabled') is True) is enabled, 'Backend intelligent recovery differs from the browser request'
    if enabled:
        assert recovery.get('mode') == requested['intelligent_recovery_mode'], 'Backend investigation role mode differs'
        assert isinstance(recovery.get('usage'), dict), 'Actual recovery usage is missing'
    repair = data.get('source_repair') or {}
    assert (repair.get('requested') is True) is requested['source_repair'], 'Backend source repair authorization differs'
    return {'requested': requested, 'actual_mode': recovery.get('mode') if enabled else 'off',
            'status': recovery.get('status', 'disabled'), 'usage': recovery.get('usage', {}),
            'requires_human': recovery.get('requires_human'),
            'source_repair_status': repair.get('status', 'not_requested')}


def outcome_levels(result):
    from src.result_conclusion import build_reproduction_conclusion
    conclusion = build_reproduction_conclusion(result.get('data') or {})
    return {key: conclusion[key] for key in ('scope', 'execution', 'comparison', 'independent', 'is_reproduced')}


def verify_author_run(result, pdf_hash, expected_repo, *, source_repair=False):
    data = result.get('data') or {}
    if data.get('execution_source') != 'patched_author_repository':
        return verify_author_completion(result, pdf_hash, expected_repo)
    assert source_repair is True, 'Repaired source execution was not authorized'
    assert result.get('state') == 'COMPLETED'
    assert data.get('reproduction_scope') == 'patched_author_code_execution'
    assert data['pdf_input']['sha256'] == pdf_hash
    assert repository_key(data['resources']['code_repo_url']) == repository_key(expected_repo)
    assert data.get('repository_executed') is True
    from src.source_repair import verify_repaired_execution
    validation = verify_repaired_execution(data['grounded_repository_plan'], data, data['execution'])
    for field in ('status', 'source_modified', 'source_repair_manifest_sha256', 'completion_checks',
                  'protocol_pass', 'independent_metrics_pass', 'is_reproduced'):
        assert validation[field] == data['validation'].get(field), 'Saved repaired execution evidence differs: ' + field
    return validation


def configure_experiment_request(page, args):
    from src.experiment_request import normalize_experiment_request
    requested = normalize_experiment_request({
        'method': getattr(args, 'experiment_method', None) or '',
        'dataset': getattr(args, 'experiment_dataset', None) or ''})
    for name, label in [('method', '实验方法 / 模型（可选）'), ('dataset', '实验数据集（可选）')]:
        field = page.get_by_label(label, exact=True)
        field.fill(requested.get(name, ''))
        field.press('Enter')
        assert field.input_value() == requested.get(name, '')
    return requested


def verify_experiment_request(result, requested, strict_validation):
    data = result.get('data') or {}
    assert data.get('experiment_request', {}) == requested, 'The backend received a different experiment request'
    if strict_validation and requested:
        from src.experiment_request import bind_experiment_request
        bind_experiment_request(data['grounded_repository_plan']['experiment_protocol'], requested)


def terminal_outcome(result):
    """Describe the service's reported outcome; evidence is audited separately."""
    data = result.get('data') or {}
    strict = data.get('strict_validation') or {}
    validation = data.get('validation') or {}
    if result.get('state') != 'COMPLETED':
        return 'execution_failed'
    if strict.get('requested') is not True:
        return 'ordinary_completed'
    if strict.get('status') == 'unavailable':
        return 'strict_unavailable'
    if strict.get('status') == 'failed':
        return 'strict_failed'
    if strict.get('status') == 'verified' and type(validation.get('is_reproduced')) is bool:
        return 'strict_verified_reference_met' if validation['is_reproduced'] else 'strict_verified_reference_not_met'
    return 'strict_incomplete'


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


def verify_ordinary_conclusion(validation):
    """AI judgments remain distinct from protocol/independent verification."""
    assert validation.get('paper_metrics_verified') is False
    assert validation.get('protocol_pass') is None and validation.get('independent_metrics_pass') is None
    assessment = validation.get('reproduction_assessment') or {}
    if assessment.get('status') == 'completed':
        mapping = {'reproduced': ('ai_reproduced', True),
                   'partially_reproduced': ('ai_partially_reproduced', None),
                   'not_reproduced': ('ai_not_reproduced', False),
                   'insufficient_evidence': ('ai_insufficient_evidence', None)}
        assert assessment.get('verdict') in mapping, 'Unknown ordinary AI verdict'
        state, reproduced = mapping[assessment['verdict']]
        assert validation.get('execution_status') == 'author_code_completed'
        assert validation.get('status') == validation.get('result_level') == state
        assert validation.get('verdict_source') == 'ai'
        assert validation.get('is_reproduced') is reproduced
    else:
        assert validation.get('status') == 'author_code_completed'
        assert validation.get('is_reproduced') is None


def verify_author_completion(result, pdf_hash, expected_repo):
    data = result.get("data") or {}
    validation = data.get("validation") or {}
    assert result.get("state") == "COMPLETED", result.get("error")
    verify_ordinary_conclusion(validation)
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


def verify_strict_completion(result, pdf_hash, expected_repo):
    """Replay the formal verifier against retained source, models and receipts.

    A recorded service request, separate inference process, complete native runs
    and frozen numerical comparison are required. Printed author comparisons
    cannot substitute for these artifacts. Missing the numerical target remains
    a completed strict verification, with is_reproduced=False.
    """
    from src.strict_validation import verify_strict_execution

    data = result.get('data') or {}
    strict = data.get('strict_validation') or {}
    assert strict.get('requested') is True, 'The backend did not receive strict_validation=true'
    assert result.get('state') == 'COMPLETED', result.get('error') or terminal_outcome(result)
    assert strict.get('status') == 'verified', (
        'Strict acceptance did not complete: ' + terminal_outcome(result) + ': ' + str(strict.get('reason', '')))
    assert (data.get('source_repair') or {}).get('requested') is False, 'Strict acceptance cannot include source repair'
    validation = data.get('validation') or {}
    assert strict.get('result') == validation.get('status')
    assert validation.get('status') in {'reproduced', 'reference_not_met'}
    assert validation.get('is_reproduced') is (validation['status'] == 'reproduced')
    assert validation.get('strict_validation') is True and validation.get('paper_metrics_verified') is True
    assert validation.get('protocol_pass') is True and validation.get('independent_metrics_pass') is True
    assert data.get('reproduction_scope') == 'selected_paper_experiment'
    assert data.get('execution_source') == 'author_repository' and data.get('repository_executed') is True
    assert data['pdf_input']['sha256'] == pdf_hash
    assert repository_key(data['resources']['code_repo_url']) == repository_key(expected_repo)

    run_dir = Path(data['run_dir'])
    frozen = json.loads((run_dir / 'experiment_spec.json').read_text('utf-8'))
    plan = frozen['spec']
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()
    assert frozen['sha256'] == data['spec_sha256'] == digest
    assert plan.get('scope') == 'selected_paper_experiment' and plan.get('pdf_sha256') == pdf_hash
    repository = data['repository']
    assert repository_key(repository['url']) == repository_key(expected_repo)
    assert plan['repository'] == {key: repository[key] for key in ('url', 'revision')}
    packet = json.loads((run_dir / 'source_packet.json').read_text('utf-8'))
    workspace = Path(repository['path'])
    # Replays source/AST bindings, complete required process records, model and
    # tensor hashes, all run receipts and the frozen numerical rule; no training
    # or model calls are performed here.
    replayed = verify_strict_execution(plan, packet, workspace, data['execution'], digest)
    assert all(validation.get(key) == value for key, value in replayed.items()), 'Strict validation differs from artifact replay'
    assert strict.get('binding') == replayed['strict_plan_verification'], 'Strict request receipt lost its source binding'
    assert data['execution'].get('strict_execution_verification') == validation
    for name, expected in [('strict_validation.json', strict), ('strict_execution_verification.json', validation)]:
        assert json.loads((run_dir / name).read_text('utf-8')) == expected, 'Saved strict receipt differs: ' + name
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
        actual, reference = row['actual'], row['reference']
        for observation in (actual, reference):
            number = observation.get('value')
            assert type(number) in (int, float) and math.isfinite(number), (
                'A paper comparison requires both original finite numerical observations')
        difference = row.get('difference')
        if row['status'] == 'comparable' or difference is not None:
            assert type(difference) in (int, float) and math.isfinite(difference), (
                'A recorded numerical difference must be finite')
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
    # Apply the same independent arithmetic and unit/metric checks used for
    # downloaded V6 reports. Related values with unknown units or different
    # metrics can remain visible as reference-only observations without a delta.
    comparison_rows(validation)
    verify_ordinary_conclusion(validation)
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
        for key in ('validation', 'execution', 'pdf_input', 'spec_sha256', 'strict_validation',
                    'source_repair', 'reproduction_scope',
                    'report', 'report_length', 'runtime_preparation'):
            assert saved.get('data', {}).get(key) == result['data'].get(key), 'Saved result differs: ' + key
    if (result['data'].get('strict_validation') or {}).get('requested') is True:
        verify_strict_report(text, result['data']['validation'])
    elif (result['data'].get('validation', {}).get('author_claim_comparison') or {}).get('comparisons'):
        verify_comparison_report(text, result['data']['validation'])
    if ((result['data'].get('validation', {}).get('author_claim_comparison') or {}).get('comparisons')
            or (result['data'].get('strict_validation') or {}).get('requested') is True):
        verify_report_conclusion(text, result['data'])
    return {'saved_reports_identical': True, 'structured_records_identical': True}


COMPARISON_HEADER = ['指标', '论文参考', '实际值', '单位（参考 / 实际）',
                     '划分（参考 / 实际）', '次数（参考 / 实际）', '差值', '可比较性']


def comparison_rows(validation, *, strict=False):
    """Build V6 expectations from original result fields, without using its renderer."""
    from src.metric_keys import norm_metric_key
    from src.runtime_metrics import metric_unit

    comparisons = (validation.get('comparisons') if strict else
                   (validation.get('author_claim_comparison') or {}).get('comparisons'))
    assert isinstance(comparisons, list) and comparisons, 'Report requires authoritative numerical comparisons'
    assert all(isinstance(row, dict) for row in comparisons)
    if strict:
        assert type(validation.get('is_reproduced')) is bool
        assert all(type(row.get('pass')) is bool for row in comparisons)
        assert all(row['pass'] for row in comparisons) is validation['is_reproduced']
        assert validation.get('status') == ('reproduced' if validation['is_reproduced'] else 'reference_not_met')
        names = [norm_metric_key(row.get('name') or row.get('metric')) for row in comparisons]
        assert len(names) == len(set(names)), 'Strict comparison metric names must be unique'

    def value(node):
        return node.get('value') if isinstance(node, dict) else node

    def count(node):
        for key in ('runs', 'run_count', 'n_runs', 'count', 'repetitions'):
            number = node.get(key)
            number = number.get('count') if isinstance(number, dict) else number
            if type(number) is int and number > 0:
                return number

    def unconfirmed_f1(reference, name):
        if name not in {'f1_score', 'f1_macro', 'f1_micro', 'f1_weighted'}:
            return False
        # Re-read only the paper excerpts selected for this row. A missing
        # averaging label is useful context, never permission to invent a delta.
        sources = {ref.get('id'): ref for ref in
                   (validation.get('author_claim_comparison') or {}).get('citations', [])
                   if isinstance(ref, dict) and ref.get('source') == 'pdf'}
        identifiers = reference.get('citations')
        if not isinstance(identifiers, list) or not identifiers:
            return False
        selected = [sources.get(key) for key in identifiers]
        if any(not ref or not isinstance(ref.get('quote'), str) or not ref['quote'] for ref in selected):
            return False
        quote = reference.get('value_quote')
        if not isinstance(quote, str) or not quote or not any(quote in ref['quote'] for ref in selected):
            return False
        metric_quote = reference.get('metric_quote')
        if metric_quote is not None:
            if (not isinstance(metric_quote, str) or not 4 <= len(metric_quote) <= 800
                    or not any(quote in ref['quote'] and metric_quote in ref['quote'] for ref in selected)):
                return False
            evidence = metric_quote
        else:
            evidence = '\n'.join(ref['quote'] for ref in selected)
        labels = re.findall(
            r'(?<!\w)(?:(?:macro|micro|weighted)[_\s-]?f1(?:[_\s-]?score)?|'
            r'f1(?:[_\s-]?score)?(?:[_\s-](?:macro|micro|weighted))?)(?!\w)', evidence, re.I)
        return name not in {norm_metric_key(label) for label in labels}

    rows = []
    for item in comparisons:
        reference, actual = value(item.get('reference')), value(item.get('actual'))
        nodes = {side: item.get(side) if isinstance(item.get(side), dict) else {}
                 for side in ('reference', 'actual')}
        units = {side: nodes[side].get('unit') or item.get(side + '_unit') or item.get('unit')
                 for side in nodes}
        splits = {side: nodes[side].get('split') or item.get(side + '_split') or
                  (item.get('split') if strict else None) for side in nodes}
        runs = {side: count(nodes[side]) or (item.get(side + '_runs') if strict else None)
                for side in nodes}
        for number in (reference, actual):
            assert number is None or type(number) in (int, float) and math.isfinite(number), (
                'Comparison contains a nonnumeric or nonfinite observation')
            if strict:
                assert number is not None, 'Strict comparisons require both verified numerical observations'
        state = 'comparable' if strict else item.get('status') or 'unrecorded'
        assert state in {'comparable', 'reference_only', 'not_comparable', 'unrecorded'}
        difference = value(item.get('difference')) if state in {'comparable', 'reference_only'} else None
        difference_unit = item.get('difference_unit') or (item.get('unit') if strict else None)
        if strict and difference is None and reference is not None and actual is not None:
            if units['reference'] == units['actual']:
                difference = actual - reference
        if state in {'comparable', 'reference_only'} and reference is not None and actual is not None:
            left, right = (metric_unit(explicit=units[side]) for side in ('reference', 'actual'))
            expected = None
            reference_name = nodes['reference'].get('name') or item.get('metric') or item.get('name')
            actual_name = nodes['actual'].get('name') or item.get('metric') or item.get('name')
            same_metric = norm_metric_key(reference_name) == norm_metric_key(actual_name)
            if not same_metric or not left or not right:
                assert difference is None, 'Different metrics or unknown units must retain raw values without a difference'
            elif not strict and unconfirmed_f1(nodes['reference'], norm_metric_key(reference_name)):
                assert state == 'reference_only' and difference is None, (
                    'Unconfirmed F1 averaging must retain reference-only raw values without a difference')
            elif left == right:
                expected = actual - reference
            elif {left, right} == {'percent', 'fraction'}:
                expected = ((actual / 100 if right == 'percent' else actual) -
                            (reference / 100 if left == 'percent' else reference)) * 100
            if expected is not None:
                assert type(difference) in (int, float) and math.isfinite(difference), (
                    'Comparable and reference-only observations must retain their numerical difference')
                assert math.isclose(difference, expected, rel_tol=1e-9, abs_tol=1e-10), (
                    'Structured difference disagrees with the authoritative reference and actual values')
        delta = _comparison_text(difference, '未取得')
        if difference is not None and difference_unit:
            delta += ' ' + _comparison_text({'percentage_points': '个百分点'}.get(difference_unit, difference_unit))
        metric = item.get('metric') or item.get('name')
        paper_name = nodes['reference'].get('name')
        if not strict and paper_name and norm_metric_key(paper_name) != norm_metric_key(metric):
            metric = f'论文 {paper_name} / 实际 {metric}'
        rows.append([_comparison_text(metric),
                     _comparison_text(reference, '未取得'), _comparison_text(actual, '未取得'),
                     ' / '.join(_comparison_text(units[side]) for side in nodes),
                     ' / '.join(_comparison_text(splits[side]) for side in nodes),
                     ' / '.join(_comparison_text(runs[side]) for side in nodes), delta,
                     {'comparable': '可比较', 'reference_only': '仅供参考',
                      'not_comparable': '不可比较', 'unrecorded': '未记录'}[state]])
    if not strict:
        # Audit unpaired observations independently of the report view model.
        # IDs identify outputs; legacy signature joins require uniqueness on
        # both sides and never collapse two different explicit IDs.
        def identifier(node):
            candidate = node.get('id')
            return candidate if isinstance(candidate, str) and candidate else None

        def signature(node):
            return (norm_metric_key(node.get('name') or node.get('metric') or ''), value(node),
                    node.get('unit') or None, node.get('split') or None, count(node),
                    node.get('std'), node.get('aggregation') or None,
                    node.get('dataset') or None, node.get('method') or None)

        groups = []
        source = validation.get('author_claim_comparison') or {}
        for collection in (source.get('metric_records'), validation.get('metric_records')):
            group, seen = [], set()
            for record in collection if isinstance(collection, list) else []:
                if not isinstance(record, dict) or not (record.get('name') or record.get('metric')):
                    continue
                key = identifier(record)
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                group.append(record)
            groups.append(group)
        primary, secondary = groups
        observations = list(primary)
        known_ids = {identifier(record) for record in primary} - {None}
        for record in secondary:
            key, stamp = identifier(record), signature(record)
            if key and key in known_ids:
                continue
            positions = [index for index, original in enumerate(primary) if signature(original) == stamp]
            if len(positions) == 1 and sum(signature(other) == stamp for other in secondary) == 1:
                position = positions[0]
                if not key or not identifier(primary[position]):
                    if key:
                        observations[position] = record
                        known_ids.add(key)
                    continue
            observations.append(record)
            if key:
                known_ids.add(key)
        paired = []
        for item in comparisons:
            node = item.get('actual') if isinstance(item.get('actual'), dict) else {}
            paired.append({**node, 'name': node.get('name') or item.get('metric') or item.get('name'),
                           'value': value(item.get('actual')),
                           'unit': node.get('unit') or item.get('actual_unit') or item.get('unit'),
                           'split': node.get('split') or item.get('actual_split')})
        paired_ids = {identifier(node) for node in paired} - {None}
        for record in observations:
            key, stamp = identifier(record), signature(record)
            matches = [node for node in paired if signature(node) == stamp]
            if key and key in paired_ids:
                continue
            if (len(matches) == 1 and sum(signature(other) == stamp for other in observations) == 1
                    and (not key or not identifier(matches[0]))):
                continue
            observed = value(record)
            assert observed is None or type(observed) in (int, float) and math.isfinite(observed)
            rows.append([_comparison_text(record.get('name') or record.get('metric')),
                         '未取得', _comparison_text(observed, '未取得'),
                         '未记录 / ' + _comparison_text(record.get('unit')),
                         '未记录 / ' + _comparison_text(record.get('split')),
                         '未记录 / ' + _comparison_text(count(record)), '未取得', '未记录'])
    return rows


def _comparison_text(value, missing='未记录'):
    if value is None or value == '':
        return missing
    text = re.sub(r'\s+', ' ', str(value)).strip()
    return text if len(text) <= 240 else text[:240] + '…（完整记录见详情附件）'


def strict_comparison_rows(validation):
    return comparison_rows(validation, strict=True)


def _report_lines(report):
    """Ignore fenced examples when auditing generated sections and conclusions."""
    lines, fence = [], None
    for line in report.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None:
            lines.append(line.strip())
    return lines


def _markdown_plain(text):
    return re.sub(r'\\([\\`*_{}\[\]()<>!|])', r'\1', text)


def verify_report_conclusion(report, data):
    from src.result_conclusion import build_reproduction_conclusion
    conclusion = build_reproduction_conclusion(data)
    lines = _report_lines(report)
    for key, labels in (('comparison', ('是否达到参考', '论文参考比较')),
                        ('independent', ('独立核验状态', '独立核验'))):
        for label in labels:
            prefix = '- **' + label + '**: '
            assert [_markdown_plain(line[len(prefix):]) for line in lines if line.startswith(prefix)] == [
                conclusion[key]['label']], 'Downloaded report has a stale ' + key + ' conclusion'


def verify_comparison_report(report, validation, *, strict=False):
    """Bind all eight V6 table columns to authoritative result records."""
    lines = _report_lines(report)
    headings = [index for index, line in enumerate(lines) if line == '## 指标与参考比较']
    assert len(headings) == 1, 'Report requires one unambiguous numerical comparison table'
    rows = []
    for line in lines[headings[0] + 1:]:
        if not line and not rows:
            continue
        if not line.startswith('|'):
            break
        cells, cell, escaped = [], '', False
        for char in line[1:-1]:
            if char == '|' and not escaped:
                cells.append(_markdown_plain(cell.strip()))
                cell = ''
            else:
                cell += char
            escaped = not escaped if char == '\\' else False
        cells.append(_markdown_plain(cell.strip()))
        if all(re.fullmatch(r':?-{3,}:?', value) for value in cells):
            continue
        rows.append(cells)
    expected = [COMPARISON_HEADER, *comparison_rows(validation, strict=strict)]
    assert rows == expected, 'Downloaded metric values, differences or comparability differ from authoritative comparisons'
    return expected


def verify_strict_report(report, validation):
    """A current strict report must retain both independent and numeric conclusions."""
    expected = verify_comparison_report(report, validation, strict=True)
    lines = _report_lines(report)
    state = ('已有核验指标达到参考规则' if validation['is_reproduced'] else '已有核验指标未全部达到参考规则')
    for label, value in (('是否达到参考', state), ('论文参考比较', state),
                         ('独立核验状态', '协议与独立指标复算已核验'), ('独立核验', '协议与独立指标复算已核验')):
        prefix = '- **' + label + '**: '
        assert [line for line in lines if line.startswith(prefix)] == [prefix + value], (
            'Downloaded strict report has a stale numerical or independent conclusion')
    return expected


def verify_comparison_table(table, validation, *, strict=False):
    """Inspect real DOM cells, including units, before accepting the live/reloaded UI."""
    rows = [[re.sub(r'\s+', ' ', cell).strip() for cell in row.locator('th, td').all_inner_texts()]
            for row in table.locator('tr').all()]
    assert rows == [COMPARISON_HEADER, *comparison_rows(validation, strict=strict)], (
        'Displayed metric values, differences or comparability differ from authoritative comparisons')


def verify_strict_table(table, validation):
    verify_comparison_table(table, validation, strict=True)


def verify_report_reconnect(page, args, progress, runtime, result):
    """Real browser downloads and a reload must retain the saved comparison."""
    strict = getattr(args, 'strict_validation', False)

    def show_report():
        page.get_by_role('tab', name=re.compile('复现报告')).click()
        table = page.get_by_role('table')
        table = table.filter(has_text='论文参考').filter(has_text='实际值').filter(has_text='可比较性').first
        table.wait_for(timeout=30000)
        verify_comparison_table(table, result['data']['validation'], strict=strict)
        if strict:
            state_text = ('已有核验指标达到参考规则' if result['data']['validation']['is_reproduced'] else
                          '已有核验指标未全部达到参考规则')
            for marker in ('协议与独立指标复算已核验', state_text):
                page.get_by_text(re.compile(re.escape(marker))).first.wait_for(timeout=30000)
        return table.inner_text()
    displayed = show_report()
    content = []
    for label, suffix in [('下载报告', '.md'), ('下载报告与证据（ZIP）', '.zip')]:
        with page.expect_download(timeout=30000) as received:
            page.get_by_role('button', name=label, exact=True).click()
        path = runtime / (args.tag + '_downloaded_report' + suffix)
        received.value.save_as(str(path))
        content.append(path.read_bytes())
    with zipfile.ZipFile(io.BytesIO(content[1])) as archive:
        report_path = Path(result['report_path'])
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Downloaded ZIP contains duplicate member names'
        assert report_path.name in names and archive.read(report_path.name) == content[0], (
            'Downloaded ZIP main report differs from the Markdown download')
        manifest_name = report_path.stem + '.manifest.json'
        evidence_name = report_path.stem + '.evidence.json'
        assert manifest_name in names and evidence_name in names, 'Downloaded ZIP has no manifest or evidence'
        manifest = json.loads(archive.read(manifest_name))
        assert manifest.get('report') == report_path.name and manifest.get('evidence') == evidence_name
        assert manifest.get('hash_algorithm') == 'sha256'
        members = manifest.get('files')
        assert isinstance(members, list) and all(isinstance(row, dict) for row in members)
        paths = [row.get('path') for row in members]
        assert len(paths) == len(set(paths)) and set(paths) == set(names) - {manifest_name}, (
            'Downloaded ZIP manifest does not cover every payload member exactly once')
        for row in members:
            raw = archive.read(row['path'])
            assert type(row.get('bytes')) is int and row['bytes'] == len(raw), 'ZIP member byte length differs'
            assert row.get('sha256') == hashlib.sha256(raw).hexdigest(), 'ZIP member SHA-256 differs'
        from src.author_claim_comparison import _redact_evidence
        from src.report_revisions import sanitize_report_result
        from src.result_conclusion import build_reproduction_conclusion
        safe = _redact_evidence(sanitize_report_result(result), '')
        evidence = json.loads(archive.read(evidence_name))
        assert evidence.get('state') == safe.get('state') and evidence.get('error') == safe.get('error')
        assert evidence.get('report') == {'filename': report_path.name,
                                         'sha256': hashlib.sha256(content[0]).hexdigest()}
        saved_evidence = evidence.get('data') or {}
        assert saved_evidence.get('validation') == safe['data'].get('validation'), (
            'Downloaded evidence validation differs from the saved result')
        assert saved_evidence.get('reproduction_conclusion') == build_reproduction_conclusion(safe['data']), (
            'Downloaded evidence has a stale reproduction conclusion')
        attachment = report_path.with_name(report_path.stem + '_execution.txt')
        assert archive.read(attachment.name) == attachment.read_bytes()
        assert (Path(result['data']['run_dir']) / attachment.name).read_bytes() == attachment.read_bytes()
    saved = verify_saved_report(result, content[0])
    before = set(runtime.glob('progress_*.jsonl'))
    page.goto(f'http://localhost:{args.port}/?run={progress.name}', wait_until='domcontentloaded')
    assert show_report() == displayed
    assert set(runtime.glob('progress_*.jsonl')) == before
    return {'comparison_table': displayed, 'markdown_zip_identical': True,
            'bundle_manifest_verified': True, 'bundle_evidence_consistent': True,
            'full_output_attachment_identical': True,
            'report_sha256': hashlib.sha256(content[0]).hexdigest(), 'reconnect_preserved_comparison': True,
            **saved}


def verify_terminal_page(page, validation, strict_validation=False):
    """Wait for the actual terminal conclusion rather than a generic success."""
    if strict_validation:
        if validation['is_reproduced']:
            page.get_by_text('完整实验已完成，论文数值验收通过。', exact=True).wait_for(timeout=30000)
        else:
            page.get_by_text(re.compile(re.escape('选定论文实验已完成，独立评估数值未达到固定参考：'))).wait_for(timeout=30000)
    elif (validation.get('reproduction_assessment') or {}).get('status') == 'completed':
        assessment = validation['reproduction_assessment']
        labels = {'reproduced': 'AI 判断复现成功', 'partially_reproduced': 'AI 判断部分复现',
                  'not_reproduced': 'AI 判断尚未复现', 'insufficient_evidence': 'AI 判断证据不足'}
        label = labels[assessment['verdict']]
        page.get_by_text(re.compile('^' + re.escape(label + '：'))).wait_for(timeout=30000)
        page.get_by_text('判断范围为本次实际执行的实验。AI 根据具体论文选择合理误差；固定 5% 参考线不作为普通复现的硬门槛。',
                         exact=True).wait_for(timeout=30000)
        if validation.get('paper_metrics_verified') is not True:
            page.get_by_text('严格独立复算为可选核验，本次未完成该项。', exact=True).wait_for(timeout=30000)
    else:
        page.get_by_text('所选仓库代码已运行完成，命令已正常退出。', exact=True).wait_for(timeout=30000)
        caption = ('已对比作者输出与论文宣称，详见报告中的差值和实验设置；本次未进行论文数值审核。'
                   if (validation.get('author_claim_comparison') or {}).get('comparisons') else
                   '本次完成的是代码运行，论文数值未做验收。')
        page.get_by_text(caption, exact=True).wait_for(timeout=30000)
        if validation['training_observed'] is False:
            page.get_by_text('本次未记录到优化器更新；详见作者程序输出。', exact=True).wait_for(timeout=30000)
    page.get_by_text('任务完成', exact=True).wait_for(timeout=30000)


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
    strict_requested = getattr(args, 'strict_validation', False)
    settings = execution_settings(args)

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
                    page.get_by_text(re.compile(re.escape("已上传：" + pdf.name))).wait_for(timeout=240000)
                else:
                    page.get_by_text(re.compile(
                        r"PDF (?:原文代码声明|仓库候选)链接：" + re.escape(args.repo), re.I)).wait_for(timeout=240000)
            except Exception:
                page.screenshot(path=str(runtime / f"{args.tag}_preview_failed.png"), full_page=True)
                save("preview_failed", {"body_text": page.locator("body").inner_text(), "task_submitted": False})
                raise
            configure_execution_mode(page, strict_requested,
                recovery_mode=settings['intelligent_recovery_mode'], source_repair=settings['source_repair'])
            experiment_request = configure_experiment_request(page, args)
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
                             "author_execution_only": not strict_requested,
                             "strict_validation": strict_requested, "source_repair": settings['source_repair'], "llm": config,
                             "intelligent_recovery_mode": settings['intelligent_recovery_mode'],
                             "experiment_request": experiment_request,
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
                    save("terminal_observation", {"reported_outcome": terminal_outcome(result),
                        "backend_strict_request": (result.get('data', {}).get('strict_validation') or {}).get('requested'),
                        "strict_validation_required": strict_requested, "evidence_audited": False})
                    cleanup = cleanup_evidence(result, data_root, deps_root)
                    save("cleanup", cleanup)
                    eligibility = verify_frozen()
                    validation = (verify_strict_completion(result, pdf_hash, args.repo) if strict_requested else
                                  verify_author_run(result, pdf_hash, args.repo, source_repair=settings['source_repair']))
                    recovery = verify_recovery_request(result, settings)
                    verify_experiment_request(result, experiment_request, strict_requested)
                    if strict_requested and freeze.get('paper'):
                        eligibility['repository'] = verify_trial_repository(result, freeze)
                    if getattr(args, 'require_independent', False):
                        observed = ((validation.get('training_summary') or {}).get('optimizer_steps')
                                    if strict_requested else validation.get('training_observed'))
                        assert observed is True or type(observed) is int and observed > 0, 'Independent acceptance requires actual training updates'
                        if not strict_requested:
                            eligibility['repository'] = verify_trial_repository(result, freeze)
                    comparison = (validation.get('comparisons') if strict_requested else
                                  verify_claim_comparison(result) if args.require_comparison else None)
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
                    verify_terminal_page(page, validation, strict_requested)
                    report_check = verify_report_reconnect(page, args, progress, runtime, result) if comparison else None
                    page.screenshot(path=str(runtime / f"{args.tag}_final.png"), full_page=True)
                    record = {"input": "real_browser_pdf_upload", "state": result["state"], "progress_file": str(progress),
                              "freeze": freeze, "pdf_sha256": pdf_hash, "validation": validation,
                              "eligibility": eligibility,
                              "recovery": recovery, "outcome_levels": outcome_levels(result),
                              "run_dir": result["data"]["run_dir"], "execution_run_starts": len(starts),
                              "execution_run_ends": len(ends), "observations": observations,
                              "cleanup": cleanup, "live_page_completed": True,
                              "claim_comparison_required": args.require_comparison, "report_checks": report_check,
                              "strict_validation_required": strict_requested,
                              "backend_strict_request": (result['data'].get('strict_validation') or {}).get('requested'),
                              "audited_outcome": terminal_outcome(result),
                              "installation_checks": installation_checks,
                              "live_page_paper_accepted": strict_requested and validation.get('is_reproduced') is True,
                              "live_page_author_code_completed": True}
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
    if independent and not (args.require_comparison or getattr(args, 'strict_validation', False)):
        raise ValueError('--require-independent also requires --require-comparison or --strict-validation')
    return verify_freeze(ROOT, freeze, paper_id=getattr(args, 'paper_id', None),
                         pdf_sha256=pdf_hash, repository=args.repo, require_independent=independent)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pdf", "repo", "freeze", "tag"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--paper-id", help="Paper identity in the shared acceptance registry")
    parser.add_argument("--require-independent", action="store_true",
                        help="Require unused-paper eligibility; separate from independent numerical validation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execution-only", action="store_true", help="Compatibility flag for default author execution")
    mode.add_argument("--strict-validation", action="store_true",
                      help="Request the real browser strict mode and require complete independent numerical evidence and matching reports; regression papers are allowed")
    parser.add_argument("--allow-no-preview-link", action="store_true")
    parser.add_argument('--intelligent-recovery-mode', choices=['off', 'single', 'collaborative'], default='off',
                        help='Set the real PDF investigation mode; default off. Does not authorize source patches.')
    parser.add_argument('--allow-source-repair', action='store_true',
                        help='Explicitly enable automatic bounded compatibility repair; incompatible with strict validation')
    parser.add_argument("--require-comparison", action="store_true",
                        help="Require verified paper comparisons, matching downloads and browser reconnect")
    parser.add_argument("--require-install-progress", action="store_true",
                        help="Require measured installer events and the live browser progress panel")
    parser.add_argument("--credential-stdin", action="store_true", help='Read {"api_key": "..."} from stdin instead of hidden input')
    parser.add_argument("--base-url", help="Override the API URL shown on the page")
    parser.add_argument("--model", help="Override the model shown on the page")
    parser.add_argument("--experiment-method", help="Select an original method/model through the real PDF page")
    parser.add_argument("--experiment-dataset", help="Select an original dataset through the real PDF page")
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
