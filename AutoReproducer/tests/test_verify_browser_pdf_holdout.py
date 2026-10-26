"""Local browser-audit regressions; no browser, network, or training required."""
from contextlib import contextmanager
from copy import deepcopy
import getpass
import hashlib
import io
import json
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace
import warnings
import zipfile

import pytest

from scripts import verify_browser_pdf_holdout as audit


def compared_result():
    observed = {'id': 'train:stdout:1', 'value': 0.8328, 'aggregation': 'mean', 'runs': 10,
                'std': 0.0004, 'source': 'train:stdout:22', 'source_quote': 'mean 0.83280, std 0.00040'}
    row = {'status': 'reference_only', 'difference': -0.02,
           'settings': {'status': 'different', 'reason': 'CPU execution; paper reports GPU.'},
           'limitations': ['CPU execution; paper reports GPU.'],
           'reference': {'value_quote': 'DGC 83.3', 'citations': ['paper']},
           'actual': {**observed, 'citations': ['author']}}
    return {'data': {'validation': {'is_reproduced': None, 'paper_metrics_verified': False,
        'author_claim_comparison': {'status': 'partial', 'metric_records': [observed], 'comparisons': [row],
                                  'citations': [{'id': 'paper', 'source': 'pdf'},
                                                {'id': 'author', 'source': 'repository'}]}}}}


def test_comparison_acceptance_requires_nonempty_verified_results():
    result = compared_result()
    assert audit.verify_claim_comparison(result)['status'] == 'partial'
    unavailable = deepcopy(result)
    unavailable['data']['validation']['author_claim_comparison'].update(status='unavailable', comparisons=[])
    with pytest.raises(AssertionError):
        audit.verify_claim_comparison(unavailable)


@pytest.mark.parametrize('field,wrong', [('value', 0.833), ('runs', 1), ('std', 0.01),
                                       ('aggregation', 'single'), ('source_quote', 'invented output')])
def test_comparison_acceptance_rejects_rewritten_observed_summary(field, wrong):
    result = compared_result()
    result['data']['validation']['author_claim_comparison']['comparisons'][0]['actual'][field] = wrong
    with pytest.raises(AssertionError):
        audit.verify_claim_comparison(result)


@pytest.mark.parametrize('field,value', [('settings', {}), ('settings', {'status': 'different', 'reason': ''}),
                                        ('limitations', []), ('limitations', [''])])
def test_reference_only_cannot_hide_setting_assessments_and_limitations(field, value):
    result = compared_result()
    result['data']['validation']['author_claim_comparison']['comparisons'][0][field] = value
    with pytest.raises(AssertionError):
        audit.verify_claim_comparison(result)


@pytest.mark.parametrize('mismatch', ['runtime_revision', 'plan_revision', 'repository'])
def test_independent_trial_binds_executed_repository_and_plan_to_registered_revision(tmp_path, mismatch):
    expected = {'url': 'https://github.com/fixture/author', 'revision': '1' * 40}
    actual, planned = deepcopy(expected), deepcopy(expected)
    if mismatch == 'runtime_revision':
        actual['revision'] = '2' * 40
    elif mismatch == 'plan_revision':
        planned['revision'] = '2' * 40
    else:
        actual['url'] = 'https://github.com/fixture/other'
    result = {'data': {'repository': actual, 'run_dir': str(tmp_path)}}
    (tmp_path / 'experiment_spec.json').write_text(json.dumps({'spec': {'repository': planned}}))
    freeze = {'paper': {'repository_url': expected['url'], 'repository_revision': expected['revision']}}
    with pytest.raises(AssertionError):
        audit.verify_trial_repository(result, freeze)


def test_independent_trial_accepts_matching_pinned_repository(tmp_path):
    repository = {'url': 'https://github.com/fixture/author', 'revision': '1' * 40}
    result = {'data': {'repository': repository, 'run_dir': str(tmp_path)}}
    (tmp_path / 'experiment_spec.json').write_text(json.dumps({'spec': {'repository': repository}}))
    freeze = {'paper': {'repository_url': repository['url'], 'repository_revision': repository['revision']}}
    assert audit.verify_trial_repository(result, freeze) == repository


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
        if value == 'Space':
            self.checked = not self.checked

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
            "Docker 沙箱执行（暂未开放）": Field(checked=False),
            "严格核验：重载模型并独立复算指标": Field(),
            "兼容性源码修复：失败后自动重试": Field(checked=True),
            "实验方法 / 模型（可选）": Field(),
            "实验数据集（可选）": Field(),
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
    assert not page.fields["Docker 沙箱执行（暂未开放）"].checked


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


@pytest.mark.parametrize('strict', [False, True])
def test_pdf_mode_controls_are_set_explicitly_and_exclude_source_repair(strict):
    page = Page()
    page.fields['严格核验：重载模型并独立复算指标'].checked = not strict
    audit.configure_execution_mode(page, strict)
    assert page.fields['严格核验：重载模型并独立复算指标'].checked is strict
    assert page.fields['兼容性源码修复：失败后自动重试'].checked is False


def test_strict_cli_is_separate_from_unused_paper_and_execution_only(monkeypatch):
    options = ['--pdf', 'a', '--repo', 'b', '--freeze', 'c', '--tag', 'd']
    args = audit.build_parser().parse_args(options + ['--strict-validation'])
    assert args.strict_validation is True and args.require_independent is False
    monkeypatch.setattr(audit, 'verify_freeze', lambda *a, **kw: kw)
    assert audit.verify_acceptance_source(args, {}, 'pdf')['require_independent'] is False
    args.require_independent = True
    assert audit.verify_acceptance_source(args, {}, 'pdf')['require_independent'] is True
    with pytest.raises(SystemExit):
        audit.build_parser().parse_args(options + ['--strict-validation', '--execution-only'])
    ordinary = audit.build_parser().parse_args(options + ['--require-independent', '--require-comparison'])
    assert ordinary.strict_validation is False
    assert audit.verify_acceptance_source(ordinary, {}, 'pdf')['require_independent'] is True


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


@pytest.fixture
def strict_completed_run(completed_run, monkeypatch):
    """Harness boundaries only; the formal verifier has separate native tests."""
    from unittest.mock import Mock

    result, _, _ = completed_run
    result = deepcopy(result)
    data = result['data']
    run_dir = Path(data['run_dir'])
    repository = {'url': 'https://github.com/fixture/author', 'revision': 'a' * 40}
    data['repository'].update(repository)
    data['resources']['code_repo_url'] = repository['url']
    plan = {'scope': 'selected_paper_experiment', 'pdf_sha256': 'pdf-digest', 'repository': repository}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()
    (run_dir / 'experiment_spec.json').write_text(json.dumps({'spec': plan, 'sha256': digest}), 'utf-8')
    (run_dir / 'source_packet.json').write_text('{"original_source": "fixture"}', 'utf-8')
    binding = {'status': 'verified', 'pdf_sha256': 'pdf-digest', 'repository': repository}
    validation = {'status': 'reproduced', 'is_reproduced': True, 'strict_validation': True,
                  'paper_metrics_verified': True, 'protocol_pass': True, 'independent_metrics_pass': True,
                  'strict_plan_verification': binding, 'comparisons': [
                      {'name': 'accuracy', 'reference': 83, 'actual': 82, 'unit': 'percent', 'pass': True}]}
    data.update(validation=validation, reproduction_scope='selected_paper_experiment', spec_sha256=digest,
                strict_validation={'requested': True, 'status': 'verified', 'result': 'reproduced', 'binding': binding},
                source_repair={'requested': False})
    data['execution']['strict_execution_verification'] = validation
    for name, value in [('strict_validation.json', data['strict_validation']),
                        ('strict_execution_verification.json', validation)]:
        (run_dir / name).write_text(json.dumps(value), 'utf-8')
    replay = Mock(return_value=deepcopy(validation))
    monkeypatch.setattr('src.strict_validation.verify_strict_execution', replay)
    return result, replay, plan


@pytest.mark.parametrize('matched', [True, False])
def test_strict_audit_replays_real_artifact_verifier_and_preserves_numeric_conclusion(strict_completed_run, matched):
    result, replay, plan = strict_completed_run
    data = result['data']
    if not matched:
        validation = data['validation']
        validation.update(status='reference_not_met', is_reproduced=False)
        validation['comparisons'][0].update(actual=70, **{'pass': False})
        data['strict_validation']['result'] = 'reference_not_met'
        replay.return_value = deepcopy(validation)
        for name, value in [('strict_validation.json', data['strict_validation']),
                            ('strict_execution_verification.json', validation)]:
            (Path(data['run_dir']) / name).write_text(json.dumps(value), 'utf-8')
    validation = audit.verify_strict_completion(result, 'pdf-digest', 'https://github.com/fixture/author/')
    assert validation['is_reproduced'] is matched
    assert validation['paper_metrics_verified'] is True
    replay.assert_called_once_with(plan, {'original_source': 'fixture'}, Path(data['repository']['path']),
                                   data['execution'], data['spec_sha256'])
    expected = 'strict_verified_reference_met' if matched else 'strict_verified_reference_not_met'
    assert audit.terminal_outcome(result) == expected


@pytest.mark.parametrize('failure', ['backend_request', 'fallback', 'strict_failed', 'execution_failed',
    'source_repair', 'author_output', 'not_verified', 'protocol', 'independent', 'pdf', 'repository',
    'spec', 'binding', 'replay', 'execution_receipt', 'strict_file', 'validation_file'])
def test_strict_audit_cannot_promote_missing_or_stale_evidence(strict_completed_run, failure):
    result, replay, _ = strict_completed_run
    data = result['data']
    strict, validation = data['strict_validation'], data['validation']
    if failure == 'backend_request':
        strict['requested'] = False
    elif failure == 'fallback':
        strict.update(status='unavailable', reason='No full model export')
        validation.update(status='author_code_completed', paper_metrics_verified=False, is_reproduced=None)
        assert audit.terminal_outcome(result) == 'strict_unavailable'
    elif failure == 'strict_failed':
        strict['status'] = 'failed'
        assert audit.terminal_outcome(result) == 'strict_failed'
    elif failure == 'execution_failed':
        result.update(state='ERROR', error='Training did not complete')
        assert audit.terminal_outcome(result) == 'execution_failed'
    elif failure == 'source_repair':
        data['source_repair']['requested'] = True
    elif failure == 'author_output':
        validation['status'] = strict['result'] = 'author_code_completed'
    elif failure == 'not_verified':
        validation['paper_metrics_verified'] = False
    elif failure in {'protocol', 'independent'}:
        validation[failure + ('_pass' if failure == 'protocol' else '_metrics_pass')] = False
    elif failure == 'pdf':
        data['pdf_input']['sha256'] = 'different'
    elif failure == 'repository':
        data['repository']['revision'] = 'b' * 40
    elif failure == 'spec':
        data['spec_sha256'] = 'different'
    elif failure == 'binding':
        strict['binding'] = {}
    elif failure == 'replay':
        replay.return_value['comparisons'][0]['actual'] = 84
    elif failure == 'execution_receipt':
        data['execution']['strict_execution_verification'] = {}
    elif failure in {'strict_file', 'validation_file'}:
        name = 'strict_validation.json' if failure == 'strict_file' else 'strict_execution_verification.json'
        (Path(data['run_dir']) / name).write_text('{}', 'utf-8')
    with pytest.raises(AssertionError):
        audit.verify_strict_completion(result, 'pdf-digest', 'https://github.com/fixture/author')


def test_author_logs_cannot_replace_actual_independent_predictions(strict_completed_run):
    result, replay, _ = strict_completed_run
    result['data']['validation']['author_claim_comparison'] = compared_result()['data']['validation']['author_claim_comparison']
    replay.side_effect = ValueError('A native repetition lacks fresh model predictions')
    with pytest.raises(ValueError, match='fresh model predictions'):
        audit.verify_strict_completion(result, 'pdf-digest', 'https://github.com/fixture/author')


@pytest.mark.parametrize('strict,matched', [(False, None), (True, True), (True, False)])
def test_live_terminal_view_checks_actual_numeric_or_author_conclusion(strict, matched):
    labels = []

    class TerminalPage:
        def get_by_text(self, name, **kwargs):
            labels.append(name.pattern if isinstance(name, re.Pattern) else name)
            return SimpleNamespace(wait_for=lambda **kwargs: None)

    validation = {'is_reproduced': matched, 'training_observed': False}
    audit.verify_terminal_page(TerminalPage(), validation, strict)
    if not strict:
        assert '本次完成的是代码运行，论文数值未做验收。' in labels
    elif matched:
        assert '完整实验已完成，论文数值验收通过。' in labels
    else:
        assert any('独立评估数值未达到固定参考' in label for label in labels)
    assert labels[-1] == '任务完成'


@pytest.mark.parametrize('changed', [None, 'download', 'comparison', 'run_report', 'stored_report', 'runtime', 'strict'])
def test_browser_download_must_match_saved_reports_and_structured_comparison(completed_run, changed):
    result, _, _ = completed_run
    report = Path(result['report_path'])
    run_dir = Path(result['data']['run_dir'])
    downloaded = report.read_text('utf-8').encode('utf-8')
    (run_dir / 'report.md').write_text(downloaded.decode('utf-8'), encoding='utf-8')
    if changed == 'download':
        downloaded += b'changed'
    elif changed == 'comparison':
        path = report.with_suffix('.result.json')
        saved = json.loads(path.read_text('utf-8'))
        saved['data']['validation']['author_claim_comparison'] = {'comparisons': []}
        path.write_text(json.dumps(saved), encoding='utf-8')
    elif changed == 'run_report':
        (run_dir / 'report.md').write_text('another report', encoding='utf-8')
    elif changed in {'stored_report', 'runtime', 'strict'}:
        path = report.with_suffix('.result.json')
        saved = json.loads(path.read_text('utf-8'))
        key = {'stored_report': 'report', 'runtime': 'runtime_preparation', 'strict': 'strict_validation'}[changed]
        saved['data'][key] = 'stale rendering'
        path.write_text(json.dumps(saved), encoding='utf-8')
    if changed:
        with pytest.raises(AssertionError):
            audit.verify_saved_report(result, downloaded)
    else:
        assert audit.verify_saved_report(result, downloaded)['structured_records_identical'] is True


@pytest.mark.parametrize('matched', [True, False])
@pytest.mark.parametrize('wrong', ['actual', 'reference', 'unit', 'metric', 'status', 'fenced_table', 'duplicate'])
def test_consistent_downloads_cannot_override_formal_strict_metrics(strict_completed_run, matched, wrong):
    """Every persisted/downloaded copy agrees, but disagrees with formal evidence."""
    result = strict_completed_run[0]
    validation = result['data']['validation']
    validation.update(status='reproduced' if matched else 'reference_not_met', is_reproduced=matched)
    validation['comparisons'][0].update(actual=82 if matched else 70, **{'pass': matched})
    status = '选定论文实验数值复现通过' if matched else '选定论文实验已完成，数值参考未达到'
    report = ('- **复现状态**: ' + status + '\n- **严格验收**: 完整独立复算证据已核验\n'
              '### 指标对比\n| 指标 | 论文声明 | 实际运行 |\n|---|---|---|\n'
              '| accuracy | 83% | ' + str(82 if matched else 70) + '% |\n')
    if wrong == 'actual':
        report = report.replace(str(82 if matched else 70) + '% |', '999% |')
    elif wrong == 'reference':
        report = report.replace('83%', '84%')
    elif wrong == 'unit':
        report = report.replace('%', ' (比例)')
    elif wrong == 'metric':
        report = report.replace('accuracy', 'cross_entropy')
    elif wrong == 'status':
        report = report.replace(status, '选定论文实验已完成，数值参考未达到' if matched else '选定论文实验数值复现通过')
    elif wrong == 'fenced_table':
        report = report.replace('### 指标对比', '```markdown\n### 指标对比') + '```\n'
    else:
        report += '| accuracy | 83% | ' + str(82 if matched else 70) + '% |\n'
    result['data']['report'] = report
    report_path = Path(result['report_path'])
    run_dir = Path(result['data']['run_dir'])
    for path in (report_path, run_dir / 'report.md'):
        path.write_text(report, 'utf-8')
    for path in (report_path.with_suffix('.result.json'), run_dir / 'result.json'):
        path.write_text(json.dumps(result), 'utf-8')
    with pytest.raises(AssertionError):
        audit.verify_saved_report(result, report.encode())


def test_strict_table_and_markdown_keep_full_renderer_precision_and_explicit_units():
    validation = {'status': 'reference_not_met', 'is_reproduced': False, 'comparisons': [
        {'name': 'cross_entropy', 'reference': 2.1e-08, 'actual': 0.12345678901234567,
         'unit': 'cross_entropy', 'pass': False},
        {'name': 'accuracy', 'reference': 0.82, 'actual': 0.8123456789012345,
         'unit': 'fraction', 'pass': True}]}
    rows = [['指标', '论文声明', '实际运行'],
            ['accuracy', '0.82 (比例)', '0.8123456789012345 (比例)'],
            ['cross_entropy', '2.1e-08 cross_entropy', '0.12345678901234566 cross_entropy']]
    report = ('- **复现状态**: ⚠️ 选定论文实验已完成，数值参考未达到\n'
              '- **严格验收**: 完整独立复算证据已核验\n### 指标对比\n' +
              '\n'.join('| ' + ' | '.join(row) + ' |' for row in [rows[0], ['---'] * 3, *rows[1:]]))
    assert audit.verify_strict_report(report, validation) == rows

    class Row:
        def __init__(self, cells):
            self.cells = cells

        def locator(self, selector):
            assert selector == 'th, td'
            return SimpleNamespace(all_inner_texts=lambda: self.cells)

    table = SimpleNamespace(locator=lambda selector: SimpleNamespace(all=lambda: [Row(row) for row in rows]))
    audit.verify_strict_table(table, validation)
    rows[1][2] = '0.8123 (比例)'
    with pytest.raises(AssertionError, match='Displayed strict metric'):
        audit.verify_strict_table(table, validation)


@pytest.mark.parametrize('strict,matched,failure', [
    (strict, matched, failure)
    for strict, matched in [(False, None), (True, True), (True, False)]
    for failure in [None, 'reload', 'zip', 'attachment', 'manifest_hash', 'manifest_bytes',
                    'evidence_validation', 'evidence_conclusion']
] + [
    (True, matched, failure)
    for matched in [True, False]
    for failure in ['wrong_actual', 'wrong_reference', 'wrong_unit', 'wrong_metric',
                    'wrong_live_actual', 'wrong_reloaded_actual', 'wrong_conclusion', 'wrong_pass']
])
def test_browser_report_download_and_reload_preserve_each_mode_conclusion(
        completed_run, strict_completed_run, tmp_path, strict, matched, failure):
    result = strict_completed_run[0] if strict else completed_run[0]
    validation = result['data']['validation']
    if strict:
        validation.update(is_reproduced=matched, status='reproduced' if matched else 'reference_not_met')
        validation['comparisons'][0].update(actual=82 if matched else 70, **{'pass': matched})
        actual = 82 if matched else 70
        cells = ['accuracy', '83%', str(actual) + '%']
        if failure == 'wrong_actual':
            cells[2] = str(actual + 1) + '%'
        elif failure == 'wrong_reference':
            cells[1] = '84%'
        elif failure == 'wrong_unit':
            cells[2] = str(actual) + ' (比例)'
        elif failure == 'wrong_metric':
            cells[0] = 'cross_entropy'
        elif failure == 'wrong_pass':
            validation['comparisons'][0]['pass'] = not matched
        state_text = ('选定论文实验数值复现通过' if matched else '选定论文实验已完成，数值参考未达到')
        if failure == 'wrong_conclusion':
            state_text = ('选定论文实验已完成，数值参考未达到' if matched else '选定论文实验数值复现通过')
        report = ('# Report\n- **严格验收**: 完整独立复算证据已核验\n- **复现状态**: ' +
                  state_text + '\n### 指标对比\n| 指标 | 论文声明 | 实际运行 |\n|---|---|---|\n| ' +
                  ' | '.join(cells) + ' |')
    else:
        cells = ['accuracy', '83', '82']
        report = '# Report\n作者输出与论文宣称对比\n| 指标 | 论文宣称 | 作者运行输出 |\n|---|---|---|\n| accuracy | 83 | 82 |'
    result['data']['report'] = report
    report_path = Path(result['report_path'])
    run_dir = Path(result['data']['run_dir'])
    report_path.write_text(report, 'utf-8')
    (run_dir / 'report.md').write_text(report, 'utf-8')
    for path in (report_path.with_suffix('.result.json'), run_dir / 'result.json'):
        path.write_text(json.dumps(result), 'utf-8')
    attachment = report_path.with_name(report_path.stem + '_execution.txt')
    attachment.write_bytes(b'Full original stdout and stderr')
    (run_dir / attachment.name).write_bytes(attachment.read_bytes())
    from src.report_bundle import report_evidence_bytes, report_manifest_bytes
    evidence_name = report_path.stem + '.evidence.json'
    manifest_name = report_path.stem + '.manifest.json'
    evidence = json.loads(report_evidence_bytes(result, report, report_path.name))
    if failure == 'evidence_validation':
        evidence['data']['validation']['is_reproduced'] = 'tampered'
    elif failure == 'evidence_conclusion':
        evidence['data']['reproduction_conclusion']['is_reproduced'] = 'tampered'
    payloads = {
        report_path.name: report.encode() + (b'changed' if failure == 'zip' else b''),
        attachment.name: b'changed' if failure == 'attachment' else attachment.read_bytes(),
        evidence_name: json.dumps(evidence).encode(),
        'report_assets/paper_markdown_fixture.md': b'Additional PDF reading evidence',
    }
    manifest = json.loads(report_manifest_bytes(payloads, report_path.name, evidence_name))
    if failure == 'manifest_hash':
        manifest['files'][0]['sha256'] = '0' * 64
    elif failure == 'manifest_bytes':
        manifest['files'][0]['bytes'] += 1
    payloads[manifest_name] = json.dumps(manifest).encode()
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, 'w') as archive:
        for name, raw in payloads.items():
            archive.writestr(name, raw)
    downloads = {'下载报告': report.encode(), '下载报告与证据（ZIP）': zipped.getvalue()}

    class Table:
        def filter(self, *, has_text):
            assert has_text in report
            return self

        @property
        def first(self):
            return self

        def wait_for(self, **kwargs):
            pass

        def inner_text(self):
            return 'stale comparison' if page.reloaded and failure == 'reload' else 'accuracy 83 82'

        def locator(self, selector):
            assert selector == 'tr'
            visible = cells.copy()
            if failure == 'wrong_live_actual' or (page.reloaded and failure == 'wrong_reloaded_actual'):
                visible[2] = '999%'
            rows = [['指标', '论文声明', '实际运行'], visible]
            return SimpleNamespace(all=lambda: [SimpleNamespace(locator=lambda selector, row=row:
                SimpleNamespace(all_inner_texts=lambda: row)) for row in rows])

    class ReportPage:
        reloaded = False
        download_label = None

        def get_by_role(self, role, *, name=None, exact=None):
            if role == 'table':
                return Table()
            if role == 'tab':
                return SimpleNamespace(click=lambda: None)
            assert role == 'button'
            return SimpleNamespace(click=lambda: setattr(self, 'download_label', name))

        def get_by_text(self, marker):
            assert marker.search(report)
            return SimpleNamespace(first=SimpleNamespace(wait_for=lambda **kwargs: None))

        @contextmanager
        def expect_download(self, **kwargs):
            yield SimpleNamespace(value=SimpleNamespace(
                save_as=lambda path: Path(path).write_bytes(downloads[self.download_label])))

        def goto(self, *args, **kwargs):
            self.reloaded = True

    page = ReportPage()
    args = SimpleNamespace(strict_validation=strict, tag='report', port=8501)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    if failure:
        with pytest.raises(AssertionError):
            audit.verify_report_reconnect(page, args, Path('progress_fixture.jsonl'), runtime, result)
    else:
        checked = audit.verify_report_reconnect(page, args, Path('progress_fixture.jsonl'), runtime, result)
        assert checked['reconnect_preserved_comparison'] is True
        assert checked['structured_records_identical'] is True and checked['markdown_zip_identical'] is True
        assert checked['bundle_manifest_verified'] is True and checked['bundle_evidence_consistent'] is True


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
