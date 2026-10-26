"""Real web/worker request wiring and acceptance mode boundaries."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import subprocess

import pytest

import frontend.backend_pipeline as backend
import src.runtime_preparation as runtime
from scripts import verify_browser_pdf_holdout as audit
from src.intelligent_recovery import (current_investigation, investigating_run,
                                      normalize_intelligent_recovery_request)
from test_pdf_input_routing import write_pdf
from test_strict_validation_entrypoint import GENERIC_TITLE, strict_app
from test_verify_browser_pdf_holdout import Page
from test_source_repair_service import repair_case
from src import source_repair as repair


@pytest.mark.parametrize('mode', ['single', 'collaborative'])
def test_real_pdf_backend_forwards_role_mode(monkeypatch, tmp_path, mode):
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
        intelligent_recovery=True, intelligent_recovery_mode=mode, _managed_runtime=True)
    request = service.run.call_args.args[0]
    assert request['intelligent_recovery'] is True
    assert request['intelligent_recovery_mode'] == mode


@pytest.mark.parametrize('mode', ['single', 'collaborative'])
def test_role_mode_survives_background_and_owned_worker(monkeypatch, tmp_path, mode):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    progress = tmp_path / 'progress.jsonl'
    prepared = runtime.RuntimePreparation(str(tmp_path / 'safe-python.exe'), True, False,
                                          'isolated runtime fixture', 'cpython-312')
    monkeypatch.setattr(runtime, 'runtime_requirement', lambda: 'isolated runtime fixture')
    monkeypatch.setattr(runtime, 'prepare_runtime', Mock(return_value=prepared))
    requests = []

    def worker(command, **kwargs):
        request = json.loads(kwargs['input'])
        requests.append(request)
        backend.ProgressStore(request['progress_path'], reset=False).emit({
            'type': 'done', 'result': {'state': 'COMPLETED', 'data': {}}})
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(runtime, 'run_owned_process', worker)
    thread = backend.run_pipeline_background(str(progress), pdf_path=str(pdf),
        intelligent_recovery=True, intelligent_recovery_mode=mode, cleanup_pdf=False)
    thread.join(timeout=5)
    assert not thread.is_alive()
    request, = requests
    assert request['_managed_runtime'] is True
    assert request['intelligent_recovery_mode'] == mode
    assert backend.ProgressStore.read_snapshot(str(progress))['done'] is True


@pytest.mark.parametrize('mode', ['single', 'collaborative'])
def test_real_pdf_page_submits_selected_mode(strict_app, mode):
    make, capture = strict_app
    app = make()
    assert 'pdf_intelligent_recovery_mode' not in [row.key for row in app.sidebar.selectbox]
    app.sidebar.checkbox(key='pdf_intelligent_recovery').check().run()
    app.sidebar.selectbox(key='pdf_intelligent_recovery_mode').select(mode).run()
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception
    assert capture.call_args.kwargs['intelligent_recovery'] is True
    assert capture.call_args.kwargs['intelligent_recovery_mode'] == mode


@pytest.mark.parametrize('mode', ['', 'off', 'analyst', None, [], True])
def test_invalid_role_mode_cannot_initialize_models(monkeypatch, tmp_path, mode):
    model = Mock(side_effect=AssertionError('invalid mode initialized model'))
    monkeypatch.setattr(backend, 'LLMClient', model)
    result = backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'),
        intelligent_recovery=True, intelligent_recovery_mode=mode)
    assert result['state'] == 'ERROR'
    assert 'intelligent_recovery_mode' in result['error']
    model.assert_not_called()


def test_disabled_recovery_drops_stale_single_selection():
    assert normalize_intelligent_recovery_request({'intelligent_recovery': False,
        'intelligent_recovery_mode': 'single'})['intelligent_recovery_mode'] == 'collaborative'


@pytest.mark.parametrize('mode', ['single', 'collaborative'])
def test_service_scope_uses_requested_role_mode(mode):
    class Service:
        llm = object()

        @investigating_run
        def run(self, request):
            return current_investigation().mode

    assert Service().run({'intelligent_recovery': True, 'intelligent_recovery_mode': mode}) == mode
    assert current_investigation() is None


@pytest.mark.parametrize('mode', ['off', 'single', 'collaborative'])
@pytest.mark.parametrize('repair', [False, True])
def test_browser_controls_set_actual_role_and_separate_patch_authorization(mode, repair):
    page = Page()
    audit.configure_execution_mode(page, False, recovery_mode=mode, source_repair=repair)
    assert page.fields['智能调查与排障'].checked is (mode != 'off')
    assert page.fields['兼容性源码修复：失败后自动重试'].checked is repair
    if mode != 'off':
        assert page.fields['调查方式'].value == mode
    if repair:
        assert page.fields['限定修复文件（可选，每行一个仓库相对路径）'].value == ''


def test_cli_defaults_off_and_strict_patch_conflict_is_rejected():
    args = audit.build_parser().parse_args(['--pdf', 'p', '--repo', 'r', '--freeze', 'f', '--tag', 't'])
    assert audit.execution_settings(args) == {'intelligent_recovery_mode': 'off',
        'source_repair': False, 'strict_validation': False}
    args.intelligent_recovery_mode = 'single'
    args.allow_source_repair = args.strict_validation = True
    with pytest.raises(ValueError, match='cannot be combined'):
        audit.execution_settings(args)
    with pytest.raises(ValueError, match='cannot be combined'):
        audit.configure_execution_mode(Page(), True, source_repair=True)


def test_receipt_rejects_wrong_backend_mode_and_unrequested_repair():
    requested = {'intelligent_recovery_mode': 'single', 'source_repair': False,
                 'strict_validation': False}
    result = {'data': {'intelligent_recovery': {'enabled': True, 'mode': 'collaborative',
        'usage': {}}, 'source_repair': {'requested': False}}}
    with pytest.raises(AssertionError, match='role mode differs'):
        audit.verify_recovery_request(result, requested)
    result['data']['intelligent_recovery']['mode'] = 'single'
    assert audit.verify_recovery_request(result, requested)['actual_mode'] == 'single'
    result['data']['source_repair']['requested'] = True
    with pytest.raises(AssertionError, match='source repair authorization differs'):
        audit.verify_recovery_request(result, requested)


def test_four_outcome_levels_are_recomputed_not_taken_from_saved_success():
    result = {'data': {'reproduction_conclusion': {'is_reproduced': True},
        'execution': {'success': False}, 'validation': {'status': 'execution_failed'}}}
    levels = audit.outcome_levels(result)
    assert set(levels) == {'scope', 'execution', 'comparison', 'independent', 'is_reproduced'}
    assert levels['is_reproduced'] is None
    assert levels['execution']['status'] == 'failed'


@pytest.mark.parametrize('tamper', [None, 'manifest', 'command'])
def test_author_audit_replays_real_compatibility_repair_without_claiming_strict_success(repair_case, tamper):
    case = repair_case
    execution = case['run']()
    data = case['data']
    data.update(grounded_repository_plan=case['plan'], execution=execution,
                pdf_input={'sha256': 'a' * 64}, repository_executed=True,
                resources={'code_repo_url': case['plan']['repository']['url']})
    data['validation'] = repair.verify_repaired_execution(case['plan'], data, execution)
    result = {'state': 'COMPLETED', 'data': data}
    with pytest.raises(AssertionError, match='not authorized'):
        audit.verify_author_run(result, 'a' * 64, case['plan']['repository']['url'])
    if tamper == 'manifest':
        from pathlib import Path
        Path(data['source_repair']['manifest_path']).write_text('{}', encoding='utf-8')
    elif tamper == 'command':
        execution['steps'][0]['argv'].append('--different-training-setting')
    if tamper:
        with pytest.raises(ValueError):
            audit.verify_author_run(result, 'a' * 64, case['plan']['repository']['url'], source_repair=True)
    else:
        validation = audit.verify_author_run(result, 'a' * 64,
            case['plan']['repository']['url'], source_repair=True)
        assert validation['source_modified'] is True
        assert audit.outcome_levels(result)['is_reproduced'] is None
