"""Recovery permission survives web and worker boundaries without stale consent."""
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import frontend.backend_pipeline as backend
import frontend.pipeline_entrypoint as loader
import src.runtime_preparation as runtime
from src.corpus import list_papers
from test_pdf_input_routing import TITLE, write_pdf
from test_strict_validation_entrypoint import GENERIC_TITLE, strict_app


@pytest.mark.parametrize('enabled', [False, True])
def test_backend_passes_explicit_recovery_permission_to_service(monkeypatch, tmp_path, enabled):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
                              intelligent_recovery=enabled, _managed_runtime=True,
                              use_llm_review=True, allow_result_summary_review=True)
    request = service.run.call_args.args[0]
    assert request['intelligent_recovery'] is enabled
    assert request['use_llm_review'] is False
    assert request['allow_result_summary_review'] is False


@pytest.mark.parametrize('invalid', ['true', 'false', 1, 0, None, {}, []])
def test_backend_rejects_implicit_recovery_consent_before_loading_models(monkeypatch, tmp_path, invalid):
    model = Mock(side_effect=AssertionError('invalid input must not load a model'))
    worker = Mock(side_effect=AssertionError('invalid input must not start a worker'))
    monkeypatch.setattr(backend, 'LLMClient', model)
    monkeypatch.setattr(backend, '_run_prepared_pipeline', worker)
    result = backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'),
                                      intelligent_recovery=invalid)
    assert result['state'] == 'ERROR'
    assert 'intelligent_recovery' in result['error']
    model.assert_not_called()
    worker.assert_not_called()


def test_backend_default_leaves_recovery_disabled(monkeypatch, tmp_path):
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), paper_title=GENERIC_TITLE,
                              _managed_runtime=True)
    assert service.run.call_args.args[0]['intelligent_recovery'] is False


@pytest.mark.parametrize('enabled', [False, True])
def test_recovery_permission_survives_background_and_managed_worker(monkeypatch, tmp_path, enabled):
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
        intelligent_recovery=enabled, cleanup_pdf=False)
    thread.join(timeout=5)
    assert not thread.is_alive()
    request, = requests
    assert request['_managed_runtime'] is True
    assert request['intelligent_recovery'] is enabled
    assert backend.ProgressStore.read_snapshot(str(progress))['done'] is True


def test_cached_backend_is_upgraded_for_new_recovery_argument(monkeypatch):
    from types import ModuleType
    import sys

    old = ModuleType('frontend.backend_pipeline')
    old.BACKEND_API_VERSION = 12
    old.ProgressStore = Mock()
    old.run_pipeline_background = Mock()
    monkeypatch.setitem(sys.modules, 'frontend.backend_pipeline', old)
    upgraded = loader.load_backend_pipeline()
    assert upgraded is not old
    assert upgraded.BACKEND_API_VERSION == loader.BACKEND_API_VERSION
    assert sys.modules['frontend.backend_pipeline'] is old


@pytest.mark.parametrize('enabled', [False, True])
def test_generic_pdf_recovery_checkbox_defaults_off_and_submits_bool(strict_app, enabled):
    make, capture = strict_app
    app = make()
    checkbox = app.sidebar.checkbox(key='pdf_intelligent_recovery')
    assert checkbox.value is False
    if enabled:
        checkbox.check().run()
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception
    request = capture.call_args.kwargs
    assert request['intelligent_recovery'] is enabled
    assert request['use_llm_review'] is False


@pytest.mark.parametrize('route', ['title', 'preset', 'known_pdf', 'mock', 'corpus'])
def test_other_routes_ignore_stale_pdf_recovery_selection(strict_app, route):
    make, capture = strict_app
    options = {
        'title': {'mode': '论文标题'}, 'preset': {'mode': '官方仓库预设'},
        'known_pdf': {'title': TITLE}, 'mock': {'mock': True},
        'corpus': {'corpus': list_papers()[0]['id']},
    }[route]
    app = make(**options)
    app.session_state['pdf_intelligent_recovery'] = True
    app.run()
    assert not app.exception
    assert 'pdf_intelligent_recovery' not in [row.key for row in app.sidebar.checkbox]
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception
    assert capture.call_args.kwargs['intelligent_recovery'] is False
