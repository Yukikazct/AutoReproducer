"""Strict requests survive web/worker handoff without leaking across input modes."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

import frontend.backend_pipeline as backend
import frontend.history_manager as history
import frontend.pipeline_entrypoint as loader
import frontend.repository_entrypoint as repository_entrypoint
import src.local_llm_settings as settings
import src.runtime_preparation as runtime
from src.corpus import list_papers
from test_pdf_input_routing import TITLE, write_pdf


APP = Path(__file__).resolve().parents[1] / 'app.py'
GENERIC_TITLE = 'A New Independent Evaluation Study'


@pytest.mark.parametrize('strict', [False, True])
def test_backend_forwards_explicit_strict_request_and_keeps_legacy_reviews_disabled(
        monkeypatch, tmp_path, strict):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
                              experiment_request={'method': 'Fixture', 'dataset': 'Grid'},
                              strict_validation=strict, _managed_runtime=True,
                              use_llm_review=True, allow_result_summary_review=True,
                              enable_optimization=True, optimization_mode='validate')
    request = service.run.call_args.args[0]
    assert request['strict_validation'] is strict
    assert request['experiment_request'] == {'method': 'Fixture', 'dataset': 'Grid'}
    assert request['author_execution_only'] is True
    assert request['use_llm_review'] is False and request['allow_result_summary_review'] is False
    assert request['enable_optimization'] is False and request['optimization_mode'] == 'off'


@pytest.mark.parametrize('title', ['Neural Ordinary Differential Equations',
    'Are Transformers Effective for Time Series Forecasting?', GENERIC_TITLE])
def test_strict_title_backend_does_not_auto_select_named_experiment(monkeypatch, tmp_path, title):
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), paper_title=title,
                              strict_validation=True, _managed_runtime=True)
    request = service.run.call_args.args[0]
    assert request['paper_title'] == title
    assert request['strict_validation'] is True
    assert request['experiment_profile'] is None
    assert not request['pdf_path']


def test_explicit_named_profile_is_retained_for_orchestrator_strict_rejection(monkeypatch, tmp_path):
    service = SimpleNamespace(run=Mock(return_value={'state': 'ERROR', 'error': 'strict mode requires PDF',
                                                   'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    result = backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'),
        paper_title='Neural Ordinary Differential Equations', experiment_profile='neural_ode_spiral',
        strict_validation=True, _managed_runtime=True)
    request = service.run.call_args.args[0]
    assert request['strict_validation'] is True
    assert request['experiment_profile'] == 'neural_ode_spiral'
    assert result['state'] == 'ERROR'


def test_strict_request_survives_background_and_prepared_runtime_handoff(monkeypatch, tmp_path):
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
        assert command[0] == prepared.executable
        assert request['strict_validation'] is True
        assert request['experiment_request'] == {'dataset': 'Grid'}
        backend.ProgressStore(request['progress_path'], reset=False).emit({
            'type': 'done', 'result': {'state': 'COMPLETED', 'data': {}}})
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(runtime, 'run_owned_process', worker)
    thread = backend.run_pipeline_background(str(progress), pdf_path=str(pdf),
                                              experiment_request={'dataset': 'Grid'},
                                              strict_validation=True, cleanup_pdf=False)
    thread.join(timeout=5)
    assert not thread.is_alive()
    request, = requests
    assert request['_managed_runtime'] is True
    assert request['strict_validation'] is True
    assert backend.ProgressStore.read_snapshot(str(progress))['done'] is True


def test_cached_backend_before_strict_mode_gets_private_upgrade(monkeypatch):
    from types import ModuleType
    import sys

    old = ModuleType('frontend.backend_pipeline')
    old.BACKEND_API_VERSION = 6
    old.ProgressStore = Mock()
    old.run_pipeline_background = Mock()
    monkeypatch.setitem(sys.modules, 'frontend.backend_pipeline', old)
    upgraded = loader.load_backend_pipeline()
    assert upgraded is not old and upgraded.BACKEND_API_VERSION == loader.BACKEND_API_VERSION
    assert sys.modules['frontend.backend_pipeline'] is old
    old.run_pipeline_background.assert_not_called()


@pytest.fixture
def strict_app(monkeypatch, tmp_path):
    capture = Mock()
    monkeypatch.setattr(repository_entrypoint, 'background_entrypoint', lambda previous: capture)
    monkeypatch.setattr(settings, 'load_local_llm_settings', Mock())
    monkeypatch.setattr(history, 'get_project_data_dir', lambda: tmp_path)
    monkeypatch.delenv('AUTOREPRO_RESUME_PROGRESS', raising=False)
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.invalid')
    monkeypatch.setenv('LLM_MODEL', 'fixture-model')
    monkeypatch.setenv('LLM_API_KEY', 'fixture-key')
    def make(mode='上传PDF', *, title=GENERIC_TITLE, mock=False, corpus=None):
        pdf = write_pdf(tmp_path / 'paper.pdf', [title, 'Abstract'])
        upload = UploadedFile(UploadedFileRec('strict-upload', 'paper.pdf', 'application/pdf',
                                             pdf.read_bytes()), FileURLs())

        def file_uploader(*args, **kwargs):
            st.session_state[kwargs['key']] = upload
            return upload

        monkeypatch.setattr(st, 'file_uploader', file_uploader)
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state['input_mode'] = mode
        app.session_state['mock_mode'] = mock
        app.session_state['paper_title'] = title
        app.session_state['experiment_profile'] = 'dlinear_etth1_smoke'
        if corpus:
            app.session_state['corpus_paper_select'] = corpus
        app.run()
        assert not app.exception
        return app

    yield make, capture
    if capture.called:
        path = capture.call_args.kwargs.get('pdf_path')
        if path:
            Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize('strict', [False, True])
def test_generic_real_pdf_checkbox_requests_strict_validation_only_when_selected(strict_app, strict):
    make, capture = strict_app
    app = make()
    checkbox = app.sidebar.checkbox(key='pdf_strict_validation')
    assert checkbox.value is False
    if strict:
        checkbox.check().run()
        assert any('结果以实际检查为准' in row.value for row in app.caption)
    app.sidebar.text_input(key='pdf_requested_method').set_value('Fixture').run()
    app.sidebar.text_input(key='pdf_requested_dataset').set_value('Grid').run()
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception
    request = capture.call_args.kwargs
    assert request['strict_validation'] is strict
    assert request['experiment_request'] == {'method': 'Fixture', 'dataset': 'Grid'}
    assert request['author_execution_only'] is True
    assert request['experiment_profile'] is None
    assert request['use_llm_review'] is False and request['enable_optimization'] is False


@pytest.mark.parametrize('route', ['title', 'preset', 'known_pdf', 'mock', 'corpus'])
def test_other_input_modes_ignore_stale_pdf_strict_selection(strict_app, route):
    make, capture = strict_app
    options = {
        'title': {'mode': '论文标题'},
        'preset': {'mode': '官方仓库预设'},
        'known_pdf': {'title': TITLE},
        'mock': {'mock': True},
        'corpus': {'corpus': list_papers()[0]['id']},
    }[route]
    app = make(**options)
    app.session_state['pdf_strict_validation'] = True
    app.session_state['pdf_requested_method'] = 'Stale method'
    app.session_state['pdf_requested_dataset'] = 'Stale data'
    app.run()
    assert not app.exception
    assert 'pdf_strict_validation' not in [row.key for row in app.sidebar.checkbox]
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception
    assert capture.call_args.kwargs['strict_validation'] is False
    assert capture.call_args.kwargs['experiment_request'] == {}
