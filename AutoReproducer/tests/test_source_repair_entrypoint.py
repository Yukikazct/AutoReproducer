"""Explicit PDF source-repair permissions survive the web and worker boundary."""
import json
from pathlib import Path
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
def test_backend_forwards_only_enabled_source_repair_permissions(monkeypatch, tmp_path, enabled):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    editable = ['src/model.py', 'train.py']
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
                              source_repair=enabled, source_repair_editable=editable,
                              strict_validation=not enabled, _managed_runtime=True)
    request = service.run.call_args.args[0]
    assert request['source_repair'] is enabled
    assert request['source_repair_editable'] == (editable if enabled else [])
    assert request['strict_validation'] is (not enabled)
    assert editable == ['src/model.py', 'train.py']


def test_backend_default_does_not_grant_source_repair_permission(monkeypatch, tmp_path):
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), paper_title=GENERIC_TITLE,
                              _managed_runtime=True)
    request = service.run.call_args.args[0]
    assert request['source_repair'] is False
    assert request['source_repair_editable'] == []


@pytest.mark.parametrize('extra', [{}, {'source_repair_editable': []}])
def test_backend_forwards_automatic_failure_selection_to_worker(monkeypatch, tmp_path, extra):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    monkeypatch.setattr(runtime, 'runtime_requirement', lambda: 'fixture worker')
    handoff = Mock(return_value={'state': 'COMPLETED', 'data': {}})
    monkeypatch.setattr(backend, '_run_prepared_pipeline', handoff)
    backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
                              source_repair=True, source_repair_auto=True,
                              **extra)
    request = handoff.call_args.args[1]
    assert request['source_repair_auto'] is True
    service = SimpleNamespace(run=Mock(return_value={'state': 'COMPLETED', 'data': {}}))
    monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: service)
    request['_managed_runtime'] = True
    backend.run_pipeline_core(**request)
    assert service.run.call_args.args[0]['source_repair_auto'] is True
    assert service.run.call_args.args[0]['source_repair_editable'] == []


def test_backend_rejects_conflicting_modes_before_orchestration(monkeypatch, tmp_path):
    pdf = write_pdf(tmp_path / 'paper.pdf', [GENERIC_TITLE, 'Abstract'])
    create = Mock(side_effect=AssertionError('invalid modes must not start'))
    monkeypatch.setattr(backend, '_create_orchestrator', create)
    result = backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=str(pdf),
        source_repair=True, source_repair_editable=['train.py'], strict_validation=True,
        _managed_runtime=True)
    assert result['state'] == 'ERROR'
    assert '不能同时启用' in result['error']
    create.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_source_repair_survives_background_and_prepared_runtime(monkeypatch, tmp_path, enabled):
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
        backend.ProgressStore(request['progress_path'], reset=False).emit({
            'type': 'done', 'result': {'state': 'COMPLETED', 'data': {}}})
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(runtime, 'run_owned_process', worker)
    thread = backend.run_pipeline_background(str(progress), pdf_path=str(pdf),
        source_repair=enabled, source_repair_editable=['train.py'],
        strict_validation=not enabled, cleanup_pdf=False)
    thread.join(timeout=5)
    assert not thread.is_alive()
    request, = requests
    assert request['_managed_runtime'] is True
    assert request['source_repair'] is enabled
    assert request['source_repair_editable'] == (['train.py'] if enabled else [])
    assert request['strict_validation'] is (not enabled)
    assert backend.ProgressStore.read_snapshot(str(progress))['done'] is True


def test_cached_backend_without_source_repair_gets_private_upgrade(monkeypatch):
    from types import ModuleType
    import sys

    old = ModuleType('frontend.backend_pipeline')
    old.BACKEND_API_VERSION = 7
    old.ProgressStore = Mock()
    old.run_pipeline_background = Mock()
    monkeypatch.setitem(sys.modules, 'frontend.backend_pipeline', old)
    upgraded = loader.load_backend_pipeline()
    assert upgraded is not old and upgraded.BACKEND_API_VERSION == loader.BACKEND_API_VERSION
    assert sys.modules['frontend.backend_pipeline'] is old
    old.run_pipeline_background.assert_not_called()


def _start(app):
    next(button for button in app.sidebar.button if button.label == '开始复现').click().run()
    assert not app.exception


def test_pdf_source_repair_defaults_to_bounded_automatic_selection(strict_app):
    make, capture = strict_app
    app = make()
    assert app.sidebar.checkbox(key='pdf_source_repair').value is True
    _start(app)
    request = capture.call_args.kwargs
    assert request['source_repair'] is True
    assert request['source_repair_auto'] is True
    assert request['source_repair_editable'] == []


def test_pdf_source_repair_blank_list_uses_failure_traceback(strict_app):
    make, capture = strict_app
    app = make()
    app.sidebar.checkbox(key='pdf_source_repair').check().run()
    app.sidebar.text_area(key='pdf_source_repair_editable').set_value('  \n\n').run()
    _start(app)
    assert capture.call_args.kwargs['source_repair_auto'] is True
    assert capture.call_args.kwargs['source_repair_editable'] == []


def test_pdf_source_repair_sends_only_explicit_file_paths(strict_app):
    make, capture = strict_app
    app = make()
    app.sidebar.checkbox(key='pdf_source_repair').check().run()
    app.sidebar.text_area(key='pdf_source_repair_editable').set_value(
        '  src/model.py \ntrain.py\n\nsrc/model.py\n').run()
    _start(app)
    request = capture.call_args.kwargs
    assert request['source_repair'] is True
    assert request['source_repair_editable'] == ['src/model.py', 'train.py']
    assert request['source_repair_auto'] is False
    assert request['strict_validation'] is False
    assert request['author_execution_only'] is True
    assert request['experiment_profile'] is None


@pytest.mark.parametrize('last_selected', ['source_repair', 'strict_validation'])
def test_pdf_execution_modes_are_mutually_exclusive(strict_app, last_selected):
    make, capture = strict_app
    app = make()
    first = 'strict_validation' if last_selected == 'source_repair' else 'source_repair'
    app.sidebar.checkbox(key=f'pdf_{first}').check().run()
    app.sidebar.checkbox(key=f'pdf_{last_selected}').check().run()
    assert app.sidebar.checkbox(key=f'pdf_{first}').value is False
    assert app.sidebar.checkbox(key=f'pdf_{last_selected}').value is True
    if last_selected == 'source_repair':
        app.sidebar.text_area(key='pdf_source_repair_editable').set_value('train.py').run()
        assert any('源码修复与严格核验互斥' in row.value for row in app.caption)
    _start(app)
    request = capture.call_args.kwargs
    assert request['source_repair'] is (last_selected == 'source_repair')
    assert request['strict_validation'] is (last_selected == 'strict_validation')
    assert request['source_repair_editable'] == (['train.py'] if last_selected == 'source_repair' else [])


def test_disabled_pdf_repair_does_not_reuse_old_editable_paths(strict_app):
    make, capture = strict_app
    app = make()
    app.sidebar.checkbox(key='pdf_source_repair').uncheck().run()
    app.session_state['pdf_source_repair_editable'] = 'old.py'
    app.run()
    _start(app)
    assert capture.call_args.kwargs['source_repair'] is False
    assert capture.call_args.kwargs['source_repair_editable'] == []


@pytest.mark.parametrize('route', ['title', 'preset', 'known_pdf', 'mock', 'corpus'])
def test_other_routes_ignore_stale_pdf_source_repair_permissions(strict_app, route):
    make, capture = strict_app
    options = {
        'title': {'mode': '论文标题'},
        'preset': {'mode': '官方仓库预设'},
        'known_pdf': {'title': TITLE},
        'mock': {'mock': True},
        'corpus': {'corpus': list_papers()[0]['id']},
    }[route]
    app = make(**options)
    app.session_state['pdf_source_repair'] = True
    app.session_state['pdf_source_repair_editable'] = 'old.py'
    app.run()
    assert not app.exception
    assert 'pdf_source_repair' not in [row.key for row in app.sidebar.checkbox]
    _start(app)
    request = capture.call_args.kwargs
    assert request['source_repair'] is False
    assert request['source_repair_editable'] == []
