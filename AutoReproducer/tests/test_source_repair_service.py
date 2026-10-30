"""Opt-in web source repair, using real owned Python subprocesses.

Fixtures are invented compatibility failures. Their successful execution does
not establish reproduction of any paper or equivalence of arbitrary patches.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from unittest.mock import Mock
import zipfile

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.execution_plan import build_plan
from src.method_adapters import digest, read_json
from src.repository_runner import RepositoryRunner
from src import source_repair as repair
from test_discovered_repository_service import (
    cpu_dependency_cache, setup_service, use_preinstalled_cache,
)

MODEL_SOURCE = "class Model:\n    def value(self):\n        return dict(answer=7).iteritems()\n"
TRAIN_SOURCE = ("from model import Model\nprint(list(Model().value())[0][1])\n"
                "print(list(dict(answer=7).iteritems())[0][1])\n")


class CompatibilityLLM:
    def __init__(self, propose=None):
        self.calls = []
        self.propose = propose or self.good

    @staticmethod
    def good(context, _round):
        return {"model.py": MODEL_SOURCE.replace(".iteritems()", ".items()"),
                "train.py": TRAIN_SOURCE.replace(".iteritems()", ".items()")}

    def chat(self, prompt, **kwargs):
        assert kwargs["task"] == "repository_repair"
        context = json.loads(prompt)
        self.calls.append(context)
        changes = self.propose(context, len(self.calls))
        return json.dumps({"reason": "Align the existing caller and callee API spelling.",
            "patches": [{"path": name, "base_sha256": context["sources"][name]["base_sha256"],
                         "content": content} for name, content in changes.items()]})


@pytest.fixture
def repair_case(tmp_path):
    workspace, run_dir = tmp_path / "original", tmp_path / "run"
    workspace.mkdir()
    run_dir.mkdir()
    sources = {"model.py": MODEL_SOURCE.encode(), "train.py": TRAIN_SOURCE.encode()}
    for name, source in sources.items():
        (workspace / name).write_bytes(source)
    # Downloaded frozen data is deliberately absent from the source snapshot.
    # Repair must copy it by its verified provenance, without generated outputs.
    (workspace / "input.json").write_bytes(b'{"values": [1, 2, 3]}\n')
    (workspace / "failed-output.txt").write_text("original failed output", encoding="utf-8")
    snapshot = {"url": "https://github.com/invented/compatibility", "revision": "a" * 40,
                "files": {name: digest(workspace / name) for name in sources}}
    data_source = {"kind": "https", "target": "input.json", "sha256": digest(workspace / "input.json"),
                   "bytes": (workspace / "input.json").stat().st_size}
    plan = {"scope": "author_code_execution", "repository": {key: snapshot[key] for key in ("url", "revision")},
            "steps": [{"id": "train", "kind": "train", "argv": ["python", "train.py"], "cwd": "."}],
            "capture": {"device": "cpu"}, "author_environment": {}}
    request = repair.normalize_source_repair_request({"pdf_path": "fixture.pdf", "source_repair": True,
                                                      "source_repair_editable": ["model.py", "train.py"]})
    frozen = build_plan(profile="original_author_command", spec_sha256="b" * 64,
                        workspace=workspace, repository=plan["repository"], steps=plan["steps"])
    data = {"source_repair": {"requested": True, "status": "pending", "max_rounds": 2,
                              "editable": request["source_repair_editable"]},
            "spec_sha256": "b" * 64, "dataset_provenance": {"sources": [data_source]},
            "env_config": {}, "execution_plan": frozen, "execution_source": "author_repository"}
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, "PYTHONIOENCODING": "utf-8"})
    executor._ensure_local_deps = Mock(return_value=None)
    runner = RepositoryRunner(executor=executor)
    initial = runner.run(workspace, frozen, {})
    assert not initial["success"] and "AttributeError" in initial["final"]["stderr"]

    def verify():
        for name, sha in snapshot["files"].items():
            assert digest(workspace / name) == sha, name
        assert digest(workspace / "input.json") == data_source["sha256"]

    def run(llm=None, **kwargs):
        return repair.repair_author_failure(runner, llm or CompatibilityLLM(), request, plan,
            snapshot, workspace, run_dir, data, initial, verify, kwargs.get("on_event", lambda _event: None))

    return {"workspace": workspace, "run_dir": run_dir, "snapshot": snapshot, "request": request,
            "plan": plan, "data": data, "runner": runner, "initial": initial, "run": run, "verify": verify}


def test_source_repair_copy_retains_automatically_recovered_inputs(repair_case):
    case = repair_case
    original = case['data']['dataset_provenance']['sources'].pop()
    case['data']['dataset_provenance']['recovered_files'] = [original]
    case['data']['self_repair'] = {'status': 'blocked', 'attempts': []}
    result = case['run']()
    assert result['success']
    assert case['data']['self_repair']['status'] == 'resolved'
    assert case['data']['self_repair']['attempts'][-1]['action'] == 'repair_compatibility_source'
    target = Path(case['data']['source_repair']['workspace']) / original['target']
    assert digest(target) == original['sha256']
    assert target.stat().st_size == original['bytes']
    case['verify']()


def test_real_two_file_repair_preserves_original_sources_commands_and_evidence(repair_case):
    case = repair_case
    original_plan = deepcopy(case["data"]["execution_plan"])
    llm = CompatibilityLLM()
    result = case["run"](llm)
    assert result["success"], result
    assert result["repair_status"] == "succeeded"
    assert result["final"]["stdout"].strip() == "7\n7"
    assert len(llm.calls) == 1
    case["verify"]()
    state = case["data"]["source_repair"]
    copied = Path(state["workspace"])
    assert copied != case["workspace"]
    assert (copied / "input.json").read_bytes() == (case["workspace"] / "input.json").read_bytes()
    assert not (copied / "failed-output.txt").exists()
    assert case["data"]["original_execution_plan"] == original_plan
    assert case["data"]["execution_plan"]["steps"][0]["argv"] == original_plan["steps"][0]["argv"]
    assert case["data"]["execution_plan"]["steps"][0]["cwd"] == original_plan["steps"][0]["cwd"]
    assert case["data"]["reproduction_scope"] == "patched_author_code_execution"
    assert case["data"]["execution_source"] == "patched_author_repository"
    assert {item["path"] for item in state["changed_files"]} == {"model.py", "train.py"}
    for item in state["changed_files"]:
        assert item["base_sha256"] == case["snapshot"]["files"][item["path"]]
        assert item["sha256"] == digest(copied / item["path"])
    manifest = read_json(state["manifest_path"])
    assert digest(state["manifest_path"]) == state["manifest_sha256"]
    assert manifest["original_source_files"] == case["snapshot"]["files"]
    assert manifest["frozen_input_files"]["input.json"]["sha256"] == digest(copied / "input.json")
    records = read_json(state["records_path"])
    diff = (Path(state["records_path"]).parent / "round_1" / "candidate.diff").read_text(encoding="utf-8")
    assert "a/model.py" in diff and "b/train.py" in diff
    assert records["initial_execution"]["success"] is False
    assert Path(records["initial_execution"]["final"]["stderr_path"]).is_file()
    assert Path(result["final"]["stdout_path"]).is_file()
    assert Path(state["report_path"]).is_file()
    validation = repair.verify_repaired_execution(case["plan"], case["data"], result)
    assert validation["source_modified"] is True
    assert validation["paper_metrics_verified"] is False
    assert validation["is_reproduced"] is None
    assert validation["training_observed"] is False


def test_failed_round_is_restored_before_second_model_call(repair_case):
    case = repair_case
    def changes(context, number):
        if number == 1:
            return {"model.py": MODEL_SOURCE.replace(".iteritems()", ".items()")}
        copy = case["run_dir"] / "repair_workspace"
        for name, sha in case["snapshot"]["files"].items():
            assert digest(copy / name) == sha
            assert hashlib.sha256(context["sources"][name]["content"].encode()).hexdigest() == sha
        return CompatibilityLLM.good(context, number)
    llm = CompatibilityLLM(changes)
    result = case["run"](llm)
    assert result["success"]
    assert [attempt["status"] for attempt in result["repair_attempts"]] == ["failed", "succeeded"]
    assert len(llm.calls) == 2
    case["verify"]()


def test_second_round_reuses_all_actually_observed_missing_apis(repair_case):
    case = repair_case
    original = TRAIN_SOURCE.replace('.iteritems()', '.iterkeys()')
    (case['workspace'] / 'train.py').write_bytes(original.encode())
    case['snapshot']['files']['train.py'] = digest(case['workspace'] / 'train.py')
    def propose(context, number):
        changes = {'model.py': MODEL_SOURCE.replace('.iteritems()', '.items()')}
        if number == 2:
            assert 'iterkeys' in context['diagnostic']
            changes['train.py'] = original.replace('.iterkeys()', '.items()')
        return changes
    result = case['run'](CompatibilityLLM(propose))
    assert result['success'], result
    assert [item['status'] for item in result['repair_attempts']] == ['failed', 'succeeded']


@pytest.mark.parametrize('parameter', ['random_state', 'learning_rate', 'dropout', 'n_estimators',
                                      'max_iter', 'num_layers', 'hidden_channels', 'momentum',
                                      'n_components', 'num_leaves', 'perplexity', 'tol', 'C', 'bagging_fraction'])
def test_keyword_compatibility_cannot_reassign_frozen_hyperparameter(repair_case, parameter):
    case = repair_case
    original = ("def configure(*, compatibility=7):\n    return compatibility\n"
                f"print(configure({parameter}=7))\n")
    replacement = original.replace(f'configure({parameter}=7)', 'configure(compatibility=7)')
    path = case['workspace'] / 'train.py'
    path.write_text(original, encoding='utf-8')
    case['snapshot']['files']['train.py'] = digest(path)
    initial = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    assert not initial['success'] and 'unexpected keyword argument' in initial['final']['stderr']
    case['initial'].clear()
    case['initial'].update(initial)
    llm = CompatibilityLLM(lambda *_: {'train.py': replacement})
    result = case['run'](llm)
    assert not result['success'], 'A successful exit cannot authorize changing a frozen hyperparameter.'
    assert len(llm.calls) == 2
    assert all(item['status'] == 'rejected' and 'execution' not in item
               for item in result['repair_attempts'])
    case['verify']()


@pytest.mark.parametrize('tamper', ['interpreter', 'runtime', 'requirements'])
def test_candidate_cannot_commit_with_different_prepared_runtime(repair_case, monkeypatch, tamper):
    case = repair_case
    original_run = case['runner'].run
    def changed_runtime(*args, **kwargs):
        result = original_run(*args, **kwargs)
        assert result['success']
        if tamper == 'interpreter':
            result['steps'][0]['argv'][0] = 'other-python.exe'
        elif tamper == 'runtime':
            result['environment']['runtime'] = 'other-runtime'
        else:
            result['environment']['requirements_txt'] = 'other-package==1.0\n'
        return result
    monkeypatch.setattr(case['runner'], 'run', changed_runtime)
    result = case['run']()
    assert not result['success']
    assert all(attempt['status'] == 'rejected' for attempt in result['repair_attempts'])
    state = case['data']['source_repair']
    assert not state.get('manifest_path')
    assert all(digest(Path(state['workspace']) / name) == sha for name, sha in case['snapshot']['files'].items())
    case['verify']()


@pytest.mark.parametrize('tamper', ['missing_environment', 'inconsistent_interpreter'])
def test_source_repair_requires_actual_prepared_runtime_evidence(repair_case, tamper):
    case = repair_case
    if tamper == 'missing_environment':
        case['initial'].pop('environment')
    else:
        case['initial']['steps'][0]['argv'][0] = 'other-python.exe'
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result is case['initial'] and not llm.calls
    state = case['data']['source_repair']
    assert state['status'] == 'not_applicable' and '解释器' in state['reason']
    assert not (case['run_dir'] / 'repair_workspace').exists()


def test_absolute_author_paths_are_relocated_into_the_repair_copy(repair_case):
    case = repair_case
    original = case['workspace']
    case['data']['execution_plan']['steps'][0]['env'] = {
        'PYTHONPATH': str(original), 'DATA_ROOT': str(original / 'input.json')}
    result = case['run']()
    assert result['success']
    copied = case['run_dir'] / 'repair_workspace'
    environment = case['data']['execution_plan']['steps'][0]['env']
    assert environment == {'PYTHONPATH': str(copied), 'DATA_ROOT': str(copied / 'input.json')}
    assert {item['name'] for item in case['data']['source_repair']['environment_relocations']} == {'PYTHONPATH', 'DATA_ROOT'}
    case['verify']()


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_copy_preparation_failure_never_leaves_running_state(repair_case, monkeypatch, failure):
    case = repair_case
    original_read = Path.read_bytes
    def read(path):
        if path == case['workspace'] / 'model.py':
            raise failure('fixture copying failed')
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    # Skip source verification here to inject precisely at copy preparation.
    with pytest.raises(failure, match='copying failed'):
        repair.repair_author_failure(case['runner'], CompatibilityLLM(), case['request'], case['plan'],
            case['snapshot'], case['workspace'], case['run_dir'], case['data'], case['initial'], lambda: None, lambda e: None)
    state = read_json(case['run_dir'] / 'source_repair.json')
    assert state['status'] == ('failed' if failure is OSError else 'interrupted')


def test_unfrozen_dynamic_inputs_do_not_trigger_a_fresh_download_repair(repair_case, monkeypatch):
    case = repair_case
    case['data']['dataset_provenance']['sources'].append({'kind': 'author_runtime', 'verified': False})
    attempt = Mock(side_effect=AssertionError('unfrozen runtime input'))
    monkeypatch.setattr(case['runner'], 'run_with_repairs', attempt)
    result = case['run']()
    assert not result['success']
    assert case['data']['source_repair']['status'] == 'not_applicable'
    assert not (case['run_dir'] / 'repair_workspace').exists()
    attempt.assert_not_called()


def test_two_failed_rounds_leave_clean_copy_and_original_failure_evidence(repair_case):
    case = repair_case
    llm = CompatibilityLLM(lambda *_: {"model.py": MODEL_SOURCE.replace(".iteritems()", ".items()")})
    result = case["run"](llm)
    assert not result["success"]
    assert result["repair_status"] == "failed"
    assert len(llm.calls) == 2
    state = case["data"]["source_repair"]
    assert not state.get("manifest_path")
    assert case["data"]["execution_source"] == "patched_author_repository"
    for name, sha in case["snapshot"]["files"].items():
        assert digest(Path(state["workspace"]) / name) == sha
    assert "Status: failed" in Path(state["report_path"]).read_text(encoding="utf-8")
    assert [attempt["status"] for attempt in state["attempts"]] == ["failed", "repeated_candidate"]
    assert sum('execution' in attempt for attempt in state['attempts']) == 1
    case["verify"]()


def test_unconfirmed_candidate_cleanup_preserves_copy_without_claiming_restoration(repair_case, monkeypatch):
    case = repair_case
    run = case['runner'].run

    def unconfirmed(*args, **kwargs):
        result = run(*args, **kwargs)
        # The child actually exited in this regression; simulate an uncertain
        # cleanup receipt without leaving a process behind in the test suite.
        result.update(success=False, cleanup_unconfirmed=True, process_cleanup_confirmed=False)
        return result

    monkeypatch.setattr(case['runner'], 'run', unconfirmed)
    llm = CompatibilityLLM()
    result = case['run'](llm)
    assert result['success'] is False and len(llm.calls) == 1
    state = case['data']['source_repair']
    assert state['status'] == 'cleanup_unconfirmed'
    assert state['workspace_retained'] and state['process_cleanup_confirmed'] is False
    assert state['effective_source_files'] == {} and not state.get('manifest_path')
    assert '.items()' in (Path(state['workspace']) / 'model.py').read_text(encoding='utf-8')
    assert state['attempts'][0]['rollback_deferred']
    assert read_json(case['run_dir'] / 'source_repair.json')['status'] == 'cleanup_unconfirmed'
    case['verify']()


@pytest.mark.parametrize("replacement", [
    TRAIN_SOURCE.replace(".iteritems()", ".items()").replace("[0][1]", "[0][1] + 1"),
    TRAIN_SOURCE.replace(".iteritems()", ".items()") + "print('fake metric')\n",
    "from model import Model\nif False:\n    print(list(Model().value())[0][1])\n",
])
def test_protocol_changes_rejected_before_any_candidate_process(repair_case, replacement, monkeypatch):
    case = repair_case
    original_run = case["runner"].run
    candidate = Mock(wraps=original_run)
    monkeypatch.setattr(case["runner"], "run", candidate)
    llm = CompatibilityLLM(lambda context, number: {**CompatibilityLLM.good(context, number), "train.py": replacement})
    result = case["run"](llm)
    assert not result["success"]
    candidate.assert_not_called()
    assert len(llm.calls) == 2
    assert all(attempt["status"] == "rejected" and "execution" not in attempt
               for attempt in result["repair_attempts"])
    for number in (1, 2):
        assert (Path(result["repair_records"]).parent / f"round_{number}" / "candidate.diff").is_file()
    for name, sha in case["snapshot"]["files"].items():
        assert digest(case["run_dir"] / "repair_workspace" / name) == sha
    case["verify"]()


def test_cancellation_restores_candidate_and_retains_interrupted_record(repair_case, monkeypatch):
    case = repair_case
    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt("fixture user cancelled")
    monkeypatch.setattr(case["runner"], "run", cancel)
    with pytest.raises(KeyboardInterrupt, match="fixture user cancelled"):
        case["run"]()
    state = read_json(case["run_dir"] / "source_repair.json")
    assert state["status"] == "interrupted"
    assert state["attempts"][0]["status"] == "interrupted"
    assert Path(state["report_path"]).is_file()
    assert not state.get("manifest_path")
    for name, sha in case["snapshot"]["files"].items():
        assert digest(Path(state["workspace"]) / name) == sha
    case["verify"]()


@pytest.mark.parametrize("limit", ["MAX_REPAIR_WORKSPACE_FILES", "MAX_REPAIR_WORKSPACE_BYTES"])
def test_resource_preflight_avoids_copy_and_model(repair_case, monkeypatch, limit):
    import src.repository_repairer as repository_repairer
    case = repair_case
    monkeypatch.setattr(repository_repairer, limit, 1)
    llm = CompatibilityLLM()
    result = case["run"](llm)
    assert result is case["initial"]
    assert not llm.calls
    assert case["data"]["source_repair"]["status"] == "resource_limit"
    assert not (case["run_dir"] / "repair_workspace").exists()
    case["verify"]()


@pytest.mark.parametrize("initial", [
    {"success": True},
    {"success": False, "reason": "dependency installation failed"},
    {"success": False, "cancelled": True, "executed": True, "final": {"exit_code": 1}},
    {"success": False, "executed": True, "final": {"exit_code": 1, "timed_out": True}},
])
def test_non_source_failures_never_call_repair_or_copy(repair_case, initial):
    case = repair_case
    case["initial"].clear()
    case["initial"].update(initial)
    llm = CompatibilityLLM()
    result = case["run"](llm)
    assert result is case["initial"]
    assert not llm.calls
    assert not (case["run_dir"] / "repair_workspace").exists()


@pytest.mark.parametrize("tamper", ["manifest", "source", "frozen_data", "incomplete_execution",
                                   "interpreter", "environment_executable", "runtime", "requirements"])
def test_final_verification_rejects_tampered_repair_evidence(repair_case, tamper):
    case = repair_case
    result = case["run"]()
    assert result["success"]
    state = case["data"]["source_repair"]
    if tamper == "manifest":
        with Path(state["manifest_path"]).open("ab") as stream:
            stream.write(b"\n")
    elif tamper == "source":
        (Path(state["workspace"]) / "model.py").write_text("# changed after approval\n", encoding="utf-8")
    elif tamper == "frozen_data":
        (Path(state["workspace"]) / "input.json").write_text("{}", encoding="utf-8")
    elif tamper == "interpreter":
        result['steps'][0]['argv'][0] = str(Path(result['steps'][0]['argv'][0]).with_name('other-python.exe'))
    elif tamper == "environment_executable":
        result['environment']['executable'] = 'other-python.exe'
    elif tamper == "runtime":
        result['environment']['runtime'] = 'other-runtime'
    elif tamper == "requirements":
        result['environment']['requirements_txt'] = 'other-package==1.0\n'
    else:
        result["steps"][0]["executed"] = False
    with pytest.raises(ValueError):
        repair.verify_repaired_execution(case["plan"], case["data"], result)


@pytest.mark.parametrize("values", [[], ["*.py"], ["../train.py"], ["data/split.py"],
                                   ["train.py", "TRAIN.py"], ["train.py "]])
def test_explicit_source_whitelist_rejects_ambiguous_and_protected_targets(values):
    with pytest.raises(ValueError):
        repair.normalize_source_repair_request({"pdf_path": "fixture.pdf", "source_repair": True,
                                                "source_repair_editable": values})


def test_disabled_request_drops_stale_source_whitelist():
    request = repair.normalize_source_repair_request({"source_repair": False, "source_repair_editable": ["*.py"]})
    assert request["source_repair"] is False
    assert request["source_repair_editable"] == []


@pytest.mark.parametrize("flags", [
    {"mock_mode": True}, {"code": "print(1)"}, {"corpus_paper": "fixture"},
    {"experiment_profile": "fixture"}, {"pdf_path": ""}, {"source_repair": "yes"},
])
def test_non_pdf_routes_and_non_boolean_opt_in_cannot_request_source_changes(flags):
    with pytest.raises(ValueError):
        repair.normalize_source_repair_request({"pdf_path": "fixture.pdf", "source_repair": True,
                                                "source_repair_editable": ["train.py"], **flags})


def test_repaired_plan_retains_derived_author_environment(repair_case):
    case = repair_case
    environment = {"DGLBACKEND": "pytorch", "CUDA_VISIBLE_DEVICES": ""}
    case["data"]["execution_plan"]["steps"][0]["env"] = environment
    result = case["run"]()
    assert result["success"]
    assert case["data"]["execution_plan"]["steps"][0]["env"] == environment
    assert case["data"]["original_execution_plan"]["steps"][0]["env"] == environment


def test_strict_and_source_repair_are_mutually_exclusive():
    with pytest.raises(ValueError, match="严格"):
        repair.normalize_source_repair_request({"pdf_path": "fixture.pdf", "strict_validation": True,
            "source_repair": True, "source_repair_editable": ["train.py"]})


def test_prepare_only_never_invokes_source_repair(tmp_path, monkeypatch):
    _, runner, request, _ = setup_service(tmp_path, monkeypatch)
    request.update(prepare_only=True, source_repair=True, source_repair_editable=["train.py", "author_model.py"])
    monkeypatch.setattr(runner.runner, "run_with_repairs", Mock(side_effect=AssertionError("prepare-only repair")))
    result = runner.run(request)
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "prepared"
    assert result["data"]["validation"]["is_reproduced"] is None
    runner.runner.run_with_repairs.assert_not_called()


def legacy_api_fixture(fixture, llm):
    source_root = fixture["workspace"]
    model = (source_root / "author_model.py").read_text(encoding="utf-8")
    model += "    def compatibility_prepare(self):\n        return dict(answer=7).iteritems()\n"
    train = (source_root / "train.py").read_text(encoding="utf-8").replace(
        "model = SignalModel()\n", "model = SignalModel()\nmodel.compatibility_prepare()\ndict(answer=7).iteritems()\n")
    for name, source in (("author_model.py", model), ("train.py", train)):
        (source_root / name).write_text(source, encoding="utf-8")
        fixture["snapshot"]["files"][name] = digest(source_root / name)
        for citation in llm.proposal["citations"]:
            if citation.get("path") == name:
                citation["quote"] = source


@pytest.mark.parametrize('via_backend', [False, True])
def test_real_service_two_file_repair_keeps_pinned_export_and_labels_report(
        tmp_path, monkeypatch, cpu_dependency_cache, via_backend):
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    legacy_api_fixture(fixture, llm)
    source_root = fixture["workspace"]
    request.update(source_repair=True, source_repair_editable=["train.py", "author_model.py"])
    original_chat = llm.chat
    def chat(prompt, **kwargs):
        if kwargs["task"] != "repository_repair":
            return original_chat(prompt, **kwargs)
        context = json.loads(prompt)
        llm.calls.append((prompt, kwargs))
        changes = {name: context["sources"][name]["content"].replace(".iteritems()", ".items()")
                   for name in ("author_model.py", "train.py")}
        return json.dumps({"reason": "Align existing no-op initialization API spellings.",
            "patches": [{"path": name, "base_sha256": context["sources"][name]["base_sha256"],
                         "content": content} for name, content in changes.items()]})
    monkeypatch.setattr(llm, "chat", chat)
    if via_backend:
        from types import SimpleNamespace
        from frontend import backend_pipeline as backend
        monkeypatch.setattr(backend, 'get_data_root', lambda: tmp_path / 'web')
        monkeypatch.setattr(backend, 'AuditLogger', lambda: runner.logger)
        monkeypatch.setattr(backend, 'LLMClient', lambda **kwargs: llm)
        def run_web(web_request, **options):
            assert web_request['source_repair'] is True
            assert web_request['source_repair_editable'] == request['source_repair_editable']
            return runner.run({**web_request, 'resources': request['resources']}, **options)
        monkeypatch.setattr(backend, '_create_orchestrator', lambda **kwargs: SimpleNamespace(run=run_web))
        result = backend.run_pipeline_core(str(tmp_path / 'progress.jsonl'), pdf_path=request['pdf_path'],
            source_repair=True, source_repair_editable=request['source_repair_editable'],
            offline=True, _managed_runtime=True)
        assert backend.ProgressStore.read_snapshot(str(tmp_path / 'progress.jsonl'))['done'] is True
    else:
        result = runner.run(request)
    assert result["state"] == "COMPLETED", result["error"]
    data = result["data"]
    assert data["source_repair"]["status"] == "succeeded"
    assert data["validation"]["source_modified"] is True
    assert data["validation"]["is_reproduced"] is None
    assert data["validation"]["paper_metrics_verified"] is False
    assert data["validation"]["training_observed"] is True
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert data["validation"]["author_claim_comparison"]["basis"] == "patched_author_output"
    assert data["validation"]["author_claim_comparison"]["source_modified"] is True
    assert data["execution_source"] == "patched_author_repository"
    assert data["execution"]["final"]["stdout"].find("Test Accuracy:") >= 0
    workspace = Path(data["run_dir"]) / "repo"
    for name, sha in fixture["snapshot"]["files"].items():
        assert digest(workspace / name) == sha
        assert digest(source_root / name) == sha
    state = data["source_repair"]
    assert {change["path"] for change in state["changed_files"]} == {"train.py", "author_model.py"}
    assert Path(state["manifest_path"]).is_file()
    assert data['validation']['author_execution'][0]['step']['argv'] == fixture['proposal']['steps'][0]['argv']
    assert data['validation']['author_execution'][0]['step']['cwd'] == fixture['proposal']['steps'][0]['cwd']
    assert data['execution_plan'] == state['observation']['execution_plan']
    report = (Path(data["run_dir"]) / "report.md").read_text(encoding="utf-8")
    assert report == data["report"]
    assert "source_repair" in report or "源码修复" in report
    assert "author_model.py" in report.replace("\\_", "_") and "train.py" in report
    assert state["manifest_sha256"] in report
    assert sum(options["task"] == "repository_repair" for _, options in llm.calls) == 1
    cleanup = read_json(Path(data["run_dir"]) / "environment_cleanup.json")
    assert cleanup["status"] == "released" and cleanup["remaining"] == [] and cleanup["errors"] == []
    artifact = state["evidence_archive"]
    raw = Path(artifact["path"]).read_bytes()
    assert len(raw) == artifact["bytes"]
    assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        assert "source_repair_manifest.json" in names
        assert "repairs/round_1/candidate.diff" in names
        assert "repairs/round_1/result.json" in names
        assert "logs/original/train.stderr.log" in names
        assert "logs/round_1/train.stdout.log" in names
        assert b"iteritems" in archive.read("logs/original/train.stderr.log")
        assert b"Test Accuracy:" in archive.read("logs/round_1/train.stdout.log")
        manifest = json.loads(archive.read("evidence_manifest.json"))
        assert set(manifest) == set(names) - {"evidence_manifest.json"}
        for name, record in manifest.items():
            content = archive.read(name)
            assert len(content) == record["bytes"]
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
    from src.source_repair_evidence import report_evidence_bytes
    companion = report_evidence_bytes(report, Path(data["run_dir"]) / "report.md")
    assert len(companion) == 1
    assert next(iter(companion.values())) == raw


def test_real_service_default_failure_never_requests_source_changes(tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch)
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    legacy_api_fixture(fixture, llm)
    request["source_repair_editable"] = ["train.py", "author_model.py"]  # stale UI state
    monkeypatch.setattr(runner.runner, "run_with_repairs", Mock(side_effect=AssertionError("not opted in")))
    result = runner.run(request)
    assert result["state"] == "ERROR"
    assert "iteritems" in result["error"]
    runner.runner.run_with_repairs.assert_not_called()
    assert result["data"]["source_repair"]["requested"] is False
    assert result["data"]["source_repair"]["status"] == "not_requested"
    assert result["data"]["source_repair"]["editable"] == []
    assert result["data"]["validation"]["is_reproduced"] is None
    assert not (Path(result["data"]["run_dir"]) / "repair_workspace").exists()
    assert all(options["task"] != "repository_repair" for _, options in llm.calls)
    for name, sha in fixture["snapshot"]["files"].items():
        assert digest(Path(result["data"]["run_dir"]) / "repo" / name) == sha
