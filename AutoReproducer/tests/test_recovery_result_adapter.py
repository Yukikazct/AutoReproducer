"""Result evidence is scored from real completed runs, never model success text."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from test_recovery_adapter import frozen_products
from src.benchmark.recovery_result_evidence import (validate_result_evidence, run_result_evidence,
    evaluate_result_evidence, _observe, _digest, _unconfirmed)
from src.runtime_preparation import run_owned_process


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {"metric": "accuracy", "value": .9, "unit": "fraction", "method": "author-model",
             "dataset": "toy", "split": "test", "aggregation": "mean", "runs": 3}


def prepare(tmp_path, *, actual=None, reference=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    actual = deepcopy(REFERENCE if actual is None else actual)
    content = json.dumps(actual, sort_keys=True)
    script = "from pathlib import Path\nPath('metrics.json').write_text(" + repr(content) + ", encoding='utf-8')\nprint('author completed')\n"
    (workspace / "train.py").write_text(script, "utf-8")
    raw = (workspace / "train.py").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    repository = {"url": "https://github.com/example/result-development", "revision": "d" * 40}
    evidence = {"adapter_version": 1, "plan": {"repository": repository,
        "steps": [{"id": "train", "kind": "train", "argv": ["python", "train.py"], "cwd": ".", "timeout_s": None}]},
        "packet": {"file_inventory": {"train.py": {"sha256": sha, "text_available": True}}},
        "environment": {}, "editable": [], "result_contract": {"version": 1, "artifact": "metrics.json",
            "format": "metric_summary_json", "reference": deepcopy(REFERENCE if reference is None else reference),
            "relative_tolerance": .05, "absolute_tolerance": 0}}
    case = {"id": "result-development", "repository_url": repository["url"], "repository_revision": repository["revision"],
        "category": "result_evidence", "origin": "injected", "split": "dev", "source_files": {"train.py": sha},
        "initial_evidence": evidence}
    validate_result_evidence(case, workspace)
    return case, workspace, content


WORKER = r'''
import hashlib, io, json, os, runpy, sys
from pathlib import Path
from contextlib import redirect_stdout, nullcontext
payload = json.loads(sys.stdin.read())
product, workspace, environment = (Path(payload[key]) for key in ('product','workspace','environment'))
driver = runpy.run_path(payload['helper'])
sys.path.insert(0, str(product))
from src.repository_runner import RepositoryRunner
from src.runtime_preparation import run_owned_process
from src.execution_plan import build_plan
evidence = payload['evidence']
with redirect_stdout(io.StringIO()):
    venv = environment / 'author-venv'
    prepared = run_owned_process([sys.executable, '-I', '-m', 'venv', '--without-pip', str(venv)], timeout_s=60)
    assert prepared.returncode == 0
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = {'python_executable':str(python), 'auto_prepare':False, 'dependency_health_check':False}
    original = evidence['plan']
    plan = build_plan(profile='recovery_benchmark',spec_sha256=driver['_digest'](original),workspace=workspace,
                      repository=original['repository'],dataset={'sources':[]},steps=original['steps'])
    execution = RepositoryRunner().run(workspace,plan,env)
    assert execution.get('success') is True, execution
    original_execution = driver['_digest'](execution)
    if payload.get('execution_override'):
        execution.update(payload['execution_override'])
    class Scripted:
        api_key = ''
        last_usage = {}
        def __init__(self): self.calls=[]; self.diagnosis_context=None
        def chat(self,prompt,**kwargs):
            task=kwargs.get('task'); self.calls.append(task)
            context=json.loads(prompt)
            if task=='result_diagnosis':
                if payload.get('model_timeout'): raise TimeoutError('development timeout')
                if payload.get('model_unclean'):
                    failure=RuntimeError('development cleanup pending'); failure.process_cleanup_confirmed=False
                    raise failure
                self.diagnosis_context=context
                return json.dumps({'hypotheses':[]})
            if task=='recovery_investigation':
                if payload.get('always_finish'):
                    return json.dumps({'action':'finish','reason':'Claim success without new observations.',
                                       'hypotheses':[],'evidence_ids':['validation']})
                probes=context.get('probe_results') or []
                if not probes:
                    return json.dumps({'action':'probe','reason':'Read original output metadata before deciding.',
                         'hypotheses':[],'evidence_ids':['validation'],'tool':'artifact_contract',
                         'arguments':{'paths':['metrics.json']}})
                observations=[item for probe in probes for item in probe.get('evidence',[])]
                if not observations:
                    return json.dumps({'action':'needs_human','reason':'No actual artifact observation was available.',
                                       'hypotheses':[],'evidence_ids':['validation']})
                return json.dumps({'action':'finish','reason':'Observed artifact metadata fixes the evidence boundary.',
                     'hypotheses':[],'evidence_ids':[observations[-1]['id']]})
            if task in {'recovery_self_check','recovery_counter_review'}:
                action='reject' if payload.get('reject_review') else 'accept'
                return json.dumps({'action':action,'reason':'Check original protocol and actual observations.',
                                   'evidence_ids':context['candidate_decision']['evidence_ids']})
            raise ValueError('Unexpected task '+str(task))
    client=Scripted(); data={}; run_dir=environment/'investigation'
    session=None
    if payload['arm']!='A':
        from src.intelligent_recovery import RecoverySession
        session=RecoverySession(client,mode='single' if payload['arm']=='B' else 'collaborative',budget=payload.get('budget'))
        session.attach(data,run_dir)
    tools=None
    if session and payload.get('tool_mode'):
        from src.investigation_tools import InvestigationTools
        tools=InvestigationTools(workspace,evidence['packet'],env,run_dir)
        original_tool=tools.execute
        def execute(name,arguments):
            mode=payload['tool_mode']
            if mode=='mutate_artifact':
                (workspace/'metrics.json').write_text('{}','utf-8')
            elif mode=='mutate_protocol':
                evidence['result_contract']['relative_tolerance']=100
            elif mode=='unconfirmed':
                return {'tool':name,'status':'stopped','evidence':[],'cleanup_confirmed':False,'summary':'cleanup pending'}
            elif mode=='timeout':
                return {'tool':name,'status':'stopped','evidence':[],'cleanup_confirmed':True,'timed_out':True,'summary':'timed out'}
            elif mode=='rejected':
                return {'tool':name,'status':'rejected','evidence':[],'cleanup_confirmed':True,'summary':'unavailable'}
            return original_tool(name,arguments)
        tools.execute=execute
    def verify_inputs():
        for name,expected in payload['source_files'].items():
            assert hashlib.sha256((workspace/name).read_bytes()).hexdigest()==expected
    with session.scope() if session else nullcontext():
        result=driver['run_result_evidence'](evidence=evidence,workspace=workspace,environment=env,
            execution=execution,llm=client,session=session,run_dir=run_dir,data=data,tools=tools,
            verify_inputs=verify_inputs,cancel_check=(lambda:True) if payload.get('cancel') else None)
    modules={}
    for name,module in list(sys.modules.items()):
        if name=='src' or name.startswith('src.'):
            file=getattr(module,'__file__',None)
            if file:
                path=Path(file).resolve(); assert path.is_relative_to(product.resolve()), (name,str(path))
                modules[name]=path.relative_to(product).as_posix()
    answer={'result_evidence':result,'calls':client.calls,'modules':modules,
            'diagnosis_context':client.diagnosis_context,
            'execution_unchanged':original_execution==driver['_digest'](execution),
            'session':session.state if session else None}
print(json.dumps(answer,ensure_ascii=False,allow_nan=False))
'''


def execute(tmp_path, frozen_products, *, arm="C", actual=None, reference=None, **options):
    case, workspace, content = prepare(tmp_path, actual=actual, reference=reference)
    if "initial_validation" in options:
        case["initial_evidence"]["initial_validation"] = options.pop("initial_validation")
    environment = tmp_path / "environment"
    environment.mkdir()
    script = tmp_path / "trusted_worker.py"
    script.write_text(WORKER, "utf-8")
    payload = {"product": str(frozen_products[0][arm]), "workspace": str(workspace), "environment": str(environment),
        "helper": str(ROOT / "src/benchmark/recovery_result_evidence.py"), "arm": arm,
        "evidence": case["initial_evidence"], "source_files": case["source_files"], **options}
    observed = run_owned_process([sys.executable, "-I", str(script)], input=json.dumps(payload),
        cwd=str(tmp_path), timeout_s=90, max_output_bytes=1024 * 1024)
    assert observed.returncode == 0, observed.stderr
    result = json.loads(observed.stdout)
    return case, workspace, content, result


def oracle(status="invalid_evidence", codes=None, **kwargs):
    return {"version": 1, "expected_status": status, "required_issue_codes": ["unit_mismatch"] if codes is None else codes, **kwargs}


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_completed_author_execution_uses_actual_frozen_diagnosis_and_observed_units(tmp_path, frozen_products, arm):
    actual = {**REFERENCE, "unit": "percent", "value": 90}
    case, workspace, content, result = execute(tmp_path, frozen_products, arm=arm, actual=actual)
    receipt = result["result_evidence"]
    assert receipt["initial_success_observed"] is True, receipt
    assert receipt["completed"] is (arm != "A"), receipt
    assert receipt["assessment"]["status"] == ("unresolved" if arm == "A" else "invalid_evidence")
    assert receipt["assessment"]["issue_codes"] == ([] if arm == "A" else ["unit_mismatch"])
    assert receipt["assessment"]["unit_conversion_applied"] is False
    assert "src.result_diagnosis" in result["modules"]
    assert ("src.intelligent_recovery" in result["modules"]) is (arm != "A")
    assert result["calls"][0] == "result_diagnosis"
    assert (workspace / "metrics.json").read_text("utf-8") == content
    assert result["execution_unchanged"] is True
    if arm != "A":
        assert result["session"]["usage"]["probes"] == 1
        assert result["session"]["usage"]["model_requests"] == len(result["calls"]) == 4
        assert result["session"]["tasks"][0]["decisions"][0]["action"] == "probe"
        assert result["session"]["tasks"][0]["decisions"][1]["action"] == "finish"
        assert receipt["decision_evidence_ids"] == receipt["observed_artifact_evidence_ids"]
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=receipt, workspace=workspace, oracle=oracle())
    assert scored["oracle_passed"] is (arm != "A"), scored
    assert scored["protocol_preserved"] is True
    assert scored["numerical_equivalence"] == "unknown"
    if arm == "A":
        initial = [row for row in result["diagnosis_context"]["evidence"] if row["id"] == "comparison"]
        assert initial[0]["value"] == {"comparisons": []}
        assert receipt["assessment"]["actual"] == {}


@pytest.mark.parametrize("field,value,code", [("split", "validation", "split_mismatch"),
    ("dataset", "other", "dataset_mismatch"), ("aggregation", "best", "aggregation_mismatch"),
    ("runs", 1, "runs_mismatch"), ("method", "other", "method_mismatch")])
def test_private_oracle_recomputes_actual_correspondence_and_statistics(tmp_path, field, value, code):
    case, workspace, _ = prepare(tmp_path)
    (workspace / "metrics.json").write_text(json.dumps({**REFERENCE, field: value}), "utf-8")
    assessment = _observe(case["initial_evidence"]["result_contract"], workspace)
    assert assessment["status"] == "invalid_evidence"
    assert assessment["issue_codes"] == [code]


def test_fake_finish_without_check_is_not_independent_success(tmp_path, frozen_products):
    case, workspace, _, result = execute(tmp_path, frozen_products, actual={**REFERENCE, "unit": "percent"}, always_finish=True)
    assert result["result_evidence"]["completed"] is False
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle())
    assert scored["oracle_passed"] is False


@pytest.mark.parametrize("mode", ["mutate_artifact", "mutate_protocol"])
def test_investigation_cannot_rewrite_artifact_or_tolerance(tmp_path, frozen_products, mode):
    case, workspace, _, result = execute(tmp_path, frozen_products, tool_mode=mode)
    assert result["result_evidence"]["completed"] is False
    assert result["result_evidence"]["protocol_preserved"] is False
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle("reference_met", []))
    assert scored["oracle_passed"] is False


@pytest.mark.parametrize("options", [{"cancel": True}, {"tool_mode": "unconfirmed"},
    {"tool_mode": "timeout"}, {"execution_override": {"process_cleanup": {"status": "unconfirmed"}}}])
def test_cancel_timeout_and_cleanup_block_further_result_success(tmp_path, frozen_products, options):
    case, workspace, _, result = execute(tmp_path, frozen_products, **options)
    receipt = result["result_evidence"]
    assert receipt["completed"] is False
    assert receipt["cancelled"] or receipt["timed_out"] or not receipt["cleanup_confirmed"]
    assert "recovery_self_check" not in result["calls"] and "recovery_counter_review" not in result["calls"]
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle("reference_met", []))
    assert scored["oracle_passed"] is False


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
@pytest.mark.parametrize('wrapper', ['environment', 'attempts', 'stages'])
def test_nested_execution_cleanup_stops_before_any_result_model_or_tool_call(tmp_path, frozen_products, arm, wrapper):
    marker = {'cleanup_receipt': {'confirmed': False, 'status': 'unconfirmed'}}
    override = {wrapper: marker if wrapper == 'environment' else [marker]}
    case, workspace, content, result = execute(tmp_path, frozen_products, arm=arm, execution_override=override)
    receipt = result['result_evidence']
    assert receipt['completed'] is False
    assert receipt['cleanup_confirmed'] is False
    assert receipt['investigation_status'] == 'stopped'
    assert result['calls'] == []
    assert (workspace / 'metrics.json').read_text('utf-8') == content
    if arm != 'A':
        assert result['session']['cleanup_confirmed'] is False
        assert result['session']['status'] == 'stopped'
        assert result['session']['usage']['model_requests'] == 0
        assert result['session']['usage']['probes'] == 0
    scored = evaluate_result_evidence(evidence=case['initial_evidence'], receipt=result,
        workspace=workspace, oracle=oracle('reference_met', []))
    assert scored['oracle_passed'] is False


def test_result_cleanup_recognizes_shared_receipts_and_cycles_without_losing_receipt_context():
    shared = {'status': 'failed'}
    record = {'environment': shared, 'cleanup_receipt': shared}
    record['stages'] = [record]
    record['attempts'] = [record]
    assert _unconfirmed(record) is True
    clean = {}
    clean['attempts'] = [clean]
    assert _unconfirmed(clean) is False


def test_missing_reference_finishes_unknown_only_with_exact_private_preregistration(tmp_path, frozen_products):
    reference = {name: value for name, value in REFERENCE.items() if name not in {"value", "unit"}}
    case, workspace, _, result = execute(tmp_path, frozen_products, reference=reference)
    receipt = result["result_evidence"]
    assert receipt["assessment"]["status"] == "insufficient_reference"
    assert receipt["assessment"]["unknown_fields"] == ["reference.unit", "reference.value"]
    expected = oracle("insufficient_reference", [], allow_unknown=True,
                      expected_unknown_fields=["reference.unit", "reference.value"])
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result, workspace=workspace, oracle=expected)
    assert scored["oracle_passed"] is True
    expected.pop("allow_unknown")
    assert evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=expected)["oracle_passed"] is False


@pytest.mark.parametrize("arm", ["A", "C"])
@pytest.mark.parametrize("field", ["model_timeout", "model_unclean"])
def test_old_diagnosis_catching_model_error_cannot_hide_stop_conditions(tmp_path, frozen_products, arm, field):
    _, _, _, result = execute(tmp_path, frozen_products, arm=arm, **{field: True})
    receipt = result["result_evidence"]
    assert receipt["completed"] is False
    assert result["calls"] == ["result_diagnosis"]
    if field == "model_timeout": assert receipt["timed_out"] is True
    else: assert receipt["cleanup_confirmed"] is False


def test_shared_budget_includes_baseline_diagnosis_and_investigation_requests(tmp_path, frozen_products):
    _, _, _, result = execute(tmp_path, frozen_products, actual={**REFERENCE, "unit": "percent"}, budget={"max_model_requests": 2})
    assert result["result_evidence"]["completed"] is False
    assert len(result["calls"]) == result["session"]["usage"]["model_requests"] == 2
    assert result["session"]["stop_reason"] == "budget_exhausted"
    assert result["session"]["usage"]["roles"]["result_diagnosis"]["model_requests"] == 1


def test_expected_code_or_status_cannot_replace_actual_oracle_evidence(tmp_path, frozen_products):
    case, workspace, _, result = execute(tmp_path, frozen_products)
    assert evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle())["oracle_passed"] is False
    forged = deepcopy(result)
    forged["result_evidence"]["assessment"]["issue_codes"] = ["unit_mismatch"]
    forged["result_evidence"]["assessment"]["status"] = "invalid_evidence"
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=forged, workspace=workspace, oracle=oracle())
    assert scored["oracle_passed"] is scored["claim_matches_independent_evidence"] is False
    assert scored["protocol_preserved"] is True


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_default_adapter_dispatches_completed_result_trials_and_private_oracle(tmp_path, frozen_products, arm):
    from src.benchmark.recovery import DEFAULT_BUDGET, DEFAULT_TOOLS
    from src.benchmark.recovery_adapter import RecoveryAdapter
    case, workspace, content = prepare(tmp_path, actual={**REFERENCE, "unit": "percent", "value": 90})
    environment = tmp_path / "environment"
    environment.mkdir()
    private = tmp_path / "private_oracle.json"
    private.write_text(json.dumps(oracle()), "utf-8")
    case["oracle"] = {"kind": "result_evidence_v1", "reference": "original private result judgment",
                      "sha256": hashlib.sha256(private.read_bytes()).hexdigest()}
    artifact_id = "artifact_" + hashlib.sha256(content.encode()).hexdigest()[:16]
    replies = {"result_diagnosis": [{"hypotheses": []}], "recovery_investigation": [
        {"action": "probe", "reason": "Inspect the original public result before deciding.", "hypotheses": [],
         "evidence_ids": ["validation"], "tool": "artifact_contract", "arguments": {"paths": ["metrics.json"]}},
        {"action": "finish", "reason": "Observed units conflict with the frozen paper reference.", "hypotheses": [],
         "evidence_ids": [artifact_id]}],
        "recovery_self_check": [{"action": "accept", "reason": "Original artifact proves the unit mismatch.", "evidence_ids": [artifact_id]}],
        "recovery_counter_review": [{"action": "accept", "reason": "Original artifact proves the unit mismatch.", "evidence_ids": [artifact_id]}]}
    products, identities = frozen_products
    adapter = RecoveryAdapter(product_roots=products, oracle_paths={case["id"]: private},
                              development_responses=replies, worker_timeout_s=60)
    config = {"mode": {"A": "baseline", "B": "single", "C": "collaborative"}[arm],
        "model": "scripted-development-only", "tools": [] if arm == "A" else list(DEFAULT_TOOLS),
        "budget": None if arm == "A" else deepcopy(DEFAULT_BUDGET), "product": identities[arm]}
    result = adapter.run_trial(case={name: value for name, value in case.items() if name != "oracle"},
        arm=arm, repeat=1, workspace=workspace, environment=environment, context={}, configuration=config)
    receipt = adapter._receipts[str(workspace.resolve())]
    assert receipt["initial_success_observed"] is receipt["execution_succeeded"] is True, receipt
    assert receipt["initial_failure_observed"] is False
    assert result["claimed_success"] is (arm != "A"), receipt
    assert result["execution_scope"] == "author_result_investigation"
    assert result["usage"]["model_requests"] == (1 if arm == "A" else 4)
    assert result["usage"]["probes"] == (0 if arm == "A" else 1)
    assert result["usage"]["repair_actions"] == result["usage"]["source_repair_rounds"] == 0
    assert receipt["loaded_product_modules"]["src.result_diagnosis"] == "src/result_diagnosis.py"
    assert ("src.investigation_tools" in receipt["loaded_product_modules"]) is (arm != "A")
    scored = adapter.evaluate_trial(case=case, result=result, workspace=workspace, environment=environment)
    assert scored["oracle_passed"] is (arm != "A"), scored
    assert scored["cleanup_confirmed"] is scored["protocol_preserved"] is True
    assert (workspace / "metrics.json").read_text("utf-8") == content


@pytest.mark.parametrize("value,status", [(.5, "reference_gap"), (.9, "reference_met")])
def test_actual_selected_metric_observation_is_needed_for_numerical_reference_decision(tmp_path, frozen_products, value, status):
    case, workspace, _, result = execute(tmp_path, frozen_products, actual={**REFERENCE, "value": value})
    assert result["result_evidence"]["assessment"]["status"] == status
    scored = evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle(status, []))
    assert scored["oracle_passed"] is True


def test_protocol_known_reference_cannot_be_hidden_to_claim_safe_unknown(tmp_path):
    case, workspace, _ = prepare(tmp_path)
    case["initial_evidence"]["plan"]["experiment_protocol"] = {
        "method": {"name": REFERENCE["method"]}, "dataset": {"name": REFERENCE["dataset"], "split": REFERENCE["split"]},
        "repetitions": {"count": 3, "aggregation": "mean"},
        "metrics": [{"name": "accuracy", "reference": .9, "unit": "fraction"}]}
    case["initial_evidence"]["result_contract"]["reference"]["value"] = None
    with pytest.raises(ValueError, match="frozen experiment protocol"):
        validate_result_evidence(case, workspace)


def test_baseline_reuses_its_existing_verified_comparison_without_new_tools(tmp_path, frozen_products):
    actual = {**REFERENCE, "value": .7}
    checks = {field: {"status": "verified"} for field in ("unit", "split", "aggregation", "runs")}
    # Shape produced by the existing author_claim_comparison validator: metric
    # is on the row, final status is compared; field_evidence has four keys.
    row = {"metric": "accuracy", "status": "comparable", "meets_reference": False, "within_tolerance": False,
        "reference": {**{key: value for key, value in REFERENCE.items() if key != "metric"}, "field_evidence": checks},
        "actual": {**{key: value for key, value in actual.items() if key != "metric"}, "field_evidence": checks}}
    validation = {"status": "author_code_completed", "author_claim_comparison": {
        "status": "compared", "comparisons": [row], "relative_tolerance": .05}}
    case, workspace, _, result = execute(tmp_path, frozen_products, arm="A", actual=actual, initial_validation=validation)
    receipt = result["result_evidence"]
    assert receipt["completed"] is True
    assert receipt["assessment"]["status"] == "reference_gap"
    assert "src.investigation_tools" not in result["modules"]
    assert evaluate_result_evidence(evidence=case["initial_evidence"], receipt=result,
        workspace=workspace, oracle=oracle("reference_gap", []))["oracle_passed"] is True


@pytest.mark.parametrize('direction,value,within,meets,status', [
    ('unknown', .7, False, None, 'reference_gap'),
    ('unknown', .9, True, None, 'reference_met'),
    ('higher', 1.0, False, True, 'reference_gap'),
])
def test_baseline_keeps_existing_closeness_verdict_when_directional_verdict_differs(
        tmp_path, frozen_products, direction, value, within, meets, status):
    actual = {**REFERENCE, 'value': value}
    checks = {field: {'status': 'verified'} for field in ('unit', 'split', 'aggregation', 'runs')}
    row = {'metric': 'accuracy', 'status': 'comparable', 'direction': direction,
        'within_tolerance': within, 'meets_reference': meets,
        'reference': {**{key: item for key, item in REFERENCE.items() if key != 'metric'}, 'field_evidence': checks},
        'actual': {**{key: item for key, item in actual.items() if key != 'metric'}, 'field_evidence': checks}}
    validation = {'status': 'author_code_completed', 'author_claim_comparison': {
        'status': 'compared', 'comparisons': [row], 'relative_tolerance': .05}}
    case, workspace, _, result = execute(tmp_path, frozen_products, arm='A', actual=actual,
                                         initial_validation=validation)
    receipt = result['result_evidence']
    assert receipt['completed'] is True
    assert receipt['assessment']['status'] == status
    assert 'src.investigation_tools' not in result['modules']
    assert result['calls'] in ([], ['result_diagnosis'])
    assert evaluate_result_evidence(evidence=case['initial_evidence'], receipt=result,
        workspace=workspace, oracle=oracle(status, []))['oracle_passed'] is True


@pytest.mark.parametrize("alteration", ["no_tolerance", "editable", "path", "protocol"])
def test_result_schema_preserves_unknown_values_and_frozen_protocol(tmp_path, alteration):
    case, workspace, _ = prepare(tmp_path)
    evidence = case["initial_evidence"]
    if alteration == "no_tolerance": evidence["result_contract"].pop("relative_tolerance")
    elif alteration == "editable": evidence["editable"] = ["train.py"]
    elif alteration == "path": evidence["result_contract"]["artifact"] = "../oracle.json"
    else:
        evidence["plan"]["experiment_protocol"] = {"method": {"name": "other"},
            "metrics": [{"name": "accuracy", "reference": .9, "unit": "fraction"}]}
    with pytest.raises(ValueError):
        validate_result_evidence(case, workspace)


@pytest.mark.parametrize("content", ['{"value":NaN}', '{"value":1e999}', '{"value":1,"value":2}', '[]'])
def test_untrusted_artifact_json_cannot_supply_nonfinite_or_duplicate_values(tmp_path, content):
    case, workspace, _ = prepare(tmp_path)
    (workspace / "metrics.json").write_text(content, "utf-8")
    observed = _observe(case["initial_evidence"]["result_contract"], workspace)
    assert "artifact_invalid" in observed["issue_codes"]
    json.dumps(observed, allow_nan=False)
