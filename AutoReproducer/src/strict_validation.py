"""Explicit, evidence-bound acceptance for one complete paper experiment.

Author execution remains the default. This module connects the existing strict
planner and independent evaluator without promoting printed author metrics.
"""
from copy import deepcopy
import math

from src.agents.experiment_protocol import PROTOCOL_SCHEMA, validate_experiment_protocol
from src.discovered_repository_plan import (
    PlanEvidenceError, _SEMANTIC_GATES, _canonical, _citations, _digest,
    _references, validate_plan,
)


def normalize_strict_validation_request(request):
    """Normalize the explicit boolean mode; legacy review flags are unrelated."""
    if not isinstance(request, dict):
        raise ValueError('Reproduction request must be an object')
    strict = request.get('strict_validation', False)
    if type(strict) is not bool:
        raise ValueError('strict_validation must be an explicit boolean')
    if strict and (request.get('mock_mode') or request.get('code')
                   or request.get('corpus_paper') or request.get('experiment_profile')):
        raise ValueError('Strict validation requires a real PDF and the generic repository workflow')
    return {**request, 'strict_validation': strict,
            'author_execution_only': not strict, 'use_llm_review': False,
            'allow_result_summary_review': False}


def _without_digest(value, key):
    body = deepcopy(value)
    supplied = body.pop(key, None)
    if supplied != _digest(_canonical(body).encode('utf-8')):
        raise PlanEvidenceError('Frozen strict evidence has an invalid ' + key)
    return body


def _reviewed_candidate(plan):
    """Recover the exact pre-review candidate, including review-added citations.

    review_plan only appends newly selected citations. Trying bounded prefixes
    recovers its original candidate without dropping any executable/value field.
    """
    body = deepcopy(plan)
    review = body.pop('semantic_review')
    body.pop('plan_sha256', None)
    citations = body.get('citations')
    if not isinstance(citations, list) or not 1 <= len(citations) <= 100:
        raise PlanEvidenceError('Strict semantic review needs bounded original citations')
    for count in range(len(citations), 0, -1):
        candidate = {**body, 'citations': citations[:count]}
        digest = _digest(_canonical(candidate).encode('utf-8'))
        if digest == review.get('reviewed_plan_sha256'):
            return {**candidate, 'plan_sha256': digest}
    raise PlanEvidenceError('Strict semantic review does not bind the frozen execution plan')


def assert_strict_plan(plan, packet, workspace):
    """Recheck frozen sources, protocol selection and every independent review.

    This performs no model calls and no execution. A runtime dependency warning
    may remain deferred until the independent execution actually succeeds.
    """
    try:
        if not isinstance(plan, dict) or not isinstance(packet, dict):
            raise PlanEvidenceError('Strict validation needs a frozen plan and source packet')
        if packet.get('author_execution_only') is not False:
            raise PlanEvidenceError('Strict validation cannot use execution-only evidence')
        if packet.get('repository_identity_pending'):
            raise PlanEvidenceError('Strict repository identity is still pending')
        if packet.get('repository_identity_review'):
            from src.agents.repository_verifier import validate_identity_review
            validated = validate_identity_review(packet, packet['repository_identity_review'])
            if validated != packet['repository_identity_review']:
                raise PlanEvidenceError('Strict repository identity review is not source-bound')
        protocol = packet.get('experiment_protocol')
        if not isinstance(protocol, dict) or plan.get('experiment_protocol') != protocol:
            raise PlanEvidenceError('Strict plan needs the identical frozen experiment protocol')
        _without_digest(protocol, 'protocol_sha256')
        selected = validate_experiment_protocol(
            {key: deepcopy(protocol[key]) for key in PROTOCOL_SCHEMA}, packet, workspace)
        if selected != protocol:
            raise PlanEvidenceError('Strict protocol differs from its original source evidence')
        if plan.get('pdf_sha256') != packet['pdf']['sha256']:
            raise PlanEvidenceError('Strict plan is bound to a different PDF')
        repository = {key: packet['repository'][key] for key in ('url', 'revision')}
        if plan.get('repository') != repository or protocol.get('repository') != repository:
            raise PlanEvidenceError('Strict plan is bound to a different repository revision')
        evidence_hash = _digest(_canonical(packet).encode('utf-8'))
        if plan.get('evidence_sha256') != evidence_hash:
            raise PlanEvidenceError('Strict source packet changed after planning')
        _without_digest(plan, 'plan_sha256')
        review = plan.get('semantic_review')
        if (not isinstance(review, dict) or review.get('accepted') is not True
                or review.get('task') != 'discovered_repository_plan_review'
                or review.get('evidence_sha256') != evidence_hash):
            raise PlanEvidenceError('Strict acceptance needs an independent source-bound semantic review')
        candidate = _reviewed_candidate(plan)
        proposal = deepcopy(candidate)
        # The entrypoint resolver accepts declarations; source_path is its
        # compiled output and must be derived again from the pinned argv.
        for step in proposal['steps']:
            if isinstance(step.get('entrypoint'), dict):
                step['entrypoint'] = {key: step['entrypoint'][key] for key in ('kind', 'target')}
        if validate_plan(proposal, packet, workspace) != candidate:
            raise PlanEvidenceError('Reviewed strict plan no longer matches deterministic source validation')
        # Re-read every quotation, including source IDs added by the reviewer.
        citations = _citations({'citations': plan['citations']}, packet, workspace)
        checks = review.get('checks')
        if not isinstance(checks, dict) or set(checks) != set(_SEMANTIC_GATES):
            raise PlanEvidenceError('Strict semantic review omitted a required protocol check')
        deferred = []
        for gate, kinds in _SEMANTIC_GATES.items():
            check = checks[gate]
            warning = (gate == 'dependency_compatibility' and isinstance(check, dict)
                       and check.get('passed') is False and check.get('severity') == 'warning'
                       and check.get('verification_stage') == 'runtime_preparation_and_execution')
            if (not isinstance(check, dict) or (check.get('passed') is not True and not warning)
                    or (check.get('severity', 'blocking') != 'blocking' and not warning)
                    or not isinstance(check.get('reason'), str)
                    or not 12 <= len(check['reason'].strip()) <= 6000):
                raise PlanEvidenceError('Strict semantic review did not establish: ' + gate)
            refs = _references(check.get('citations'), citations)
            if any(not any(ref['source'] == kind for ref in refs) for kind in kinds):
                raise PlanEvidenceError('Strict semantic review lacks original evidence: ' + gate)
            if warning:
                deferred.append(gate)
        expected_aggregation = ('required_protocol_checks_with_runtime_dependency_verification'
                                if deferred else 'all_required_verified_checks')
        if (review.get('aggregation') != expected_aggregation
                or review.get('deferred_checks', []) != deferred):
            raise PlanEvidenceError('Strict semantic review has inconsistent deferred checks')
        return {'status': 'verified', 'scope': 'selected_paper_experiment',
                'pdf_sha256': plan['pdf_sha256'], 'repository': deepcopy(repository),
                'protocol_sha256': protocol['protocol_sha256'],
                'evidence_sha256': evidence_hash, 'plan_sha256': plan['plan_sha256'],
                'reviewed_plan_sha256': review['reviewed_plan_sha256'],
                'deferred_checks': deferred}
    except PlanEvidenceError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PlanEvidenceError('Strict acceptance evidence is invalid: ' + str(exc)) from exc


def finalize_strict_validation(validation, plan, packet, workspace):
    """Label an existing verify_capture result after checking its strict gates.

    Call only with the result of verify_capture. Prefer verify_strict_execution
    when owned process records and runtime artifacts are available together.
    """
    receipt = assert_strict_plan(plan, packet, workspace)
    checks = validation.get('completion_checks') if isinstance(validation, dict) else None
    required = ('required_processes_exited', 'full_training_and_test_capture',
                'all_native_runs_evaluated', 'fresh_model_predictions_verified',
                'author_metric_reconciled')
    if (not isinstance(checks, dict) or any(checks.get(key) is not True for key in required)
            or validation.get('protocol_pass') is not True
            or validation.get('independent_metrics_pass') is not True
            or type(validation.get('is_reproduced')) is not bool
            or checks.get('paper_reference_met') is not validation['is_reproduced']):
        raise PlanEvidenceError('Strict acceptance requires complete independent execution evidence')
    comparisons = validation.get('comparisons')
    if (not isinstance(comparisons, list) or len(comparisons) != len(plan['metrics'])
            or any(not isinstance(row, dict) or type(row.get('pass')) is not bool for row in comparisons)
            or all(row['pass'] for row in comparisons) is not validation['is_reproduced']):
        raise PlanEvidenceError('Strict acceptance comparisons do not reconcile with its conclusion')
    for row, metric in zip(comparisons, plan['metrics']):
        if any(row.get(key) != value for key, value in metric.items()):
            raise PlanEvidenceError('Strict comparison changed a frozen reference or metric')
        actual = row.get('actual')
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or actual < 0 or (metric['name'] == 'accuracy'
                                 and actual > (100 if metric['unit'] == 'percent' else 1))):
            raise PlanEvidenceError('Strict comparison has an invalid independent metric')
        tolerance = abs(metric['reference']) * metric['tolerance_relative']
        passed = (actual >= metric['reference'] - tolerance if metric['direction'] == 'maximize'
                  else actual <= metric['reference'] + tolerance)
        if row['pass'] is not passed or row.get('absolute_tolerance') != tolerance:
            raise PlanEvidenceError('Strict comparison does not follow the frozen numerical rule')
    result = deepcopy(validation)
    result.update(strict_validation=True, paper_metrics_verified=True,
                  scope='selected_paper_experiment', reproduction_scope='selected_paper_experiment',
                  reviews_removed=False, pre_execution_review='verified',
                  strict_plan_verification=receipt)
    return result


def verify_strict_execution(plan, packet, workspace, execution, spec_hash):
    """Verify owned full execution and fresh inference against a reviewed plan."""
    assert_strict_plan(plan, packet, workspace)
    from src.repository_reproduction import spec_digest
    from src.discovered_repository_runtime import verify_capture
    if spec_hash != spec_digest(plan):
        raise PlanEvidenceError('Strict runtime evidence uses a different frozen specification')
    validation = verify_capture(plan, workspace, execution, spec_hash)
    capture_pid = validation['capture'].get('process_pid')
    evaluated = validation['independent_metrics']
    evaluation_pid = evaluated.get('process_pid')
    if (type(capture_pid) is not int or capture_pid <= 0
            or type(evaluation_pid) is not int or evaluation_pid <= 0
            or capture_pid == evaluation_pid
            or type(evaluated.get('capture_process_pid')) is not int
            or evaluated['capture_process_pid'] != capture_pid):
        raise PlanEvidenceError('Strict metrics must be recomputed in a separate recorded process')
    return finalize_strict_validation(validation, plan, packet, workspace)
