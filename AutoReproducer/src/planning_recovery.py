"""Task-ledger integration for existing source-grounded planning attempts.

The planner still owns its deterministic validators and bounded reading loops.
Only a correction after an observed failure is a recovery action; receiving a
model response alone never verifies that correction.
"""
from copy import deepcopy
import hashlib
import json

from src.intelligent_recovery import current_investigation


def planning_evidence(packet, proposal=None):
    inventory = packet.get('file_inventory') or {}
    return {'pdf_sha256': (packet.get('pdf') or {}).get('sha256'),
            'repository': deepcopy(packet.get('repository', {}).get('url')),
            'revision': packet.get('repository', {}).get('revision'),
            'source_sha256': hashlib.sha256(json.dumps(
                {name: value.get('sha256') for name, value in inventory.items()},
                sort_keys=True).encode('utf-8')).hexdigest(),
            'read_sources': {name: value.get('sha256') for name, value in
                             (packet.get('repository', {}).get('files') or {}).items()},
            'experiment_request': deepcopy(packet.get('experiment_request') or {}),
            'previous_proposal': deepcopy(proposal)}


def planning_attempt(operation, packet, *, problem='', proposal=None, strict=False, fallback=False):
    """Execute one existing attempt, charging corrections to the shared ledger."""
    session = current_investigation()
    if session is None:
        return operation()
    with session.phase('entrypoint_planning'):
        session.ensure_running()
        if not problem:
            return operation()
        evidence = planning_evidence(packet, proposal)
        session.record_issue(problem, evidence=evidence, role='PlanValidator')
        kind = ('recover_literal_author_command' if fallback else
                'replan_strict_experiment' if strict else 'replan_original_command')

        def verify_plan(value):
            valid = bool(isinstance(value, dict) and value.get('steps') and value.get('evidence_sha256'))
            return {'success': True if valid else None,
                'reason': ('Original command passed deterministic entrypoint, arguments, data and dependency validation'
                           if valid else 'Additional source evidence was collected; no executable plan verified yet'),
                'evidence': {'plan_evidence_sha256': value.get('evidence_sha256') if valid else None},
                # Each attempt validates the complete original plan. A final
                # valid plan resolves earlier structural errors as well as the
                # most recent one; a source read alone proves none of them fixed.
                'resolved_issue_ids': [row['id'] for row in session.state['unresolved_issues']
                    if valid and row.get('phase') == 'entrypoint_planning' and row.get('status') != 'resolved']}

        return session.run_action(kind,
            operation, problem=problem, evidence=evidence, role='AuthorPlanner',
            review={'status': 'accepted', 'reviewer': 'deterministic_plan_rules',
                    'reason': 'Correct the observed validation failure using the same pinned paper, source and experiment request.'},
            verify=verify_plan)


def planning_llm(llm):
    session = current_investigation()
    return session.wrap_llm(llm) if session is not None and llm is not None else llm


def durable_planning_context():
    session = current_investigation()
    return ('\nDurable task goals, verified facts and unresolved objections (navigation only):\n'
            + json.dumps(session.critical_context(), ensure_ascii=False, sort_keys=True)
            if session else '')


def validate_plan_consistency(plan, packet, workspace, environment):
    """Recheck concrete inputs at the planner/environment handoff, without LLM gates."""
    from src.discovered_repository_plan import PlanEvidenceError
    from src.method_adapters import digest
    from src.safety.paths import workspace_path

    inventory = packet.get('file_inventory') or {}
    checked = []
    for step in plan.get('steps') or []:
        name = (step.get('entrypoint') or {}).get('source_path')
        source = workspace_path(workspace, name, 'original entrypoint', must_exist=True, forbid_git=True)
        if name not in inventory or not source.is_file() or digest(source) != inventory[name]['sha256']:
            raise PlanEvidenceError('Original entrypoint differs from pinned source: ' + str(name),
                                    category='source_changed')
        workspace_path(workspace, step.get('cwd', '.'), 'original command directory',
                       must_exist=True, forbid_git=True)
        checked.append({'entrypoint': name, 'sha256': inventory[name]['sha256'], 'cwd': step.get('cwd', '.')})
    for dataset in plan.get('datasets') or []:
        names = dataset.get('paths', []) if dataset.get('kind') == 'bundled' else [dataset.get('path')]
        for name in filter(None, names):
            source = workspace_path(workspace, name, 'original data source', must_exist=True, forbid_git=True)
            if name not in inventory or not source.is_file() or digest(source) != inventory[name]['sha256']:
                raise PlanEvidenceError('Original dataset or loader differs from pinned source: ' + name,
                                        category='source_changed')
    if (environment.get('author_requirements') or []) != (plan.get('requirements', {}).get('author') or []):
        raise PlanEvidenceError('Environment author declarations differ from the validated plan',
                                category='protocol_conflict')
    return {'status': 'verified', 'entrypoints': checked,
            'dataset_declarations': len(plan.get('datasets') or []),
            'device': plan.get('capture', {}).get('device'),
            'note': 'Checks frozen input consistency; execution and dataset completeness require runtime evidence.'}
