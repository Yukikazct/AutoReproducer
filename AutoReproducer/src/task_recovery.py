"""Durable task facts and actual recovery actions, independent of role mailboxes.

This ledger never grants execution authority. Callers supply existing controlled
operations and validators; a review or a delivered message is only a proposal.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import re
import time


_TRANSIENT = {'phase', 'role', 'run_id', 'run_dir', 'execution_record', 'execution_dir',
              'failed_execution_dir', 'timestamp', 'started_at', 'duration_seconds',
              'task_id', 'issue_id', 'reviewer', 'agent', 'stage'}


def semantic_evidence(value):
    """Ignore transport identities, retaining sources, checks and actual errors."""
    if isinstance(value, dict):
        return {key: semantic_evidence(item) for key, item in value.items() if key not in _TRANSIENT}
    if isinstance(value, (list, tuple)):
        return [semantic_evidence(item) for item in value]
    return value


def fingerprint(problem, evidence=None, kind=None):
    problem = re.sub(r'\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|\b[0-9a-f]{32}\b', '<run>', str(problem))
    value = [kind, problem, semantic_evidence(evidence)]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode('utf-8')).hexdigest()


def summarize_recovery_state(state):
    """Summarize a detached receipt without creating a session or new actions."""
    from src.execution_safety import cleanup_is_unconfirmed
    actions = [row for row in state.get('actions', []) if isinstance(row, dict)]

    def executed(row):
        return (row.get('execution') or {}).get('status') in {'completed', 'succeeded', 'failed',
            'executed', 'stopped', 'success', 'error', 'cancelled', 'timed_out'}

    def verified(row):
        execution = row.get('execution') or {}
        result = execution.get('result') or {}
        return (executed(row) and row.get('status') not in {'stopped', 'rejected', 'no_progress'}
            and execution.get('status') not in {'stopped', 'failed', 'error', 'cancelled', 'timed_out'}
            and not cleanup_is_unconfirmed([execution, result])
            and result.get('cancelled') is not True and result.get('timed_out') is not True
            and (row.get('verification') or {}).get('success') is True)

    budget, usage = state.get('budget') or {}, state.get('usage') or {}
    remaining = {}
    for label, limit, used in [('probes', 'max_probes', 'probes'),
            ('source_patch_rounds', 'max_source_repair_rounds', 'source_patch_rounds'),
            ('repair_actions', 'max_repair_actions', 'repair_actions')]:
        maximum, consumed = budget.get(limit), usage.get(used)
        remaining[label] = max(0, maximum - consumed) if type(maximum) is int and type(consumed) is int else None
    return {'recorded': True, 'status': state.get('status', 'unrecorded'),
        'actions_proposed': len(actions), 'actions_executed': sum(executed(row) for row in actions),
        'actions_verified': sum(verified(row) for row in actions),
        'actions_rejected': sum(row.get('status') == 'rejected' for row in actions),
        'actions_failed': sum(row.get('status') in {'failed', 'no_progress'} for row in actions),
        'unresolved_issues': [deepcopy(row) for row in state.get('unresolved_issues', [])
                              if isinstance(row, dict) and row.get('status') != 'resolved'],
        'remaining_budget': remaining, 'requires_human': state.get('requires_human', False),
        'stop_reason': state.get('stop_reason', '')}


class TaskRecoveryLedger:
    """Mixin for RecoverySession; one lock, allowance and journal per task."""

    @contextmanager
    def phase(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Recovery phase must be a nonempty string')
        previous = self.state['phase']
        self.state['phase'] = name
        self.state['phases'].append({'phase': name, 'status': 'entered'})
        self.persist()
        try:
            self.ensure_running()
            yield self
        finally:
            self.state['phase'] = previous
            self.persist()

    def freeze_goal(self, goal, evidence=None):
        """Freeze individual goal keys; later services may add new constraints."""
        if not isinstance(goal, dict):
            raise TypeError('Frozen experiment goal must be a dictionary')
        clean = self.safe(deepcopy(goal))
        with self._mutation_lock:
            frozen = self.state['goals']
            for key, value in clean.items():
                if key in frozen and frozen[key] != value:
                    raise ValueError('Frozen experiment goal changed: ' + str(key))
            frozen.update(clean)
            if evidence is not None:
                self.remember_fact('goal_evidence', evidence)
            self.persist()
        return deepcopy(frozen)

    def remember_fact(self, key, value, evidence_ids=()):
        if not isinstance(key, str) or not key:
            raise ValueError('Task fact must have a stable name')
        with self._mutation_lock:
            record = self.safe({'value': deepcopy(value), 'evidence_ids': list(evidence_ids),
                                'phase': self.state['phase']})
            self.state['facts'][key] = record
            self.persist()
            return deepcopy(record)

    def record_issue(self, problem, evidence=None, role=None):
        clean_problem, clean_evidence = self.safe(str(problem)), self.safe(evidence or {})
        key = fingerprint(clean_problem, clean_evidence)
        with self._mutation_lock:
            existing = next((row for row in self.state['unresolved_issues']
                             if row['fingerprint'] == key), None)
            if existing is not None:
                return existing
            row = {'id': 'issue_' + str(len(self.state['unresolved_issues']) + 1),
                   'problem': clean_problem, 'evidence': clean_evidence, 'fingerprint': key,
                   'task_id': (self._active_task or {}).get('id'),
                   'phase': self.state['phase'], 'role': role or 'Coordinator', 'status': 'unresolved'}
            self.state['unresolved_issues'].append(row)
            self.state['requires_human'] = True
            self.persist()
            return row

    def record_decision(self, kind, *, reason='', evidence=None, status='proposed', role='Coordinator'):
        row = self.safe({'id': 'decision_' + str(len(self.state['decisions']) + 1),
                         'kind': kind, 'reason': reason, 'evidence': evidence or {},
                         'status': status, 'role': role, 'phase': self.state['phase']})
        self.state['decisions'].append(row)
        self.persist()
        return row

    def critical_context(self):
        """Facts and unresolved objections survive the bounded recent mailbox."""
        return self.safe({'goals': deepcopy(self.state['goals']),
            'facts': deepcopy(self.state['facts']),
            'unresolved_issues': [deepcopy(row) for row in self.state['unresolved_issues']
                                  if row['status'] != 'resolved'],
            'rejected_actions': [deepcopy(row) for row in self.state['actions']
                                 if row['status'] in {'rejected', 'no_progress'}],
            'remaining_budget': self.recovery_summary()['remaining_budget']})

    def recovery_summary(self):
        return summarize_recovery_state(self.state)

    def begin_action(self, kind, *, problem='', evidence=None, role='Coordinator', review=None,
                     source_patch=False):
        """Reserve a real recovery attempt before execution, not on receipt."""
        from src.intelligent_recovery import InvestigationStopped
        with self._mutation_lock:
            self.ensure_running()
            issue = self.record_issue(problem, evidence, role) if problem else None
            key = fingerprint(self.safe(problem), self.safe(evidence or {}), kind)
            duplicate = any(row['fingerprint'] == key for row in self.state['actions'])
            row = self.safe({'id': 'action_' + str(len(self.state['actions']) + 1),
                'kind': kind, 'phase': self.state['phase'], 'role': role,
                'task_id': (self._active_task or {}).get('id'),
                'problem': problem, 'issue_id': issue['id'] if issue else None,
                'evidence': evidence or {}, 'fingerprint': key, 'source_patch': bool(source_patch),
                'status': 'proposed', 'review': review or {'status': 'unrecorded'},
                'execution': {'status': 'not_executed'},
                'verification': {'success': None, 'reason': 'Not verified'},
                'started_at': time.time()})
            self.state['actions'].append(row)
            if duplicate:
                row['status'] = 'no_progress'
                self.stop('no_progress', requires_human=True)
                raise InvestigationStopped('Unchanged recovery action cannot be repeated without new evidence')
            if row['review'].get('status') in {'rejected', 'reject'}:
                row['status'] = 'rejected'
                self.persist()
                raise ValueError('Recovery action rejected: ' + str(row['review'].get('reason') or kind))
            enclosing = self._running_action
            self._running_action = None
            try:
                self.consume_repair(source_patch=source_patch)
            except BaseException:
                row['status'] = 'rejected'
                row['review'] = {'status': 'rejected', 'reviewer': 'shared_budget_controller',
                                 'reason': 'No shared recovery allowance remains'}
                self.persist()
                raise
            finally:
                self._running_action = enclosing
            row['status'] = 'executing'
            self.persist()
            return row

    def finish_action(self, action, *, result=None, verification=None, error=None, status=None,
                      executed=True, original_command_rerun=False):
        """Attach actual effect; unknown verification never becomes success."""
        with self._mutation_lock:
            if action.get('status') == 'stopped' or self.state.get('status') == 'stopped':
                status = 'stopped'
                verification = {**(verification or {}), 'success': False,
                    'reason': 'Task stopped before recovery verification completed'}
            if error is not None:
                action['execution'] = self.safe({'status': 'failed', 'error_type': type(error).__name__,
                                                  'error': str(error)})
                action['status'] = status or 'failed'
                action['verification'] = self.safe(verification or {
                    'success': None, 'reason': 'Execution failed before verification'})
            else:
                # Complete execution records live in existing verified attachments.
                record = {key: result.get(key) for key in ('success', 'run_dir', 'executed',
                    'repair_status', 'cancelled', 'timed_out', 'cleanup_confirmed') if key in result} \
                    if isinstance(result, dict) else {'returned': type(result).__name__}
                action['execution'] = self.safe({'status': 'completed', 'result': record})
                action['verification'] = self.safe(verification or {'success': None, 'reason': 'Not verified'})
                success = action['verification'].get('success')
                action['status'] = status or ('verified' if success is True else 'failed' if success is False else 'executed')
                if success is True:
                    resolved_ids = action['verification'].get('resolved_issue_ids') or []
                    for issue in self.state['unresolved_issues']:
                        if issue['id'] == action.get('issue_id') or issue['id'] in resolved_ids:
                            issue.update(status='resolved', resolved_by=action['id'])
                    if not self._blocked:
                        self.state['requires_human'] = (any(issue['status'] != 'resolved'
                            for issue in self.state['unresolved_issues']) or any(task['status'] in
                            {'needs_human', 'repair_proposed'} for task in self.state['tasks']))
            action['duration_seconds'] = max(0.0, time.time() - action['started_at'])
            if not executed:
                action['execution']['status'] = 'not_executed'
            if original_command_rerun:
                action['execution']['original_command_rerun'] = True
            self.persist()
            return action

    def record_external_repair(self, kind, *, problem, evidence=None, role='CodeExecutor', reason=''):
        action = self.begin_action(kind, problem=problem, evidence=evidence, role=role,
            review={'status': 'accepted', 'reviewer': 'deterministic_repair_rules', 'reason': reason})
        self._legacy_action = action
        return action

    def verify_external_repair(self, result):
        action = self._legacy_action
        if action is None:
            return
        from src.repository_repairer import cleanup_is_unconfirmed, repair_is_stopped
        stopped = repair_is_stopped(result)
        reran = result.get('executed') is True or (result.get('final') or {}).get('executed') is True
        if stopped:
            self.stop('execution_stopped_or_cleanup_unconfirmed',
                cleanup_confirmed=False if cleanup_is_unconfirmed(result) else None)
        self.finish_action(action, result=result, verification={
            'success': (result.get('success') is True and not stopped) if reran else False if stopped else None,
            'reason': ('Original author command rerun; numerical equivalence remains unknown'
                       if reran else 'Original author command did not execute; repair effect remains unverified'),
            'evidence': {'execution_dir': result.get('run_dir'),
                         'exit_code': (result.get('final') or {}).get('exit_code')}},
            status='stopped' if stopped else None, original_command_rerun=reran)
        self._legacy_action = None

    def run_action(self, kind, operation, *, problem='', evidence=None, role='Coordinator',
                   review=None, verify=None, source_patch=False):
        """Serialize controlled mutations and record the validator's actual result."""
        from src.intelligent_recovery import InvestigationStopped
        with self._mutation_lock:
            action = self.begin_action(kind, problem=problem, evidence=evidence, role=role,
                                       review=review, source_patch=source_patch)
            previous = self._running_action
            self._running_action = action
            operation_started = False
            try:
                self.ensure_running()
                operation_started = True
                result = operation()
                from src.repository_repairer import cleanup_is_unconfirmed, repair_is_stopped
                if isinstance(result, dict) and repair_is_stopped(result):
                    self.stop('execution_stopped_or_cleanup_unconfirmed',
                        cleanup_confirmed=False if cleanup_is_unconfirmed(result) else None)
                    self.finish_action(action, result=result, verification={'success': False,
                        'reason': 'Execution stopped or cleanup unconfirmed'}, status='stopped')
                    self.ensure_running()
                self.ensure_running()
                verification = verify(result) if verify is not None else None
                self.finish_action(action, result=result, verification=verification)
                return result
            except BaseException as exc:
                if action['status'] != 'stopped':
                    interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit, InvestigationStopped))
                    cleanup_failed = getattr(exc, 'process_cleanup_confirmed', None) is False
                    timed_out = getattr(exc, 'timed_out', False) is True
                    self.finish_action(action, error=exc,
                        status='stopped' if interrupted or cleanup_failed or timed_out else 'failed',
                        executed=operation_started)
                    if cleanup_failed:
                        self.stop('action_cleanup_unconfirmed', requires_human=True, cleanup_confirmed=False)
                    elif timed_out:
                        self.state['timed_out'] = True
                        self.stop('action_timed_out', requires_human=True)
                    elif interrupted and not self._blocked:
                        self.stop('action_interrupted')
                raise
            finally:
                self._running_action = previous

    def consume_probe(self, *, tool='source_context', arguments=None):
        """Charge pinned source rereads and active observations to one allowance."""
        from src.intelligent_recovery import RecoveryBudgetExceeded
        with self._mutation_lock:
            self.ensure_running()
            if self.state['usage']['probes'] >= self.budget['max_probes']:
                self.stop('probe_budget_exhausted', requires_human=True)
                raise RecoveryBudgetExceeded('Shared observation budget exhausted')
            key = fingerprint('', arguments, tool)
            if key in self._probe_keys:
                self.stop('no_progress', requires_human=True)
                raise RecoveryBudgetExceeded('Repeated observation without new evidence')
            self._probe_keys.add(key)
            self.state['usage']['probes'] += 1
            self.persist()
