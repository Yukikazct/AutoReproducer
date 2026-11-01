"""Run one independently frozen correction of a documented author invocation."""
from copy import deepcopy
from pathlib import Path

from src.method_adapters import write_json
from src.repository_reproduction import export_repository, spec_digest


def recover_outcome(service, request, data, plan, packet, snapshot, workspace, run_dir, emit, event):
    from src.outcome_recovery import prepare_outcome_recovery
    from src.discovered_repository_reproduction import environment_for_plan, prepare_plan_datasets
    from src.intelligent_recovery import current_investigation
    from src.planning_recovery import validate_plan_consistency

    if (data.get('outcome_recovery') or {}).get('attempts'):
        return  # A resumed/re-entered service cannot reserve another attempt.
    # All decisions, including explicit unsupported outcomes, remain reviewable.
    decision = prepare_outcome_recovery(plan, deepcopy(packet), workspace,
        data.get('execution') or {}, data.get('validation') or {}, request)
    if decision.get('status') == 'proposed' and (data.get('data_repair_attempts') or data.get('data_recovery_analyses')):
        decision = {'status': 'unresolved', 'reason': '已恢复的作者外部输入尚无独立工作副本合同，保留本次数据与执行证据。'}
    public = {key: deepcopy(value) for key, value in decision.items() if key not in {'plan', 'packet'}}
    record = {'version': 1, **public, 'max_attempts': 1, 'attempts': [],
              'original_attempt_dir': str(run_dir), 'numerical_equivalence_verified': False}
    data['outcome_recovery'] = record
    destination = Path(run_dir) / 'outcome_recovery.json'
    write_json(destination, record)
    if decision.get('status') != 'proposed':
        return
    investigation = current_investigation()
    corrected, evidence = decision['plan'], decision['packet']
    attempt_dir = Path(run_dir) / 'outcome_recovery' / 'attempt_1'
    attempt = {'id': 1, 'status': 'preparing', 'run_dir': str(attempt_dir),
               'original_spec_sha256': data['spec_sha256'],
               'corrected_spec_sha256': spec_digest(corrected),
               'original_validation_status': (data.get('validation') or {}).get('status')}
    record.update(status='running', reason='已找到唯一的作者原文完整命令，正在独立冻结后完整复跑。')
    record['attempts'].append(attempt)
    baseline = deepcopy(data)
    child = deepcopy(data)
    for key in ('execution', 'validation', 'outcome_diagnosis', 'metric_parse_errors',
                'execution_plan', 'runtime_dataset_evidence', 'source_repair_workspace'):
        child.pop(key, None)
    if 'intelligent_recovery' in data:
        child['intelligent_recovery'] = data['intelligent_recovery']
    child['outcome_recovery'] = record
    child['source_repair'] = {'requested': request.get('source_repair', False),
        'auto': request.get('source_repair_auto', False),
        'editable': list(request.get('source_repair_editable') or []), 'max_rounds': 2,
        'status': 'pending' if request.get('source_repair') else 'not_requested'}
    child.update(active_attempt_dir=str(attempt_dir),
                 grounded_repository_plan=corrected, spec_sha256=attempt['corrected_spec_sha256'])
    try:
        if investigation:
            investigation.consume_repair()
            investigation.ensure_running()
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json(destination, record)
        event('assess_reproduction', 'running', record['reason'])
        new_workspace = attempt_dir / 'repo'
        exported = export_repository(service.root, {'repository': corrected['repository']}, new_workspace,
            offline=bool(request.get('offline')), required_files=[], include_submodules=False)
        if (exported['files'] != snapshot['files'] or exported['resolved_sha'] != snapshot['resolved_sha']
                or exported['url'] != snapshot['url']):
            raise ValueError('Outcome correction cannot change the pinned author source or bundled inputs')
        child['repository'] = exported
        write_json(attempt_dir / 'repository.json', exported)
        write_json(attempt_dir / 'source_packet.json', evidence)
        write_json(attempt_dir / 'experiment_spec.json', {'spec': corrected, 'sha256': child['spec_sha256']})
        environment = environment_for_plan(corrected)
        environment['offline'] = bool(request.get('offline'))
        child['env_config'] = deepcopy(environment)
        child['plan_consistency'] = validate_plan_consistency(corrected, evidence, new_workspace, environment)
        write_json(attempt_dir / 'plan_consistency.json', child['plan_consistency'])
        child['dataset_provenance'] = prepare_plan_datasets(service.root, new_workspace, corrected, exported,
                                                          offline=bool(request.get('offline')))
        write_json(attempt_dir / 'dataset.json', child['dataset_provenance'])
        attempt['status'] = 'executing'
        write_json(destination, record)
        service._execute_frozen(request, child, corrected, exported, new_workspace, attempt_dir,
                                environment, emit, event)
        validation = child.get('validation') or {}
        assessment = validation.get('reproduction_assessment') or {}
        # Do not select whichever attempt looks best. The actual correction is
        # authoritative, and model judgement remains separate from strict proof.
        completed = (child.get('execution') or {}).get('success') is True
        accepted = completed and assessment.get('status') == 'completed' and assessment.get('verdict') == 'reproduced'
        attempt.update(status='completed' if completed else 'failed',
            validation_status=validation.get('status'),
            comparison_status=(validation.get('author_claim_comparison') or {}).get('status'))
        record.update(status='corrected' if accepted else 'unresolved',
            reason=('作者完整命令已独立复跑并重新完成比较；普通 AI 判断为复现，严格数值核验仍以单独证据为准。'
                    if accepted else '作者完整命令已复跑；本次偏差或证据缺口仍未解决，不再尝试其他参数。'))
    except BaseException as exc:
        attempt.update(status='failed', error_type=type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            attempt['interrupted'] = True
        record.update(status='failed', reason='完整作者命令纠正未完成；原始运行和失败证据均已保留。')
        if not child.get('execution'):
            child = baseline  # Preparation never became another author run.
            record['original_execution_preserved'] = True
        raise
    finally:
        child['outcome_recovery'] = record
        data.clear()
        data.update(child)
        for path in (destination, attempt_dir / 'outcome_recovery.json'):
            try:
                write_json(path, record)
            except OSError as error:
                record.setdefault('persistence_errors', []).append({'file': path.name, 'error_type': type(error).__name__})
        if attempt_dir.exists():
            try:
                write_json(attempt_dir / 'run_status.json', {'status': ('interrupted' if attempt.get('interrupted')
                    else 'completed' if attempt['status'] == 'completed' else 'failed'), 'kind': 'outcome_correction'})
            except OSError as error:
                record.setdefault('persistence_errors', []).append({'file': 'run_status.json', 'error_type': type(error).__name__})
        # Original attempt's files stay immutable; the final data points at
        # the new spec and all its execution/comparison artifacts explicitly.
