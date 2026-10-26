"""Error-driven dependency retries for the explicit author execution objective."""
from copy import deepcopy
import re
import sys

from src.agents.code_executor import _safe_install_diagnostic
from src.agents.dependency_resolver import find_missing_module
from src.author_dependencies import declared_requirements, import_distribution
from src.discovered_repository_plan import GRAPH_WHEEL_LINKS, _RUNTIME_DEPENDENCIES
from src.dependency_versions import compatibility_pins
from src.method_adapters import write_json
from src.author_self_repair import (failure_diagnostic, record_repair, repair_local_import,
                                    save_repair_state, terminal_missing_import, verify_repair_result)

MAX_DEPENDENCY_REPAIRS = 8


def repair_requirement(stderr, plan, packet):
    # Missing symbols in an installed package indicate an API/version problem,
    # not an absent dependency. Do not download a different package blindly.
    stderr = terminal_missing_import(stderr or '')
    if not stderr:
        return None
    module = find_missing_module(stderr)
    if not module or module in sys.stdlib_module_names:
        return None
    inventory = packet['file_inventory']
    declarations = declared_requirements(module, plan['requirements'].get('author', []))
    if len(declarations) > 1 or any(item.url for item in declarations):
        return None  # Conflicting versions or direct sources need further evidence.
    if not declarations and (any(name == module + '.py' or name.endswith('/' + module + '.py')
            or name.startswith(module + '/') or '/' + module + '/' in name for name in inventory)
            or any(item['name'] == module for item in plan['requirements'].get('repository_namespaces', []))):
        return None
    package = import_distribution(module)
    declared = str(declarations[0]) if declarations else None
    device = plan['capture'].get('device', 'cpu')
    pins = compatibility_pins(device)
    # Known native pairs retain their compatible pins; other packages prefer
    # the author's requirement. Undeclared dependencies are installed only
    # after their genuine runtime error, then cache health checks their import.
    requirement = package + '==' + pins[package] if package in pins else declared or package
    additions = [requirement]
    for dependency in _RUNTIME_DEPENDENCIES.get(package, ()):
        additions.append(dependency + '==' + pins[dependency])
    options = (["--find-links " + GRAPH_WHEEL_LINKS[device]]
               if any(item.split('==')[0] in {'torch-scatter', 'torch-sparse'} for item in additions)
               else [])
    return {'module': module, 'package': package, 'requirement': requirement,
            'additions': additions, 'requirement_options': options, 'author_declaration': declared,
            'source': 'actual_missing_module_error'}


def run_with_dependency_repair(runner, workspace, steps, environment, plan, packet,
                               run_dir, data, on_event, verify_inputs, repair_numeric=None, repair_path=None,
                               analyze_failure=None, repair_pyg=None, prepare_environment=None):
    """Retry after cleanup, retaining every failed run and the unchanged argv."""
    effective = deepcopy(environment)
    # Retry allowance counts actual failures, never time spent running code.
    effective.pop('deadline_monotonic', None)
    data.setdefault('dependency_repair_attempts', [])
    repairs = data['dependency_repair_attempts']
    numeric_repairs = data.setdefault('numeric_repair_attempts', [])
    path_repairs = data.setdefault('path_repair_attempts', [])
    model_repairs = data.setdefault('model_dependency_repairs', [])
    pyg_repairs = data.setdefault('pyg_repair_attempts', [])
    import_repairs = data.setdefault('import_repair_attempts', [])
    seen = {repair['package'] for repair in repairs}
    for attempt in range(MAX_DEPENDENCY_REPAIRS + 1):
        verify_inputs()
        if prepare_environment is not None:
            effective = prepare_environment(effective)
        try:
            result = runner.run(workspace, steps, effective, on_event=on_event)
        except (KeyboardInterrupt, SystemExit):
            verify_repair_result(run_dir, data, on_event, {'interrupted': True})
            save_repair_state(run_dir, data, on_event, status='interrupted', reason='运行已取消，自修复停止。')
            raise
        effective = deepcopy(result.get('effective_env_config') or effective)
        result.setdefault('effective_env_config', effective)
        verify_repair_result(run_dir, data, on_event, result)
        if result.get('success'):
            return result
        def stop(status, reason):
            save_repair_state(run_dir, data, on_event, status=status, reason=reason, current=result)
            return result
        if result.get('cancelled') or result.get('interrupted'):
            return stop('interrupted', '运行已取消，自修复停止。')
        if effective.get('dependency_failure_kind') == 'installation_timeout':
            # A killed installer is not evidence of a missing module or a bad
            # version pin. Retrying a model suggestion would restart downloads.
            return stop('timed_out', '依赖安装已超时；保留安装日志，未重新启动相同下载。')
        final = result.get('final') or {}
        if final.get('cancelled'):
            return stop('interrupted', '原命令已取消，自修复停止。')
        if final.get('timed_out'):
            return stop('timed_out', '原命令已超时，自修复停止；保留已产生的结果。')
        stderr = failure_diagnostic(result)
        used = sum(len(items) for items in (repairs, numeric_repairs, path_repairs,
                                            model_repairs, pyg_repairs, import_repairs))
        attempts = (data.get('self_repair') or {}).get('attempts') or []
        if attempts and attempts[-1].get('status') == 'no_progress':
            return stop('no_progress', '修复后原命令仍出现相同错误；停止重复尝试，保留诊断与失败日志。')
        if used >= MAX_DEPENDENCY_REPAIRS:
            return stop('budget_exhausted', '自修复已达到本任务共享的八次上限；保留所有失败与修复记录。')
        from src.intelligent_recovery import current_investigation
        investigation = current_investigation()
        if investigation:
            investigation.ensure_running()
            if investigation.state['usage'].get('repair_actions', 0) >= investigation.budget['max_repair_actions']:
                return stop('budget_exhausted', '共享修复动作已达上限，未启动新的修复。')
        local = repair_local_import(result, workspace, packet, plan['requirements'])
        if local and local['root'] not in effective.get('repository_pythonpath', []):
            effective.setdefault('repository_pythonpath', []).append(local['root'])
            import_repairs.append({**local, 'attempt': used + 1, 'failed_execution_dir': result.get('run_dir')})
            write_json(run_dir / 'import_repairs.json', import_repairs)
            write_json(run_dir / 'runtime_repair_environment.json', effective)
            record_repair(run_dir, data, on_event, result, local['action'], local['reason'])
            continue
        fix = repair_requirement(stderr, plan, packet)
        if not fix:
            pyg = repair_pyg(result) if repair_pyg else None
            if pyg is not None:
                pyg_repairs.append({**pyg, 'failed_execution_dir': result.get('run_dir'), 'attempt': used + 1})
                write_json(run_dir / 'pyg_repairs.json', pyg_repairs)
                on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_pyg_data_state',
                    'status': 'RUNNING', 'detail': '旧 PyG 数据对象缺少新版容器结构；保持原数据与划分，在内存中兼容后重试原命令。',
                    'data': pyg_repairs[-1]}})
                record_repair(run_dir, data, on_event, result, 'repair_pyg_data_state',
                              '根据实际错误兼容旧 PyG 容器，重跑原命令验证。')
                continue
            path_fix = repair_path(result) if repair_path else None
            if path_fix is not None:
                path_repairs.append({**path_fix, 'failed_execution_dir': result.get('run_dir'),
                    'attempt': used + 1})
                write_json(run_dir / 'path_repairs.json', path_repairs)
                on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_data_path',
                    'status': 'RUNNING', 'detail': '作者临时数据目录不存在；改用任务内目录并重试原命令。',
                    'data': path_repairs[-1]}})
                record_repair(run_dir, data, on_event, result, 'repair_data_path',
                              '修复任务内数据路径，重跑原命令验证。')
                continue
            numeric = repair_numeric(result) if repair_numeric else None
            if numeric is None:
                # Deterministic evidence-backed repairs take priority. Unknown
                # failures receive bounded source/log analysis only afterwards.
                suggestion = analyze_failure(result, effective) if analyze_failure else None
                if suggestion and suggestion.get('action') == 'replace_compatibility_pin':
                    current = effective.get('requirements_txt', '').splitlines()
                    old, replacement = suggestion['previous'], suggestion['requirement']
                    if current.count(old) == 1 and replacement not in current:
                        effective['requirements_txt'] = '\n'.join(replacement if line == old else line for line in current) + '\n'
                        effective.update(auto_prepare=True, require_prepared=False,
                                         exclude_environment_preparation_from_deadline=True)
                        model_repairs.append({**suggestion, 'attempt': used + 1,
                            'failed_execution_dir': result.get('run_dir')})
                        write_json(run_dir / 'model_dependency_repairs.json', model_repairs)
                        write_json(run_dir / 'runtime_repair_environment.json', effective)
                        record_repair(run_dir, data, on_event, result, suggestion['action'],
                                      f'根据实际解析器证据调整项目兼容依赖 {old} → {replacement}，重跑原命令验证。')
                        continue
                analyses = data.get('failure_analyses') or []
                reason = ((analyses[-1].get('reason') or analyses[-1].get('analysis_error')) if analyses else '')
                return stop('blocked', reason or '当前错误没有通过校验的自动修复方案；保留实际错误和源码依据。')
            numeric_repairs.append({**numeric, 'failed_execution_dir': result.get('run_dir'),
                                    'attempt': used + 1})
            write_json(run_dir / 'numeric_repairs.json', numeric_repairs)
            on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_numeric_type',
                'status': 'RUNNING', 'detail': '原命令发生 NumPy 兼容错误；记录兼容修复后重试同一命令。',
                'data': numeric_repairs[-1]}})
            record_repair(run_dir, data, on_event, result, 'repair_numeric_type',
                          '应用经过校验的 NumPy 兼容修复，重跑原命令验证。')
            continue
        if fix['package'] in seen:
            return stop('no_progress', '同一模块在依赖修复后仍缺失；停止重复安装，保留原错误。')
        seen.add(fix['package'])
        repair = {**fix, 'attempt': used + 1,
                  'failed_execution_dir': result.get('run_dir'),
                  'diagnostic': _safe_install_diagnostic(stderr)[-4000:]}
        repairs.append(repair)
        write_json(run_dir / 'dependency_repairs.json', data['dependency_repair_attempts'])
        on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_dependency',
                  'status': 'RUNNING', 'detail': f"缺少 {fix['module']}；安装 {fix['requirement']} 后重试原作者命令",
                  'data': repair}})
        existing = effective.get('requirements_txt', '')
        lines = existing.splitlines()
        for requirement in [*fix.get('requirement_options', []), *fix['additions']]:
            if requirement not in lines:
                lines.append(requirement)
        effective.update(requirements_txt='\n'.join(lines) + '\n', auto_prepare=True, require_prepared=False,
                         exclude_environment_preparation_from_deadline=True)
        effective.pop('deadline_monotonic', None)
        write_json(run_dir / 'runtime_repair_environment.json', effective)
        record_repair(run_dir, data, on_event, result, 'install_missing_dependency',
                      f"根据实际错误安装 {fix['requirement']}，重跑原命令验证。")
    return result
