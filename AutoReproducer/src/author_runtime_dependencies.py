"""Error-driven dependency retries for the explicit author execution objective."""
from copy import deepcopy
import re
import sys

from packaging.requirements import InvalidRequirement, Requirement

from src.agents.code_executor import _safe_install_diagnostic
from src.agents.dependency_resolver import find_missing_module, python_package_for
from src.discovered_repository_plan import GRAPH_WHEEL_LINKS, _PACKAGE_IMPORTS, _RUNTIME_DEPENDENCIES
from src.dependency_versions import compatibility_pins
from src.method_adapters import write_json

MAX_DEPENDENCY_REPAIRS = 8


def repair_requirement(stderr, plan, packet):
    # Missing symbols in an installed package indicate an API/version problem,
    # not an absent dependency. Do not download a different package blindly.
    if not re.search(r'(?:ModuleNotFoundError|ImportError):\s*No module named', stderr or ''):
        return None
    module = find_missing_module(stderr)
    if not module or module in sys.stdlib_module_names:
        return None
    inventory = packet['file_inventory']
    if (module + '.py' in inventory or any(name.startswith(module + '/') for name in inventory)
            or any(item['name'] == module for item in plan['requirements'].get('repository_namespaces', []))):
        return None
    inverse = {value: key for key, value in _PACKAGE_IMPORTS.items()}
    package = inverse.get(module, python_package_for(module)).lower()
    normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
    declared = None
    for line in plan['requirements'].get('author', []):
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        if normalize(requirement.name) == normalize(package) and not requirement.url:
            declared = str(requirement)
            break
    pins = compatibility_pins(plan['capture']['device'])
    # Known native pairs retain their compatible pins; other packages prefer
    # the author's requirement. Undeclared dependencies are installed only
    # after their genuine runtime error, then cache health checks their import.
    requirement = package + '==' + pins[package] if package in pins else declared or package
    additions = [requirement]
    for dependency in _RUNTIME_DEPENDENCIES.get(package, ()):
        additions.append(dependency + '==' + pins[dependency])
    options = (["--find-links " + GRAPH_WHEEL_LINKS[plan['capture']['device']]]
               if any(item.split('==')[0] in {'torch-scatter', 'torch-sparse'} for item in additions)
               else [])
    return {'module': module, 'package': package, 'requirement': requirement,
            'additions': additions, 'requirement_options': options, 'author_declaration': declared,
            'source': 'actual_missing_module_error'}


def run_with_dependency_repair(runner, workspace, steps, environment, plan, packet,
                               run_dir, data, on_event, verify_inputs, repair_numeric=None, repair_path=None,
                               analyze_failure=None):
    """Retry after cleanup, retaining every failed run and the unchanged argv."""
    effective = deepcopy(environment)
    # Retry allowance counts actual failures, never time spent running code.
    effective.pop('deadline_monotonic', None)
    data.setdefault('dependency_repair_attempts', [])
    repairs = data['dependency_repair_attempts']
    numeric_repairs = data.setdefault('numeric_repair_attempts', [])
    path_repairs = data.setdefault('path_repair_attempts', [])
    model_repairs = data.setdefault('model_dependency_repairs', [])
    seen = {repair['package'] for repair in repairs}
    for attempt in range(MAX_DEPENDENCY_REPAIRS + 1):
        verify_inputs()
        result = runner.run(workspace, steps, effective, on_event=on_event)
        effective = deepcopy(result.get('effective_env_config') or effective)
        result.setdefault('effective_env_config', effective)
        if result.get('success') or result.get('cancelled') or result.get('interrupted'):
            return result
        if effective.get('dependency_failure_kind') == 'installation_timeout':
            # A killed installer is not evidence of a missing module or a bad
            # version pin. Retrying a model suggestion would restart downloads.
            return result
        final = result.get('final') or {}
        if final.get('timed_out') or final.get('cancelled'):
            return result
        stderr = str(final.get('stderr') or result.get('reason') or '')
        used = len(repairs) + len(numeric_repairs) + len(path_repairs) + len(model_repairs)
        suggestion = analyze_failure(result, effective) if analyze_failure else None
        if used >= MAX_DEPENDENCY_REPAIRS:
            return result
        if suggestion and suggestion.get('action') == 'replace_compatibility_pin':
            current = effective.get('requirements_txt', '').splitlines()
            old, replacement = suggestion['previous'], suggestion['requirement']
            if current.count(old) == 1 and replacement not in current:
                effective['requirements_txt'] = '\n'.join(replacement if line == old else line for line in current) + '\n'
                effective.update(auto_prepare=True, require_prepared=False,
                                 exclude_environment_preparation_from_deadline=True)
                effective.pop('deadline_monotonic', None)
                model_repairs.append({**suggestion, 'attempt': used + 1,
                    'failed_execution_dir': result.get('run_dir')})
                write_json(run_dir / 'model_dependency_repairs.json', model_repairs)
                write_json(run_dir / 'runtime_repair_environment.json', effective)
                on_event({'type': 'log', 'log': {'agent': 'CodeExecutor',
                    'action': 'repair_derived_dependency', 'status': 'RUNNING',
                    'detail': f'根据失败分析验证项目推导依赖 {old} → {replacement}；保留作者版本与原命令',
                    'data': model_repairs[-1]}})
                continue
        fix = repair_requirement(stderr, plan, packet)
        if not fix:
            path_fix = repair_path(result) if repair_path else None
            if path_fix is not None:
                path_repairs.append({**path_fix, 'failed_execution_dir': result.get('run_dir'),
                    'attempt': used + 1})
                write_json(run_dir / 'path_repairs.json', path_repairs)
                on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_data_path',
                    'status': 'RUNNING', 'detail': '作者临时数据目录不存在；改用任务内目录并重试原命令。',
                    'data': path_repairs[-1]}})
                continue
            numeric = repair_numeric(result) if repair_numeric else None
            if numeric is None:
                return result
            numeric_repairs.append({**numeric, 'failed_execution_dir': result.get('run_dir'),
                                    'attempt': used + 1})
            write_json(run_dir / 'numeric_repairs.json', numeric_repairs)
            on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_numeric_type',
                'status': 'RUNNING', 'detail': '原命令发生 NumPy 兼容错误；记录兼容修复后重试同一命令。',
                'data': numeric_repairs[-1]}})
            continue
        if fix['package'] in seen:
            return result
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
    return result
