"""Error-driven dependency retries for the explicit author execution objective."""
from copy import deepcopy
import re
import sys
import time

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
                               run_dir, data, on_event, verify_inputs, repair_numeric=None):
    """Retry after cleanup, retaining every failed run and the unchanged argv."""
    effective = deepcopy(environment)
    data.setdefault('dependency_repair_attempts', [])
    repairs = data['dependency_repair_attempts']
    numeric_repairs = data.setdefault('numeric_repair_attempts', [])
    seen = {repair['package'] for repair in repairs}
    for attempt in range(MAX_DEPENDENCY_REPAIRS + 1):
        verify_inputs()
        result = runner.run(workspace, steps, effective, on_event=on_event)
        effective = deepcopy(result.get('effective_env_config') or effective)
        result.setdefault('effective_env_config', effective)
        if result.get('success') or result.get('cancelled') or result.get('interrupted'):
            return result
        final = result.get('final') or {}
        if final.get('timed_out') or final.get('cancelled'):
            return result
        stderr = str(final.get('stderr') or result.get('reason') or '')
        if len(repairs) + len(numeric_repairs) >= MAX_DEPENDENCY_REPAIRS:
            return result
        fix = repair_requirement(stderr, plan, packet)
        if not fix:
            numeric = repair_numeric(result) if repair_numeric else None
            if numeric is None:
                return result
            numeric_repairs.append({**numeric, 'failed_execution_dir': result.get('run_dir'),
                                    'attempt': len(repairs) + len(numeric_repairs) + 1})
            write_json(run_dir / 'numeric_repairs.json', numeric_repairs)
            on_event({'type': 'log', 'log': {'agent': 'CodeExecutor', 'action': 'repair_numeric_type',
                'status': 'RUNNING', 'detail': '原命令发生 NumPy 兼容错误；记录兼容修复后重试同一命令。',
                'data': numeric_repairs[-1]}})
            if effective.get('deadline_monotonic') is not None:
                raw_steps = steps.get('steps', []) if isinstance(steps, dict) else steps
                effective['deadline_monotonic'] = time.monotonic() + sum(step['timeout_s'] for step in raw_steps) + 2
            continue
        if fix['package'] in seen:
            return result
        seen.add(fix['package'])
        repair = {**fix, 'attempt': len(repairs) + 1, 'failed_execution_dir': result.get('run_dir'),
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
        if effective.get('deadline_monotonic') is not None:
            raw_steps = steps.get('steps', []) if isinstance(steps, dict) else steps
            effective['deadline_monotonic'] = time.monotonic() + sum(step['timeout_s'] for step in raw_steps) + 2
        write_json(run_dir / 'runtime_repair_environment.json', effective)
    return result
