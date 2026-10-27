"""Select an isolated interpreter for pinned author code, separately from the app."""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import sys

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from src.runtime_platform import runtime_fingerprint
from src.runtime_preparation import run_owned_process


SUPPORTED = ((3, 12), (3, 11), (3, 10), (3, 9))


class TrainingPythonUnavailable(RuntimeError):
    category = 'python_incompatible'


def _python_bounds(packet):
    """Read literal author declarations, never execute setup.py or source text."""
    declarations = []
    for name, source in packet['repository']['files'].items():
        text = source['text']
        if Path(name).name == 'pyproject.toml':
            try:
                import tomllib
                project = tomllib.loads(text).get('project') or {}
                if isinstance(project.get('requires-python'), str):
                    declarations.append((name, project['requires-python']))
            except (ValueError, ImportError):
                pass
        elif Path(name).name == 'setup.cfg':
            match = re.search(r'(?m)^\s*python_requires\s*=\s*([^\r\n#]+)', text)
            if match:
                declarations.append((name, match[1].strip()))
        elif Path(name).name == 'setup.py':
            try:
                for node in ast.walk(ast.parse(text)):
                    if (isinstance(node, ast.keyword) and node.arg == 'python_requires'
                            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                        declarations.append((name, node.value.value))
            except SyntaxError:
                pass
        elif Path(name).name.lower() in {'environment.yml', 'environment.yaml'}:
            match = re.search(r'(?mi)^\s*-\s*python\s*=\s*(3\.(?:9|10|11|12))(?:\s|$)', text)
            if match:
                major, minor = match[1].split('.')
                declarations.append((name, f'=={major}.{minor}.*'))
    bounds = []
    for name, declared in declarations:
        try:
            bounds.append((name, declared, SpecifierSet(declared)))
        except InvalidSpecifier:
            continue
    return bounds


def _probe(executable):
    try:
        outcome = run_owned_process([str(executable), '-I', '-c',
            'import json,sys; print(json.dumps([sys.implementation.name,*sys.version_info[:3],sys.maxsize>2**32]))'],
            timeout_s=15)
        if outcome.returncode:
            return None
        name, major, minor, micro, bits = json.loads(outcome.stdout.strip())
        version = (major, minor, micro)
        if name != 'cpython' or version[:2] not in SUPPORTED or not bits:
            return None
        return version
    except (OSError, ValueError, RuntimeError):
        return None


def _uv_command(root, *, offline):
    uv = shutil.which('uv')
    if uv:
        return [uv]
    target = root / 'tools' / 'uv'
    if not (target / '.ready').is_file():
        if offline:
            raise TrainingPythonUnavailable('离线模式下缺少 uv 解释器管理器缓存')
        from src.agents.code_executor import _public_install_sources, _safe_install_diagnostic
        target.mkdir(parents=True, exist_ok=True)
        failures = []
        for index, _ in _public_install_sources('uv==0.11.19'):
            command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                       '--no-user', '--upgrade', '--target', str(target), '-i', index, 'uv==0.11.19']
            from src.training_environment import training_process_environment
            result = run_owned_process(command, timeout_s=300, env=training_process_environment(root))
            if result.returncode == 0:
                (target / '.ready').write_text('uv==0.11.19\n', encoding='utf-8')
                break
            failures.append(_safe_install_diagnostic(result.stderr or result.stdout)[-300:])
        else:
            raise TrainingPythonUnavailable('自动准备 uv 失败：' + '; '.join(failures))
    binary = next((path for path in (target / 'bin/uv', target / 'bin/uv.exe',
                                   target / 'Scripts/uv.exe', target / 'uv/_uv')
                   if path.is_file()), None)
    if binary:
        return [str(binary)]
    raise TrainingPythonUnavailable('uv 缓存缺少可执行文件')


def _ensure_pip(executable, root):
    from src.training_environment import training_process_environment
    environment = training_process_environment(root)
    check = run_owned_process([str(executable), '-m', 'pip', '--version'], timeout_s=15, env=environment)
    if check.returncode == 0:
        return
    installed = run_owned_process([str(executable), '-m', 'ensurepip', '--upgrade'], timeout_s=120, env=environment)
    if installed.returncode or run_owned_process([str(executable), '-m', 'pip', '--version'],
                                                timeout_s=15, env=environment).returncode:
        raise TrainingPythonUnavailable('隔离训练解释器缺少 pip，自动准备失败')


def _managed_venv(root, base, version):
    from filelock import FileLock
    from src.training_environment import training_process_environment
    directory = root / 'envs' / ('cpython-' + '.'.join(map(str, version[:2])))
    executable = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.python.lock')):
        if _probe(executable) != version:
            if directory.exists():
                raise TrainingPythonUnavailable('已有隔离训练 Python 缓存不完整：' + str(directory))
            result = run_owned_process([str(base), '-m', 'venv', str(directory)], timeout_s=180,
                                       env=training_process_environment(root))
            if result.returncode or _probe(executable) != version:
                raise TrainingPythonUnavailable('创建隔离训练 Python 失败：' + (result.stderr or result.stdout)[-800:])
        _ensure_pip(executable, root)
    # Resolving a POSIX venv executable symlink would select the host Python.
    return executable.absolute()


def _family_can_match(family, bounds):
    """Check final CPython patches at every constraint boundary, not only x.y.0."""
    patches = {0}
    for _, _, specifications in bounds:
        for specification in specifications:
            try:
                boundary = Version(specification.version.removesuffix('.*'))
            except InvalidVersion:
                # An arbitrary equality that cannot describe a CPython release
                # is rejected by contains() below without guessing a version.
                continue
            if boundary.release[:2] == family:
                patch = boundary.release[2] if len(boundary.release) > 2 else 0
                patches.update({max(0, patch - 1), patch, patch + 1})
    return any(all(spec.contains('.'.join(map(str, (*family, patch))))
                   for _, _, spec in bounds) for patch in patches)


def select_author_python(packet, managed_root, *, offline=False, mirror=None,
                         exclude_versions=(), required_python=None,
                         required_python_source='actual_requested_dependency',
                         additional_python_requirements=()):
    """Reuse a matching Python; download one managed version only when needed."""
    bounds = _python_bounds(packet)
    requested = [(required_python_source, required_python)] if required_python else []
    requested.extend((entry['path'], entry['specifier']) for entry in additional_python_requirements)
    for source, requirement in requested:
        try:
            bounds.append((source, requirement, SpecifierSet(requirement)))
        except InvalidSpecifier as exc:
            raise TrainingPythonUnavailable('无法核验 Python 约束 ' + str(source) + '：' + requirement) from exc
    admissible = [version for version in SUPPORTED
                  if version not in exclude_versions and _family_can_match(version, bounds)]
    if not admissible:
        raise TrainingPythonUnavailable('作者 Python 声明与支持的 CPython 3.9–3.12 均不兼容：' +
                                        ', '.join(f'{name}: {value}' for name, value, _ in bounds))
    from src.training_environment import prepare_training_environment, training_process_environment
    root = prepare_training_environment(managed_root)

    def matches(version):
        # A minor family such as 3.12 cannot stand in for the actual patch
        # release when evaluating declarations such as >=3.12.1 or !=3.12.10.
        return (version is not None and version[:2] in admissible
                and all(spec.contains('.'.join(map(str, version))) for _, _, spec in bounds))

    choices = [Path(sys.executable)]
    choices.extend(Path(path) for version in admissible
                   if (path := shutil.which('python' + '.'.join(map(str, version)))))
    for version in admissible:
        choices.append(root / 'interpreters' / ('cpython-' + '.'.join(map(str, version))) /
                       ('python.exe' if os.name == 'nt' else 'bin/python3'))
    seen = set()
    for candidate in choices:
        if str(candidate) in seen or not candidate.is_file():
            continue
        seen.add(str(candidate))
        version = _probe(candidate)
        if matches(version):
            isolated = _managed_venv(root, candidate, version)
            return {'python_executable': str(isolated), 'python_version': '.'.join(map(str, version[:2])),
                    'runtime_fingerprint': runtime_fingerprint(isolated), 'source': 'isolated_existing',
                    'requirements': [{'path': name, 'specifier': value} for name, value, _ in bounds]}
    if offline:
        raise TrainingPythonUnavailable('离线模式缺少符合作者版本声明的训练用 Python；请预先准备隔离解释器')
    uv = _uv_command(root, offline=offline)
    # Let uv select an available patch release from the exact intersection;
    # testing only x.y.0 would discard otherwise compatible Python families.
    lowest, highest = min(SUPPORTED), max(SUPPORTED)
    constraints = [f'>={lowest[0]}.{lowest[1]}', f'<{highest[0]}.{highest[1] + 1}']
    constraints.extend(value for _, value, _ in bounds)
    constraints.extend(f'!={major}.{minor}.*' for major, minor in SUPPORTED if (major, minor) not in admissible)
    request = 'cpython@' + ','.join(constraints)
    install_dir = root / 'interpreters'
    command = [*uv, 'python', 'install', request,
               '--install-dir', str(install_dir), '--no-bin', '--no-registry']
    mirror = mirror or os.environ.get('AUTOREPRO_PYTHON_MIRROR', '').strip()
    if mirror:
        command += ['--mirror', mirror]
    result = run_owned_process(command, timeout_s=1800, env=training_process_environment(root))
    if result.returncode:
        raise TrainingPythonUnavailable('自动准备训练用 Python 失败：' + (result.stderr or result.stdout)[-1200:])
    from src.runtime_preparation import safe_runtime_diagnostic
    candidates = list(install_dir.glob('cpython-*/bin/python3')) + list(install_dir.glob('cpython-*/python.exe'))
    for candidate in candidates:
        version = _probe(candidate)
        if matches(version):
            _ensure_pip(candidate, root)
            return {'python_executable': str(candidate.absolute()), 'python_version': '.'.join(map(str, version[:2])),
                    'runtime_fingerprint': runtime_fingerprint(candidate), 'source': 'uv_managed',
                    'requirements': [{'path': name, 'specifier': value} for name, value, _ in bounds]}
    raise TrainingPythonUnavailable(safe_runtime_diagnostic('uv did not provide the selected CPython version'))


def retry_requested_python_conflict(packet, managed_root, environment, selected_python,
                                    preparation, run_preflight, *, offline=False, on_switch=None):
    """Retry the unchanged preflight only for a proven requested-package conflict."""
    from src.agents.code_executor import _requested_python_conflict
    from src.intelligent_recovery import current_investigation
    from src.repository_repairer import repair_is_stopped
    attempted = {tuple(int(part) for part in selected_python['python_version'].split('.'))}
    while preparation.get('success') is not True:
        # A cancelled installer or unconfirmed process cleanup is not evidence
        # authorizing another interpreter and another installation process.
        if repair_is_stopped(preparation):
            session = current_investigation()
            if session:
                session.record_execution(preparation)
                session.ensure_running()
            break
        effective = preparation.get('effective_env_config') or {}
        if effective.get('dependency_failure_kind') != 'python_incompatible':
            break
        conflict = None
        for item in reversed(effective.get('dependency_install_attempts') or []):
            if item.get('failure_kind') == 'python_incompatible':
                conflict = _requested_python_conflict(item.get('diagnostic', ''),
                    effective.get('requirements_txt', ''))
                if conflict:
                    break
        if not conflict:
            break
        def switch_and_verify():
            replacement = select_author_python(packet, managed_root, offline=offline,
                exclude_versions=set(attempted), required_python=conflict.get('requires_python'),
                additional_python_requirements=([{'path': environment.get('required_python_source',
                    'actual_requested_dependency'), 'specifier': environment['required_python']}]
                    if environment.get('required_python') else []))
            attempted.add(tuple(int(part) for part in replacement['python_version'].split('.')))
            replacement['selected_for'] = conflict
            environment.update(replacement)
            if on_switch:
                on_switch(replacement)
            return run_preflight(environment)

        session = current_investigation()
        if session:
            with session.phase('environment_preparation'):
                preparation = session.run_action('select_compatible_author_python', switch_and_verify,
                    problem='Requested author dependency is incompatible with the selected Python',
                    evidence={'conflict': conflict, 'python_version': environment.get('python_version'),
                              'requirements_txt': effective.get('requirements_txt'),
                              'repository': packet.get('repository', {}).get('revision')},
                    role='EnvBuilder', review={'status': 'accepted',
                        'reviewer': 'requested_dependency_rules',
                        'reason': 'A real installer diagnostic proves the interpreter constraint; author requirements remain unchanged.'},
                    verify=lambda result: {'success': result.get('success') is True,
                        'reason': 'Unchanged native preflight rerun under the selected interpreter',
                        'evidence': {'execution_dir': result.get('run_dir'),
                                     'cleanup_confirmed': result.get('cleanup_confirmed')}})
                session.record_execution(preparation)
                session.ensure_running()
        else:
            preparation = switch_and_verify()
    return preparation
