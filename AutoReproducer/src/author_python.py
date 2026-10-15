"""Select an isolated interpreter for pinned author code, separately from the app."""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import sys

from packaging.specifiers import InvalidSpecifier, SpecifierSet

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
            'import json,sys; print(json.dumps([sys.implementation.name,*sys.version_info[:2],sys.maxsize>2**32]))'],
            timeout_s=15)
        if outcome.returncode:
            return None
        name, major, minor, bits = json.loads(outcome.stdout.strip())
        version = (major, minor)
        if name != 'cpython' or version not in SUPPORTED or not bits:
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
    binary = next((path for path in (target / 'bin/uv', target / 'Scripts/uv.exe', target / 'uv/_uv')
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
    directory = root / 'envs' / ('cpython-' + '.'.join(map(str, version)))
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


def select_author_python(packet, managed_root, *, offline=False, mirror=None,
                         exclude_versions=(), required_python=None):
    """Reuse a matching Python; download one managed version only when needed."""
    bounds = _python_bounds(packet)
    if required_python:
        try:
            bounds.append(('actual_requested_dependency', required_python, SpecifierSet(required_python)))
        except InvalidSpecifier as exc:
            raise TrainingPythonUnavailable('无法核验所请求依赖的 Requires-Python 约束：' + required_python) from exc
    admissible = [version for version in SUPPORTED
                  if version not in exclude_versions
                  and all(spec.contains('.'.join(map(str, version)) + '.0') for _, _, spec in bounds)]
    if not admissible:
        raise TrainingPythonUnavailable('作者 Python 声明与支持的 CPython 3.9–3.12 均不兼容：' +
                                        ', '.join(f'{name}: {value}' for name, value, _ in bounds))
    from src.training_environment import prepare_training_environment, training_process_environment
    root = prepare_training_environment(managed_root)
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
        if version in admissible:
            isolated = _managed_venv(root, candidate, version)
            return {'python_executable': str(isolated), 'python_version': '.'.join(map(str, version)),
                    'runtime_fingerprint': runtime_fingerprint(isolated), 'source': 'isolated_existing',
                    'requirements': [{'path': name, 'specifier': value} for name, value, _ in bounds]}
    if offline:
        raise TrainingPythonUnavailable('离线模式缺少符合作者版本声明的训练用 Python；请预先准备隔离解释器')
    uv = _uv_command(root, offline=offline)
    version = admissible[0]
    install_dir = root / 'interpreters'
    command = [*uv, 'python', 'install', '.'.join(map(str, version)),
               '--install-dir', str(install_dir), '--no-bin']
    mirror = mirror or os.environ.get('AUTOREPRO_PYTHON_MIRROR', '').strip()
    if mirror:
        command += ['--mirror', mirror]
    result = run_owned_process(command, timeout_s=1800, env=training_process_environment(root))
    if result.returncode:
        raise TrainingPythonUnavailable('自动准备训练用 Python 失败：' + (result.stderr or result.stdout)[-1200:])
    from src.runtime_preparation import safe_runtime_diagnostic
    candidates = list(install_dir.glob('cpython-*/bin/python3')) + list(install_dir.glob('cpython-*/python.exe'))
    for candidate in candidates:
        if _probe(candidate) == version:
            _ensure_pip(candidate, root)
            return {'python_executable': str(candidate.absolute()), 'python_version': '.'.join(map(str, version)),
                    'runtime_fingerprint': runtime_fingerprint(candidate), 'source': 'uv_managed',
                    'requirements': [{'path': name, 'specifier': value} for name, value, _ in bounds]}
    raise TrainingPythonUnavailable(safe_runtime_diagnostic('uv did not provide the selected CPython version'))


def retry_requested_python_conflict(packet, managed_root, environment, selected_python,
                                    preparation, run_preflight, *, offline=False, on_switch=None):
    """Retry the unchanged preflight only for a proven requested-package conflict."""
    from src.agents.code_executor import _requested_python_conflict
    attempted = {tuple(int(part) for part in selected_python['python_version'].split('.'))}
    while preparation.get('success') is not True:
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
        replacement = select_author_python(packet, managed_root, offline=offline,
            exclude_versions=set(attempted), required_python=conflict.get('requires_python'))
        attempted.add(tuple(int(part) for part in replacement['python_version'].split('.')))
        replacement['selected_for'] = conflict
        environment.update(replacement)
        if on_switch:
            on_switch(replacement)
        preparation = run_preflight(environment)
    return preparation
