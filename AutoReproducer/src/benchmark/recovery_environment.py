"""Trusted missing-dependency adapter and independent environment contract.

Only stdlib is imported at module load: the frozen driver may load this file
with runpy before choosing an old product checkout which lacks this module.
Production policy imports deliberately occur inside run_environment_recovery.
"""
from copy import deepcopy
from email.parser import BytesParser
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


_OBSERVATIONS = {
    'dependency_install_attempts', 'dependency_health_attempts', 'dependency_cache_repairs',
    'dependency_cache_maintenance', 'dependency_temporary_files', 'dependency_reuse',
    'dependency_layer_fallbacks', 'dependency_failure_kind',
}
_REPAIR_CONTROLS = {'requirements_txt', 'auto_prepare', 'require_prepared',
                    'exclude_environment_preparation_from_deadline'}
_BASELINE_KEY = '_recovery_environment_baseline'
_SNAPSHOT_MAX_FILES = 25000
_SNAPSHOT_MAX_BYTES = 512 * 1024 * 1024


class UnsupportedEnvironmentRepair(ValueError):
    """The selected product proposed a change outside the registered envelope."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())


def _name(value):
    return re.sub(r'[-_.]+', '-', value).lower()


def _pin(value):
    from packaging.requirements import Requirement
    from packaging.version import Version
    if not isinstance(value, str):
        raise ValueError('Runtime dependency must be an exact public version pin')
    parsed = Requirement(value)
    specs = list(parsed.specifier)
    if (parsed.url or parsed.marker or parsed.extras or len(specs) != 1
            or specs[0].operator != '==' or '*' in specs[0].version):
        raise ValueError('Runtime dependency must be an exact public version pin')
    version = str(Version(specs[0].version))
    return _name(parsed.name) + '==' + version


def _path(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or any(p in {'', '..'} for p in relative.split('/'))):
        raise ValueError('Expected a safe relative environment input')
    rel = PurePosixPath(relative)
    if rel.is_absolute():
        raise ValueError('Absolute environment inputs are unsupported')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*rel.parts)
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink() or getattr(parent, 'is_junction', lambda: False)():
            raise ValueError('Environment input links are unsupported')
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('Environment input escaped its root')
    return path


def _initial_pins(evidence):
    environment = evidence.get('environment') or {}
    requirements = environment.get('requirements_txt') or ''
    packages = environment.get('required_packages') or []
    if not isinstance(requirements, str) or not isinstance(packages, list):
        raise ValueError('Invalid initial runtime dependency list')
    pins = [_pin(line.strip()) for line in requirements.splitlines() if line.strip()]
    pins.extend(_pin(line) for line in packages)
    if len({pin.split('==')[0] for pin in pins}) != len(pins):
        raise ValueError('Initial runtime dependencies contain duplicate distributions')
    return pins


def validate_environment_evidence(case, workspace, evidence):
    """Validate a preregistered missing-module envelope; never infer new rights."""
    if case.get('category') != 'runtime_environment':
        raise ValueError('Expected runtime_environment category')
    raw = evidence.get('runtime_environment')
    if (not isinstance(raw, dict) or raw.get('version') != 1
            or set(raw) - {'version', 'allowed_repairs', 'development_wheels'}):
        raise ValueError('Preregister runtime_environment version 1 and its allowed repairs')
    if evidence.get('editable'):
        raise ValueError('Runtime environment recovery cannot authorize source edits')
    failure = evidence.get('initial_failure') or {}
    if failure.get('error_type') != 'ModuleNotFoundError':
        raise ValueError('Only missing-module environment failures are currently supported')
    repairs = raw.get('allowed_repairs')
    if not isinstance(repairs, list) or not 1 <= len(repairs) <= 8:
        raise ValueError('Register one to eight missing dependency repairs')
    allowed, modules, versions = [], set(), {}
    initial = _initial_pins(evidence)
    for pin in initial:
        name, version = pin.split('==')
        versions[name] = version
    for repair in repairs:
        if (not isinstance(repair, dict) or set(repair) - {'module', 'requirement', 'additions'}
                or not re.fullmatch(r'[A-Za-z_]\w*', str(repair.get('module', '')))
                or repair['module'] in modules):
            raise ValueError('Each allowed dependency needs a unique top-level missing module')
        requirement = _pin(repair.get('requirement'))
        additions = repair.get('additions', [requirement])
        if not isinstance(additions, list) or not additions or len(additions) > 32:
            raise ValueError('Register the exact dependency additions')
        additions = [_pin(pin) for pin in additions]
        if requirement not in additions or len(set(additions)) != len(additions):
            raise ValueError('The primary dependency must occur once in its registered additions')
        for pin in additions:
            name, version = pin.split('==')
            if name in versions and versions[name] != version:
                raise ValueError('Runtime recovery cannot replace an existing dependency version')
            versions[name] = version
        modules.add(repair['module'])
        allowed.append({'module': repair['module'], 'requirement': requirement, 'additions': additions})
    wheels = raw.get('development_wheels', [])
    if not isinstance(wheels, list) or len(wheels) > 64:
        raise ValueError('Invalid development wheel list')
    if wheels and (case.get('split') != 'dev' or case.get('origin') != 'injected'):
        raise ValueError('Development wheels are restricted to injected development cases')
    names, checked = set(), []
    inventory = case.get('source_files') or {}
    for record in wheels:
        if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
            raise ValueError('Development wheels need an inventory path and SHA-256')
        path = _path(workspace, record['path'])
        if (path.suffix != '.whl' or record['sha256'] != inventory.get(record['path'])
                or _sha(path.read_bytes()) != record['sha256'] or path.name.casefold() in names):
            raise ValueError('Development wheel bytes must match the frozen source inventory')
        names.add(path.name.casefold())
        checked.append(deepcopy(record))
    return {'version': 1, 'initial_pins': initial, 'allowed_repairs': allowed, 'development_wheels': checked}


def _wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if (len(names) > 4096 or len({name.casefold() for name in names}) != len(names)
                or sum(item.file_size for item in members) > 128 * 1024 * 1024):
            raise ValueError('Development wheel exceeds its archive envelope')
        for item in members:
            name = item.filename.rstrip('/')
            if (not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()
                    or any(part in {'', '.', '..'} for part in name.split('/'))
                    or stat.S_ISLNK(item.external_attr >> 16) or '.data/' in item.filename
                    or item.filename.endswith('.pth')):
                raise ValueError('Development wheels require ordinary contained package files')
        candidates = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(candidates) != 1:
            raise ValueError('Development wheel needs one distribution metadata record')
        meta = BytesParser().parsebytes(archive.read(candidates[0]))
        return _pin(meta['Name'] + '==' + meta['Version']), meta


def _transport(contract, workspace, trial_environment, *, create=True):
    wheels = contract.get('development_wheels') or []
    if not wheels:
        return []
    root = Path(trial_environment).resolve(strict=True)
    destination = root / 'wheelhouse'
    if destination.is_symlink() or getattr(destination, 'is_junction', lambda: False)():
        raise ValueError('Trial wheelhouse cannot be a link')
    creating = not destination.exists()
    if creating:
        if not create:
            raise ValueError('Registered trial wheelhouse is missing')
        destination.mkdir()
    registered = set(contract['initial_pins'])
    for repair in contract['allowed_repairs']:
        registered.update(repair['additions'])
    expected = set()
    for record in wheels:
        original = _path(workspace, record['path'])
        raw = original.read_bytes()
        if _sha(raw) != record['sha256']:
            raise ValueError('Frozen development wheel changed')
        pin, _ = _wheel_metadata(original)
        if pin not in registered:
            raise ValueError('A development wheel is outside the registered dependency set')
        target = destination / original.name
        expected.add(original.name)
        if target.exists():
            if _path(root, 'wheelhouse/' + original.name).read_bytes() != raw:
                raise ValueError('Staged development wheel changed')
        elif creating:
            target.write_bytes(raw)
        else:
            raise ValueError('Staged development wheel is missing')
    if {path.name for path in destination.iterdir()} != expected:
        raise ValueError('Trial wheelhouse contains unregistered inputs')
    # Requirements files use shlex parsing: raw Windows backslashes disappear.
    # A locally reconstructed file URI also handles spaces without extra flags.
    return ['--no-index', '--find-links ' + destination.as_uri()]


def _environment_pins(environment, prefix):
    text = environment.get('requirements_txt') or ''
    if not isinstance(text, str):
        raise ValueError('Invalid effective requirements')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and prefix:
        if lines[:len(prefix)] != prefix:
            raise ValueError('Runtime dependency transport options changed')
        lines = lines[len(prefix):]
    if any(line.startswith('-') for line in lines):
        raise ValueError('Unregistered runtime install options or indexes')
    if not lines:
        if prefix and text.strip():
            raise ValueError('Transport options cannot install an empty dependency set')
        lines = environment.get('required_packages') or []
    pins = [_pin(line) for line in lines]
    if len({pin.split('==')[0] for pin in pins}) != len(pins):
        raise ValueError('Effective dependency list contains duplicate distributions')
    return pins


def _file_identity(path):
    """Hash ordinary files without accepting a changing or special input."""
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Trial interpreter snapshot requires ordinary files')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > _SNAPSHOT_MAX_BYTES:
                raise ValueError('Trial interpreter snapshot exceeds its byte limit')
            digest.update(chunk)
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
            (after.st_size, after.st_mtime_ns, after.st_ino) or size != after.st_size):
        raise ValueError('Trial interpreter file changed while being inspected')
    return {'sha256': digest.hexdigest(), 'size': size, 'mode': stat.S_IMODE(after.st_mode)}


def _venv_config(path):
    return dict(line.split(' = ', 1) for line in path.read_text(encoding='utf-8').splitlines()
                if ' = ' in line)


def _interpreter_chain(base):
    """Windows can create a venv from another venv's redirecting launcher."""
    records, seen = [], set()
    for _ in range(8):
        if not base.is_absolute() or not base.is_file() or str(base) in seen:
            raise ValueError('Fresh trial venv has an invalid base interpreter chain')
        seen.add(str(base))
        record = {'path': str(base), 'resolved': str(base.resolve(strict=True)), **_file_identity(base)}
        config = base.parent.parent / 'pyvenv.cfg'
        if base.parent.name.lower() not in {'scripts', 'bin'} or not config.is_file():
            records.append(record)
            return records
        record['venv_config'] = {'path': str(config), **_file_identity(config)}
        records.append(record)
        value = _venv_config(config).get('executable')
        if not value:
            raise ValueError('Base venv must identify its actual interpreter')
        next_base = Path(value)
        if next_base == base:
            raise ValueError('Base venv interpreter chain cannot refer to itself')
        base = next_base
    raise ValueError('Fresh trial venv interpreter chain exceeds its depth limit')


def _runtime_snapshot(trial_environment, environment):
    """Bind fresh-venv bytes and base interpreter without changing its launcher."""
    root = Path(trial_environment).resolve(strict=True)
    venv = root / 'author-venv'
    python = Path(os.path.abspath(environment.get('python_executable') or ''))
    expected = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if python != expected:
        raise ValueError('Author interpreter must use the registered fresh venv launcher')
    for path in (venv, python.parent):
        if (not path.is_dir() or path.is_symlink()
                or getattr(path, 'is_junction', lambda: False)()):
            raise ValueError('Trial venv directories cannot be filesystem links')
    entries, total = {}, 0
    pending = [venv]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(venv).as_posix()
            if len(entries) >= _SNAPSHOT_MAX_FILES:
                raise ValueError('Trial interpreter snapshot exceeds its file limit')
            if getattr(path, 'is_junction', lambda: False)():
                raise ValueError('Trial venv cannot contain directory junctions')
            record = path.lstat()
            if stat.S_ISLNK(record.st_mode):
                target = path.resolve(strict=True)
                # CPython venvs commonly link bin/python to a system binary.
                # Other links must stay inside the venv; never walk them.
                launcher = (relative.startswith('bin/') and '/' not in relative[4:]
                            and re.fullmatch(r'python(?:\d+(?:\.\d+)?)?', path.name))
                if not target.is_relative_to(venv) and not (launcher and target.is_file()):
                    raise ValueError('Trial venv contains an unregistered external link')
                entries[relative] = {'kind': 'link', 'target': os.readlink(path),
                                     'resolved': str(target)}
                if launcher:
                    entries[relative]['interpreter'] = _file_identity(target)
            elif stat.S_ISDIR(record.st_mode):
                entries[relative] = {'kind': 'directory', 'mode': stat.S_IMODE(record.st_mode)}
                pending.append(path)
            elif stat.S_ISREG(record.st_mode):
                identity = _file_identity(path)
                total += identity['size']
                if total > _SNAPSHOT_MAX_BYTES:
                    raise ValueError('Trial interpreter snapshot exceeds its byte limit')
                entries[relative] = {'kind': 'file', **identity}
            else:
                raise ValueError('Trial venv contains an unsupported special file')
    config = venv / 'pyvenv.cfg'
    if 'pyvenv.cfg' not in entries or not python.is_file():
        raise ValueError('Fresh trial venv is missing its interpreter or configuration')
    values = _venv_config(config)
    if values.get('include-system-site-packages', '').lower() != 'false':
        raise ValueError('Fresh trial venv cannot enable system site packages')
    base = Path(values.get('executable') or '')
    if not base.is_absolute() or not base.is_file():
        raise ValueError('Fresh trial venv must identify its actual base interpreter')
    return {'version': 1, 'python_executable': str(python), 'venv': str(venv),
            'inventory_sha256': _digest(entries), 'entries': len(entries), 'bytes': total,
            'base_interpreters': _interpreter_chain(base)}


def _verify_runtime_snapshot(trial_environment, environment):
    baseline = environment.get(_BASELINE_KEY)
    if not isinstance(baseline, dict) or _runtime_snapshot(trial_environment, environment) != baseline:
        raise ValueError('Fresh trial venv or interpreter bytes changed')


def prepare_environment_runtime(contract, workspace, trial_environment, environment):
    """Apply fixed offline transport, identically for all arms, before first run."""
    effective = deepcopy(environment)
    prefix = _transport(contract, workspace, trial_environment)
    text = effective.get('requirements_txt') or ''
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if prefix and lines[:len(prefix)] == prefix:
        pins = _environment_pins(effective, prefix)
    else:
        pins = _environment_pins(effective, [])
    if pins:
        effective['requirements_txt'] = '\n'.join([*prefix, *pins]) + '\n'
    if _BASELINE_KEY in effective:
        _verify_runtime_snapshot(trial_environment, effective)
    else:
        effective[_BASELINE_KEY] = _runtime_snapshot(trial_environment, effective)
    return effective


def _controls(environment):
    return {key: deepcopy(value) for key, value in environment.items()
            if key not in _OBSERVATIONS | _REPAIR_CONTROLS}


def _check_environment(actual, baseline, pins, prefix, *, repaired):
    if _controls(actual) != _controls(baseline):
        raise ValueError('Protected runtime environment controls changed')
    if _environment_pins(actual, prefix) != pins:
        raise ValueError('Effective dependencies differ from the registered repair sequence')
    flags = {'auto_prepare': True, 'require_prepared': False,
             'exclude_environment_preparation_from_deadline': True} if repaired else {
                 key: baseline.get(key) for key in _REPAIR_CONTROLS - {'requirements_txt'}}
    if any(actual.get(key) != value for key, value in flags.items()):
        raise ValueError('Runtime preparation controls differ from the registered transition')


def _source_files(workspace, packet):
    for name, record in packet['file_inventory'].items():
        if _sha(_path(workspace, name).read_bytes()) != record['sha256']:
            raise ValueError('Frozen author input changed during environment recovery')


def _failure_module(diagnostic, workspace, packet):
    """Independent literal-import check; do not reuse the product's proposal."""
    traces = list(re.finditer(r'(?m)^Traceback \(most recent call last\):\s*$', diagnostic))
    if not traces:
        raise UnsupportedEnvironmentRepair('A real original traceback is required')
    trace = diagnostic[traces[-1].start():]
    errors = list(re.finditer(r'(?m)^([\w.]+(?:Error|Exception|Interrupt|Exit)):[^\n]*', trace))
    if not errors:
        raise UnsupportedEnvironmentRepair('Terminal missing-module evidence is absent')
    match = re.fullmatch(r"ModuleNotFoundError: No module named ['\"]([A-Za-z_]\w*)['\"]", errors[-1].group().strip())
    if not match:
        raise UnsupportedEnvironmentRepair('Only terminal top-level missing-module errors are supported')
    module = match[1]
    importing = []
    root = Path(workspace).resolve(strict=True)
    for filename, line in re.findall(r'File "([^"\n]+)", line (\d+)', trace):
        try:
            name = Path(filename).resolve().relative_to(root).as_posix()
            if name not in packet['file_inventory'] or not name.endswith('.py'):
                continue
            raw = _path(root, name).read_bytes()
            if _sha(raw) != packet['file_inventory'][name]['sha256']:
                raise ValueError('Frozen importing source changed')
            location = int(line)
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno <= location <= getattr(node, 'end_lineno', node.lineno):
                    if (isinstance(node, ast.Import) and any(alias.name.split('.')[0] == module for alias in node.names)
                            or isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                            and node.module.split('.')[0] == module):
                        importing.append(name)
        except (OSError, SyntaxError):
            continue
        except ValueError:
            if Path(filename).resolve().is_relative_to(root):
                raise
    if not importing:
        raise UnsupportedEnvironmentRepair('Missing dependency needs a literal import in original frozen source')
    return module, sorted(set(importing))


def _descriptor(fix):
    if fix.get('requirement_options'):
        raise UnsupportedEnvironmentRepair('Native wheel/index options are not supported by this category adapter')
    return {'module': fix.get('module'), 'requirement': _pin(fix.get('requirement')),
            'additions': [_pin(pin) for pin in fix.get('additions', [])]}


def _unclean(execution):
    pending, seen, diagnostics = [(execution, False)], set(), []
    while pending:
        record, receipt = pending.pop()
        if isinstance(record, (list, dict)):
            identity = (id(record), receipt)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(record, list):
            pending.extend((item, receipt) for item in record)
        elif isinstance(record, dict):
            if (record.get('process_cleanup_confirmed') is False or record.get('cleanup_confirmed') is False
                    or any(record.get(key) for key in ('cleanup_failed', 'cleanup_unconfirmed', 'cleanup_error', 'cleanup_errors'))
                    or record.get('status') in {'cleanup_failed', 'cleanup_unconfirmed'}
                    or receipt and (record.get('confirmed') is False or record.get('status') in {'failed', 'unconfirmed'})):
                return True
            diagnostics.extend(str(record.get(key) or '') for key in ('reason', 'stdout', 'stderr'))
            pending.extend((record.get(key), True) for key in ('cleanup', 'process_cleanup', 'cleanup_receipt'))
            pending.extend((record.get(key), False) for key in ('final', 'steps', 'attempts', 'stages', 'environment', 'effective_env_config'))
    return bool(re.search(r'process[- ](?:tree|group) cleanup (?:failed|was denied)|\bcleanup_unconfirmed\b',
                          '\n'.join(diagnostics), re.I))


def _runtime(execution):
    observed = execution.get('environment') or {}
    return {key: observed.get(key) for key in ('executable', 'runtime')}


def record_environment_execution(execution, workspace, trial_environment, environment, *, contract):
    """Record an unchanged execution even when its expected failure did not occur."""
    baseline = deepcopy(environment)
    actual = deepcopy(execution.get('effective_env_config') or environment)
    proof = {'contract': deepcopy(contract), 'initial_environment': baseline,
             'final_environment': actual, 'initial_runtime': _runtime(execution),
             'final_runtime': _runtime(execution), 'proofs': [], 'preserved': True, 'status': 'not_attempted'}
    # Cleanup uncertainty wins over filesystem errors and never permits a new
    # observation, installer, or model call against the affected environment.
    if _unclean(execution):
        proof.update(preserved=False, status='cleanup_unconfirmed')
        return proof
    prefix = _transport(contract, workspace, trial_environment, create=False)
    if _environment_pins(baseline, prefix) != contract['initial_pins'] or baseline.get('repository_pythonpath'):
        raise ValueError('Initial environment differs from preregistration')
    try:
        _check_environment(actual, baseline, contract['initial_pins'], prefix, repaired=False)
        _verify_runtime_snapshot(trial_environment, baseline)
    except (OSError, ValueError) as exc:
        proof.update(preserved=False, status='environment_changed', reason=str(exc))
    else:
        # Preserve initial runner observations for the ordered transition
        # receipt, after validating every protected field against trusted input.
        proof['initial_environment'] = deepcopy(actual)
    return proof


def run_environment_recovery(runner, execution, steps, plan, packet, workspace, environment,
                             run_dir, data, verify_inputs, analyze_failure, on_event, *,
                             contract, trial_environment):
    """Run the selected frozen product's real loop behind an exact change gate."""
    from src.author_runtime_dependencies import repair_requirement, run_with_dependency_repair
    from src.author_self_repair import failure_diagnostic
    from src.repository_repairer import repair_is_stopped
    frozen_steps, frozen_plan, frozen_packet = deepcopy(steps), deepcopy(plan), deepcopy(packet)
    proof = record_environment_execution(execution, workspace, trial_environment, environment, contract=contract)
    baseline = proof['initial_environment']
    prefix = _transport(contract, workspace, trial_environment, create=False)
    pins = _environment_pins(baseline, prefix)
    data['_runtime_environment'] = proof
    data['_runtime_environment_last_execution'] = execution
    allowed = {item['module']: item for item in contract['allowed_repairs']}
    if repair_is_stopped(execution) or _unclean(execution):
        proof['status'] = 'cleanup_unconfirmed' if _unclean(execution) else 'stopped'
        return execution
    if not proof['preserved']:
        return execution
    proof['status'] = 'pending'

    def checked_inputs():
        verify_inputs()
        _source_files(workspace, frozen_packet)
        if _digest(plan) != _digest(frozen_plan) or _digest(packet) != _digest(frozen_packet):
            raise ValueError('Original plan or evidence packet changed')
        try:
            _verify_runtime_snapshot(trial_environment, baseline)
        except (OSError, ValueError) as exc:
            proof.update(preserved=False, reason=str(exc))
            raise

    class CheckedRunner:
        def __init__(self):
            self.seed = True
            self.last = execution
            self.pins = list(pins)
            self.seen = set()

        def run(self, selected_workspace, selected_steps, selected_env, **kwargs):
            checked_inputs()
            if Path(selected_workspace).resolve() != Path(workspace).resolve() or _digest(selected_steps) != _digest(frozen_steps):
                raise ValueError('Environment recovery changed original command plan')
            if self.seed:
                self.seed = False
                _check_environment(selected_env, baseline, self.pins, prefix, repaired=False)
                return deepcopy(execution)
            if repair_is_stopped(self.last) or _unclean(self.last):
                error = RuntimeError('Stopped execution cannot authorize another environment repair')
                if _unclean(self.last):
                    error.process_cleanup_confirmed = False
                raise error
            diagnostic = failure_diagnostic(self.last)
            module, importing = _failure_module(diagnostic, workspace, frozen_packet)
            fix = repair_requirement(diagnostic, frozen_plan, frozen_packet)
            if not fix or module in self.seen or _descriptor(fix) != allowed.get(module):
                raise UnsupportedEnvironmentRepair('Selected product repair is outside the registered dependency additions')
            candidate = list(self.pins)
            for pin in allowed[module]['additions']:
                if pin not in candidate:
                    candidate.append(pin)
            _check_environment(selected_env, baseline, candidate, prefix, repaired=True)
            item = {'failure': {'final': {'stderr': diagnostic}}, 'repair': deepcopy(allowed[module]),
                    'importing_sources': importing, 'before_environment': deepcopy(proof['final_environment']),
                    'requested_environment': deepcopy(selected_env)}
            self.seen.add(module)
            self.pins = candidate
            proof['proofs'].append(item)
            observed = runner.run(selected_workspace, selected_steps, selected_env, **kwargs)
            self.last = observed
            data['_runtime_environment_last_execution'] = observed
            actual = deepcopy(observed.get('effective_env_config') or selected_env)
            proof.update(final_environment=actual, final_runtime=_runtime(observed))
            item['actual_environment'] = deepcopy(actual)
            if _unclean(observed):
                proof['preserved'] = False
                error = RuntimeError('Environment retry process cleanup was not confirmed')
                error.process_cleanup_confirmed = False
                raise error
            checked_inputs()
            _check_environment(actual, baseline, self.pins, prefix, repaired=True)
            if _runtime(observed) != proof['initial_runtime']:
                raise ValueError('Actual author interpreter/runtime changed during dependency recovery')
            return observed

    checked = CheckedRunner()
    try:
        result = run_with_dependency_repair(checked, workspace, frozen_steps, baseline,
            plan=frozen_plan, packet=frozen_packet, run_dir=run_dir, data=data,
            on_event=on_event, verify_inputs=checked_inputs, analyze_failure=analyze_failure,
            prepare_environment=lambda value: prepare_environment_runtime(contract, workspace, trial_environment, value))
        terminal = (data.get('self_repair') or {}).get('status')
        proof['status'] = ('recovered' if result.get('success') else terminal
                           if terminal in {'budget_exhausted', 'interrupted', 'timed_out', 'no_progress', 'blocked'}
                           else 'blocked')
        return result
    except UnsupportedEnvironmentRepair as exc:
        proof.update(status='unsupported', reason=str(exc))
        return checked.last
    except BaseException:
        proof['status'] = 'stopped'
        raise
    finally:
        proof['preserved'] = (proof['preserved'] and _controls(proof['final_environment']) == _controls(baseline)
                              and proof['final_runtime'] == proof['initial_runtime'])


def _distribution_inventory(paths):
    result = {}
    for path in paths:
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                candidate = Path(directory) / name
                if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
                    raise ValueError('Installed dependency contains an unregistered filesystem link')
        for distribution in metadata.distributions(path=[str(path)]):
            name = _name(distribution.metadata['Name'])
            if name in result:
                raise ValueError('Duplicate installed dependency metadata across trial paths')
            result[name] = distribution.version
    return result


def validate_environment_receipt(receipt, case, workspace, environment):
    """Independent oracle: re-read frozen imports, allowed transitions and installs."""
    evidence = case['initial_evidence']
    contract = validate_environment_evidence(case, workspace, evidence)
    recorded = receipt.get('runtime_environment')
    if (_unclean(receipt) or not isinstance(recorded, dict) or recorded.get('contract') != contract
            or recorded.get('preserved') is not True):
        raise ValueError('Missing independent runtime environment preservation evidence')
    prefix = _transport(contract, workspace, environment, create=False)
    initial, final = recorded['initial_environment'], recorded['final_environment']
    pins = list(contract['initial_pins'])
    _check_environment(initial, initial, pins, prefix, repaired=False)
    expected_python = receipt.get('author_python')
    if not expected_python or initial.get('python_executable') != expected_python:
        raise ValueError('Recorded author interpreter differs from the fresh trial interpreter')
    _verify_runtime_snapshot(environment, initial)
    python = Path(os.path.abspath(expected_python))
    if not python.parent.resolve().is_relative_to(Path(environment).resolve()):
        raise ValueError('Recorded author interpreter escaped the trial environment')
    if (recorded['initial_runtime'] != recorded['final_runtime']
            or recorded['initial_runtime'].get('executable') != expected_python
            or not recorded['initial_runtime'].get('runtime')):
        raise ValueError('Actual trial runtime was not preserved')
    repairs = recorded.get('proofs')
    if not isinstance(repairs, list) or len(repairs) > 8:
        raise ValueError('Invalid environment recovery proof count')
    seen, previous = set(), initial
    allowed = {repair['module']: repair for repair in contract['allowed_repairs']}
    for proof in repairs:
        diagnostic = proof['failure']['final']['stderr']
        module, importing = _failure_module(diagnostic, workspace, evidence['packet'])
        repair = allowed.get(module)
        if (module in seen or not repair or proof.get('repair') != repair
                or proof.get('importing_sources') != importing
                or proof.get('before_environment') != previous):
            raise ValueError('Dependency repair is not justified by original failure and preregistration')
        seen.add(module)
        for pin in repair['additions']:
            if pin not in pins:
                pins.append(pin)
        _check_environment(proof['requested_environment'], initial, pins, prefix, repaired=True)
        _check_environment(proof['actual_environment'], initial, pins, prefix, repaired=True)
        previous = proof['actual_environment']
    if final != previous:
        raise ValueError('Final environment differs from independently validated transitions')
    _check_environment(final, initial, pins, prefix, repaired=bool(repairs))
    paths, root = [], Path(environment).resolve(strict=True)
    for entry in receipt.get('dependency_paths', []):
        path = Path(entry).resolve(strict=True)
        if not path.is_relative_to(root / 'dependencies'):
            raise ValueError('Installed dependency path escaped the trial environment')
        relative = path.relative_to(root).as_posix()
        _path(root, relative)
        if not path.is_dir() or path in paths:
            raise ValueError('Invalid or duplicate installed dependency path')
        paths.append(path)
    expected = dict(pin.split('==') for pin in pins)
    if _distribution_inventory(paths) != expected:
        raise ValueError('Actual installed distributions differ from registered exact pins')
    for record in contract['development_wheels']:
        wheel = _path(workspace, record['path'])
        pin, _ = _wheel_metadata(wheel)
        if pin not in pins:
            continue
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.endswith('.dist-info/RECORD'):
                    continue
                targets = [path / member.filename for path in paths if (path / member.filename).is_file()]
                if len(targets) != 1 or _path(root, targets[0].relative_to(root).as_posix()).read_bytes() != archive.read(member):
                    raise ValueError('Installed development dependency differs from its frozen wheel bytes')
    _source_files(workspace, evidence['packet'])
    return [str(path) for path in paths]
