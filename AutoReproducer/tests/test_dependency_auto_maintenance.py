"""Automatic dependency GC at actual execution boundaries; no network installs."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
import src.repository_runner as repository
from src.resource_events import ResourceEventLogger
from src.dependency_cache import cache_guard, DependencyCacheBusy
from src.dependency_management import prune_cache, read_maintenance
from src.dependency_temporary import OWNER_MARKER, TEMP_NAMESPACE, pip_temporary, reap_pip_temporary


def environment(root, name, *, size=15000, recent=False, **extra):
    path = root / name
    path.mkdir(parents=True)
    (path / '.ready').write_text('ok', encoding='utf-8')
    (path / 'package.bin').write_bytes(b'x' * size)
    (path / 'meta.json').write_text(json.dumps({
        'last_used': '2099-01-01T00:00:00+00:00' if recent else '2000-01-01T00:00:00+00:00',
        **extra}), encoding='utf-8')
    return path


@pytest.fixture
def executor(tmp_path, monkeypatch):
    root = tmp_path / 'deps'
    monkeypatch.setattr(ce, 'DEPS_CACHE_ROOT', root)
    monkeypatch.setattr(repository, 'DEPS_CACHE_ROOT', root)
    monkeypatch.setenv('AUTOREPRO_DEPS_QUOTA_GB', '0.00001')
    monkeypatch.setenv('AUTOREPRO_DEPS_MIN_FREE_GB', '0')
    monkeypatch.setenv('AUTOREPRO_DEPS_INSTALL_RESERVE_GB', '0')
    monkeypatch.setenv('AUTOREPRO_DEPS_KEEP_RECENT', '0')
    ce._INSTALLED_DEPS.clear()
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {'requirements_txt': 'fixture-package==1', 'dependency_health_check': False}
    instance.resource_events = ResourceEventLogger(tmp_path / 'events.jsonl')
    yield instance
    ce._INSTALLED_DEPS.clear()


@pytest.mark.parametrize('in_memory', [False, True])
def test_cache_hit_enforces_quota_without_health_review_or_pip(executor, tmp_path, monkeypatch, in_memory):
    root = executor._dependency_lock_root()
    current = environment(root, ce.reqs_digest(executor.env_config['requirements_txt']), size=300)
    stale = environment(root, 'repository/cpython-311-win32-AMD64/idle')
    if in_memory:
        ce._INSTALLED_DEPS[executor.env_config['requirements_txt']] = ''
    monkeypatch.setattr(executor, '_run_dependency_command', lambda *a, **k: pytest.fail('cache hit ran pip'))
    work = tmp_path / 'prepare'
    work.mkdir()
    assert executor._ensure_local_deps(str(work)) is None
    assert current.is_dir() and not stale.exists()
    receipts = executor.env_config['dependency_cache_maintenance']
    assert receipts[-1]['quota_satisfied']
    assert any(item['removed'] for item in receipts)


@pytest.mark.parametrize('exit_code', [0, 1])
def test_actual_repository_exit_reclaims_idle_cache_created_during_run(executor, tmp_path, exit_code):
    root = executor._dependency_lock_root()
    current = environment(root, f'repository/{ce.runtime_fingerprint()}/{ce.reqs_digest(executor.env_config["requirements_txt"])}', size=300)
    work = tmp_path / 'repo'
    work.mkdir()
    stale = root / 'repository/another-runtime/idle'
    def event(item):
        if item.get('type') == 'repository_step' and item.get('status') in {'success', 'error'}:
            environment(root, 'repository/another-runtime/idle')
    result = repository.RepositoryRunner(executor=executor).run(work, [
        {'id': 'run', 'argv': ['python', '-c', f'print("fixture");raise SystemExit({exit_code})']}],
        executor.env_config, on_event=event)
    assert result['success'] is (exit_code == 0)
    assert current.is_dir() and not stale.exists()
    receipts = result['effective_env_config']['dependency_cache_maintenance']
    assert receipts[-1]['phase'] == 'scope_finished'
    assert receipts[-1]['quota_satisfied']
    saved = json.loads((Path(result['run_dir']) / 'environment.json').read_text('utf-8'))
    assert saved['dependency_cache_maintenance'] == receipts
    assert saved['preparation_elapsed_s'] == result['environment']['preparation_elapsed_s']


def test_successive_runs_evict_and_rebuild_without_erasing_history(executor, tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOREPRO_DEPS_QUOTA_GB', '0.00002')
    installed = []
    def pip(command, **kwargs):
        target = Path(command[command.index('--target') + 1])
        requirements = Path(command[command.index('-r') + 1]).read_text('utf-8').strip()
        version = int(requirements.rsplit('==', 1)[1])
        (target / 'fixture_dependency.py').write_text(f'VERSION = {version}\n', encoding='utf-8')
        (target / 'payload').write_bytes(b'x' * 15000)
        installed.append(requirements)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(executor, '_invoke_dependency_command', pip)
    root = executor._dependency_lock_root()
    work = tmp_path / 'author'
    work.mkdir()
    report = work / 'report.md'
    report.write_text('retained report', encoding='utf-8')
    runs = []
    runner = repository.RepositoryRunner(executor=executor)
    for version in (1, 2, 1):
        requirements = f'fixture-package=={version}'
        result = runner.run(work, [{'id': 'train', 'argv': ['python', '-c',
            'from fixture_dependency import VERSION;print(VERSION)']}],
            {'requirements_txt': requirements, 'dependency_health_check': False})
        assert result['success'], result['final']['stderr']
        assert result['final']['stdout'].strip() == str(version)
        assert result['environment']['dependency_cache_maintenance'][-1]['quota_satisfied']
        current = root / f'repository/{ce.runtime_fingerprint()}/{ce.reqs_digest(requirements)}'
        assert current.is_dir()
        other = root / f'repository/{ce.runtime_fingerprint()}/{ce.reqs_digest(f"fixture-package=={3-version}")}'
        assert not other.exists()
        runs.append(result)
    assert installed == ['fixture-package==1', 'fixture-package==2', 'fixture-package==1']
    assert report.read_text('utf-8') == 'retained report'
    for result in runs:
        saved = json.loads((Path(result['run_dir']) / 'environment.json').read_text('utf-8'))
        assert saved['requirements_txt'] == result['environment']['requirements_txt']
        assert Path(result['final']['stdout_path']).read_text('utf-8') == result['final']['stdout']


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout', 'cancel'])
def test_pip_temporary_files_are_owned_and_released_on_every_exit(executor, tmp_path, monkeypatch, outcome):
    inherited = tmp_path / 'other-application-temp'
    inherited.mkdir()
    sentinel = inherited / 'keep'
    sentinel.write_text('keep', encoding='utf-8')
    monkeypatch.setenv('TMPDIR', str(inherited))
    observed = {}
    def pip(command, **kwargs):
        env = kwargs['env']
        owned = Path(env['TMPDIR'])
        assert not os.path.samefile(owned, inherited)
        assert os.path.samefile(owned.parent.parent, executor._dependency_lock_root())
        assert env['TEMP'] == env['TMP'] == str(owned)
        assert Path(env['PIP_CACHE_DIR']).is_relative_to(owned)
        (owned / 'pip-unpack-fixture').mkdir()
        (owned / 'pip-unpack-fixture' / 'wheel.bin').write_bytes(b'x' * 2000)
        observed['path'] = owned
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        if outcome == 'cancel':
            raise KeyboardInterrupt('fixture cancellation')
        return subprocess.CompletedProcess(command, 0 if outcome == 'success' else 1, '', '')
    monkeypatch.setattr(executor, '_invoke_dependency_command', pip)
    command = [ce.sys.executable, '-m', 'pip', 'install', 'fixture-package==1']
    if outcome in {'timeout', 'cancel'}:
        with pytest.raises(subprocess.TimeoutExpired if outcome == 'timeout' else KeyboardInterrupt):
            executor._run_dependency_command(command, timeout=1, env=executor._pip_env())
    else:
        result = executor._run_dependency_command(command, timeout=1, env=executor._pip_env())
        assert result.returncode == (0 if outcome == 'success' else 1)
    assert not observed['path'].exists() and sentinel.read_text('utf-8') == 'keep'
    assert executor.env_config['dependency_temporary_files'][-1]['freed_bytes'] >= 2000


@pytest.mark.parametrize('outcome', ['failure', 'cancel'])
def test_failed_install_discards_partial_target_without_losing_diagnostic(executor, tmp_path, monkeypatch, outcome):
    def pip(command, **kwargs):
        target = Path(command[command.index('--target') + 1])
        (target / 'partial-wheel.bin').write_bytes(b'x' * 4000)
        if outcome == 'cancel':
            raise KeyboardInterrupt('installer cancelled')
        return subprocess.CompletedProcess(command, 1, '', 'wheel installation failed')
    monkeypatch.setattr(executor, '_run_dependency_command', pip)
    work = tmp_path / 'prepare'
    work.mkdir()
    if outcome == 'cancel':
        with pytest.raises(KeyboardInterrupt, match='installer cancelled'):
            executor._ensure_local_deps(str(work))
    else:
        diagnostic = executor._ensure_local_deps(str(work))
        assert 'wheel installation failed' in diagnostic
        assert executor.env_config['dependency_install_attempts'][0]['success'] is False
    target = executor._dependency_lock_root() / ce.reqs_digest(executor.env_config['requirements_txt'])
    assert not target.exists()
    assert executor.env_config['dependency_cache_maintenance'][-1]['phase'] == 'scope_finished'


def test_install_peak_reserve_checked_with_health_disabled(executor, tmp_path, monkeypatch):
    import src.dependency_management as management
    monkeypatch.setenv('AUTOREPRO_DEPS_MIN_FREE_GB', '1')
    monkeypatch.setenv('AUTOREPRO_DEPS_INSTALL_RESERVE_GB', '2')
    monkeypatch.setattr(management.shutil, 'disk_usage', lambda path: SimpleNamespace(free=2 * 1024**3))
    monkeypatch.setattr(executor, '_run_dependency_command', lambda *a, **k: pytest.fail('insufficient space ran pip'))
    work = tmp_path / 'prepare'
    work.mkdir()
    assert '空间不足' in executor._ensure_local_deps(str(work))
    receipt = next(item for item in executor.env_config['dependency_cache_maintenance'] if item['phase'] == 'before_install')
    assert receipt['min_free_bytes'] == 3 * 1024**3 and receipt['install_reserve_bytes'] == 2 * 1024**3
    assert receipt['space_satisfied'] is False


def test_retention_and_references_explain_unmet_global_budget(executor):
    root = executor._dependency_lock_root()
    base = environment(root, 'repository/runtime-mac/base')
    delta = environment(root, 'repository/runtime-mac/delta', base_environment='repository/runtime-mac/base')
    newest = environment(root, 'repository/runtime-win/cuda', recent=True)
    idle = environment(root, 'legacy')
    with cache_guard(root):
        receipt = prune_cache(root, protected=[delta], quota_bytes=1, min_free_bytes=0, keep_recent=1)
    assert base.exists() and delta.exists() and newest.exists() and not idle.exists()
    assert not receipt['quota_satisfied'] and '仍未满足' in receipt['reason']
    kept = {row['environment']: row for row in receipt['retained']}
    assert 'referenced_base' in kept['repository/runtime-mac/base']['reasons']
    assert 'protected' in kept['repository/runtime-mac/delta']['reasons']
    assert 'recent' in kept['repository/runtime-win/cuda']['reasons']
    assert receipt['freed_bytes'] == receipt['removed'][0]['bytes']
    assert receipt['initial_bytes'] - receipt['remaining_bytes'] == receipt['freed_bytes']


def owned_orphan(root, *, created_at=0, kind='autorepro_pip'):
    token = uuid.uuid4().hex
    path = root / TEMP_NAMESPACE / ('install-' + token)
    path.mkdir(parents=True)
    (path / OWNER_MARKER).write_text(json.dumps({
        'kind': kind, 'version': 1, 'id': token, 'created_at': created_at}), encoding='utf-8')
    (path / '.owner.lock').touch()
    (path / 'payload').write_bytes(b'x' * 3000)
    return path


def test_cache_parent_alias_supports_gc_and_temporary_cleanup(tmp_path, require_symlinks):
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    root = alias / 'deps'
    idle = environment(root, 'idle')
    orphan = owned_orphan(root)
    with cache_guard(root):
        receipt = prune_cache(root, quota_bytes=1, min_free_bytes=0)
        recovered = reap_pip_temporary(root)
    assert not idle.exists() and not orphan.exists()
    assert receipt['removed'][0]['environment'] == 'idle'
    assert recovered['freed_bytes'] >= 3000
    cleanup = []
    with pip_temporary(root, on_cleanup=cleanup.append) as work:
        (work / 'unpacked').write_bytes(b'x' * 3000)
    assert not work.exists()
    assert cleanup[0]['status'] == 'removed' and cleanup[0]['freed_bytes'] >= 3000


def test_linked_cache_root_is_rejected_before_cleanup(tmp_path, require_symlinks):
    outside = tmp_path / 'external'
    idle = environment(outside, 'idle')
    alias = tmp_path / 'linked-cache'
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink or junction'):
        prune_cache(alias, quota_bytes=1, min_free_bytes=0)
    with pytest.raises(ValueError, match='symlink or junction'):
        reap_pip_temporary(alias)
    with pytest.raises(ValueError, match='symlink or junction'), pip_temporary(alias):
        pytest.fail('accepted a linked cache root')
    assert idle.exists()


def test_restart_reaps_only_old_owned_temporaries_and_preserves_other_resources(executor, tmp_path):
    import time
    root = executor._dependency_lock_root()
    abandoned = owned_orphan(root)
    recent = owned_orphan(root, created_at=time.time())
    unknown = owned_orphan(root, kind='other_app')
    immutable = []
    for name in ('reports/report.md', 'runs/model.pt', 'repos/author.py', 'datasets/input.csv', 'papers/paper.pdf'):
        path = tmp_path / 'data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unchanged')
        immutable.append(path)
    with executor.dependency_scope():
        assert not abandoned.exists() and recent.exists() and unknown.exists()
    receipts = executor.env_config['dependency_cache_maintenance']
    assert receipts[0]['temporary_files']['freed_bytes'] >= 3000
    assert all(path.read_bytes() == b'unchanged' for path in immutable)
    assert read_maintenance(root) == receipts[-1]


def test_real_active_process_protects_temporaries_and_kill_allows_recovery(executor):
    root = executor._dependency_lock_root()
    program = (
        'import json,sys\nfrom src.dependency_temporary import pip_temporary, OWNER_MARKER\n'
        'with pip_temporary(sys.argv[1]) as work:\n'
        ' marker = work / OWNER_MARKER\n record = json.loads(marker.read_text())\n'
        ' record["created_at"] = 0\n marker.write_text(json.dumps(record))\n'
        ' (work / "unpacked").write_bytes(b"x"*3000)\n'
        ' print(str(work),flush=True)\n sys.stdin.readline()\n')
    process = subprocess.Popen([ce.sys.executable, '-c', program, str(root)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8')
    try:
        line = process.stdout.readline().strip()
        assert line, 'temporary owner failed to start'
        work = Path(line)
        with pytest.raises(DependencyCacheBusy), cache_guard(root, cleanup=True):
            pytest.fail('cleanup acquired an active installer lock')
        assert work.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    with executor.dependency_scope():
        assert not work.exists()


def test_orphan_lease_link_and_missing_ownership_are_not_deleted(executor, require_symlinks):
    from filelock import FileLock
    root = executor._dependency_lock_root()
    active = owned_orphan(root)
    linked = root / TEMP_NAMESPACE / ('install-' + uuid.uuid4().hex)
    outside = root.parent / 'external-temp'
    outside.mkdir()
    (outside / 'keep').write_text('keep', encoding='utf-8')
    linked.symlink_to(outside, target_is_directory=True)
    unknown = root / TEMP_NAMESPACE / 'other-app'
    unknown.mkdir()
    with cache_guard(root), FileLock(str(active / '.owner.lock')):
        receipt = reap_pip_temporary(root)
    assert active.exists() and unknown.exists() and linked.is_symlink()
    assert (outside / 'keep').read_text('utf-8') == 'keep'
    assert any(item['reason'] == 'active' for item in receipt['retained'])
    assert receipt['errors'] and not receipt['removed']


def test_cli_reports_unmet_budget_and_keeps_preview_read_only(executor, tmp_path):
    root = executor._dependency_lock_root()
    recent = environment(root, 'recent', recent=True)
    old = environment(root, 'old')
    orphan = owned_orphan(root)
    env = {**os.environ, 'AUTOREPRO_DEPS_ROOT': str(root), 'PYTHONIOENCODING': 'utf-8'}
    script = Path(__file__).parents[1] / 'scripts/resource_cli.py'
    command = [ce.sys.executable, str(script), 'deps-prune', '--keep-recent', '1']
    preview = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert preview.returncode == 2
    assert recent.exists() and old.exists() and orphan.exists()
    assert not (root / '.maintenance.json').exists()
    actual = subprocess.run([*command, '--yes'], env=env, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert actual.returncode == 2
    receipt = json.loads(actual.stdout)
    assert not receipt['quota_satisfied'] and receipt['temporary_files']['freed_bytes'] >= 3000
    assert recent.exists() and not old.exists() and not orphan.exists()
    assert receipt['retained'][0]['reasons'] == ['recent']


def test_history_reads_policy_and_latest_receipt_without_reclaiming(executor, monkeypatch):
    from frontend.history_manager import dependency_maintenance_status
    root = executor._dependency_lock_root()
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(root))
    with executor.dependency_scope():
        pass
    abandoned = owned_orphan(root)
    status = dependency_maintenance_status()
    assert abandoned.exists()
    assert status['latest'] == executor.env_config['dependency_cache_maintenance'][-1]
    assert status['policy']['quota_bytes'] == int(0.00001 * 1024**3) and status['free_bytes'] >= 0
