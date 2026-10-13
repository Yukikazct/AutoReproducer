"""Exercise real layered imports, fixed wheel resolution and reference-aware GC."""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
from src.dependency_cache import cache_guard
from src.dependency_management import (
    choose_base, compatible_candidates, directory_bytes, environment_directories,
    environment_paths, prune_cache, requirement_signature, wheel_resolution,
)
from frontend import history_manager as history


def package(directory, name, version='1.0', value=7):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name.replace('-', '_') + '.py')).write_text(f'VALUE = {value}\n', 'utf-8')
    metadata = directory / f'{name.replace("-", "_")}-{version}.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', 'utf-8')


def ready(directory, requirements, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / '.ready').write_text('ok\n', 'utf-8')
    ce._write_deps_meta(directory, 'reqs', requirements)
    meta = ce.read_deps_meta(directory)
    meta.update(extra)
    (directory / 'meta.json').write_text(json.dumps(meta), 'utf-8')


def report(*packages):
    return {'install': [{'metadata': {'name': name, 'version': version},
                         'download_info': {'url': f'https://files.pythonhosted.org/{name}-{version}.whl',
                                           'archive_info': {'hashes': {'sha256': 'a' * 64}}}}
                        for name, version in packages]}


@pytest.fixture
def executor(tmp_path, monkeypatch):
    ce._INSTALLED_DEPS.clear()
    root = tmp_path / 'deps'
    monkeypatch.setattr(ce, 'DEPS_CACHE_ROOT', root)
    monkeypatch.setattr(ce, 'PIP_INDEX_URL', ce._DEFAULT_PUBLIC_PIP_INDEX)
    monkeypatch.setattr(ce, 'PIP_FIND_LINKS', '')
    monkeypatch.setenv('AUTOREPRO_DEPS_MIN_FREE_GB', '0')
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.resource_events = Mock()
    instance.deps_cache_root = root / 'repository' / ce.runtime_fingerprint()
    instance.deps_lock_root = root
    instance.env_config = {'requirements_txt': 'native-base==1.0\nnative-extra==1.0',
                           'dependency_health_check': True}
    yield instance
    ce._INSTALLED_DEPS.clear()


def test_new_requirement_installs_only_delta_and_reuses_real_base_import(executor, tmp_path, monkeypatch):
    base = executor.deps_cache_root / 'base'
    package(base, 'native-base')
    ready(base, 'native-base==1.0')
    work = tmp_path / 'prepare'
    work.mkdir()
    real_run = executor._run_dependency_command
    calls = []
    def command(argv, **kwargs):
        if argv[1:4] == ['-m', 'pip', 'install']:
            calls.append(argv)
            if '--dry-run' in argv:
                Path(argv[argv.index('--report') + 1]).write_text(
                    json.dumps(report(('native-base', '1.0'), ('native-extra', '1.0'))), 'utf-8')
            else:
                assert '--no-deps' in argv
                manifest = Path(argv[argv.index('-r') + 1]).read_text('utf-8')
                assert 'native-extra @ ' in manifest and 'native-base' not in manifest
                package(Path(argv[argv.index('--target') + 1]), 'native-extra', value=8)
            return subprocess.CompletedProcess(argv, 0, '', '')
        return real_run(argv, **kwargs)
    monkeypatch.setattr(executor, '_run_dependency_command', command)
    assert executor._ensure_local_deps(str(work)) is None
    assert len(calls) == 2 and executor._base_deps_dirs == [str(base)]
    delta = Path(executor._deps_dir)
    assert not (delta / 'native_base.py').exists()
    actual = subprocess.run([ce.sys.executable, '-c', 'import native_base,native_extra;print(native_base.VALUE+native_extra.VALUE)'],
                            env=executor._exec_env(), capture_output=True, text=True)
    assert actual.returncode == 0 and actual.stdout.strip() == '15'
    calls.clear()
    assert executor._ensure_local_deps(str(work)) is None and not calls
    assert executor._base_deps_dirs == [str(base)]
    meta = ce.read_deps_meta(delta)
    assert meta['shared_packages'] == ['native-base']
    assert meta['resolved_versions'] == {'native-base': '1.0', 'native-extra': '1.0'}


@pytest.mark.parametrize('offline', [False, True])
def test_superset_environment_reuses_pins_without_invoking_pip(executor, tmp_path, monkeypatch, offline):
    base = executor.deps_cache_root / 'superset'
    package(base, 'native-base')
    package(base, 'native-extra')
    ready(base, executor.env_config['requirements_txt'])
    executor.env_config['requirements_txt'] = 'native-base==1.0'
    executor.env_config['offline'] = offline
    work = tmp_path / 'prepare'
    work.mkdir()
    original = executor._run_dependency_command
    def command(argv, **kwargs):
        assert argv[1:4] != ['-m', 'pip', 'install']
        return original(argv, **kwargs)
    monkeypatch.setattr(executor, '_run_dependency_command', command)
    assert executor._ensure_local_deps(str(work)) is None
    assert executor._deps_dir == str(base)
    assert executor.env_config['dependency_reuse']['kind'] == 'compatible_environment'


def test_pins_sources_and_python_roots_are_never_combined(tmp_path):
    root = tmp_path / 'runtime311'
    base = root / 'base'
    package(base, 'torch', '2.5.1+cpu')
    ready(base, '--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu')
    assert compatible_candidates(root, 'torch==2.5.1+cpu') == []
    assert compatible_candidates(root, '--extra-index-url https://download.pytorch.org/whl/cu121\ntorch==2.5.1+cu121') == []
    other = tmp_path / 'runtime312'
    other.mkdir()
    assert choose_base(other, 'torch==2.5.1+cpu', wheel_resolution(report(('torch', '2.5.1+cpu')))) is None
    for unsupported in ('-r requirements.txt', 'package @ https://example.com/package.whl',
                        'package[extra]', 'package; python_version>="3"', '--index-url https://private.example/simple'):
        assert requirement_signature(unsupported) is None


def test_delta_health_rejects_missing_resolved_transitive_version(executor):
    delta = executor.deps_cache_root / 'delta'
    package(delta, 'native-base')
    ready(delta, 'native-base==1.0', resolved_versions={'native-base': '1.0', 'transitive': '2'})
    error = executor._check_local_dependency_health(delta, 'native-base==1.0', 'cached')
    assert 'resolved wheel version missing' in error


def test_base_path_escape_and_missing_base_are_rejected(tmp_path):
    root, delta = tmp_path / 'deps', tmp_path / 'deps' / 'delta'
    ready(delta, 'package', base_environment='../outside')
    with pytest.raises(ValueError):
        environment_paths(root, delta, require_ready=True)
    ready(delta, 'package', base_environment='missing')
    with pytest.raises(FileNotFoundError):
        environment_paths(root, delta, require_ready=True)


def test_gc_preserves_active_and_referenced_environments_and_frees_incomplete_first(tmp_path):
    root = tmp_path / 'deps'
    for name in ('base', 'delta', 'active'):
        ready(root / name, 'package', last_used='2000-01-01T00:00:00',
              **({'base_environment': 'base'} if name == 'delta' else {}))
    incomplete = root / 'failed'
    incomplete.mkdir()
    (incomplete / 'payload').write_bytes(b'x' * 1000)
    with cache_guard(root):
        before = prune_cache(root, protected=[root / 'active', root / 'delta'], quota_bytes=1,
                             min_free_bytes=0, dry_run=True)
        assert incomplete.exists() and before['removed'][0]['environment'] == 'failed'
        actual = prune_cache(root, protected=[root / 'active', root / 'delta'], quota_bytes=1, min_free_bytes=0)
    assert actual['removed'] == before['removed']
    assert (root / 'base').exists() and (root / 'active').exists() and (root / 'delta').exists()
    assert not incomplete.exists()


def test_manual_cleanup_cannot_delete_retained_base(tmp_path, monkeypatch):
    root = tmp_path / 'deps'
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(root))
    ready(root / 'base', 'package')
    ready(root / 'delta', 'package', base_environment='base')
    notices = []
    assert history.delete_deps_cache(['base'], on_skip=notices.append) == (0, 0)
    assert notices and (root / 'base').exists()
    assert history.delete_deps_cache(['base', 'delta'])[0] == 2


def test_supplemental_inventory_has_only_environment_leaves(tmp_path):
    root = tmp_path / 'deps'
    leaf = root / 'repository' / 'runtime' / 'supplemental' / 'hash' / 'heal-module'
    ready(leaf, 'module')
    assert list(environment_directories(root)) == [leaf]


def test_gc_never_traverses_dependency_link(tmp_path):
    root, outside = tmp_path / 'deps', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (outside / 'keep').write_text('keep', 'utf-8')
    try:
        (root / 'linked').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlinks unavailable')
    with cache_guard(root):
        assert not prune_cache(root, quota_bytes=0, min_free_bytes=0)['removed']
    assert (outside / 'keep').read_text('utf-8') == 'keep'


def test_resolver_wheel_must_have_source_hash():
    invalid = report(('package', '1'))
    invalid['install'][0]['download_info']['archive_info']['hashes'] = {}
    with pytest.raises(ValueError, match='SHA-256'):
        wheel_resolution(invalid)


@pytest.mark.parametrize('requirements', [None, [], {'package': '1.0'}, 7])
def test_reuse_skips_malformed_requirement_metadata(tmp_path, requirements):
    ready(tmp_path / 'broken', 'package==1.0')
    (tmp_path / 'broken' / 'meta.json').write_text(json.dumps({'requirements': requirements}), 'utf-8')
    assert compatible_candidates(tmp_path, 'package==1.0') == []
    assert choose_base(tmp_path, 'package==1.0', wheel_resolution(report(('package', '1.0')))) is None


def test_gc_uses_directory_recency_when_metadata_timestamp_is_invalid(tmp_path):
    root = tmp_path / 'deps'
    ready(root / 'base', 'package', last_used='2000-01-01T00:00:00')
    ready(root / 'delta', 'package', last_used={'invalid': 'timestamp'}, base_environment='base')
    with cache_guard(root):
        receipt = prune_cache(root, protected=[root / 'delta'], quota_bytes=0, min_free_bytes=0)
    assert receipt['removed'] == []
    assert (root / 'base').is_dir()


def test_gc_normalizes_portable_base_references(tmp_path):
    root = tmp_path / 'deps'
    runtime = root / 'repository' / 'runtime'
    ready(runtime / 'base', 'package', last_used='2000-01-01T00:00:00')
    ready(runtime / 'delta', 'package', base_environment=r'repository\runtime\base')
    assert environment_paths(root, runtime / 'delta', require_ready=True)[1] == runtime / 'base'
    with cache_guard(root):
        receipt = prune_cache(root, protected=[runtime / 'delta'], quota_bytes=0, min_free_bytes=0)
    assert receipt['removed'] == []


def test_gc_preserves_references_when_an_environment_cannot_be_measured(tmp_path, require_symlinks):
    root = tmp_path / 'deps'
    ready(root / 'base', 'package', last_used='2000-01-01T00:00:00')
    ready(root / 'delta', 'package', base_environment='base')
    outside = tmp_path / 'outside.txt'
    outside.write_text('keep', 'utf-8')
    (root / 'delta' / 'linked.txt').symlink_to(outside)
    with cache_guard(root):
        receipt = prune_cache(root, quota_bytes=0, min_free_bytes=0)
    assert receipt['removed'] == []
    assert receipt['errors'] and receipt['errors'][0]['environment'] == 'delta'
    assert receipt['unmeasured_environments'] == ['delta']
    assert not receipt['quota_satisfied']
    assert (root / 'base').is_dir() and outside.read_text('utf-8') == 'keep'


def test_gc_ignores_invalid_base_metadata_without_aborting_cleanup(tmp_path):
    root = tmp_path / 'deps'
    ready(root / 'broken', 'package', base_environment=['not-a-path'])
    with cache_guard(root):
        receipt = prune_cache(root, quota_bytes=0, min_free_bytes=0)
    assert [row['environment'] for row in receipt['removed']] == ['broken']


@pytest.mark.parametrize('limits', [
    {'quota_bytes': -1}, {'min_free_bytes': -1}, {'keep_days': -1},
    {'quota_bytes': float('nan')}, {'min_free_bytes': float('inf')}, {'keep_days': '30'},
])
def test_gc_rejects_invalid_limits_before_removing_environments(tmp_path, limits):
    root = tmp_path / 'deps'
    ready(root / 'keep', 'package', last_used='2000-01-01T00:00:00')
    with cache_guard(root), pytest.raises(ValueError, match='finite non-negative'):
        prune_cache(root, **limits)
    assert (root / 'keep').is_dir()
