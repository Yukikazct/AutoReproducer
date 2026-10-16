"""Committed file aliases are copied from verified Git blobs, never followed on disk."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from src import repository_reproduction as reproduction
from src.repository_profiles import get_profile
from src.resource_manager import ResourceManager
from test_repository_reproduction import fake_git, legacy_repository


def archive_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            if kind == 'file':
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = {'alias': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE,
                               'directory': tarfile.DIRTYPE, 'fifo': tarfile.FIFOTYPE}[kind]
                if kind in {'alias', 'hardlink'}:
                    member.linkname = value
                archive.addfile(member)
    return stream.getvalue()


def export_fixture(tmp_path, monkeypatch, entries):
    profile = get_profile('dlinear_etth1_smoke')
    root = tmp_path / 'data'
    source = legacy_repository(root)
    archive = archive_bytes(entries)
    fake_git(monkeypatch, source, profile, archive)
    destination = tmp_path / 'export'
    return root, profile, destination, archive


def test_internal_file_aliases_copy_original_bytes_and_keep_provenance(tmp_path, monkeypatch):
    payload = b'VALUE = "committed author bytes"\n'
    root, profile, destination, archive = export_fixture(tmp_path, monkeypatch, [
        ('gat/logger.py', 'alias', '../logger.py'),
        ('other/logger.py', 'alias', '../gat/logger.py'),
        ('logger.py', 'file', payload),
    ])
    snapshot = reproduction.export_repository(root, profile, destination, offline=True, required_files=[])
    expected_sha = hashlib.sha256(payload).hexdigest()
    assert snapshot['archive_sha256'] == hashlib.sha256(archive).hexdigest()
    assert snapshot['files'] == {name: expected_sha for name in ('logger.py', 'gat/logger.py', 'other/logger.py')}
    for name in snapshot['files']:
        assert (destination / name).read_bytes() == payload
        assert not (destination / name).is_symlink()
    aliases = {row['path']: row for row in snapshot['file_aliases']}
    assert aliases['gat/logger.py'] == {
        'path': 'gat/logger.py', 'link_target': '../logger.py', 'resolved_target': 'logger.py',
        'target_sha256': expected_sha, 'materialization': 'committed_file_copy',
    }
    assert aliases['other/logger.py']['link_target'] == '../gat/logger.py'
    assert aliases['other/logger.py']['resolved_target'] == 'logger.py'


@pytest.mark.parametrize('target', [
    '../../outside.py', '/logger.py', '//server/share/logger.py', 'C:/logger.py',
    'C:logger.py', r'..\logger.py', '../.git/config', '../.git./config', '../NUL',
    '../logger.py:stream', '../missing.py', '../directory', '../logger.py/',
    '../logger.py/../logger.py', '',
])
def test_invalid_file_alias_rejected_before_any_export(tmp_path, monkeypatch, target):
    root, profile, destination, _ = export_fixture(tmp_path, monkeypatch, [
        ('logger.py', 'file', b'original\n'), ('directory', 'directory', None),
        ('gat/logger.py', 'alias', target),
    ])
    with pytest.raises(ValueError, match='symlink|链接|别名'):
        reproduction.export_repository(root, profile, destination, offline=True, required_files=[])
    assert not destination.exists()
    assert not (tmp_path / 'outside.py').exists()


@pytest.mark.parametrize('entries', [
    [('one', 'alias', 'two'), ('two', 'alias', 'one')],
    [('one', 'alias', 'one')],
    [('one', 'hardlink', 'logger.py')],
    [('one', 'fifo', None)],
    [('directory', 'alias', 'logger.py'), ('directory/child.py', 'file', b'x')],
    [('logger.py', 'file', b'x'), ('logger.py', 'file', b'y')],
    [('Case/logger.py', 'file', b'x'), ('case/other.py', 'file', b'y')],
])
def test_archive_alias_cycles_and_path_collisions_are_atomic(tmp_path, monkeypatch, entries):
    root, profile, destination, _ = export_fixture(tmp_path, monkeypatch, entries)
    with pytest.raises(ValueError):
        reproduction.export_repository(root, profile, destination, offline=True, required_files=[])
    assert not destination.exists()


def test_real_git_archive_materializes_alias_without_os_symlink_support(tmp_path):
    source = tmp_path / 'author'
    source.mkdir()

    def git(*args, input=None):
        return subprocess.check_output(['git', '-C', str(source), *args], input=input,
                                       stderr=subprocess.PIPE).decode().strip()

    git('init')
    git('config', 'user.name', 'Alias Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'core.autocrlf', 'false')
    payload = b'print("ORIGINAL_ALIAS_OK")\n'
    (source / 'logger.py').write_bytes(payload)
    git('add', 'logger.py')
    blob = git('hash-object', '-w', '--stdin', input=b'../logger.py')
    git('update-index', '--add', '--cacheinfo', f'120000,{blob},gat/logger.py')
    git('commit', '-m', 'author committed internal alias')
    revision = git('rev-parse', 'HEAD')
    manager = ResourceManager(data_root=tmp_path / 'data')
    snapshot = manager.fetch_code('original_author', str(source), revision=revision)
    assert snapshot['commit'] == revision, snapshot
    exported = Path(snapshot['path'])
    assert (exported / 'gat/logger.py').read_bytes() == payload
    assert not (exported / 'gat/logger.py').is_symlink()
    execution = subprocess.check_output([sys.executable, str(exported / 'gat/logger.py')], text=True)
    assert execution.strip() == 'ORIGINAL_ALIAS_OK'
    marker = json.loads((exported / '.autorepro-repo-source.json').read_text('utf-8'))
    assert marker['file_aliases'] == snapshot['file_aliases']
    assert snapshot['file_aliases'][0]['target_sha256'] == hashlib.sha256(payload).hexdigest()


def test_pinned_submodule_alias_keeps_qualified_provenance(tmp_path):
    from test_pinned_submodules import git, initialize
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    initialize(child)
    (child / 'source.py').write_bytes(b'VALUE = 17\n')
    git(child, 'add', 'source.py')
    blob = subprocess.check_output(['git', '-C', str(child), 'hash-object', '-w', '--stdin'],
                                   input=b'source.py').decode().strip()
    git(child, 'update-index', '--add', '--cacheinfo', f'120000,{blob},alias.py')
    git(child, 'commit', '-m', 'committed child alias')
    revision = git(child, 'rev-parse', 'HEAD')
    initialize(parent)
    (parent / '.gitmodules').write_text(
        '[submodule "library"]\n path = vendor/library\n url = ../child\n', encoding='utf-8')
    git(parent, 'add', '.')
    git(parent, 'update-index', '--add', '--cacheinfo', f'160000,{revision},vendor/library')
    git(parent, 'commit', '-m', 'pinned child')
    snapshot = reproduction.export_repository(tmp_path / 'cache', {'repository': {
        'url': str(parent), 'revision': git(parent, 'rev-parse', 'HEAD')}},
        tmp_path / 'export', required_files=[])
    alias = snapshot['file_aliases'][0]
    assert alias['path'] == 'vendor/library/alias.py'
    assert alias['resolved_target'] == 'vendor/library/source.py'
    assert alias['link_target'] == 'source.py'
    assert snapshot['files'][alias['path']] == alias['target_sha256']
    assert snapshot['submodules'][0]['resolved_sha'] == revision
