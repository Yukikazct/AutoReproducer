"""Exercise real Git gitlinks rather than treating .gitmodules as a refusal."""
import hashlib
from pathlib import Path
import subprocess

from src.repository_reproduction import export_repository, materialize_submodules


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], stderr=subprocess.PIPE).decode().strip()


def initialize(path):
    path.mkdir()
    git(path, 'init')
    git(path, 'config', 'user.name', 'Submodule Fixture')
    git(path, 'config', 'user.email', 'fixture@example.invalid')
    git(path, 'config', 'core.autocrlf', 'false')


def commit(path):
    git(path, 'add', '.')
    git(path, 'commit', '-m', 'fixture')
    return git(path, 'rev-parse', 'HEAD')


def test_deferred_identity_export_then_fixed_child_export_and_offline_replay(tmp_path):
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    initialize(child)
    (child / 'library.py').write_bytes(b'VALUE = 17\n')
    child_revision = commit(child)
    (child / 'library.py').write_bytes(b'VALUE = 99\n')
    commit(child)  # The newest version must never replace the committed gitlink.
    initialize(parent)
    (parent / 'README.md').write_text('Paper implementation with a vendored library.', encoding='utf-8')
    (parent / '.gitmodules').write_text(
        '[submodule "library"]\n path = vendor/library\n url = ../child\n', encoding='utf-8')
    git(parent, 'add', '.')
    git(parent, 'update-index', '--add', '--cacheinfo', f'160000,{child_revision},vendor/library')
    git(parent, 'commit', '-m', 'pinned gitlink')
    parent_revision = git(parent, 'rev-parse', 'HEAD')
    profile = {'repository': {'url': str(parent), 'revision': parent_revision}}
    root = tmp_path / 'cache'
    snapshot = export_repository(root, profile, tmp_path / 'review', required_files=[], include_submodules=False)
    assert snapshot['submodules_pending']
    assert not (Path(snapshot['path']) / 'vendor/library/library.py').exists()
    materialize_submodules(root, snapshot)
    original = (Path(snapshot['path']) / 'vendor/library/library.py').read_bytes()
    assert original.splitlines() == [b'VALUE = 17']
    assert snapshot['submodules'][0]['resolved_sha'] == child_revision
    assert snapshot['files']['vendor/library/library.py'] == hashlib.sha256(original).hexdigest()
    assert not snapshot['submodules_pending']
    replay = export_repository(root, profile, tmp_path / 'offline', offline=True,
                               required_files=['vendor/library/library.py'])
    assert replay['files'] == snapshot['files']
    assert not list(Path(replay['path']).rglob('.git'))


def test_gitmodules_without_gitlinks_does_not_block_repository(tmp_path):
    parent = tmp_path / 'parent'
    initialize(parent)
    (parent / '.gitmodules').write_text('# No active submodules\n', encoding='utf-8')
    revision = commit(parent)
    snapshot = export_repository(tmp_path / 'cache', {'repository': {'url': str(parent), 'revision': revision}},
        tmp_path / 'repo', required_files=[])
    assert snapshot['submodules'] == [] and not snapshot['submodules_pending']
