"""Narrow child-only RECORD normalization, retaining the requested pip CLI."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.pip_windows_target as target


@pytest.mark.parametrize('path, expected', [
    (r'\\?\D:\owned\pip-target\lib\python\../../bin/numba',
     r'\\?\D:\owned\pip-target\bin\numba'),
    (r'\\?\UNC\server\share\owned\lib\python\../../bin/numba',
     r'\\?\UNC\server\share\owned\bin\numba'),
    (r'D:\owned\lib\python\../../bin/numba', r'D:\owned\lib\python\../../bin/numba'),
    (r'\\server\share\owned\lib\python\../../bin/numba', r'\\server\share\owned\lib\python\../../bin/numba'),
    ('/owned/lib/python/../../bin/numba', '/owned/lib/python/../../bin/numba'),
])
def test_only_extended_record_paths_are_normalized(path, expected):
    assert target.normalize_record_filesystem_path(path) == expected


@pytest.mark.parametrize('option', [['--target', 'owned deps'], ['--target=owned deps'], ['-t', 'owned deps']])
def test_windows_target_bootstrap_keeps_every_pip_argument_and_utf8(monkeypatch, option):
    monkeypatch.setattr(target, 'os', SimpleNamespace(name='nt'))
    command = ['selected python.exe', '-m', 'pip', 'install', Path('local wheel.whl'),
               '--no-index', *option, '--no-user']
    original = list(command)
    adapted = target.target_install_command(command)
    assert adapted[:4] == [command[0], '-I', '-X', 'utf8']
    assert Path(adapted[4]).name == 'pip_windows_target.py'
    assert adapted[5:] == command[3:] and command == original


@pytest.mark.parametrize('platform, arguments', [
    ('posix', ['install', '--target', 'owned', 'wheel.whl']),
    ('nt', ['install', 'wheel.whl']),
    ('nt', ['download', '--target', 'owned', 'wheel.whl']),
    ('nt', ['--version']),
])
def test_other_pip_commands_keep_the_original_invocation(monkeypatch, platform, arguments):
    monkeypatch.setattr(target, 'os', SimpleNamespace(name=platform))
    command = ['python', '-m', 'pip', *arguments]
    assert target.target_install_command(command) is command
