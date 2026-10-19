"""Recover only missing imports explicitly causing the terminal import failure."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.author_runtime_dependencies import repair_requirement
from src.author_self_repair import failure_fingerprint, repair_local_import, terminal_missing_import


DIRECT = '\n\nThe above exception was the direct cause of the following exception:\n\n'
HANDLING = '\n\nDuring handling of the above exception, another exception occurred:\n\n'


def traceback_for(exception, filename='loader.py', line=3):
    return f'Traceback (most recent call last):\n  File "{filename}", line {line}, in <module>\n{exception}'


def wrapped(module, depth=1, separator=DIRECT):
    diagnostic = traceback_for(f"ModuleNotFoundError: No module named '{module}'")
    for _ in range(depth):
        diagnostic += separator + traceback_for('ImportError: optional backend could not be loaded')
    return diagnostic + '\n'


@pytest.mark.parametrize('module', ['transport_client', 'image_codec', 'matrix_backend'])
@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_explicit_import_causes_recover_the_actual_missing_distribution(module, depth, newline):
    diagnostic = wrapped(module, depth).replace('\n', newline)
    plan = {'requirements': {'author': [module.replace('_', '-') + '==1.2.3']}, 'capture': {'device': 'cpu'}}
    repair = repair_requirement(diagnostic, plan, {'file_inventory': {}})
    assert repair is not None
    assert repair['module'] == module
    assert repair['requirement'] == module.replace('_', '-') + '==1.2.3'


@pytest.mark.parametrize('diagnostic', [
    wrapped('obsolete', separator=HANDLING),
    wrapped('obsolete') + '\n' + traceback_for('ValueError: invalid shape'),
    traceback_for("ModuleNotFoundError: No module named 'obsolete'") + DIRECT + traceback_for('ValueError: invalid shape'),
    traceback_for("ModuleNotFoundError: No module named 'obsolete'") + DIRECT + traceback_for('RuntimeError: init failed')
        + DIRECT + traceback_for('ImportError: backend unavailable'),
    traceback_for("ModuleNotFoundError: No module named 'obsolete'") + '\nRecovered optional import\n'
        + traceback_for('ImportError: cannot import name token'),
    traceback_for("ModuleNotFoundError: No module named 'obsolete'") + '\nRecovered optional import'
        + DIRECT + traceback_for('ImportError: cannot import name token'),
    "Quoted error: ModuleNotFoundError: No module named 'obsolete'" + DIRECT
        + traceback_for('ImportError: backend unavailable'),
    "ModuleNotFoundError: No module named 'truncated_trace'" + DIRECT
        + traceback_for('ImportError: backend unavailable'),
    traceback_for('ImportError: cannot import name token'),
    wrapped('obsolete').replace(DIRECT, '\n\n'),
])
def test_unrelated_caught_implicit_or_unproven_import_causes_are_not_installed(diagnostic):
    assert terminal_missing_import(diagnostic) == ''


def test_nearest_explicit_cause_does_not_resurrect_an_earlier_caught_import():
    earlier = traceback_for("ModuleNotFoundError: No module named 'obsolete'")
    diagnostic = earlier + HANDLING + wrapped('current_backend')
    assert terminal_missing_import(diagnostic) == "ModuleNotFoundError: No module named 'current_backend'"


def test_same_wrapper_with_a_different_missing_dependency_is_progress():
    first = {'final': {'stderr': wrapped('transport_client')}}
    second = {'final': {'stderr': wrapped('image_codec')}}
    assert failure_fingerprint(first) != failure_fingerprint(second)
    first_with_progress = {'final': {'stderr': 'progress: 2 seconds\n' + wrapped('transport_client')}}
    assert failure_fingerprint(first) == failure_fingerprint(first_with_progress)


def test_real_wrapped_local_import_uses_the_causal_source_without_changing_the_command(tmp_path):
    module = tmp_path / 'lib/isolated_author_helper.py'
    module.parent.mkdir()
    module.write_text('VALUE = 17\n', encoding='utf-8')
    script = tmp_path / 'train.py'
    script.write_text('try:\n    from isolated_author_helper import VALUE\n'
                      'except ImportError as error:\n    raise ImportError("backend unavailable") from error\n'
                      'print(VALUE)\n', encoding='utf-8')
    packet = {'file_inventory': {p.relative_to(tmp_path).as_posix():
              {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [script, module]}}
    command = [sys.executable, '-u', str(script)]
    failed = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=15)
    assert failed.returncode != 0
    repair = repair_local_import({'final': {'stderr': failed.stderr}}, tmp_path, packet)
    assert repair is not None and repair['root'] == 'lib'
    assert repair['importing_sources'] == ['train.py']
    restored = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=15,
                             env={**os.environ, 'PYTHONPATH': str(tmp_path / repair['root'])})
    assert restored.returncode == 0 and restored.stdout.strip() == '17'
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == record['sha256']
               for name, record in packet['file_inventory'].items())
