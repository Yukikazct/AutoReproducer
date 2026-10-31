"""Real offline pip and owned cleanup with Win32 long paths disabled or enabled."""
import csv
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import Mock
import zipfile

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.dependency_temporary import OWNER_MARKER, TEMP_NAMESPACE, pip_temporary, reap_pip_temporary
from src.dependency_install_progress import DependencyInstallProgress
from src.safety.files import directory_bytes


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows extended-length filesystem paths")
NATIVE = ("longpath_demo.libs/libopenblas64__v0.3.23-293-gc2f4bdbb-gcc_10_3_0-"
          "2bde3a66a51006b2b53eb373ff767a3f.dll")


def managed_root(tmp_path):
    root = tmp_path / ("managed_" + "r" * max(1, 112 - len(str(tmp_path))))
    root.mkdir()
    return root


def make_wheel(tmp_path, *, with_scripts=False):
    dist = "longpath_demo-1.0.dist-info/"
    contents = {"longpath_demo/__init__.py": b"VALUE = 7\n", NATIVE: b"offline path fixture\n",
        dist + "METADATA": b"Metadata-Version: 2.1\nName: longpath-demo\nVersion: 1.0\n",
        dist + "WHEEL": b"Wheel-Version: 1.0\nGenerator: path-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    if with_scripts:
        contents['longpath_demo/__init__.py'] = b'VALUE = 7\ndef main():\n    print(VALUE)\n'
        contents['longpath_demo-1.0.data/scripts/path-fixture-legacy'] = (
            b'#!python\nfrom longpath_demo import main\nmain()\n')
        contents[dist + 'entry_points.txt'] = b'[console_scripts]\npath-fixture-entry = longpath_demo:main\n'
    rows = io.StringIO(newline="")
    writer = csv.writer(rows)
    for name in [*contents, dist + "RECORD"]:
        writer.writerow([name, "", ""])
    contents[dist + "RECORD"] = rows.getvalue().encode()
    wheel = tmp_path / "longpath_demo-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return wheel


def write_long_payload(work):
    path = work / "unpacked" / NATIVE
    assert len(str(path)) > 260
    extended = Path("\\\\?\\" + str(path.absolute()))
    extended.parent.mkdir(parents=True)
    extended.write_bytes(b"long temporary payload")


def clean_owned_fixture(root, tmp_path):
    assert root.resolve().is_relative_to(tmp_path.resolve())
    if root.exists():
        shutil.rmtree("\\\\?\\" + str(root.absolute()))


def test_managed_pip_installs_wheel_with_a_long_extraction_path(tmp_path):
    root = managed_root(tmp_path)
    instance = CodeExecutorAgent(None, logger=Mock())
    instance.deps_cache_root = root
    instance.env_config = {"auto_prepare": False}
    wheel = make_wheel(tmp_path)
    target = tmp_path / "target"
    try:
        result = instance._run_dependency_command([sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--no-user", "--no-index", "--no-deps",
            "--target", str(target), str(wheel)], timeout=60, env=instance._pip_env())
        assert result.returncode == 0, result.stderr
        assert (target / NATIVE).read_bytes() == b"offline path fixture\n"
        imported = subprocess.run([sys.executable, "-I", "-c",
            f"import sys; sys.path.insert(0, {str(target)!r}); import longpath_demo; print(longpath_demo.VALUE)"],
            capture_output=True, text=True, timeout=10, check=True)
        assert imported.stdout.strip() == "7"
        receipt, = instance.env_config["dependency_temporary_files"]
        assert receipt["status"] == "removed"
        assert not Path(receipt["path"]).exists()
    finally:
        clean_owned_fixture(root, tmp_path)


def test_managed_pip_rehashes_legacy_script_record_and_retains_long_extraction(tmp_path):
    root = managed_root(tmp_path)
    instance = CodeExecutorAgent(None, logger=Mock())
    instance.deps_cache_root = root
    instance.env_config = {'auto_prepare': False}
    wheel = make_wheel(tmp_path, with_scripts=True)
    target = tmp_path / 'target'
    try:
        result = instance._run_dependency_command([sys.executable, '-m', 'pip', 'install',
            '--disable-pip-version-check', '--no-user', '--no-index', '--no-deps',
            '--target', str(target), str(wheel)], timeout=60, env=instance._pip_env())
        assert result.returncode == 0, result.stderr
        assert (target / NATIVE).read_bytes() == b'offline path fixture\n'
        legacy = target / 'bin/path-fixture-legacy'
        assert legacy.read_bytes().startswith(b'#!') and not legacy.read_bytes().startswith(b'#!python\n')
        imported = subprocess.run([sys.executable, '-I', '-c',
            f'import sys,runpy; sys.path.insert(0, {str(target)!r}); runpy.run_path({str(legacy)!r}, run_name="__main__")'],
            capture_output=True, text=True, timeout=10, check=True)
        assert imported.stdout.strip() == '7'
        entry = target / 'bin/path-fixture-entry.exe'
        launched = subprocess.run([str(entry)], env={**os.environ, 'PYTHONPATH': str(target)},
            capture_output=True, text=True, timeout=10, check=True)
        assert launched.stdout.strip() == '7'
        rows = list(csv.reader((target / 'longpath_demo-1.0.dist-info/RECORD').read_text('utf-8').splitlines()))
        script_record, = [row for row in rows if row[0].endswith('/path-fixture-legacy')]
        expected = base64.urlsafe_b64encode(hashlib.sha256(legacy.read_bytes()).digest()).rstrip(b'=').decode('ascii')
        assert script_record[1] == 'sha256=' + expected and int(script_record[2]) == legacy.stat().st_size
        receipt, = instance.env_config['dependency_temporary_files']
        assert receipt['status'] == 'removed' and not Path(receipt['path']).exists()
    finally:
        clean_owned_fixture(root, tmp_path)


def test_owned_long_temporary_files_are_counted_and_removed_after_error(tmp_path):
    root = managed_root(tmp_path)
    receipts = []
    try:
        with pytest.raises(RuntimeError, match="installer failed"):
            with pip_temporary(root, on_cleanup=receipts.append) as work:
                write_long_payload(work)
                assert directory_bytes(work) >= len(b"long temporary payload")
                raise RuntimeError("installer failed")
        assert receipts[0]["status"] == "removed", receipts
        assert receipts[0]["freed_bytes"] >= len(b"long temporary payload")
        assert not work.exists()
    finally:
        clean_owned_fixture(root, tmp_path)


def test_download_monitor_measures_files_past_max_path(tmp_path):
    root = managed_root(tmp_path)
    name = "longpath_demo-" + "1" * 100 + "-py3-none-any.whl"
    work = root / TEMP_NAMESPACE / ("install-" + "b" * 32)
    path = work / "pip-unpack-fixture" / name
    assert len(str(path)) > 260
    extended = Path("\\\\?\\" + str(path.absolute()))
    extended.parent.mkdir(parents=True)
    extended.write_bytes(b"downloaded wheel bytes")
    events = []
    try:
        monitor = DependencyInstallProgress(work, events.append, 60)
        monitor.log_path.write_text("Downloading https://example.org/" + name + "\n", encoding="utf-8")
        monitor(None)
        assert events[-1]["downloaded_bytes"] == len(b"downloaded wheel bytes")
    finally:
        clean_owned_fixture(root, tmp_path)


def test_abandoned_long_temporary_files_are_reaped_from_owned_namespace(tmp_path):
    root = managed_root(tmp_path)
    token = "a" * 32
    work = root / TEMP_NAMESPACE / ("install-" + token)
    work.mkdir(parents=True)
    (work / OWNER_MARKER).write_text(json.dumps({"kind": "autorepro_pip", "version": 1,
        "id": token, "created_at": time.time() - 90000}), encoding="utf-8")
    write_long_payload(work)
    try:
        result = reap_pip_temporary(root)
        assert not result["errors"], result
        assert len(result["removed"]) == 1
        assert result["freed_bytes"] >= len(b"long temporary payload")
        assert not work.exists()
    finally:
        clean_owned_fixture(root, tmp_path)
