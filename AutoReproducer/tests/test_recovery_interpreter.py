"""A benchmark worker must keep the selected virtual environment's imports."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.benchmark.recovery_adapter import RecoveryAdapter, _python_entrypoint


def test_selected_venv_keeps_its_prefix_and_private_imports(tmp_path):
    environment = tmp_path / "product-venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(environment)], check=True)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Put an import in this product environment only, without any pip/network.
    site_directory = subprocess.check_output(
        [str(executable), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
    Path(site_directory, "recovery_product_marker.py").write_text("value = 'selected-product'\n", "utf-8")
    baseline, enhanced = tmp_path / "baseline", tmp_path / "enhanced"
    baseline.mkdir()
    enhanced.mkdir()
    adapter = RecoveryAdapter(product_roots={"A": baseline, "B": enhanced, "C": enhanced},
        oracle_paths={}, product_python=executable)
    assert adapter.author_python == adapter.python
    result = subprocess.run([adapter.python, "-I", "-c",
        "import json,sys,recovery_product_marker; print(json.dumps([sys.prefix,recovery_product_marker.value]))"],
        capture_output=True, text=True, check=True)
    prefix, marker = json.loads(result.stdout)
    assert Path(prefix).resolve() == environment.resolve()
    assert marker == "selected-product"
    if os.name != "nt":
        assert executable.is_symlink(), "The POSIX regression must exercise a venv symlink"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_interpreter_is_rejected_before_starting_a_worker(tmp_path, kind):
    target = tmp_path / kind
    if kind == "directory":
        target.mkdir()
    with pytest.raises(FileNotFoundError):
        _python_entrypoint(target)
