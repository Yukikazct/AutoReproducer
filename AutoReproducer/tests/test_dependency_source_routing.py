"""Reviewed wheel routing preserves pins and keeps ordinary packages on PyPI."""
from __future__ import annotations

from functools import partial
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest

from src.dependency_sources import scope_pytorch_sources


@pytest.mark.parametrize("flavor", ["cpu", "cu121"])
def test_reviewed_extra_index_is_scoped_to_explicit_native_packages(flavor):
    original = (f"# author pins\r\n--extra-index-url https://download.pytorch.org/whl/{flavor}\r\n"
                f"torch==2.5.1+{flavor}\r\ntorchvision==0.20.1+{flavor}\r\n"
                "numpy==1.26.4 # keep this comment\r\n"
                f"--find-links https://data.pyg.org/whl/torch-2.5.0+{flavor}.html\r\n")
    routed = scope_pytorch_sources(original, default_cpu_links=True)
    assert routed == original.replace(
        f"--extra-index-url https://download.pytorch.org/whl/{flavor}",
        f"--find-links https://download.pytorch.org/whl/{flavor}/torch/\r\n"
        f"--find-links https://download.pytorch.org/whl/{flavor}/torchvision/")
    assert "/torchaudio/" not in routed
    if flavor == "cu121":
        assert "/whl/cpu/" not in routed


def test_default_cpu_page_scopes_unsuffixed_pin_without_changing_requirement():
    original = "torch==2.5.1\nnumpy==1.26.4"
    assert scope_pytorch_sources(original) == original
    assert scope_pytorch_sources(original, default_cpu_links=True) == (
        original + "\n--find-links https://download.pytorch.org/whl/cpu/torch/\n")


def test_markers_comments_and_spacing_are_preserved():
    original = ('  --extra-index-url=https://download.pytorch.org/whl/cpu/  # native wheels\n'
                'torch == 2.5.1+cpu ; python_version >= "3.10"\n'
                'torchaudio==2.5.1+cpu; python_version >= "3.10"\n'
                'numpy==1.26.4\n')
    routed = scope_pytorch_sources(original)
    assert routed == original.replace(
        '  --extra-index-url=https://download.pytorch.org/whl/cpu/  # native wheels\n',
        '  --find-links https://download.pytorch.org/whl/cpu/torch/  # native wheels\n'
        '  --find-links https://download.pytorch.org/whl/cpu/torchaudio/\n')


@pytest.mark.parametrize("flavor", ["cpu", "cu121"])
def test_reviewed_graph_stack_preserves_matching_pyg_wheels_and_page(flavor):
    original = (f"--extra-index-url https://download.pytorch.org/whl/{flavor}\n"
                f"--find-links https://data.pyg.org/whl/torch-2.5.0+{flavor}.html\n"
                f"torch==2.5.1+{flavor}\ntorch-geometric==2.6.1\n"
                f"torch-scatter==2.1.2+pt25{flavor}\ntorch-sparse==0.6.18+pt25{flavor}\n"
                "numpy==1.26.4\n")
    assert scope_pytorch_sources(original) == original.replace(
        f"--extra-index-url https://download.pytorch.org/whl/{flavor}",
        f"--find-links https://download.pytorch.org/whl/{flavor}/torch/")


@pytest.mark.parametrize("extra", [
    "torch-scatter==2.1.2+pt25cpu",  # Missing the reviewed PyG source.
    "--find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html\ntorch-scatter==2.1.2+pt25cu121",
    "--find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html\ntorch-scatter==2.1.2+pt24cpu",
    "--find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html\ntorch-scatter==2.1.2",
    "--find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html\ntorch-scatter",
])
def test_unreviewed_pyg_source_or_abi_retains_original_routing(extra):
    original = "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu\n" + extra
    assert scope_pytorch_sources(original) == original


@pytest.mark.parametrize("native", [
    "torch", "torch>=2.5.1", "torch==2.5.*", "torch===2.5.1", "torch[extra]==2.5.1",
    "torch==2.5.1+cu121", "torch==2.5.1\ntorchvision", "torch==2.5.1\ntorch==2.5.1",
    "torchvision==0.20.1", "torch==2.5.1\npytorch-triton==3.1.0",
    "torch==2.5.1\ntorchtext==0.18.0", "torch==2.5.1\ntriton==3.1.0",
    'torch==2.5.1; python_version < "3.10"\ntorchvision==0.20.1',
])
def test_ambiguous_or_additional_native_dependencies_retain_original_text(native):
    original = "--extra-index-url https://download.pytorch.org/whl/cpu\n" + native
    assert scope_pytorch_sources(original, default_cpu_links=True) == original


@pytest.mark.parametrize("extra", [
    "--index-url https://private.invalid/simple", "-r nested.txt", "--no-index",
    "--extra-index-url https://private.invalid/simple", "--find-links https://private.invalid/wheels",
    "--extra-index-url https://download.pytorch.org/whl/cu118",
    "numpy @ https://private.invalid/numpy.whl", "not a requirement",
    "torchvision==0.20.1 \\\n    --hash=sha256:fixture",
])
def test_unreviewed_sources_and_formats_are_not_rewritten(extra):
    original = "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1\n" + extra
    assert scope_pytorch_sources(original, default_cpu_links=True) == original


def _wheel(directory: Path, name: str, version: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}/__init__.py", "")
        archive.writestr(f"{metadata}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{metadata}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{metadata}/RECORD", "")
    (directory / "index.html").write_text(f'<a href="{path.name}">{path.name}</a>', encoding="utf-8")
    return path


def test_real_pip_keeps_numpy_on_primary_and_exact_torch_on_scoped_page(tmp_path):
    # Both indexes offer NumPy. The supplemental index must cease to be an
    # ordinary package source, while its exact native wheel remains available.
    _wheel(tmp_path / "primary/numpy", "numpy", "1.26.4")
    _wheel(tmp_path / "supplemental/numpy", "numpy", "1.26.4")
    _wheel(tmp_path / "supplemental/torch", "torch", "2.5.1+cpu")
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    original = "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu\nnumpy==1.26.4\n"
    routed = scope_pytorch_sources(original).replace("https://download.pytorch.org/whl/cpu", base + "/supplemental")
    requirements = tmp_path / "installer.txt"
    requirements.write_text(routed, encoding="utf-8")
    report = tmp_path / "resolution.json"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONIOENCODING="utf-8", PIP_NO_INPUT="1")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--no-deps",
             "--disable-pip-version-check", "--no-cache-dir", "--report", str(report),
             "-i", base + "/primary", "-r", str(requirements)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment, timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    selected = {item["metadata"]["name"]: item for item in json.loads(report.read_text("utf-8"))["install"]}
    assert selected["numpy"]["download_info"]["url"].startswith(base + "/primary/numpy/")
    assert selected["torch"]["download_info"]["url"].startswith(base + "/supplemental/torch/")
    assert selected["torch"]["metadata"]["version"] == "2.5.1+cpu"
