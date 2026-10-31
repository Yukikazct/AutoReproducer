"""Real pip backtracking can reject wheels before downloading their payloads."""
from __future__ import annotations

import http.server
import io
import json
import os
import re
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit
import zipfile

import pytest


def _candidate(name, version, dependency=None):
    buffer = io.BytesIO()
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        # A real payload makes the difference between metadata ranges and a
        # full wheel transfer measurable, without installing any package.
        archive.writestr(f"{name}/payload.bin", b"\0" * (1024 * 1024))
        fields = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        if dependency:
            fields += "Requires-Dist: " + dependency + "\n"
        archive.writestr(f"{metadata}/METADATA", fields)
        archive.writestr(f"{metadata}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{metadata}/RECORD", "")
    return buffer.getvalue()


@pytest.mark.parametrize("fast_deps", [False, True])
def test_pip_backtracking_uses_only_metadata_for_rejected_wheels(tmp_path, fast_deps):
    wheels = {
        "numpy-1.26.4-py3-none-any.whl": _candidate("numpy", "1.26.4"),
        "scipy-1.18.1-py3-none-any.whl": _candidate("scipy", "1.18.1", "numpy>=2"),
        "scipy-1.18.0-py3-none-any.whl": _candidate("scipy", "1.18.0", "numpy>=2"),
        "scipy-1.17.1-py3-none-any.whl": _candidate("scipy", "1.17.1", "numpy>=1.26"),
    }
    transfers = []

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self._serve(head=True)

        def do_GET(self):
            self._serve(head=False)

        def _serve(self, *, head):
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/simple/"):
                name = path.strip("/").split("/")[-1]
                payload = "\n".join(
                    f'<a href="/wheels/{filename}">{filename}</a>'
                    for filename in wheels if filename.startswith(name + "-"))
                encoded = payload.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                if not head:
                    self.wfile.write(encoded)
                return
            filename = path.rsplit("/", 1)[-1]
            if filename not in wheels:
                self.send_error(404)
                return
            payload = wheels[filename]
            requested = self.headers.get("Range")
            start, end = 0, len(payload) - 1
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                assert match is not None
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
            self.send_response(206 if requested else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            transfers.append({"filename": filename, "head": head, "range": requested,
                              "bytes": 0 if head else end - start + 1})
            if not head:
                self.wfile.write(payload[start:end + 1])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    report = tmp_path / "resolution.json"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONIOENCODING="utf-8", PIP_NO_INPUT="1")
    command = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
               "--disable-pip-version-check", "--no-cache-dir", "--report", str(report),
               "-i", f"http://127.0.0.1:{server.server_port}/simple", "numpy==1.26.4", "scipy"]
    if fast_deps:
        command.append("--use-feature=fast-deps")
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=environment, timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    selected = {item["metadata"]["name"]: item["metadata"]["version"]
                for item in json.loads(report.read_text("utf-8"))["install"]}
    assert selected == {"numpy": "1.26.4", "scipy": "1.17.1"}
    for version in ("1.18.1", "1.18.0"):
        filename = f"scipy-{version}-py3-none-any.whl"
        downloads = [row for row in transfers if row["filename"] == filename and not row["head"]]
        assert downloads, "The resolver must actually inspect and reject both incompatible candidates"
        if fast_deps:
            assert all(row["range"] for row in downloads)
            assert sum(row["bytes"] for row in downloads) < len(wheels[filename]) // 4
        else:
            assert any(row["range"] is None and row["bytes"] == len(wheels[filename]) for row in downloads)
