"""Resource helpers use the same owned process boundary as Windows training."""
import sys

import pytest

import src.resource_manager as resources
from src.runtime_preparation import run_owned_process


def test_python_dataset_script_runs_in_a_chinese_path_without_file_associations(tmp_path):
    script = tmp_path / "数据 下载" / "download dataset.PY"
    script.parent.mkdir()
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1], 'rows.csv').write_text('value\\n1\\n2\\n', encoding='utf-8')\n"
        "print('下载完成')\n", encoding="utf-8")
    manager = resources.ResourceManager(data_root=tmp_path / "data")

    result = manager._download_real_dataset(
        "paper", "fixture", {"entry": "script:" + str(script)}, tmp_path / "数据 集")

    assert result["state"] == "downloaded", result
    assert result["rows"] == 2
    assert (tmp_path / "数据 集" / "dataset_full" / "rows.csv").exists()


@pytest.mark.parametrize("kind", ["dataset", "weights"])
def test_download_failure_keeps_non_utf8_diagnostics_without_reader_thread_crash(tmp_path, monkeypatch, kind):
    calls = []

    def localized_failure(command, **options):
        calls.append(command)
        return run_owned_process([sys.executable, "-c",
            "import sys;sys.stderr.buffer.write(b'download failed: \\xb4\\xed\\xce\\xf3');sys.exit(7)"],
            **options)

    monkeypatch.setattr(resources, "run_owned_process", localized_failure)
    manager = resources.ResourceManager(data_root=tmp_path / "data")
    if kind == "dataset":
        result = manager._download_real_dataset(
            "paper", "fixture", {"entry": "url:https://example.invalid/data.csv"}, tmp_path / "dataset")
        assert result["state"] == "unavailable"
        assert "download failed:" in result["detail"]
    else:
        result = manager.fetch_weights("paper", "https://example.invalid/weights.bin")
        assert result["state"] == "skipped"
    assert len(calls) == 1 and calls[0][0] == "curl"


def test_cancelled_dataset_script_cannot_leave_a_live_downloader(tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    survived = tmp_path / "survived"
    script = tmp_path / "下载器.py"
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "subprocess.Popen([sys.executable, '-c', " + repr(
            "import time;from pathlib import Path;time.sleep(1);Path(" + repr(str(survived)) + ").touch()") + "])\n"
        "Path(" + repr(str(ready)) + ").touch()\ntime.sleep(30)\n", encoding="utf-8")

    def cancellable(command, **options):
        def cancel_when_ready(process):
            if ready.exists():
                raise KeyboardInterrupt("user cancellation")
        return run_owned_process(command, on_poll=cancel_when_ready, **options)

    monkeypatch.setattr(resources, "run_owned_process", cancellable)
    manager = resources.ResourceManager(data_root=tmp_path / "data")
    with pytest.raises(KeyboardInterrupt, match="user cancellation"):
        manager._download_real_dataset(
            "paper", "fixture", {"entry": "script:" + str(script)}, tmp_path / "dataset")

    # A separate process waits beyond the child's scheduled write; the owned
    # job/group must already have removed it before cancellation is returned.
    run_owned_process([sys.executable, "-c", "import time;time.sleep(1.2)"])
    assert ready.exists() and not survived.exists()
