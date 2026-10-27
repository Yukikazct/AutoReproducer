"""The session cleanup contract also applies to startup and normal child exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src import runtime_preparation as runtime
from src.posix_process_scope import _session_members


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX worker sessions")


def test_normal_worker_exit_preserves_status_and_reaps_nested_managed_group(tmp_path):
    ready = tmp_path / "owned.json"
    program = (
        "import json,os,subprocess,sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(Path(runtime.__file__).resolve().parents[1])!r});"
        "from src.posix_process_scope import managed_process_options;"
        "os.environ['AUTOREPRO_WORKER_SESSION']=str(os.getsid(0));"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "**managed_process_options());"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),child.pid]));"
        "sys.exit(7)"
    )
    confirmations = []

    def confirmed():
        leader, child = json.loads(ready.read_text())
        assert not any(not state.startswith("Z") for _, state in _session_members(leader).values())
        confirmations.append(child)

    result = runtime.run_owned_process([sys.executable, "-c", program],
                                       contain_session=True, on_cleanup=confirmed)
    assert result.returncode == 7
    assert len(confirmations) == 1


def test_failed_session_initialization_cannot_downgrade_to_group_cleanup(monkeypatch):
    notify = Mock()
    created = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    monkeypatch.setattr(runtime, "PosixSessionScope", Mock(side_effect=OSError("session query failed")))
    with pytest.raises(runtime.ProcessCleanupError) as failure:
        runtime.run_owned_process([sys.executable, "-c", "import time;time.sleep(60)"],
                                  contain_session=True, on_cleanup=notify)
    assert failure.value.process_cleanup_confirmed is False
    assert failure.value.cleanup_receipt["process_cleanup_confirmed"] is False
    notify.assert_not_called()
    assert len(created) == 1 and created[0].returncode is not None
