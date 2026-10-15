"""Execute tiny author notebooks with a real kernel when one is available."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest


PYTHON = shutil.which('python')
AUTHOR_NOTEBOOK = Path(__file__).resolve().parents[1] / 'src/experiments/author_notebook.py'
KERNEL_HOOK = Path(__file__).resolve().parents[1] / 'src/experiments/notebook_kernel_hook.py'


@pytest.fixture
def notebook_python():
    if not PYTHON or subprocess.run([PYTHON, '-c', 'import nbclient,ipykernel,nbformat'],
                                    capture_output=True).returncode:
        pytest.skip('No installed training Python with nbclient/ipykernel')
    return PYTHON


def notebook(path, sources):
    path.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                   'outputs': [], 'source': source} for source in sources]}), encoding='utf-8')


def command(python, source, runtime):
    return [python, '-c', 'import sys; sys.path.insert(0, sys.argv[1]); '
            'from author_notebook import run_notebook; '
            'print(run_notebook(sys.argv[2], __import__("pathlib").Path(sys.argv[3]), "train", '
            '__import__("pathlib").Path(sys.argv[4])))',
            str(AUTHOR_NOTEBOOK.parent), str(source), str(runtime), str(source.parent)]


def test_notebook_runs_in_order_and_stops_at_failing_cell(tmp_path, notebook_python):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    shutil.copyfile(KERNEL_HOOK, runtime / 'notebook_kernel_hook.py')
    source = tmp_path / 'train.ipynb'
    notebook(source, ["from pathlib import Path\nPath('order.txt').write_text('first')",
                      "from pathlib import Path\nPath('order.txt').write_text(Path('order.txt').read_text() + ',second')",
                      "raise ValueError('intentional cell failure')",
                      "from pathlib import Path\nPath('skipped.txt').write_text('must not run')"])
    run = subprocess.run(command(notebook_python, source, runtime), capture_output=True,
                         text=True, timeout=45)
    assert run.returncode != 0 and '第 3 单元' in run.stderr
    assert (tmp_path / 'order.txt').read_text() == 'first,second'
    assert not (tmp_path / 'skipped.txt').exists()
    assert json.loads((runtime / 'notebook_failure_train.json').read_text())['cell'] == 3
    executed = json.loads((runtime / 'executed_train.ipynb').read_text())
    assert len(executed['cells']) == 4
    assert executed['cells'][0]['execution_count'] == 1
    assert executed['cells'][1]['execution_count'] == 2


@pytest.mark.skipif(os.name != 'posix', reason='POSIX signal and process probe')
def test_notebook_interrupt_releases_kernel(tmp_path, notebook_python):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    shutil.copyfile(KERNEL_HOOK, runtime / 'notebook_kernel_hook.py')
    source = tmp_path / 'long.ipynb'
    notebook(source, ["import os\nfrom pathlib import Path\nPath('kernel.pid').write_text(str(os.getpid()))",
                      'import time\ntime.sleep(120)'])
    process = subprocess.Popen(command(notebook_python, source, runtime),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    kernel_pid = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (tmp_path / 'kernel.pid').exists():
            if process.poll() is not None:
                pytest.fail('Notebook process exited before creating its kernel')
            time.sleep(.1)
        assert (tmp_path / 'kernel.pid').exists()
        kernel_pid = int((tmp_path / 'kernel.pid').read_text())
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=20)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(kernel_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            pytest.fail('Notebook kernel survived interruption')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        if kernel_pid is not None:
            try:
                os.kill(kernel_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
