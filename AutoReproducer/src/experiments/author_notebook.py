"""Execute every code cell of a pinned author notebook in the managed Python."""
import json
import os
from pathlib import Path
import sys


def run_notebook(script, runtime, identifier, directory):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpec

    notebook = nbformat.read(str(script), as_version=4)
    if (notebook.nbformat != 4 or not notebook.cells
            or any(cell.cell_type not in {'code', 'markdown', 'raw'} for cell in notebook.cells)):
        raise ValueError('Pinned author notebook has an unsupported structure')
    kernel = AsyncKernelManager(kernel_name='python3')
    observer = runtime / 'notebook_kernel_hook.py'
    os.environ['AUTOREPRO_CAPTURE_DIR'] = str(runtime)
    os.environ['AUTOREPRO_NOTEBOOK_STEP'] = identifier
    kernel._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher',
        '-f', '{connection_file}', '--IPKernelApp.exec_files=' + str(observer)],
        display_name='AutoReproducer author Python', language='python')
    client = NotebookClient(notebook, km=kernel, timeout=None, allow_errors=False,
        resources={'metadata': {'path': str(directory)}})
    output = runtime / ('executed_' + identifier + '.ipynb')
    failure = runtime / ('notebook_failure_' + identifier + '.json')
    failure.unlink(missing_ok=True)
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                try:
                    client.execute_cell(cell, index)
                except BaseException as exc:
                    failure.write_text(json.dumps({'cell': index + 1, 'error': str(exc)[-4000:]},
                                                  ensure_ascii=False), encoding='utf-8')
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    raise RuntimeError(f'原作者 Notebook 第 {index + 1} 单元执行失败：{exc}') from exc
            # A separate observer request runs after every original cell; it
            # is never appended to or substituted for the author notebook.
            observer_cell = nbformat.v4.new_code_cell('_autorepro_write_steps()')
            notebook.cells.append(observer_cell)
            try:
                client.execute_cell(observer_cell, len(notebook.cells) - 1)
            finally:
                notebook.cells.pop()
    finally:
        nbformat.write(notebook, str(output))
    count_path = runtime / ('notebook_steps_' + identifier + '.json')
    try:
        observed = json.loads(count_path.read_text('utf-8'))
        return int(observed['optimizer_steps'])
    except (ValueError, OSError, KeyError):
        return 0
