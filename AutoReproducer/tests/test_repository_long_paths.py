"""Actual Windows author processes and retained records beyond MAX_PATH."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.repository_runner import RepositoryRunner
from src.safety.paths import filesystem_path


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows extended-length paths')


@pytest.mark.parametrize('exit_code', [0, 7])
def test_author_process_retains_complete_logs_and_artifacts_in_deep_workspace(tmp_path, exit_code):
    parent = tmp_path / ('experiment_' + 'x' * max(1, 205 - len(str(tmp_path)) - 12))
    workspace = parent / 'repo'
    workspace.mkdir(parents=True)
    source = ("from pathlib import Path\n"
              "Path('output').mkdir()\n"
              "Path('output/predictions.json').write_text('{\"samples\": 8}', encoding='utf-8')\n"
              "print('作者命令完成', flush=True)\n"
              f"raise SystemExit({exit_code})\n")
    author = workspace / 'train.py'
    author.write_text(source, encoding='utf-8')
    records = parent / '.autorepro_repository_runs'
    assert len(str(records / ('a' * 32) / 'environment_prepare')) > 260
    executor = CodeExecutorAgent(None)
    runner = RepositoryRunner(executor=executor)
    try:
        result = runner.run(workspace, [{'id': 'train', 'kind': 'train', 'required': True,
            'argv': ['python', '-u', 'train.py'], 'cwd': '.', 'timeout_s': 15,
            'env': {}, 'artifacts': ['output/predictions.json']}],
            {'auto_prepare': False, 'requirements_txt': ''})
        assert result['success'] is (exit_code == 0), result['final']['stderr']
        step, = result['steps']
        assert step['executed'] and step['exit_code'] == exit_code
        assert step['cwd'] == str(workspace)
        assert step['stdout'] == '作者命令完成\n'
        assert Path(step['stdout_path']).read_text(encoding='utf-8') == step['stdout']
        run_dir = Path(result['run_dir'])
        assert len(str(run_dir / 'train.json')) > 260
        assert json.loads((run_dir / 'train.json').read_text(encoding='utf-8'))['exit_code'] == exit_code
        assert json.loads((run_dir / 'environment.json').read_text(encoding='utf-8'))['status'] == 'success'
        declaration, = step['artifacts']
        assert declaration['matched'], declaration
        artifact, = declaration['files']
        kept = Path(artifact['captured_to'])
        assert kept.is_relative_to(run_dir)
        payload = kept.read_bytes()
        assert json.loads(payload) == {'samples': 8}
        assert artifact['sha256'] == hashlib.sha256(payload).hexdigest()
        assert author.read_text(encoding='utf-8') == source
    finally:
        assert parent.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(filesystem_path(parent))


@pytest.mark.parametrize('deep', [False, True])
def test_record_junction_is_rejected_before_author_execution(tmp_path, deep):
    parent = tmp_path / ('p' * max(1, 240 - len(str(tmp_path)) - 1)) if deep else tmp_path
    workspace = parent / 'repo'
    workspace.mkdir(parents=True)
    external = tmp_path / 'external'
    external.mkdir()
    sentinel = external / 'keep.txt'
    sentinel.write_text('preserved', encoding='utf-8')
    records = Path(filesystem_path(parent / '.autorepro_repository_runs'))
    if deep:
        assert len(str(records)) > 260
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(records), str(external)],
                   capture_output=True, check=True)
    try:
        result = RepositoryRunner().run(workspace, [{'id': 'train',
            'argv': ['python', '-c', "raise AssertionError('must not execute')"]}], {})
        assert not result['executed']
        assert 'junction' in result['reason']
        assert list(external.iterdir()) == [sentinel]
        assert sentinel.read_text(encoding='utf-8') == 'preserved'
    finally:
        # Remove only the directory entry, never the target contents.
        records.rmdir()
