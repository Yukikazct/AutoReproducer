"""A wrapped download failure relocates only the failing author's data root."""
import hashlib
from pathlib import Path

import pytest

from src.experiments.author_numeric_compatibility import run_entrypoint
from src.experiments.author_path_compatibility import DATA_ROOT, diagnose_temporary_root_error
from test_author_execution_plan import AuthorLLM
from test_discovered_repository_service import setup_service, use_preinstalled_cache, cpu_dependency_cache


LOADER = '''
from pathlib import Path
def load_dataset(root):
    if root == '/tmp/':
        try:
            raise FileNotFoundError(2, 'No such file or directory', '/tmp/cora.npz')
        except FileNotFoundError:
            raise Exception('Download failed! Make sure you have stable Internet connection')
    target = Path(root) / 'cora.npz'
    target.write_bytes(b'original-download-content')
    assert target.read_bytes() == b'original-download-content'
    return root
result = load_dataset(root='/tmp/')
'''


def failed_loader(tmp_path):
    script = tmp_path / 'train.py'
    script.write_text(LOADER, encoding='utf-8')
    sources = {'train.py': hashlib.sha256(script.read_bytes()).hexdigest()}
    try:
        run_entrypoint(script, [], tmp_path, sources)
    except Exception as exc:
        repair = diagnose_temporary_root_error(exc, script, tmp_path, sources)
        assert repair is not None
        return script, sources, repair
    pytest.fail('Expected the real wrapped missing-directory failure')


def test_wrapped_file_error_retries_with_task_local_root_and_preserves_source(tmp_path):
    script, sources, repair = failed_loader(tmp_path)
    original = script.read_bytes()
    result = run_entrypoint(script, [], tmp_path, sources, [repair])
    assert Path(result['result']) == tmp_path / DATA_ROOT
    assert script.read_bytes() == original
    assert (tmp_path / DATA_ROOT / 'cora.npz').read_bytes() == b'original-download-content'


@pytest.mark.parametrize('change', ['source', 'replacement', 'filename'])
def test_data_root_repair_rejects_changed_source_or_unrelated_destination(tmp_path, change):
    script, sources, repair = failed_loader(tmp_path)
    if change == 'source':
        script.write_text(LOADER + '\n# changed\n', encoding='utf-8')
    elif change == 'replacement':
        repair['replacement_root'] = '../shared-data'
    else:
        repair['failed_filename'] = '/tmp2/cora.npz'
    with pytest.raises(ValueError, match='Temporary data root'):
        run_entrypoint(script, [], tmp_path, sources, [repair])
    assert not (tmp_path / DATA_ROOT).exists()


def test_unrelated_missing_checkpoint_is_not_a_temporary_data_root_failure(tmp_path):
    assert diagnose_temporary_root_error(FileNotFoundError(2, 'missing', '/checkpoints/model.pt'),
                                        tmp_path / 'train.py', tmp_path, {}) is None


def test_service_retries_wrapped_download_path_error_and_finishes_training(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    script = fixture['workspace'] / 'train.py'
    source = LOADER + script.read_text('utf-8')
    script.write_text(source, encoding='utf-8')
    pinned = hashlib.sha256(script.read_bytes()).hexdigest()
    fixture['snapshot']['files']['train.py'] = pinned
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    runner.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert len(data['path_repair_attempts']) == 1
    repair = data['path_repair_attempts'][0]
    assert repair['failed_execution_dir'] != data['execution']['run_dir']
    assert 'Download failed!' in (Path(repair['failed_execution_dir']) / 'train.stderr.log').read_text('utf-8')
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert data['validation']['path_compatibility'][0]['original_root'] == '/tmp/'
    assert data['validation']['paper_metrics_verified'] is False
    assert (Path(data['run_dir']) / 'repo/train.py').read_text('utf-8') == source
    assert any('临时数据目录' in item for item in data['validation']['warnings'])
