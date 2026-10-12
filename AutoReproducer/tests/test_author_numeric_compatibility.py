"""Recorded type repairs preserve source bytes, command and native training."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.experiments.author_numeric_compatibility import diagnose_reciprocal_error, run_entrypoint
from src.method_adapters import read_json
from test_author_execution_plan import AuthorLLM
from test_discovered_repository_service import setup_service, use_preinstalled_cache, cpu_dependency_cache


def failed_call(tmp_path, expression='np.power(rowsum, -1)', values='[2, 4]'):
    script = tmp_path / 'train.py'
    script.write_text('import numpy as np\nrowsum = np.array(' + values + ', dtype=np.int32)\n'
                      'result = ' + expression + '\n', encoding='utf8')
    sources = {'train.py': hashlib.sha256(script.read_bytes()).hexdigest()}
    try:
        run_entrypoint(script, [], tmp_path, sources)
    except ValueError as exc:
        return script, sources, diagnose_reciprocal_error(exc, script, tmp_path, sources)
    raise AssertionError('Expected an actual original integer-power failure')


def test_recorded_float_reciprocal_preserves_math_and_original_file(tmp_path):
    script, sources, repair = failed_call(tmp_path)
    assert repair['base_dtype'] == 'int32'
    original = script.read_bytes()
    result = run_entrypoint(script, [repair], tmp_path, sources)['result']
    np.testing.assert_array_equal(result, [.5, .25])
    assert script.read_bytes() == original


@pytest.mark.parametrize('expression,values', [
    ('np.power(rowsum, -2)', '[2, 4]'),
    ('np.power(rowsum, -1, dtype=np.int64)', '[2, 4]'),
])
def test_other_power_expressions_are_not_automatically_repaired(tmp_path, expression, values):
    _, _, repair = failed_call(tmp_path, expression, values)
    assert repair is None


def test_recorded_repair_cannot_apply_after_source_changes(tmp_path):
    script, sources, repair = failed_call(tmp_path)
    script.write_text(script.read_text() + '\nprint("changed")\n')
    with pytest.raises(ValueError, match='pinned entrypoint'):
        run_entrypoint(script, [repair], tmp_path, sources)


def test_full_service_retries_real_numeric_failure_then_observes_original_training(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    script = fixture['workspace'] / 'train.py'
    prefix = ('import numpy as np\n'
              'rowsum = np.array([0, 2, 4], dtype=np.uint32)\n'
              'inverse = np.power(rowsum, -1)\n'
              'inverse[np.isinf(inverse)] = 0.\n'
              'assert np.array_equal(inverse, [0., .5, .25])\n')
    source = prefix + script.read_text('utf8')
    script.write_text(source, encoding='utf8')
    pinned = hashlib.sha256(script.read_bytes()).hexdigest()
    fixture['snapshot']['files']['train.py'] = pinned
    next(item for item in fixture['proposal']['citations'] if item['id'] == 'script')['quote'] = source
    runner.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['validation']['paper_metrics_verified'] is False
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    assert len(data['numeric_repair_attempts']) == 1
    repair = data['numeric_repair_attempts'][0]
    assert repair['original_expression'] == 'np.power(rowsum, -1)'
    assert repair['failed_execution_dir'] != data['execution']['run_dir']
    assert 'Integers to negative' in (Path(repair['failed_execution_dir']) / 'train.stderr.log').read_text('utf-8')
    assert hashlib.sha256((Path(data['run_dir']) / 'repo/train.py').read_bytes()).hexdigest() == pinned
    assert data['validation']['numeric_compatibility'][0]['source_sha256'] == pinned
    manifest = read_json(Path(data['run_dir']) / 'adapter_manifest.json')
    config = Path(data['run_dir']) / 'repo/_autorepro_capture/config.json'
    assert manifest['files']['_autorepro_capture/config.json'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert any('浮点倒数兼容' in item for item in data['validation']['warnings'])
