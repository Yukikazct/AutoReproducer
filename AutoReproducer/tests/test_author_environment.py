"""Author shell configuration is evidence, not a shell to execute."""
import hashlib
from pathlib import Path

import pytest

from src.author_environment import author_environment
from src.discovered_repository_plan import build_evidence_packet
from test_author_execution_plan import AuthorLLM
from test_discovered_repository_service import setup_service, use_preinstalled_cache, cpu_dependency_cache


def add_configuration(fixture, script):
    workspace = fixture['workspace']
    readme = workspace / 'README.md'
    readme.write_text(readme.read_text('utf-8') + '\nsource set_env.sh\n', encoding='utf-8')
    (workspace / 'set_env.sh').write_text(script, encoding='utf-8')
    train = workspace / 'train.py'
    train.write_text("import os\nfrom pathlib import Path\n"
                     "assert Path(os.environ['AUTHORDATA']).joinpath('input.txt').read_text() == 'complete original data'\n"
                     + train.read_text('utf-8'), encoding='utf-8')
    (workspace / 'inputs').mkdir()
    (workspace / 'inputs/input.txt').write_text('complete original data', encoding='utf-8')
    for name in ('README.md', 'set_env.sh', 'train.py', 'inputs/input.txt'):
        fixture['snapshot']['files'][name] = hashlib.sha256((workspace / name).read_bytes()).hexdigest()


def test_full_service_loads_pinned_author_environment_without_running_shell(tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    add_configuration(fixture, 'export AUTHOR_HOME=$(pwd)\nexport AUTHORDATA="${AUTHOR_HOME}/inputs"\n'
                      'export PYTHONPATH="$AUTHOR_HOME:$PYTHONPATH"\nsource activate obsolete_env\n')
    runner.llm = AuthorLLM(fixture)
    request['author_execution_only'] = True
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    data = result['data']
    assert data['validation']['author_execution'][0]['optimizer_steps'] == 40
    record = data['grounded_repository_plan']['author_environment']['AUTHORDATA']
    assert record['source_path'] == 'set_env.sh'
    assert record['source_sha256'] == fixture['snapshot']['files']['set_env.sh']
    assert Path(record['value']).name == 'inputs'


@pytest.mark.parametrize('value', ['$(touch should_not_exist)', '$UNKNOWN/inputs', '../outside', '.git'])
def test_unresolved_or_external_environment_does_not_execute_or_escape(tmp_path, monkeypatch, value):
    fixture, _, _, _ = setup_service(tmp_path, monkeypatch)
    add_configuration(fixture, 'export AUTHORDATA="' + value + '"\n')
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    records, warnings = author_environment(packet, fixture['workspace'])
    assert 'AUTHORDATA' not in records
    assert not (fixture['workspace'] / 'should_not_exist').exists()
    assert warnings
