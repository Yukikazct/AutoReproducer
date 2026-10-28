"""Exercise installer source routing without changing the frozen requirements."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce


REQUIREMENTS = 'numpy==1.26.4\ntorch==2.5.1\n'
CPU_LINKS = 'https://mirrors.aliyun.com/pytorch-wheels/cpu/'
MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple'


@pytest.mark.parametrize('supplemental', ['', '--extra-index-url https://download.pytorch.org/whl/cpu\n'])
def test_owned_install_scopes_wheels_and_preserves_original_requirements(monkeypatch, tmp_path, supplemental):
    monkeypatch.setattr(ce, 'PIP_INDEX_URL', MIRROR)
    monkeypatch.setattr(ce, 'PIP_FIND_LINKS', CPU_LINKS)
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {'auto_prepare': True}
    instance.deps_cache_root = tmp_path / 'deps'
    original = tmp_path / 'requirements.txt'
    original.write_text(supplemental + REQUIREMENTS, encoding='utf-8')
    seen = []

    def invoke(command, **kwargs):
        installer_file = Path(command[command.index('-r') + 1])
        assert installer_file != original
        installer = installer_file.read_text('utf-8')
        assert '--extra-index-url' not in installer
        assert '--find-links https://download.pytorch.org/whl/cpu/torch/' in installer
        assert REQUIREMENTS in installer
        assert CPU_LINKS not in command
        assert command[command.index('-i') + 1] == MIRROR
        assert '--use-feature=fast-deps' in command
        assert kwargs['env']['PIP_RETRIES'] == '1'
        assert kwargs['env']['PIP_DEFAULT_TIMEOUT'] == '15'
        seen.append(installer_file)
        return subprocess.CompletedProcess(command, 0, 'done', '')

    monkeypatch.setattr(instance, '_invoke_dependency_command', invoke)
    instance._run_dependency_command(
        ['python', '-m', 'pip', 'install', '-i', MIRROR, '--find-links', CPU_LINKS,
         '-r', str(original)], timeout=1800, env={})
    assert original.read_text('utf-8') == supplemental + REQUIREMENTS
    assert len(seen) == 1 and not seen[0].exists()


@pytest.mark.parametrize('auto_prepare,source', [(False, MIRROR), (True, 'https://private.invalid/simple')])
def test_explicit_sources_and_unreviewed_installs_keep_routing(monkeypatch, tmp_path, auto_prepare, source):
    monkeypatch.setattr(ce, 'PIP_INDEX_URL', source)
    monkeypatch.setattr(ce, 'PIP_FIND_LINKS', CPU_LINKS)
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {'auto_prepare': auto_prepare}
    instance.deps_cache_root = tmp_path / 'deps'
    original = tmp_path / 'requirements.txt'
    original.write_text(REQUIREMENTS, encoding='utf-8')

    def invoke(command, **kwargs):
        assert command[command.index('-r') + 1] == str(original)
        assert CPU_LINKS in command
        assert '--use-feature=fast-deps' not in command
        assert kwargs['env']['PIP_RETRIES'] == '7'
        return subprocess.CompletedProcess(command, 0, 'done', '')

    monkeypatch.setattr(instance, '_invoke_dependency_command', invoke)
    instance._run_dependency_command(
        ['python', '-m', 'pip', 'install', '-i', source, '--find-links', CPU_LINKS,
         '-r', str(original)], timeout=300, env={'PIP_RETRIES': '7'})
