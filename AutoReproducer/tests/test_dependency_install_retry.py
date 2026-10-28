"""Source fallback and user-run retry contracts; every pip invocation is mocked."""
import base64
import csv
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
from src.agents.code_executor import CodeExecutorAgent


REQUIREMENTS = "numpy==1.26.4\npandas==2.2.3\ntorch==2.5.1"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
ALIYUN = "https://mirrors.aliyun.com/pypi/simple/"
USTC = "https://pypi.mirrors.ustc.edu.cn/simple/"
OFFICIAL = "https://pypi.org/simple"
CUDA_REQUIREMENTS = ("--extra-index-url https://download.pytorch.org/whl/cu121\n"
                     "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\nnumpy==1.26.4")
NO_CANDIDATES = ("ERROR: Could not find a version that satisfies the requirement numpy==1.26.4 "
                 "(from versions: none)\nERROR: No matching distribution found for numpy==1.26.4")


@pytest.fixture(autouse=True)
def isolated_cache_and_public_defaults(monkeypatch, tmp_path):
    ce._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce, "DEPS_CACHE_ROOT", tmp_path / "deps")
    monkeypatch.setattr(ce, "PIP_INDEX_URL", MIRROR)
    monkeypatch.setattr(ce, "PIP_FIND_LINKS", "https://mirrors.aliyun.com/pytorch-wheels/cpu/")
    yield
    ce._INSTALLED_DEPS.clear()


def executor():
    instance = CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {"requirements_txt": REQUIREMENTS}
    return instance


def prepare(instance, tmp_path, name="prepare"):
    directory = tmp_path / name
    directory.mkdir()
    return instance._ensure_local_deps(str(directory))


def install_responses(monkeypatch, responses):
    calls = []

    def run(cmd, **kwargs):
        assert cmd[1:4] == ["-m", "pip", "install"], "tests may only invoke mocked pip"
        index = cmd[cmd.index("-i") + 1]
        req_file = Path(cmd[cmd.index("-r") + 1])
        calls.append({"cmd": cmd, "index": index, "requirements": req_file.read_text(encoding="utf-8"),
                      "kwargs": kwargs})
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return subprocess.CompletedProcess(cmd, response[0], stdout=response[1], stderr=response[2])

    monkeypatch.setattr(ce.subprocess, "run", run)
    monkeypatch.setattr(CodeExecutorAgent, "_run_dependency_command",
                        lambda instance, command, **kwargs: run(command, **kwargs))
    return calls


def test_missing_default_mirror_candidates_try_another_domestic_source_with_identical_pins(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", "")])
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [c["index"] for c in calls] == [MIRROR, ALIYUN]
    assert [c["requirements"] for c in calls] == [REQUIREMENTS, REQUIREMENTS]
    assert "--find-links" in calls[0]["cmd"]
    assert "--find-links" not in calls[1]["cmd"]
    assert "--upgrade" in calls[1]["cmd"]
    assert all(c["kwargs"]["timeout"] == ce.LOCAL_PIP_TIMEOUT for c in calls)
    assert (Path(instance._deps_dir) / ".ready").is_file()
    records = instance.env_config["dependency_install_attempts"]
    assert [r["success"] for r in records] == [False, True]
    assert "from versions: none" in records[0]["diagnostic"]


@pytest.mark.parametrize("wheel_index", ["cpu", "cu121"])
def test_reviewed_pytorch_extra_index_is_preserved_when_primary_mirror_fails(monkeypatch, tmp_path, wheel_index):
    instance = executor()
    requirements = f"--extra-index-url https://download.pytorch.org/whl/{wheel_index}\n" + REQUIREMENTS
    instance.env_config["requirements_txt"] = requirements
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", "")])
    assert prepare(instance, tmp_path) is None
    assert [call["index"] for call in calls] == [MIRROR, ALIYUN]
    assert [call["requirements"] for call in calls] == [requirements, requirements]


def test_cuda_install_total_timeout_does_not_restart_completed_download_on_another_source(monkeypatch, tmp_path):
    first_timeout = subprocess.TimeoutExpired(["pip"], 1800, output=b"extracting CUDA wheel",
                                             stderr=b"partial target copy")
    calls = install_responses(monkeypatch, [first_timeout])
    instance = executor()
    instance.env_config.update(requirements_txt=CUDA_REQUIREMENTS, auto_prepare=True)
    diagnostic = prepare(instance, tmp_path)
    assert [call["index"] for call in calls] == [MIRROR]
    assert calls[0]["kwargs"]["timeout"] == 1800
    assert calls[0]["requirements"] == CUDA_REQUIREMENTS
    records = instance.env_config["dependency_install_attempts"]
    assert len(records) == 1 and records[0]["timeout_s"] == 1800
    assert records[0]["timed_out"] and records[0]["failure_kind"] == "installation_timeout"
    assert "extracting CUDA wheel" in records[0]["diagnostic"]
    assert "partial target copy" in records[0]["diagnostic"]
    assert "安装进程总时限" in diagnostic
    assert "当前公共镜像未能提供依赖" not in repr(instance.logger.mock_calls)
    assert instance.env_config["dependency_failure_kind"] == "installation_timeout"


def test_explicit_official_cuda_install_has_same_long_budget_without_self_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "PIP_INDEX_URL", OFFICIAL)
    timeout = subprocess.TimeoutExpired(["pip"], 1800, output="partial wheel copy")
    calls = install_responses(monkeypatch, [timeout])
    instance = executor()
    instance.env_config.update(requirements_txt=CUDA_REQUIREMENTS, auto_prepare=True)
    diagnostic = prepare(instance, tmp_path)
    assert len(calls) == 1 and calls[0]["kwargs"]["timeout"] == 1800
    assert calls[0]["requirements"] == CUDA_REQUIREMENTS
    assert "依赖安装超时(1800s)" in diagnostic and "partial wheel copy" in diagnostic
    assert instance.env_config["dependency_install_attempts"][0]["timeout_s"] == 1800


def test_fixed_default_cpu_torch_gets_cold_install_budget_without_changing_pins(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(0, 'installed', '')])
    instance = executor()
    instance.env_config['auto_prepare'] = True
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 1 and calls[0]['kwargs']['timeout'] == ce.LOCAL_CPU_PIP_TIMEOUT
    assert calls[0]['requirements'] == REQUIREMENTS


def test_small_auto_prepared_stack_keeps_generic_budget(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(0, 'installed', '')])
    instance = executor()
    instance.env_config.update(auto_prepare=True, requirements_txt='numpy==1.26.4')
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 1 and calls[0]['kwargs']['timeout'] == ce.LOCAL_PIP_TIMEOUT


@pytest.mark.parametrize("starting_source", [MIRROR, ALIYUN, USTC, OFFICIAL])
@pytest.mark.parametrize("wheel", ["cpu", "cu121"])
def test_reviewed_stack_receives_full_budget_on_its_first_source(monkeypatch, tmp_path, starting_source, wheel):
    monkeypatch.setattr(ce, "PIP_INDEX_URL", starting_source)
    requirements = f"--extra-index-url https://download.pytorch.org/whl/{wheel}\ntorch==2.5.1+{wheel}"
    calls = install_responses(monkeypatch, [(0, "download and extraction completed", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=True)
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 1 and calls[0]["index"] == starting_source
    assert calls[0]["kwargs"]["timeout"] == 1800
    assert calls[0]["requirements"] == requirements
    assert instance.env_config["dependency_install_attempts"][0]["timeout_s"] == 1800


@pytest.mark.parametrize("private_source", ["links", "requirements", "extra", "compact_index", "compact_links",
                                           "abbreviated_index", "abbreviated_links"])
def test_explicit_official_index_with_private_cuda_sources_keeps_generic_budget(monkeypatch, tmp_path, private_source):
    monkeypatch.setattr(ce, "PIP_INDEX_URL", OFFICIAL)
    requirements = CUDA_REQUIREMENTS
    if private_source == "links":
        monkeypatch.setattr(ce, "PIP_FIND_LINKS", "https://packages.company.invalid/wheels")
    elif private_source == "requirements":
        requirements = "--index-url https://packages.company.invalid/simple\n" + requirements
    elif private_source == "compact_index":
        requirements = "-ihttps://packages.company.invalid/simple\n" + requirements
    elif private_source == "compact_links":
        requirements = "-fhttps://packages.company.invalid/wheels\n" + requirements
    elif private_source == "abbreviated_index":
        requirements = "--index-u=https://packages.company.invalid/simple\n" + requirements
    elif private_source == "abbreviated_links":
        requirements = "--find-li=https://packages.company.invalid/wheels\n" + requirements
    else:
        requirements += "\n--extra-index-url https://packages.company.invalid/simple"
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES)])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=True)
    assert prepare(instance, tmp_path) is not None
    assert len(calls) == 1 and calls[0]["kwargs"]["timeout"] == 300


@pytest.mark.parametrize("requirements,auto_prepare", [
    ("--extra-index-url https://download.pytorch.org/whl/cu121\n"
     "torch==2.5.1+cpu\nnumpy==1.26.4", True),
    ("--extra-index-url https://download.pytorch.org/whl/cu121\n" + REQUIREMENTS, True),
    ("torch==2.5.1+cu121\nnumpy==1.26.4", True),
    ("--extra-index-url https://download.pytorch.org/whl/cu121\n"
     "torch==2.5.1+cu118", True),
    (CUDA_REQUIREMENTS, False),
])
def test_only_matching_auto_prepared_torch_wheels_get_long_retry(monkeypatch, tmp_path, requirements, auto_prepare):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=auto_prepare)
    assert prepare(instance, tmp_path) is None
    assert [call["kwargs"]["timeout"] for call in calls] == [300, 300]
    assert [record["timeout_s"] for record in instance.env_config["dependency_install_attempts"]] == [300, 300]


def test_auto_prepared_cpu_stack_gets_long_budget_on_first_and_fallback_mirrors(monkeypatch, tmp_path):
    requirements = "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu\nnumpy==1.26.4"
    calls = install_responses(monkeypatch, [(1, "", "ConnectionError: mirror unavailable"), (0, "installed", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=True)
    assert prepare(instance, tmp_path) is None
    assert [call["kwargs"]["timeout"] for call in calls] == [1800, 1800]
    assert [call["requirements"] for call in calls] == [requirements] * 2


@pytest.mark.parametrize("private_source", ["index", "links", "requirements", "extra", "no-index",
                                           "compact_index", "compact_links", "abbreviated_index", "abbreviated_links"])
def test_pinned_cuda_retains_explicit_sources_and_budget(monkeypatch, tmp_path, private_source):
    requirements = CUDA_REQUIREMENTS
    if private_source == "index":
        monkeypatch.setattr(ce, "PIP_INDEX_URL", "https://packages.company.invalid/simple")
    elif private_source == "links":
        monkeypatch.setattr(ce, "PIP_FIND_LINKS", "https://packages.company.invalid/wheels")
    elif private_source == "requirements":
        requirements = "--index-url https://packages.company.invalid/simple\n" + requirements
    elif private_source == "extra":
        requirements += "\n--extra-index-url https://packages.company.invalid/simple"
    elif private_source == "compact_index":
        requirements = "-ihttps://packages.company.invalid/simple\n" + requirements
    elif private_source == "compact_links":
        requirements = "-fhttps://packages.company.invalid/wheels\n" + requirements
    elif private_source == "abbreviated_index":
        requirements = "--index-u=https://packages.company.invalid/simple\n" + requirements
    elif private_source == "abbreviated_links":
        requirements = "--find-li=https://packages.company.invalid/wheels\n" + requirements
    else:
        requirements = "--no-index\n" + requirements
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES)])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=True)
    assert prepare(instance, tmp_path) is not None
    assert len(calls) == 1 and calls[0]["requirements"] == requirements
    assert calls[0]["kwargs"]["timeout"] == 300


def test_cuda_native_health_rebuild_reuses_successful_domestic_source(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", ""),
                                           (0, "rebuilt", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=CUDA_REQUIREMENTS, auto_prepare=True,
                               dependency_health_check=True)
    monkeypatch.setattr(instance, "_check_local_dependency_health",
                        Mock(side_effect=["broken native DLL", None]))
    assert prepare(instance, tmp_path) is None
    assert [call["index"] for call in calls] == [MIRROR, ALIYUN, ALIYUN]
    assert [call["kwargs"]["timeout"] for call in calls] == [1800, 1800, 1800]
    assert [call["requirements"] for call in calls] == [CUDA_REQUIREMENTS] * 3


def graphbolt_wheel_fixture(directory, record_state):
    """An import failure plus real wheel metadata; no native library is loaded."""
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / 'dgl'
    graphbolt = package / 'graphbolt'
    graphbolt.mkdir(parents=True)
    (package / '__init__.py').write_text(
        'from pathlib import Path\nimport torch\n'
        "version = torch.__version__.split('+')[0]\n"
        "library = Path(__file__).parent / 'graphbolt' / f'graphbolt_pytorch_{version}.dll'\n"
        "raise FileNotFoundError(f'Cannot find DGL C++ graphbolt library at {library}')\n",
        encoding='utf-8')
    (graphbolt / '__init__.py').write_text('# Original GraphBolt loader location.\n', encoding='utf-8')
    (graphbolt / 'graphbolt_pytorch_2.2.1.dll').write_bytes(b'fixture binary for a different Torch version')
    (directory / 'torch.py').write_text("__version__ = '2.5.1'\n", encoding='utf-8')
    for name, version in [('dgl', '2.2.1'), ('torch', '2.5.1')]:
        metadata = directory / f'{name}-{version}.dist-info'
        metadata.mkdir()
        (metadata / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', encoding='utf-8')
        if name == 'dgl' and record_state != 'missing':
            paths = ['dgl/__init__.py', 'dgl/graphbolt/__init__.py',
                     'dgl/graphbolt/graphbolt_pytorch_2.2.1.dll', 'dgl-2.2.1.dist-info/METADATA']
            with (metadata / 'RECORD').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                for path in paths:
                    raw = (directory / path).read_bytes()
                    digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode('ascii')
                    writer.writerow([path, 'sha256=' + digest, str(len(raw))])
                if record_state == 'declared_missing':
                    writer.writerow(['dgl/graphbolt/graphbolt_pytorch_2.5.1.dll', '', ''])
                writer.writerow(['dgl-2.2.1.dist-info/RECORD', '', ''])


def graphbolt_install_responses(monkeypatch, record_state):
    real_run = subprocess.run
    calls = []

    def run(instance, command, **kwargs):
        if command[1:4] == ['-m', 'pip', 'install']:
            directory = Path(command[command.index('--target') + 1])
            manifest = Path(command[command.index('-r') + 1]).read_text(encoding='utf-8')
            assert not list(directory.iterdir()), 'every mocked install must receive a clean target'
            graphbolt_wheel_fixture(directory, record_state)
            calls.append({'command': command, 'requirements': manifest, 'directory': directory})
            return subprocess.CompletedProcess(command, 0, 'Successfully installed dgl-2.2.1 torch-2.5.1', '')
        assert command[1:3] == ['-I', '-c'], 'only the real isolated dependency health probe may execute'
        return real_run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', **kwargs)

    monkeypatch.setattr(CodeExecutorAgent, '_run_dependency_command', run)
    return calls


def test_graphbolt_wheel_without_the_requested_torch_abi_stops_identical_reinstallation(monkeypatch, tmp_path):
    calls = graphbolt_install_responses(monkeypatch, 'other_versions_only')
    instance = executor()
    requirements = 'dgl==2.2.1\ntorch==2.5.1'
    instance.env_config.update(requirements_txt=requirements, dependency_health_check=True)
    diagnostic = prepare(instance, tmp_path)
    assert len(calls) == 1, 'the installed wheel RECORD proves that reinstalling it cannot supply this Torch ABI'
    assert diagnostic and 'graphbolt_pytorch_2.5.1.dll' in diagnostic
    assert 'graphbolt_pytorch_2.2.1.dll' in diagnostic
    assert instance.env_config['dependency_failure_kind'] == 'native_abi_incompatible'
    attempts = instance.env_config['dependency_install_attempts']
    assert len(attempts) == 1 and attempts[0]['exit_code'] == 0 and not attempts[0]['success']
    assert attempts[0]['failure_kind'] == 'native_abi_incompatible'
    assert 'graphbolt_pytorch_2.5.1.dll' in attempts[0]['health_diagnostic']
    assert not instance.env_config.get('dependency_cache_repairs')
    assert instance._deps_dir is None and not (calls[0]['directory'] / '.ready').exists()
    assert calls[0]['requirements'] == requirements


@pytest.mark.parametrize('record_state', ['declared_missing', 'missing'])
def test_graphbolt_missing_declared_file_or_record_still_allows_one_cache_rebuild(monkeypatch, tmp_path, record_state):
    calls = graphbolt_install_responses(monkeypatch, record_state)
    instance = executor()
    requirements = 'dgl==2.2.1\ntorch==2.5.1'
    instance.env_config.update(requirements_txt=requirements, dependency_health_check=True)
    diagnostic = prepare(instance, tmp_path)
    assert diagnostic and 'graphbolt_pytorch_2.5.1.dll' in diagnostic
    assert len(calls) == 2, 'a missing file or manifest is not evidence that the wheel lacks the requested ABI'
    assert all(call['requirements'] == requirements for call in calls)
    assert instance.env_config['dependency_failure_kind'] != 'native_abi_incompatible'
    assert all(attempt['failure_kind'] != 'native_abi_incompatible'
               for attempt in instance.env_config['dependency_install_attempts'])
    assert [attempt['phase'] for attempt in instance.env_config['dependency_health_attempts']] == ['installed', 'rebuild']
    assert instance.env_config['dependency_cache_repairs'][0]['status'] == 'failed'
    assert instance._deps_dir is None and not (calls[-1]['directory'] / '.ready').exists()


@pytest.mark.parametrize('absolute', [False, True])
def test_graphbolt_unsafe_record_cannot_establish_an_abi_mismatch(monkeypatch, tmp_path, absolute):
    directory = tmp_path / 'wheel'
    graphbolt_wheel_fixture(directory, 'other_versions_only')
    outside = tmp_path / 'outside.dll'
    outside.write_bytes(b'outside evidence must not be read')
    record = directory / 'dgl-2.2.1.dist-info/RECORD'
    with record.open('a', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerow([str(outside) if absolute else '../outside.dll', '', ''])
    real_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        assert path != outside, 'ABI evidence must not read files outside the managed environment'
        return real_read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded_read_bytes)
    detail = f'FileNotFoundError: Cannot find DGL C++ graphbolt library at {directory / "dgl/graphbolt/graphbolt_pytorch_2.5.1.dll"}'
    assert ce._graphbolt_abi_evidence([directory], detail) is None
    assert outside.stat().st_size == len(b'outside evidence must not be read')


@pytest.mark.parametrize('filename', ['METADATA', 'RECORD'])
def test_graphbolt_linked_metadata_cannot_establish_an_abi_mismatch(tmp_path, require_symlinks, filename):
    directory = tmp_path / 'wheel'
    graphbolt_wheel_fixture(directory, 'other_versions_only')
    source = directory / 'dgl-2.2.1.dist-info' / filename
    outside = tmp_path / filename
    source.rename(outside)
    source.symlink_to(outside)
    before = outside.read_bytes()
    detail = f'FileNotFoundError: Cannot find DGL C++ graphbolt library at {directory / "dgl/graphbolt/graphbolt_pytorch_2.5.1.dll"}'
    assert ce._graphbolt_abi_evidence([directory], detail) is None
    assert outside.read_bytes() == before


def test_graphbolt_mismatched_torch_metadata_cannot_establish_an_abi_mismatch(tmp_path):
    directory = tmp_path / 'wheel'
    graphbolt_wheel_fixture(directory, 'other_versions_only')
    (directory / 'torch-2.5.1.dist-info/METADATA').write_text(
        'Metadata-Version: 2.1\nName: torch\nVersion: 2.2.1\n', encoding='utf-8')
    detail = f'FileNotFoundError: Cannot find DGL C++ graphbolt library at {directory / "dgl/graphbolt/graphbolt_pytorch_2.5.1.dll"}'
    assert ce._graphbolt_abi_evidence([directory], detail) is None


def test_graphbolt_proven_abi_mismatch_in_ready_cache_does_not_invoke_pip(monkeypatch, tmp_path):
    requirements = 'dgl==2.2.1\ntorch==2.5.1'
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest(requirements)
    graphbolt_wheel_fixture(directory, 'other_versions_only')
    (directory / '.ready').write_text('ok\n', encoding='utf-8')
    calls = graphbolt_install_responses(monkeypatch, 'other_versions_only')
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, dependency_health_check=True)
    diagnostic = prepare(instance, tmp_path)
    assert diagnostic and 'graphbolt_pytorch_2.5.1.dll' in diagnostic
    assert calls == [] and instance.env_config['dependency_install_attempts'] == []
    assert instance.env_config['dependency_failure_kind'] == 'native_abi_incompatible'
    assert instance.env_config['dependency_health_attempts'][0]['phase'] == 'cached'
    assert not instance.env_config.get('dependency_cache_repairs')
    assert instance._deps_dir is None and not (directory / '.ready').exists()
    assert (directory / 'dgl-2.2.1.dist-info/RECORD').is_file()


def test_graphbolt_many_available_libraries_keep_bounded_diagnostics(monkeypatch, tmp_path):
    requirements = 'dgl==2.2.1\ntorch==2.5.1'
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest(requirements)
    graphbolt_wheel_fixture(directory, 'other_versions_only')
    record_path = directory / 'dgl-2.2.1.dist-info/RECORD'
    with record_path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        for minor in range(40):
            relative = f'dgl/graphbolt/graphbolt_pytorch_1.{minor}.0.dll'
            (directory / relative).write_bytes(b'another declared ABI')
            writer.writerow([relative, '', ''])
    graphbolt_install_responses(monkeypatch, 'other_versions_only')
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, dependency_health_check=True)
    assert instance._check_local_dependency_health(directory, requirements, 'cached')
    attempt = instance.env_config['dependency_health_attempts'][0]
    assert len(attempt['native_abi_evidence']['available_libraries']) == 41
    assert len(attempt['diagnostic']) <= 1600
    assert 'graphbolt_pytorch_2.5.1.dll' in attempt['diagnostic']
    assert 'graphbolt_pytorch_1.0.0.dll' in attempt['diagnostic']


def test_private_extra_index_retains_source_policy(monkeypatch, tmp_path):
    instance = executor()
    instance.env_config["requirements_txt"] = "--extra-index-url https://private.invalid/simple\n" + REQUIREMENTS
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES)])
    assert prepare(instance, tmp_path) is not None
    assert len(calls) == 1


@pytest.mark.parametrize("first_error", [
    (1, "", "WARNING: Retrying after NewConnectionError: connection refused"),
    subprocess.TimeoutExpired(["pip"], 300, output=b"partial download", stderr=b"Read timed out"),
])
def test_public_mirror_network_failure_tries_next_domestic_source(monkeypatch, tmp_path, first_error):
    calls = install_responses(monkeypatch, [first_error, (0, "installed", "")])
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [c["index"] for c in calls] == [MIRROR, ALIYUN]
    assert len(instance.env_config["dependency_install_attempts"]) == 2
    first_record = instance.env_config["dependency_install_attempts"][0]
    if isinstance(first_error, subprocess.TimeoutExpired):
        assert first_record["failure_kind"] == "installation_timeout"
        assert "日志包含来源失败" in repr(instance.logger.mock_calls)
    else:
        assert first_record["failure_kind"] == "source_unavailable"


def test_generic_install_total_timeout_keeps_its_distinct_failure_without_switching_sources(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [subprocess.TimeoutExpired(["pip"], 300,
                               output="Successfully downloaded; copying installed files")])
    instance = executor()
    assert prepare(instance, tmp_path) is not None
    assert len(calls) == 1 and calls[0]["kwargs"]["timeout"] == 300
    assert instance.env_config["dependency_failure_kind"] == "installation_timeout"
    assert "当前公共镜像未能提供依赖" not in repr(instance.logger.mock_calls)


def test_all_public_sources_fail_with_full_diagnostics_and_no_ready_marker(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES),
                                           (2, "", "ALIYUN_NETWORK_FAILURE: ConnectionError"),
                                           (1, "", NO_CANDIDATES),
                                           (2, "", "OFFICIAL_NETWORK_FAILURE: ConnectionError")])
    instance = executor()
    diagnostic = prepare(instance, tmp_path)
    assert [call['index'] for call in calls] == [MIRROR, ALIYUN, USTC, OFFICIAL]
    assert "已尝试 4 轮" in diagnostic
    assert "第 1 轮" in diagnostic and "第 4 轮" in diagnostic
    assert "from versions: none" in diagnostic
    assert "OFFICIAL_NETWORK_FAILURE" in diagnostic
    assert MIRROR in diagnostic and OFFICIAL in diagnostic
    assert ce._INSTALLED_DEPS[REQUIREMENTS] == diagnostic
    assert instance._deps_dir is None
    assert not (ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS) / ".ready").exists()


def test_official_source_is_used_only_after_domestic_sources_fail(monkeypatch, tmp_path):
    responses = [(1, '', NO_CANDIDATES)] * 3 + [(0, 'installed', '')]
    calls = install_responses(monkeypatch, responses)
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [item['index'] for item in calls] == [MIRROR, ALIYUN, USTC, OFFICIAL]
    assert all(item['requirements'] == REQUIREMENTS for item in calls)


def test_python_conflict_and_wheel_absence_have_distinct_diagnostics():
    assert ce._install_failure_kind('ERROR: package requires a different Python') == 'python_incompatible'
    assert ce._install_failure_kind('ERROR: No matching distribution found for torch==2.5.1+cpu') == 'wheel_unavailable'
    assert ce._install_failure_kind('ERROR: ResolutionImpossible') == 'dependency_conflict'
    assert ce._install_failure_kind('ERROR: Could not build wheels for package') == 'build_failed'


def test_process_deadline_is_distinct_from_network_request_timeout():
    assert ce._install_failure_kind("extracting wheel", timed_out=True) == "installation_timeout"
    assert not ce._public_mirror_failure("extracting wheel", timed_out=True)
    diagnostic = "ReadTimeoutError: HTTPSConnectionPool(host='pypi.example')"
    assert ce._install_failure_kind(diagnostic) == "source_unavailable"
    assert ce._public_mirror_failure(diagnostic)


def test_ignored_unrequested_python_versions_do_not_trigger_interpreter_switch():
    diagnostic = ("ERROR: Ignored the following versions that require a different python version: "
                  "0.8.0 Requires-Python <3.11; 0.9.0 Requires-Python <3.12\n"
                  "ERROR: No matching distribution found for example==1.2.0")
    assert ce._requested_python_conflict(diagnostic, 'example==1.2.0') is None
    assert ce._install_failure_kind(diagnostic, requirements='example==1.2.0') == 'wheel_unavailable'
    assert ce._public_mirror_failure(diagnostic, requirements='example==1.2.0') is True
    requested = diagnostic.replace('0.8.0 Requires-Python', '1.2.0 Requires-Python')
    assert ce._requested_python_conflict(requested, 'example==1.2.0')['version'] == '1.2.0'
    assert ce._install_failure_kind(requested, requirements='example==1.2.0') == 'python_incompatible'
    assert ce._public_mirror_failure(requested, requirements='example==1.2.0') is False
    explicit = "ERROR: Package 'example' requires a different Python: 3.12.0 not in '<3.10'"
    assert ce._requested_python_conflict(explicit, 'example==1.2.0')['requires_python'] == '<3.10'
    assert ce._requested_python_conflict(explicit, 'another==1.2.0') is None


def test_failed_install_can_succeed_on_next_user_run_in_same_process(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(2, "", "first-run failure"), (0, "installed", "")])
    first = executor()
    assert "first-run failure" in prepare(first, tmp_path, "first")
    assert ce._INSTALLED_DEPS[REQUIREMENTS]
    second = executor()
    assert prepare(second, tmp_path, "second") is None
    assert len(calls) == 2
    assert ce._INSTALLED_DEPS[REQUIREMENTS] == ""
    assert second._deps_dir == str(ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS))


def test_offline_missing_cache_is_reported_without_pip_attempt(tmp_path):
    instance = executor()
    instance.env_config['offline'] = True
    diagnostic = prepare(instance, tmp_path)
    assert '离线模式缺少' in diagnostic
    assert instance.env_config['dependency_failure_kind'] == 'offline_cache_missing'
    assert instance.env_config['dependency_install_attempts'] == []


def test_first_success_uses_one_request_and_success_cache_restores_dependency_path(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(0, "installed", "")])
    first = executor()
    assert prepare(first, tmp_path, "first") is None
    second = executor()
    assert prepare(second, tmp_path, "second") is None
    assert len(calls) == 1
    assert first._deps_dir == second._deps_dir
    assert not (tmp_path / "second" / "requirements.txt").exists()


def test_memory_success_without_matching_ready_directory_reinstalls(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(0, "installed", "")])
    ce._INSTALLED_DEPS[REQUIREMENTS] = ""
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 1
    assert (Path(instance._deps_dir) / ".ready").is_file()


def test_separate_runtime_cache_roots_do_not_share_ready_or_failure_state(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(0, "installed first", ""), (0, "installed second", "")])
    first = executor()
    first.deps_cache_root = tmp_path / "runtime_first"
    assert prepare(first, tmp_path, "first") is None
    second = executor()
    second.deps_cache_root = tmp_path / "runtime_second"
    assert prepare(second, tmp_path, "second") is None
    assert len(calls) == 2
    assert first._deps_dir != second._deps_dir
    assert Path(second._deps_dir).is_relative_to(second.deps_cache_root)


def test_incomplete_previous_install_replaces_files_under_the_same_pins(monkeypatch, tmp_path):
    incomplete = ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS)
    incomplete.mkdir(parents=True)
    (incomplete / "partial_package").mkdir()
    calls = install_responses(monkeypatch, [(0, "installed", "")])
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert "--upgrade" in calls[0]["cmd"]
    assert calls[0]["requirements"] == REQUIREMENTS
    assert (incomplete / ".ready").is_file()


@pytest.mark.parametrize("private_source", ["index", "links", "requirements"])
def test_custom_package_sources_do_not_fall_back_to_public_pypi(monkeypatch, tmp_path, private_source):
    instance = executor()
    if private_source == "index":
        monkeypatch.setattr(ce, "PIP_INDEX_URL", "https://packages.company.invalid/simple")
    elif private_source == "links":
        monkeypatch.setattr(ce, "PIP_FIND_LINKS", "https://packages.company.invalid/wheels")
    else:
        instance.env_config["requirements_txt"] = "--index-url https://packages.company.invalid/simple\n" + REQUIREMENTS
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES)])
    assert prepare(instance, tmp_path) is not None
    assert len(calls) == 1
    assert len(instance.env_config["dependency_install_attempts"]) == 1


def test_official_pypi_configuration_does_not_retry_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "PIP_INDEX_URL", OFFICIAL)
    calls = install_responses(monkeypatch, [(1, "", "ConnectionError")])
    assert prepare(executor(), tmp_path) is not None
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    "ERROR: ResolutionImpossible: conflicting dependencies",
    "ERROR: Could not build wheels for broken-package",
    "ERROR: package requires a different Python: 3.12 not in '<3.9'",
])
def test_dependency_constraints_and_build_failures_do_not_change_sources(monkeypatch, tmp_path, error):
    calls = install_responses(monkeypatch, [(1, "", error)])
    assert prepare(executor(), tmp_path) is not None
    assert len(calls) == 1


def test_private_url_credentials_and_tokens_are_removed_from_new_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "PIP_INDEX_URL", "https://user:secretpass@packages.company.invalid/token-in-path/simple?token=querysecret")
    error = ("Could not fetch URL https://user:secretpass@packages.company.invalid/token-in-path/wheel?token=querysecret\n"
             "Authorization: Bearer headersecret\nAPI_KEY=keysecret\nTOKEN=labelsecret\n" + NO_CANDIDATES)
    install_responses(monkeypatch, [(1, "", error)])
    instance = executor()
    diagnostic = prepare(instance, tmp_path)
    persisted = diagnostic + repr(instance.env_config["dependency_install_attempts"]) + repr(instance.logger.mock_calls)
    for secret in ["secretpass", "token-in-path", "querysecret", "headersecret", "keysecret", "labelsecret"]:
        assert secret not in persisted
    assert "packages.company.invalid" in persisted
    assert "from versions: none" in diagnostic
