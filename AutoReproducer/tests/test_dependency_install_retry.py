"""Source fallback and user-run retry contracts; every pip invocation is mocked."""
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
from src.agents.code_executor import CodeExecutorAgent


REQUIREMENTS = "numpy==1.26.4\npandas==2.2.3\ntorch==2.5.1"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
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


def test_missing_default_mirror_candidates_fall_back_once_with_identical_pins(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", "")])
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [c["index"] for c in calls] == [MIRROR, OFFICIAL]
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
    assert [call["index"] for call in calls] == [MIRROR, OFFICIAL]
    assert [call["requirements"] for call in calls] == [requirements, requirements]


def test_auto_prepared_cuda_timeout_gets_long_official_retry_with_same_pins(monkeypatch, tmp_path):
    first_timeout = subprocess.TimeoutExpired(["pip"], 300, output=b"extracting CUDA wheel",
                                             stderr=b"partial target copy")
    calls = install_responses(monkeypatch, [first_timeout, (0, "installed", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=CUDA_REQUIREMENTS, auto_prepare=True)
    assert prepare(instance, tmp_path) is None
    assert [call["index"] for call in calls] == [MIRROR, OFFICIAL]
    assert [call["kwargs"]["timeout"] for call in calls] == [300, 1800]
    assert [call["requirements"] for call in calls] == [CUDA_REQUIREMENTS] * 2
    records = instance.env_config["dependency_install_attempts"]
    assert [record["timeout_s"] for record in records] == [300, 1800]
    assert records[0]["timed_out"] and not records[1]["timed_out"]
    assert "extracting CUDA wheel" in records[0]["diagnostic"]
    assert "partial target copy" in records[0]["diagnostic"]


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
    ("--extra-index-url https://download.pytorch.org/whl/cpu\n"
     "torch==2.5.1+cpu\nnumpy==1.26.4", True),
    ("--extra-index-url https://download.pytorch.org/whl/cu121\n" + REQUIREMENTS, True),
    ("torch==2.5.1+cu121\nnumpy==1.26.4", True),
    ("--extra-index-url https://download.pytorch.org/whl/cu121\n"
     "torch==2.5.1+cu118", True),
    (CUDA_REQUIREMENTS, False),
])
def test_only_matching_auto_prepared_cuda_wheels_get_long_retry(monkeypatch, tmp_path, requirements, auto_prepare):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=requirements, auto_prepare=auto_prepare)
    assert prepare(instance, tmp_path) is None
    assert [call["kwargs"]["timeout"] for call in calls] == [300, 300]
    assert [record["timeout_s"] for record in instance.env_config["dependency_install_attempts"]] == [300, 300]


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


def test_cuda_native_health_rebuild_retains_long_official_retry(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES), (0, "installed", ""),
                                           (0, "rebuilt", "")])
    instance = executor()
    instance.env_config.update(requirements_txt=CUDA_REQUIREMENTS, auto_prepare=True,
                               dependency_health_check=True)
    monkeypatch.setattr(instance, "_check_local_dependency_health",
                        Mock(side_effect=["broken native DLL", None]))
    assert prepare(instance, tmp_path) is None
    assert [call["index"] for call in calls] == [MIRROR, OFFICIAL, OFFICIAL]
    assert [call["kwargs"]["timeout"] for call in calls] == [300, 1800, 1800]
    assert [call["requirements"] for call in calls] == [CUDA_REQUIREMENTS] * 3


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
def test_public_mirror_network_failure_has_one_official_fallback(monkeypatch, tmp_path, first_error):
    calls = install_responses(monkeypatch, [first_error, (0, "installed", "")])
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [c["index"] for c in calls] == [MIRROR, OFFICIAL]
    assert len(instance.env_config["dependency_install_attempts"]) == 2


def test_double_failure_reports_both_attempts_and_writes_no_ready_marker(monkeypatch, tmp_path):
    calls = install_responses(monkeypatch, [(1, "", NO_CANDIDATES),
                                           (2, "", "OFFICIAL_NETWORK_FAILURE: ConnectionError")])
    instance = executor()
    diagnostic = prepare(instance, tmp_path)
    assert len(calls) == 2
    assert "已尝试 2 轮" in diagnostic
    assert "第 1 轮" in diagnostic and "第 2 轮" in diagnostic
    assert "from versions: none" in diagnostic
    assert "OFFICIAL_NETWORK_FAILURE" in diagnostic
    assert MIRROR in diagnostic and OFFICIAL in diagnostic
    assert ce._INSTALLED_DEPS[REQUIREMENTS] == diagnostic
    assert instance._deps_dir is None
    assert not (ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS) / ".ready").exists()


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
