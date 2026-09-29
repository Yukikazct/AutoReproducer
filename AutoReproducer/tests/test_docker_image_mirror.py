"""Docker 镜像可用性加固测试（镜像源回退 + 拉取失败分类）。

背景缺陷（真实运行实测）：本机直连 docker.io 不通（curl 000），
`docker run python:3.11-slim` 拉取失败报
`failed to resolve reference "docker.io/library/python:3.11-slim": context
deadline exceeded`，exit_code=125。而 125 不在 `_NON_CODE_REPAIR_EXIT_CODES`
里，诊断层将它判成"可修复"——3 轮 LLM 修复全烧在改论文代码上（实测报告里
4 次 exit 125，每次都在改代码）；计划路径则整链跳过、报告把基础设施故障
写成 execution_error。

覆盖：
1. 镜像源候选解析：canonical 名才拼镜像源，显式 registry 名不拼；
2. 预拉：本地已有不拉；缺失则按候选顺序拉取，成功 tag 回**原名**；
3. 拉取失败分类：docker_pull_failed + repairable=False（不烧 LLM 修复）；
4. 回归：unknown flag（加固降级链同样用 125）不得被误判成拉取失败；
5. 计划步预拉失败：专属退出码 + plan_fail_reason 非空 + 报告显式提示；
6. en2vbuilder：docker build 前对 FROM 基础镜像预拉，失败短路不构建。

运行: python -m pytest tests/test_docker_image_mirror.py -v
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent,
    EXIT_DOCKER_IMAGE_UNAVAILABLE,
)
from src.agents.env_builder import EnvBuilderAgent  # noqa: E402
from src.agents.report_generator import ReportGeneratorAgent  # noqa: E402
from src.base_agent import BaseAgent  # noqa: E402
from src.execution_plan import PlanStep  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402

_IMAGE = "python:3.11-slim"
_MIRROR = "docker.m.daocloud.io"
# 用户实测的那条原始报错（docker.io 不可达）
_DOCKER_IO_ERR = (
    'docker: Error response from daemon: failed to resolve reference '
    '"docker.io/library/python:3.11-slim": failed to do request: Head '
    '"https://registry-1.docker.io/v2/library/python/manifests/3.11-slim": '
    'dial tcp 31.13.94.35:443: i/o timeout')


def _executor() -> CodeExecutorAgent:
    executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
    executor.use_docker = True
    executor.env_config = {"image_tag": _IMAGE}
    return executor


def _plan(units, steps) -> dict:
    return {"paper_id": "p", "source": "heuristic", "units": units,
            "steps": steps, "entry": {}, "notes": []}


def _unit(tmp_path, unit_id="main") -> dict:
    repo = tmp_path / unit_id
    repo.mkdir(exist_ok=True)
    (repo / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return {"unit_id": unit_id, "role": "main", "local_path": str(repo)}


# ---------------- 1. 镜像源候选解析 ----------------

class TestResolveImageWithMirror:
    def test_no_mirror_keeps_original(self):
        assert BaseAgent.resolve_image_with_mirror(_IMAGE, []) == [_IMAGE]

    def test_canonical_gets_mirror_prefix(self):
        assert BaseAgent.resolve_image_with_mirror(
            _IMAGE, [_MIRROR]) == [_IMAGE, f"{_MIRROR}/{_IMAGE}"]

    def test_multiple_mirrors_keep_order(self):
        assert BaseAgent.resolve_image_with_mirror(
            _IMAGE, [_MIRROR, "dockerproxy.net"]) == [
                _IMAGE, f"{_MIRROR}/{_IMAGE}", f"dockerproxy.net/{_IMAGE}"]

    def test_registry_prefixed_image_not_rewritten(self):
        """带显式 registry 的镜像名不能再往上拼镜像源（拼出来是无效名）。"""
        image = "docker.m.daocloud.io/library/python:3.11-slim"
        assert BaseAgent.resolve_image_with_mirror(image, [_MIRROR]) == [image]
        assert BaseAgent.resolve_image_with_mirror(
            "gcr.io/proj/img:1.0", [_MIRROR]) == ["gcr.io/proj/img:1.0"]
        assert BaseAgent.resolve_image_with_mirror(
            "localhost:5000/foo", [_MIRROR]) == ["localhost:5000/foo"]

    def test_docker_hub_namespace_is_canonical(self):
        """`library/python` 这类 Docker Hub 命名空间仍是 canonical（无 registry）。"""
        assert BaseAgent.is_canonical_image("library/python:3.11-slim") is True
        assert BaseAgent.is_canonical_image("autorepro-base:latest") is True
        assert BaseAgent.is_canonical_image("gcr.io/x/y") is False
        assert BaseAgent.is_canonical_image("") is False


# ---------------- 2. 预拉：本地命中 / 拉取 + tag ----------------

def _fake_docker(monkeypatch, *, existing: bool = False,
                 pull_results: dict = None):
    """打桩 subprocess.run，模拟 docker images / pull / tag。

    pull_results: {候选名: (returncode, stderr)}；未列出的一律成功。
    """
    calls: list = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        op = cmd[1] if len(cmd) > 1 else ""
        if op == "images":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="sha256:deadbeef" if existing else "",
                stderr="")
        if op == "pull":
            rc, err = (pull_results or {}).get(cmd[2], (0, ""))
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


class TestEnsureImagePulled:
    def test_local_hit_skips_pull(self, monkeypatch):
        calls = _fake_docker(monkeypatch, existing=True)
        assert BaseAgent.ensure_image_pulled("docker", _IMAGE) is None
        assert calls == [["docker", "images", "-q", _IMAGE]]

    def test_pull_then_tag_back_to_canonical(self, monkeypatch):
        """镜像源拉下来后必须 tag 回原名：白名单与执行路径全按原名判定。"""
        calls = _fake_docker(monkeypatch, pull_results={
            _IMAGE: (125, _DOCKER_IO_ERR)})
        assert BaseAgent.ensure_image_pulled(
            "docker", _IMAGE, mirrors=[_MIRROR]) is None
        assert calls == [
            ["docker", "images", "-q", _IMAGE],
            ["docker", "pull", _IMAGE],                     # 原名先试（docker.io）
            ["docker", "pull", f"{_MIRROR}/{_IMAGE}"],      # 失败后换镜像源
            ["docker", "tag", f"{_MIRROR}/{_IMAGE}", _IMAGE],
        ]

    def test_first_mirror_fails_falls_through(self, monkeypatch):
        calls = _fake_docker(monkeypatch, pull_results={
            _IMAGE: (125, _DOCKER_IO_ERR),
            f"{_MIRROR}/{_IMAGE}": (1, "no such host"),
        })
        assert BaseAgent.ensure_image_pulled(
            "docker", _IMAGE, mirrors=[_MIRROR, "dockerproxy.net"]) is None
        assert ["docker", "pull", "dockerproxy.net/" + _IMAGE] in calls
        assert ["docker", "tag",
                "dockerproxy.net/" + _IMAGE, _IMAGE] in calls

    def test_all_candidates_fail_returns_human_message(self, monkeypatch):
        _fake_docker(monkeypatch, pull_results={
            _IMAGE: (125, _DOCKER_IO_ERR),
            f"{_MIRROR}/{_IMAGE}": (1, "no such host")})
        err = BaseAgent.ensure_image_pulled(
            "docker", _IMAGE, mirrors=[_MIRROR])
        assert err and _IMAGE in err
        assert _MIRROR in err                       # 尝试过哪些候选要说清楚
        assert "AUTOREPRO_DOCKER_IMAGE_MIRROR" in err   # 给出可操作的下一步

    def test_no_mirror_configured_hints_env_var(self, monkeypatch):
        _fake_docker(monkeypatch, pull_results={_IMAGE: (125, _DOCKER_IO_ERR)})
        err = BaseAgent.ensure_image_pulled("docker", _IMAGE, mirrors=[])
        assert err and "AUTOREPRO_DOCKER_IMAGE_MIRROR" in err

    def test_pull_timeout_falls_through(self, monkeypatch):
        """单个镜像源拉取超时不得卡死整条流水线，继续试下一个。"""
        def fake_run(cmd, **kw):
            if cmd[1] == "images":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[1] == "pull" and cmd[2] == _IMAGE:
                raise subprocess.TimeoutExpired(cmd, 600)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert BaseAgent.ensure_image_pulled(
            "docker", _IMAGE, mirrors=[_MIRROR]) is None


# ---------------- 3. 拉取失败分类：不烧 LLM ----------------

class _ExplodingLLM(LLMClient):
    """任何 chat 调用都视为缺陷（镜像拉取失败不该触发代码修复）。"""

    def __init__(self):
        super().__init__(mock_mode=True)

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        raise AssertionError("镜像拉取失败不应触发 LLM 修复")


class TestPullFailureDiagnosis:
    @staticmethod
    def _pull_result(exit_code: int = 125, stderr: str = _DOCKER_IO_ERR) -> dict:
        return {"success": False, "stdout": "", "stderr": stderr,
                "exit_code": exit_code}

    def test_diagnosed_as_unrepairable_environment_issue(self):
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        diag = executor._diagnose_execution_error(self._pull_result())
        assert diag["error_type"] == "docker_pull_failed"
        assert diag["repairable"] is False

    def test_no_llm_call_on_pull_failure(self):
        executor = CodeExecutorAgent(_ExplodingLLM(), mock_mode=True)
        code, diag, candidate = executor._repair_after_execution(
            {}, "print(1)\n", "smoke", self._pull_result())
        assert candidate is None
        assert diag["error_type"] == "docker_pull_failed"

    def test_dedicated_exit_code_is_not_repairable(self):
        """预拉失败是我们自己返回的人话文本，原始特征串对不上，靠退出码兜底。"""
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        diag = executor._diagnose_execution_error(
            {"stdout": "", "stderr": "镜像本地不存在且拉取失败",
             "exit_code": EXIT_DOCKER_IMAGE_UNAVAILABLE})
        assert diag["error_type"] == "docker_pull_failed"
        assert diag["repairable"] is False

    def test_unknown_flag_still_degrades(self):
        """回归：加固降级链同样用 125，不得被误判成镜像拉取失败。

        old Docker 不支持 --pids-limit 时报 "unknown flag"，那条路径的正确
        行为是**降级重跑**——整体按 125 判定会把可跑的用例判死。
        """
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        diag = executor._diagnose_execution_error(
            {"stdout": "", "exit_code": 125,
             "stderr": "Error: unknown flag: --pids-limit"})
        assert diag["error_type"] != "docker_pull_failed"
        assert diag["repairable"] is True

    def test_plan_step_repair_classified_non_repairable(self):
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        step = PlanStep.from_dict(
            {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
             "cwd": "/app/main", "unit_id": "main", "timeout_s": 30})
        new_step, diag = executor._repair_plan_step(
            step, self._pull_result())
        assert new_step is None
        assert diag["error_type"] == "docker_pull_failed"
        assert diag["strategy"] == "non_repairable"
        assert "AUTOREPRO_DOCKER_IMAGE_MIRROR" in diag["detail"]

    def test_plan_step_by_exit_code(self):
        """预拉失败是我们自己返回的人话文本，特征串对不上，靠退出码兜底。"""
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        step = PlanStep.from_dict(
            {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
             "cwd": "/app/main", "unit_id": "main", "timeout_s": 30})
        _new, diag = executor._repair_plan_step(step, {
            "success": False, "stdout": "",
            "stderr": f"镜像 {_IMAGE} 本地不存在且拉取失败",
            "exit_code": EXIT_DOCKER_IMAGE_UNAVAILABLE})
        assert diag["error_type"] == "docker_pull_failed"


# ---------------- 4. 计划执行：预拉失败 → 回退原因非空 ----------------

class TestPlanStepImageUnavailable:
    def _run(self, tmp_path, monkeypatch, message: str) -> dict:
        executor = _executor()
        monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
        monkeypatch.setattr(BaseAgent, "docker_engine_available",
                            staticmethod(lambda *a, **k: (True, None)))
        monkeypatch.setattr(BaseAgent, "ensure_image_pulled",
                            staticmethod(lambda *a, **k: message))
        steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
                  "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
        return executor._execute_plan(_plan([_unit(tmp_path)], steps), {})

    def test_step_record_and_fail_reason(self, tmp_path, monkeypatch):
        message = f"镜像 {_IMAGE} 本地不存在且拉取失败（已尝试：{_IMAGE}）"
        result = self._run(tmp_path, monkeypatch, message)
        assert result["success"] is False
        assert result["plan_failed_irreparably"] is True
        record = result["stages"][0]
        assert record["exit_code"] == EXIT_DOCKER_IMAGE_UNAVAILABLE
        assert record["image_unavailable"] is True
        # 回退原因此前在正常失败路径上根本没有这个键（报告里永远是空的）
        assert "镜像" in result["plan_fail_reason"]
        assert "AUTOREPRO_DOCKER_IMAGE_MIRROR" in result["plan_fail_reason"]

    def test_report_renders_image_pull_failure(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, f"镜像 {_IMAGE} 拉取失败")
        report = ReportGeneratorAgent().run({
            "paper_info": {}, "resources": {}, "env_config": {},
            "validation": {}, "optimization": {}, "audit_stats": {},
            "execution": {**result, "execution_mode": "plan"},
        })["report"]
        assert "🐳 镜像拉取失败" in report
        assert "运行环境问题，不是论文代码问题" in report
        assert "AUTOREPRO_DOCKER_IMAGE_MIRROR" in report

    def test_fail_reason_present_for_other_failures_too(self, tmp_path,
                                                        monkeypatch):
        """回退原因补全不止服务镜像故障：跑挂了也必须有人话原因。"""
        executor = CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=True)
        executor.use_docker = False
        monkeypatch.setattr(executor, "_execute_plan_step", lambda *a: {
            "step_id": "run_0", "kind": "run", "cmd": "python run.py",
            "unit_id": "main", "success": False, "stdout": "",
            "stderr": "RuntimeError: boom", "exit_code": 1, "repairs": []})
        steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
                  "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
        result = executor._execute_plan(_plan([_unit(tmp_path)], steps), {})
        assert result["plan_failed_irreparably"] is True
        assert result["plan_fail_reason"]
        assert "run_0" in result["plan_fail_reason"]


# ---------------- 5. EnvBuilder：FROM 基础镜像预拉 ----------------

class TestEnvBuilderBaseImage:
    def test_parse_from_image(self):
        assert EnvBuilderAgent._parse_from_image(
            "FROM python:3.11-slim\nRUN pip install x\n") == "python:3.11-slim"
        assert EnvBuilderAgent._parse_from_image(
            "FROM python:3.11-slim AS build\n") == "python:3.11-slim"
        assert EnvBuilderAgent._parse_from_image(
            "# FROM ignored\nFROM pytorch/pytorch:2.1\n") == \
            "pytorch/pytorch:2.1"
        assert EnvBuilderAgent._parse_from_image("RUN echo hi\n") == ""
        assert EnvBuilderAgent._parse_from_image("FROM ${BASE}\n") == ""

    def test_build_short_circuits_when_base_unavailable(self, monkeypatch,
                                                        tmp_path):
        """基础镜像拉不到时不再白等 1800s 构建超时，直接给人话原因。"""
        calls: list = []

        def fake_run(cmd, **kw):
            calls.append([str(c) for c in cmd])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        builder = EnvBuilderAgent(LLMClient(mock_mode=True))
        monkeypatch.setattr(builder, "_resolve_docker_cmd", lambda: "docker")
        monkeypatch.setattr(BaseAgent, "docker_engine_available",
                            staticmethod(lambda *a, **k: (True, None)))
        monkeypatch.setattr(
            BaseAgent, "ensure_image_pulled",
            staticmethod(lambda *a, **k: f"镜像 {_IMAGE} 本地不存在且拉取失败"))
        result = builder._build_dockerfile(
            "FROM python:3.11-slim\nRUN echo hi\n", tag="t")
        assert result["success"] is False
        assert "拉取失败" in result["error"]
        assert calls == []          # 预拉失败即短路，不进入 docker build

    def test_self_built_base_image_skips_pull(self, monkeypatch):
        """autorepro-base 只存在于本地，对它 pull 必然失败——必须跳过预拉。"""
        pulled: list = []

        def fake_run(cmd, **kw):
            joined = " ".join(str(c) for c in cmd)
            if "build" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="ok",
                                                   stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        builder = EnvBuilderAgent(LLMClient(mock_mode=True))
        monkeypatch.setattr(builder, "_resolve_docker_cmd", lambda: "docker")
        monkeypatch.setattr(BaseAgent, "docker_engine_available",
                            staticmethod(lambda *a, **k: (True, None)))

        def spy(docker_cmd, image, **kw):
            pulled.append(image)
            return None

        monkeypatch.setattr(BaseAgent, "ensure_image_pulled",
                            staticmethod(spy))
        result = builder._build_dockerfile(
            "FROM autorepro-base:latest\nRUN echo hi\n", tag="t")
        assert result["success"] is True
        assert pulled == []
