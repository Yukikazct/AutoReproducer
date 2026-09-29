"""CodeExecutor 计划执行模式测试（官方代码多单元整体调用）。

覆盖（mock 本地 + fake docker 两条路径）：
1. 计划端到端：run 步输出 mse/mae -> actual_metrics 提取 + parse 步指标；
2. depends_on：前置失败 -> 后续步骤 skipped_deps；
3. 超时 -> smoke_args 缩参重试（repair_attempts 记录）；
4. 缺模块 -> install_pkgs 追加包重试；
5. GPU-only 特征 -> 不可修复 -> plan_failed_irreparably -> run() 回退
   生成脚本路径（execution_mode="generated_fallback"）；
6. 真实模式无 Docker -> 拒绝执行官方代码（EXIT_ISOLATION_REQUIRED）；
7. Docker 路径：安装前缀与脚本本体两次独立 run、预算分离、逐步 -w。

运行: python -m pytest tests/test_code_executor_plan.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent,
    EXIT_ISOLATION_REQUIRED,
)
from src.base_agent import BaseAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


class _ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]

    def get_call_count(self) -> int:
        return self.call_count


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    ce_mod._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce_mod, "DEPS_CACHE_ROOT", tmp_path / "deps")
    monkeypatch.setattr(BaseAgent, "docker_engine_available",
                        staticmethod(lambda *a, **k: (True, None)))
    # 每次 docker run 前都会先探一次镜像可用性（docker images -q，缺失则
    # 按镜像源拉取）——那是一次真实 subprocess，会污染本文件的调用序列
    # 断言，也会让用例依赖本机 Docker 状态。镜像可用性由
    # tests/test_docker_image_mirror.py 专门覆盖，这里打桩为"镜像已就绪"。
    monkeypatch.setattr(BaseAgent, "ensure_image_pulled",
                        staticmethod(lambda *a, **k: None))
    yield
    ce_mod._INSTALLED_DEPS.clear()


def _executor(mock_mode=True) -> CodeExecutorAgent:
    return CodeExecutorAgent(LLMClient(mock_mode=True), mock_mode=mock_mode)


def _plan(units, steps) -> dict:
    return {"paper_id": "p", "source": "heuristic",
            "units": units, "steps": steps, "entry": {},
            "notes": []}


def _unit(tmp_path, unit_id="main", content="print('mse: 0.01, mae: 0.02')\n"):
    repo = tmp_path / unit_id
    repo.mkdir(exist_ok=True)
    (repo / "run.py").write_text(content, encoding="utf-8")
    return {"unit_id": unit_id, "role": "main",
            "local_path": str(repo), "url": f"file://{repo}"}


# ---------------- 1. 端到端 ----------------

def test_plan_end_to_end_metrics(tmp_path):
    unit = _unit(tmp_path)
    steps = [
        {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30},
        {"step_id": "parse", "kind": "parse", "cmd": "",
         "cwd": "/app/main", "unit_id": "main", "depends_on": ["run_0"]},
    ]
    executor = _executor()
    result = executor._execute_plan(_plan([unit], steps), {})
    assert result["success"] is True
    assert result["plan_failed_irreparably"] is False
    assert result["actual_metrics"]["mse"] == 0.01
    assert result["actual_metrics"]["mae"] == 0.02
    assert [s["step_id"] for s in result["stages"]] == ["run_0", "parse"]
    assert result["stages"][1]["metrics"]["mse"] == 0.01
    assert "mse: 0.01" in result["final"]["stdout"]


def test_run_method_prefers_plan_over_generation(tmp_path):
    """run() 带执行计划且计划成功 -> 直接返回计划结果，不调用 LLM 生成。"""
    unit = _unit(tmp_path)
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    llm = _ScriptedLLM(["print('must not be used')"])
    executor = CodeExecutorAgent(llm, mock_mode=True)
    result = executor.run({
        "paper_info": {}, "env_config": {},
        "execution_plan": _plan([unit], steps),
    })
    assert result["execution_mode"] == "plan"
    assert result["success"] is True
    assert llm.call_count == 0


# ---------------- 2. depends_on ----------------

def test_depends_on_skips_downstream(tmp_path):
    unit = _unit(tmp_path, content="import sys\nsys.exit(1)\n")
    steps = [
        {"step_id": "bad", "kind": "run", "cmd": "python run.py",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30},
        {"step_id": "after", "kind": "run", "cmd": "python run.py",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30,
         "depends_on": ["bad"]},
    ]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["success"] is False
    records = {s["step_id"]: s for s in result["stages"]}
    assert records["after"]["skipped_deps"] is True
    assert records["bad"]["success"] is False


# ---------------- 3. 超时缩参修复 ----------------

def test_timeout_repairs_with_smoke_args(tmp_path):
    unit = _unit(tmp_path, content=(
        "import sys, time\n"
        "time.sleep(3 if '--fast' not in sys.argv else 0.05)\n"
        "print('mse: 0.01')\n"))
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 1,
              "smoke_args": {"--fast": "1"}}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["success"] is True
    record = result["stages"][0]
    assert record["repair_attempts"] == 1
    assert record["repairs"][0]["strategy"] == "smoke_args"
    assert record["final_cmd"] == "python run.py --fast 1"


def test_timeout_without_smoke_args_non_repairable(tmp_path):
    unit = _unit(tmp_path, content="import time\ntime.sleep(30)\n")
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 1}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["success"] is False
    assert result["plan_failed_irreparably"] is True
    record = result["stages"][0]
    assert record["timed_out"] is True


# ---------------- 4. 缺模块自愈 ----------------

def test_missing_module_appends_install_pkg(tmp_path):
    unit = _unit(tmp_path, content="import no_such_module_xyz\n")
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    record = result["stages"][0]
    # 第一轮修复：追加包；本地 mock 不真实安装 -> 第二轮同模块停止
    assert record["repair_attempts"] == 2
    assert record["repairs"][0]["strategy"] == "install_pkg"


# ---------------- 5. GPU-only 与回退 ----------------

def test_gpu_only_is_irreparable(tmp_path):
    unit = _unit(tmp_path, content=(
        "import sys\n"
        "sys.stderr.write('RuntimeError: CUDA out of memory. Tried to "
        "allocate 2 GiB on GPU 0')\nsys.exit(1)\n"))
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["plan_failed_irreparably"] is True
    record = result["stages"][0]
    assert record["repairs"][0]["error_type"] == "gpu_only"


def test_run_falls_back_to_generated_script(tmp_path):
    """官方计划不可修复失败 -> run() 回退生成脚本路径并合并记录。"""
    unit = _unit(tmp_path, content="import sys\nsys.exit(1)\n")
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    llm = _ScriptedLLM(["print('Training complete. Test accuracy: 85.2%')\n"])
    executor = CodeExecutorAgent(llm, mock_mode=True)
    result = executor.run({
        "paper_info": {"method": "演示", "dataset": "合成", "metrics": {}},
        "env_config": {},
        "execution_plan": _plan([unit], steps),
    })
    assert result["execution_mode"] == "generated_fallback"
    assert result["plan_execution"]["plan_failed_irreparably"] is True
    assert result["success"] is True          # 生成脚本路径成功
    assert llm.call_count >= 1


# ---------------- 6. 真实模式隔离门 ----------------

def test_real_mode_without_docker_refuses_plan(tmp_path):
    unit = _unit(tmp_path)
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    executor = _executor(mock_mode=False)
    executor.use_docker = False
    result = executor._execute_plan(_plan([unit], steps), {})
    assert result["plan_failed_irreparably"] is True
    assert result["final"]["exit_code"] == EXIT_ISOLATION_REQUIRED


# ---------------- 7. Docker 路径：两次独立 run ----------------

def test_docker_plan_step_splits_install_and_run(tmp_path, monkeypatch):
    unit = _unit(tmp_path)
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 42,
              "install_pkgs": ["gdown"]}]
    calls = []

    def fake_docker(base_cmd, image, runner, timeout):
        calls.append({"base": base_cmd, "image": image, "runner": runner,
                      "timeout": timeout})
        return subprocess.CompletedProcess(
            runner, 0, stdout="mse: 0.01\n", stderr=""), \
            {"hardened": True, "level": 0, "degraded": False}

    executor = _executor()
    executor.use_docker = True
    executor.env_config = {"image_tag": "python:3.11-slim"}
    monkeypatch.setattr(executor, "_resolve_docker_cmd", lambda: "docker")
    monkeypatch.setattr(executor, "_run_docker_cmd_with_sandbox", fake_docker)
    result = executor._execute_plan(_plan([unit], steps), {})
    assert result["success"] is True
    assert len(calls) == 2                      # 安装 run + 脚本 run
    install_call, run_call = calls
    assert "pip install" in " ".join(install_call["runner"])
    assert "gdown" in " ".join(install_call["runner"])
    assert install_call["timeout"] == ce_mod.DOCKER_INSTALL_TIMEOUT
    assert "-w" in install_call["base"] and "/app/main" in install_call["base"]
    assert run_call["timeout"] == 42            # 脚本预算与安装预算分离
    assert "python run.py" in " ".join(run_call["runner"])
    assert run_call["image"] == "python:3.11-slim"


# ---------------- 8. 缩参走 smoke_cmd（bash 入口） ----------------

def test_timeout_repairs_with_smoke_cmd(tmp_path):
    """bash 入口超时 -> 用 smoke_cmd 单次调用重试（不是追加到 bash 后面）。

    官方 .sh 不转发 "$@"，把缩参追加到 `bash x.sh` 后面会被 shell 忽略，
    等于原样重跑一遍。这里断言修复动作走的是 smoke_cmd。
    """
    unit = _unit(tmp_path)
    steps = [{"step_id": "run_0", "kind": "run",
              "cmd": "sleep 5",              # 模拟官方 .sh 整体超时
              "smoke_cmd": "python run.py",  # 缩参后的单次调用
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 1}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["success"] is True
    record = result["stages"][0]
    assert record["repairs"][0]["strategy"] == "smoke_cmd"
    assert record["final_cmd"] == "python run.py"
    assert result["actual_metrics"]["mse"] == 0.01   # 缩参命令真的产出了指标


def test_smoke_cmd_takes_precedence_over_smoke_args(tmp_path):
    unit = _unit(tmp_path)
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "sleep 5",
              "smoke_cmd": "python run.py",
              "smoke_args": {"--fast": "1"},
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 1}]
    record = _executor()._execute_plan(_plan([unit], steps),
                                       {})["stages"][0]
    assert record["repairs"][0]["strategy"] == "smoke_cmd"
    assert record["final_cmd"] == "python run.py"


# ---------------- 9. 数据集缺失是独立失败类 ----------------

def test_missing_dataset_diagnosed(tmp_path):
    """官方脚本读 ./dataset/... 但数据集在网盘 -> 明确归类，不当作代码缺陷。"""
    unit = _unit(tmp_path, content=(
        "import sys\n"
        "sys.stderr.write(\"FileNotFoundError: [Errno 2] No such file or \"\n"
        "                \"directory: './dataset/ETT-small/ETTh1.csv'\")\n"
        "sys.exit(1)\n"))
    steps = [{"step_id": "run_0", "kind": "run", "cmd": "python run.py",
              "cwd": "/app/main", "unit_id": "main", "timeout_s": 30}]
    result = _executor()._execute_plan(_plan([unit], steps), {})
    assert result["plan_failed_irreparably"] is True
    diagnosis = result["stages"][0]["repairs"][0]
    assert diagnosis["error_type"] == "missing_dataset"
    assert "网盘" in diagnosis["detail"]


# ---------------- 10. 容器挂载目录权限（nobody 必须能进能写） ----------------

def test_chmod_tree_writable_grants_traverse_and_write(tmp_path):
    """0700 的 mkdtemp 只加 o+w 会得到 0702——nobody 连 /app 都进不去。"""
    root = tmp_path / "ws"
    root.mkdir(mode=0o700)
    sub = root / "main"
    sub.mkdir()
    script = sub / "run.py"
    script.write_text("print('mse: 0.1')\n", encoding="utf-8")

    CodeExecutorAgent._chmod_tree_writable(root)

    assert root.stat().st_mode & 0o001          # o+x：容器用户要 traverse
    assert sub.stat().st_mode & 0o001
    assert script.stat().st_mode & 0o004        # o+r：要能读脚本
    assert script.stat().st_mode & 0o002        # o+w：脚本要能写产出


def test_prepare_pip_target_lives_in_mounted_volume(tmp_path):
    """pip 安装目标必须在挂载卷内（/app 下），才能从安装步活到脚本步。"""
    target = CodeExecutorAgent._prepare_pip_target(str(tmp_path))
    assert target == "/app/.autorepro_site"
    created = tmp_path / ".autorepro_site"
    assert created.is_dir()
    assert created.stat().st_mode & 0o007 == 0o007


# ---------------- 10. 计划执行 -> 验证 -> 报告 全链路 ----------------

def test_plan_result_flows_to_validation_and_report(tmp_path):
    """计划模式跑出的指标必须一路走到验证结论并进报告。

    这里刻意用官方仓库最常见的形态：多个 run 块（iTransformer 的
    pred_len 96/192/336/720 就是四段）+ 末尾一个 parse 步。第一个 run 块
    成功并吐出指标，第二个失败 -> 依赖它的 parse 步被跳过（stdout 为空，
    且它就是 stages[-1]）。

    VALIDATE 原先只读 stages[-1]，于是：明明跑出了 mse/mae，却因为末步
    是空 stdout 的跳过步而判成"执行未产出任何输出"（not_runnable），把
    一次部分成功的复现说成压根没跑起来。报告第 4 节同样只认生成路径的
    execution 结构，会把计划模式打成 "None: ❌ 失败(退出码 None)"。
    """
    from src.agents.report_generator import ReportGeneratorAgent
    from src.agents.result_validator import ResultValidatorAgent

    unit = _unit(tmp_path)                       # run.py 打印 mse/mae
    execution = _executor()._execute_plan(_plan([unit], [
        {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30,
         "expects": {"metrics": ["mse", "mae"]}},
        {"step_id": "parse", "kind": "parse", "cmd": "",
         "cwd": "/app/main", "unit_id": "main", "depends_on": ["run_0"]},
    ]), {})
    assert execution["success"] is True
    assert execution["final"]["stdout"].strip()              # 有汇总输出

    # 换成"末步被跳过"的形态：run_1 失败 -> parse 依赖它 -> 末步 stdout 为空
    # 第一个 run 块成功、第二个（--pred_len 720 那一段）失败
    bad_unit = _unit(tmp_path, content=(
        "import sys\n"
        "print('mse: 0.01, mae: 0.02')\n"
        "sys.exit(1 if '--pred_len' in sys.argv else 0)\n"))
    bad_execution = _executor()._execute_plan(_plan([bad_unit], [
        {"step_id": "run_0", "kind": "run", "cmd": "python run.py",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30},
        {"step_id": "run_1", "kind": "run",
         "cmd": "python run.py --pred_len 720",
         "cwd": "/app/main", "unit_id": "main", "timeout_s": 30},
        {"step_id": "parse", "kind": "parse", "cmd": "",
         "cwd": "/app/main", "unit_id": "main", "depends_on": ["run_1"]},
    ]), {})
    assert bad_execution["stages"][-1].get("skipped_deps") is True
    assert bad_execution["stages"][-1].get("stdout", "") == ""
    assert [s["step_id"] for s in bad_execution["stages"]] == \
        ["run_0", "run_1", "parse"]
    assert bad_execution["stages"][0]["success"] is True     # 第一段跑通了
    assert bad_execution["final"]["success"] is False        # 但整体有真失败

    validation = ResultValidatorAgent(LLMClient(mock_mode=True)).run({
        "paper_info": {"metrics": {"mse": 0.01, "mae": 0.02}},
        "execution": execution,
    })
    assert validation["metrics_comparison"]["actual"] == {"mse": 0.01,
                                                          "mae": 0.02}
    assert validation["is_reproduced"] is True

    # 末步空 stdout 的失败执行：不能报成"未产出任何输出"
    failed_validation = ResultValidatorAgent(LLMClient(mock_mode=True)).run({
        "paper_info": {"metrics": {"mse": 0.01}},
        "execution": bad_execution,
    })
    assert failed_validation["status"] != "not_runnable"
    assert failed_validation["metrics_comparison"]["actual"] == {"mse": 0.01,
                                                                 "mae": 0.02}

    report = ReportGeneratorAgent().run({
        "paper_info": {"title": "合成论文", "metrics": {"mse": 0.01}},
        "resources": {}, "env_config": {}, "validation": validation,
        "optimization": {}, "audit_stats": {}, "execution": execution,
    })["report"]
    sec = report.split("## 4. 代码执行")[1].split("## 5.")[0]
    assert "官方代码执行计划" in sec
    assert "None: " not in sec and "(退出码 None)" not in sec
    assert "mse=0.01" in sec                                 # 实际指标可见
