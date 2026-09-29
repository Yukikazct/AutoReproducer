"""计划模式下执行输出流向 VALIDATE 的回归测试。

背景缺陷：`ResultValidatorAgent.run` 只读 `execution["stdout"]`，为空时
退回 `execution["stages"][-1]["stdout"]`。计划模式的 execution 没有顶层
stdout——汇总输出在 `final` 里（`CodeExecutor._execute_plan` 把各 run 步的
stdout 拼好放进去），而 `stages[-1]` 往往是 parse 步或被跳过的步骤，stdout
恒为空。后果有二：

1. 明明跑出了 `mse:…, mae:…`，`actual_metrics` 却是空 -> 有声明指标但无实测值
   -> 判"未复现"（把成功的复现报成失败）；
2. 失败的 run 步若把错误写进 `final.stdout`，会被 `_detect_not_runnable` 判成
   "执行未产出任何输出"，抹掉真实原因。

注意生成路径不受影响：`_execute_with_repair` 失败时 `return stages[-1]`，
成功时 `final` 就是最后一个 stage 的 dict，即 `final` 恒等于 `stages[-1]`，
优先级调整对它是零行为变化（由 tests/test_reproduction_core.py 覆盖）。

运行: python -m pytest tests/test_validator_plan_stdout.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.result_validator import ResultValidatorAgent  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402


def _plan_execution(final: dict, stages=None) -> dict:
    """计划模式 execution：无顶层 stdout，输出只在 final。"""
    execution = {
        "success": bool(final.get("success")),
        "execution_mode": "plan",
        "final": final,
        "stages": stages if stages is not None else [
            {"step_id": "run_0", "kind": "run", "success": True,
             "stdout": "mse:0.428, mae:0.421", "stderr": "", "exit_code": 0},
            # parse 步（或 skipped_deps 的跳过步）排在最后，stdout 为空：
            # 只看 stages[-1] 就会把上面的指标丢掉
            {"step_id": "parse", "kind": "parse", "success": True,
             "stdout": "", "stderr": "", "exit_code": 0},
        ],
        "actual_metrics": {"mse": 0.428, "mae": 0.421},
    }
    return execution


@pytest.fixture()
def validator():
    return ResultValidatorAgent(LLMClient(mock_mode=True))


def test_plan_final_stdout_reaches_metrics_extraction(validator):
    """跑出指标的计划执行必须被判定为复现成功，而不是"无实测值"。"""
    result = validator.run({
        "paper_info": {"metrics": {"mse": 0.428}},
        "execution": _plan_execution(
            {"success": True, "stdout": "mse:0.428, mae:0.421",
             "stderr": "", "exit_code": 0}),
    })
    assert result["metrics_comparison"]["actual"] == {"mse": 0.428, "mae": 0.421}
    assert result["is_reproduced"] is True
    assert result["status"] == "reproduced"


def test_plan_last_stage_empty_does_not_mask_metrics(validator):
    """stages[-1] 是空 stdout 的 parse 步时，也不能丢掉 run 步的指标。"""
    execution = _plan_execution(
        {"success": True, "stdout": "mse:0.428", "stderr": "", "exit_code": 0})
    execution["stages"] = [execution["stages"][1]]     # 只剩那个空的 parse 步
    result = validator.run({"paper_info": {"metrics": {"mse": 0.428}},
                            "execution": execution})
    assert result["metrics_comparison"]["actual"] == {"mse": 0.428}
    assert result["is_reproduced"] is True


def test_failed_plan_with_output_is_not_reported_as_no_output(validator):
    """失败的 run 步有输出时，不能报成"执行未产出任何输出"。"""
    result = validator.run({
        "paper_info": {"metrics": {"mse": 0.428}},
        "execution": _plan_execution(
            {"success": False, "stdout": "step run_0 failed\nmse: 9.9",
             "stderr": "RuntimeError: CUDA out of memory", "exit_code": 1},
            stages=[{"step_id": "run_0", "kind": "run", "success": False,
                     "stdout": "", "stderr": "", "exit_code": 1}]),
    })
    assert result["status"] != "not_runnable"
    # 有输出、指标对不上 -> 如实判"未复现"，而不是"无法验证"
    assert result["is_reproduced"] is False
    assert result["metrics_comparison"]["actual"] == {"mse": 9.9}


def test_top_level_stdout_still_wins(validator):
    """旧结构（顶层 stdout）优先级不变。"""
    execution = _plan_execution(
        {"success": True, "stdout": "mse:9.9", "stderr": "", "exit_code": 0})
    execution["stdout"] = "mse:0.428"
    result = validator.run({"paper_info": {"metrics": {"mse": 0.428}},
                            "execution": execution})
    assert result["metrics_comparison"]["actual"] == {"mse": 0.428}
    assert result["is_reproduced"] is True


def test_final_stderr_used_when_top_level_missing(validator):
    """顶层 stderr 缺失时，final.stderr 要能支撑 not_runnable 的原因文案。"""
    execution = {
        "success": False, "execution_mode": "plan",
        "final": {"success": False, "stdout": "", "exit_code": 1,
                  "stderr": "bash: scripts/x.sh: No such file or directory"},
        "stages": [],
    }
    result = validator.run({"paper_info": {"metrics": {}},
                            "execution": execution})
    assert result["status"] == "not_runnable"
    assert "No such file or directory" in result["reason"]
