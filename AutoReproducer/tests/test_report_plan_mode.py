"""报告在官方代码执行计划模式下的渲染测试。

背景缺陷：第 4 节原先只认生成脚本路径的 execution 结构——
- 计划模式的 stages 记录键是 step_id/kind/cmd/unit_id/skipped_deps/repairs，
  没有 stage/exit_code，沿用生成路径渲染会打成
  "None: ❌ 失败 (退出码 None)"；
- execution.code 在计划模式下是官方入口脚本原文（.sh 居多），却被套进
  ```python 代码块，等于把 shell 脚本当 Python 展示；
- 代码单元、actual_metrics、plan_fail_reason 三个字段完全没有出口，
  读者看不出"官方代码到底跑了什么、跑出什么、为什么回退"。

运行: python -m pytest tests/test_report_plan_mode.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.report_generator import ReportGeneratorAgent  # noqa: E402


def _base_data(execution: dict) -> dict:
    return {
        "paper_info": {"title": "iTransformer", "method": "iTransformer",
                       "dataset": "ETTh1", "metrics": {"mse": 0.428}},
        "resources": {}, "env_config": {}, "validation": {},
        "optimization": {}, "audit_stats": {}, "execution": execution,
    }


def _section4(report: str) -> str:
    return report.split("## 4. 代码执行")[1].split("## 5.")[0]


def _plan_execution() -> dict:
    """计划模式的成功结果（结构与 CodeExecutor._execute_plan 返回一致）。"""
    return {
        "success": True,
        "execution_mode": "plan",
        "code": "export CUDA_VISIBLE_DEVICES=1\npython -u run.py --seq_len 96\n",
        "plan": {
            "paper_id": "pid", "source": "heuristic",
            "units": [{"unit_id": "main", "role": "main",
                       "url": "https://github.com/thuml/iTransformer",
                       "fetch_state": "cloned"}],
            "steps": [
                {"step_id": "install_main", "kind": "install",
                 "cmd": "pip install -q -r requirements.txt"},
                {"step_id": "run_0", "kind": "run",
                 "cmd": "bash scripts/multivariate_forecasting/ETT/"
                        "iTransformer_ETTh1.sh"},
            ],
            "entry": {"script": "scripts/multivariate_forecasting/ETT/"
                                "iTransformer_ETTh1.sh",
                      "unit_id": "main", "interp": "bash"},
            "notes": ["数据集需从 README 网盘链接下载"],
        },
        "stages": [
            {"step_id": "install_main", "kind": "install", "cmd": "pip install",
             "unit_id": "main", "success": True, "stdout": "", "stderr": "",
             "exit_code": 0, "repairs": [], "repair_attempts": 0},
            {"step_id": "run_0", "kind": "run",
             "cmd": "bash scripts/multivariate_forecasting/ETT/"
                    "iTransformer_ETTh1.sh",
             "unit_id": "main", "success": True, "stdout": "mse:0.428, mae:0.421",
             "stderr": "", "exit_code": 0,
             "repairs": [], "repair_attempts": 0,
             "final_cmd": "bash scripts/multivariate_forecasting/ETT/"
                          "iTransformer_ETTh1.sh"},
        ],
        "final": {"success": True, "stdout": "mse:0.428, mae:0.421",
                  "stderr": "", "exit_code": 0},
        "actual_metrics": {"mse": 0.428, "mae": 0.421},
        "plan_failed_irreparably": False,
    }


def test_plan_mode_renders_units_steps_and_metrics():
    report = ReportGeneratorAgent().run(
        _base_data(_plan_execution()))["report"]
    sec = _section4(report)
    assert "官方代码执行计划" in sec
    assert "确定性启发式兜底" in sec
    # 代码单元表：多代码块整体调用的可见性来源
    assert "### 代码单元" in sec
    assert "thuml/iTransformer" in sec and "cloned" in sec
    # 逐步记录
    assert "### 执行步骤" in sec
    assert "install_main" in sec and "run_0" in sec
    assert "✅ 通过" in sec
    # 实际指标
    assert "实际指标" in sec and "mse=0.428" in sec and "mae=0.421" in sec
    # 入口脚本用 bash 代码块，不当成 Python
    assert "### 官方入口脚本" in sec
    assert "```bash" in sec and "```python" not in sec
    # 规划备注
    assert "网盘链接" in sec


def test_plan_mode_does_not_print_none_stage_fields():
    """计划模式的 stages 没有 stage/exit_code 键，不能打成 None。"""
    sec = _section4(ReportGeneratorAgent().run(
        _base_data(_plan_execution()))["report"])
    assert "None: " not in sec
    assert "(退出码 None)" not in sec


def test_plan_mode_renders_repairs_and_skipped_deps():
    execution = _plan_execution()
    execution["stages"][1].update({
        "success": False, "exit_code": -1, "timed_out": True,
        "final_cmd": "python -u run.py --seq_len 32 --pred_len 32",
        "repair_attempts": 1,
        "repairs": [{"error_type": "timeout", "strategy": "smoke_cmd",
                     "detail": "缩参单次调用重试"}],
    })
    execution["stages"].append({
        "step_id": "parse", "kind": "parse", "cmd": "", "unit_id": "main",
        "success": False, "skipped_deps": True, "repairs": []})
    sec = _section4(ReportGeneratorAgent().run(
        _base_data(execution))["report"])
    assert "修复记录" in sec and "smoke_cmd" in sec
    # 缩参后的真实命令必须可见（否则"修复了什么"只能靠猜）
    assert "--seq_len 32" in sec
    assert "⏭️ 跳过（前置步骤失败）" in sec


def test_missing_dataset_diagnosis_visible_in_report():
    execution = _plan_execution()
    execution["stages"][1].update({
        "success": False, "exit_code": 1,
        "repairs": [{"error_type": "missing_dataset",
                     "strategy": "non_repairable",
                     "detail": "缺少数据集文件（官方脚本从 README 的网盘链接"
                               "下载，仓库内不含数据）"}],
        "repair_attempts": 1,
    })
    sec = _section4(ReportGeneratorAgent().run(
        _base_data(execution))["report"])
    assert "missing_dataset" in sec
    assert "缺少数据集文件" in sec


def test_generated_fallback_renders_reason_and_both_paths():
    """回退路径：既要显示官方执行记录，也要显示回退原因与生成脚本结果。"""
    plan_result = _plan_execution()
    plan_result["success"] = False
    plan_result["plan_failed_irreparably"] = True
    plan_result["plan_fail_reason"] = "官方脚本需要 GPU，本环境不可修复"
    execution = {
        "success": True, "execution_mode": "generated_fallback",
        "code": "print('mse: 0.5')",
        "plan_execution": plan_result,
        "plan_fail_reason": "官方脚本需要 GPU，本环境不可修复",
        "stages": [{"stage": "full", "success": True, "exit_code": 0}],
        "final": {"success": True, "stdout": "mse: 0.5", "stderr": "",
                  "exit_code": 0},
        "best_effort": False, "fallback_used": False, "sanitize_stats": {},
    }
    sec = _section4(ReportGeneratorAgent().run(
        _base_data(execution))["report"])
    assert "回退到生成脚本路径" in sec
    assert "官方脚本需要 GPU" in sec              # 回退原因
    assert "run_0" in sec                          # 官方执行记录仍在
    assert "### 生成代码" in sec                   # 生成路径照常渲染
    assert "### 执行输出(full)" in sec


def test_generated_mode_unchanged():
    """生成路径（无 execution_mode）渲染必须与改动前一致。"""
    execution = {
        "code": "print('mse: 0.5')\nprint('done')",
        "stages": [{"stage": "full", "success": True, "exit_code": 0}],
        "final": {"success": True, "stdout": "mse: 0.5\nlast line",
                  "stderr": "", "exit_code": 0},
        "sanitize_stats": {"prose_dropped": 2, "code_dropped": 0},
        "best_effort": False, "fallback_used": False,
    }
    sec = _section4(ReportGeneratorAgent().run(
        _base_data(execution))["report"])
    assert "代码长度" in sec
    assert "叙述行 2 行" in sec
    assert "  - full: ✅ 通过 (退出码 0)" in sec
    assert "### 生成代码" in sec and "```python" in sec
    assert "### 执行输出(full)" in sec
    assert "print('done')" in sec                  # 代码末行必须在
    assert "last line" in sec                      # 输出末行必须在
    assert "官方" not in sec                        # 不掺计划模式的措辞
