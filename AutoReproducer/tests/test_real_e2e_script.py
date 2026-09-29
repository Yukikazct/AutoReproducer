"""real_e2e 脚本的可见性输出测试。

这个脚本是真实模式唯一的端到端入口，它读的键名一旦和 CodeExecutor 实际
产出的结构对不上（例如把 `plan_execution` 当成顶层 `plan`），结果就是
"跑了一小时，屏幕上一句有用的话都没有"——而它本身不被任何测试覆盖。
这里用合成 execution 字典把三件事钉死：

1. 计划模式：unit 表、入口、步骤状态（✅/跳过/未到达）、修复记录、实际指标；
2. 回退模式：官方执行记录在 `plan_execution` 里，仍要打印出来；
3. 生成模式：不报错、不留空段。

运行: python -m pytest tests/test_real_e2e_script.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import real_e2e  # noqa: E402


def _plan() -> dict:
    return {
        "source": "heuristic",
        "units": [{"unit_id": "main", "role": "main", "fetch_state": "cloned",
                   "url": "https://github.com/thuml/iTransformer"}],
        "entry": {"script": "scripts/multivariate_forecasting/ETT/"
                            "iTransformer_ETTh1.sh",
                  "interp": "bash", "unit_id": "main"},
        "notes": ["数据集需从 README 网盘链接下载"],
        "steps": [
            {"step_id": "install_main", "kind": "install",
             "cmd": "pip install -q -r requirements.txt"},
            {"step_id": "run_0", "kind": "run",
             "cmd": "bash scripts/multivariate_forecasting/ETT/"
                    "iTransformer_ETTh1.sh"},
            {"step_id": "parse", "kind": "parse", "cmd": ""},
        ],
    }


def _stages():
    return [
        {"step_id": "install_main", "kind": "install", "success": True,
         "exit_code": 0, "repairs": []},
        {"step_id": "run_0", "kind": "run", "success": False, "exit_code": 1,
         "repairs": [{"error_type": "missing_dataset",
                      "strategy": "non_repairable",
                      "detail": "缺少数据集文件"}]},
        {"step_id": "parse", "kind": "parse", "success": False,
         "skipped_deps": True, "repairs": []},
    ]


def test_plan_mode_prints_units_steps_and_metrics(capsys):
    data = {"execution": {
        "success": False, "execution_mode": "plan", "plan": _plan(),
        "stages": _stages(),
        "final": {"success": False, "stdout": "", "stderr": "", "exit_code": 1},
        "actual_metrics": {"mse": 0.428},
        "plan_failed_irreparably": True,
        "plan_fail_reason": "数据集不在仓库内，无法自动下载",
    }}
    real_e2e._print_plan(data)
    out = capsys.readouterr().out
    assert "thuml/iTransformer" in out and "cloned" in out
    assert "iTransformer_ETTh1.sh" in out and "interp=bash" in out
    assert "网盘链接" in out
    assert "✅" in out and "退出码 1" in out
    assert "⏭️ 跳过（前置失败）" in out          # skipped_deps 分支
    assert "missing_dataset -> non_repairable" in out
    assert "{'mse': 0.428}" in out               # 实际指标不能丢
    assert "数据集不在仓库内" in out


def test_fallback_reads_plan_execution(capsys):
    """回退时官方记录在 plan_execution 里，仍必须打印。"""
    plan_result = {"plan": _plan(), "stages": _stages(),
                   "actual_metrics": {},
                   "plan_failed_irreparably": True,
                   "plan_fail_reason": "官方脚本需要 GPU"}
    data = {"execution": {
        "success": True, "execution_mode": "generated_fallback",
        "code": "print('mse: 0.5')",
        "plan_execution": plan_result,
        "plan_fail_reason": "官方脚本需要 GPU",
        "stages": [{"stage": "full", "success": True, "exit_code": 0}],
        "final": {"success": True, "stdout": "mse: 0.5", "stderr": "",
                  "exit_code": 0},
    }}
    real_e2e._print_plan(data)
    out = capsys.readouterr().out
    assert "generated_fallback" in out
    assert "iTransformer_ETTh1.sh" in out        # 不是"(无计划)"
    assert "官方脚本需要 GPU" in out

    real_e2e._print_generated(data)
    out = capsys.readouterr().out
    assert "⚠ 官方计划失败后回退" in out
    assert "print('mse: 0.5')" in out            # 代码末行
    assert "mse: 0.5" in out                     # stdout 尾部


def test_generated_mode_without_plan_is_quiet(capsys):
    """纯生成路径（无计划）：不炸、不打印计划段正文。"""
    data = {"execution": {
        "code": "print('hi')\nprint('done')",
        "stages": [{"stage": "full", "success": True, "exit_code": 0}],
        "final": {"success": True, "stdout": "hi\ndone", "stderr": "",
                  "exit_code": 0},
        "best_effort": False, "fallback_used": False, "sanitize_stats": {},
    }}
    real_e2e._print_plan(data)
    out = capsys.readouterr().out
    assert "execution_mode: generated" in out
    assert "无计划" in out
    assert "run_0" not in out

    real_e2e._print_generated(data)
    out = capsys.readouterr().out
    assert "done" in out


def test_arg_parsing_defaults_and_overrides():
    args = real_e2e._parse_args([])
    assert args.pdf == real_e2e.DEFAULT_PDF      # 什么都不给 -> 内置样例
    assert args.use_docker is False

    args = real_e2e._parse_args(["--paper-title", "iTransformer",
                                 "--use-docker", "--repo-url", "https://x/y"])
    assert args.paper_title == "iTransformer"
    assert args.pdf == ""                        # 有标题就不塞默认 PDF
    assert args.use_docker is True
    assert args.repo_url == "https://x/y"

    args = real_e2e._parse_args(["p.pdf", "--max-trials", "5"])
    assert args.pdf == "p.pdf" and args.max_trials == 5
