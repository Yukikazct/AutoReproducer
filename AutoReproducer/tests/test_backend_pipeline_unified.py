"""驱动器统一后的 UI 路径回归测试（修复 iTransformer 失败的根因）。

覆盖：
1. UI 路径（run_pipeline_core -> Orchestrator）也会执行资源懒加载：
   storage.fetched 被填充（此前 backend_pipeline 自持循环漏掉了
   Orchestrator._fetch_resources，官方仓库被发现却从未 clone）；
2. 进度事件序列与展示名保持兼容（每个 Agent 卡片都有状态事件）；
3. 结果键完整（state/error/data/audit_logs/audit_stats/report_path/
   session_id），FINISH ledger 终态落盘；
4. Orchestrator 的 progress_cb 通知序列：running -> success 依次出现。

运行: python -m pytest tests/test_backend_pipeline_unified.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.backend_pipeline import (  # noqa: E402
    AGENTS,
    ProgressStore,
    run_pipeline_core,
)
from src.orchestrator import Orchestrator, STAGE_DISPLAY  # noqa: E402


def _ledger_records(session_id: str):
    root = Path(__file__).resolve().parents[1]
    ledger_file = (root / "data" / "experiment_ledger"
                   / f"ledger_{session_id}.jsonl")
    return [json.loads(line) for line in
            ledger_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_ui_path_populates_storage_fetched(tmp_path):
    """UI 路径必须执行资源懒加载（iTransformer 失败根因回归）。

    即使仓库发现结果为空（Dummy 标题 + mock 离线模式），storage.fetched
    也应携带 code/dataset/weights/units 键——证明 _fetch_resources 确实
    被调用，而不是被 UI 循环跳过。
    """
    progress = tmp_path / "p.jsonl"
    result = run_pipeline_core(str(progress), paper_title="Dummy Paper",
                               mock_mode=True, max_trials=1)
    assert result["state"] == "COMPLETED", result.get("error")

    fetched = result["data"].get("storage", {}).get("fetched")
    assert fetched is not None, "UI 路径未执行资源懒加载（storage.fetched 缺失）"
    # 子集断言：Stage 2 多单元拉取会在此基础上追加 units 键
    assert {"code", "dataset", "weights"} <= set(fetched)


def test_ui_progress_events_keep_display_names(tmp_path):
    """进度事件序列：每个展示名（含 Verifier/Optimizer/ReportGenerator）
    都有状态事件，且 running 先于 success 出现。"""
    progress = tmp_path / "p.jsonl"
    result = run_pipeline_core(str(progress), paper_title="Dummy Paper",
                               mock_mode=True, max_trials=1)
    assert result["state"] == "COMPLETED", result.get("error")

    events = []
    for line in progress.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "state":
            events.append(ev)

    names = [a[1] for a in AGENTS] + ["Verifier", "Optimizer",
                                      "ReportGenerator"]
    seen_names = {ev.get("agent") for ev in events}
    for n in names:
        assert n in seen_names, f"缺少 {n} 的状态事件"

    # 每个展示名的状态按 running -> success/waiting/error 顺序出现
    # （Verifier 例外：UI 契约只发一条 success，与旧事件序列一致）
    order = {n: [] for n in names}
    for ev in events:
        if ev.get("agent") in order:
            order[ev["agent"]].append(ev.get("status"))
    for n, statuses in order.items():
        if not statuses or n == "Verifier":
            continue
        assert statuses[0] == "running", f"{n} 首条事件应为 running"


def test_ui_result_has_complete_keys_and_finish_ledger(tmp_path):
    """结果键完整 + ledger 末条 FINISH 携带终态。"""
    progress = tmp_path / "p.jsonl"
    result = run_pipeline_core(str(progress), paper_title="Dummy Paper",
                               mock_mode=True, max_trials=1)
    assert result["state"] == "COMPLETED", result.get("error")
    for key in ("state", "error", "data", "audit_logs", "audit_stats",
                "report_path", "session_id"):
        assert key in result, f"结果缺少键 {key}"
    assert result["report_path"]  # 报告已落盘
    assert Path(result["report_path"]).exists()

    records = _ledger_records(result["session_id"])
    assert records[-1]["phase"] == "FINISH"
    assert records[-1]["result"]["state"] == "COMPLETED"


def test_orchestrator_progress_cb_sequence():
    """Orchestrator.progress_cb：各阶段按 running -> success 通知。"""
    events = []

    def _cb(state_name, display_name, status):
        events.append((state_name, display_name, status))

    orch = Orchestrator(mock_mode=True, progress_cb=_cb,
                        max_trials=1)
    result = orch.run({"paper_title": "Dummy Paper"})
    assert result["state"] == "COMPLETED"

    states = [ev[0] for ev in events]
    for expected in ("READ_PAPER", "FIND_RESOURCES", "BUILD_ENV",
                     "EXECUTE_CODE", "VALIDATE", "GENERATE_REPORT",
                     "COMPLETED"):
        assert expected in states, f"缺少阶段事件 {expected}"
    # 每阶段先 running 后 success；展示名与 STAGE_DISPLAY 一致
    by_state = {}
    for state_name, display_name, status in events:
        by_state.setdefault(state_name, []).append(status)
        assert display_name == STAGE_DISPLAY.get(state_name, state_name)
    for state_name in ("READ_PAPER", "EXECUTE_CODE", "GENERATE_REPORT"):
        statuses = by_state[state_name]
        assert statuses[0] == "running" and "success" in statuses


def test_curated_itransformer_selected_first():
    """iTransformer 标题的 curated 回退必须选中官方仓库（最长关键词优先）。"""
    from src.agents.repo_discovery import (
        curated_repo_fallback_candidates,
        discover_repositories,
    )
    query = ("iTransformer: Inverted Transformers Are Effective for "
             "Time Series Forecasting")
    curated = curated_repo_fallback_candidates(query)
    urls = [c.repo_urls[0] for c in curated]
    assert urls[0] == "https://github.com/thuml/iTransformer", urls
    assert "https://github.com/thuml/Time-Series-Library" in urls

    discovery = discover_repositories(query, offline=True)
    assert discovery["selected_repo"] == "https://github.com/thuml/iTransformer"
    assert discovery["fallback_used"] is True


def test_timeout_diagnosis_is_repairable():
    """超时诊断必须可修复，且不被退出码 -1 的不可修复门覆盖。"""
    from src.agents.code_executor import CodeExecutorAgent
    diagnosis = CodeExecutorAgent._diagnose_execution_error({
        "stdout": "", "stderr": "执行超时(30s, smoke)", "exit_code": -1,
    })
    assert diagnosis["error_type"] == "timeout"
    assert diagnosis["repairable"] is True
