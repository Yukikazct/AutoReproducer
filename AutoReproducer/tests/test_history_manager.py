"""历史记录管理模块单元测试：list_sessions / get_storage_stats / cleanup_runtime / get_session_detail / format_size。

通过 monkeypatch 将数据目录指向临时目录，不触碰真实 data/。
"""
import json
import time
from pathlib import Path

import pytest

from frontend.history_manager import (
    cleanup_runtime,
    format_size,
    get_project_data_dir,
    get_session_detail,
    get_storage_stats,
    list_sessions,
    delete_session,
    clear_sessions,
    _is_finished_progress,
    _session_id_from_progress,
)


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch):
    """构造临时数据目录并注入 history_manager."""
    monkeypatch.setattr("frontend.history_manager.get_project_data_dir",
                        lambda: tmp_path)
    return tmp_path


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mk_ledger(data_dir: Path, sid: str, title: str = "测试论文", state: str = "COMPLETED"):
    record = {
        "type": "start",
        "outputs": {"title": title, "paper_info": {"title": title}},
        "result": {"state": state, "duration_sec": 12.5, "llm_calls": 35},
        "timestamp": "2026-09-10T10:00:00",
    }
    _write_jsonl(data_dir / "experiment_ledger" / f"ledger_{sid}.jsonl", [record])


def _mk_real_ledger(data_dir: Path, sid: str, title: str = "梯度下降实验",
                    state: str = "COMPLETED", duration: float = 12.5,
                    llm_calls: int = 35):
    """构造「真实结构」ledger：首条 READ_PAPER（outputs.title + 空 result），
    末条 FINISH（result 携带终态 state/duration/llm_calls）。"""
    records = [
        {
            "session_id": sid,
            "phase": "READ_PAPER",
            "decision": "解析论文",
            "inputs": {"paper_title": title},
            "outputs": {"title": title, "method": "梯度下降", "metrics": {}},
            "result": {},
        },
        {
            "session_id": sid,
            "phase": "FINISH",
            "decision": "流水线终止",
            "inputs": {"paper_title": title},
            "outputs": {},
            "result": {"state": state, "duration_sec": duration,
                       "llm_calls": llm_calls},
        },
    ]
    _write_jsonl(data_dir / "experiment_ledger" / f"ledger_{sid}.jsonl", records)


# ---------- list_sessions ----------

def test_list_sessions_empty(fake_data: Path):
    assert list_sessions() == []


def test_list_sessions_detects_ledger(fake_data: Path):
    _mk_ledger(fake_data, "20260910_100000", "梯度下降实验")
    sessions = list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["session_id"] == "20260910_100000"
    assert s["paper_title"] == "梯度下降实验"
    assert s["state"] == "COMPLETED"
    assert s["duration_sec"] == 12.5
    assert s["llm_calls"] == 35


def test_list_sessions_counts_log_lines(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl",
                 [{"type": "log"} for _ in range(5)])
    sessions = list_sessions()
    assert sessions[0]["log_entries"] == 1 + 5  # ledger 1 行 + log 5 行


def test_list_sessions_matches_report(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    report = fake_data / "reports" / f"线性回归复现_{sid}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# 报告", encoding="utf-8")
    sessions = list_sessions()
    assert sessions[0]["report_path"] == str(report)


def test_list_sessions_ignores_unrelated_files(fake_data: Path):
    _mk_ledger(fake_data, "20260910_100000")
    (fake_data / "experiment_ledger" / "random.txt").write_text("x", encoding="utf-8")
    assert len(list_sessions()) == 1


def test_list_sessions_sorted_desc(fake_data: Path):
    _mk_ledger(fake_data, "20260910_090000", "旧")
    _mk_ledger(fake_data, "20260910_110000", "新")
    sids = [s["session_id"] for s in list_sessions()]
    assert sids == ["20260910_110000", "20260910_090000"]


def test_list_sessions_real_ledger_structure(fake_data: Path):
    """真实 ledger：标题在首条 READ_PAPER、终态在末条 FINISH —— 不再恒显未知/0/0。"""
    _mk_real_ledger(fake_data, "20260910_100000", "梯度下降实验",
                    state="COMPLETED", duration=12.5, llm_calls=35)
    sessions = list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["paper_title"] == "梯度下降实验"
    assert s["state"] == "COMPLETED"
    assert s["duration_sec"] == 12.5
    assert s["llm_calls"] == 35


def test_list_sessions_real_ledger_error_state(fake_data: Path):
    """终态 ERROR 也要如实回填，而非「未知」。"""
    _mk_real_ledger(fake_data, "20260910_110000", "某论文", state="ERROR",
                    duration=2.0, llm_calls=3)
    sessions = list_sessions()
    assert sessions[0]["state"] == "ERROR"
    assert sessions[0]["llm_calls"] == 3


def test_list_sessions_real_ledger_running_fallback(fake_data: Path):
    """末条 result 缺失终态时回退 RUNNING/0/0，不崩溃。"""
    _write_jsonl(
        fake_data / "experiment_ledger" / "ledger_20260910_120000.jsonl",
        [{"phase": "READ_PAPER", "outputs": {"title": "半途论文"}, "result": {}}])
    s = list_sessions()[0]
    assert s["state"] == "RUNNING"
    assert s["duration_sec"] == 0
    assert s["llm_calls"] == 0


# ---------- get_storage_stats ----------

def test_storage_stats_counts_files_and_bytes(fake_data: Path):
    p = fake_data / "experiment_ledger" / "ledger_20260910_100000.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * 1024, encoding="utf-8")
    stats = get_storage_stats()
    assert stats["experiment_ledger"]["files"] == 1
    assert stats["experiment_ledger"]["bytes"] == 1024
    assert stats["total"]["bytes"] == 1024


def test_storage_stats_missing_dirs_zero(fake_data: Path):
    stats = get_storage_stats()
    for key in ("logs", "runtime", "reports"):
        assert stats[key] == {"files": 0, "bytes": 0}


# ---------- cleanup_runtime ----------

def test_cleanup_runtime_removes_old_only(fake_data: Path):
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    old = runtime / "progress_old.jsonl"
    new = runtime / "progress_new.jsonl"
    old.write_text("o" * 100, encoding="utf-8")
    new.write_text("n" * 50, encoding="utf-8")
    # 把 old 文件修改时间改为 10 天前
    past = time.time() - 10 * 86400
    import os as _os
    _os.utime(old, (past, past))
    removed, freed = cleanup_runtime(keep_days=7)
    assert removed == 1
    assert freed == 100
    assert not old.exists()
    assert new.exists()


def test_cleanup_runtime_no_dir(fake_data: Path):
    assert cleanup_runtime() == (0, 0)


def test_cleanup_runtime_removes_finished_progress(fake_data: Path):
    """已终态（done/error 事件）的 progress 文件应立即清理，不等待超期。"""
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    finished = runtime / "progress_finished.jsonl"
    running = runtime / "progress_running.jsonl"
    finished.write_text(
        json.dumps({"type": "state", "state": "EXECUTE_CODE", "status": "success"}) + "\n"
        + json.dumps({"type": "done", "result": {"state": "COMPLETED"}}) + "\n",
        encoding="utf-8")
    running.write_text(
        json.dumps({"type": "state", "state": "EXECUTE_CODE", "status": "running"}) + "\n",
        encoding="utf-8")
    expected = finished.stat().st_size
    removed, freed = cleanup_runtime(keep_days=7)
    assert removed == 1
    assert freed == expected
    assert not finished.exists()
    assert running.exists()


def test_cleanup_runtime_removes_error_progress(fake_data: Path):
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    err = runtime / "progress_err.jsonl"
    err.write_text(json.dumps({"type": "error", "error": "boom"}) + "\n",
                   encoding="utf-8")
    removed, freed = cleanup_runtime()
    assert removed == 1
    assert freed > 0
    assert not err.exists()


# ---------- delete_session / clear_sessions ----------

def test_delete_session_removes_related_files(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid, "线性回归复现")
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl", [{"type": "log"}])
    report = fake_data / "reports" / f"线性回归复现_{sid}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# 报告", encoding="utf-8")
    # runtime progress 通过内容时间戳关联
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    progress = runtime / "progress_1234567890123.jsonl"
    progress.write_text(json.dumps({"type": "log", "log": {
        "timestamp": "2026-09-10T10:00:00"}}) + "\n",
        encoding="utf-8")

    removed, freed = delete_session(sid)
    assert removed == 4  # ledger + log + report + progress
    assert freed > 0
    # 全部删除
    assert not (fake_data / "experiment_ledger" / f"ledger_{sid}.jsonl").exists()
    assert not (fake_data / "logs" / f"session_{sid}.jsonl").exists()
    assert not report.exists()
    assert not progress.exists()
    # 其他会话不受影响
    other = fake_data / "reports" / "其他_{20260911_000000}.md"
    other.write_text("x", encoding="utf-8")
    assert other.exists()


def test_delete_session_unknown_returns_zero(fake_data: Path):
    assert delete_session("20260101_000000") == (0, 0)


def test_delete_session_skip_other_sessions_progress(fake_data: Path):
    sid = "20260910_100000"
    other_sid = "20260911_000000"
    _mk_ledger(fake_data, sid)
    _mk_ledger(fake_data, other_sid)
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    mine = runtime / "progress_mine.jsonl"
    theirs = runtime / "progress_theirs.jsonl"
    mine.write_text(json.dumps({"type": "log", "log": {
        "timestamp": "2026-09-10T10:00:00"}}) + "\n", encoding="utf-8")
    theirs.write_text(json.dumps({"type": "log", "log": {
        "timestamp": "2026-09-11T00:00:00"}}) + "\n", encoding="utf-8")
    removed, _ = delete_session(sid)
    assert removed == 2  # ledger + mine progress
    assert mine.exists() is False
    assert theirs.exists()  # 其他会话的 progress 保留


def test_clear_sessions_removes_everything(fake_data: Path):
    _mk_ledger(fake_data, "20260910_100000", "A")
    _mk_ledger(fake_data, "20260911_000000", "B")
    _write_jsonl(fake_data / "logs" / "session_20260910_100000.jsonl", [{"a": 1}])
    _write_jsonl(fake_data / "logs" / "session_20260911_000000.jsonl", [{"b": 2}])
    reports = fake_data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "A_20260910_100000.md").write_text("r1", encoding="utf-8")
    (reports / "B_20260911_000000.md").write_text("r2", encoding="utf-8")
    # 保留一个仍在运行的 progress（无终态、非超期）
    runtime = fake_data / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "progress_running.jsonl").write_text(
        json.dumps({"type": "state", "status": "running"}) + "\n", encoding="utf-8")
    # 一个已终态 progress 应被清掉
    (runtime / "progress_done.jsonl").write_text(
        json.dumps({"type": "done", "result": {}}) + "\n", encoding="utf-8")

    removed, freed = clear_sessions()
    assert removed == 7  # 2 ledger + 2 log + 2 report + 1 finished progress
    assert freed > 0
    assert list_sessions() == []
    # 仍在运行的 progress 保留
    assert (runtime / "progress_running.jsonl").exists()
    assert not (runtime / "progress_done.jsonl").exists()


# ---------- 终态判断辅助 ----------

def test_is_finished_progress(fake_data: Path):
    d = fake_data / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    done = d / "a.jsonl"
    done.write_text(json.dumps({"type": "done", "result": {}}) + "\n", encoding="utf-8")
    assert _is_finished_progress(done)
    err = d / "b.jsonl"
    err.write_text(json.dumps({"type": "error", "error": "x"}) + "\n", encoding="utf-8")
    assert _is_finished_progress(err)
    running = d / "c.jsonl"
    running.write_text(json.dumps({"type": "state", "status": "running"}) + "\n",
                       encoding="utf-8")
    assert not _is_finished_progress(running)
    broken = d / "d.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    assert not _is_finished_progress(broken)


def test_session_id_from_progress_strict(fake_data: Path):
    d = fake_data / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "p.jsonl"
    p.write_text(json.dumps({"type": "log", "log": {
        "timestamp": "2026-09-10T10:00:00"}}) + "\n", encoding="utf-8")
    assert _session_id_from_progress(p) == "20260910_100000"
    # 无 log 时间戳时返回 None（不 mtime 回退）
    empty = d / "empty.jsonl"
    empty.write_text(json.dumps({"type": "state", "status": "running"}) + "\n",
                     encoding="utf-8")
    assert _session_id_from_progress(empty) is None


# ---------- get_session_detail ----------

def test_get_session_detail_returns_none_for_unknown(fake_data: Path):
    assert get_session_detail("20260101_000000") is None


def test_get_session_detail_reads_ledger_and_logs(fake_data: Path):
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl", [{"type": "log", "msg": "hi"}])
    detail = get_session_detail(sid)
    assert detail is not None
    assert len(detail["ledger"]) == 1
    assert len(detail["logs"]) == 1
    assert detail["logs"][0]["msg"] == "hi"


# ---------- format_size ----------

@pytest.mark.parametrize("bytes_,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (1024, "1.00 KB"),
    (2 * 1024 * 1024, "2.00 MB"),
    (3 * 1024 * 1024 * 1024, "3.00 GB"),
])
def test_format_size(bytes_: int, expected: str):
    assert format_size(bytes_) == expected


# ---------- 解析辅助函数 ----------

def test_parse_session_id_variants():
    from frontend.history_manager import _parse_session_id
    assert _parse_session_id("ledger_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("session_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("progress_20260910_100000.jsonl") == "20260910_100000"
    assert _parse_session_id("random.txt") is None