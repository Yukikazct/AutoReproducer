"""历史记录管理模块单元测试：list_sessions / get_storage_stats / cleanup_runtime / get_session_detail / format_size / 依赖缓存管理。

通过 monkeypatch 将数据目录指向临时目录，不触碰真实 data/。
"""
import json
import os
import time
from datetime import datetime, timedelta
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
    delete_sessions,
    clear_sessions,
    _is_finished_progress,
    _session_id_from_progress,
    cleanup_deps_cache,
    delete_deps_cache,
    list_deps_cache,
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


# ---------- delete_sessions（批量删除） ----------

def _mk_report(data_dir: Path, sid: str, title: str) -> Path:
    reports = data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"{title}_{sid}.md"
    path.write_text("# 报告", encoding="utf-8")
    return path


def test_delete_sessions_removes_multiple(fake_data: Path):
    """一次删多个会话：各自的账本/日志/报告全部清理。"""
    sids = ["20260910_100000", "20260911_000000"]
    reports = [_mk_report(fake_data, sid, f"论文{i}")
               for i, sid in enumerate(sids)]
    for i, sid in enumerate(sids):
        _mk_ledger(fake_data, sid, f"论文{i}")
        _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl",
                     [{"type": "log"}])

    removed, freed = delete_sessions(sids)

    assert removed == 6      # 2 账本 + 2 日志 + 2 报告
    assert freed > 0
    for i, sid in enumerate(sids):
        assert not (fake_data / "experiment_ledger"
                    / f"ledger_{sid}.jsonl").exists()
        assert not (fake_data / "logs" / f"session_{sid}.jsonl").exists()
        assert not reports[i].exists()


def test_delete_sessions_skips_unrelated(fake_data: Path):
    """未传入的会话必须原样保留（批量删除不得误伤）。"""
    keep = "20260912_000000"
    for sid in ("20260910_100000", "20260911_000000"):
        _mk_ledger(fake_data, sid)
    _mk_ledger(fake_data, keep, "保留论文")
    _write_jsonl(fake_data / "logs" / f"session_{keep}.jsonl", [{"k": 1}])
    kept_report = _mk_report(fake_data, keep, "保留论文")

    delete_sessions(["20260910_100000", "20260911_000000"])

    assert (fake_data / "experiment_ledger" / f"ledger_{keep}.jsonl").exists()
    assert (fake_data / "logs" / f"session_{keep}.jsonl").exists()
    assert kept_report.exists()


def test_delete_sessions_empty_and_unknown(fake_data: Path):
    """空列表与未知会话 id 都返回 (0, 0)，不抛异常。"""
    assert delete_sessions([]) == (0, 0)
    assert delete_sessions(["20260101_000000"]) == (0, 0)
    assert delete_sessions(["20260101_000000", "20260102_000000"]) == (0, 0)


def test_delete_sessions_dedupes_input_ids(fake_data: Path):
    """重复的会话 id 只删一份，removed 不翻倍。"""
    sid = "20260910_100000"
    _mk_ledger(fake_data, sid)
    _write_jsonl(fake_data / "logs" / f"session_{sid}.jsonl", [{"a": 1}])

    removed, _ = delete_sessions([sid, sid])

    assert removed == 2      # ledger + log，而不是 4
    assert not (fake_data / "experiment_ledger" / f"ledger_{sid}.jsonl").exists()


def test_delete_sessions_matches_progress_by_content(fake_data: Path):
    """批量删除同样按内容时间戳关联 progress，不误删其他会话的。"""
    sid, other_sid = "20260910_100000", "20260911_000000"
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

    removed, _ = delete_sessions([sid])

    assert removed == 2      # ledger + mine progress
    assert mine.exists() is False
    assert theirs.exists()


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


# ---------- 依赖缓存管理（data/deps/） ----------

@pytest.fixture
def fake_deps(tmp_path: Path, monkeypatch) -> Path:
    """依赖缓存根指向 tmp_path/deps（覆盖 AUTOREPRO_DEPS_ROOT，双保险）。

    conftest 已把该环境变量指向一次性临时目录，这里再显式指到本用例的
    tmp_path，让每条断言都只针对自己造的数据。
    """
    root = tmp_path / "deps"
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(root))
    return root


def _mk_deps_dir(root: Path, name: str, *, packages=(), payload: int = 10,
                 meta=None, mtime: float = None) -> Path:
    """造一个隔离依赖目录：meta.json + <pkg>.dist-info/ + 若干字节。"""
    d = root / name
    (d / "somepkg").mkdir(parents=True, exist_ok=True)
    (d / "somepkg" / "__init__.py").write_text("x" * payload, encoding="utf-8")
    for pkg in packages:
        info = d / f"{pkg}-1.0.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text("Name: " + pkg, encoding="utf-8")
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def test_list_deps_cache_reads_meta_and_packages(fake_deps: Path):
    """条目要能看出「装的是什么、占多大、什么时候用的」。"""
    _mk_deps_dir(fake_deps, "abc123", packages=("numpy", "scipy"),
                 meta={"kind": "reqs", "installed_at": "2026-09-01T10:00:00",
                       "last_used": "2026-09-19T10:00:00",
                       "requirements": "numpy\nscipy"})

    items = list_deps_cache()

    assert len(items) == 1
    it = items[0]
    assert it["name"] == "abc123"
    assert it["kind"] == "reqs"
    assert it["packages"] == ["numpy", "scipy"]
    assert it["requirements"] == "numpy\nscipy"
    assert it["last_used"] == "2026-09-19T10:00:00"
    assert it["files"] == 4          # __init__ + 2 个 METADATA + meta.json
    assert it["bytes"] > 0


def test_list_deps_cache_legacy_dir_falls_back_to_mtime(fake_deps: Path):
    """改造前建的目录没有 meta.json：不能消失、也不能报错，
    类型标 legacy、最后使用回退到目录 mtime（否则永远清不掉）。"""
    _mk_deps_dir(fake_deps, "old_no_meta", packages=("numpy",), mtime=1000.0)

    items = list_deps_cache()

    assert len(items) == 1
    assert items[0]["kind"] == "legacy"
    assert items[0]["packages"] == ["numpy"]
    assert items[0]["last_used"].startswith("1970-01-01")   # mtime=1000s


def test_list_deps_cache_sorted_by_last_used_desc(fake_deps: Path):
    _mk_deps_dir(fake_deps, "hot", meta={"kind": "reqs",
                                         "last_used": "2026-09-19T10:00:00"})
    _mk_deps_dir(fake_deps, "cold", meta={"kind": "reqs",
                                          "last_used": "2026-01-01T10:00:00"})
    assert [i["name"] for i in list_deps_cache()] == ["hot", "cold"]


def test_list_deps_cache_missing_root_is_empty(fake_deps: Path):
    assert not fake_deps.exists()      # 前提：根目录还没建
    assert list_deps_cache() == []


def test_delete_deps_cache_removes_only_named(fake_deps: Path):
    _mk_deps_dir(fake_deps, "keep", packages=("numpy",))
    victim = _mk_deps_dir(fake_deps, "victim", packages=("numpy",))
    expected_bytes = sum(f.stat().st_size for f in victim.rglob("*") if f.is_file())

    removed, freed = delete_deps_cache(["victim"])

    assert removed == 1
    assert freed == expected_bytes
    assert not victim.exists()
    assert (fake_deps / "keep").is_dir()


def test_delete_deps_cache_ignores_unknown_names(fake_deps: Path):
    assert delete_deps_cache([]) == (0, 0)
    assert delete_deps_cache(["nope"]) == (0, 0)


def test_delete_deps_cache_rejects_path_escape(fake_deps: Path, tmp_path: Path):
    """只接受直接子目录名：路径穿越/嵌套路径一律拒绝，绝不越界删除。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important.txt").write_text("别删我", encoding="utf-8")
    inside = _mk_deps_dir(fake_deps, "inside")

    removed, _ = delete_deps_cache(
        ["../outside", str(outside), "inside/../inside", "", "."])

    assert removed == 0
    assert outside.is_dir() and (outside / "important.txt").exists()
    assert inside.is_dir()          # 嵌套路径同样被拒（不是「恰好删对了」）


def test_cleanup_deps_cache_keeps_recent(fake_deps: Path):
    """「保留最近 N 天」按 last_used 判断冷热，旧目录（无 meta）按 mtime。"""
    now = datetime.now()
    _mk_deps_dir(fake_deps, "cold_meta", meta={
        "kind": "reqs",
        "last_used": (now - timedelta(days=200)).isoformat(timespec="seconds")})
    _mk_deps_dir(fake_deps, "hot_meta", meta={
        "kind": "reqs",
        "last_used": (now - timedelta(days=1)).isoformat(timespec="seconds")})
    _mk_deps_dir(fake_deps, "cold_legacy", mtime=time.time() - 200 * 86400)

    removed, freed = cleanup_deps_cache(keep_days=30)

    assert removed == 2
    assert freed > 0
    assert sorted(i["name"] for i in list_deps_cache()) == ["hot_meta"]


def test_deps_meta_name_matches_executor():
    """`_DEPS_META_NAME` 是跨模块契约：执行器写入、前端读取。

    history_manager 刻意不 import src.*（保持纯 stdlib、前端加载更轻），
    所以用这条测试钉住两边的常量一致，而不是靠跨层 import。
    """
    import frontend.history_manager as hm
    import src.agents.code_executor as ce

    assert ce._DEPS_META_NAME == hm._DEPS_META_NAME


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