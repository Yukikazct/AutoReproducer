"""ExperienceStore 经验库测试（融合 ScholarAgent optimizer/experience_store）。

运行: python -m pytest tests/test_experience_store.py -v
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experience.experience_store import (
    DEFAULT_EXPERIENCE_PATH,
    MAX_RECORDS,
    ExperienceStore,
)


@pytest.fixture()
def store(tmp_path: Path) -> ExperienceStore:
    return ExperienceStore(tmp_path / "exp.jsonl")


def _record(**over):
    base = {
        "repo": "repo-A", "direction": "batch_size",
        "params": {"bs": 128}, "metric": "acc",
        "score": 0.91, "direction_mode": "maximize",
        "validated": True, "context": {"trial": 1},
    }
    base.update(over)
    return base


# ---------------- 基础持久化 ----------------

def test_append_and_all_roundtrip(store: ExperienceStore):
    store.append(_record())
    records = store.all(repo="repo-A")
    assert len(records) == 1
    r = records[0]
    assert r["repo"] == "repo-A"
    assert r["direction"] == "batch_size"
    assert r["score"] == 0.91 and r["metric"] == "acc"
    assert r["validated"] is True


def test_append_whitelist_fields(store: ExperienceStore):
    """脏键不落盘，只保留白名单字段。"""
    store.append(_record(evil_key="x", params="not-a-dict"))
    r = store.all()[0]
    assert "evil_key" not in r
    assert r["params"] == {}                 # 非 dict 归一为空


def test_append_auto_timestamp(store: ExperienceStore):
    store.append(_record(timestamp=""))
    r = store.all()[0]
    assert r["timestamp"]                    # 自动补时间戳


def test_corrupt_line_skipped(store: ExperienceStore):
    store.append(_record())
    store.path.open("a", encoding="utf-8").write("{\"broken\":\n")
    store.append(_record(score=0.5))
    assert len(store.all()) == 2             # 半行写入被跳过


def test_count_and_clear(store: ExperienceStore):
    store.append(_record())
    store.append(_record(score=0.6))
    assert store.count() == 2
    store.clear()
    assert store.count() == 0


# ---------------- validated 过滤 ----------------

def test_validated_filter(store: ExperienceStore):
    store.append(_record(validated=True))
    store.append(_record(validated=False, score=0.99))
    assert len(store.all()) == 1             # 默认只要 validated
    assert len(store.all(validated_only=False)) == 2


def test_repo_filter(store: ExperienceStore):
    store.append(_record(repo="A"))
    store.append(_record(repo="B"))
    assert len(store.all(repo="A")) == 1
    assert len(store.all(repo="nope")) == 0


# ---------------- best / summarize ----------------

def test_best_maximize(store: ExperienceStore):
    store.append(_record(score=0.80))
    store.append(_record(score=0.95))
    store.append(_record(score=0.90))
    best = store.best("repo-A")
    assert best is not None and best["score"] == 0.95


def test_best_minimize(store: ExperienceStore):
    store.append(_record(score=0.8, direction_mode="minimize"))
    store.append(_record(score=0.3, direction_mode="minimize"))
    store.append(_record(score=0.6, direction_mode="minimize"))
    best = store.best("repo-A", direction_mode="minimize")
    assert best is not None and best["score"] == 0.3


def test_best_only_validated(store: ExperienceStore):
    store.append(_record(validated=False, score=1.0))
    assert store.best("repo-A") is None     # 无效记录不算经验


def test_best_ignores_non_numeric_score(store: ExperienceStore):
    store.append(_record(score="n/a"))
    assert store.best("repo-A") is None


def test_summarize_by_direction(store: ExperienceStore):
    store.append(_record(direction="bs", score=0.8))
    store.append(_record(direction="bs", score=0.9))
    store.append(_record(direction="lr", score=0.6))
    store.append(_record(direction="bs", valid=True, score="bad"))  # 非数值剔除
    s = store.summarize(repo="repo-A")
    assert s["record_count"] == 4
    d = s["directions"]["bs"]
    assert d["count"] == 3 and d["best"] == 0.9
    assert abs(d["mean"] - 0.85) < 1e-9
    assert s["directions"]["lr"]["best"] == 0.6


# ---------------- 上限截断 ----------------

def test_truncate_keeps_latest(store: ExperienceStore):
    store.max_records = 5
    for i in range(8):
        store.append(_record(score=i / 10))
    assert store.count() == 5
    scores = sorted(r["score"] for r in store.all())
    assert scores == [0.3, 0.4, 0.5, 0.6, 0.7]


def test_default_max_records_positive():
    assert MAX_RECORDS >= 1000


def test_experience_default_path_under_data_root():
    assert str(DEFAULT_EXPERIENCE_PATH).endswith(
        str(Path("experience") / "experience.jsonl"))


# ---------------- 线程安全 ----------------

def test_concurrent_append_no_lost_lines(store: ExperienceStore):
    def worker(idx: int):
        for _ in range(20):
            store.append(_record(repo="repo-A", score=idx / 100))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 80
    assert len(store.all(repo="repo-A")) == 80