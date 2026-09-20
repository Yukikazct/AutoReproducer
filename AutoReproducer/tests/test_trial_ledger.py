"""TrialLedger 试验账本测试（融合 ScholarAgent research/ledger.py）。

运行: python -m pytest tests/test_trial_ledger.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experience.trial_ledger import (
    DEFAULT_LEDGER_PATH,
    LEDGER_VERSION,
    TrialLedger,
)


@pytest.fixture()
def ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / "trials.jsonl")


def _ev(score=None, valid=True, **over):
    ev = {"valid": valid, "score": score, "metric": "acc", "stdout_tail": "..."}
    ev.update(over)
    return ev


def _candidate(**over):
    c = {"arm": "loss_fn", "params": {"k": 1}}
    c.update(over)
    return c


def test_record_keep_roundtrip(ledger: TrialLedger):
    rec = ledger.record_trial(
        repo="pinrepo", candidate=_candidate(), evaluation=_ev(score=0.95),
        kept=True, reason="mse 下降 5%", restored=True,
        applied_files=["optimized_patches/loss_fn.py"])
    assert rec["kept"] is True
    assert rec["version"] == LEDGER_VERSION
    assert rec["trial_id"] and rec["timestamp"]

    rows = ledger.all(repo="pinrepo")
    assert len(rows) == 1
    r = rows[0]
    assert r["candidate"]["arm"] == "loss_fn"
    assert r["evaluation"]["valid"] is True
    assert r["restored"] is True
    assert r["applied_files"] == ["optimized_patches/loss_fn.py"]


def test_record_reject(ledger: TrialLedger):
    rec = ledger.record_trial(
        repo="R", candidate=_candidate(), evaluation=_ev(score=None, valid=False),
        kept=False, reason="运行崩溃", rejected_files=["run.py"])
    assert rec["kept"] is False
    assert ledger.rejected() and not ledger.kept()


def test_kept_and_rejected_filters(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.9),
                        kept=True)
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.2),
                        kept=False)
    kept = ledger.kept()
    rejected = ledger.rejected()
    assert len(kept) == 1 and kept[0]["kept"] is True
    assert len(rejected) == 1 and rejected[0]["kept"] is False


def test_repo_filter(ledger: TrialLedger):
    ledger.record_trial(repo="A", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=True)
    ledger.record_trial(repo="B", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=True)
    assert len(ledger.all(repo="A")) == 1
    assert len(ledger.all(repo="C")) == 0


def test_applied_files_normalization(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=True, applied_files="single/path.py")   # 字符串归一为列表
    row = ledger.all()[0]
    assert row["applied_files"] == ["single/path.py"]

    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=False, rejected_files=None)
    assert ledger.all()[-1]["rejected_files"] == []


def test_valid_keep_trials_gate(ledger: TrialLedger):
    """只有 evaluation.valid=True 且 kept=True 才可播种经验。"""
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.9),
                        kept=True)                       # valid keep ✓
    ledger.record_trial(repo="R", candidate=_candidate(),
                        evaluation=_ev(0.9, valid=False),
                        kept=True, reason="无真实执行")   # kept 但 valid=False ✗
    ledger.record_trial(repo="R", candidate=_candidate(arm="fake"),
                        evaluation=_ev(0.99, valid=False),
                        kept=True)                       # kept 但 valid=False ✗
    rows = ledger.valid_keep_trials(repo="R")
    assert len(rows) == 1 and rows[0]["candidate"]["arm"] == "loss_fn"


def test_best_picks_max_valid_keep(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(arm="a"),
                        evaluation=_ev(0.6), kept=True)
    ledger.record_trial(repo="R", candidate=_candidate(arm="b"),
                        evaluation=_ev(0.9), kept=True)
    ledger.record_trial(repo="R", candidate=_candidate(arm="c"),
                        evaluation=_ev(score=None, valid=False), kept=True)
    best = ledger.best("R")
    assert best is not None and best["candidate"]["arm"] == "b"


def test_best_none_when_no_valid_keep(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(),
                        evaluation=_ev(0.9), kept=False)
    assert ledger.best("R") is None


def test_summary_counts(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.9),
                        kept=True)
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.1),
                        kept=False)
    summary = ledger.summary(repo="R")
    assert summary["trials_total"] == 2
    assert summary["kept"] == 1 and summary["rejected"] == 1
    assert summary["valid_keep"] == 1


def test_corrupt_line_skipped(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=True)
    ledger.path.open("a", encoding="utf-8").write('{"broken"\n')
    assert len(ledger.all()) == 1


def test_truncate_keeps_latest(ledger: TrialLedger):
    ledger.max_records = 4
    for i in range(6):
        ledger.record_trial(repo="R", candidate=_candidate(arm=str(i)),
                            evaluation=_ev(i), kept=True)
    trials = ledger.all()
    assert len(trials) == 4
    assert [t["candidate"]["arm"] for t in trials] == ["2", "3", "4", "5"]


def test_count_and_clear(ledger: TrialLedger):
    ledger.record_trial(repo="R", candidate=_candidate(), evaluation=_ev(0.5),
                        kept=True)
    assert ledger.count() == 1
    ledger.clear()
    assert ledger.count() == 0


def test_custom_trial_id(ledger: TrialLedger):
    rec = ledger.record_trial(repo="R", candidate=_candidate(),
                              evaluation=_ev(0.5), kept=True, trial_id="T-9")
    assert rec["trial_id"] == "T-9"
    assert ledger.all()[0]["trial_id"] == "T-9"


def test_default_ledger_path_under_experience_dir():
    assert str(DEFAULT_LEDGER_PATH).endswith(
        str(Path("experience") / "trials.jsonl"))