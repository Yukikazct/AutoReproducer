"""证据链模块测试：证据注册表 / 确定性裁决门控 / Claim-Criterion-Evidence 图。

移植自 ScholarAgent backend/app/evidence/graph.py (MIT)，键名对齐 AutoReproducer。

运行: python -m pytest tests/test_evidence_graph.py -v
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evidence.graph import (
    EXECUTION_EVIDENCE_KEYS,
    build_evidence_registry,
    build_graph,
    canonical_rubric_sha256,
    evidence_mode,
    normalize_findings,
    verify_frozen_rubric,
)


# ---------------------------------------------------------------------------
# rubric 冻结
# ---------------------------------------------------------------------------

def _rubric(claims=None, **over):
    rb = {
        "paper_title": "PINN 复现",
        "version": "v1",
        "claims": claims or [
            {"claim_id": "c1", "title": "精度达标",
             "statement": "复现精度与论文接近",
             "criteria": [{"criterion_id": "k1", "statement": "测试集准确率"}]},
        ],
    }
    rb.update(over)
    return rb


def test_canonical_rubric_sha256_stable_and_sorted():
    a = canonical_rubric_sha256("t", "v1", [{"x": 1, "y": 2}])
    b = canonical_rubric_sha256("t", "v1", [{"y": 2, "x": 1}])   # 键序无关
    assert a == b and len(a) == 64


def test_verify_frozen_rubric():
    rb = _rubric(frozen_before_execution=True)
    rb["rubric_sha256"] = canonical_rubric_sha256(
        rb["paper_title"], rb["version"], rb["claims"])
    ok, _ = verify_frozen_rubric(rb)
    assert ok

    # 执行后篡改 claim -> 哈希不匹配
    rb2 = dict(rb)
    rb2["claims"] = [{"claim_id": "c1", "title": "篡改", "criteria": []}]
    ok2, why = verify_frozen_rubric(rb2)
    assert not ok2 and "mismatch" in why


def test_verify_frozen_rubric_requires_freeze_flag():
    rb = _rubric(claims=[])
    rb["rubric_sha256"] = canonical_rubric_sha256(
        rb["paper_title"], rb["version"], rb["claims"])
    ok, why = verify_frozen_rubric(rb)
    assert not ok and "frozen before" in why


# ---------------------------------------------------------------------------
# 证据注册表
# ---------------------------------------------------------------------------

def test_build_evidence_registry_hashes_and_flags():
    inputs = {
        "run_metrics": json.dumps({"status": "completed",
                                   "metrics": {"acc": 0.92}}),
        "comparison_report": "# 对照报告\n指标对齐。",
        "result_plot": "",
    }
    reg = build_evidence_registry(inputs)
    keys = {e["key"] for e in reg}
    assert keys == {"run_metrics", "comparison_report"}

    by_key = {e["key"]: e for e in reg}
    rm = by_key["run_metrics"]
    assert rm["authentic"] is True
    assert rm["execution_evidence"] is True
    assert rm["sha256"] == hashlib.sha256(
        inputs["run_metrics"].encode()).hexdigest()

    comp = by_key["comparison_report"]
    assert comp["authentic"] is True          # 支撑证据非空即真实
    assert comp["execution_evidence"] is False


def test_authentic_requires_completed_and_metrics():
    # 未完成 / 无指标 -> 不 authentic
    bad = build_evidence_registry({"run_metrics": json.dumps(
        {"status": "failed", "metrics": {}})})
    assert bad[0]["authentic"] is False

    # 非 JSON -> 不 authentic
    bad2 = build_evidence_registry({"run_metrics": "oops not json"})
    assert bad2[0]["authentic"] is False


def test_trial_ledger_list_counts_as_execution_evidence():
    trial_json = json.dumps([
        {"kept": True, "evaluation": {"valid": True, "score": 0.5}},
        {"kept": False},
    ])
    reg = build_evidence_registry({"trial_ledger": trial_json})
    assert reg[0]["authentic"] is True
    assert reg[0]["execution_evidence"] is True


def test_evidence_mode_detection():
    assert evidence_mode({"reproduction_report": json.dumps(
        {"effective_mode": "full"})}) == "full"
    assert evidence_mode({"reproduction_report": json.dumps(
        {"effective_mode": "smoke"})}) == "smoke"
    assert evidence_mode({"run_metrics": "{}"}) == "unknown"


# ---------------------------------------------------------------------------
# 确定性裁决门控
# ---------------------------------------------------------------------------

def _reg_ok(key="run_metrics"):
    return build_evidence_registry({key: json.dumps(
        {"status": "completed", "metrics": {"acc": 0.9}})})


def test_no_execution_evidence_degrades_verdict():
    reg = build_evidence_registry({"comparison_report": "# 仅对照"})
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "verified",
          "evidence_keys": ["comparison_report"]}],
        reg, mode="full", blocked=[])
    assert out[0]["status"] == "unverifiable"   # 无真实执行证据 -> 降级


def test_smoke_mode_caps_verified():
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "verified",
          "evidence_keys": ["run_metrics"]}],
        _reg_ok(), mode="smoke", blocked=[])
    assert out[0]["status"] == "smoke_verified"


def test_unknown_mode_caps_verified_to_partial():
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "verified",
          "evidence_keys": ["run_metrics"]}],
        _reg_ok(), mode="unknown", blocked=[])
    assert out[0]["status"] == "partially_reproduced"


def test_full_mode_allows_verified_with_real_evidence():
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "verified",
          "evidence_keys": ["run_metrics"]}],
        _reg_ok(), mode="full", blocked=[])
    assert out[0]["status"] == "verified"


def test_invalid_status_normalized():
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "magic_verified",
          "evidence_keys": ["run_metrics"]}],
        _reg_ok(), mode="full", blocked=[])
    assert out[0]["status"] == "unverifiable"


def test_blocked_resources_attribution():
    blocked = [{"resource": "checkpoint", "kind": "weights"}]
    reg = build_evidence_registry({"comparison_report": "# x"})
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "unverifiable",
          "evidence_keys": []}],
        reg, mode="full", blocked=blocked)
    assert out[0]["status"] == "blocked_by_missing_asset"


def test_blocked_not_applied_when_execution_evidence_exists():
    blocked = [{"resource": "gpu", "kind": "hardware"}]
    out = normalize_findings(
        [{"criterion_id": "k1", "status": "unverifiable",
          "evidence_keys": ["run_metrics"]}],
        _reg_ok(), mode="full", blocked=blocked)
    assert out[0]["status"] == "unverifiable"  # 有执行证据但不支持 claim


# ---------------------------------------------------------------------------
# 图组装
# ---------------------------------------------------------------------------

def test_build_graph_full_flow():
    rubric = _rubric(frozen_before_execution=True)
    rubric["rubric_sha256"] = canonical_rubric_sha256(
        rubric["paper_title"], rubric["version"], rubric["claims"])
    findings = [{"criterion_id": "k1", "status": "verified",
                 "evidence_keys": ["run_metrics"], "confidence": 0.9}]
    reg = _reg_ok()
    graph = build_graph(rubric, findings, reg, mode="full", blocked=[])

    assert graph["version"].startswith("autorepro.claim.evidence")
    assert graph["rubric_sha256"] == rubric["rubric_sha256"]
    assert len(graph["graph_sha256"]) == 64

    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["c1"]["type"] == "claim"
    assert by_id["c1"]["verdict"] == "verified"      # criteria 全部 verified
    assert by_id["k1"]["verdict"] == "verified"
    assert by_id["evidence:run_metrics"]["authentic"] is True

    edge_types = {e["type"] for e in graph["edges"]}
    assert edge_types == {"has_criterion", "supported_by"}
    assert graph["summary"]["claims"] == 1
    assert graph["summary"]["criteria"] == 1
    assert graph["summary"]["evidence_artifacts"] == 1
    assert graph["summary"]["criterion_verdicts"] == {"verified": 1}


def test_graph_sha256_deterministic_and_sensitive():
    rubric = _rubric(frozen_before_execution=True)
    rubric["rubric_sha256"] = canonical_rubric_sha256(
        rubric["paper_title"], rubric["version"], rubric["claims"])
    kwargs = dict(rubric=rubric, findings=[
        {"criterion_id": "k1", "status": "verified",
         "evidence_keys": ["run_metrics"]}],
        evidence_registry=_reg_ok(), mode="full", blocked=[])
    g1 = build_graph(**kwargs)
    g2 = build_graph(**kwargs)
    assert g1["graph_sha256"] == g2["graph_sha256"]

    kwargs2 = dict(kwargs, mode="smoke")
    g3 = build_graph(**kwargs2)
    assert g1["graph_sha256"] != g3["graph_sha256"]   # 模式变化反映到指纹


def test_execution_evidence_keys_include_trial_ledger():
    assert "trial_ledger" in EXECUTION_EVIDENCE_KEYS