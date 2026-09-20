"""Claim / Criterion / Evidence graph with hash-anchored evidence.

移植自 ScholarAgent backend/app/evidence/graph.py (MIT License)，键名对齐
AutoReproducer 的产物管线（run_metrics / comparison_report / trial_ledger 等）。

裁决策略（确定性，绝不委托给 LLM）：
- 一个 criterion 只有在它引用的 artifacts 中至少包含一条 *authentic* 执行证据
  （真实完成的 run + metrics，每条以内容 SHA-256 锚定）时，才可能被判为
  verified / smoke_verified / partially_reproduced / contradicted；
- ``verified`` 还要求完整复现模式 full；smoke 模式的天花板是
  ``smoke_verified``，模式未知时天花板是 ``partially_reproduced``；
- 完全没有执行证据 -> ``unverifiable``；缺 checkpoint/dataset/GPU 等资产
  -> ``blocked_by_missing_asset``。

全部为纯函数，不依赖 LLM，可在 pipeline 任意环节安全调用。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GRAPH_VERSION = "autorepro.claim.evidence/v1"

# 可充当"执行证据"的产物键（authentic 判定见 _is_authentic）
EXECUTION_EVIDENCE_KEYS = {"run_metrics", "rerun_metrics", "trial_ledger"}

# 建图时考虑的完整证据清单（AutoReproducer 产物管线键名）
EVIDENCE_KEYS = (
    "run_metrics",          # 复现运行真实指标（completed + metrics）
    "rerun_metrics",        # 优化/验证重跑指标
    "trial_ledger",         # TrialLedger JSON 串（真实执行的 Keep/Reject 记录）
    "comparison_report",    # 论文指标 vs 复现指标对照
    "result_plot",          # 结果图（支撑证据）
    "repo_manifest",        # 仓库清单
    "reproduction_report",  # 主复现报告 markdown
    "dependency_install_report",  # 依赖安装报告
    "blocked_resources_report",   # 缺失资产报告
)

VALID_STATUSES = {
    "verified",
    "smoke_verified",
    "partially_reproduced",
    "contradicted",
    "unverifiable",
    "blocked_by_missing_asset",
}


# ---------------------------------------------------------------------------
# 冻结 rubric 完整性
# ---------------------------------------------------------------------------

def canonical_rubric_sha256(paper_title: str, version: str,
                            claims: list[dict[str, Any]]) -> str:
    """对 rubric 的规范 JSON 序列化求 SHA-256（键排序，保证可复算）。"""
    payload = json.dumps(
        {"version": version, "paper_title": paper_title, "claims": claims},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_rubric(rubric: dict[str, Any]) -> tuple[bool, str]:
    """复算冻结 rubric 哈希；无哈希或执行后改动都判定失败。"""
    declared = str(rubric.get("rubric_sha256", ""))
    if not declared:
        return False, "rubric is not frozen (missing rubric_sha256)"
    recomputed = canonical_rubric_sha256(
        str(rubric.get("paper_title", "")),
        str(rubric.get("version", "")),
        list(rubric.get("claims", [])),
    )
    if recomputed != declared:
        return False, (f"rubric hash mismatch: declared {declared[:16]}..., "
                       f"recomputed {recomputed[:16]}...")
    if not rubric.get("frozen_before_execution"):
        return False, "rubric was not frozen before execution"
    return True, ""


# ---------------------------------------------------------------------------
# 证据注册表（结构性：从真实 artifacts 构建，而非 LLM 声称）
# ---------------------------------------------------------------------------

def _try_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _is_authentic(key: str, content: str) -> bool:
    """被引用的 artifact 只有在记录了"真实完成的 run + 可测输出"时才算执行证据。

    - run_metrics / rerun_metrics / trial_ledger：需 JSON 内 status=completed
      且 metrics 非空（trial_ledger 允许 kept 记录聚合）；
    - 其余 artifact：真实存在（非空）即算支撑证据，但不代表执行结果本身。
    """
    if key in EXECUTION_EVIDENCE_KEYS:
        payload = _try_json(content)
        if isinstance(payload, str):
            payload = _try_json(payload)
        if isinstance(payload, dict):
            completed = str(payload.get("status", "")) == "completed"
            metrics = payload.get("metrics")
            return completed and isinstance(metrics, dict) and len(metrics) > 0
        # trial_ledger 可能是 json 数组（多个 trial 记录）
        if isinstance(payload, list) and key == "trial_ledger":
            return any(isinstance(item, dict) and item.get("kept")
                       for item in payload)
        return False
    # comparison_report / result_plot / repo 上下文类产物：
    # 真实 artifacts，但不是独立执行结果。
    return bool(content.strip())


def evidence_mode(inputs: dict[str, Any]) -> str:
    """复现模式：full | smoke | unknown（默认 unknown 限制裁决天花板）。"""
    raw = str(inputs.get("reproduction_report", "") or "")
    payload = _try_json(raw)
    if isinstance(payload, dict):
        mode = str(payload.get("effective_mode", "")).lower()
        if mode in {"full", "smoke"}:
            return mode
    return "unknown"


def build_evidence_registry(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """每个在场 artifact 生成一个证据节点：内容哈希 + authentic 判定。"""
    registry: list[dict[str, Any]] = []
    for key in EVIDENCE_KEYS:
        content = str(inputs.get(key, "") or "")
        if not content.strip():
            continue
        registry.append(
            {
                "key": key,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "bytes": len(content.encode("utf-8")),
                "authentic": _is_authentic(key, content),
                "execution_evidence": key in EXECUTION_EVIDENCE_KEYS,
            }
        )
    return registry


def blocked_resources(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """缺失资产报告（checkpoint/dataset/GPU 等），有则列出。"""
    payload = _try_json(str(inputs.get("blocked_resources_report", "") or ""))
    if isinstance(payload, dict):
        missing = payload.get("missing_resources", [])
        if isinstance(missing, list):
            return [item for item in missing if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


# ---------------------------------------------------------------------------
# 确定性裁决归一化（门控：无真实执行证据不得判 verified）
# ---------------------------------------------------------------------------

def normalize_findings(
    findings: list[dict[str, Any]],
    evidence_registry: list[dict[str, Any]],
    mode: str,
    blocked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按证据与模式对每个 finding 做确定性裁决（不调 LLM）。"""
    registry_by_key = {entry["key"]: entry for entry in evidence_registry}
    has_execution_evidence = any(
        entry["execution_evidence"] and entry["authentic"]
        for entry in evidence_registry
    )
    normalized: list[dict[str, Any]] = []
    for finding in findings:
        status = str(finding.get("status", "unverifiable"))
        cited = [key for key in finding.get("evidence_keys", [])
                 if key in registry_by_key]
        cited_authentic_execution = any(
            registry_by_key[key]["execution_evidence"]
            and registry_by_key[key]["authentic"]
            for key in cited
        )

        if status not in VALID_STATUSES:
            status = "unverifiable"
        # 门控 1：正性结论必须有真实执行证据
        if status in {"verified", "partially_reproduced", "contradicted"}:
            if not cited_authentic_execution:
                # "进程正常退出本身不足以验证一个 claim"
                status = "unverifiable"
        # 门控 2：模式天花板
        if status == "verified":
            if mode == "smoke":
                status = "smoke_verified"
            elif mode != "full":
                status = "partially_reproduced"
        # 门控 3：无执行证据 + 缺资产 -> 明确归因
        if status == "unverifiable" and blocked and not has_execution_evidence:
            status = "blocked_by_missing_asset"

        normalized.append(
            {
                **finding,
                "status": status,
                "evidence_keys": sorted(set(cited)),
                "mode_ceiling_applied": True,
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# 图组装
# ---------------------------------------------------------------------------

def _criterion_verdict_summary(verdicts: list[str]) -> str:
    """claim 的 verdict = 其 criteria verdicts 的保守聚合。"""
    if not verdicts:
        return "unverifiable"
    if any(v == "contradicted" for v in verdicts):
        return "contradicted"
    if all(v == "blocked_by_missing_asset" for v in verdicts):
        return "blocked_by_missing_asset"
    if all(v == "verified" for v in verdicts):
        return "verified"
    if all(v in {"verified", "smoke_verified"} for v in verdicts):
        return "smoke_verified"
    if any(v in {"verified", "smoke_verified", "partially_reproduced"}
           for v in verdicts):
        return "partially_reproduced"
    return "unverifiable"


def build_graph(
    rubric: dict[str, Any],
    findings: list[dict[str, Any]],
    evidence_registry: list[dict[str, Any]],
    mode: str,
    blocked: list[dict[str, Any]],
) -> dict[str, Any]:
    """组装 Claim/Criterion/Evidence 图，含全图指纹 graph_sha256。"""
    finding_by_criterion = {
        str(f.get("criterion_id")): f
        for f in findings if f.get("criterion_id")
    }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for entry in evidence_registry:
        nodes.append({
            "id": f"evidence:{entry['key']}",
            "type": "evidence",
            "artifact_key": entry["key"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
            "authentic": entry["authentic"],
            "execution_evidence": entry["execution_evidence"],
        })

    for claim in rubric.get("claims", []):
        claim_id = str(claim.get("claim_id", ""))
        criterion_verdicts: list[str] = []
        nodes.append({
            "id": claim_id,
            "type": "claim",
            "title": claim.get("title", ""),
            "statement": claim.get("statement", ""),
            "importance": claim.get("importance", 0.5),
        })
        for criterion in claim.get("criteria", []):
            criterion_id = str(criterion.get("criterion_id", ""))
            finding = finding_by_criterion.get(criterion_id, {})
            verdict = str(finding.get("status", "unverifiable"))
            criterion_verdicts.append(verdict)
            nodes.append({
                "id": criterion_id,
                "type": "criterion",
                "claim": claim_id,
                "statement": criterion.get("statement",
                                           criterion.get("description", "")),
                "verdict": verdict,
                "confidence": finding.get("confidence", 0.0),
                "observed_value": finding.get("observed_value", ""),
                "reason": finding.get("reason", ""),
            })
            edges.append({"from": claim_id, "to": criterion_id,
                          "type": "has_criterion"})
            for key in finding.get("evidence_keys", []):
                edges.append({"from": criterion_id,
                              "to": f"evidence:{key}",
                              "type": "supported_by"})

    # claim verdict 由其 criteria 聚合
    for node in nodes:
        if node["type"] != "claim":
            continue
        children = [n for n in nodes
                    if n["type"] == "criterion" and n.get("claim") == node["id"]]
        node["verdict"] = _criterion_verdict_summary(
            [c["verdict"] for c in children])

    counts: dict[str, int] = {}
    for node in nodes:
        if node["type"] == "criterion":
            counts[node["verdict"]] = counts.get(node["verdict"], 0) + 1

    graph = {
        "version": GRAPH_VERSION,
        "mode": mode,
        "rubric_sha256": rubric.get("rubric_sha256", ""),
        "blocked_resources": blocked,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "claims": sum(1 for n in nodes if n["type"] == "claim"),
            "criteria": sum(1 for n in nodes if n["type"] == "criterion"),
            "evidence_artifacts": sum(1 for n in nodes if n["type"] == "evidence"),
            "criterion_verdicts": counts,
        },
    }
    graph["graph_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in graph.items() if k != "graph_sha256"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return graph