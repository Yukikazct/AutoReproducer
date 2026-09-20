"""ResearchSpec - 冻结的优化研究契约 + 隐藏 Holdout 验收（P1-⑦，融合 ScholarAgent）。

移植自 ScholarAgent backend/app/research/spec.py（MIT License）:
- ResearchSpec: 冻结(frozen=True)的优化契约。materialize 一次并哈希, 运行期间
  任何字段(指标键/方向/阈值/预算/holdout 命令)都不得变化, 防"优化过程中
  悄悄改验收标准"的泄漏;
- run_hidden_holdout: 只对最终 best 候选进行多轮独立评估(holdout_repeats 次),
  计算均值/标准差与目标达成, 判定 passed —— 防止过度拟合到单次评估噪音。

与 evidence/graph.py 的 verify_frozen_rubric 分工: rubric 冻结管"复现验收标准",
本模块管"优化契约与最终 Holdout"; 两者都是确定性纯函数, 无 LLM。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

SPEC_VERSION = "autorepro.spec/v1"
MAX_TRIALS_LIMIT = 20
MAX_EVAL_REPEATS = 5
MAX_HOLDOUT_REPEATS = 10


class InvalidResearchSpec(ValueError):
    """spec 字段非法或缺失。"""


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(str(item).strip() for item in value if str(item).strip())


def _bounded_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ResearchSpec:
    """冻结的优化契约：一拍即定, 运行期不可变。

    必填: metric_key + direction（direction 决定胜负方向与 target 判定）；
    可选: min_delta / target_score / max_trials / holdout_repeats / 命令等。
    workspace/eval_command 允许为空：AutoReproducer 的执行器是 CodeExecutor
    而非 shell 命令, 调用方按需注入 holdout 执行器。
    """

    metric_key: str
    direction: str = "maximize"
    workspace: str = ""
    eval_command: str = ""
    holdout_command: str = ""
    holdout_repeats: int = 1
    min_delta: float = 0.0
    target_score: Optional[float] = None
    max_trials: int = 3
    max_patch_files: int = 3
    editable_files: Tuple[str, ...] = ()
    protected_files: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.metric_key.strip():
            raise InvalidResearchSpec("metric_key must not be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise InvalidResearchSpec(
                f"direction must be maximize|minimize, got {self.direction!r}")
        if not 1 <= self.max_trials <= MAX_TRIALS_LIMIT:
            raise InvalidResearchSpec(
                f"max_trials must be within 1..{MAX_TRIALS_LIMIT}")
        if self.min_delta < 0:
            raise InvalidResearchSpec("min_delta must be >= 0")
        if not 1 <= self.holdout_repeats <= MAX_HOLDOUT_REPEATS:
            raise InvalidResearchSpec(
                f"holdout_repeats must be within 1..{MAX_HOLDOUT_REPEATS}")

    # -------------------------------------------------------------- 序列化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": SPEC_VERSION,
            "metric_key": self.metric_key,
            "direction": self.direction,
            "workspace": self.workspace,
            "eval_command": self.eval_command,
            "holdout_command": self.holdout_command,
            "holdout_repeats": self.holdout_repeats,
            "min_delta": self.min_delta,
            "target_score": self.target_score,
            "max_trials": self.max_trials,
            "max_patch_files": self.max_patch_files,
            "editable_files": list(self.editable_files),
            "protected_files": list(self.protected_files),
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def sha256(self) -> str:
        """契约稳定指纹：排除 version/notes（注释性字段），其余按序哈希。

        运行前后重算一致 => 证明契约未在优化过程中被改动。
        """
        payload = {key: value for key, value in self.to_dict().items()
                   if key not in {"version", "notes"}}
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            .encode("utf-8")).hexdigest()

    # -------------------------------------------------------------- 判定

    def improves(self, score: float, reference: float) -> bool:
        """score 相对 reference 是否构成有效改进（超过 min_delta 才算）。"""
        margin = score - reference
        if self.direction == "maximize":
            return margin > self.min_delta
        return -margin > self.min_delta

    def target_reached(self, score: float) -> bool:
        """绝对目标达成判定（target_score 为 None 时永不达成）。"""
        if self.target_score is None:
            return False
        if self.direction == "maximize":
            return score >= self.target_score
        return score <= self.target_score

    # -------------------------------------------------------------- 构造

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResearchSpec":
        if not isinstance(payload, dict):
            raise InvalidResearchSpec("spec payload must be a JSON object")
        try:
            return cls(
                metric_key=str(payload.get("metric_key", "score")),
                direction=str(payload.get("direction", "maximize")),
                workspace=str(payload.get("workspace", "")),
                eval_command=str(payload.get("eval_command", "")),
                holdout_command=str(payload.get("holdout_command", "")),
                holdout_repeats=_bounded_int(
                    payload.get("holdout_repeats", 1), 1,
                    MAX_HOLDOUT_REPEATS, 1),
                min_delta=float(payload.get("min_delta", 0.0) or 0.0),
                target_score=_optional_float(payload.get("target_score")),
                max_trials=_bounded_int(
                    payload.get("max_trials", 3), 1, MAX_TRIALS_LIMIT, 3),
                max_patch_files=_bounded_int(
                    payload.get("max_patch_files", 3), 1, 10, 3),
                editable_files=_string_tuple(payload.get("editable_files", [])),
                protected_files=_string_tuple(payload.get("protected_files", [])),
                notes=str(payload.get("notes", "")),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, InvalidResearchSpec):
                raise
            raise InvalidResearchSpec(f"invalid spec field: {exc}") from exc

    @classmethod
    def from_json(cls, raw: str) -> "ResearchSpec":
        try:
            payload = json.loads(raw or "")
        except json.JSONDecodeError as exc:
            raise InvalidResearchSpec("invalid research_spec JSON") from exc
        return cls.from_dict(payload)


def verify_frozen_spec(spec: ResearchSpec,
                       declared_sha256: str = "") -> Tuple[bool, str]:
    """重算并核对契约指纹；也可声明期传入外部快照哈希做一致性校验。"""
    recomputed = spec.sha256()
    if declared_sha256 and recomputed != declared_sha256:
        return False, (f"spec hash mismatch: declared {declared_sha256[:16]}..., "
                       f"recomputed {recomputed[:16]}...")
    return True, recomputed


# ===========================================================================
# 隐藏 Holdout 验收
# ===========================================================================

HoldoutRunner = Callable[[str, int], Dict[str, Any]]
"""holdout 执行器签名: (best_arm, run_index) -> {"exit_code": int,
"metrics": {metric: value}}；metric_key 对应的数值将作为该轮得分。"""


def _pstdev(values: List[float]) -> Optional[float]:
    """总体标准差；无值 None，单值 0.0。"""
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def run_hidden_holdout(spec: ResearchSpec, best_arm: Optional[str],
                       runner: Optional[HoldoutRunner]) -> Dict[str, Any]:
    """对最终 best 候选做 holdout_repeats 轮独立评估（隐藏 Holdout）。

    只在 runner 可调用且存在 best 候选时执行；否则返回 skipped 结构。
    每轮从 runner 的 metrics 中取 spec.metric_key 作为得分。
    passed 判定 = 所有轮 exit_code==0 且得分齐全（稳定） + 均值达标。
    """
    if spec.holdout_command == "" and runner is None:
        return {"mode": "skipped", "reason": "no holdout runner configured",
                "passed": False}
    if best_arm is None:
        return {"mode": "skipped", "reason": "no valid candidate to verify",
                "passed": False}
    if runner is None:
        return {"mode": "skipped", "reason": "no holdout runner configured",
                "passed": False}

    runs: List[Dict[str, Any]] = []
    for index in range(1, spec.holdout_repeats + 1):
        try:
            result = runner(best_arm, index) or {}
        except Exception as exc:                    # 执行异常视为失败轮
            result = {"exit_code": 1, "error": str(exc)[:200]}
        metrics = result.get("metrics") or {}
        score = None
        if isinstance(metrics, dict):
            raw = metrics.get(spec.metric_key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                score = float(raw)
        entry = {
            "run": index,
            "exit_code": int(result.get("exit_code", 0)),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "score": score,
        }
        if "error" in result:
            entry["error"] = str(result["error"])[:200]
        runs.append(entry)

    scores = [r["score"] for r in runs if r["score"] is not None]
    stable = (len(scores) == spec.holdout_repeats
              and all(r["exit_code"] == 0 for r in runs))
    mean_score = sum(scores) / len(scores) if scores else None
    target_ok = True
    if spec.target_score is not None and mean_score is not None:
        target_ok = spec.target_reached(mean_score)
    return {
        "mode": "hidden_holdout",
        "candidate_arm": best_arm,
        "repeats": spec.holdout_repeats,
        "runs": runs,
        "mean_score": mean_score,
        "std_score": _pstdev(scores),
        "target_score": spec.target_score,
        "target_ok": target_ok,
        "stable": stable,
        "passed": bool(stable and target_ok),
    }