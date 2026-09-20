"""防泄漏 Benchmark 评测（P1-⑩，融合 ScholarAgent backend/app/agents/benchmark.py，MIT License）。

四道防线（全部确定性、无 LLM、无网络）：
1. **确定性 hash 切分**：每行按内容+序号取 `__benchmark_id`（sha256），
   id 尾 8 位 hex %100 映射 train/val/test——hash 桶互斥，
   同一行不会同时出现在 train 与 test，构造性防泄漏；
2. **隐藏标签私有目录**：test/preflight 只写输入特征文件
   （剥掉 target 列 + 保留 `__benchmark_id`），真实标签写入
   hidden_root（沙箱工作区外私有目录）；写盘后重读文件断言
   target 列不存在，feature id 集合与私有标签严格一致，任何
   违反直接抛 LeakageError；
3. **指标契约冻结**：metric/direction/supported/recompute_policy
   一经冻结不可变，带 sha256 指纹（与 ResearchSpec 同一模式），
   后续重算核对指纹证明契约未被偷改；
4. **后端复算**：最终指标只在后端私有标签上重算，分数来自
   隐藏标签，模型/适配器只见无标签特征文件。

与 dataset_registry 集成：Registry 提供合成行存根，配合
ResourceManager.prepare_leakage_safe_benchmark 把注册表数据集
物化为防泄漏测试集，产物登记进 L0 manifest 与分层存储路径。

依赖：仅标准库（csv/json/hashlib/statistics/urllib 无）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random as random_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCHMARK_VERSION = "autorepro.benchmark/v1"
CONTRACT_VERSION = "autorepro.metric-contract/v1"
SPLIT_RATIOS = (0.6, 0.2, 0.2)   # train / validation / test
SUPPORTED_STRATEGIES = {"hash", "random", "group", "time"}
PREFLIGHT_ROWS = 8
MISSING_VALUES = {"", "na", "n/a", "nan", "none", "null"}
TARGET_NAME_HINTS = {
    "label", "target", "y", "class", "category", "answer", "rating",
    "score", "price", "sentiment", "diagnosis", "result", "quality",
    "survived", "default", "输出", "标签", "目标",
}
INPUT_NAME_HINTS = {
    "text", "sentence", "review", "question", "body", "content",
    "comment", "title", "abstract", "summary", "input", "prompt",
    "query", "description", "文本", "内容", "输入",
}
PRIMARY_KEY_HINTS = {"id", "key", "index", "idx", "no", "uid", "uuid"}

# task type -> direction per supported metric
METRIC_DIRECTIONS: Dict[str, Dict[str, str]] = {
    "classification": {
        "accuracy": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "f1": "maximize",
        "macro_f1": "maximize",
    },
    "regression": {
        "mae": "minimize", "mse": "minimize", "rmse": "minimize",
        "r2": "maximize",
    },
}
METRIC_KEY_ALIASES = {
    "macro_f1": "macro_f1", "f1": "macro_f1",
    "precision": "precision_macro", "recall": "recall_macro",
}


class LeakageError(RuntimeError):
    """泄漏或切分契约被违反。"""


class MappingAmbiguousError(RuntimeError):
    """列角色推断歧义，拒绝猜测（fail loudly，与 ScholarAgent 一致）。"""

    def __init__(self, reason: str, candidates: List[Dict[str, Any]],
                 columns: List[str]) -> None:
        super().__init__("benchmark column mapping is ambiguous: " + reason)
        self.reason = reason
        self.candidates = candidates
        self.columns = columns


class MetricComputationError(RuntimeError):
    """指标计算失败（非法输入 / 空预测等）。"""


# ===========================================================================
# 数据读写与稳定哈希
# ===========================================================================

def _is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in MISSING_VALUES)


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return math.isfinite(number) and number or number


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def benchmark_id(row: Dict[str, Any], index: int) -> str:
    """行级稳定 id：内容 + 序号，题目相同行顺序不同 id 也不同。"""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True,
                         default=str)
    return f"bm-{_stable_hash(payload, str(index))[:16]}"


def read_rows(path: str) -> Tuple[List[str], List[Dict[str, Any]], str]:
    """读取 jsonl/json/csv/tsv；返回 (columns, rows, format)。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        columns: List[str] = []
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(payload)
                for key in payload:
                    if key not in columns:
                        columns.append(key)
        return columns, rows, "jsonl"
    if suffix == ".json":
        with open(path, encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        rows = data if isinstance(data, list) else [data]
        columns = list(rows[0].keys()) if rows else []
        return columns, rows, "json"
    delimiter = "\t" if suffix == ".tsv" else ","
    with open(path, encoding="utf-8", errors="replace",
              newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return list(columns), rows, "csv"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ===========================================================================
# 列统计 / 主键检测 / 列角色推断
# ===========================================================================

def column_stats(rows: List[Dict[str, Any]],
                 columns: List[str]) -> Dict[str, Dict[str, Any]]:
    total = len(rows)
    stats: Dict[str, Dict[str, Any]] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        present = [value for value in values if not _is_missing(value)]
        numeric = sum(1 for value in present
                      if _as_float(value) is not None)
        unique = len({str(value) for value in present})
        unique_ratio = unique / len(present) if present else 0.0
        numeric_ratio = numeric / len(present) if present else 0.0
        if present and numeric_ratio >= 0.95:
            dtype = "numeric"
        elif unique_ratio <= 0.05 and unique <= 50:
            dtype = "categorical"
        else:
            dtype = "text"
        stats[column] = {
            "missing": total - len(present),
            "unique": unique,
            "unique_ratio": round(unique_ratio, 4),
            "numeric_ratio": round(numeric_ratio, 4),
            "dtype": dtype,
        }
    return stats


def _detect_primary_key(columns: List[str], declared: str) -> str:
    if declared:
        return declared if declared in columns else ""
    lowered = {column.lower(): column for column in columns}
    for hint in PRIMARY_KEY_HINTS:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        if column.lower().endswith("_id"):
            return column
    return ""


def _score_target_candidate(column: str, stats: Dict[str, Any],
                            is_last: bool) -> int:
    score = 0
    lowered = column.lower()
    if lowered in TARGET_NAME_HINTS:
        score += 4
    elif any(hint in lowered for hint in ("label", "target")):
        score += 3
    if is_last:
        score += 1
    if stats["dtype"] == "numeric" and stats["unique_ratio"] > 0.9:
        score += 1                       # regression 特征
    elif stats["dtype"] == "categorical" and 1 < stats["unique"] <= 50:
        score += 2                       # classification 特征
    return score


def detect_mapping(columns: List[str], rows: List[Dict[str, Any]],
                   declared_target: str = "",
                   declared_input: str = "",
                   declared_task_type: str = "",
                   declared_primary_key: str = "") -> Dict[str, Any]:
    """推断 input/target 列与任务类型；歧义时抛 MappingAmbiguousError。

    声明优先；未声明时按名字/类型信号打分，歧义（并列/弱信号）
    拒绝猜测（fail loudly），返回结构化候选供调用方处理。
    """
    if not columns or not rows:
        raise MappingAmbiguousError(
            "empty dataset", [], list(columns))
    stats = column_stats(rows, columns)
    last_column = columns[-1]

    if declared_target:
        if declared_target not in columns:
            raise MappingAmbiguousError(
                f"declared target column {declared_target!r} "
                "not found in dataset", [], list(columns))
        target = declared_target
        confidence = 0.95
    else:
        scored = sorted(
            ((column, _score_target_candidate(
                column, stats[column], column == last_column))
             for column in columns),
            key=lambda item: item[1], reverse=True)
        candidates = [
            {"column": column, "score": score,
             "dtype": stats[column]["dtype"]}
            for column, score in scored if score > 0]
        if not scored or scored[0][1] <= 0:
            raise MappingAmbiguousError(
                "no column looks like a target "
                "(no name/dtype signals)", candidates, list(columns))
        if len(scored) > 1 and scored[0][1] == scored[1][1]:
            target = scored[0][0]
            raise MappingAmbiguousError(
                f"columns {scored[0][0]!r} and {scored[1][0]!r} "
                "tie as target candidates", candidates, list(columns))
        target = scored[0][0]
        confidence = min(0.9, 0.5 + 0.1 * scored[0][1])

    target_stats = stats.get(target, {})
    task_type = declared_task_type
    if not task_type:
        if (target_stats.get("dtype") == "numeric"
                and target_stats.get("unique_ratio", 0) > 0.95):
            task_type = "regression"
        else:
            task_type = "classification"

    input_column = ""
    if declared_input and declared_input in columns:
        input_column = declared_input
    else:
        input_candidates = sorted(
            ((column, (3 if column.lower() in INPUT_NAME_HINTS else 0)
              + stats[column]["unique_ratio"])
             for column in columns
             if column != target
             and stats[column]["dtype"] in {"text", "categorical"}),
            key=lambda item: item[1], reverse=True)
        input_column = (input_candidates[0][0] if input_candidates
                        else (columns[0] if columns else ""))

    return {
        "target_column": target,
        "input_column": input_column,
        "task_type": task_type,
        "primary_key": _detect_primary_key(columns, declared_primary_key),
        "mapping_confidence": round(confidence, 3),
        "column_stats": stats,
    }


# ===========================================================================
# 确定性切分
# ===========================================================================

def split_assignments(rows: List[Dict[str, Any]],
                      strategy: str = "hash",
                      group_column: str = "",
                      time_column: str = "",
                      seed: int = 42) -> List[str]:
    """每行 -> train/validation/test（按冻结策略）。

    hash（默认）：按 __benchmark_id 尾 8 位 hex 取模，行级哈希互斥；
    random：seed 固定可复现；group：按分组列哈希（同组同行）；
    time：按时间列升序切窗（时序前提，防穿越）。
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise LeakageError(
            f"unsupported split strategy {strategy!r}; "
            f"supported: {sorted(SUPPORTED_STRATEGIES)}")
    count = len(rows)

    def bucket_of(seed_key: str) -> str:
        bucket = int(seed_key[-8:], 16) % 100
        if bucket < 60:
            return "train"
        if bucket < 80:
            return "validation"
        return "test"

    def into_window(position: int) -> str:
        cut = position / max(count, 1)
        if cut < SPLIT_RATIOS[0]:
            return "train"
        if cut < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
            return "validation"
        return "test"

    if strategy == "hash":
        return [bucket_of(row["__benchmark_id"]) for row in rows]
    if strategy == "random":
        indices = list(range(count))
        random_module.Random(seed).shuffle(indices)
        assignments = [""] * count
        for position, index in enumerate(indices):
            assignments[index] = into_window(position)
        return assignments
    if strategy == "group":
        if not group_column:
            raise LeakageError("group split requires benchmark_group_column")
        for row in rows:
            if _is_missing(row.get(group_column)):
                raise LeakageError(
                    f"group column {group_column!r} has an empty value; "
                    "group split cannot proceed")
        return [bucket_of(_stable_hash(str(row.get(group_column, ""))))
                for row in rows]
    # time
    if not time_column:
        raise LeakageError("time split requires benchmark_time_column")
    parsed: List[Tuple[float, int]] = []
    for index, row in enumerate(rows):
        raw = row.get(time_column)
        number = _as_float(raw)
        if number is not None:
            parsed.append((number, index))
            continue
        try:
            stamp = datetime.fromisoformat(
                str(raw).strip().replace("Z", "+00:00"))
            parsed.append((stamp.timestamp(), index))
        except (TypeError, ValueError):
            raise LeakageError(
                f"time column {time_column!r} has unparsable value "
                f"{raw!r}; time split cannot proceed")
    ordered = [index for _, index in sorted(parsed)]
    assignments = [""] * count
    for position, index in enumerate(ordered):
        assignments[index] = into_window(position)
    return assignments


# ===========================================================================
# 物化：公开特征文件 + 私有隐藏标签
# ===========================================================================

def _feature_row(row: Dict[str, Any], row_id: str,
                 target: str) -> Dict[str, Any]:
    """只读输入视图：剥掉 target 列，保留 __benchmark_id。"""
    feature = {key: value for key, value in row.items()
               if key != target}
    feature["__benchmark_id"] = row_id
    return feature


def assert_no_labels(path: Path, target: str) -> bool:
    """重读已写盘的特征文件，target 列泄漏即抛 LeakageError。"""
    for row in read_jsonl(path):
        if target in row:
            raise LeakageError(
                f"hidden-label leak: target column {target!r} "
                f"present in {path.name}")
        if "__benchmark_id" not in row:
            raise LeakageError(
                f"feature row without __benchmark_id in {path.name}")
    return True


@dataclass
class BenchmarkPrepared:
    """防泄漏基准产物（含公开工作区路径与私有标签路径）。"""

    manifest: Dict[str, Any]
    public_dir: Path
    hidden_dir: Path
    hidden_labels_path: Path
    test_features_path: Path
    preflight_features_path: Path
    mapping: Dict[str, Any]
    contract: Dict[str, Any] = field(default_factory=dict)


def materialize_benchmark(
    rows: List[Dict[str, Any]],
    target: str,
    hidden_root: Path,
    public_dir: Optional[Path] = None,
    input_column: str = "",
    task_type: str = "classification",
    strategy: str = "hash",
    group_column: str = "",
    time_column: str = "",
    seed: int = 42,
    max_samples: int = 1000,
    primary_metric: str = "",
    target_score: Optional[float] = None,
) -> BenchmarkPrepared:
    """完整物化：赋予 id -> 切分 -> 写公开特征 + 私有标签 -> 泄漏自检。

    返回 BenchmarkPrepared；任何一步泄漏/歧义抛对应异常。
    """
    rows = list(rows)[: max_samples] if max_samples > 0 else list(rows)
    if not rows:
        raise LeakageError("materialization found no dataset rows")
    if target not in rows[0]:
        raise LeakageError(
            f"target column {target!r} not present in dataset rows")

    for index, row in enumerate(rows):
        row["__benchmark_id"] = benchmark_id(row, index)
    assignments = split_assignments(rows, strategy, group_column,
                                    time_column, seed)

    train: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    test_features_source: List[Dict[str, Any]] = []
    for row, assignment in zip(rows, assignments):
        if assignment == "train":
            train.append(row)
        elif assignment == "validation":
            validation.append(row)
        else:
            test_features_source.append(row)

    test_rows = [_feature_row(row, row["__benchmark_id"], target)
                 for row in test_features_source]
    preflight_rows = test_rows[:PREFLIGHT_ROWS]

    hidden_dir = hidden_root
    hidden_labels = [
        {"__benchmark_id": row["__benchmark_id"],
         target: row.get(target)}
        for row in test_features_source]
    hidden_labels_path = hidden_dir / "hidden_labels.jsonl"
    write_jsonl(hidden_labels_path, hidden_labels)
    hidden_labels_sha = _stable_hash(
        json.dumps(hidden_labels, ensure_ascii=False, sort_keys=True,
                   default=str))

    if public_dir is None:
        public_dir = hidden_dir / "splits"
    split_paths: Dict[str, Path] = {}
    for name, split_rows in (
            ("train", train),
            ("validation", validation),
            ("preflight_features", preflight_rows),
            ("test_features", test_rows)):
        path = public_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        split_paths[name] = path

    # 写盘后泄漏自检（对磁盘上的真实文件）
    leak_checks = {
        "test_features": assert_no_labels(
            split_paths["test_features"], target),
        "preflight_features": assert_no_labels(
            split_paths["preflight_features"], target),
    }
    feature_ids = [row["__benchmark_id"] for row in test_rows]
    label_ids = [row["__benchmark_id"] for row in hidden_labels]
    if sorted(feature_ids) != sorted(label_ids):
        raise LeakageError(
            "hidden-label leak: feature ids do not match "
            "the private label store")

    manifest = {
        "version": BENCHMARK_VERSION,
        "strategy": strategy,
        "seed": seed if strategy == "random" else None,
        "group_column": group_column if strategy == "group" else None,
        "time_column": time_column if strategy == "time" else None,
        "splits": {
            name: {"path": str(path), "rows": len(split_rows)}
            for name, path, split_rows in (
                ("train", split_paths.get("train", Path("")),
                 train),
                ("validation", split_paths.get("validation", Path("")),
                 validation),
                ("preflight_features",
                 split_paths.get("preflight_features", Path("")),
                 preflight_rows),
                ("test_features",
                 split_paths.get("test_features", Path("")),
                 test_rows),
            )},
        "input_column": input_column,
        "target_column": target,
        "task_type": task_type,
        "test_row_count": len(test_rows),
        "hidden_labels_sha256": hidden_labels_sha,
        "hidden_labels_store": "backend-private (outside sandbox)",
        "leakage_report": {
            "strategy": strategy,
            "train_validation_overlap": "hash-disjoint by construction",
            "test_contains_labels": False,
            "preflight_contains_labels": False,
            "verified_on_disk": {
                "test_features_no_target_column":
                    bool(leak_checks["test_features"]),
                "preflight_features_no_target_column":
                    bool(leak_checks["preflight_features"]),
                "feature_ids_match_hidden_labels": True,
            },
        },
    }
    return BenchmarkPrepared(
        manifest=manifest,
        public_dir=public_dir,
        hidden_dir=hidden_dir,
        hidden_labels_path=hidden_labels_path,
        test_features_path=split_paths["test_features"],
        preflight_features_path=split_paths["preflight_features"],
        mapping={"target_column": target, "input_column": input_column,
                 "task_type": task_type},
    )


# ===========================================================================
# 指标契约冻结（recompute policy 后端复算）
# ===========================================================================

def freeze_metric_contract(task_type: str, primary_metric: str,
                           target_score: Optional[float] = None
                           ) -> Dict[str, Any]:
    """冻结指标契约：metric/direction/supported/recompute_policy。

    冻结确定性内容 + sha256 指纹；后续复算前核对指纹，
    证明契约未被优化过程偷改（与 ResearchSpec 同一防泄漏模式）。
    """
    supported = METRIC_DIRECTIONS.get(task_type)
    if supported is None:
        raise LeakageError(
            f"unsupported task type {task_type!r}; "
            f"supported: {sorted(METRIC_DIRECTIONS)}")
    metric = primary_metric.strip() or (
        "accuracy" if task_type == "classification" else "mae")
    if metric not in supported:
        raise LeakageError(
            f"primary metric {metric!r} is not supported for task type "
            f"{task_type!r}; supported: {sorted(supported)}")
    contract = {
        "version": CONTRACT_VERSION,
        "task_type": task_type,
        "primary_metric": metric,
        "primary_direction": supported[metric],
        "supported_metrics": [
            {"name": name, "direction": direction}
            for name, direction in sorted(supported.items())],
        "recompute_policy": "backend_recompute_on_hidden_labels",
        "reward_policy": "candidate_ranking_only",
        "hidden_from_adapter": True,
        "frozen_before_execution": True,
    }
    if target_score is not None:
        contract["target_score"] = float(target_score)
    contract["sha256"] = hash_contract(contract)
    return contract


def hash_contract(contract: Dict[str, Any]) -> str:
    """契约指纹：排除注释性字段（version/sha256），其余按序哈希。"""
    payload = {key: value for key, value in contract.items()
               if key not in {"version", "sha256"}}
    return _stable_hash(json.dumps(payload, ensure_ascii=False,
                                   sort_keys=True))


def verify_frozen_contract(contract: Dict[str, Any],
                           declared_sha256: str = "") -> Tuple[bool, str]:
    """核对契约指纹；可通过 declared_sha256 与外部快照比对。"""
    recomputed = hash_contract(contract)
    if declared_sha256 and recomputed != declared_sha256:
        return False, (
            f"contract hash mismatch: declared "
            f"{declared_sha256[:16]}..., recomputed {recomputed[:16]}...")
    return True, recomputed


# ===========================================================================
# 后端复算：只在私有隐藏标签上计算指标
# ===========================================================================

def _binary_macro_scores(y_true: List[Any],
                         y_pred: List[Any]) -> Dict[str, float]:
    """多分类标签下计算 accuracy 与各类宏平均 precision/recall/f1。

    纯 Python，无 sklearn 依赖；按出现类别逐类算 TP/FP/FN，
    宏平均；单类退化时该类指标按定义计算。
    """
    matched = list(zip(y_true, y_pred))
    if not matched:
        raise MetricComputationError("empty predictions for metrics")
    classes = sorted({str(value) for value in y_true})
    correct = sum(1 for t, p in matched if str(t) == str(p))
    accuracy = correct / len(matched)
    if not classes:
        return {"accuracy": round(accuracy, 6)}
    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = sum(1 for t, p in matched if str(t) == cls and str(p) == cls)
        fp = sum(1 for t, p in matched if str(t) != cls and str(p) == cls)
        fn = sum(1 for t, p in matched if str(t) == cls and str(p) != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": round(accuracy, 6),
        "precision_macro": round(sum(precisions) / len(precisions), 6),
        "recall_macro": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
    }


def _regression_scores(y_true: List[float],
                       y_pred: List[float]) -> Dict[str, float]:
    if not y_true:
        raise MetricComputationError("empty ground truth for metrics")
    errors = [t - p for t, p in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in errors) / len(errors)
    mse = sum(e * e for e in errors) / len(errors)
    rmse = math.sqrt(mse)
    mean_true = sum(y_true) / len(y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def compute_metrics(task_type: str, y_true: List[Any],
                    y_pred: List[Any]) -> Dict[str, float]:
    """按任务类型计算契约指标集（后端复算唯一入口）。"""
    values = list(zip(y_true, y_pred))
    if not values:
        raise MetricComputationError("no matched predictions")
    if task_type == "regression":
        try:
            t_float = [float(value) for value in y_true]
            p_float = [float(value) for value in y_pred]
        except (TypeError, ValueError) as exc:
            raise MetricComputationError(
                "regression metrics require numeric labels") from exc
        return _regression_scores(t_float, p_float)
    return _binary_macro_scores(y_true, y_pred)


def validate_hidden(predictions: List[Dict[str, Any]],
                    labels_path: Path, target: str,
                    task_type: str = "classification",
                    primary_metric: str = "") -> Dict[str, Any]:
    """后端复算：hidden labels（私有）+ 预测行 -> 契约指标 + 覆盖核对。

    预测行格式：[{"__benchmark_id": "...", "prediction": ...}, ...]。
    """
    label_rows = read_jsonl(labels_path)
    hidden_labels: Dict[str, Any] = {}
    duplicate_ids: List[str] = []
    for row in label_rows:
        row_id = str(row.get("__benchmark_id", ""))
        if not row_id:
            raise LeakageError("hidden label row without __benchmark_id")
        if row_id in hidden_labels:
            duplicate_ids.append(row_id)
        hidden_labels[row_id] = row.get(target)

    prediction_map: Dict[str, Any] = {}
    extra_ids: List[str] = []
    seen: set = set()
    for row in predictions:
        row_id = str(row.get("__benchmark_id")
                     or row.get("id") or "")
        if not row_id:
            continue
        if row_id in seen:
            continue
        seen.add(row_id)
        prediction_map[row_id] = row.get("prediction")
        if row_id not in hidden_labels:
            extra_ids.append(row_id)

    matched_ids = [row_id for row_id in hidden_labels
                   if row_id in prediction_map]
    missing_ids = [row_id for row_id in hidden_labels
                   if row_id not in prediction_map]

    metric_name = primary_metric.strip() or (
        "accuracy" if task_type == "classification" else "mae")
    metric_key = METRIC_KEY_ALIASES.get(metric_name, metric_name)
    metrics: Dict[str, float] = {}
    score: Optional[float] = None
    if matched_ids:
        y_true = [hidden_labels[row_id] for row_id in matched_ids]
        y_pred = [prediction_map[row_id] for row_id in matched_ids]
        metrics = compute_metrics(task_type, y_true, y_pred)
        if metric_key in metrics:
            score = metrics[metric_key]
        else:
            raise MetricComputationError(
                f"primary metric {metric_name!r} not among recomputed "
                f"metrics {sorted(metrics)}")

    return {
        "version": BENCHMARK_VERSION,
        "task_type": task_type,
        "metric": metric_name,
        "metrics": metrics,
        "score": score,
        "matched": len(matched_ids),
        "missing_ids": len(missing_ids),
        "duplicate_ids": len(sorted(set(duplicate_ids))),
        "extra_ids": len(extra_ids),
        "prediction_coverage": round(
            len(matched_ids) / len(hidden_labels), 6)
        if hidden_labels else 0.0,
        "hidden_label_source": "backend-private store",
    }


# ===========================================================================
# 与 dataset_registry 集成（合成行存根 + 由 Registry 校验度量合约）
# ===========================================================================

def synthetic_rows_for(dataset_name: str, n: int = 120,
                       seed: int = 42) -> List[Dict[str, Any]]:
    """为注册表数据集生成可复现的合成行存根（防泄漏评测用）。

    语义：评测关注的是"防泄漏机制 + 契约复算"这条链路本身，
    行数据仅作确定性载体；真实数据接入时替换为 read_rows 结果即可。
    返回 [{"id": i, "text": ..., "label": ...}] 分类样例。
    """
    rng = random_module.Random(seed + len(str(dataset_name)))
    classes = ["pos", "neg", "neutral"]
    rows: List[Dict[str, Any]] = []
    for index in range(n):
        rows.append({
            "id": f"{index:04d}",
            "text": f"{dataset_name}-sample-{index} "
                    f"feature {rng.randrange(1000)}",
            "label": classes[rng.randrange(len(classes))],
        })
    return rows