"""防泄漏 Benchmark 评测测试（P1-⑩）。

覆盖：稳定 id / 列角色推断（歧义拒绝）/ 四种确定性切分 /
物化 + 泄漏自检（写盘后复查）/ 契约冻结指纹 / 后端复算指标 /
validate_hidden 覆盖核对 / dataset_registry 集成 / ResourceManager
prepare_leakage_safe_benchmark 端到端。全部纯本地，无网络。
"""
import json
from pathlib import Path

import pytest

from src.benchmark.leakage_safe import (
    LeakageError,
    MappingAmbiguousError,
    MetricComputationError,
    assert_no_labels,
    benchmark_id,
    column_stats,
    compute_metrics,
    detect_mapping,
    freeze_metric_contract,
    hash_contract,
    materialize_benchmark,
    read_jsonl,
    split_assignments,
    synthetic_rows_for,
    validate_hidden,
    verify_frozen_contract,
)
from src.dataset_registry import DatasetRegistry
from src.resource_manager import ResourceManager

ROWS = [
    {"id": "0000", "text": f"sample {i} content", "label": "pos" if i % 3 == 0 else "neg"}
    for i in range(120)
]


# ---------------- 稳定 id ----------------

class TestBenchmarkId:
    def test_stable_and_unique(self):
        ids = [benchmark_id({"a": i}, i) for i in range(50)]
        assert len(set(ids)) == 50
        assert benchmark_id({"a": 1}, 1) == benchmark_id({"a": 1}, 1)
        # 内容不同或序位不同 => id 不同
        assert benchmark_id({"a": 2}, 1) != benchmark_id({"a": 1}, 1)
        assert benchmark_id({"a": 1}, 2) != benchmark_id({"a": 1}, 1)

    def test_prefix(self):
        assert benchmark_id({"a": 1}, 0).startswith("bm-")


# ---------------- 列角色推断 ----------------

class TestDetectMapping:
    def test_declared_target_wins(self):
        mapping = detect_mapping(["text", "label"], ROWS,
                                 declared_target="label")
        assert mapping["target_column"] == "label"
        assert mapping["mapping_confidence"] == 0.95

    def test_auto_detection_uses_hint(self):
        mapping = detect_mapping(["text", "label"], ROWS)
        assert mapping["target_column"] == "label"
        assert mapping["input_column"] == "text"
        assert mapping["task_type"] == "classification"

    def test_ambiguous_tie_fails_loudly(self):
        # label 与 target 同为强 hint 且均非末列 -> 并列最高，拒绝猜测
        rows = [
            {"label": "pos", "target": "pos", "id": str(i)}
            for i in range(30)
        ]
        with pytest.raises(MappingAmbiguousError) as exc:
            detect_mapping(["label", "target", "id"], rows,
                           declared_target="")
        assert "tie" in str(exc.value)

    def test_declared_missing_rejected(self):
        with pytest.raises(MappingAmbiguousError):
            detect_mapping(["text", "label"], ROWS,
                           declared_target="nope")

    def test_empty_dataset_rejected(self):
        with pytest.raises(MappingAmbiguousError):
            detect_mapping([], [])


# ---------------- 切分策略 ----------------

class TestSplitAssignments:
    def _tagged(self):
        rows = [dict(row) for row in ROWS]
        for index, row in enumerate(rows):
            row["__benchmark_id"] = benchmark_id(row, index)
        return rows

    def test_hash_disjoint_train_test(self):
        rows = self._tagged()
        assigned = split_assignments(rows, strategy="hash")
        train_ids = {r["__benchmark_id"] for r, a in zip(rows, assigned)
                     if a == "train"}
        test_ids = {r["__benchmark_id"] for r, a in zip(rows, assigned)
                    if a == "test"}
        assert not (train_ids & test_ids)      # 构造性无重叠
        assert set(assigned) <= {"train", "validation", "test"}
        assert "test" in set(assigned)

    def test_random_deterministic_with_seed(self):
        rows = self._tagged()
        a1 = split_assignments(rows, strategy="random", seed=42)
        a2 = split_assignments(rows, strategy="random", seed=42)
        a3 = split_assignments(rows, strategy="random", seed=7)
        assert a1 == a2
        assert a1 != a3

    def test_group_requires_column(self):
        rows = self._tagged()
        with pytest.raises(LeakageError):
            split_assignments(rows, strategy="group")
        for row in rows:
            row["group"] = f"g{int(row['id']) % 10}"
        assigned = split_assignments(rows, strategy="group",
                                     group_column="group")
        assert set(assigned) <= {"train", "validation", "test"}

    def test_time_split_chronological(self):
        rows = []
        for i in range(90):
            rows.append({"__benchmark_id": f"bm-t{i:016x}",
                         "ts": f"2024-01-{i % 28 + 1:02d}T00:00:00"})
        assigned = split_assignments(rows, strategy="time",
                                     time_column="ts")
        # 时序前提：train 的时间范围不晚于 test 起点
        train_max = max(int(r["ts"][8:10])
                        for r, a in zip(rows, assigned)
                        if a == "train")
        test_min = min(int(r["ts"][8:10])
                       for r, a in zip(rows, assigned) if a == "test")
        assert train_max <= test_min

    def test_unsupported_strategy(self):
        with pytest.raises(LeakageError):
            split_assignments(self._tagged(), strategy="magic")


# ---------------- 物化与泄漏自检 ----------------

class TestMaterialize:
    def _prepared(self, tmp_path):
        return materialize_benchmark(
            rows=ROWS, target="label",
            hidden_root=tmp_path / "hidden",
            public_dir=tmp_path / "public" / "splits",
            max_samples=120)

    def test_writes_public_features_and_private_labels(self, tmp_path):
        prepared = self._prepared(tmp_path)
        assert (tmp_path / "public" / "splits" / "train.jsonl").exists()
        assert (tmp_path / "public" / "splits"
                / "validation.jsonl").exists()
        assert (tmp_path / "public" / "splits"
                / "test_features.jsonl").exists()
        assert (tmp_path / "public" / "splits"
                / "preflight_features.jsonl").exists()
        assert (tmp_path / "hidden" / "hidden_labels.jsonl").exists()

    def test_public_features_have_no_labels(self, tmp_path):
        prepared = self._prepared(tmp_path)
        for name in ("test_features", "preflight_features"):
            path = prepared.public_dir / f"{name}.jsonl"
            assert assert_no_labels(path, "label") is True
            for row in read_jsonl(path):
                assert "label" not in row
                assert "__benchmark_id" in row

    def test_feature_ids_match_hidden_labels(self, tmp_path):
        prepared = self._prepared(tmp_path)
        feature_ids = {row["__benchmark_id"]
                       for row in read_jsonl(
                           prepared.test_features_path)}
        label_ids = {row["__benchmark_id"]
                     for row in read_jsonl(prepared.hidden_labels_path)}
        assert feature_ids == label_ids
        report = prepared.manifest["leakage_report"]
        assert report["test_contains_labels"] is False
        assert report["verified_on_disk"][
            "test_features_no_target_column"] is True

    def test_leak_detected_when_target_present(self, tmp_path):
        prepared = self._prepared(tmp_path)
        leaked = prepared.public_dir / "test_features.jsonl"
        rows = read_jsonl(leaked)
        rows[0]["label"] = "pos"           # 模拟泄漏
        leaked.write_text("\n".join(
            json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        with pytest.raises(LeakageError, match="leak"):
            assert_no_labels(leaked, "label")

    def test_target_missing_rejected(self, tmp_path):
        with pytest.raises(LeakageError):
            materialize_benchmark(
                rows=[{"a": 1}], target="label",
                hidden_root=tmp_path / "h",
                public_dir=tmp_path / "p")

    def test_manifest_summary(self, tmp_path):
        prepared = self._prepared(tmp_path)
        m = prepared.manifest
        assert m["version"].startswith("autorepro.benchmark")
        assert m["test_row_count"] == len(read_jsonl(
            prepared.test_features_path))
        assert m["hidden_labels_sha256"]


# ---------------- 指标契约冻结 ----------------

class TestMetricContract:
    def test_freeze_classification(self):
        c = freeze_metric_contract("classification", "accuracy")
        assert c["primary_direction"] == "maximize"
        assert c["recompute_policy"] == \
            "backend_recompute_on_hidden_labels"
        assert c["hidden_from_adapter"] is True
        assert c["frozen_before_execution"] is True
        assert c["sha256"]

    def test_freeze_regression_default_metric(self):
        c = freeze_metric_contract("regression", "")
        assert c["primary_metric"] == "mae"
        assert c["primary_direction"] == "minimize"

    def test_unsupported_task_and_metric(self):
        with pytest.raises(LeakageError):
            freeze_metric_contract("gan", "accuracy")
        with pytest.raises(LeakageError):
            freeze_metric_contract("classification", "auc")

    def test_verify_tamper_detection(self):
        c = freeze_metric_contract("classification", "accuracy",
                                   target_score=0.9)
        ok, recomputed = verify_frozen_contract(c, c["sha256"])
        assert ok
        assert recomputed == c["sha256"]
        # 偷改契约（如把 direction 从 maximize 改为 minimize）
        c["primary_direction"] = "minimize"
        ok, _ = verify_frozen_contract(c, recomputed)
        assert ok is False
        # 未声明指纹时至少重算自洽
        ok2, _ = verify_frozen_contract(c)
        assert ok2 is True
        assert hash_contract(c) != recomputed


# ---------------- 后端复算 ----------------

class TestComputeMetrics:
    def test_classification(self):
        # a: TP=2 FP=1 FN=0 -> p=2/3 r=1 f1=0.8；b: TP=1 FP=0 FN=1
        # -> p=1 r=0.5 f1=2/3；macro_f1=(0.8+2/3)/2=0.7333
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "a"]
        m = compute_metrics("classification", y_true, y_pred)
        assert m["accuracy"] == 0.75
        assert round(m["macro_f1"], 4) == round((0.8 + 2 / 3) / 2, 4)
        assert m["precision_macro"] == pytest.approx((2 / 3 + 1) / 2)
        assert m["recall_macro"] == 0.75

    def test_regression(self):
        m = compute_metrics("regression", [1, 2, 3], [1, 2, 4])
        assert m["mae"] == pytest.approx(1 / 3)
        assert m["mse"] == pytest.approx(1 / 3)
        assert m["rmse"] == pytest.approx((1 / 3) ** 0.5)
        assert m["r2"] == 0.5

    def test_perfect_regression_r2(self):
        m = compute_metrics("regression", [1, 2, 3], [1, 2, 3])
        assert m["r2"] == 0.0 if False else m["r2"] == 1.0

    def test_empty_predictions_fail(self):
        with pytest.raises(MetricComputationError):
            compute_metrics("classification", [], [])

    def test_non_numeric_regression_fails(self):
        with pytest.raises(MetricComputationError):
            compute_metrics("regression", ["a"], ["b"])


class TestValidateHidden:
    def _labels(self, tmp_path, rows):
        hidden = tmp_path / "hidden_labels.jsonl"
        hidden.write_text("\n".join(
            json.dumps({"__benchmark_id": benchmark_id(r, i),
                        "label": r["label"]}, ensure_ascii=False)
            for i, r in enumerate(rows)) + "\n", encoding="utf-8")
        return hidden

    def test_perfect_predictions_full_coverage(self, tmp_path):
        rows = ROWS[:30]
        labels = self._labels(tmp_path, rows)
        preds = [{"__benchmark_id": benchmark_id(r, i),
                  "prediction": r["label"]}
                 for i, r in enumerate(rows)]
        report = validate_hidden(preds, labels, "label")
        assert report["score"] == 1.0
        assert report["prediction_coverage"] == 1.0
        assert report["matched"] == 30
        assert report["missing_ids"] == 0
        assert report["extra_ids"] == 0

    def test_partial_coverage_and_extra(self, tmp_path):
        rows = ROWS[:30]
        labels = self._labels(tmp_path, rows)
        preds = ([{"__benchmark_id": benchmark_id(r, i),
                   "prediction": "neg"}
                  for i, r in enumerate(rows[:20])]
                 + [{"__benchmark_id": "bm-unknown-filler",
                     "prediction": "pos"}])
        report = validate_hidden(preds, labels, "label")
        assert report["matched"] == 20
        assert report["missing_ids"] == 10
        assert report["extra_ids"] == 1
        assert report["prediction_coverage"] == pytest.approx(20 / 30)

    def test_regression_primary_metric(self, tmp_path):
        labels = tmp_path / "hidden_labels.jsonl"
        labels.write_text(
            "\n".join(json.dumps({"__benchmark_id": f"bm-r{i}",
                                  "target": float(i)})
                      for i in range(5)) + "\n",
            encoding="utf-8")
        preds = [{"__benchmark_id": f"bm-r{i}", "prediction": float(i)}
                 for i in range(4)]
        report = validate_hidden(preds, labels, "target",
                                 task_type="regression",
                                 primary_metric="mae")
        assert report["metrics"]["mae"] == 0.0
        assert report["score"] == 0.0
        assert report["metric"] == "mae"

    def test_primary_metric_not_supported_fails(self, tmp_path):
        rows = ROWS[:10]
        labels = self._labels(tmp_path, rows)
        preds = [{"__benchmark_id": benchmark_id(r, i),
                  "prediction": "pos"} for i, r in enumerate(rows)]
        with pytest.raises(MetricComputationError):
            validate_hidden(preds, labels, "label",
                            primary_metric="auc")

    def test_missing_benchmark_id_fails(self, tmp_path):
        labels = tmp_path / "hidden_labels.jsonl"
        labels.write_text(json.dumps({"label": "x"}) + "\n",
                          encoding="utf-8")
        with pytest.raises(LeakageError):
            validate_hidden([], labels, "label")


# ---------------- 与 dataset_registry / ResourceManager 集成 ----------------

class TestRegistryIntegration:
    def test_benchmark_hint_sizes(self):
        reg = DatasetRegistry()
        cifar = reg.benchmark_hint("CIFAR-10")
        assert cifar["found"] is True
        assert cifar["recommended_max_samples"] == 1000
        imagenet = reg.benchmark_hint("ImageNet-1k")
        assert imagenet["subset"] == "percent:1"
        assert imagenet["recommended_max_samples"] == 26
        synthetic = reg.benchmark_hint("SYNTHETIC-DATA")
        assert synthetic["recommended_max_samples"] == 24
        unknown = reg.benchmark_hint("不存在的数据集")
        assert unknown["found"] is False
        assert unknown["recommended_max_samples"] == 1000

    def test_synthetic_rows_deterministic(self):
        a = synthetic_rows_for("CIFAR-10", n=30)
        b = synthetic_rows_for("CIFAR-10", n=30)
        assert a == b
        assert all("label" in row and "text" in row for row in a)
        assert len(a) == 30


class TestResourceManagerBenchmark:
    def _rm(self, tmp_path):
        return ResourceManager(data_root=str(tmp_path / "data"))

    def test_prepare_end_to_end(self, tmp_path):
        rm = self._rm(tmp_path)
        info = rm.prepare_leakage_safe_benchmark(
            paper_id="p-cifar", dataset_name="CIFAR-10",
            target_score=0.9)
        assert info["state"] == "prepared", info["detail"]
        assert info["meta"]["found"] is True
        # 私有标签在 data_root/.hidden/<paper_id>/（沙箱外私有目录）
        hidden = Path(info["hidden_labels_path"])
        assert str(Path(tmp_path / "data" / ".hidden" / "p-cifar")) \
            in str(hidden.parent)
        # 公开特征无标签
        for row in read_jsonl(Path(info["test_features_path"])):
            assert "label" not in row
            assert "__benchmark_id" in row
        # 契约冻结并可核验
        contract = info["contract"]
        ok, _ = verify_frozen_contract(
            contract, info["contract"]["sha256"])
        assert ok
        # manifest 落盘
        manifest = json.loads(Path(info["manifest_path"]).read_text(
            encoding="utf-8"))
        assert manifest["dataset"]["registry_found"] is True
        assert manifest["leakage_report"]["test_contains_labels"] is False

    def test_prepare_percent_dataset_small_sample(self, tmp_path):
        rm = self._rm(tmp_path)
        info = rm.prepare_leakage_safe_benchmark(
            paper_id="p-net", dataset_name="ImageNet-1k")
        assert info["state"] == "prepared", info["detail"]
        detail = info["detail"]
        assert "26 行" in detail          # percent:1 -> 24+2

    def test_prepare_with_custom_hidden_root(self, tmp_path):
        rm = self._rm(tmp_path)
        secret = tmp_path / "secret-store"
        info = rm.prepare_leakage_safe_benchmark(
            paper_id="p1", dataset_name="MNIST",
            hidden_root=str(secret))
        assert Path(info["hidden_labels_path"]) == \
            secret / "hidden_labels.jsonl"

    def test_prepare_invalid_strategy_fails_gracefully(self, tmp_path):
        rm = self._rm(tmp_path)
        info = rm.prepare_leakage_safe_benchmark(
            paper_id="p1", dataset_name="MNIST", strategy="magic")
        assert info["state"] == "unavailable"
        assert "magic" in info["detail"]