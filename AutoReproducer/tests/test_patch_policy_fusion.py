"""融合补丁安全策略测试：ScholarAgent 完备规则 + 兼容回归。

运行: python -m pytest tests/test_patch_policy_fusion.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.patch_policy import (
    DENIED_EXTENSIONS,
    DENIED_SEGMENT_PATTERNS,
    MAX_PATCHABLE_FILE_BYTES,
    PatchDecision,
    PatchPolicy,
)


# ---------------- 兼容回归：原白名单行为保持 ----------------

def test_legacy_editable_allowed():
    policy = PatchPolicy(editable=["src/model.py", "train.py"])
    ok, reason = policy.check("src/model.py")
    assert ok
    assert "允许" in reason
    assert policy.editable == ("src/model.py", "train.py")


def test_legacy_protected_overrides_editable():
    policy = PatchPolicy(editable=["data/train.csv", ".git/config"])
    assert policy.check("data/train.csv")[0] is False
    assert policy.check(".git/config")[0] is False


def test_legacy_unknown_rejected():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("src/other.py")[0] is False


def test_legacy_traversal_and_absolute_rejected():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("../secret.txt")[0] is False
    assert policy.check("src/../../etc/passwd")[0] is False
    assert policy.check("/etc/passwd")[0] is False
    assert policy.check("C:/Windows/system32/x.dll")[0] is False


def test_legacy_backslash_and_dot_normalized():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("src\\model.py")[0] is True
    assert policy.check("./src/model.py")[0] is True


def test_legacy_default_protected_prefixes():
    policy = PatchPolicy(editable=["anything.py"])
    for path in (".git/config", "data/train.csv", "tests/test_x.py",
                 "checkpoints/best.pt", "__pycache__/x.pyc"):
        assert policy.check(path)[0] is False, path


def test_legacy_default_allow_without_whitelist():
    """无 editable 白名单时仅受强制层约束。"""
    policy = PatchPolicy()
    assert policy.check("src/model.py")[0] is True
    assert policy.check("readme.md")[0] is True


# ---------------- 融合规则：目录段禁改模式（任意层，强制优先于白名单） ----------------

def test_segment_pattern_rejects_any_level_directory():
    policy = PatchPolicy(editable=["eval/score.py", "src/tests/x.py"])
    assert policy.check("eval/score.py")[0] is False      # 显式 editable 也被拒
    assert policy.check("src/tests/x.py")[0] is False     # 非首层目录段
    assert policy.check("project/benchmark/run.py")[0] is False


def test_segment_pattern_denied_segments_cover_spec_verifier():
    policy = PatchPolicy()
    for segment in ("eval", "evaluator", "judge", "verifier",
                    "benchmark", "spec", "dataset", "gold",
                    "ground_truth", "label", "annotation"):
        assert policy.check(f"src/{segment}/x.py")[0] is False, segment
        assert policy.check(f"src/{segment}/sub/x.py")[0] is False, segment


def test_segment_pattern_does_not_match_filename():
    """只禁目录段：文件名含 data/eval 前缀但不在禁改名单时仍可白名单放行。"""
    policy = PatchPolicy(editable=["src/data_loader.py", "run_training.py"])
    assert policy.check("src/data_loader.py")[0] is True
    assert policy.check("run_training.py")[0] is True


# ---------------- 融合规则：文件名模式 ----------------

def test_filename_stem_patterns_rejected():
    policy = PatchPolicy(editable=[
        "test_model.py", "eval_cli.py", "dataset_utils.py",
        "model_test.py", "run_evaluation.py", "verify_output.py",
    ])
    for name in ("test_model.py", "eval_cli.py", "dataset_utils.py",
                 "model_test.py", "run_evaluation.py", "verify_output.py"):
        decision = policy.decide(name)
        assert decision.allowed is False, name
        assert decision.rule == "denied_filename", name


def test_filename_segment_itself_denied():
    policy = PatchPolicy(editable=["test", "eval", "data"])
    assert policy.check("test")[0] is False
    assert policy.check("eval")[0] is False
    assert policy.check("data")[0] is False


# ---------------- 融合规则：数据/权重扩展名 ----------------

def test_denied_extensions_rejected_even_if_editable():
    policy = PatchPolicy(editable=[
        "model.pt", "weigths.safetensors", "train.csv", "labels.pkl",
        "data.npy", "db.sqlite", "predictions.jsonl", "feats.parquet",
    ])
    for name in ("model.pt", "weigths.safetensors", "train.csv", "labels.pkl",
                 "data.npy", "db.sqlite", "predictions.jsonl", "feats.parquet"):
        d = policy.decide(name)
        assert d.allowed is False, name
        assert d.rule == "denied_extension", name


def test_denied_extensions_case_insensitive():
    policy = PatchPolicy(editable=["Model.PT", "DATA.CSV"])
    assert policy.check("Model.PT")[0] is False
    assert policy.check("DATA.CSV")[0] is False


def test_normal_python_files_not_denied_by_extension():
    for ext in (".csv", ".pkl", ".pt", ".safetensors"):
        assert f".{ext.lstrip('.')}" in DENIED_EXTENSIONS
    policy = PatchPolicy(editable=["train.py", "src/model.py"])
    assert policy.check("train.py")[0] is True
    assert policy.check("src/model.py")[0] is True


# ---------------- 融合规则：单文件体积预算 ----------------

def test_size_budget_requires_workspace_and_rejects_large(tmp_path: Path):
    big = tmp_path / "big.py"
    big.write_text("x = 1" * (MAX_PATCHABLE_FILE_BYTES // 5 + 10),
                   encoding="utf-8")
    small = tmp_path / "small.py"
    small.write_text("x = 1\n", encoding="utf-8")

    policy = PatchPolicy()                    # 未绑工作区：不做体积校验
    assert policy.check("big.py")[0] is True

    bound = policy._with_workspace(str(tmp_path))
    d_big = bound.decide("big.py")
    assert d_big.allowed is False and d_big.rule == "size_budget"
    assert bound.decide("small.py").allowed is True

    # decide 的 workspace 参数同样生效
    assert policy.decide("big.py", workspace=str(tmp_path)).rule == "size_budget"


def test_size_budget_traversal_does_not_raise(tmp_path: Path):
    """体积校验不得逃出工作区根：越界路径仅判 illegal。"""
    policy = PatchPolicy(workspace=str(tmp_path))
    assert policy.check("../outside.py", workspace=str(tmp_path))[0] is False


# ---------------- 结构化决策 ----------------

def test_decision_is_dataclass_with_rule():
    policy = PatchPolicy(editable=["src/model.py"])
    d = policy.decide("src/model.py")
    assert isinstance(d, PatchDecision)
    assert d.path == "src/model.py"
    assert d.allowed is True and d.rule == "editable"
    assert d.to_dict()["rule"] == "editable"


def test_check_matches_decide_reason():
    policy = PatchPolicy(editable=["src/model.py"])
    ok, reason = policy.check("benchmark/x.py")
    assert ok is False
    assert "benchmark" in reason and "禁止" in reason


# ---------------- 批量切片 filter_patches ----------------

def test_filter_patches_splits_allowed_and_rejected():
    policy = PatchPolicy(editable=["src/model.py"])
    patches = [
        {"path": "src/model.py", "content": "new"},
        {"path": "data/train.csv", "content": "x"},   # protected
        {"path": "test_model.py", "content": "x"},    # filename
        {"path": "model.pt", "content": "x"},         # extension
    ]
    allowed, rejected = policy.filter_patches(patches)
    assert [p["path"] for p in allowed] == ["src/model.py"]
    assert len(rejected) == 3
    assert all(not d.allowed for d in rejected)
    assert {d.rule for d in rejected} == {"protected", "denied_filename",
                                          "denied_extension"}


def test_filter_patches_dedupes_and_limits_candidate_files():
    policy = PatchPolicy()
    patches = [{"path": f"src/model{i}.py", "content": "x"} for i in range(5)]
    allowed, rejected = policy.filter_patches(patches)
    assert len(allowed) == 3                    # ≤ MAX_PATCH_FILES_PER_CANDIDATE
    assert len({p["path"] for p in allowed}) == 3

    dup = [{"path": "src/a.py"}, {"path": "src/a.py"}, {"path": "src/b.py"}]
    allowed2, _ = policy.filter_patches(dup)
    assert [p["path"] for p in allowed2] == ["src/a.py", "src/b.py"]


# ---------------- 常量完整性（迁移自 ScholarAgent 的关键清单） ----------------

def test_denied_segment_patterns_cover_core_protected_dirs():
    for key in ("eval", "test", "data", "benchmark", "label",
                "gold", "verifier", "spec"):
        assert key in DENIED_SEGMENT_PATTERNS, key


def test_denied_extensions_cover_data_and_weight_formats():
    for key in (".csv", ".jsonl", ".parquet", ".pkl", ".pt", ".pth",
                ".ckpt", ".safetensors", ".bin", ".npy"):
        assert key in DENIED_EXTENSIONS, key