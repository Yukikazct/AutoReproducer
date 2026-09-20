"""数据集注册表与复现级别数据策略测试（P1-2）。

覆盖：
1. 注册表别名归一化（cifar-10 / CIFAR10 / cifar_10 -> CIFAR-10）；
2. 未知数据集返回 None（ResourceManager 将降级合成冒烟集）；
3. 超大数据集（ImageNet）percent:N 子集策略 + HF 入口 + 体积预估；
4. 零数据合成标记（synthetic / 前缀自动归类）；
5. ResourceManager 默认自动启用注册表；
6. fetch_dataset 按 kind 决策：torchvision -> 懒加载不预下载；
   synthetic -> 零数据；未知 -> 冒烟降级；
7. url 入口真实下载（mock curl）；hf 入口不可用 -> 冒烟降级。

运行: python -m pytest tests/test_dataset_registry.py -v
"""
import builtins
import sys
from pathlib import Path

import pytest

sys_path_fix = Path(__file__).parent.parent
if str(sys_path_fix) not in sys.path:
    sys.path.insert(0, str(sys_path_fix))

from src.dataset_registry import DatasetRegistry  # noqa: E402
from src.resource_manager import ResourceManager  # noqa: E402


@pytest.fixture()
def registry():
    return DatasetRegistry()


@pytest.fixture()
def mgr(tmp_path):
    return ResourceManager(data_root=str(tmp_path / "data"),
                           quota_bytes=1024 * 1024)


# ---------------- 1. 别名归一化 ----------------

def test_alias_normalization(registry):
    for name in ("CIFAR-10", "cifar10", " CIFAR10 ",
                 "cifar_10", "cifar-10"):
        meta = registry.lookup(name)
        assert meta is not None
        assert meta["name"] == "CIFAR-10"
        assert meta["torchvision_builtin"] is True
        assert meta["synthetic"] is False


def test_mnist_alias(registry):
    meta = registry.lookup("MNIST")
    assert meta is not None
    assert meta["kind"] == "torchvision"
    assert meta["subset"] == "lazy"
    assert meta["size_gb"] == pytest.approx(0.011, abs=1e-3)


def test_unknown_returns_none(registry):
    assert registry.lookup("some-random-dataset-xyz") is None
    assert registry.lookup("") is None
    assert registry.lookup("  ") is None


# ---------------- 2. 大数据集子集策略 ----------------

def test_imagenet_subset_strategy(registry):
    meta = registry.lookup("ImageNet-1k")
    assert meta["kind"] == "huggingface"
    assert meta["subset"] == "percent:1"      # 1% 降采样验证
    assert meta["entry"].startswith("hf:")
    assert meta["size_gb"] == pytest.approx(150.0)
    assert "流程验证" in meta["reason"] or "非数值" in meta["reason"]


def test_coco_subset_strategy(registry):
    meta = registry.lookup("coco2017")
    assert meta["kind"] == "huggingface"
    assert meta["subset"] == "percent:5"


def test_small_nlp_full(registry):
    meta = registry.lookup("SQuAD v2")
    assert meta["subset"] == "full"


# ---------------- 3. 零数据合成标记 ----------------

def test_synthetic_marker(registry):
    for name in ("synthetic", "generated", "合成数据", "生成数据", "模拟数据"):
        meta = registry.lookup(name)
        assert meta is not None
        assert meta["synthetic"] is True
        assert meta["size_gb"] == 0.0


def test_synthetic_prefix_autoclassify(registry):
    # 未知名称以合成词打头 -> 自动归类为运行时可合成（零数据）
    meta = registry.lookup("synthetic-sine-wave")
    assert meta is not None
    assert meta["synthetic"] is True


def test_registered_and_estimate(registry):
    names = registry.registered()
    assert len(names) >= 10
    assert registry.estimate_size_gb("MNIST") > 0
    assert registry.estimate_size_gb("no-such-dataset") == 0.0


# ---------------- 4. ResourceManager 默认接线 ----------------

def test_default_registry_enabled(mgr):
    assert mgr.dataset_registry is not None


def test_smoke_path_unchanged_with_registry(mgr):
    pid = "reg000000001"
    info = mgr.fetch_dataset(pid, "CIFAR-10", level="smoke")
    assert info["state"] == "smoke-synth"
    assert "meta" in info
    assert info["meta"]["name"] == "CIFAR-10"


# ---------------- 5. fetch_dataset kind 决策 ----------------

def test_full_torchvision_lazy_no_download(mgr):
    pid = "reg000000002"
    info = mgr.fetch_dataset(pid, "CIFAR-10", level="full")
    assert info["state"] == "lazy-torchvision"
    assert info["path"] == ""
    assert info["level"] == "full"
    # 不预下载：dataset_full / dataset_smoke 均不应产生
    ds_dir = mgr._dataset_dir(pid)
    assert not (ds_dir / "dataset_full").exists()
    assert not (ds_dir / "dataset_smoke").exists()


def test_full_synthetic_zero_data(mgr):
    pid = "reg000000003"
    info = mgr.fetch_dataset(pid, "synthetic", level="full")
    assert info["state"] == "synthetic"
    assert info["path"] == ""
    assert info["level"] == "full"


def test_full_unknown_falls_back_smoke(mgr):
    pid = "reg000000004"
    info = mgr.fetch_dataset(pid, "my-weird-dataset-xyz", level="full")
    assert info["state"] == "smoke-synth"
    assert info["rows"] > 0


# ---------------- 6. url 入口真实下载 ----------------

def test_full_url_entry_downloads(mgr, monkeypatch, tmp_path):
    pid = "reg000000005"
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # curl -L --fail -sS -o <target> <url>：在 -o 处落地文件
        args = [str(c) for c in cmd]
        if "-o" in args:
            target = Path(args[args.index("-o") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        return type("P", (), {"returncode": 0, "stdout": "",
                              "stderr": ""})()

    monkeypatch.setattr("src.resource_manager.subprocess.run", fake_run)
    info = mgr.fetch_dataset(pid, "Tiny ImageNet", level="full")
    assert info["state"] == "downloaded"
    assert info["level"] == "full"
    assert any("tiny-imagenet-200.zip" in " ".join(map(str, c)) for c in calls)


def test_url_download_failure_falls_back_smoke(mgr, monkeypatch):
    pid = "reg000000006"

    def fake_run(cmd, **kw):
        return type("P", (), {"returncode": 1, "stdout": "",
                              "stderr": "404 not found"})()

    monkeypatch.setattr("src.resource_manager.subprocess.run", fake_run)
    info = mgr.fetch_dataset(pid, "Tiny ImageNet", level="full")
    assert info["state"] == "smoke-synth"      # 诚实降级，不伪造大文件
    assert "降级合成冒烟集" in info["detail"]


# ---------------- 7. hf 入口不可用 -> 冒烟降级 ----------------

def test_hf_entry_unavailable_falls_back_smoke(mgr, monkeypatch):
    pid = "reg000000007"

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub blocked for test")
        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    info = mgr.fetch_dataset(pid, "ImageNet-1k", level="full")
    assert info["state"] == "smoke-synth"
    assert "ImageNet" in info["detail"] or "1%" in info["detail"]


def test_disabled_registry_still_smoke(mgr):
    """显式传 False 关闭注册表：一切数据集走冒烟降级。"""
    import tempfile
    rm = ResourceManager(data_root=str(tempfile.mkdtemp()),
                         dataset_registry=False)
    info = rm.fetch_dataset("reg000000008", "CIFAR-10", level="full")
    assert info["state"] == "smoke-synth"