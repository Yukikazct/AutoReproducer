"""Frozen paper experiment: ReZero CIFAR-10 superconvergence, not enwiki8."""
from copy import deepcopy

from src.experiments.rezero_runtime import REFERENCE_PARAMETERS
from src.rezero_adapter import SOURCE_SHA256


REZERO_PROFILE_ID = "rezero_cifar10_reference"
REZERO_LABELS = {REZERO_PROFILE_ID: "ReZero · CIFAR-10 · 论文完整实验（45 轮）"}
REZERO_REPOSITORY = "https://github.com/tbachlechner/ReZero-Superconvergence"
REZERO_REVISION = "6c0212669ac8c23d3db6f2b99255bcf6e3c5e6e6"
# Full canonical archive, verified against its 170498071-byte size and official MD5.
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
REZERO_REQUIREMENTS = (
    "--extra-index-url https://download.pytorch.org/whl/cu121\n"
    "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\n"
    "numpy==1.26.4\nscipy==1.14.1\nmatplotlib==3.9.2\nPillow==10.4.0\n"
)


def rezero_profile(profile_id):
    if profile_id not in REZERO_LABELS:
        raise ValueError(f"未适配的 ReZero 论文预设: {profile_id}")
    reference = f"{REZERO_REPOSITORY}/blob/{REZERO_REVISION}/Faster_SuperC.ipynb"
    return deepcopy({
        "version": 1, "id": profile_id, "adapter_id": "rezero", "label": REZERO_LABELS[profile_id],
        "paper": {
            "title": "ReZero is All You Need: Fast Convergence at Large Depth",
            "url": "https://arxiv.org/abs/2003.04887",
            "method": "ReZero PreActResNet18 superconvergence", "dataset": "CIFAR-10",
            "metrics": {"top1_accuracy_pct": 94.0},
            "required_metrics": ["top1_accuracy_pct", "cross_entropy"],
            "reference_source": reference,
            "experiment_locator": "arXiv:2003.04887v2 §5 / Appendix E.2; published paper §5.2",
        },
        "repository": {"url": REZERO_REPOSITORY, "revision": REZERO_REVISION},
        "source_sha256s": SOURCE_SHA256,
        "required_files": ["README.md", "models/__init__.py", "models/fixup_resnet_cifar.py",
                           "models/resnet_cifar.py", "models/rezero_resnet_cifar.py",
                           "models/rezero_dpn.py", "models/dpn.py", "models/preact_resnet.py",
                           *SOURCE_SHA256],
        "dataset": {
            "name": "cifar10_official", "kind": "real",
            "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "mirrors": ["https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"],
            "sha256": CIFAR10_SHA256, "md5": "c58f30108f718f92721af3b95e74349a",
            "bytes": 170498071, "target": "cifar-10-python.tar.gz",
            "train_samples": 50000, "test_samples": 10000,
            "split": "官方完整 50,000 张训练集 / 10,000 张测试集；按作者协议选择 45 轮内测试准确率最高的 checkpoint",
        },
        "parameters": REFERENCE_PARAMETERS,
        "environment": {
            "requirements_txt": REZERO_REQUIREMENTS,
            "note": "Python 3.12、PyTorch 2.5.1、CUDA 12.1 兼容环境；论文使用 PyTorch 1.2。固定作者模型和学习率调度器，完整 FP32、batch 512 训练。",
        },
        "validation": {
            "level": "reference", "scope": "selected_paper_experiment",
            "target_accuracy_pct": 94.0, "relative_tolerance": 0.0,
            "reference_aggregation": "single_seed", "aggregation_source": reference,
            "implementation_note": "保留作者模型、初始化、增强、SGD/Adagrad 分组与 OneCycleLR；适配 Windows 进程入口、现代依赖、受控数据准备和独立评估。",
            "note": "仅复现 ReZero CIFAR-10 超收敛实验：seed 6892、完整 45 轮/4410 步；独立重载作者协议选出的 checkpoint，在完整测试集测得 top-1 ≥94.00% 才通过。该结论不覆盖 enwiki8 或整篇论文。",
        },
        "optimization_modes": ["off"], "search_space": {},
        "budget": {"total_s": 8100, "baseline_s": 7800, "advice_s": 0},
    })
