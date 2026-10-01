"""DatasetRegistry - 数据集注册表（P1-2：复现级别数据策略）。

对齐方案「按需懒加载 + 体积瘦身 + 镜像级共享」中的数据侧决策点：
- **别名归一化**：cifar-10 / CIFAR10 / cifar_10 均可命中 CIFAR-10；
- **体积预估 size_gb**：fetch 前即可用于 L0 配额规划与选型决策
  （ImageNet 150GB vs CIFAR-10 170MB -> 优先小数据集的依据）；
- **子集策略 subset**：
  - lazy：torchvision 等内建数据集，训练代码运行时时懒加载，
    预下载反而浪费磁盘 -> 不拉；
  - full：体积可控，可直接全量；
  - percent:N：超大数据集降采样 N% 验证流程趋势（语义为
    "流程验证"而非"数值复现"，与方案 L1 冒烟级别一致）；
  - synthetic：零数据，训练代码运行时合成（PINN/GAN 等），
    天然体积为 0——体积瘦身的最优形态；
- **国内镜像 mirror**：ModelScope / 镜像站备注，受限网络优先走镜像；
- **下载入口 entry**：""（内建/合成，无需入口）/ url:<file> /
  hf:<repo_id>[/subpath]（HuggingFace 子集）/ script:<cmd>；
- **未知数据集返回 None**：ResourceManager 降级为本地合成冒烟集，
  绝不静默伪造大文件。

体积与镜像信息为公开常识量级（近似值，仅供规划），
不替代真实下载时的磁盘核对。
"""
import re
from typing import Dict, Optional

# ---------------- 注册表数据 ----------------
# kind: torchvision=内建懒加载 / huggingface=HF 子集 / url=直链单文件 /
#       synthetic=运行时合成零数据
# subset: lazy / full / percent:N / synthetic

KNOWN_DATASETS: tuple = (
    # ---- 视觉：torchvision 内建（训练时懒加载，不预下载） ----
    {
        "name": "MNIST",
        "aliases": ("mnist", "minist", "mnist手写数字"),
        "size_gb": 0.011,
        "kind": "torchvision",
        "entry": "",
        "mirror": "torchvision 自动下载，失败可设 HF_ENDPOINT 镜像",
        "subset": "lazy",
        "reason": "torchvision 内建，训练代码运行时懒加载，无需预下载",
    },
    {
        "name": "Fashion-MNIST",
        "aliases": ("fashionmnist", "fashion-mnist", "fashion"),
        "size_gb": 0.03,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上",
        "subset": "lazy",
        "reason": "torchvision 内建，训练时懒加载",
    },
    {
        "name": "EMNIST",
        "aliases": ("emnist",),
        "size_gb": 0.15,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上",
        "subset": "lazy",
        "reason": "torchvision 内建，训练时懒加载",
    },
    {
        "name": "CIFAR-10",
        "aliases": ("cifar10", "cifar", "cifar-10", "cifar_10"),
        "size_gb": 0.17,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上；也可用 ModelScope 镜像数据集",
        "subset": "lazy",
        "reason": "仅 170MB；torchvision 内建懒加载，全量也可接受",
    },
    {
        "name": "CIFAR-100",
        "aliases": ("cifar100", "cifar-100", "cifar_100"),
        "size_gb": 0.17,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上",
        "subset": "lazy",
        "reason": "仅 170MB；torchvision 内建懒加载",
    },
    {
        "name": "SVHN",
        "aliases": ("svhn",),
        "size_gb": 1.8,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上",
        "subset": "lazy",
        "reason": "torchvision 内建懒加载",
    },
    {
        "name": "STL-10",
        "aliases": ("stl10", "stl-10"),
        "size_gb": 2.6,
        "kind": "torchvision",
        "entry": "",
        "mirror": "同上",
        "subset": "lazy",
        "reason": "torchvision 内建懒加载",
    },
    # ---- 视觉：直链 / HF（需要子集策略） ----
    {
        "name": "Tiny ImageNet",
        "aliases": ("tinyimagenet", "tiny-imagenet", "tiny-imagenet-200"),
        "size_gb": 0.24,
        "kind": "url",
        "entry": "url:http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "mirror": "暂无稳定国内镜像，可搜索 ModelScope 重传仓库",
        "subset": "full",
        "reason": "237MB 全量可接受；冒烟级走 1% 采样验证流程",
    },
    {
        "name": "ImageNet-1k",
        "aliases": ("imagenet", "imagenet1k", "imagenet-1k", "imagenet-1k-2012"),
        "size_gb": 150.0,
        "kind": "huggingface",
        "entry": "hf:ILSVRC/imagenet-1k",
        "mirror": "需官方授权；国内走学术数据集镜像站或内网缓存",
        "subset": "percent:1",
        "reason": "150GB 全量仅 L3 深复现；L1/L2 用 1% 降采样验证流程"
                  "趋势，语义为流程验证而非数值复现",
    },
    {
        "name": "COCO 2017",
        "aliases": ("coco", "coco2017", "coco-2017", "mscoco"),
        "size_gb": 25.0,
        "kind": "huggingface",
        "entry": "hf:detection-datasets/coco",
        "mirror": "国内多按需取 annotations + 抽样图，避免全量 19GB 图片",
        "subset": "percent:5",
        "reason": "25GB 图片全量仅 L3；趋势验证用 5% 图像子集",
    },
    # ---- NLP：HF（体积小，可全量） ----
    {
        "name": "SQuAD v1",
        "aliases": ("squad", "squadv1", "squad-v1"),
        "size_gb": 0.04,
        "kind": "huggingface",
        "entry": "hf:stanfordnlp/squad",
        "mirror": "ModelScope: modelscope.cn/datasets 有重传",
        "subset": "full",
        "reason": "约 40MB，全量无压力",
    },
    {
        "name": "SQuAD v2",
        "aliases": ("squad2", "squadv2", "squad-v2"),
        "size_gb": 0.05,
        "kind": "huggingface",
        "entry": "hf:stanfordnlp/squad_v2",
        "mirror": "同上",
        "subset": "full",
        "reason": "约 50MB，全量无压力",
    },
    {
        "name": "IMDb",
        "aliases": ("imdb",),
        "size_gb": 0.08,
        "kind": "huggingface",
        "entry": "hf:stanfordnlp/imdb",
        "mirror": "同上",
        "subset": "full",
        "reason": "约 80MB，全量无压力",
    },
    {
        "name": "AG News",
        "aliases": ("agnews", "ag-news", "ag_news"),
        "size_gb": 0.12,
        "kind": "huggingface",
        "entry": "hf:fancyzhx/ag_news",
        "mirror": "同上",
        "subset": "full",
        "reason": "约 120MB，全量无压力",
    },
    {
        "name": "GLUE",
        "aliases": ("glue",),
        "size_gb": 0.1,
        "kind": "huggingface",
        "entry": "hf:glue",
        "mirror": "同上",
        "subset": "full",
        "reason": "基准子任务合计约百 MB，按子任务取",
    },
    {
        "name": "WikiText-103",
        "aliases": ("wikitext103", "wikitext-103", "wt103"),
        "size_gb": 0.5,
        "kind": "huggingface",
        "entry": "hf:wikitext",
        "mirror": "ModelScope 有重传",
        "subset": "percent:10",
        "reason": "约 500MB；语言建模冒烟可 10% 采样",
    },
    # ---- 零数据：运行时合成（体积瘦身最优形态） ----
    {
        "name": "SYNTHETIC-DATA",
        "aliases": ("synthetic", "generated", "synthesized",
                    "self-synthesized", "self-generated",
                    "合成数据", "生成数据", "模拟数据"),
        "size_gb": 0.0,
        "kind": "synthetic",
        "entry": "",
        "mirror": "无（本地生成）",
        "subset": "synthetic",
        "reason": "零数据：训练代码运行时合成（PINN/GAN/仿真等），"
                  "无需下载，体积为 0",
    },
)

# 归一化别名 -> 条目
_ALIAS_INDEX: Dict[str, dict] = {}


def _normalize(name: str) -> str:
    """别名归一化：小写 + 去空白 + 去非单词字符（保留中文）。"""
    name = (name or "").strip().lower().replace(" ", "")
    return re.sub(r"[\W_]+", "", name, flags=re.UNICODE)


def _build_index() -> None:
    for entry in KNOWN_DATASETS:
        for alias in entry["aliases"]:
            key = _normalize(alias)
            if key:
                _ALIAS_INDEX.setdefault(key, entry)


_build_index()

# 前缀自动归类：名称以这些词打头的未知数据集视为"运行时可合成"
_SYNTHETIC_PREFIXES = ("synthetic", "generated", "synth", "self-gen")


class DatasetRegistry:
    """数据集注册表：别名解析 + 子集策略 + 体积预估 + 国内镜像。"""

    def lookup(self, dataset_name: str) -> Optional[Dict]:
        """归一化查表；未知数据集（含合成前缀未命中）返回 None。

        返回（命中时）：{"name","size_gb","kind","entry","mirror",
        "subset","reason","torchvision_builtin","synthetic"}。
        """
        key = _normalize(dataset_name)
        if not key:
            return None
        entry = _ALIAS_INDEX.get(key)
        if entry is None and key.startswith(_SYNTHETIC_PREFIXES):
            entry = _ALIAS_INDEX.get("synthetic")
        if entry is None:
            return None
        return {
            "name": entry["name"],
            "size_gb": entry["size_gb"],
            "kind": entry["kind"],
            "entry": entry["entry"],
            "mirror": entry["mirror"],
            "subset": entry["subset"],
            "reason": entry["reason"],
            "torchvision_builtin": entry["kind"] == "torchvision",
            "synthetic": entry["kind"] == "synthetic",
        }

    def registered(self) -> list:
        """注册表一览：以精简字段返回，供 CLI status 文档展示。"""
        return [
            {"name": e["name"], "size_gb": e["size_gb"],
             "kind": e["kind"], "subset": e["subset"]}
            for e in KNOWN_DATASETS
        ]

    def estimate_size_gb(self, dataset_name: str) -> float:
        """体积预估；未知数据集返回 0.0（将由合成冒烟集兜底）。"""
        meta = self.lookup(dataset_name)
        return meta["size_gb"] if meta else 0.0

    def benchmark_hint(self, dataset_name: str) -> Dict:
        """防泄漏 Benchmark 评测接入点（P1-⑩）。

        返回该数据集的评测采样建议与元信息，供
        ResourceManager.prepare_leakage_safe_benchmark 决策：
        - synthetic/零数据：评测链路用小样本（24 行）即可；
        - percent:N 超大真实集：N 成比例小样本（24 + 2N），
          语义为"链路验证"而非"数值复现"；
        - 其他（lazy/full 小数据）：默认 1000 行上限。
        未知数据集返回 found=False + 默认建议（合成存根可用）。
        """
        meta = self.lookup(dataset_name)
        if meta is None:
            return {"found": False, "dataset_name": dataset_name,
                    "kind": "unknown", "subset": "unknown",
                    "size_gb": 0.0, "recommended_max_samples": 1000}
        subset = meta.get("subset", "")
        if subset == "synthetic":
            recommended = 24
        elif subset.startswith("percent:"):
            try:
                percent = int(subset.split(":", 1)[1])
            except (TypeError, ValueError):
                percent = 1
            recommended = 24 + 2 * percent
        else:
            recommended = 1000
        return {
            "found": True,
            "dataset_name": meta.get("name", dataset_name),
            "kind": meta.get("kind", ""),
            "subset": subset,
            "size_gb": meta.get("size_gb", 0.0),
            "mirror": meta.get("mirror", ""),
            "recommended_max_samples": recommended,
        }