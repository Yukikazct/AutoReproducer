"""ResourceManager - L0 热缓存资源管理（对齐方案「三层存储」）。

职责：
- 懒加载：不预下载全部资源，fetch_code / fetch_dataset / fetch_weights
  只拉当前任务最小集；重复 fetch 幂等复用已有缓存，绝不重复下载；
- 统一缓存根：代码 data/repos/<paper_id>/、数据集 data/datasets/<paper_id>/、
  依赖 data/deps/<hash>/（CodeExecutor 写入）、清单 data/manifests/<paper_id>.json；
- manifest：生成 / 查询 / 列出，字段与 2026-09-09 存量格式兼容
  （paper_id / created_at / resources{code,dataset,weights} / paper_title /
  code_url / dataset_name / cleaned_at）；
- 清理与归档：cleanup 清理 L0 条目并在 manifest 记录 cleaned_at；
  archive / restore 与 L1 温存储（移动硬盘 / NAS）对接；
- L0 配额守护：AUTOREPRO_L0_QUOTA_GB 可配（默认 20GB），
  enforce_quota 超限时按最近使用排序返回建议归档清单（不自动删除）；
- 统计：stats() 输出按缓存根 / 按论文聚合的占用与条目数。

所有真实网络下载（git clone / HuggingFace snapshot / ModelScope / 单文件
URL）均为"尽力而为"：执行器或网络不可用时诚实降级为本地最小冒烟集
（dataset_smoke），并在返回 info 的 state 字段标注实际状态，绝不静默伪造大文件。
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# src/resource_manager.py -> parents[1] 为仓库内 AutoReproducer 包根
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 统一缓存根（data 之下，与存量 manifest 绝对路径一致）
DATA_ROOT = Path(os.environ.get("AUTOREPRO_DATA_ROOT", str(_PROJECT_ROOT / "data")))
REPOS_ROOT = DATA_ROOT / "repos"          # 代码仓库：data/repos/<paper_id>/
DATASETS_ROOT = DATA_ROOT / "datasets"    # 数据集：data/datasets/<paper_id>/
MANIFESTS_ROOT = DATA_ROOT / "manifests"  # 资源清单：data/manifests/<paper_id>.json
ARCHIVE_ROOT = DATA_ROOT / "archive"      # L1 归档暂存：data/archive/<paper_id>.zip

# L0 配额（字节），AUTOREPRO_L0_QUOTA_GB 环境变量可配，默认 20GB
_DEFAULT_QUOTA_GB = 20
_L0_QUOTA = int(float(os.environ.get(
    "AUTOREPRO_L0_QUOTA_GB", _DEFAULT_QUOTA_GB)) * 1024 ** 3)

# 冒烟数据集最小样本行数（合成数据，用于离线验证流程而非数值复现）
_SMOKE_ROWS = 8

# 明显是占位符的代码仓库 URL（存量测试语料常用），不应真实克隆：
# host 为 example.com/example.org，或路径含 /example/
_PLACEHOLDER_HOSTS = ("example.com", "example.org")


def _is_placeholder_url(url: str) -> bool:
    if any(h in url for h in _PLACEHOLDER_HOSTS):
        return True
    path = url.split("//", 1)[-1].split("/", 1)
    return len(path) == 2 and path[1].startswith("example/")


def _fmt_bytes(n: float) -> str:
    """字节 -> 人类可读（B/KB/MB/GB）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _dir_bytes(path: Path) -> int:
    """递归统计目录总字节数；不存在或不可读返回 0。"""
    if not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _git_available() -> bool:
    return shutil.which("git") is not None


class ResourceManager:
    """论文资源（代码 / 数据集 / 权重）的懒加载与 L0 缓存管理。

    dataset_registry=None 时自动启用默认 DatasetRegistry（P1-2）；
    显式传 False 可关闭注册表（未知数据集一律降级合成冒烟集）。
    接入后按注册表决定真实子集入口与体积预估（fetch 前即可配额决策）。
    """

    def __init__(self, data_root: Optional[str] = None,
                 quota_bytes: Optional[int] = None,
                 dataset_registry: Optional[Any] = None,
                 logger: Optional[Any] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.repos_root = self.data_root / "repos"
        self.datasets_root = self.data_root / "datasets"
        self.manifests_root = self.data_root / "manifests"
        self.archive_root = self.data_root / "archive"
        self.quota_bytes = quota_bytes or _L0_QUOTA
        if dataset_registry is None:
            from src.dataset_registry import DatasetRegistry
            dataset_registry = DatasetRegistry()
        self.dataset_registry = dataset_registry
        self.logger = logger
        for d in (self.repos_root, self.datasets_root,
                  self.manifests_root, self.archive_root):
            os.makedirs(d, exist_ok=True)

    # ---------------- 基础工具 ----------------

    def _log(self, event: str, status: str, detail: str = "") -> None:
        if self.logger is not None and hasattr(self.logger, "log"):
            try:
                self.logger.log(event, status, detail)
            except Exception:
                pass

    @staticmethod
    def paper_id_for(paper_title: str, corpus_key: str = "") -> str:
        """论文稳定 ID：corpus 语料键直接可用，否则 sha1(title) 前 12 位
        （与存量 manifest 的 12 位十六进制 paper_id 风格一致）。"""
        if corpus_key:
            return corpus_key
        digest = hashlib.sha1(paper_title.encode("utf-8")).hexdigest()
        return digest[:12]

    def _repo_dir(self, paper_id: str) -> Path:
        return self.repos_root / paper_id

    def _dataset_dir(self, paper_id: str) -> Path:
        return self.datasets_root / paper_id

    def _manifest_path(self, paper_id: str) -> Path:
        return self.manifests_root / f"{paper_id}.json"

    # ---------------- 懒加载：代码 ----------------

    def fetch_code(self, paper_id: str, code_url: str,
                   target: Optional[str] = None,
                   revision: str = "") -> Dict:
        """拉取论文代码仓库到 L0；已存在则幂等复用。

        revision 非空时，clone 后尝试浅拉取并 detach 到该 revision
        （commit sha / tag）；失败仅降级为 HEAD 并标注，不阻断。
        成功克隆/复用后写入源标记 .autorepro-repo-source.json
        （repo_url / commit / acquisition），供证据链与报告溯源。

        返回 {"path": 本地目录或"" , "state": 状态, "detail": 说明,
              "commit": 当前 HEAD sha, "revision": 请求的 pin}。
        state 取值：cached（已有缓存）/ cloned（新克隆）/ placeholder-skip
        （占位 URL 不下载）/ clone-failed（下载失败）。
        """
        repo_dir = Path(target) if target else self._repo_dir(paper_id)
        info: Dict = {"path": "", "state": "skipped", "detail": "",
                      "commit": "", "revision": revision or ""}
        url = (code_url or "").strip()
        if not url:
            info["detail"] = "无代码仓库 URL"
            return info
        if repo_dir.exists() and any(repo_dir.iterdir()):
            info.update(path=str(repo_dir), state="cached",
                        detail="缓存命中，复用已有仓库")
            self._record_repo_source(repo_dir, url, info, "cached")
            return info
        if _is_placeholder_url(url):
            info.update(state="placeholder-skip",
                        detail=f"占位 URL 不下载: {url}")
            return info
        if not _git_available():
            info["detail"] = "git 不可用，跳过代码下载"
            return info
        try:
            repo_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch",
                 url, str(repo_dir)],
                capture_output=True, text=True, timeout=600)
            if proc.returncode == 0:
                info.update(path=str(repo_dir), state="cloned",
                            detail=f"克隆成功: {url}")
                self._pin_and_record(repo_dir, url, info, revision)
            else:
                info.update(state="clone-failed",
                            detail=(proc.stderr or proc.stdout).strip()[-300:])
        except Exception as exc:       # 网络 / 超时 / git 异常
            info.update(state="clone-failed", detail=str(exc)[-300:])
        return info

    @staticmethod
    def _git_head_commit(repo_dir: Path) -> str:
        """返回仓库当前 HEAD 的 commit sha；非 git 仓库返回空串。"""
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15)
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def _pin_and_record(self, repo_dir: Path, url: str, info: Dict,
                        revision: str) -> None:
        """pin revision（尽力而为）+ 写入源标记 + 填充 commit。"""
        info["commit"] = self._git_head_commit(repo_dir)
        if revision:
            try:
                fetch = subprocess.run(
                    ["git", "-C", str(repo_dir), "fetch", "--depth", "1",
                     "origin", revision],
                    capture_output=True, text=True, timeout=120)
                checkout = subprocess.run(
                    ["git", "-C", str(repo_dir), "checkout", "--detach",
                     revision],
                    capture_output=True, text=True, timeout=60)
                if fetch.returncode == 0 and checkout.returncode == 0:
                    info["commit"] = self._git_head_commit(repo_dir)
                    info["detail"] += f"; pinned revision {revision}"
                else:
                    info["detail"] += (f"; pin 失败({revision})，"
                                       "保留 HEAD")
            except (OSError, subprocess.SubprocessError):
                info["detail"] += f"; pin 失败({revision})，保留 HEAD"
        self._record_repo_source(repo_dir, url, info, info.get("state", ""))

    def _record_repo_source(self, repo_dir: Path, url: str, info: Dict,
                            acquisition: str) -> None:
        """写入仓库溯源标记（供证据链 / 报告引用，失败静默）。"""
        try:
            marker = {
                "repo_url": url,
                "commit": info.get("commit", ""),
                "acquisition": acquisition,
                "revision": info.get("revision", ""),
            }
            (repo_dir / ".autorepro-repo-source.json").write_text(
                json.dumps(marker, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError:
            pass

    # ---------------- 懒加载：数据集 ----------------

    def fetch_dataset(self, paper_id: str, dataset_name: str,
                      level: str = "smoke",
                      target: Optional[str] = None) -> Dict:
        """拉取论文数据集到 L0，按复现级别决定子集规模。

        level=smoke 只取最小冒烟集（合成样本，验证流程不验证数值）；
        level=full 时优先走 dataset_registry 的真实下载入口，
        注册表缺失或下载失败则同样降级为冒烟集 + 标注。

        返回 {"path", "state", "detail", "level", "rows"}。
        state：cached / smoke-synth（合成冒烟集）/ downloaded（真实子集）/
        unavailable。
        """
        ds_dir = Path(target) if target else self._dataset_dir(paper_id)
        smoke_dir = ds_dir / "dataset_smoke"
        info: Dict = {"path": "", "state": "skipped",
                      "detail": "", "level": level, "rows": 0}
        name = (dataset_name or "").strip()
        if not name:
            info["detail"] = "无数据集名称"
            return info
        if smoke_dir.exists() and any(smoke_dir.iterdir()):
            info.update(path=str(smoke_dir), state="cached",
                        level="smoke", detail="冒烟集缓存命中",
                        rows=_SMOKE_ROWS)
            return info

        meta = self._registry_lookup(name)
        if meta:
            info["meta"] = {k: meta.get(k) for k in (
                "name", "size_gb", "kind", "subset", "mirror", "reason")}
        if level == "full" and meta:
            # 注册表子集策略：torchvision 内建 -> 训练时懒加载不预下载；
            # 零数据合成 -> 无需下载；有真实入口 -> 按入口拉子集。
            if meta.get("torchvision_builtin"):
                info.update(path="", state="lazy-torchvision",
                            level="full",
                            detail=f"{meta.get('name')} 为 torchvision 内建"
                                   "数据集，训练代码运行时懒加载，无需预下载"
                                   "（避免重复占用 L0）")
                return info
            if meta.get("synthetic"):
                info.update(path="", state="synthetic", level="full",
                            detail=f"{meta.get('name')} 零数据：训练代码运行时"
                                   "合成，无需下载（体积为 0）")
                return info
            if meta.get("entry"):
                result = self._download_real_dataset(
                    paper_id, name, meta, ds_dir)
                if result["state"] == "downloaded":
                    info.update(result)
                    info["path"] = str(ds_dir)
                    return info

        # 降级：合成最小冒烟集（离线验证流程）
        try:
            os.makedirs(smoke_dir, exist_ok=True)
            self._write_smoke_dataset(smoke_dir, name)
            info.update(path=str(smoke_dir), state="smoke-synth",
                        level="smoke",
                        detail=("本地合成冒烟集（离线流程验证，非数值复现）"
                                if not meta else
                                f"真实数据源不可用，降级合成冒烟集: "
                                f"{meta.get('reason', '')}"),
                        rows=_SMOKE_ROWS)
        except Exception as exc:
            info.update(state="unavailable", detail=str(exc)[-300:])
        return info

    def _registry_lookup(self, name: str) -> Optional[Dict]:
        if self.dataset_registry is None:
            return None
        lookup = getattr(self.dataset_registry, "lookup", None)
        if callable(lookup):
            try:
                result = lookup(name)
                return result if isinstance(result, dict) else None
            except Exception:
                return None
        return None

    def _download_real_dataset(self, paper_id: str, name: str,
                               meta: Dict, ds_dir: Path) -> Dict:
        """按注册表真实下载数据集子集；失败返回 unavailable 状态。"""
        entry = meta.get("entry", "")
        try:
            os.makedirs(ds_dir, exist_ok=True)
            if entry.startswith("hf:"):
                # HuggingFace 按需子集：仅拉数据文件通配形，不拉全仓
                # （对齐 ImageNet/COCO 等超大数据的 percent:N 子集策略，
                # 语义为流程验证而非数值复现）
                from huggingface_hub import snapshot_download  # 延迟导入
                repo = entry[3:].strip("/").split("/", 1)
                repo_id = f"{repo[0]}/{repo[1]}" if len(repo) > 1 else repo[0]
                target_dir = ds_dir / "dataset_full"
                os.makedirs(target_dir, exist_ok=True)
                local = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["*.csv", "*.jsonl", "*.parquet",
                                    "*.txt", "*.json", "*.zip"],
                    local_dir=str(target_dir))
                return {"state": "downloaded",
                        "detail": f"HF 数据集子集: {repo_id}",
                        "level": "full", "rows": 0,
                        "path": str(local)}
            if entry.startswith("script:"):
                script = entry.split(":", 1)[1]
                target_dir = ds_dir / "dataset_full"
                os.makedirs(target_dir, exist_ok=True)
                proc = subprocess.run(
                    [script, str(target_dir), str(_SMOKE_ROWS * 8)],
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode != 0:
                    return {"state": "unavailable",
                            "detail": f"数据集脚本失败: {(proc.stderr or '')[-200:]}"}
                rows = sum(1 for f in target_dir.glob("*.csv")
                           for _ in f.open(encoding="utf-8")) - 1
                return {"state": "downloaded",
                        "detail": f"注册表脚本下载: {name}",
                        "level": "full", "rows": max(rows, 0)}
            if entry.startswith("url:"):
                target_file = ds_dir / entry.split(":", 1)[1].rsplit("/", 1)[-1]
                proc = subprocess.run(
                    ["curl", "-L", "--fail", "-sS", "-o", str(target_file),
                     entry.split(":", 1)[1]],
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode != 0:
                    return {"state": "unavailable",
                            "detail": f"数据集下载失败: {(proc.stderr or '')[-200:]}"}
                return {"state": "downloaded",
                        "detail": f"注册表 URL 下载: {name}",
                        "level": "full", "rows": 0}
        except Exception as exc:
            return {"state": "unavailable", "detail": str(exc)[-200:]}
        return {"state": "unavailable", "detail": f"未实现的下载入口: {entry}"}

    def _write_smoke_dataset(self, smoke_dir: Path, dataset_name: str) -> None:
        """写入合成冒烟集：极小 CSV 样本 + dataset_info.json。"""
        rows = []
        for i in range(_SMOKE_ROWS):
            rows.append(f"{i},{i * 0.1:.1f},{i % 3}")
        (smoke_dir / "samples.csv").write_text(
            "id,feature,label\n" + "\n".join(rows) + "\n", encoding="utf-8")
        (smoke_dir / "dataset_info.json").write_text(
            json.dumps({
                "dataset": dataset_name,
                "subset": "smoke",
                "rows": _SMOKE_ROWS,
                "note": "合成最小样本，仅用于流程验证，不用于数值复现",
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 防泄漏 Benchmark：L0 物化 + 私有标签 ----------------

    def prepare_leakage_safe_benchmark(
            self, paper_id: str, dataset_name: str = "CIFAR-10",
            target_column: str = "",
            hidden_root: Optional[str] = None,
            public_dir: Optional[str] = None,
            strategy: str = "hash",
            task_type: str = "classification",
            primary_metric: str = "",
            target_score: Optional[float] = None,
            max_samples: Optional[int] = None,
            seed: int = 42) -> Dict:
        """把注册表数据集物化为防泄漏测试集（P1-⑩，与 dataset_registry 集成）。

        语义：评测关注"防泄漏机制 + 契约复算"链路本身，行数据用
        合成存根（synthetic_rows_for，依数据集名稳定生成）；接入真实
        数据时替换行来源即可，本方法签名与切分/泄漏逻辑不变。

        布局（L0 内）：
          public  = <data_root>/benchmark/<paper_id>/splits/
                    （train / validation / preflight_features /
                     test_features 公开特征，无 target 列）
          private = hidden_root（默认 <data_root>/.hidden/<paper_id>/，
                    沙箱工作区之外的私有目录，仅后端复算读取）

        返回 {"state","detail","hidden_labels_path","test_features_path",
              "leakage_report","contract","meta","manifest_path"}。
        """
        info: Dict = {"state": "skipped", "detail": ""}
        try:
            # 1. 注册表决策：数据集存在性 / 采样规模 / 元信息
            hint: Dict = {}
            lookup = getattr(self.dataset_registry, "benchmark_hint", None)
            if callable(lookup):
                hint = dict(lookup(dataset_name) or {})
            else:
                hint = {"found": False, "recommended_max_samples": 1000}
            if max_samples is None:
                max_samples = int(hint.get("recommended_max_samples", 1000))

            # 合成行来源（真实数据接入时替换此步）
            from src.benchmark.leakage_safe import (
                materialize_benchmark,
                freeze_metric_contract,
                synthetic_rows_for,
            )
            rows = synthetic_rows_for(dataset_name, n=max_samples, seed=seed)
            eval_task_type = task_type or "classification"
            if not target_column:
                target_column = "label"      # synthetic_rows_for 固定列

            # 2. 物化：公开特征 + 私有隐藏标签 + 泄漏自检
            bench_root = Path(public_dir) if public_dir else \
                self.data_root / "benchmark" / (paper_id or "default")
            hidden_root_path = Path(hidden_root) if hidden_root else \
                self.data_root / ".hidden" / (paper_id or "default")
            prepared = materialize_benchmark(
                rows=rows,
                target=target_column,
                hidden_root=hidden_root_path,
                public_dir=bench_root / "splits",
                input_column="text",
                task_type=eval_task_type,
                strategy=strategy,
                seed=seed,
                max_samples=max_samples,
                primary_metric=primary_metric,
                target_score=target_score,
            )
            contract = freeze_metric_contract(
                eval_task_type, primary_metric, target_score)

            # 3. manifest 落盘（L0 可审计）
            manifest_path = bench_root / "manifest.json"
            manifest = dict(prepared.manifest)
            manifest["dataset"] = {
                "name": hint.get("dataset_name", dataset_name),
                "registry_found": bool(hint.get("found")),
                "kind": hint.get("kind", ""),
                "subset": hint.get("subset", ""),
                "size_gb": hint.get("size_gb", 0.0),
            }
            manifest["contract"] = contract
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8")

            info.update(
                state="prepared",
                detail=(f"防泄漏基准已物化: {dataset_name} "
                        f"({strategy} 切分, {len(rows)} 行, "
                        f"test {manifest['test_row_count']} 行)"),
                hidden_labels_path=str(prepared.hidden_labels_path),
                hidden_labels_sha256=manifest["hidden_labels_sha256"],
                test_features_path=str(prepared.test_features_path),
                preflight_features_path=str(
                    prepared.preflight_features_path),
                train_path=str(prepared.public_dir / "train.jsonl"),
                validation_path=str(
                    prepared.public_dir / "validation.jsonl"),
                leakage_report=manifest["leakage_report"],
                contract=contract,
                meta=hint,
                manifest_path=str(manifest_path),
            )
        except Exception as exc:      # 物化失败不伪造
            info.update(state="unavailable",
                        detail=f"基准物化失败: {str(exc)[-300:]}")
        return info

    # ---------------- 懒加载：权重 ----------------

    def fetch_weights(self, paper_id: str, weights_ref: str,
                      quantized: bool = True) -> Dict:
        """拉取预训练权重到 L0。

        weights_ref 形态："" / "none" 跳过；本地路径则复制；http(s) 单
        文件则下载；hf://repo[/subpath] 则尝试 HuggingFace snapshot
        （库不可用或失败时诚实跳过，不伪造）。

        返回 {"path", "state", "detail", "quantized"}。
        state：none / copied / downloaded / skipped（离线降级）。
        """
        info: Dict = {"path": "", "state": "skipped",
                      "detail": "", "quantized": quantized}
        ref = (weights_ref or "").strip()
        if not ref or ref.lower() in ("none", "无", "null"):
            info.update(state="none", detail="无权重引用，跳过")
            return info
        target_dir = self._repo_dir(paper_id) / "weights"
        os.makedirs(target_dir, exist_ok=True)

        if ref.startswith(("http://", "https://")):
            try:
                fname = ref.rsplit("/", 1)[-1] or "weights.bin"
                target = target_dir / fname
                proc = subprocess.run(
                    ["curl", "-L", "--fail", "-sS", "-o", str(target), ref],
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode == 0 and target.stat().st_size > 0:
                    info.update(path=str(target), state="downloaded",
                                detail=f"下载权重: {fname}")
                else:
                    info.update(state="skipped",
                                detail="权重下载失败，离线跳过（不伪造）")
            except Exception:
                info.update(state="skipped",
                            detail="权重下载异常，离线跳过（不伪造）")
            return info

        if ref.startswith("hf://"):
            try:
                from huggingface_hub import snapshot_download  # 延迟导入
                repo = ref[5:].split("/", 1)
                repo_id = f"{repo[0]}/{repo[1]}" if len(repo) > 1 else repo[0]
                # 仅下载权重子路径（按需懒加载，不拉全仓）
                local = snapshot_download(
                    repo_id=repo_id, allow_patterns=["*.safetensors",
                                                     "*.bin", "*.pt"],
                    local_dir=str(target_dir))
                info.update(path=str(local), state="downloaded",
                            detail=f"HF snapshot: {repo_id}")
            except Exception as exc:
                info.update(state="skipped",
                            detail=f"HF 下载不可用: {str(exc)[-120:]}")
            return info

        # 本地路径引用：直接复制（移动硬盘/已挂载目录的权重引入 L0）
        local_path = Path(ref)
        if local_path.is_file():
            target = target_dir / local_path.name
            try:
                shutil.copy2(local_path, target)
                info.update(path=str(target), state="copied",
                            detail=f"复制本地权重: {local_path.name}")
            except Exception as exc:
                info.update(state="skipped", detail=str(exc)[-200:])
        else:
            info.update(state="skipped",
                        detail=f"权重引用未解析: {ref[:120]}")
        return info

    # ---------------- manifest ----------------

    def build_manifest(self, paper_id: str, paper_title: str = "",
                       code_url: str = "", dataset_name: str = "",
                       weights_ref: str = "") -> Dict:
        """生成 manifest（磁盘状态为主，引用字段为辅）。"""
        repo_dir = self._repo_dir(paper_id)
        ds_dir = self._dataset_dir(paper_id)
        code_path = str(repo_dir) if (repo_dir.exists()
                                      and any(repo_dir.iterdir())) else ""
        ds_path = str(ds_dir / "dataset_smoke") if (
            (ds_dir / "dataset_smoke").exists()) else (
            str(ds_dir) if ds_dir.exists() and any(ds_dir.iterdir()) else "")
        w_path = ""
        w_dir = repo_dir / "weights"
        if w_dir.exists():
            files = list(w_dir.iterdir())
            if files:
                w_path = str(files[0])
        return {
            "paper_id": paper_id,
            "created_at": _now_iso(),
            "resources": {"code": code_path,
                          "dataset": ds_path,
                          "weights": w_path},
            "paper_title": paper_title or "",
            "code_url": code_url or "",
            "dataset_name": dataset_name or "",
        }

    def save_manifest(self, manifest: Dict) -> str:
        path = self._manifest_path(manifest["paper_id"])
        path.write_text(json.dumps(manifest, ensure_ascii=False,
                                   indent=2), encoding="utf-8")
        return str(path)

    def get_manifest(self, paper_id: str) -> Optional[Dict]:
        path = self._manifest_path(paper_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_manifests(self) -> List[Dict]:
        out: List[Dict] = []
        if not self.manifests_root.is_dir():
            return out
        for f in sorted(self.manifests_root.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("paper_id"):
                out.append(data)
        return out

    # ---------------- 清理 / 归档 ----------------

    def cleanup(self, paper_id: str, keep_manifest: bool = True) -> Dict:
        """清理 L0 条目（代码仓库 + 数据集 + 本地权重），manifest 记录
        cleaned_at（兼容存量 test_cleanup.json 字段）。"""
        removed: List[str] = []
        for root in (self._repo_dir(paper_id), self._dataset_dir(paper_id)):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
                removed.append(str(root))
        result = {"paper_id": paper_id, "removed": removed}
        if keep_manifest:
            manifest = self.get_manifest(paper_id)
            if manifest:
                manifest["cleaned_at"] = _now_iso()
                for key in ("code", "dataset", "weights"):
                    manifest["resources"][key] = ""
                self.save_manifest(manifest)
                result["manifest"] = str(self._manifest_path(paper_id))
        return result

    def archive(self, paper_id: str, dest_dir: Optional[str] = None,
                include_manifest: bool = True) -> Dict:
        """打包整篇论文的 L0 资源为 zip（L1 温存储对接）。

        返回 {"archive", "bytes", "entries"}；缺资源时仅打包存在的部分。
        """
        dest = Path(dest_dir) if dest_dir else self.archive_root
        os.makedirs(dest, exist_ok=True)
        zip_path = dest / f"{paper_id}.zip"
        entries = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root in (self._repo_dir(paper_id), self._dataset_dir(paper_id)):
                if root.exists():
                    arc_dir = root.relative_to(self.data_root).as_posix()
                    for f in sorted(root.rglob("*")):
                        if f.is_file():
                            arc = f"{arc_dir}/{f.relative_to(root).as_posix()}"
                            zf.write(f, arc)
                            entries += 1
            if include_manifest:
                manifest = self.get_manifest(paper_id)
                if manifest:
                    zf.writestr("manifest.json",
                                json.dumps(manifest, ensure_ascii=False,
                                           indent=2))
                    entries += 1
        return {"archive": str(zip_path),
                "bytes": zip_path.stat().st_size, "entries": entries}

    def restore(self, archive_path: str, paper_id: Optional[str] = None) -> Dict:
        """从 L1 归档 zip 恢复回 L0（repos / datasets / manifests）。"""
        archive = Path(archive_path)
        if not archive.is_file():
            return {"ok": False, "detail": f"归档不存在: {archive_path}"}
        restored: List[str] = []
        manifest: Optional[Dict] = None
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member == "manifest.json":
                    continue
                parts = member.split("/", 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                if parts[0] == "repos":
                    target = self.repos_root / parts[1]
                elif parts[0] == "datasets":
                    target = self.datasets_root / parts[1]
                else:
                    continue
                os.makedirs(target.parent, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                restored.append(str(target))
            if "manifest.json" in zf.namelist():
                manifest = json.loads(
                    zf.read("manifest.json").decode("utf-8"))
        if manifest:
            self.save_manifest(manifest)
        return {"ok": True, "restored": restored,
                "manifest": manifest or self.get_manifest(
                    paper_id or Path(archive).stem)}

    # ---------------- L0 配额守护 ----------------

    def quota_usage(self) -> Dict:
        """各缓存根占用与总占用（相对配额）。"""
        roots = {"repos": self.repos_root,
                 "datasets": self.datasets_root,
                 "deps": self.data_root / "deps"}
        per_root = {}
        total = 0
        for name, root in roots.items():
            size = _dir_bytes(root)
            per_root[name] = {"bytes": size, "human": _fmt_bytes(size)}
            total += size
        return {"bytes": total, "human": _fmt_bytes(total),
                "quota": self.quota_bytes,
                "quota_human": _fmt_bytes(self.quota_bytes),
                "percent": round(100.0 * total / max(self.quota_bytes, 1), 1),
                "per_root": per_root}

    def enforce_quota(self, dry_run: bool = True,
                      excess_bytes: Optional[int] = None) -> List[Dict]:
        """超限时按最近使用（mtime 旧->新）返回建议归档清单。

        excess_bytes 用于"预计超限"场景（如拉取前配额预检）：
        传入预计超额字节，即使当前未超限也能给出按 LRU 排列的
        释放建议；不传则按当前实际占用计算。

        默认 dry_run=True 只建议不删除（P2 CLI 中 prune 需用户显式确认）。
        返回 [{paper_id, bytes, human, last_used, reason}]。
        """
        usage = self.quota_usage()
        if usage["bytes"] <= self.quota_bytes and excess_bytes is None:
            return []
        excess = (excess_bytes if excess_bytes is not None
                  else usage["bytes"] - self.quota_bytes)
        candidates: List[Dict] = []
        for root in (self.repos_root, self.datasets_root):
            if not root.is_dir():
                continue
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    mtime = child.stat().st_mtime
                except OSError:
                    mtime = 0
                size = _dir_bytes(child)
                if size:
                    candidates.append({
                        "paper_id": child.name, "base": root.name,
                        "bytes": size, "human": _fmt_bytes(size),
                        "last_used": mtime,
                    })
        candidates.sort(key=lambda c: c["last_used"])
        suggestions: List[Dict] = []
        freed = 0
        for c in candidates:
            if freed >= excess:
                break
            freed += c["bytes"]
            c["reason"] = "L0 配额超限，按最近使用建议归档（dry_run）" \
                if dry_run else "L0 配额超限，已清理"
            c["dry_run"] = dry_run
            suggestions.append(c)
        return suggestions

    # ---------------- 统计 ----------------

    def stats(self) -> Dict:
        """按缓存根与按论文聚合的统计。"""
        usage = self.quota_usage()
        manifests = self.list_manifests()
        papers = []
        for m in manifests:
            pid = m["paper_id"]
            repo = self._repo_dir(pid)
            dset = self._dataset_dir(pid)
            papers.append({
                "paper_id": pid,
                "title": m.get("paper_title", ""),
                "code_bytes": _dir_bytes(repo),
                "dataset_bytes": _dir_bytes(dset),
                "has_manifest": True,
                "created_at": m.get("created_at", ""),
                "cleaned_at": m.get("cleaned_at", ""),
            })
        return {
            "usage": usage,
            "manifest_count": len(manifests),
            "papers": papers,
        }