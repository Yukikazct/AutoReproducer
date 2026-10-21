"""ResourceManager - L0 热缓存资源管理（对齐方案「三层存储」）。

职责：
- 懒加载：不预下载全部资源，fetch_code / fetch_dataset / fetch_weights
  只拉当前任务最小集；代码复用固定 commit 缓存，每次导出独立工作区；
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

指定代码版本获取失败不会回退 HEAD。其他资源下载（HuggingFace snapshot /
ModelScope / 单文件 URL）为"尽力而为"：执行器或网络不可用时诚实降级为本地最小冒烟集
（dataset_smoke），并在返回 info 的 state 字段标注实际状态，绝不静默伪造大文件。
"""
import datetime
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import unicodedata
import uuid
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
from filelock import FileLock
from src.project_paths import get_data_root
from src.resource_events import ResourceEventLogger
from src.safety.files import remove_tree
from src.safety.paths import is_link, workspace_path
from src.runtime_preparation import run_owned_process

# src/resource_manager.py -> parents[1] 为仓库内 AutoReproducer 包根
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 统一缓存根（data 之下，与存量 manifest 绝对路径一致）
DATA_ROOT = get_data_root()
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
_WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³"}


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


def _portable_component(value: str) -> bool:
    """A component usable without aliases on both Windows and POSIX."""
    if (not isinstance(value, str) or not value or value in (".", "..")
            or value.endswith((" ", "."))
            or any(char in value for char in '/\\:<>"|?*')
            or any(ord(char) < 32 for char in value)):
        return False
    device = value.split(".", 1)[0].rstrip(" ").upper()
    return device not in _WINDOWS_DEVICE_NAMES


def _validate_paper_id(paper_id: str) -> str:
    """资源 ID 必须是跨平台可用的单个目录名。"""
    if not _portable_component(paper_id):
        raise ValueError(f"无效的 paper_id: {paper_id!r}")
    return paper_id


def _archive_parts(name: str, aliases: Dict[str, str]) -> List[str]:
    """Reject Windows devices, separators and case/Unicode path aliases."""
    parts = name.rstrip("/").split("/")
    if not all(_portable_component(part) for part in parts):
        raise ValueError(f"归档路径不能跨平台安全恢复: {name}")
    for length in range(1, len(parts) + 1):
        prefix = "/".join(parts[:length])
        key = unicodedata.normalize("NFC", prefix).casefold()
        if key in aliases and aliases[key] != prefix:
            raise ValueError(f"归档路径大小写或 Unicode 别名冲突: {name}")
        aliases[key] = prefix
    return parts


def _replace_file(source: Path, destination: Path) -> None:
    """Briefly retry Windows sharing conflicts; permanent locks still fail."""
    for attempt in range(11):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if attempt == 10 or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(0.05)


def _atomic_json(path: Path, value: Dict) -> None:
    """同目录临时文件 + 原子替换，读者只能看到完整的新/旧 JSON。"""
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        self.data_root = get_data_root(data_root)
        self.repos_root = self.data_root / "repos"
        self.datasets_root = self.data_root / "datasets"
        self.manifests_root = self.data_root / "manifests"
        self.archive_root = self.data_root / "archive"
        self.quota_bytes = _L0_QUOTA if quota_bytes is None else quota_bytes
        if self.quota_bytes < 0:
            raise ValueError("L0 配额不能为负数")
        if dataset_registry is None:
            from src.dataset_registry import DatasetRegistry
            dataset_registry = DatasetRegistry()
        self.dataset_registry = dataset_registry
        self.logger = logger
        self.resource_events = ResourceEventLogger(
            str(self.data_root / "resource_events.jsonl"))
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

    def _resource_event(self, resource_type: str, resource_id: str,
                        operation: str, state: str, **details: Any) -> None:
        self.resource_events.emit(resource_type, resource_id, operation,
                                  state, **details)

    @staticmethod
    def paper_id_for(paper_title: str, corpus_key: str = "") -> str:
        """论文稳定 ID：corpus 语料键直接可用，否则 sha1(title) 前 12 位
        （与存量 manifest 的 12 位十六进制 paper_id 风格一致）。"""
        if corpus_key:
            return _validate_paper_id(corpus_key)
        digest = hashlib.sha1(paper_title.encode("utf-8")).hexdigest()
        return digest[:12]

    def _repo_dir(self, paper_id: str) -> Path:
        return self._managed_path(self.repos_root, paper_id)

    def _dataset_dir(self, paper_id: str) -> Path:
        return self._managed_path(self.datasets_root, paper_id)

    def _manifest_path(self, paper_id: str) -> Path:
        return self._managed_path(self.manifests_root, paper_id, ".json")

    def _managed_path(self, root: Path, paper_id: str, suffix: str = "") -> Path:
        path = root / (_validate_paper_id(paper_id) + suffix)
        workspace_path(self.data_root, path.relative_to(self.data_root),
                       "资源路径不能包含符号链接")
        return path

    # ---------------- 懒加载：代码 ----------------

    def fetch_code(self, paper_id: str, code_url: str,
                   target: Optional[str] = None,
                   revision: str = "") -> Dict:
        """复用 URL/commit 源码缓存，逐次导出独立的干净工作区。

        指定版本无法获取时返回 pin-failed 和空路径，绝不回退 HEAD。
        未指定版本时先解析远端 HEAD，再固定其完整 commit。
        显式 target 必须尚不存在；已有实验和旧缓存不被修改。
        """
        self._repo_dir(paper_id)  # Validate the identity and managed root even for an explicit target.
        info: Dict = {"path": "", "state": "skipped", "detail": "",
                      "commit": "", "revision": revision or ""}
        url = (code_url or "").strip()
        if not url:
            info["detail"] = "无代码仓库 URL"
            return info
        if _is_placeholder_url(url):
            info.update(state="placeholder-skip", detail=f"占位 URL 不下载: {url}")
            return info
        if not _git_available():
            info["state"] = "pin-failed" if revision else "clone-failed"
            info["detail"] = "git 不可用，跳过代码下载"
            return info
        try:
            from src.repository_reproduction import export_repository, resolve_repository_revision

            repo_dir = (Path(target).absolute() if target else workspace_path(
                self.data_root, Path("repos") / paper_id /
                "workspaces" / uuid.uuid4().hex / "repo", "repository workspace"))
            if repo_dir.exists() or is_link(repo_dir):
                raise ValueError("运行工作区已经存在，拒绝覆盖或复用旧实验")
            resolved = resolve_repository_revision(self.data_root, url, revision)
            snapshot = export_repository(self.data_root,
                {"repository": {"url": url, "revision": resolved}}, repo_dir, required_files=[])
            acquisition = snapshot["acquisition"]
            prepared = {**info, **snapshot, "commit": resolved, "revision": revision or "",
                        "state": acquisition, "detail": f"已导出独立工作区；固定 commit {resolved}"}
            self._record_repo_source(repo_dir, url, prepared, acquisition)
            info = prepared
        except Exception as exc:
            info.update(path="", commit="", state="pin-failed" if revision else "clone-failed",
                        detail=f"固定源码获取失败，停止使用该仓库：{str(exc)[-300:]}")
        return info

    def _record_repo_source(self, repo_dir: Path, url: str, info: Dict,
                            acquisition: str) -> None:
        """原子保存实际来源；无法保存时不能报获取成功。"""
        if ".autorepro-repo-source.json" in info.get("files", {}):
            raise ValueError("固定源码占用来源记录路径，拒绝覆盖作者文件")
        marker = {
            "repo_url": url,
            "commit": info.get("commit", ""),
            "acquisition": acquisition,
            "revision": info.get("revision", ""),
            "archive_sha256": info.get("archive_sha256", ""),
            "files": info.get("files", {}),
            "file_aliases": info.get("file_aliases", []),
        }
        _atomic_json(repo_dir / ".autorepro-repo-source.json", marker)

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
        self._resource_event("dataset", f"{paper_id}:{name}", "download",
                             "running", paper_id=paper_id, level=level,
                             source=name)
        if smoke_dir.exists() and any(smoke_dir.iterdir()):
            info.update(path=str(smoke_dir), state="cached",
                        level="smoke", detail="冒烟集缓存命中",
                        rows=_SMOKE_ROWS)
            self._resource_event("dataset", f"{paper_id}:{name}", "download",
                                 "cached", path=str(smoke_dir),
                                 bytes=self._path_bytes(smoke_dir))
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
                    self._resource_event(
                        "dataset", f"{paper_id}:{name}", "download",
                        "succeeded", path=str(ds_dir),
                        bytes=self._path_bytes(ds_dir), detail=info["detail"])
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
        self._resource_event(
            "dataset", f"{paper_id}:{name}", "download",
            "succeeded" if info["state"] == "smoke-synth" else "failed",
            path=info.get("path"), bytes=self._path_bytes(ds_dir),
            detail=info.get("detail"))
        return info

    @staticmethod
    def _path_bytes(path: Path) -> int:
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        if not path.is_dir():
            return 0
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        return total

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
                command = ([sys.executable, script] if Path(script).suffix.casefold() == ".py"
                           else [script])
                proc = run_owned_process(
                    [*command, str(target_dir), str(_SMOKE_ROWS * 8)], timeout_s=1800)
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
                proc = run_owned_process(
                    ["curl", "-L", "--fail", "-sS", "-o", str(target_file),
                     entry.split(":", 1)[1]],
                    timeout_s=1800)
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
                proc = run_owned_process(
                    ["curl", "-L", "--fail", "-sS", "-o", str(target), ref],
                    timeout_s=1800)
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
                       weights_ref: str = "", code_snapshot: Optional[Dict] = None) -> Dict:
        """生成 manifest（磁盘状态为主，引用字段为辅）。"""
        repo_dir = self._repo_dir(paper_id)
        ds_dir = self._dataset_dir(paper_id)
        code_path = str(repo_dir) if (repo_dir.exists()
                                      and any(repo_dir.iterdir())) else ""
        if code_snapshot is not None:
            code_path = code_snapshot.get("path", "")
        ds_path = str(ds_dir / "dataset_smoke") if (
            (ds_dir / "dataset_smoke").exists()) else (
            str(ds_dir) if ds_dir.exists() and any(ds_dir.iterdir()) else "")
        w_path = ""
        w_dir = repo_dir / "weights"
        if w_dir.exists():
            files = list(w_dir.iterdir())
            if files:
                w_path = str(files[0])
        manifest = {
            "paper_id": paper_id,
            "created_at": _now_iso(),
            "resources": {"code": code_path,
                          "dataset": ds_path,
                          "weights": w_path},
            "paper_title": paper_title or "",
            "code_url": code_url or "",
            "dataset_name": dataset_name or "",
        }
        if code_snapshot is not None:
            manifest["repository"] = dict(code_snapshot)
        return manifest

    def save_manifest(self, manifest: Dict) -> str:
        path = self._manifest_path(manifest["paper_id"])
        with FileLock(str(path.with_suffix(".lock")), timeout=10):
            _atomic_json(path, manifest)
        return str(path)

    def get_manifest(self, paper_id: str) -> Optional[Dict]:
        path = self._manifest_path(paper_id)
        # Windows can deny opening a file during an otherwise atomic replace.
        # Readers and writers share a process-independent lock so a transient
        # sharing conflict cannot turn an existing manifest into a missing one.
        with FileLock(str(path.with_suffix(".lock")), timeout=10):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict) and manifest.get("paper_id") == paper_id:
                    return manifest
                return None
            except (OSError, json.JSONDecodeError):
                return None

    def list_manifests(self) -> List[Dict]:
        out: List[Dict] = []
        if not self.manifests_root.is_dir():
            return out
        for f in sorted(self.manifests_root.glob("*.json")):
            try:
                data = self.get_manifest(f.stem)
            except (OSError, ValueError):
                continue
            if data is not None:
                out.append(data)
        return out

    # ---------------- 清理 / 归档 ----------------

    def cleanup(self, paper_id: str, keep_manifest: bool = True) -> Dict:
        """清理 L0 条目（代码仓库 + 数据集 + 本地权重），manifest 记录
        cleaned_at（兼容存量 test_cleanup.json 字段）。"""
        removed: List[str] = []
        errors: List[str] = []
        for root in (self._repo_dir(paper_id), self._dataset_dir(paper_id)):
            if root.exists():
                try:
                    remove_tree(root)
                    removed.append(str(root))
                except OSError as exc:
                    errors.append(f"{root}: {exc}")
        result = {"paper_id": paper_id, "removed": removed,
                  "ok": not errors, "errors": errors}
        if keep_manifest:
            manifest = self.get_manifest(paper_id)
            if manifest:
                resources = manifest.setdefault("resources", {})
                if not self._repo_dir(paper_id).exists():
                    resources.update(code="", weights="")
                if not self._dataset_dir(paper_id).exists():
                    resources["dataset"] = ""
                if not errors:
                    manifest["cleaned_at"] = _now_iso()
                self.save_manifest(manifest)
                result["manifest"] = str(self._manifest_path(paper_id))
        elif not errors:
            self._manifest_path(paper_id).unlink(missing_ok=True)
        return result

    def archive(self, paper_id: str, dest_dir: Optional[str] = None,
                include_manifest: bool = True) -> Dict:
        """打包整篇论文的 L0 资源为 zip（L1 温存储对接）。

        校验完整归档后原子替换目标；失败时保留旧归档。
        符号链接无法可靠往返，拒绝归档以避免后续清理丢失资源。
        返回 {"archive", "bytes", "entries"}；缺资源时仅打包存在的部分。
        """
        _validate_paper_id(paper_id)
        roots = (self._repo_dir(paper_id), self._dataset_dir(paper_id))
        dest = Path(dest_dir) if dest_dir else self.archive_root
        for root in roots:
            if dest.resolve().is_relative_to(root.resolve()):
                raise ValueError("归档目标不能位于待归档资源内部")
        os.makedirs(dest, exist_ok=True)
        zip_path = dest / f"{paper_id}.zip"
        entries = 0
        resource_entries = 0
        aliases: Dict[str, str] = {}
        fd, name = tempfile.mkstemp(prefix=f".{paper_id}.", suffix=".zip.tmp", dir=dest)
        os.close(fd)
        temporary = Path(name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
                for root in roots:
                    if root.exists():
                        pending = [root]
                        while pending:
                            path = pending.pop()
                            if is_link(path) or not (path.is_dir() or path.is_file()):
                                raise ValueError(f"归档不支持符号链接或特殊文件: {path}")
                            member = path.relative_to(self.data_root).as_posix()
                            _archive_parts(member, aliases)
                            zf.write(path, member)
                            entries += 1
                            resource_entries += 1
                            if path.is_dir():
                                pending.extend(reversed(sorted(path.iterdir())))
                if include_manifest:
                    manifest = self.get_manifest(paper_id)
                    if manifest:
                        zf.writestr("manifest.json", json.dumps(
                            manifest, ensure_ascii=False, indent=2))
                        entries += 1
            if not resource_entries and zip_path.exists():
                raise ValueError("L0 无资源，保留已有归档")
            with zipfile.ZipFile(temporary) as zf:
                corrupt = zf.testzip()
                if corrupt:
                    raise ValueError(f"归档校验失败: {corrupt}")
            with temporary.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            _replace_file(temporary, zip_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"archive": str(zip_path),
                "bytes": zip_path.stat().st_size, "entries": entries}

    def restore(self, archive_path: str, paper_id: Optional[str] = None) -> Dict:
        """先验证并暂存完整 zip，再恢复 L0；拒绝越界、链接和论文混包。

        paper_id 指定时校验归档身份。支持旧版无目录条目的 zip，恢复后
        将 manifest 的资源路径重定位到当前 data_root。
        """
        archive = Path(archive_path)
        if not archive.is_file():
            return {"ok": False, "detail": f"归档不存在: {archive_path}"}
        restored: List[str] = []
        manifest: Optional[Dict] = None
        try:
            if paper_id is not None:
                _validate_paper_id(paper_id)
            with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory(
                    prefix=".restore-", dir=self.data_root) as temporary:
                members = []
                identities = set()
                seen = set()
                aliases: Dict[str, str] = {}
                for member in zf.infolist():
                    name = member.filename
                    parts = _archive_parts(name, aliases)
                    mode = member.external_attr >> 16
                    if (name.startswith("/") or "\\" in name
                            or any(part in ("", ".", "..") or ":" in part for part in parts)
                            or any(ord(char) < 32 for char in name)
                            or name.rstrip("/") in seen
                            or stat.S_ISLNK(mode)
                            or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                        raise ValueError(f"不安全或重复的归档路径: {name}")
                    seen.add(name.rstrip("/"))
                    if name == "manifest.json":
                        manifest = json.loads(zf.read(member).decode("utf-8"))
                        if (not isinstance(manifest, dict)
                                or not isinstance(manifest.get("resources", {}), dict)):
                            raise ValueError("归档 manifest 格式无效")
                        identities.add(_validate_paper_id(manifest.get("paper_id")))
                        continue
                    if (len(parts) < 2 or parts[0] not in ("repos", "datasets")
                            or (len(parts) == 2 and not member.is_dir())):
                        raise ValueError(f"不支持的归档路径: {name}")
                    identities.add(_validate_paper_id(parts[1]))
                    target = self.data_root.joinpath(*parts)
                    self._check_restore_target(target, member.is_dir())
                    members.append((member, Path(*parts), target))
                if len(identities) != 1:
                    raise ValueError("归档必须包含且仅包含一篇论文的资源")
                identity = identities.pop()
                if paper_id is not None and paper_id != identity:
                    raise ValueError(f"归档 paper_id 不匹配: {identity}")
                self._manifest_path(identity)

                # 完整解压到暂存目录；CRC/格式失败不覆盖现有资源。
                stage = Path(temporary)
                for member, relative, _target in members:
                    target = stage / relative
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output)
                        permissions = (member.external_attr >> 16) & 0o777
                        # Windows chmod only toggles read-only; POSIX execute
                        # bits have no meaning and may make cleanup fail there.
                        if permissions and os.name != "nt":
                            target.chmod(permissions)

                for member, relative, target in members:
                    self._check_restore_target(target, member.is_dir())
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _replace_file(stage / relative, target)
                        restored.append(str(target))
                manifest = manifest or self.get_manifest(identity) or self.build_manifest(identity)
                self._rebase_manifest(manifest)
                self.save_manifest(manifest)
            return {"ok": True, "restored": restored, "manifest": manifest}
        except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, zlib.error,
                NotImplementedError) as exc:
            return {"ok": False, "detail": f"归档恢复失败: {exc}", "restored": restored}

    def _check_restore_target(self, target: Path, is_directory: bool) -> None:
        workspace_path(self.data_root, target.relative_to(self.data_root),
                       "恢复目标不能包含符号链接")
        current = target
        while current != self.data_root:
            if current != target and current.exists() and not current.is_dir():
                raise ValueError(f"恢复目标父路径不是目录: {current}")
            current = current.parent
        if target.exists() and target.is_dir() != is_directory:
            raise ValueError(f"恢复目标文件类型冲突: {target}")

    def _rebase_manifest(self, manifest: Dict) -> None:
        """兼容旧绝对路径，保留可识别资源的相对后缀。"""
        identity = manifest["paper_id"]
        resources = self.build_manifest(identity)["resources"]
        for key, base in (("code", "repos"), ("weights", "repos"),
                          ("dataset", "datasets")):
            old = manifest.get("resources", {}).get(key, "")
            if not isinstance(old, str):
                continue
            parts = PurePosixPath(old.replace("\\", "/")).parts
            for index in range(len(parts) - 1):
                if parts[index:index + 2] == (base, identity):
                    tail = parts[index + 2:]
                    if ".." not in tail:
                        candidate = self.data_root / base / identity
                        candidate = candidate.joinpath(*tail)
                        if candidate.exists() and candidate.resolve().is_relative_to(
                                (self.data_root / base / identity).resolve()):
                            resources[key] = str(candidate)
                    break
        manifest["resources"] = resources
        if isinstance(manifest.get("repository"), dict):
            manifest["repository"]["path"] = resources["code"]
        manifest.pop("cleaned_at", None)
        manifest["restored_at"] = _now_iso()

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
                if not child.is_dir() or is_link(child):
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
                if dry_run else "L0 配额超限，待归档后清理"
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
