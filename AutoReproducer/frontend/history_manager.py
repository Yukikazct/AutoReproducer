"""复现历史管理：扫描 ledger/runtime/reports 目录，生成历史记录列表，
提供存储状态、一键清理、报告下载链接。

无 Streamlit 依赖，纯 Python 工具函数，便于单元测试。
"""
import functools
import json
import logging
import os
import shutil
import re
import stat
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dependency_cache import cache_guard, DependencyCacheBusy
from src.dependency_management import environment_directories, read_metadata
from src.safety.paths import is_link, workspace_path
from src.safety.files import remove_tree
from src.project_paths import get_data_root
from src.resource_events import read_resource_events


def get_project_data_dir() -> Path:
    return get_data_root()


def _parse_session_id(filename: str) -> Optional[str]:
    """从文件名提取 session_id 格式 (YYYYMMDD_HHMMSS)。"""
    name = Path(filename).stem
    for prefix in ("ledger_", "session_", "progress_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _session_time(session_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(session_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def list_sessions() -> List[Dict[str, Any]]:
    """扫描 experiment_ledger/、logs/、runtime/，反推所有历史复现会话。"""
    base = get_project_data_dir()
    sessions: Dict[str, Dict[str, Any]] = {}

    # 扫描 experiment_ledger（最权威来源：含论文标题、结果等）
    ledger_dir = base / "experiment_ledger"
    if ledger_dir.exists():
        for f in sorted(ledger_dir.glob("ledger_*.jsonl"), reverse=True):
            sid = _parse_session_id(f.name)
            if not sid:
                continue
            sessions.setdefault(sid, {
                "session_id": sid,
                "paper_title": "",
                "state": "未知",
                "duration_sec": 0.0,
                "llm_calls": 0,
                "log_entries": 0,
                "ledger_size_bytes": 0,
                "progress_file": "",
                "report_path": "",
            })
            # 读取 ledger 全量记录：标题取首个含 outputs.title 的记录，
            # 终态（state/duration/llm_calls）取末条 FINISH 记录的 result。
            sessions[sid]["ledger_size_bytes"] = f.stat().st_size
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    records = [json.loads(line) for line in fh if line.strip()]
                for rec in records:
                    outputs = rec.get("outputs", {}) or {}
                    title = outputs.get("title") or \
                        (outputs.get("paper_info") or {}).get("title", "")
                    if title:
                        sessions[sid]["paper_title"] = title
                        break
                if records:
                    result = records[-1].get("result", {}) or {}
                    sessions[sid]["state"] = result.get("state", "RUNNING")
                    sessions[sid]["duration_sec"] = result.get("duration_sec", 0)
                    sessions[sid]["llm_calls"] = result.get("llm_calls", 0)
            except Exception:
                pass
            sessions[sid]["log_entries"] += _count_lines(f)

    # 扫描 logs/ 补充日志条目数
    log_dir = base / "logs"
    if log_dir.exists():
        for f in sorted(log_dir.glob("session_*.jsonl"), reverse=True):
            sid = _parse_session_id(f.name)
            if not sid or sid not in sessions:
                continue
            sessions[sid]["log_entries"] += _count_lines(f)

    # 扫描 runtime/ 关联 progress 文件
    runtime_dir = base / "runtime"
    if runtime_dir.exists():
        for f in sorted(runtime_dir.glob("progress_*.jsonl"), reverse=True):
            # runtime 文件名用毫秒时间戳，无法精确匹配 session_id
            # 但可以通过文件内容中的 done 事件提取 session_id
            try:
                sid_from_file = _extract_session_from_progress(f)
                if sid_from_file and sid_from_file in sessions:
                    sessions[sid_from_file]["progress_file"] = str(f)
            except Exception:
                pass

    # 扫描 reports/ 关联报告文件
    reports_dir = base / "reports"
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*.md"), reverse=True):
            # 报告文件名: {title}_{YYYYMMDD_HHMMSS}.md
            # 提取末尾的时间戳部分
            stem = f.stem
            # 从末尾往前找最后一个符合 YYYYMMDD_HHMMSS 的位置
            # 注意：YYYYMMDD_HHMMSS 共 15 字符（8 + 1 + 6）
            sid = ""
            for i in range(len(stem) - 14, -1, -1):
                candidate = stem[i:i+15]
                if len(candidate) == 15 and candidate[8] == '_' and candidate[:8].isdigit() and candidate[9:].isdigit():
                    sid = candidate
                    break
            if sid and sid in sessions:
                sessions[sid]["report_path"] = str(f)
                # 如果 ledger 没抓到标题，尝试从报告文件名推断
                if not sessions[sid]["paper_title"]:
                    title_guess = stem[:len(stem)-len(sid)-1]  # 去掉 _{sid}
                    sessions[sid]["paper_title"] = title_guess.replace("_", " ")

    # 按时间倒序排列
    return sorted(sessions.values(), key=lambda s: s["session_id"], reverse=True)


def get_session_report_result(session):
    """Load saved inputs for viewing/regenerating a historical report, locally."""
    from src.report_revisions import load_report_revision_source, resolve_report_revision

    path = session.get("report_path")
    result = None
    if path:
        candidates = [Path(path).with_suffix(".result.json"), Path(path).parent / "result.json"]
        for candidate in candidates:
            try:
                saved = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(saved, dict) and isinstance(saved.get("data"), dict):
                    result = saved
                    break
            except (OSError, ValueError):
                continue
    if result is None and session.get("progress_file"):
        from frontend.progress_state import read_progress_snapshot
        snapshot = read_progress_snapshot(session["progress_file"])
        if snapshot.get("done"):
            result = snapshot.get("result")
    if result is None and path:
        result = load_report_revision_source(path)
    if isinstance(result, dict):
        # History filenames are the permanent report anchor; the same run may
        # also have an internal report under runs/<id>/.
        if path:
            result = {**result, "report_path": path}
        if result.get("state") is None and session.get("state") in {"COMPLETED", "ERROR"}:
            result = {**result, "state": session["state"]}
        return resolve_report_revision(result, data_root=get_project_data_dir())
    if path:
        try:
            return {"state": session.get("state"), "report_path": path,
                    "data": {"report": Path(path).read_text(encoding="utf-8")}}
        except OSError:
            pass
    return None


def _count_lines(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def _extract_session_from_progress(path: Path) -> Optional[str]:
    """从进度文件内容提取 session_id（如果日志条目中有）。"""
    try:
        sid = _session_id_from_progress(path)
        if sid:
            return sid
        # 没找到就回退到文件修改时间
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return None


def get_storage_stats(*, _sizes=None) -> Dict[str, Any]:
    """各数据子目录的存储占用统计。"""
    base = get_project_data_dir()
    stats = {}
    total = 0
    for sub in ("experiment_ledger", "logs", "runtime", "reports",
                "optimization_demo", "pinn-output",
                "datasets", "deps", "repos", "archive", "manifests"):
        d = base / sub
        if not d.exists():
            stats[sub] = {"files": 0, "bytes": 0}
            continue
        count, total_bytes = _shared_dir_size(d, {} if _sizes is None else _sizes)
        stats[sub] = {"files": count, "bytes": total_bytes}
        total += total_bytes
    stats["total"] = {"bytes": total}
    return stats


def _is_finished_progress(path: Path) -> bool:
    """判断 runtime 进度文件是否已终态（含 type=done 或 type=error 事件）。

    只读取不修改，供 cleanup_runtime / clear_sessions 复用。
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("type") in ("done", "error"):
                    return True
    except Exception:
        return False
    return False


def cleanup_runtime(keep_days: int = 7) -> Tuple[int, int]:
    """清理 runtime 中的过期/终态进度文件，返回 (删除文件数, 释放字节数)。

    清理范围（两条规则任一命中即删）：
    1. 进度文件已终态（内容含 type=done / type=error 事件）——
       会话结束后 progress 只是展示缓存，ledger/logs 才是权威记录；
    2. 超过 keep_days 天未更新的遗留文件（含异常中断、卡死残留）。
    """
    cutoff = time.time() - keep_days * 86400
    runtime_dir = get_project_data_dir() / "runtime"
    removed = 0
    freed = 0
    if not runtime_dir.exists():
        return 0, 0
    for f in runtime_dir.glob("*.jsonl"):
        try:
            stale = f.stat().st_mtime < cutoff
            if stale or _is_finished_progress(f):
                count, size = _unlink_files([f])
                freed += size
                removed += count
        except Exception:
            pass
    return removed, freed


@functools.lru_cache(maxsize=4096)
def _cached_progress_sid(path_str: str, _mtime_ns: int,
                         _size: int) -> Optional[str]:
    """按「路径 + mtime + 大小」缓存 progress 文件的归属解析结果。

    后两个参数**只作缓存键**、不参与解析：progress 是追加写的，内容一变
    mtime/大小必变，所以旧键自然失效，不会返回过期结果。

    需要缓存是因为 _related_files 每处理一个会话都要重扫一遍全部
    progress 文件，批量删除 332 条会话时是约 10 万次开文件。
    """
    return _scan_progress_sid(Path(path_str))


def _session_id_from_progress(path: Path) -> Optional[str]:
    """优先读取完成结果的 session_id，回退到 log 时间戳（无 mtime 回退）。

    仅当文件中存在明确会话 ID 或可解析的 log.timestamp 时才返回，避免把不同会话的
    progress 误归到目标会话（删除操作宁可少删、不可误删）。
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return _cached_progress_sid(str(path), st.st_mtime_ns, st.st_size)


def _scan_progress_sid(path: Path) -> Optional[str]:
    """真正读取并解析 progress 文件（_session_id_from_progress 的缓存后端）。"""
    fallback = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                sid = (rec.get("result") or {}).get("session_id")
                if rec.get("type") == "done" and isinstance(sid, str) and re.fullmatch(r"\d{8}_\d{6}", sid):
                    return sid
                if rec.get("type") == "log":
                    ts = (rec.get("log") or {}).get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(
                                ts.replace("Z", "+00:00"))
                            fallback = fallback or dt.strftime("%Y%m%d_%H%M%S")
                        except ValueError:
                            continue
    except Exception:
        pass
    return fallback


def _related_files(session_id: str) -> List[Path]:
    """收集某会话关联的全部文件（ledger/logs/reports/runtime 四类）。"""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
        return []
    base = get_project_data_dir()
    files: List[Path] = []

    ledger = base / "experiment_ledger" / f"ledger_{session_id}.jsonl"
    if ledger.exists():
        files.append(ledger)

    log = base / "logs" / f"session_{session_id}.jsonl"
    if log.exists():
        files.append(log)

    # reports: {title}_{sid}.md —— 匹配文件名尾部 _{sid}.md
    reports_dir = base / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("*.md"):
            if f.stem.endswith(f"_{session_id}"):
                files.append(f)
                if f.with_suffix(".result.json").exists():
                    files.append(f.with_suffix(".result.json"))
                revisions = f.parent / (f.stem + ".report-revisions")
                if revisions.is_dir() and not is_link(revisions):
                    files.extend(p for p in revisions.rglob("*") if p.is_file() and not is_link(p))
        # Figures are kept alongside reports and owned by the same session.
        if re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
            for directory in (reports_dir / "artifacts").glob(f"figures_{session_id}_*"):
                if directory.is_dir() and not directory.is_symlink():
                    files.extend(p for p in directory.iterdir()
                                 if p.is_file() and not p.is_symlink())

    # runtime: progress_*.jsonl 文件名是毫秒时间戳，需按内容提取 session_id
    runtime_dir = base / "runtime"
    if runtime_dir.exists():
        for f in runtime_dir.glob("progress_*.jsonl"):
            if _session_id_from_progress(f) == session_id:
                files.append(f)

    # 去重（保留顺序）
    seen = set()
    unique = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _unlink_files(paths: List[Path]) -> Tuple[int, int]:
    """删除文件列表，返回 (删除数, 释放字节)。"""
    removed = 0
    freed = 0
    for p in paths:
        try:
            base = get_project_data_dir().absolute()
            relative = p.absolute().relative_to(base)
            p = workspace_path(base, relative, "history file", must_exist=True)
            size = p.stat().st_size
            p.unlink()
            freed += size
            removed += 1
        except Exception:
            pass
    return removed, freed


def delete_session(session_id: str) -> Tuple[int, int]:
    """删除单次历史复现会话的全部关联文件（ledger/logs/reports/runtime）。

    返回 (删除文件数, 释放字节数)。文件全部不存在时返回 (0, 0)。
    """
    return _unlink_files(_related_files(session_id))


def delete_sessions(session_ids: List[str]) -> Tuple[int, int]:
    """批量删除多个会话的全部关联文件，返回 (删除文件数, 释放字节数)。

    与逐条调用 delete_session 等价，但只做一次删除动作：先按
    _related_files 取并集（输入会话 id 去重、跨会话文件路径去重），
    最后统一 _unlink_files。空列表或全部为未知会话时返回 (0, 0)。

    单个文件删不掉（如仍被运行中的进程占用）只跳过、不中断整批，
    与 _unlink_files 的既有行为一致。
    """
    files: List[Path] = []
    seen = set()
    for sid in dict.fromkeys(session_ids or []):     # 保序去重
        for f in _related_files(sid):
            key = str(f)
            if key not in seen:
                seen.add(key)
                files.append(f)
    return _unlink_files(files)


def clear_sessions() -> Tuple[int, int]:
    """清空全部历史复现会话：ledger/logs/reports 全部文件 + runtime 终态/过期文件。

    保留 runtime 中仍在运行（未终态且未过期）的进度文件，避免破坏正在
    执行的复现任务。返回 (删除文件数, 释放字节数)。
    """
    base = get_project_data_dir()
    targets: List[Path] = []
    for sub in ("experiment_ledger", "logs", "reports"):
        d = base / sub
        if d.exists():
            targets.extend(f for f in d.glob("**/*") if f.is_file())
    targets.extend(f for f in (base / "runtime").glob("*.jsonl")
                   if f.is_file() and (_is_finished_progress(f) or
                                       f.stat().st_mtime <
                                       time.time() - 7 * 86400))
    return _unlink_files(targets)


# ---------------- 依赖缓存管理（data/deps/） ----------------
# 与写入方 code_executor 保持一致，避免加载执行器及其 Agent 依赖。
_DEPS_META_NAME = "meta.json"


def _deps_root(data_root=None) -> Path:
    """依赖缓存根目录（尊重 AUTOREPRO_DEPS_ROOT 覆盖，与执行器同口径）。"""
    override = os.environ.get("AUTOREPRO_DEPS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().absolute()
    return (get_data_root(data_root) if data_root is not None else get_project_data_dir()) / "deps"


def _dir_size(path: Path) -> Tuple[int, int]:
    """返回 (文件数, 字节数)。"""
    return _shared_dir_size(path, {})


def _shared_dir_size(path: Path, sizes) -> Tuple[int, int]:
    """Scan each directory once and retain child totals for this UI snapshot.

    DirEntry reuses directory-entry metadata on Windows, avoiding repeated
    Path.is_file/stat calls for every installed dependency. Symbolic links and
    Windows junctions are excluded; vanished directories tolerate live cleanup.
    """
    key = os.path.normcase(os.path.abspath(path))
    if key in sizes:
        return sizes[key]
    if is_link(path):
        return 0, 0
    count = total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                    if (entry.is_symlink() or getattr(info, "st_file_attributes", 0)
                            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        child_count, child_bytes = _shared_dir_size(Path(entry.path), sizes)
                        count += child_count
                        total += child_bytes
                    elif entry.is_file(follow_symlinks=False):
                        count += 1
                        total += info.st_size
                except OSError:
                    continue
    except OSError:
        pass
    sizes[key] = (count, total)
    return sizes[key]


def _dependency_directories(root):
    """Only environment leaves are selectable; namespace directories never are."""
    yield from environment_directories(root)


def list_deps_cache(*, _sizes=None, data_root=None) -> List[Dict[str, Any]]:
    """列出依赖缓存里每个隔离安装目录。

    依赖缓存按「归一化后的依赖清单」哈希寻址，**跨论文跨会话共享**，因此
    它不属于任何一次复现记录，不随删除历史一起清（删了下次要重新下载安装）。
    但会随论文数量无限增长，所以需要一个看得见、删得掉的入口。

    每个条目：
      name        相对缓存根的环境路径（哈希 / heal-<模块> / repository/<runtime>/<哈希>）
      bytes/files 占用
      kind        reqs | heal | legacy（无 meta.json 的旧目录）
      packages    装了哪些包（取 *.dist-info 名字）
      requirements 原始依赖清单（旧目录为空）
      installed_at / last_used  ISO 时间串（旧目录回退到目录 mtime）
    """
    root = _deps_root(data_root)
    items: List[Dict[str, Any]] = []
    if not root.is_dir():
        return items
    for p in _dependency_directories(root):
        try:
            name = p.relative_to(root).as_posix()
            p = workspace_path(root, name, "dependency cache", must_exist=True)
        except (OSError, ValueError):
            continue
        meta = {}
        try:
            meta = json.loads((p / _DEPS_META_NAME).read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {}
        except (OSError, ValueError):
            pass
        try:
            files, size = _dir_size(p) if _sizes is None else _shared_dir_size(p, _sizes)
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            continue  # A concurrent cleanup may have removed the displayed item.
        items.append({
            "name": name,
            "path": str(p),
            "bytes": size,
            "files": files,
            "kind": meta.get("kind", "legacy"),
            # dist-info 目录名是 `<包名>-<版本>.dist-info`（包名里的 - 已被
            # 归一为 _），所以从右往左切一次即得包名，不带版本号。
            "packages": sorted(d.name[:-len(".dist-info")].rsplit("-", 1)[0]
                               for d in p.glob("*.dist-info")),
            "requirements": meta.get("requirements", ""),
            "module": meta.get("module", ""),
            "installed_at": meta.get("installed_at", ""),
            "last_used": meta.get("last_used", "") or mtime,
            "base_environment": meta.get('base_environment', ''),
            "shared_packages": meta.get('shared_packages', []),
        })
    return sorted(items, key=lambda i: i["last_used"], reverse=True)


def _delete_deps_cache(names=None, *, keep_days=None, on_skip=None):
    root = _deps_root()
    if not root.is_dir():
        return 0, 0
    removed, freed = 0, 0

    def skipped(message):
        logging.getLogger(__name__).info(message)
        if on_skip is not None:
            on_skip(message)

    selected = set(names or [])
    cutoff = datetime.now().astimezone() - timedelta(days=keep_days) if keep_days is not None else None
    try:
        with cache_guard(root, cleanup=True):
            # Discover again under the same lock used by installers and runners.
            inventory = list_deps_cache()
            # Remove deltas first; a base still referenced by a retained delta
            # cannot be removed even when selected explicitly.
            inventory.sort(key=lambda row: not bool(row['base_environment']))
            for item in inventory:
                if cutoff is None and item["name"] not in selected:
                    continue
                if cutoff is not None:
                    try:
                        used = datetime.fromisoformat(item["last_used"]).astimezone()
                    except (ValueError, TypeError):
                        continue
                    if used >= cutoff:
                        continue
                try:
                    references = []
                    for directory in _dependency_directories(root):
                        base = read_metadata(directory).get('base_environment')
                        if not base:
                            continue
                        try:
                            reference = workspace_path(root, base, "base dependency environment")
                            if reference == (root / item['name']).resolve():
                                references.append(directory)
                        except (OSError, ValueError, TypeError):
                            continue
                    if references:
                        skipped(f"依赖缓存 {item['name']} 仍被增量环境引用，已保留")
                        continue
                    target = workspace_path(root, item["name"], "dependency cache", must_exist=True)
                    _, size = _dir_size(target)
                    # Do not trust an earlier UI inventory or a rewritten symlink.
                    target = workspace_path(root, item["name"], "dependency cache", must_exist=True)
                    remove_tree(target)
                    removed += 1
                    freed += size
                except (OSError, ValueError) as exc:
                    skipped(f"依赖缓存 {item['name']} 未删除: {exc}")
    except DependencyCacheBusy as exc:
        skipped(str(exc))
    return removed, freed


def delete_deps_cache(names: List[str], *, on_skip=None) -> Tuple[int, int]:
    """Delete only discovered environment IDs; busy caches are left intact.

    Existing direct-directory IDs remain valid. Nested IDs must exactly match
    an environment returned by list_deps_cache(), never a namespace or root.
    """
    return _delete_deps_cache(names, on_skip=on_skip)


def dependency_maintenance_status():
    """Read policy and the latest atomic receipt without starting collection."""
    from src.dependency_management import cache_policy, read_maintenance
    root = _deps_root()
    policy = cache_policy()
    return {'policy': policy, 'latest': read_maintenance(root),
            'free_bytes': shutil.disk_usage(root).free if root.is_dir() else None}


def cleanup_deps_cache(keep_days: int = 30, *, on_skip=None) -> Tuple[int, int]:
    """清理超过 keep_days 天未使用过的依赖缓存目录。

    冷热以 meta.json 的 last_used 为准（每次命中缓存都会刷新），
    没有 meta.json 的旧目录回退到目录 mtime。返回 (删除目录数, 释放字节数)。
    """
    return _delete_deps_cache(keep_days=keep_days, on_skip=on_skip)


def list_resource_events(limit: int = 500, *, data_root=None) -> List[Dict[str, Any]]:
    """读取下载/安装事件，供前端和 CLI 展示最近资源活动。"""
    base = get_data_root(data_root) if data_root is not None else get_project_data_dir()
    return list(reversed(read_resource_events(base / "resource_events.jsonl", limit)))


def list_resource_inventory(*, deps_items=None, _sizes=None, data_root=None) -> List[Dict[str, Any]]:
    """从磁盘和 manifest 汇总依赖、数据集、代码和权重资源。"""
    base = get_data_root(data_root) if data_root is not None else get_project_data_dir()
    rows: List[Dict[str, Any]] = []
    for item in list_deps_cache(data_root=data_root) if deps_items is None else deps_items:
        rows.append({
            "type": "dependency", "id": item["name"], "paper_id": "",
            "state": "ready" if (Path(item["path"]) / ".ready").is_file()
            else "partial", "bytes": item["bytes"],
            "last_used": item["last_used"],
            "detail": ", ".join(item["packages"]) or item["requirements"],
        })
    manifests = base / "manifests"
    if manifests.is_dir():
        for path in manifests.glob("*.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(manifest, dict) or not isinstance(manifest.get("resources", {}), dict):
                continue
            pid = manifest.get("paper_id", path.stem)
            resources = manifest.get("resources", {})
            for kind, resource_path in resources.items():
                if not isinstance(resource_path, str) or not resource_path:
                    continue
                p = Path(resource_path)
                rows.append({
                    "type": kind, "id": f"{pid}:{kind}", "paper_id": pid,
                    "state": "present" if p.exists() else "missing",
                    "bytes": (_dir_size(p) if _sizes is None else _shared_dir_size(p, _sizes))[1] if p.is_dir()
                    else (p.stat().st_size if p.is_file() else 0),
                    "last_used": manifest.get("created_at", ""),
                    "detail": manifest.get(
                        "dataset_name" if kind == "dataset" else
                        f"{kind}_url", ""),
                })
    return rows


def collect_storage_snapshot() -> Dict[str, Any]:
    """Collect history-tab storage views with one shared in-memory size scan.

    The caller controls refresh/invalidation. No disk cache is written. The
    established storage categories are unchanged; external dependency roots
    and manifest resources are scanned separately only when they are needed.
    """
    sizes = {}
    storage = get_storage_stats(_sizes=sizes)
    deps_items = list_deps_cache(_sizes=sizes)
    inventory = list_resource_inventory(deps_items=deps_items, _sizes=sizes)
    return {"storage": storage, "deps_items": deps_items, "inventory": inventory}


def get_session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    """获取单次会话的详细记录（ledger + logs）。"""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
        return None
    base = get_project_data_dir()
    detail: Dict[str, Any] = {"session_id": session_id, "ledger": [], "logs": []}

    ledger_file = base / "experiment_ledger" / f"ledger_{session_id}.jsonl"
    if ledger_file.exists():
        try:
            with open(ledger_file, "r", encoding="utf-8") as fh:
                detail["ledger"] = [json.loads(line) for line in fh]
        except Exception:
            pass

    log_file = base / "logs" / f"session_{session_id}.jsonl"
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as fh:
                detail["logs"] = [json.loads(line) for line in fh]
        except Exception:
            pass

    return detail if detail["ledger"] or detail["logs"] else None


def format_size(bytes_: int) -> str:
    if bytes_ >= 1024 ** 3:
        return f"{bytes_ / (1024 ** 3):.2f} GB"
    if bytes_ >= 1024 ** 2:
        return f"{bytes_ / (1024 ** 2):.2f} MB"
    if bytes_ >= 1024:
        return f"{bytes_ / 1024:.2f} KB"
    return f"{bytes_} B"
