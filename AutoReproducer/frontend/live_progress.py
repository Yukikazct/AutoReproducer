"""Read-only monitoring helpers; reconnecting never starts a pipeline."""
import json
import re
import time
from datetime import datetime
from pathlib import Path

from frontend.display_text import display_text
from src.safety.paths import is_link, workspace_path


_PROGRESS_NAME = re.compile(r"progress_\d+\.jsonl")
_SCOPED_PROGRESS_NAME = re.compile(r"([0-9a-f]{32})__(progress_\d+\.jsonl)")


def _monitor_root(data_root):
    root = Path(data_root).absolute()
    if any(is_link(parent) for parent in (root, *root.parents)):
        raise ValueError("monitor data root contains a symlink or junction")
    return root


def _recovery_root(data_root, identifier):
    if not re.fullmatch(r"[0-9a-f]{32}", identifier):
        raise ValueError("invalid recovery scope")
    root = workspace_path(_monitor_root(data_root), "recovery/" + identifier,
                          "monitor recovery root", must_exist=True)
    marker = workspace_path(root, ".restart.json", "monitor recovery marker", must_exist=True)
    if not root.is_dir() or not marker.is_file() or marker.stat().st_size > 16384:
        raise ValueError("invalid recovery marker")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or type(record.get("version")) is not int
            or record["version"] != 1 or record.get("original_cleanup") != "unconfirmed"):
        raise ValueError("invalid recovery marker")
    return root


def isolated_progress_root(data_root, path):
    """Return a marked recovery root only for one of its own progress files."""
    try:
        base = _monitor_root(data_root)
        candidate = Path(path).absolute()
        relative = candidate.relative_to(base)
        if (len(relative.parts) != 4 or relative.parts[0] != "recovery"
                or relative.parts[2] != "runtime"
                or not _PROGRESS_NAME.fullmatch(relative.parts[3])):
            return None
        root = _recovery_root(base, relative.parts[1])
        checked = workspace_path(root, "runtime/" + relative.parts[3],
                                 "monitor recovery progress", must_exist=True)
        return root if checked.is_file() else None
    except (OSError, ValueError, TypeError, RuntimeError):
        return None


def progress_run_id(data_root, path):
    """Encode an existing, unlinked progress file as a bounded monitor token."""
    try:
        base = _monitor_root(data_root)
        candidate = Path(path).absolute()
        relative = candidate.relative_to(base)
        if (len(relative.parts) == 2 and relative.parts[0] == "runtime"
                and _PROGRESS_NAME.fullmatch(relative.parts[1])):
            checked = workspace_path(base, relative, "monitor progress", must_exist=True)
            return checked.name if checked.is_file() else None
        root = isolated_progress_root(base, candidate)
        if root is not None:
            return root.name + "__" + candidate.name
    except (OSError, ValueError, TypeError, RuntimeError):
        pass
    return None


def progress_signature(path):
    try:
        info = Path(path).stat()
        return str(path), info.st_mtime_ns, info.st_size
    except (OSError, TypeError):
        return str(path), None, None


def resolve_progress_run(data_root, name):
    if not isinstance(name, str):
        return None
    try:
        if _PROGRESS_NAME.fullmatch(name):
            candidate = workspace_path(_monitor_root(data_root), "runtime/" + name,
                                       "monitor progress", must_exist=True)
        else:
            scoped = _SCOPED_PROGRESS_NAME.fullmatch(name)
            if scoped is None:
                return None
            root = _recovery_root(data_root, scoped[1])
            candidate = workspace_path(root, "runtime/" + scoped[2],
                                       "monitor recovery progress", must_exist=True)
        return str(candidate) if candidate.is_file() else None
    except (OSError, ValueError, TypeError, RuntimeError):
        return None


def recent_progress_runs(data_root):
    try:
        base = _monitor_root(data_root)
        files = []
        try:
            directory = workspace_path(base, "runtime", "monitor runtime directory")
            files.extend(directory.glob("progress_*.jsonl"))
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
        try:
            recovery = workspace_path(base, "recovery", "monitor recovery directory")
            scopes = list(recovery.iterdir()) if recovery.is_dir() else []
        except (OSError, ValueError, TypeError, RuntimeError):
            scopes = []
        if scopes:
            for scope in scopes:
                if re.fullmatch(r"[0-9a-f]{32}", scope.name):
                    try:
                        root = _recovery_root(base, scope.name)
                        directory = workspace_path(root, "runtime", "monitor recovery runtime")
                        files.extend(directory.glob("progress_*.jsonl"))
                    except (OSError, ValueError, TypeError, RuntimeError):
                        continue
        rows = [(path.stat().st_mtime_ns, identifier) for path in files
                if (identifier := progress_run_id(base, path)) is not None]
        return [identifier for _, identifier in sorted(rows, reverse=True)[:10]]
    except (OSError, ValueError, TypeError, RuntimeError):
        return []


def elapsed_seconds(start, end=None):
    try:
        beginning = datetime.fromisoformat(start)
        ending = datetime.fromisoformat(end) if end else datetime.now(beginning.tzinfo)
        return max(0, int((ending - beginning).total_seconds()))
    except (TypeError, ValueError):
        return None


def _byte_size(value):
    if value is None:
        return "未报告"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def render_dependency_progress(snapshot, *, st_module):
    """Display measured installer activity; a quiet process is not a failure."""
    installations = snapshot.get("dependency_installations", [])
    if not installations:
        return
    st = st_module
    current = next((row for row in reversed(installations) if row.get("status") == "running"), installations[-1])
    status = current.get("status")
    phase = {"resolving": "解析依赖", "downloading": "下载文件", "installing": "安装软件包"}.get(current.get("phase"), "准备依赖")
    labels = {"running": "进行中", "success": "安装完成", "error": "安装失败", "cancelled": "已取消",
              "interrupted": "已中断", "unconfirmed": "任务已结束，安装结果未确认"}
    with st.container(border=True):
        st.markdown("### 依赖安装进度")
        st.text(f"{labels.get(status, '等待状态')} · {phase}")
        package = current.get("package") or ("安装/复制中（未提供逐包进度）"
                    if current.get("phase") == "installing" else "等待安装器报告包名")
        st.caption("当前软件包：" + package)
        if current.get("source_domain"):
            st.caption("软件源 / 下载来源：" + current["source_domain"])
        downloaded, total = current.get("downloaded_bytes"), current.get("total_bytes")
        amount = _byte_size(downloaded)
        speed = current.get("download_speed_bps")
        elapsed = current.get("elapsed_s")
        if elapsed is not None and status == "running":
            elapsed += elapsed_seconds(current.get("at")) or 0
        budget = current.get("timeout_s")
        columns = st.columns(4)
        columns[0].metric("当前文件下载量", amount)
        columns[1].metric("最近下载速度", _byte_size(speed) + "/s" if speed is not None else "未报告")
        columns[2].metric("安装已用时", f"{int(elapsed)} 秒" if elapsed is not None else "未报告")
        columns[3].metric("安装时限", f"{int(budget)} 秒" if budget is not None else "未报告")
        if total is not None and total > 0:
            st.caption("文件总大小：" + _byte_size(total))
        elif current.get("phase") == "downloading":
            st.caption("文件总大小：未知（安装器未报告）")
        if (current.get("phase") == "downloading" and total is not None and total > 0
                and downloaded is not None and 0 <= downloaded <= total):
            ratio = downloaded / total
            st.progress(ratio, text=f"当前文件下载 {ratio:.1%}")
        # This panel runs in the live fragment, so the activity age advances
        # even when the process has no new output or progress-file updates.
        last_activity = current.get("last_activity_at")
        age = max(0, int(time.time() - last_activity)) if last_activity is not None else None
        if status == "running":
            heartbeat_age = elapsed_seconds(current.get("at"))
            if heartbeat_age is not None and heartbeat_age >= 10:
                st.info(f"监控记录已 {heartbeat_age} 秒未更新，暂时无法确认安装进程状态。")
            if age is None:
                st.info("尚未收到安装活动时间，继续等待安装器反馈。")
            elif age >= 30:
                st.info(f"已有 {age} 秒未收到下载字节增长或安装输出；尚未收到安装结束事件，尚不能据此判定失败。")
            else:
                st.caption(f"最近活动：{age} 秒前（安装输出或下载字节增长）。")
        message = current.get("message")
        if message:
            (st.error if status == "error" else st.caption)(display_text(message))
        history = current.get("history", [])
        if history:
            with st.expander("最近安装记录", expanded=True):
                lines = [f"{row.get('at', '')[11:19]}  {row.get('package') or ''}  {display_text(row.get('message', ''))}"
                         for row in history[-20:]]
                st.text("\n".join(lines))


def render_activity(snapshot, *, st_module):
    st = st_module
    stages = snapshot.get("pipeline_stages", [])
    titles = {row["id"]: display_text(row.get("title", row["id"])) for row in stages}
    activities = snapshot.get("activities", [])
    st.markdown("### 各阶段运行记录")
    options = [""] + list(titles)
    # A recovered/new task can have a different set of stages.
    if st.session_state.get("live_phase_filter", "") not in options:
        st.session_state.live_phase_filter = ""
    selected = st.selectbox("查看阶段", options, key="live_phase_filter",
                            format_func=lambda value: titles.get(value, "全部阶段"))
    rows = [row for row in activities if not selected or row["phase_id"] == selected]
    if rows:
        labels = {"running": "进行中", "start": "开始", "success": "完成",
                  "completed": "完成", "error": "错误", "warning": "提醒",
                  "waiting": "等待", "interrupted": "中断", "skipped": "跳过"}
        lines = []
        for row in rows[-100:]:
            name = titles.get(row["phase_id"]) or display_text(row.get("agent", "")) or "准备任务"
            status = labels.get(row.get("status", "").lower(), row.get("status", ""))
            lines.append(f"{row.get('at', '')[11:19]}  {name} · {status}\n{display_text(row['message'])}")
        with st.container(height=280):
            st.text("\n\n".join(lines))
        st.caption("显示所选阶段最近 100 条记录；完整决策详情见“审计日志”，训练输出见下方。")
    else:
        st.info("该阶段尚未产生过程记录。页面会继续等待后台事件。")
