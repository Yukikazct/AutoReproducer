"""Read-only monitoring helpers; reconnecting never starts a pipeline."""
import re
import time
from datetime import datetime
from pathlib import Path

from frontend.display_text import display_text


def progress_signature(path):
    try:
        info = Path(path).stat()
        return str(path), info.st_mtime_ns, info.st_size
    except (OSError, TypeError):
        return str(path), None, None


def resolve_progress_run(data_root, name):
    if not isinstance(name, str) or not re.fullmatch(r"progress_\d+\.jsonl", name):
        return None
    directory = (Path(data_root) / "runtime").resolve()
    candidate = (directory / name).resolve()
    return str(candidate) if candidate.parent == directory and candidate.is_file() else None


def recent_progress_runs(data_root):
    directory = Path(data_root) / "runtime"
    try:
        files = sorted(directory.glob("progress_*.jsonl"),
                       key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return [path.name for path in files[:10]
                if resolve_progress_run(data_root, path.name)]
    except OSError:
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
