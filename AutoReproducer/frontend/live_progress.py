"""Read-only monitoring helpers; reconnecting never starts a pipeline."""
import re
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
