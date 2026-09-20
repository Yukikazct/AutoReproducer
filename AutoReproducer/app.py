"""AutoReproducer - Streamlit 前端界面

修复与增强：
- 论文标题 / 上传 PDF 正确传递到 PaperReader；
- 侧边栏 LLM API 配置（OpenAI 兼容端点 / Key / 模型）真实生效；
- 展示优化结果（最优方向/改进幅度）与 LLM 预算统计；
- 复现流水线后台线程执行，前端轮询进度文件实时展示当前阶段
  （OpenAI 兼容端点 / Key / 模型真实生效）。
"""
import os
import shutil
import sys
import tempfile
import time

import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger
from src.corpus import list_papers
from frontend.llm_config import (
    resolve_llm_config,
    config_missing,
    test_llm_connection,
)
from frontend.backend_pipeline import (
    AGENTS,
    ProgressStore,
    run_pipeline_background,
)
from frontend.history_manager import (
    list_sessions,
    get_storage_stats,
    cleanup_runtime,
    format_size,
    get_session_detail,
    delete_session,
    clear_sessions,
)

# 页面自动刷新（可选依赖）：未安装时退化为手动刷新
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# 页面配置
st.set_page_config(
    page_title="AutoReproducer - 论文自动复现系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 样式
st.markdown("""
<style>
    .status-ok { color: #00ff00; font-weight: bold; }
    .status-error { color: #ff0000; font-weight: bold; }
    .status-running { color: #ffaa00; font-weight: bold; }
    .status-waiting { color: #888888; }
    .agent-card {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .stApp header {display: none;}
    .main-title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        margin-top: 0;
    }
    div[data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 400px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session 状态
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "current_state" not in st.session_state:
    st.session_state.current_state = "INIT"
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "mock_mode" not in st.session_state:
    st.session_state.mock_mode = True
if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""
if "connection_result" not in st.session_state:
    st.session_state.connection_result = None
if "progress_file" not in st.session_state:
    st.session_state.progress_file = None
if "pipeline_note" not in st.session_state:
    st.session_state.pipeline_note = None


# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/idea.png", width=60)
    st.markdown("## ⚙️ 控制面板")

# 模式选择
    st.session_state.mock_mode = st.toggle(
        "🧪 Mock模式（无需API）",
        value=st.session_state.mock_mode,
        help="启用Mock模式可直接演示，无需连接任何LLM服务")

    # Docker 沙箱执行开关（真实模式生效）
    if "use_docker" not in st.session_state:
        st.session_state.use_docker = True
    docker_available = shutil.which("docker") is not None
    st.session_state.use_docker = st.toggle(
        "🐳 Docker 沙箱执行（真实模式）",
        value=st.session_state.use_docker,
        disabled=st.session_state.mock_mode or not docker_available,
        help="真实模式下启用 Docker 隔离执行：依赖在容器内安装，"
             "不污染本机环境；未安装 Docker 或 Mock 模式时自动降级为本地隔离执行")
    if st.session_state.mock_mode:
        st.caption("🧪 Mock 模式不执行真实代码，无需 Docker")
    elif not docker_available:
        st.caption("⚠️ 未检测到 Docker，将使用本地隔离执行（依赖安装在隔离目录）")
    else:
        st.caption(f"✅ Docker 已就绪 ({'将使用' if st.session_state.use_docker else '未启用，将使用本地隔离执行'})")

    # LLM API 配置（真实模式；OpenAI 兼容接口，不依赖本地部署）
    with st.expander("🔗 LLM API 配置", expanded=not st.session_state.mock_mode):
        base_url = st.text_input(
            "API 地址（OpenAI 兼容）",
            value=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            placeholder="如 https://api.deepseek.com",
            help="支持 DeepSeek / 千帆 / OpenAI 等任意 OpenAI 兼容端点",
            disabled=st.session_state.mock_mode)
        api_key = st.text_input(
            "API Key",
            value=os.environ.get("LLM_API_KEY", ""),
            type="password",
            help="远程 API 的访问密钥（无鉴权服务可留空）",
            disabled=st.session_state.mock_mode)
        model_name = st.text_input(
            "模型名称",
            value=os.environ.get("LLM_MODEL", "deepseek-chat"),
            placeholder="如 deepseek-chat / ernie-4.0-8k / gpt-4o-mini",
            disabled=st.session_state.mock_mode)

        # 连接测试：真实调用一次 Chat Completions，验证 API 配置可用
        st.markdown("---")
        link_btn = st.button(
            "🔌 测试 AI 连接",
            disabled=st.session_state.mock_mode,
            use_container_width=True,
            help="真实调用一次 LLM API，验证地址/Key/模型配置可用")
        if st.session_state.mock_mode:
            st.caption("🧪 Mock 模式不调用真实 LLM，连接测试不可用；"
                       "关闭 Mock 开关后可输入 API 并测试")
        else:
            _cfg = resolve_llm_config(base_url, api_key, model_name)
            st.caption(f"当前生效: `{_cfg['model']}` @ `{_cfg['base_url']}`"
                       "（输入留空时回退环境变量）")
            _cr = st.session_state.connection_result
            if _cr:
                ok, msg = _cr
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # 预算上限
    max_trials = st.slider(
        "🎯 优化预算（UCB 尝试次数）", min_value=3, max_value=20, value=10,
        help="Optimizer 在复现成功后最多尝试的优化方向次数")

    # 论文输入
    st.markdown("### 📄 论文输入")
    input_mode = st.radio("输入方式", ["论文标题", "上传PDF"], key="input_mode")

    paper_title = st.session_state.paper_title
    uploaded_file = None

    if input_mode == "论文标题":
        paper_title = st.text_input(
            "论文标题",
            value=st.session_state.paper_title,
            placeholder="输入论文标题...",
            key="paper_title_input")
        st.session_state.paper_title = paper_title
    else:
        uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"],
                                         key="pdf_uploader")

    # 语料对照层（可选）：选择真实论文作为轻量锚点
    st.markdown("### 🗂️ 语料对照(可选)")
    _corpus = [p["id"] for p in list_papers()]
    _corpus_choice = st.selectbox(
        "选择 PaperGuru-Benchmark 论文", ["无"] + _corpus, index=0,
        key="corpus_paper_select")
    corpus_paper = None if _corpus_choice == "无" else _corpus_choice

    # 启动 / 重置按钮
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始复现", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.running)
    with col2:
        reset_btn = st.button("🔄 重置", use_container_width=True)

    # 系统状态
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    state_colors = {
        "INIT": "⚪", "READ_PAPER": "📖", "FIND_RESOURCES": "🔍",
        "BUILD_ENV": "🔧", "EXECUTE_CODE": "⚡", "VALIDATE": "✅",
        "OPTIMIZING": "🧪", "OPTIMIZED": "🏆",
        "GENERATE_REPORT": "📝", "COMPLETED": "🎉", "ERROR": "❌",
    }
    st.markdown(
        f"**当前状态**: {state_colors.get(st.session_state.current_state, '⚪')} "
        f"`{st.session_state.current_state}`")


# ========== 主界面 ==========
st.markdown('<p class="main-title">🔬 AutoReproducer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于多智能体协作的论文自动复现与优化系统</p>',
            unsafe_allow_html=True)

# 标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 流水线状态", "📄 复现报告", "📜 审计日志", "🔍 状态机", "📂 历史记录",
])

# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("### 🏗️ 复现流水线（复现 -> 验证 -> 优化 -> 报告）")

    AGENT_DESC = {
        "📖 PaperReader": ("论文解析", "从PDF/标题中提取结构化信息"),
        "🔍 ResourceFinder": ("资源查找", "定位代码仓库和数据集"),
        "🔧 EnvBuilder": ("环境构建", "自动搭建环境 + 依赖诊断"),
        "⚡ CodeExecutor": ("代码执行", "smoke + full 双阶段执行"),
        "✅ ResultValidator": ("结果验证", "比对论文声明值与运行结果"),
        "🛡️ Verifier": ("质量验证", "Prompt-Free 检查质量 + 修正闭环"),
        "🧪 Optimizer": ("智能优化", "UCB 预算调度, Keep/Reject"),
        "📝 ReportGenerator": ("报告生成", "生成复现+优化 Markdown 报告"),
    }
    names = [a[1] for a in AGENTS] + ["🛡️ Verifier", "🧪 Optimizer",
                                      "📝 ReportGenerator"]

    # ---------- 后台复现实时进度（轮询进度文件） ----------
    pf = st.session_state.progress_file
    snap = None
    if pf:
        snap = ProgressStore.read_snapshot(pf)
    if snap:
        if snap["result"]:
            st.session_state.result = snap["result"]
        if not snap["running"]:
            # done 或 error 两种终态都会把 running 置 False；后台提前失败时
            # 没有 result 事件，但必须清除 running，否则「开始复现」永久禁用。
            st.session_state.running = False
        if snap.get("agent_status"):
            for k, v in snap["agent_status"].items():
                st.session_state.agent_status[k] = v
        if snap.get("state"):
            st.session_state.current_state = snap["state"]
        if snap.get("logs"):
            st.session_state.logs = snap["logs"]
        if snap.get("error"):
            st.error(f"❌ 后台流水线异常: {snap['error']}")

    if st.session_state.running and pf:
        if st_autorefresh is not None:
            st_autorefresh(interval=2000, key=f"ar_{pf}")
            st.info("🔄 复现流水线正在后台运行，页面每 2 秒自动刷新，"
                    "实时展示各 Agent 进度。")
        else:
            st.warning("未安装 streamlit-autorefresh，页面不会自动刷新；"
                       "可刷新浏览器页面查看最新进度。")

    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            status = st.session_state.agent_status.get(name, "waiting")
            status_icons = {"success": "✅", "error": "❌",
                            "running": "🔄", "waiting": "⏳"}
            status_colors = {
                "success": "border-left: 4px solid #4CAF50;",
                "error": "border-left: 4px solid #f44336;",
                "running": "border-left: 4px solid #FF9800;",
                "waiting": "border-left: 4px solid #9E9E9E;",
            }
            icon = status_icons.get(status, "⏳")
            border = status_colors.get(status, "")
            title, desc = AGENT_DESC.get(name, ("", ""))
            st.markdown(f"""
            <div class="agent-card" style="{border}">
                <h4>{icon} {name}</h4>
                <small>{title}</small><br>
                <span style="color: #888;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    completed = sum(1 for a in names
                    if st.session_state.agent_status.get(a) == "success")
    progress = completed / len(names) if names else 0
    st.progress(progress, text=f"整体进度: {completed}/{len(names)}")

    # 运行结果展示
    if st.session_state.result:
        result = st.session_state.result
        st.markdown("---")
        st.markdown("### 📊 运行摘要")
        stats = result.get("audit_stats", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总步骤数", stats.get("total_steps", 0))
        c2.metric("成功", stats.get("success", 0))
        c3.metric("错误", stats.get("errors", 0))
        c4.metric("耗时(秒)", f"{stats.get('duration_sec', 0):.1f}")
        c5.metric("LLM调用", stats.get("llm_calls", 0))

        data = result.get("data", {}) or {}
        optimization = data.get("optimization", {}) or {}
        if optimization.get("optimized"):
            o1, o2, o3 = st.columns(3)
            o1.metric("最优优化方向", str(optimization.get("best_arm", "无"))[:18])
            o2.metric("改进幅度", f"{optimization.get('improvement', 0):.2%}")
            o3.metric("预算使用", f"{optimization.get('budget_used', 0)}/"
                      f"{optimization.get('budget', 0)}")

        if result.get("state") == "COMPLETED":
            st.success("🎉 复现流程成功完成！")
        elif result.get("state") == "ERROR":
            st.error(f"❌ 流程出错: {result.get('error', '未知错误')}")

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown("### 📄 复现与优化报告")
    if st.session_state.result and st.session_state.result.get("data", {}).get("report"):
        report = st.session_state.result["data"]["report"]
        st.markdown(report)
    else:
        st.info("运行复现流程后，这里将显示完整的复现与优化报告。")

# ===== Tab 3: 审计日志 =====
with tab3:
    st.markdown("### 📜 审计日志")
    if st.session_state.logs:
        col1, col2 = st.columns(2)
        with col1:
            filter_agent = st.selectbox(
                "按Agent筛选",
                ["全部"] + sorted(
                    set(l.get("agent", "") for l in st.session_state.logs)),
                key="filter_agent_tab3")
        with col2:
            filter_status = st.selectbox(
                "按状态筛选",
                ["全部", "SUCCESS", "ERROR", "START", "RUNNING", "WARNING"],
                key="filter_status_tab3")

        filtered_logs = st.session_state.logs
        if filter_agent != "全部":
            filtered_logs = [l for l in filtered_logs
                             if l.get("agent") == filter_agent]
        if filter_status != "全部":
            filtered_logs = [l for l in filtered_logs
                             if l.get("status") == filter_status]

        for log in filtered_logs:
            status_color = {"SUCCESS": "🟢", "ERROR": "🔴", "START": "🟡",
                            "RUNNING": "🔄", "WARNING": "🟠"}.get(
                                log.get("status", ""), "⚪")
            with st.expander(
                f"{status_color} [{log.get('elapsed_sec', 0):.1f}s] "
                f"{log.get('agent', '?')} - {log.get('action', '?')}"
            ):
                st.json(log)
    else:
        st.info("运行复现流程后，这里将显示详细的审计日志。")

# ===== Tab 4: 状态机 =====
with tab4:
    st.markdown("### 🔍 状态机定义")
    st.markdown("系统使用有限状态机（FSM）管理 Agent 的流转。")
    state_info = """
```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> READ_PAPER
    READ_PAPER --> FIND_RESOURCES
    FIND_RESOURCES --> BUILD_ENV
    BUILD_ENV --> EXECUTE_CODE
    EXECUTE_CODE --> VALIDATE
    VALIDATE --> OPTIMIZING: 复现成功
    VALIDATE --> GENERATE_REPORT: 复现失败
    OPTIMIZING --> OPTIMIZED
    OPTIMIZED --> GENERATE_REPORT
    GENERATE_REPORT --> COMPLETED
    READ_PAPER --> ERROR
    FIND_RESOURCES --> ERROR
    BUILD_ENV --> ERROR
    EXECUTE_CODE --> ERROR
    VALIDATE --> ERROR
    OPTIMIZING --> ERROR
    GENERATE_REPORT --> ERROR
    ERROR --> INIT
    COMPLETED --> [*]
```
"""
    st.markdown(state_info)

    st.markdown("### 状态说明")
    state_data = [
        {"状态": "INIT", "说明": "初始化，等待输入", "Agent": "—"},
        {"状态": "READ_PAPER", "说明": "解析论文PDF/标题,提取结构化信息", "Agent": "PaperReader"},
        {"状态": "FIND_RESOURCES", "说明": "查找代码仓库和数据集", "Agent": "ResourceFinder"},
        {"状态": "BUILD_ENV", "说明": "构建环境 + 5轮依赖诊断", "Agent": "EnvBuilder"},
        {"状态": "EXECUTE_CODE", "说明": "smoke+full 双阶段执行", "Agent": "CodeExecutor"},
        {"状态": "VALIDATE", "说明": "验证结果与论文一致性", "Agent": "ResultValidator"},
        {"状态": "OPTIMIZING", "说明": "复现成功后,UCB 预算调度优化", "Agent": "Optimizer"},
        {"状态": "OPTIMIZED", "说明": "优化完成,产出最优方案", "Agent": "Optimizer"},
        {"状态": "GENERATE_REPORT", "说明": "生成Markdown复现+优化报告", "Agent": "ReportGenerator"},
        {"状态": "COMPLETED", "说明": "流水线完成", "Agent": "—"},
        {"状态": "ERROR", "说明": "出错状态，可重试", "Agent": "—"},
    ]
    st.table(state_data)


# ===== Tab 5: 历史记录 =====
with tab5:
    st.markdown("### 📂 复现历史与存储管理")

    # -- 当前会话报告下载（如果本次复现已完成） --
    if st.session_state.result and st.session_state.result.get("report_path"):
        rp = st.session_state.result["report_path"]
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as fh:
                report_content = fh.read()
            st.download_button(
                "⬇️ 下载本次复现报告",
                data=report_content,
                file_name=os.path.basename(rp),
                mime="text/markdown",
                use_container_width=True,
            )

# -- 存储仪表板 --
    st.markdown("#### 💾 存储占用")
    try:
        storage = get_storage_stats()
        cols = st.columns(3)
        metrics = [
            ("实验账本", "experiment_ledger"),
            ("审计日志", "logs"),
            ("实时进度", "runtime"),
            ("复现报告", "reports"),
            ("优化产物", "optimization_demo"),
            ("数据集", "datasets"),
            ("依赖缓存", "deps"),
            ("代码仓库", "repos"),
            ("清单/归档", "manifests"),
            ("归档快照", "archive"),
            ("其他", "pinn-output"),
        ]
        for i, (label, key) in enumerate(metrics):
            with cols[i % 3]:
                info = storage.get(key, {"files": 0, "bytes": 0})
                st.metric(label, f"{info['files']} 文件", format_size(info["bytes"]))
        st.caption(f"总计: {format_size(storage.get('total', {}).get('bytes', 0))}")
    except Exception as e:
        st.error(f"获取存储统计失败: {e}")

    # -- 一键清理 --
    with st.expander("🧹 清理管理"):
        keep_days = st.slider("保留 runtime 文件天数", 1, 30, 7)
        if st.button("清理过期/终态 runtime 文件", use_container_width=True,
                     help="删除已结束复现（done/error）遗留的进度文件，以及超过保留天数的旧进度文件"):
            removed, freed = cleanup_runtime(keep_days=keep_days)
            st.success(f"已删除 {removed} 个文件，释放 {format_size(freed)}")
            st.rerun()

        st.markdown("---")
        st.markdown("**🗑️ 历史会话清理**（危险操作，请谨慎）")
        confirm_clear = st.checkbox(
            "我确认清空全部历史复现会话（实验账本/审计日志/复现报告/runtime 进度）",
            key="confirm_clear_all")
        if st.button("🧹 清空全部历史", use_container_width=True,
                     type="secondary",
                     disabled=not confirm_clear):
            removed, freed = clear_sessions()
            st.success(f"已清空全部历史：删除 {removed} 个文件，"
                       f"释放 {format_size(freed)}")
            st.rerun()

    # -- 历史会话列表 --
    st.markdown("#### 📋 历史复现会话")
    try:
        sessions = list_sessions()
        if not sessions:
            st.info("暂无历史复现记录。完成一次复现后将在此展示。")
        else:
            for sess in sessions:
                sid = sess["session_id"]
                title = sess.get("paper_title", "未知论文")
                state = sess.get("state", "未知")
                state_icon = {"COMPLETED": "✅", "ERROR": "❌"}.get(state, "⏳")
                duration = sess.get("duration_sec", 0)
                llm_calls = sess.get("llm_calls", 0)
                log_entries = sess.get("log_entries", 0)
                report_path = sess.get("report_path", "")

                with st.expander(f"{state_icon} [{sid}] {title[:40] or '无标题'}..."):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("状态", state)
                    c2.metric("耗时", f"{duration:.1f}s")
                    c3.metric("LLM调用", llm_calls)
                    st.caption(f"日志条目: {log_entries}")

                    # 报告下载
                    if report_path and os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as fh:
                            report_data = fh.read()
                        st.download_button(
                            "⬇️ 下载报告",
                            data=report_data,
                            file_name=os.path.basename(report_path),
                            mime="text/markdown",
                            key=f"dl_{sid}",
                        )
                    else:
                        st.caption("无报告文件")

# 详情按钮
                    if st.button("查看详情", key=f"detail_{sid}"):
                        detail = get_session_detail(sid)
                        if detail:
                            st.markdown("**审计日志片段（最近5条）:**")
                            for entry in detail.get("logs", [])[-5:]:
                                st.json(entry)
                        else:
                            st.warning("未找到详情")

                    # 删除本会话（危险操作：需勾选确认）
                    st.markdown("---")
                    confirm_del = st.checkbox(
                        "确认删除本会话（账本/日志/报告/progress 一并删除）",
                        key=f"confirm_del_{sid}")
                    if st.button("🗑️ 删除本会话", key=f"del_btn_{sid}",
                                 type="secondary",
                                 disabled=not confirm_del):
                        removed, freed = delete_session(sid)
                        st.success(f"已删除该会话 {removed} 个文件，"
                                   f"释放 {format_size(freed)}")
                        st.rerun()
    except Exception as e:
        st.error(f"加载历史记录失败: {e}")


# ========== 事件处理 ==========
def _save_uploaded_pdf(uploaded_file) -> str:
    """将上传的 PDF 保存为临时文件，返回路径。"""
    suffix = os.path.splitext(uploaded_file.name or "paper.pdf")[1] or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="autorepro_paper_")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


def _new_progress_file() -> str:
    """创建本次复现的进度文件路径（data/runtime/progress_<毫秒>.jsonl）。"""
    runtime_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    return os.path.join(runtime_dir,
                        f"progress_{int(time.time() * 1000)}.jsonl")


# 启动按钮处理：后台线程执行流水线，主线程立即返回并轮询进度
if start_btn:
    pt = st.session_state.paper_title or ""
    if not pt and not uploaded_file:
        st.error("请先输入论文标题或上传PDF文件")
    elif not st.session_state.mock_mode and config_missing(base_url, model_name):
        st.error("真实模式缺少 LLM 配置（"
                 + "、".join(config_missing(base_url, model_name))
                 + "）。请在侧边栏填写，或设置环境变量 "
                   "LLM_BASE_URL / LLM_MODEL 后重试。")
    else:
        # 真实模式且未填 API Key：多数云端端点（DeepSeek/OpenAI 等）会返回
        # 401，且流水线会把错误文本当 LLM 输出继续跑，表象类似「没反应」。
        # 此处不阻断（部分自建端点无需鉴权），但给出明确预警。
        if (not st.session_state.mock_mode
                and not (api_key.strip()
                         or os.environ.get("LLM_API_KEY", "").strip())):
            st.warning("⚠️ 未填写 API Key：如果上游服务需要鉴权"
                       "（如 DeepSeek/OpenAI），调用会返回 401 错误文本；"
                       "建议先在侧边栏填写 Key 并点击「🔌 测试 AI 连接」验证。")
        tmp_pdf = _save_uploaded_pdf(uploaded_file) if uploaded_file else ""
        progress_file = _new_progress_file()
        st.session_state.progress_file = progress_file
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.logs = []
        st.session_state.agent_status = {}
        st.session_state.current_state = "INIT"
        st.session_state.pipeline_note = (
            "复现流水线已在后台启动，进度实时刷新中…")
        run_pipeline_background(
            progress_file,
            paper_title=pt, pdf_path=tmp_pdf,
            corpus_paper=corpus_paper,
            model_name=model_name, base_url=base_url,
            api_key=api_key,
            mock_mode=st.session_state.mock_mode,
            max_trials=max_trials,
            use_docker=(not st.session_state.mock_mode
                        and docker_available
                        and st.session_state.use_docker),
            cleanup_pdf=True)   # 临时 PDF 由后台线程负责删除
        st.rerun()

# 测试连接按钮处理
if link_btn:
    st.session_state.connection_result = None
    with st.spinner("正在测试 API 连接..."):
        ok, msg = test_llm_connection(base_url, api_key, model_name)
    st.session_state.connection_result = (ok, msg)
    st.rerun()

# 重置按钮处理
if reset_btn:
    st.session_state.orchestrator = None
    st.session_state.result = None
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.current_state = "INIT"
    st.session_state.agent_status = {}
    st.session_state.connection_result = None
    st.session_state.progress_file = None
    st.session_state.pipeline_note = None
    st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    AutoReproducer v0.2.0 | 基于多智能体协作的论文自动复现与优化系统
</div>
""", unsafe_allow_html=True)