"""AutoReproducer - Streamlit 前端界面

修复与增强：
- 论文标题 / 上传 PDF 正确传递到 PaperReader；
- 侧边栏 LLM API 配置（OpenAI 兼容端点 / Key / 模型）真实生效；
- 展示优化结果（最优方向/改进幅度）与 LLM 预算统计；
- 复现流水线后台线程执行，前端轮询进度文件实时展示当前阶段
  （OpenAI 兼容端点 / Key / 模型真实生效）。
"""
import os
import sys
import tempfile
import time
from html import escape
from pathlib import Path

import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger
from src.base_agent import BaseAgent
from src.corpus import list_papers
from src.repository_profiles import PROFILE_LABELS, PAPER_TITLE
from frontend.report_renderer import render_report, build_report_bundle
from frontend.llm_config import (
    CONNECT_TEST_TIMEOUT,
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
    list_deps_cache,
    delete_deps_cache,
    cleanup_deps_cache,
    list_resource_events,
    list_resource_inventory,
    format_size,
    get_session_detail,
    delete_session,
    delete_sessions,
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

# 页面样式独立维护，避免业务逻辑与大段 CSS 混在一起。
_styles = Path(__file__).resolve().parent / "frontend" / "styles.css"
st.markdown(f"<style>{_styles.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

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
    st.session_state.mock_mode = False
if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""
if "connection_result" not in st.session_state:
    st.session_state.connection_result = None
if "progress_file" not in st.session_state:
    # 定向重启服务时可恢复已完成报告，不重新调用模型或启动任务。
    resume = os.environ.get("AUTOREPRO_RESUME_PROGRESS", "")
    previous = ProgressStore.read_snapshot(resume) if resume else None
    st.session_state.progress_file = (
        resume if previous and previous["done"] and previous["result"] else None)
if "pipeline_note" not in st.session_state:
    st.session_state.pipeline_note = None

# 先读取后台状态，再渲染侧边栏和主区域。终态时停止轮询后也能立即
# 更新「开始复现」按钮、系统状态和顶部提示，避免留在上一阶段。
pf = st.session_state.progress_file
snap = ProgressStore.read_snapshot(pf) if pf else None
if snap:
    if snap["result"]:
        st.session_state.result = snap["result"]
    if not snap["running"]:
        st.session_state.running = False
    if snap.get("agent_status"):
        st.session_state.agent_status.update(snap["agent_status"])
    if snap.get("state"):
        st.session_state.current_state = snap["state"]
    if snap.get("logs"):
        st.session_state.logs = snap["logs"]


@st.fragment
def render_llm_settings():
    """配置和连接测试局部刷新，反馈直接显示在按钮旁。"""
    with st.expander("🔗 LLM API 配置", expanded=not st.session_state.mock_mode):
        base_url = st.text_input(
            "API 地址（OpenAI 兼容）",
            value=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            key="llm_base_url",
            placeholder="如 https://api.deepseek.com",
            help="支持 DeepSeek / 千帆 / OpenAI 等任意 OpenAI 兼容端点",
            disabled=st.session_state.mock_mode)
        api_key = st.text_input(
            "API Key", value=os.environ.get("LLM_API_KEY", ""),
            key="llm_api_key", type="password",
            help="远程 API 的访问密钥（无鉴权服务可留空）",
            disabled=st.session_state.mock_mode)
        model_name = st.text_input(
            "模型名称", value=os.environ.get("LLM_MODEL", "deepseek-chat"),
            key="llm_model",
            placeholder="如 deepseek-chat / ernie-4.0-8k / gpt-4o-mini",
            disabled=st.session_state.mock_mode)

        cfg = resolve_llm_config(base_url, api_key, model_name)
        # 改配置后清掉旧结果，避免新地址/模型仍显示上次的「连接成功」。
        if cfg != st.session_state.get("connection_config"):
            st.session_state.connection_result = None
            st.session_state.connection_config = cfg

        st.markdown("---")
        test_clicked = st.button(
            "🔌 测试 AI 连接", key="test_llm_connection",
            disabled=st.session_state.mock_mode,
            use_container_width=True,
            help="真实调用一次 LLM API，验证地址/Key/模型配置可用")
        feedback = st.empty()
        if test_clicked:
            st.session_state.connection_result = None
            feedback.info("正在测试 AI 连接，请稍候…")
            with st.spinner(f"正在请求 API（超时 {CONNECT_TEST_TIMEOUT} 秒）…"):
                st.session_state.connection_result = test_llm_connection(
                    base_url, api_key, model_name)

        if st.session_state.mock_mode:
            st.caption("🧪 Mock 模式不调用真实 LLM，连接测试不可用；"
                       "关闭 Mock 开关后可输入 API 并测试")
        else:
            result = st.session_state.connection_result
            if result:
                ok, msg = result
                if ok:
                    feedback.success(msg)
                else:
                    feedback.error(f"连接失败：{msg}")
            st.caption(f"当前生效: `{cfg['model']}` @ `{cfg['base_url']}`"
                       "（输入留空时回退环境变量）")


def render_paper_input():
    """输入方式切换与页面一起刷新，保持上传入口和启动按钮状态同步。"""
    st.markdown('<div class="sidebar-section-label"><span>01</span> 论文输入</div>',
                unsafe_allow_html=True)
    input_mode = st.radio("输入方式", ["论文标题", "上传PDF", "官方仓库预设"], key="input_mode")
    if input_mode == "论文标题":
        st.session_state.paper_title = st.text_input(
            "论文标题", value=st.session_state.paper_title,
            placeholder="输入论文标题...", key="paper_title_input")
    elif input_mode == "上传PDF":
        uploaded_file = st.file_uploader(
            "上传PDF文件", type=["pdf"], key="pdf_uploader",
            help="点击 Upload 选择本地 PDF，或将 PDF 拖入上传区域")
        if uploaded_file is None:
            st.caption("请点击 Upload 选择 PDF 文件，上传后再开始复现。")
        else:
            st.success(f"已上传：{uploaded_file.name}"
                       f"（{format_size(uploaded_file.size)}）")
    else:
        st.selectbox("真实论文实验", list(PROFILE_LABELS),
                     format_func=lambda key: PROFILE_LABELS[key], key="experiment_profile")
        st.caption(PAPER_TITLE)
        st.markdown("[论文](https://arxiv.org/abs/2205.13504) · "
                    "[作者代码](https://github.com/cure-lab/LTSF-Linear)")
        st.checkbox("仅准备代码、数据和命令（不训练）", key="repository_prepare_only")
        st.checkbox("使用真实多 Agent 分析论文、仓库和环境", value=True, key="repository_llm_review")
        st.checkbox("允许 API 分析本次指标、轮数和核验状态摘要", key="repository_result_review")
        st.caption("完整分析流程需要真实 API；关闭分析选项可直接执行作者实验。"
                   "使用本地 CPU，首次训练会安装隔离依赖；默认执行完整作者训练协议。")


# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">✳</span>
        <div><strong>AutoReproducer</strong><small>RESEARCH WORKSPACE</small></div>
    </div>
    <div class="sidebar-intro">配置复现任务</div>
    <p class="sidebar-help">选择运行模式、提交论文，然后启动研究流水线。</p>
    """, unsafe_allow_html=True)

# 模式选择
    st.session_state.mock_mode = st.toggle(
        "🧪 Mock模式（无需API）",
        value=st.session_state.mock_mode,
        help="启用Mock模式可直接演示，无需连接任何LLM服务")

    # Docker 沙箱执行开关（真实模式生效）。
    # 默认**关闭**：容器沙箱的代价是每次运行都要在容器里现装依赖（tmpfs 随
    # `--rm` 清空，numpy 级别约 20 秒、torch 级别几分钟），默认开着会让用户
    # 的第一次真实运行无声地落在容器里、慢且难解释；本地隔离执行（依赖装进
    # data/deps/<hash>/，多论文共享）才是开箱即跑的那条路。
    # 要用容器随时手动打开——今天它已在真实容器里实测跑通。
    if "use_docker" not in st.session_state:
        st.session_state.use_docker = False
    if "docker_probe" not in st.session_state:
        st.session_state.docker_probe = None       # None = 尚未探测

    # 引擎存活探测：CLI 二进制在 PATH 上 ≠ Docker Desktop 的引擎在跑。
    # 只看 `shutil.which("docker")` 会把「装了没启动」报成「✅ 已就绪」，
    # 随后 `docker run` 甩出 npipe 原始报错（用户实测踩到）。
    # 只在真实模式探测：Mock 模式不执行代码、用不上 Docker，也不必让
    # Mock 用例背上真实探测。结果缓存进 session_state —— 每轮 rerun 都
    # 跑一次探测不划算，而引擎可能被用户中途启动，所以留「重新检测」按钮
    # 作为显式刷新路径。
    if st.session_state.mock_mode:
        docker_available, docker_reason = False, ""
    else:
        if st.session_state.docker_probe is None:
            st.session_state.docker_probe = BaseAgent.docker_engine_available()
        docker_available, docker_reason = st.session_state.docker_probe
        if not docker_available:
            # 引擎不在就把开关拉回关闭：显示开着却跑不了是最坏的组合
            st.session_state.use_docker = False

    st.session_state.use_docker = st.toggle(
        "🐳 Docker 沙箱执行（真实模式）",
        value=st.session_state.use_docker,
        disabled=st.session_state.mock_mode or not docker_available,
        help="真实模式下启用 Docker 隔离执行：依赖在容器内安装，"
             "不污染本机环境；未安装 Docker 或 Mock 模式时自动降级为本地隔离执行")
    if st.session_state.mock_mode:
        st.caption("🧪 Mock 模式不执行真实代码，无需 Docker")
    elif not docker_available:
        st.caption(f"⚠️ {docker_reason}，将使用本地隔离执行（依赖安装在隔离目录）")
        if st.button("🔄 重新检测 Docker", key="docker_recheck",
                     use_container_width=True,
                     help="启动 Docker Desktop 后点此重新探测，无需刷新页面"):
            st.session_state.docker_probe = None
            st.rerun()
    else:
        st.caption("✅ Docker 已就绪 ("
                   + ("将使用容器沙箱，首跑会在容器内安装依赖，偏慢"
                      if st.session_state.use_docker
                      else "未启用容器沙箱，将使用本地隔离执行")
                   + ")")

    render_llm_settings()
    base_url = st.session_state.llm_base_url
    api_key = st.session_state.llm_api_key
    model_name = st.session_state.llm_model

    # 预留可选入口；当前版本不执行优化，也不消耗优化预算。
    st.session_state.enable_optimization = False
    st.checkbox("启用智能优化（预留）", key="enable_optimization", disabled=True)
    st.caption("智能优化暂未开放，当前只执行复现、核验和报告生成。")

    # 论文输入
    render_paper_input()
    input_mode = st.session_state.input_mode
    uploaded_file = (st.session_state.get("pdf_uploader")
                     if input_mode == "上传PDF" else None)
    experiment_profile = (st.session_state.get("experiment_profile")
                          if input_mode == "官方仓库预设" else None)
    requires_api = not experiment_profile or (
        bool(st.session_state.get("repository_llm_review"))
        and not bool(st.session_state.get("repository_prepare_only")))


    # 语料对照层（可选）：选择真实论文作为轻量锚点
    st.markdown('<div class="sidebar-section-label"><span>02</span> 语料对照 <em>可选</em></div>',
                unsafe_allow_html=True)
    _corpus = [p["id"] for p in list_papers()]
    _corpus_choice = st.selectbox(
        "选择 PaperGuru-Benchmark 论文", ["无"] + _corpus, index=0,
        key="corpus_paper_select")
    corpus_paper = None if _corpus_choice == "无" else _corpus_choice
    if corpus_paper:
        st.caption(f"当前任务会使用 {corpus_paper} 的依赖和参考指标。"
                   "仅运行上传的 PDF 时，请将语料对照设为“无”。")

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
    st.markdown('<div class="sidebar-section-label"><span>03</span> 系统状态</div>',
                unsafe_allow_html=True)
    state_colors = {
        "INIT": "⚪", "READ_PAPER": "📖", "FIND_RESOURCES": "🔍",
        "BUILD_ENV": "🔧", "EXECUTE_CODE": "⚡", "VALIDATE": "✅",
        "OPTIMIZING": "🧪", "OPTIMIZED": "🏆",
        "GENERATE_REPORT": "📝", "COMPLETED": "🎉", "ERROR": "❌",
    }
    st.markdown(
        f'<div class="sidebar-status"><span class="status-pulse"></span>'
        f'<span>当前状态</span><strong>'
        f'{state_colors.get(st.session_state.current_state, "⚪")} '
        f'{escape(st.session_state.current_state)}</strong></div>',
        unsafe_allow_html=True)


# ========== 主界面 ==========
_state_label = {
    "INIT": "等待开始", "READ_PAPER": "解析论文", "FIND_RESOURCES": "查找资源",
    "BUILD_ENV": "构建环境", "EXECUTE_CODE": "执行代码", "VALIDATE": "验证结果",
    "OPTIMIZING": "智能优化", "OPTIMIZED": "优化完成",
    "GENERATE_REPORT": "生成报告", "COMPLETED": "任务完成", "ERROR": "运行异常",
}.get(st.session_state.current_state, "运行中")
_hero_status = ("error" if st.session_state.current_state == "ERROR" else
                "success" if st.session_state.current_state == "COMPLETED" else
                "running" if st.session_state.running else "idle")
st.markdown(f"""
<div class="hero">
    <div class="hero-top">
        <span class="hero-eyebrow"><span class="eyebrow-dot"></span> AUTO REPRODUCER / 研究工作台</span>
        <span class="hero-status hero-status-{_hero_status}">{escape(_state_label)}</span>
    </div>
    <div class="hero-content">
        <div>
            <h1>让研究成果，<br><span>被可靠地复现。</span></h1>
            <p>从论文解析到实验执行、结果核验与报告生成，<br>在一个工作台中追踪完整的复现过程。</p>
        </div>
        <div class="hero-visual" aria-hidden="true">
            <div class="visual-orbit orbit-one"></div><div class="visual-orbit orbit-two"></div>
            <div class="visual-core">✳</div>
            <span class="visual-node node-one"></span><span class="visual-node node-two"></span>
            <span class="visual-node node-three"></span>
        </div>
    </div>
    <div class="hero-steps"><span><b>01</b> 解析论文</span><i>→</i><span><b>02</b> 准备实验</span><i>→</i><span><b>03</b> 执行与核验</span><i>→</i><span><b>04</b> 生成报告</span></div>
</div>
""", unsafe_allow_html=True)

# 标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 流水线状态", "📄 复现报告", "📜 审计日志", "🔍 状态机", "📂 历史记录",
])

# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("""<div class="section-heading"><div><span class="section-kicker">LIVE PIPELINE</span>
    <h2>复现流水线</h2><p>每个 Agent 的执行状态都会在这里实时更新。</p></div>
    <span class="section-aside">论文解析 · 验证 · 优化 · 报告</span></div>""",
                unsafe_allow_html=True)

    AGENT_DESC = {
        "📖 PaperReader": ("论文解析", "从PDF/标题中提取结构化信息"),
        "🔍 ResourceFinder": ("资源查找", "定位代码仓库和数据集"),
        "🔧 EnvBuilder": ("环境构建", "自动搭建环境 + 依赖诊断"),
        "⚡ CodeExecutor": ("代码执行", "smoke + full 双阶段执行"),
        "✅ ResultValidator": ("结果验证", "比对论文声明值与运行结果"),
        "🛡️ Verifier": ("质量验证", "Prompt-Free 检查质量 + 修正闭环"),
        "🧪 Optimizer": ("智能优化（预留）", "当前未开放，不执行优化"),
        "📝 ReportGenerator": ("报告生成", "生成复现+优化 Markdown 报告"),
    }
    names = [a[1] for a in AGENTS] + ["🛡️ Verifier", "🧪 Optimizer",
                                      "📝 ReportGenerator"]

    # ---------- 后台复现实时进度（轮询进度文件） ----------
    if snap and snap.get("error"):
        st.error(f"❌ 后台流水线异常: {snap['error']}")

    if st.session_state.running and pf:
        if st_autorefresh is not None:
            st_autorefresh(interval=2000, key=f"ar_{pf}")
            st.info("🔄 复现流水线正在后台运行，页面每 2 秒自动刷新，"
                    "实时展示各 Agent 进度。")
        else:
            st.warning("未安装 streamlit-autorefresh，页面不会自动刷新；"
                       "可刷新浏览器页面查看最新进度。")

    agent_cards = []
    for i, name in enumerate(names):
        status = st.session_state.agent_status.get(name, "waiting")
        status_text = {"success": "已完成", "error": "出现错误",
                       "running": "进行中", "waiting": "等待中"}.get(status, "等待中")
        status_class = status if status in {"success", "error", "running"} else "waiting"
        title, desc = AGENT_DESC.get(name, ("", ""))
        agent_cards.append(f"""
            <div class="agent-card agent-{status_class}">
                <div class="agent-top"><span class="agent-index">{i + 1:02d} / {len(names):02d}</span>
                <span class="agent-status">{status_text}</span></div>
                <div class="agent-name">{escape(name)}</div>
                <div class="agent-title">{escape(title)}</div>
                <p>{escape(desc)}</p>
            </div>
            """)
    st.markdown('<div class="agent-grid">' + ''.join(card.strip() for card in agent_cards) + '</div>',
                unsafe_allow_html=True)

    completed = sum(1 for a in names
                    if st.session_state.agent_status.get(a) == "success")
    progress = completed / len(names) if names else 0
    st.progress(progress, text=f"整体进度: {completed}/{len(names)}")

    if snap and snap.get("execution_output"):
        with st.expander("官方仓库实时输出（最近16000字符，完整日志保存在运行目录）", expanded=True):
            st.code(snap["execution_output"], language="text")

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
        if result.get("state") == "COMPLETED":
            validation = data.get("validation") or {}
            if validation.get("result_level") == "failed":
                st.error("实验未通过：" + validation.get("reason", "查看报告和执行日志"))
            elif validation.get("status") == "not_reproduced":
                st.warning("完整实验已运行，论文数值验收未通过：" + validation.get("reason", "查看指标差异"))
            elif validation.get("status") in {"prepared", "inconclusive", "smoke_passed"}:
                st.info(validation.get("reason", "流程已结束，请查看实验结论"))
            elif validation.get("is_reproduced") is True:
                st.success("🎉 完整实验已完成，论文数值验收通过！")
            else:
                st.info("流水线已结束，请查看报告中的复现结论。")
        elif result.get("state") == "ERROR":
            st.error(f"❌ 流程出错: {result.get('error', '未知错误')}")

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">RESEARCH OUTPUT</span><h2>复现与优化报告</h2><p>查看实验结论、验证结果与优化建议。</p></div></div>', unsafe_allow_html=True)
    if st.session_state.result and st.session_state.result.get("data", {}).get("report"):
        report = st.session_state.result["data"]["report"]
        # 深色 IDE 面板渲染已回滚（见 CHANGELOG [2026.09.20-12]）：
        # 面板在真实浏览器里代码不可见，改回原生 Markdown 渲染。
        result = st.session_state.result
        report_path = result.get("report_path") or result["data"].get("report_path")
        render_report(report, report_path, st_module=st)
    else:
        st.markdown('<div class="empty-state"><span>▤</span><div class="empty-title">报告将在这里生成</div><p>在左侧提交论文并启动复现，完成后即可查看实验报告。</p></div>', unsafe_allow_html=True)

# ===== Tab 3: 审计日志 =====
with tab3:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">TRACE & EVIDENCE</span><h2>审计日志</h2><p>按 Agent 和状态筛选，查看每一步的运行记录。</p></div></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="empty-state"><span>≡</span><div class="empty-title">暂无运行记录</div><p>启动复现后，这里会记录各 Agent 的执行过程。</p></div>', unsafe_allow_html=True)

# ===== Tab 4: 状态机 =====
with tab4:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">WORKFLOW MAP</span><h2>状态机定义</h2><p>系统通过有限状态机管理 Agent 的流转与异常恢复。</p></div></div>', unsafe_allow_html=True)
    state_info = """
```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> READ_PAPER
    READ_PAPER --> FIND_RESOURCES
    FIND_RESOURCES --> BUILD_ENV
    BUILD_ENV --> EXECUTE_CODE
    EXECUTE_CODE --> VALIDATE
    VALIDATE --> GENERATE_REPORT
    GENERATE_REPORT --> COMPLETED
    READ_PAPER --> ERROR
    FIND_RESOURCES --> ERROR
    BUILD_ENV --> ERROR
    EXECUTE_CODE --> ERROR
    VALIDATE --> ERROR
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
        {"状态": "GENERATE_REPORT", "说明": "生成Markdown复现报告", "Agent": "ReportGenerator"},
        {"状态": "COMPLETED", "说明": "流水线完成", "Agent": "—"},
        {"状态": "ERROR", "说明": "出错状态，可重试", "Agent": "—"},
    ]
    st.table(state_data)
    st.caption("智能优化为预留可选接口，当前未启用；上图展示当前主流程。详细阶段以流水线状态为准。")


# ===== Tab 5: 历史记录 =====
with tab5:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">ARCHIVE & STORAGE</span><h2>复现历史与存储</h2><p>回顾已运行的会话，并管理实验产物与缓存。</p></div></div>', unsafe_allow_html=True)

    # 删除类操作的反馈：必须跨 rerun 传递。删除后紧跟 st.rerun()，
    # 当次运行的 st.success 会被新一次运行整棵树丢弃，用户看不到任何提示。
    _hist_flash = st.session_state.pop("_hist_flash", "")
    if _hist_flash:
        st.success(_hist_flash)
    # 上一轮刚做过删除 -> 在此复位确认勾选，避免下次一键误删。
    # 必须在这里做：widget 一旦实例化，就不能再改它的 session_state 了。
    for _ck in st.session_state.pop("_reset_confirm_keys", []):
        st.session_state[_ck] = False

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

            try:
                bundle = build_report_bundle(report_content, rp)
                st.download_button("⬇️ 下载报告和图片（ZIP）", data=bundle,
                                   file_name=Path(rp).stem + ".zip", mime="application/zip",
                                   use_container_width=True)
            except ValueError as exc:
                st.warning(str(exc))

# -- 存储仪表板 --
    # -- 资源下载/安装监控 --
    with st.expander("📡 资源下载与安装监控", expanded=True):
        if st.button("🔄 刷新资源状态", key="resource_refresh",
                     use_container_width=True):
            st.rerun()
        events = list_resource_events(limit=100)
        inventory = list_resource_inventory()
        running = [e for e in events if e.get("state") == "running"]
        if running:
            st.warning(f"当前有 {len(running)} 个资源操作进行中")
        else:
            st.caption("当前没有正在记录的下载或依赖安装操作")
        if inventory:
            st.dataframe(
                [{"类型": row["type"], "资源": row["id"],
                  "论文": row["paper_id"] or "共享缓存",
                  "状态": row["state"],
                  "占用": format_size(row["bytes"]),
                  "最后使用": (row["last_used"] or "")[:19].replace("T", " "),
                  "来源/内容": row["detail"] or "—"}
                 for row in inventory],
                use_container_width=True, hide_index=True)
        else:
            st.info("暂无资源清单。运行一次复现后，依赖和数据集会显示在这里。")
        if events:
            st.markdown("**最近资源事件**")
            st.dataframe(
                [{"时间": (e.get("timestamp") or "")[:19].replace("T", " "),
                  "类型": e.get("resource_type", ""),
                  "资源": e.get("resource_id", ""),
                  "操作": e.get("operation", ""),
                  "状态": e.get("state", ""),
                  "详情": e.get("detail", "") or e.get("error", "")}
                 for e in events[:30]],
                use_container_width=True, hide_index=True)

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
            st.session_state["_hist_flash"] = (
                f"已删除 {removed} 个文件，释放 {format_size(freed)}")
            st.rerun()

        # -- 依赖缓存：不随历史记录一起删（跨会话共享，删了要重新下载） --
        st.markdown("---")
        st.markdown("**📦 依赖缓存**（跨会话共享，不随历史记录删除）")
        st.caption("隔离安装按**依赖清单内容**哈希寻址，同一份依赖跨论文复用。"
                   "删掉后下次执行同一依赖要重新下载安装，因此不并入"
                   "「清空历史」；需要腾空间时才在这里单独清。")
        try:
            deps_items = list_deps_cache()
        except Exception as e:
            deps_items = []
            st.error(f"读取依赖缓存失败: {e}")
        if deps_items:
            total_deps = sum(d["bytes"] for d in deps_items)
            st.dataframe(
                [{"目录": d["name"],
                  "类型": {"reqs": "依赖清单", "heal": "自愈补装",
                           "legacy": "旧目录（无元数据）"}.get(d["kind"], d["kind"]),
                  "包含包": ", ".join(d["packages"]) or "—",
                  "占用": format_size(d["bytes"]),
                  "最后使用": (d["last_used"] or "")[:16].replace("T", " ")}
                 for d in deps_items],
                use_container_width=True, hide_index=True)
            st.caption(f"{len(deps_items)} 个目录 · 合计 {format_size(total_deps)}")

            deps_keep = st.slider("清理多少天未使用的依赖缓存", 1, 180, 30,
                                  key="deps_keep_days")
            if st.button("清理冷缓存（按最后使用时间）",
                         use_container_width=True,
                         help="依赖缓存每次命中都会刷新「最后使用」时间；"
                              "无元数据的旧目录按目录修改时间判断"):
                n, freed = cleanup_deps_cache(keep_days=deps_keep)
                st.session_state["_hist_flash"] = (
                    f"已清理 {n} 个冷依赖目录，释放 {format_size(freed)}")
                st.rerun()

            picked_deps = st.multiselect(
                "选择要删除的依赖目录（下次执行会重新下载安装）",
                [d["name"] for d in deps_items], key="deps_del_pick")
            confirm_deps = st.checkbox(
                "我确认删除所选依赖缓存（下次执行同一依赖需重新下载安装）",
                key="confirm_deps_del")
            if st.button(f"🗑️ 删除所选 {len(picked_deps)} 个依赖目录",
                         key="deps_del_btn", use_container_width=True,
                         type="secondary",
                         disabled=not picked_deps or not confirm_deps):
                with st.spinner("正在删除..."):
                    n, freed = delete_deps_cache(picked_deps)
                st.session_state["_reset_confirm_keys"] = ["confirm_deps_del"]
                st.session_state["_hist_flash"] = (
                    f"已删除 {n} 个依赖目录，释放 {format_size(freed)}")
                st.rerun()
        else:
            st.caption("暂无依赖缓存。")

        st.markdown("---")
        st.markdown("**🗑️ 历史会话清理**（危险操作，请谨慎）")
        confirm_clear = st.checkbox(
            "我确认清空全部历史复现会话（实验账本/审计日志/复现报告/runtime 进度）",
            key="confirm_clear_all")
        if st.button("🧹 清空全部历史", use_container_width=True,
                     type="secondary",
                     disabled=not confirm_clear):
            removed, freed = clear_sessions()
            st.session_state["_reset_confirm_keys"] = ["confirm_clear_all"]
            st.session_state["_hist_flash"] = (
                f"已清空全部历史：删除 {removed} 个文件，"
                f"释放 {format_size(freed)}")
            st.rerun()

    # -- 历史会话列表 --
    st.markdown("#### 📋 历史复现会话")
    try:
        sessions = list_sessions()
        if not sessions:
            st.info("暂无历史复现记录。完成一次复现后将在此展示。")
        else:
            # --- 筛选 ---
            # 「其他」兜住状态机中间态（INIT / GENERATE_CODE 等），
            # 「未知」是 list_sessions 在 ledger 为空时的字面量兜底值。
            known_states = ("COMPLETED", "ERROR", "RUNNING", "未知")
            f1, f2 = st.columns([1, 3])
            state_q = f1.selectbox("状态筛选", ["全部"] + list(known_states)
                                   + ["其他"], key="hist_state_filter")
            title_q = f2.text_input(
                "按标题筛选（不区分大小写，子串匹配）",
                key="hist_title_filter").strip().lower()

            def _match(sess: dict) -> bool:
                st_ = sess.get("state", "未知")
                if state_q == "其他":
                    if st_ in known_states:
                        return False
                elif state_q != "全部" and st_ != state_q:
                    return False
                return title_q in (sess.get("paper_title") or "").lower()

            visible = [s for s in sessions if _match(s)]

            # --- 选择操作 ---
            # 必须在下方 checkbox 实例化「之前」写 session_state：Streamlit
            # 不允许在 widget 创建后修改它的状态（会抛 StreamlitAPIException）。
            b1, b2, _pad = st.columns([1, 1, 3])
            if b1.button("☑️ 全选可见", key="hist_select_all",
                         use_container_width=True, disabled=not visible):
                for s in visible:
                    st.session_state[f"hist_chk_{s['session_id']}"] = True
                st.rerun()
            if b2.button("▫️ 清除选择", key="hist_clear_sel",
                         use_container_width=True):
                for s in sessions:   # 含被筛选隐藏的，不留幽灵勾选
                    st.session_state[f"hist_chk_{s['session_id']}"] = False
                st.rerun()

            st.caption(f"共 {len(sessions)} 条 · 可见 {len(visible)} 条"
                       f"（勾选左侧方框即可，无需展开）")
            if any(s.get("state") == "RUNNING" for s in visible):
                st.caption("⚠️ 可见列表含 RUNNING 会话：多为异常中断的残留；"
                           "若该复现仍在运行，删除会丢失它的记录"
                           "（被占用的文件自动跳过）。")

            if not visible:
                st.info("没有符合筛选条件的会话。")
            else:
                for sess in visible:
                    sid = sess["session_id"]
                    title = sess.get("paper_title", "未知论文")
                    state = sess.get("state", "未知")
                    state_icon = {"COMPLETED": "✅", "ERROR": "❌"}.get(state, "⏳")
                    duration = sess.get("duration_sec", 0)
                    llm_calls = sess.get("llm_calls", 0)
                    log_entries = sess.get("log_entries", 0)
                    report_path = sess.get("report_path", "")

                    c_chk, c_body = st.columns([0.04, 0.96])
                    # 勾选框放在 expander 外：批量删除时不必逐条展开
                    c_chk.checkbox(f"选择 {sid}", key=f"hist_chk_{sid}",
                                   label_visibility="collapsed")
                    with c_body.expander(
                            f"{state_icon} [{sid}] {title[:40] or '无标题'}..."):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("状态", state)
                        m2.metric("耗时", f"{duration:.1f}s")
                        m3.metric("LLM调用", llm_calls)
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
                            st.session_state["_hist_flash"] = (
                                f"已删除该会话 {removed} 个文件，"
                                f"释放 {format_size(freed)}")
                            st.rerun()

                # --- 批量删除 ---
                # 已选必须在循环「之后」统计：checkbox 的勾选值在 widget
                # 实例化时才写入 session_state，循环前算会滞后一次交互。
                st.markdown("---")
                st.markdown("**🗑️ 批量删除所选会话**（危险操作，请谨慎）")
                selected = [
                    s for s in visible
                    if st.session_state.get(f"hist_chk_{s['session_id']}", False)
                ]
                st.caption(f"已选 {len(selected)} 条（仅统计上方可见列表；"
                           f"被筛选隐藏的已勾选会话不会被删除）")

                confirm_batch = st.checkbox(
                    "我确认批量删除以上所选会话（账本/日志/报告/progress 一并删除）",
                    key="confirm_batch_del")
                if st.button(f"🗑️ 删除所选 {len(selected)} 条",
                             key="batch_del_btn", type="secondary",
                             disabled=not selected or not confirm_batch):
                    with st.spinner("正在删除..."):
                        removed, freed = delete_sessions(
                            [s["session_id"] for s in selected])
                    st.session_state["_reset_confirm_keys"] = ["confirm_batch_del"]
                    st.session_state["_hist_flash"] = (
                        f"已批量删除 {len(selected)} 个会话，共 {removed} 个文件，"
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
    pt = st.session_state.paper_title.strip() if input_mode == "论文标题" else ""
    if experiment_profile and st.session_state.mock_mode:
        st.sidebar.error("官方仓库预设需要关闭 Mock 模式，才能执行真实论文代码。")
    elif experiment_profile and st.session_state.use_docker:
        st.sidebar.error("本轮官方仓库预设仅支持本地 CPU；请关闭 Docker 开关后运行。")
    elif not experiment_profile and not pt and not uploaded_file:
        st.sidebar.error("请先上传PDF文件" if input_mode == "上传PDF"
                         else "请先输入论文标题")
    elif requires_api and not st.session_state.mock_mode and config_missing(base_url, model_name):
        st.error("真实模式缺少 LLM 配置（"
                 + "、".join(config_missing(base_url, model_name))
                 + "）。请在侧边栏填写，或设置环境变量 "
                   "LLM_BASE_URL / LLM_MODEL 后重试。")
    else:
        # 真实模式且未填 API Key：多数云端端点（DeepSeek/OpenAI 等）会返回
        # 401，且流水线会把错误文本当 LLM 输出继续跑，表象类似「没反应」。
        # 此处不阻断（部分自建端点无需鉴权），但给出明确预警。
        if (requires_api and not st.session_state.mock_mode
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
            experiment_profile=experiment_profile,
            prepare_only=(bool(st.session_state.get("repository_prepare_only"))
                          if experiment_profile else False),
            use_llm_review=(bool(st.session_state.get("repository_llm_review"))
                            if experiment_profile else False),
            allow_result_summary_review=(bool(st.session_state.get("repository_result_review"))
                                         if experiment_profile else False),
            model_name=model_name, base_url=base_url,
            api_key=api_key,
            mock_mode=st.session_state.mock_mode,
            enable_optimization=False,
            use_docker=(not st.session_state.mock_mode
                        and docker_available
                        and st.session_state.use_docker),
            cleanup_pdf=True)   # 临时 PDF 由后台线程负责删除
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
st.markdown("""
<div class="app-footer">
    <span>✳ AutoReproducer</span><span>v0.2.0 · 让复现过程清晰可见</span>
</div>
""", unsafe_allow_html=True)
