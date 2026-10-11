"""AutoReproducer - Streamlit 前端界面

修复与增强：
- 论文标题 / 上传 PDF 正确传递到 PaperReader；
- 侧边栏 LLM API 配置（OpenAI 兼容端点 / Key / 模型）真实生效；
- 按实际执行阶段展示进度与 LLM 调用统计；
- 复现流水线后台线程执行，前端轮询进度文件实时展示当前阶段
  （OpenAI 兼容端点 / Key / 模型真实生效）。
"""
import inspect
import importlib
import os
import sys
import tempfile
import time
from html import escape
from pathlib import Path

import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.local_llm_settings import load_local_llm_settings
load_local_llm_settings()

from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger
from src.base_agent import BaseAgent
from src.corpus import list_papers
from src.repository_profiles import PROFILE_LABELS, PAPER_TITLE, get_profile
from src.title_routing import resolve_title_request
from src.method_budget import CONFIRMATION_RESERVE_SECONDS, optimization_budget_error
from frontend.report_renderer import render_report, build_report_bundle
from frontend.llm_config import (
    CONNECT_TEST_TIMEOUT,
    resolve_llm_config,
    config_missing,
    test_llm_connection,
)
from frontend import pipeline_entrypoint

# Only refresh the small loader. Older active pipeline modules keep their globals.
if pipeline_entrypoint.BACKEND_API_VERSION != 4:
    pipeline_entrypoint = importlib.reload(pipeline_entrypoint)

_pipeline_backend = pipeline_entrypoint.load_backend_pipeline()
ProgressStore = _pipeline_backend.ProgressStore
run_pipeline_background = _pipeline_backend.run_pipeline_background
from frontend.history_manager import (
    list_sessions,
    collect_storage_snapshot,
    cleanup_runtime,
    delete_deps_cache,
    cleanup_deps_cache,
    list_resource_events,
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
if "pipeline_stages" not in st.session_state:
    st.session_state.pipeline_stages = []
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
    # 每次使用当前任务的完整快照，避免切换进度文件时残留上次的完成状态。
    st.session_state.result = snap["result"]
    st.session_state.running = snap["running"]
    st.session_state.agent_status = snap.get("agent_status", {})
    st.session_state.pipeline_stages = snap.get("pipeline_stages", [])
    st.session_state.current_state = snap["state"]
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


@st.cache_data(show_spinner=False, max_entries=8)
def _preview_pdf_input(payload: bytes, mock_mode: bool):
    """Inspect uploaded bytes once; the backend independently verifies its file."""
    from frontend.pdf_entrypoint import load_pdf_input
    pdf_module = load_pdf_input()
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(payload)
            temporary = handle.name
        document = pdf_module.extract_pdf_input(temporary)
        request = pdf_module.resolve_pdf_request({"pdf_path": temporary, "mock_mode": mock_mode})
        return {"resolution": request.get("pdf_resolution", {}),
                "repository_links": list(document.repository_links),
                "pdf_input": {"sha256": document.sha256, "pages": document.page_count}}
    except pdf_module.PDFParserUnavailable:
        return {"parser_pending": True}
    except pdf_module.PDFInputError as exc:
        return {"error": str(exc)}
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


@st.fragment
def render_paper_input():
    """只刷新论文输入；点击启动时主页面重新读取最新选项。"""
    st.markdown('<div class="sidebar-section-label"><span>01</span> 论文输入</div>',
                unsafe_allow_html=True)
    input_mode = st.radio("输入方式", ["论文标题", "上传PDF", "官方仓库预设"], key="input_mode")
    if input_mode == "论文标题":
        st.session_state.paper_title = st.text_input(
            "论文标题", value=st.session_state.paper_title,
            placeholder="输入论文标题...", key="paper_title_input")
        match = resolve_title_request({"paper_title": st.session_state.paper_title,
                    "mock_mode": st.session_state.mock_mode,
                    "corpus_paper": None if st.session_state.get("corpus_paper_select", "无") == "无"
                                    else st.session_state.get("corpus_paper_select")})
        if match.get("title_resolution"):
            selected = get_profile(match["experiment_profile"])
            st.success("已匹配官方作者仓库：" + selected["repository"]["url"])
            st.markdown(f"[论文]({selected['paper']['url']}) · [作者代码]({selected['repository']['url']})")
            st.caption("将核验固定版本并执行：" + selected["label"])
            if selected.get("adapter_id"):
                st.caption("结论限于选定官方方法实验；论文完整基准需要另行适配。")
            st.caption("该在线审核使用作者源码与项目适配证据。")
            st.checkbox("在线核验匹配的作者代码与实验依据", value=True, key="title_llm_review")
    elif input_mode == "上传PDF":
        uploaded_file = st.file_uploader(
            "上传PDF文件", type=["pdf"], key="pdf_uploader",
            help="点击 Upload 选择本地 PDF，或将 PDF 拖入上传区域")
        if uploaded_file is None:
            st.caption("请点击 Upload 选择 PDF 文件，上传后再开始复现。")
        else:
            st.success(f"已上传：{uploaded_file.name}"
                       f"（{format_size(uploaded_file.size)}）")
            preview = _preview_pdf_input(uploaded_file.getvalue(), st.session_state.mock_mode)
            if preview.get("error"):
                st.error(preview["error"])
            elif preview.get("resolution"):
                selected = get_profile(preview["resolution"]["profile"])
                st.success("PDF 首页已匹配作者仓库：" + selected["repository"]["url"])
                st.markdown(f"[论文]({selected['paper']['url']}) · [作者代码]({selected['repository']['url']})")
                st.caption("将执行：" + selected["label"] + "；结论按该实验的验收范围给出。")
                st.caption("该在线审核使用作者源码与项目适配证据。")
            else:
                st.caption("将解析论文正文并核验资源；信息不足会明确报告，不能据占位代码认定复现。")
            links = preview.get("repository_links", [])
            if links:
                primary = next((link for link in links if link.get("is_author_code")), None)
                if primary:
                    st.success("PDF 原文代码声明链接：" + primary["url"])
                    st.caption(f"来源：第 {primary['page']} 页；将优先于模型猜测和关键词搜索。")
                with st.expander("PDF 仓库链接提取依据"):
                    for link in links[:12]:
                        kind = {"author_code_statement": "代码公开声明", "reference": "参考文献",
                                "repository_link": "仓库链接"}.get(link["evidence_type"], "仓库链接")
                        st.markdown(f"[第 {link['page']} 页 · {kind}]({link['url']})")
                        if link.get("context"):
                            st.text(link["context"])
            if not preview.get("error"):
                st.checkbox("在线核验匹配的作者代码与实验依据", value=True, key="pdf_llm_review")
    else:
        st.selectbox("真实论文实验", list(PROFILE_LABELS),
                     format_func=lambda key: PROFILE_LABELS[key], key="experiment_profile")
        selected = get_profile(st.session_state.experiment_profile)
        st.caption(selected["paper"]["title"])
        st.markdown(f"[论文]({selected['paper']['url']}) · [作者代码]({selected['repository']['url']})")
        if selected.get("adapter_id") in {"siren", "neural_ode"}:
            st.selectbox("实验操作", ["运行实验", "准备实验环境", "仅准备源码和命令"], key="method_action")
            st.selectbox("智能优化", ["off", "suggest", "validate"], index=1,
                         format_func=lambda value: {"off": "关闭", "suggest": "真实基线 + 智能建议", "validate": "真实训练验证建议（长任务）"}[value],
                         key="method_optimization")
            st.checkbox("运行前进行在线论文与代码分析（完整模式）", value=False, key="method_llm_review")
            if st.session_state.method_optimization == "validate":
                st.number_input("最多候选数", min_value=1, max_value=3, value=3, key="method_max_candidates")
                minutes = st.number_input("本篇总预算（分钟）", min_value=1, max_value=120, value=120,
                                          key="method_budget_minutes",
                                          help="包含在线分析、基线、候选训练和最终确认；必须大于40分钟，建议120分钟。")
                st.caption(f"总预算包含分析、基线与参数验证，其中固定预留"
                           f"{CONFIRMATION_RESERVE_SECONDS // 60}分钟用于两种子确认。"
                           "建议120分钟；这是耗时上限，提前完成不会等满。")
                if st.session_state.method_action == "运行实验":
                    budget_error = optimization_budget_error(minutes * 60)
                    if budget_error:
                        st.warning(budget_error)
            st.caption("运行时会自动准备缺失环境、检查依赖并修复缓存；也可单独准备环境。"
                       "快速档目标为五分钟，建议尚未实测有效；验证建议使用独立长任务预算。"
                       "智能建议会把指标与训练摘要交给已配置的 API。")
            return
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
    st.caption("通用流程暂不执行优化；SIREN 与 Neural ODE 预设提供真实建议和参数验证选项。")

    # 论文输入
    render_paper_input()
    input_mode = st.session_state.input_mode
    uploaded_file = (st.session_state.get("pdf_uploader")
                     if input_mode == "上传PDF" else None)
    experiment_profile = (st.session_state.get("experiment_profile")
                          if input_mode == "官方仓库预设" else None)
    method_selected = bool(experiment_profile and get_profile(experiment_profile).get("adapter_id") in {"siren", "neural_ode"})
    requires_api = not experiment_profile or (
        bool(st.session_state.get("repository_llm_review"))
        and not bool(st.session_state.get("repository_prepare_only")))
    if method_selected:
        requires_api = st.session_state.get("method_action") == "运行实验" and (
            st.session_state.get("method_optimization", "off") != "off" or st.session_state.get("method_llm_review", False))

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

    title_resolution = None
    if input_mode == "论文标题":
        match = resolve_title_request({"paper_title": st.session_state.paper_title,
                                      "corpus_paper": corpus_paper,
                                      "mock_mode": st.session_state.mock_mode})
        if match.get("title_resolution"):
            experiment_profile = match["experiment_profile"]
            title_resolution = match["title_resolution"]
            # Title input runs the matched baseline. Preset controls belong to
            # that input mode and must not leak a previous long optimization.
            method_selected = False
            requires_api = bool(st.session_state.get("title_llm_review", True))
    elif input_mode == "上传PDF" and uploaded_file and not corpus_paper:
        preview = _preview_pdf_input(uploaded_file.getvalue(), st.session_state.mock_mode)
        if preview.get("resolution"):
            requires_api = bool(st.session_state.get("pdf_llm_review", True))

    # 启动 / 重置按钮
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始复现", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.running)
    with col2:
        reset_btn = st.button("🔄 重置", use_container_width=True,
                              disabled=st.session_state.running)
        if st.session_state.running:
            st.caption("任务结束后可重置")

    # 系统状态
    st.markdown("---")
    st.markdown('<div class="sidebar-section-label"><span>03</span> 系统状态</div>',
                unsafe_allow_html=True)
    state_colors = {
        "INIT": "⚪", "READ_PAPER": "📖", "FIND_RESOURCES": "🔍",
        "BUILD_ENV": "🔧", "EXECUTE_CODE": "⚡", "VALIDATE": "✅",
        "OPTIMIZING": "🧪", "OPTIMIZED": "🏆",
        "OPTIMIZE": "🧪",
        "GENERATE_REPORT": "📝", "COMPLETED": "🎉", "ERROR": "❌",
    }
    st.markdown(
        f'<div class="sidebar-status"><span class="status-pulse"></span>'
        f'<span>当前状态</span><strong>'
        f'{state_colors.get(st.session_state.current_state, "⚪")} '
        f'{escape(st.session_state.current_state)}</strong></div>',
        unsafe_allow_html=True)


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
if start_btn and not st.session_state.running:
    pt = st.session_state.paper_title.strip() if input_mode == "论文标题" else ""
    budget_error = (optimization_budget_error(st.session_state.get("method_budget_minutes", 120) * 60)
                    if method_selected and st.session_state.get("method_action") == "运行实验"
                    and st.session_state.get("method_optimization") == "validate" else "")
    if experiment_profile and st.session_state.mock_mode:
        st.sidebar.error("官方仓库预设需要关闭 Mock 模式，才能执行真实论文代码。")
    elif experiment_profile and st.session_state.use_docker:
        st.sidebar.error("本轮官方仓库预设使用本地执行；请关闭 Docker 开关后运行。")
    elif not experiment_profile and not pt and not uploaded_file:
        st.sidebar.error("请先上传PDF文件" if input_mode == "上传PDF"
                         else "请先输入论文标题")
    elif budget_error:
        st.sidebar.error(budget_error)
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
        st.session_state.pipeline_stages = []
        st.session_state.current_state = "INIT"
        st.session_state.pipeline_note = (
            "复现流水线已在后台启动，进度实时刷新中…")
        run_pipeline_background(
            progress_file,
            paper_title=pt, pdf_path=tmp_pdf,
            corpus_paper=corpus_paper,
            experiment_profile=experiment_profile,
            title_resolution=title_resolution,
            prepare_only=False if title_resolution else (st.session_state.get("method_action") == "仅准备源码和命令" if method_selected else
                          bool(st.session_state.get("repository_prepare_only")) if experiment_profile else False),
            prepare_environment=method_selected and st.session_state.get("method_action") == "准备实验环境",
            optimization_mode=st.session_state.get("method_optimization", "off") if method_selected else "off",
            max_candidates=int(st.session_state.get("method_max_candidates", 3)),
            budget_seconds=int(st.session_state.get("method_budget_minutes", 120)) * 60,
            use_llm_review=(bool(st.session_state.get("title_llm_review", True)) if title_resolution else
                             bool(st.session_state.get("pdf_llm_review", True)) if tmp_pdf else
                            bool(st.session_state.get("method_llm_review")) if method_selected else
                            bool(st.session_state.get("repository_llm_review")) if experiment_profile else False),
            allow_result_summary_review=(bool(st.session_state.get("repository_result_review"))
                                         if experiment_profile and not title_resolution else False),
            model_name=model_name, base_url=base_url,
            api_key=api_key,
            mock_mode=st.session_state.mock_mode,
            enable_optimization=False,
            use_docker=(not st.session_state.mock_mode
                        and docker_available
                        and st.session_state.use_docker),
            cleanup_pdf=True)   # 临时 PDF 由后台线程负责删除
        st.session_state.workspace_tabs = "📋 流水线状态"
        st.rerun()

# 重置按钮处理
if reset_btn and not st.session_state.running:
    st.session_state.orchestrator = None
    st.session_state.result = None
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.current_state = "INIT"
    st.session_state.agent_status = {}
    st.session_state.pipeline_stages = []
    st.session_state.connection_result = None
    st.session_state.progress_file = None
    st.session_state.pipeline_note = None
    st.rerun()

# ========== 主界面 ==========
_state_label = {
    "INIT": "等待开始", "READ_PAPER": "解析论文", "FIND_RESOURCES": "查找资源",
    "BUILD_ENV": "构建环境", "EXECUTE_CODE": "执行代码", "VALIDATE": "验证结果",
    "OPTIMIZING": "智能优化", "OPTIMIZED": "优化完成",
    "OPTIMIZE": "参数优化中",
    "GENERATE_REPORT": "生成报告", "COMPLETED": "任务完成", "ERROR": "运行异常",
}.get(st.session_state.current_state, "运行中")
if st.session_state.running:
    _state_label = "整体任务运行中"
_hero_status = ("running" if st.session_state.running else
                "error" if st.session_state.current_state == "ERROR" else
                "success" if st.session_state.current_state == "COMPLETED" else
                "idle")
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

# 新版 Streamlit 支持按所选标签延迟执行；旧版保留显式加载入口。
_lazy_tabs = "on_change" in inspect.signature(st.tabs).parameters
_tab_options = {"key": "workspace_tabs", "on_change": "rerun"} if _lazy_tabs else {}
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 流水线状态", "📄 复现报告", "📜 审计日志", "🔍 状态机", "📂 历史记录",
], **_tab_options)

# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("""<div class="section-heading"><div><span class="section-kicker">LIVE PIPELINE</span>
    <h2>复现流水线</h2><p>按实际执行顺序追踪各阶段；同一 Agent 的不同任务分别显示。</p></div>
    <span class="section-aside">准备 · 执行 · 核验 · 报告</span></div>""",
                unsafe_allow_html=True)

    # ---------- 后台复现实时进度（轮询进度文件） ----------
    if snap and snap.get("error"):
        st.error(f"❌ 后台流水线异常: {snap['error']}")

    if st.session_state.running and pf:
        if st_autorefresh is not None:
            st_autorefresh(interval=2000, key=f"ar_{pf}")
            st.info("🔄 复现流水线正在后台运行，页面每 2 秒自动刷新，"
                    "实时展示各阶段进度。")
        else:
            st.warning("未安装 streamlit-autorefresh，页面不会自动刷新；"
                       "可刷新浏览器页面查看最新进度。")

    stages = st.session_state.pipeline_stages
    execution_context = (snap or {}).get("execution_context") or {}
    active_executions = (snap or {}).get("active_executions") or []
    active_runs = (snap or {}).get("active_runs") or []
    active_workers = (snap or {}).get("active_workers") or []
    pdf_source = (snap or {}).get("pdf_resolution") or {}
    if pdf_source:
        selected = get_profile(pdf_source["profile"])
        st.caption("已根据 PDF 首页核验论文：" + pdf_source["title"])
        st.markdown(f"作者仓库：[源码]({selected['repository']['url']}) · PDF SHA-256：`{pdf_source['sha256']}`")
    has_method_study = any(stage["id"] == "method_advice" and stage.get("status") != "skipped"
                           for stage in stages)
    if st.session_state.running and stages:
        st.info("整体任务仍在运行：代码仍在执行。" if active_executions else
                "整体任务仍在运行：执行轮次尚未结束，正在等待后续步骤或清理。" if active_runs else
                "整体任务仍在运行：等待后台进程返回并完成清理。" if active_workers and (snap or {}).get("finalization_pending") else
                "整体任务仍在运行。单个阶段或步骤完成，不代表整个复现任务已结束。")
        for context in active_executions:
            st.caption("当前执行：" + context["label"])
    if (snap or {}).get("finalization_pending"):
        st.warning("已收到流水线结束请求，正在等待所属执行退出并清理；尚未确认任务结束。")
    agent_cards = []
    for i, stage in enumerate(stages):
        status = stage.get("status", "waiting")
        status_text = {"success": "本阶段完成", "error": "出现错误",
                       "running": "进行中", "waiting": "等待中",
                       "skipped": "已跳过", "blocked": "未执行"}.get(status, "等待中")
        if status == "error" and stage.get("outcome") in {"rejected", "fail"}:
            status_text = "核验未通过"
        elif status == "error" and stage.get("outcome") == "execution_failed":
            status_text = "执行失败"
        status_class = status if status in {"success", "error", "running"} else "waiting"
        title = stage.get("title", stage.get("agent", "执行阶段"))
        desc = stage.get("description", "")
        agent = stage.get("agent", "")
        phase_executions = [context for context in active_executions
                            if context.get("phase_id") == stage["id"]]
        phase_runs = [run for run in active_runs if run.get("phase_id") == stage["id"]]
        # Historical plans may call this role "complete training". Its actual
        # responsibility is code execution, regardless of the paper or commands.
        if stage.get("agent") in {"CodeExecutor", "⚡ CodeExecutor"}:
            title = "代码执行"
            if has_method_study and stage["id"] == "execute_repository":
                title = "基线代码执行"
                desc = "基线训练与测试；后续优化试验在下方单独显示。"
                if status == "success":
                    status_text = "基线执行完成"
            elif status == "success" and st.session_state.running and active_executions:
                title = "代码执行（此前轮次）"
                status_text = "此前轮次完成"
        if has_method_study and stage["id"] == "verify_protocol":
            title = "基线协议与产物核验"
            desc = "核验基线轮次的协议与产物。"
            if status == "success":
                status_text = "基线核验完成"
        if stage["id"] == "method_advice" and status == "running" and phase_executions:
            title = "优化试验：代码执行"
            status_text = "代码执行中"
            agent += " · ⚡ CodeExecutor"
            desc = "正在执行优化试验代码，当前轮次完成后继续后续处理。"
        elif stage["id"] == "method_advice" and status == "running" and phase_runs:
            title = "优化试验：执行轮次未结束"
            status_text = "执行轮次未结束"
            agent += " · ⚡ CodeExecutor"
        if stage.get("completion_pending"):
            status_text = "结束确认中"
        details = []
        if stage.get("attempt", 0) > 1:
            details.append(f"第 {stage['attempt']} 次尝试")
        if stage.get("calls"):
            details.append(f"LLM 调用 {stage['calls']} 次")
        outcome = {"reproduced": "数值验收通过", "not_reproduced": "数值验收未通过",
                   "inconclusive": "证据不足", "prepared": "仅完成准备"}.get(stage.get("outcome"))
        if outcome:
            details.append(outcome)
        if stage.get("reason"):
            details.append(str(stage["reason"]))
        if stage.get("completion_reason"):
            details.append(stage["completion_reason"])
        if status == "running":
            details.extend("当前执行：" + context["label"] for context in phase_executions)
        detail_text = escape(" · ".join(details)).replace("\r", "").replace("\n", "<br>")
        detail_html = f'<p>{detail_text}</p>' if details else ""
        # 连续 HTML 不插入空行，避免 Markdown 将后续卡片识别为缩进代码块。
        agent_cards.append(
            f'<div class="agent-card agent-{status_class}" data-stage-id="{escape(stage["id"], quote=True)}">'
            f'<div class="agent-top"><span class="agent-index">{i + 1:02d} / {len(stages):02d}</span>'
            f'<span class="agent-status">{status_text}</span></div>'
            f'<div class="agent-name">{escape(title)}</div>'
            f'<div class="agent-title">{escape(agent)}</div>'
            f'<p>{escape(desc)}</p>{detail_html}</div>')
    if agent_cards:
        st.markdown('<div class="agent-grid">' + ''.join(card.strip() for card in agent_cards) + '</div>',
                    unsafe_allow_html=True)
        completed = sum(stage.get("status") == "success" for stage in stages)
        enabled_count = sum(stage.get("status") != "skipped" for stage in stages)
        progress = completed / enabled_count if enabled_count else 0
        task_status = "整体任务仍在运行；" if st.session_state.running else ""
        st.progress(progress, text=f"阶段完成: {completed}/{enabled_count}（{task_status}复现结论以数值验收为准）")
        st.caption("上方仅按阶段计数，不代表耗时比例；执行轮数和步骤数以实际运行记录为准。")
        with st.expander("各部分结束判定"):
            st.caption("阶段负责人返回结果，且所属执行轮次退出并完成清理，才确认该部分结束。"
                       "步骤间隙、日志中的最终轮数和完成请求都不能代替结束确认。")
            st.table([{"阶段": stage.get("title", stage["id"]),
                       "结束条件": stage.get("completion_rule", "负责人返回结果且所属执行全部结束"),
                       "结束已确认": "是" if stage.get("completion_confirmed") else "否"}
                      for stage in stages])
    else:
        st.info("启动后将按实际执行顺序显示阶段状态。")
    if snap and snap.get("execution_output"):
        with st.expander("代码执行输出（最近16000字符）", expanded=True):
            if active_executions:
                for context in active_executions:
                    st.caption(f"正在执行：{context['label']} · 进行中")
            elif execution_context:
                context_status = {"running": "进行中", "success": "本步骤完成", "error": "失败",
                                  "interrupted": "未正常结束"}.get(
                    execution_context.get("status"), "")
                st.caption(f"最近执行：{execution_context['label']} · {context_status}")
            st.caption("日志中的计数只属于对应的执行步骤，不代表整个任务的完成进度。")
            with st.container(height=240):
                st.text(snap["execution_output"])

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
            elif validation.get("status") == "insufficient_evidence":
                st.warning("论文证据不足，尚未完成论文实验：" + validation.get("reason", "缺少可核验实现"))
            elif validation.get("status") == "best_effort":
                st.warning("已尝试按现有证据重建代码，但不能判定论文复现：" + validation.get("reason", "缺少完整实验协议"))
            elif validation.get("status") in {"no_reference_metrics", "inconclusive"}:
                st.warning("复现结论无法验收：" + validation.get("reason", "缺少论文指标或完整执行证据"))
            elif validation.get("status") in {"not_runnable", "execution_incomplete", "execution_failed"}:
                st.warning("论文实验未完成：" + validation.get("reason", "代码未运行或执行证据不完整"))
            elif validation.get("status") in {"prepared", "environment_prepared", "smoke_passed"}:
                st.info(validation.get("reason", "流程已结束，请查看实验结论"))
            elif validation.get("status") == "method_experiment_completed":
                optimization = data.get("optimization") or {}
                st.success(("基线方法实验已完成" if optimization.get("mode") == "validate"
                            else "官方方法实验已完成") + "，协议与独立指标核验通过。")
                st.caption("本结论限于选定方法实验，不代表整篇论文数值复现。")
                for metric in validation.get("metric_records", []):
                    st.metric(f"{metric['name'].upper()} · {metric['split']}", f"{metric['value']:.6f} {metric.get('unit','')}")
                if optimization.get("status") == "validated_gain" and optimization.get("optimized"):
                    st.success("选定候选已通过两个随机种子的留出确认，详细数值见优化报告。")
                elif optimization.get("status") in {"budget_insufficient", "budget_exhausted"}:
                    st.warning("参数优化验证未完成：" + optimization.get("reason", "可用预算不足"))
                    tried = len(optimization.get("trials", []))
                    completed = sum(row.get("status") == "completed"
                                    for row in optimization.get("trials", []))
                    confirmed = sum(row.get("status") == "completed"
                                    for row in optimization.get("confirmation", []))
                    st.caption(f"候选训练：已尝试 {tried} 个、已完成 {completed} 个；"
                               f"两种子留出确认：已完成 {confirmed}/2。上方指标来自基线，不能据此判断优化有效。")
                    st.info("如需完成参数验证，请调整总预算（建议120分钟）后重新开始。预算是上限，不会强制运行到时限。")
                elif optimization.get("status") == "suggested":
                    st.info("智能建议已生成，尚未通过训练验证。")
                elif optimization.get("mode") in {"suggest", "validate"}:
                    st.info(optimization.get("reason", "请查看优化记录"))
            elif validation.get("status") == "quality_target_not_met":
                st.warning("方法实验已运行，但未达到预设工程效果门槛。")
            elif validation.get("is_reproduced") is True:
                st.success("🎉 完整实验已完成，论文数值验收通过！")
            else:
                st.info("流水线已结束，请查看报告中的复现结论。")
        elif result.get("state") == "ERROR":
            st.error(f"❌ 流程出错: {result.get('error', '未知错误')}")

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">RESEARCH OUTPUT</span><h2>复现报告</h2><p>查看实验结论、执行证据与验证结果。</p></div></div>', unsafe_allow_html=True)
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
    st.caption("上图展示通用主流程；方法预设可在核验后进入智能建议或参数验证阶段，详细阶段以实际流水线为准。")


# ===== Tab 5: 历史记录 =====
def render_history():
    """历史页按需加载，目录统计在本次会话中复用，刷新或清理后更新。"""
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
            st.session_state.pop("_history_storage", None)
        if "_history_storage" not in st.session_state:
            with st.spinner("正在统计历史资源与存储占用…"):
                st.session_state._history_storage = collect_storage_snapshot()
        snapshot = st.session_state._history_storage
        st.caption("存储占用为上次统计结果；安装或训练后可点击刷新更新。")
        events = list_resource_events(limit=100)
        inventory = snapshot["inventory"]
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
        storage = snapshot["storage"]
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
            st.session_state.pop("_history_storage", None)
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
            deps_items = snapshot["deps_items"]
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
                skipped = []
                n, freed = cleanup_deps_cache(keep_days=deps_keep, on_skip=skipped.append)
                st.session_state.pop("_history_storage", None)
                st.session_state["_hist_flash"] = (
                    f"已清理 {n} 个冷依赖目录，释放 {format_size(freed)}"
                    + ("；" + "；".join(skipped) if skipped else ""))
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
                    skipped = []
                    n, freed = delete_deps_cache(picked_deps, on_skip=skipped.append)
                    st.session_state.pop("_history_storage", None)
                st.session_state["_reset_confirm_keys"] = ["confirm_deps_del"]
                st.session_state["_hist_flash"] = (
                    f"已删除 {n} 个依赖目录，释放 {format_size(freed)}"
                    + ("；" + "；".join(skipped) if skipped else ""))
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
            st.session_state.pop("_history_storage", None)
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
                            st.session_state.pop("_history_storage", None)
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
                    st.session_state.pop("_history_storage", None)
                    st.session_state["_reset_confirm_keys"] = ["confirm_batch_del"]
                    st.session_state["_hist_flash"] = (
                        f"已批量删除 {len(selected)} 个会话，共 {removed} 个文件，"
                        f"释放 {format_size(freed)}")
                    st.rerun()
    except Exception as e:
        st.error(f"加载历史记录失败: {e}")


with tab5:
    if _lazy_tabs:
        if tab5.open:
            render_history()
    else:
        # 较早版本的标签切换不会通知后端，显式加载避免在首页扫描磁盘。
        if st.button("加载历史与存储", key="history_load"):
            st.session_state._history_loaded = True
        if st.session_state.get("_history_loaded"):
            render_history()


# 底部信息
st.markdown("""
<div class="app-footer">
    <span>✳ AutoReproducer</span><span>v0.2.0 · 让复现过程清晰可见</span>
</div>
""", unsafe_allow_html=True)
