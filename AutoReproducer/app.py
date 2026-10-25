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
import hashlib
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
from src.project_paths import get_data_root
from src.corpus import list_papers
from src.repository_profiles import PAPER_TITLE
from src.title_routing import resolve_title_request
from frontend.display_text import display_text
from frontend.live_progress import (elapsed_seconds, progress_signature,
                                    recent_progress_runs, render_activity, render_dependency_progress,
                                    resolve_progress_run)
from frontend import progress_state
if getattr(progress_state, "PROGRESS_VIEW_VERSION", 0) != 4:
    progress_state = importlib.reload(progress_state)
from frontend.report_renderer import render_report, report_downloads
from src.report_revisions import (can_regenerate_report, needs_report_upgrade,
                                  regenerate_report, resolve_report_revision)
from frontend.llm_config import (
    CONNECT_TEST_TIMEOUT,
    resolve_llm_config,
    config_missing,
    test_llm_connection,
)
from frontend import pipeline_entrypoint
from frontend import repository_entrypoint

if getattr(repository_entrypoint, "REPOSITORY_ENTRYPOINT_API_VERSION", 0) != 2:
    repository_entrypoint = importlib.reload(repository_entrypoint)

PROFILE_LABELS = repository_entrypoint.profile_labels()
get_profile = repository_entrypoint.get_profile

# Only refresh the small loader. Older active pipeline modules keep their globals.
if pipeline_entrypoint.BACKEND_API_VERSION != 10:
    pipeline_entrypoint = importlib.reload(pipeline_entrypoint)

_pipeline_backend = pipeline_entrypoint.load_backend_pipeline()
ProgressStore = _pipeline_backend.ProgressStore
run_pipeline_background = repository_entrypoint.background_entrypoint(
    _pipeline_backend.run_pipeline_background)
from frontend.history_manager import (
    list_sessions,
    collect_storage_snapshot,
    cleanup_runtime,
    delete_deps_cache,
    cleanup_deps_cache,
    dependency_maintenance_status,
    list_resource_events,
    format_size,
    get_session_detail,
    delete_session,
    delete_sessions,
    clear_sessions,
    get_session_report_result,
    get_project_data_dir,
)
from frontend.task_control import request_cancel, task_status as pipeline_task_status, recover_orphaned_task

# 页面配置
st.set_page_config(
    page_title="AutoReproducer | 论文复现工作台",
    page_icon=str(Path(__file__).resolve().parent / "frontend" / "favicon.png"),
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
    linked_run = resolve_progress_run(get_project_data_dir(), st.query_params.get("run", ""))
    if linked_run:
        st.session_state.progress_file = linked_run
if "pipeline_note" not in st.session_state:
    st.session_state.pipeline_note = None

# 先读取后台状态，再渲染侧边栏和主区域。终态时停止轮询后也能立即
# 更新「开始复现」按钮、系统状态和顶部提示，避免留在上一阶段。
pf = st.session_state.progress_file
if pf:
    recover_orphaned_task(pf, ProgressStore)
_task_control_status = pipeline_task_status(pf)
_progress_signature = progress_signature(pf)
snap = ProgressStore.read_snapshot(pf) if pf else None
if pf and resolve_progress_run(get_project_data_dir(), Path(pf).name) == str(Path(pf).resolve()):
    if st.query_params.get("run") != Path(pf).name:
        st.query_params["run"] = Path(pf).name
if snap:
    # 每次使用当前任务的完整快照，避免切换进度文件时残留上次的完成状态。
    st.session_state.result = snap["result"]
    st.session_state.running = (snap["running"] or _task_control_status["active"]
                                or _task_control_status.get("cleanup_failed", False))
    st.session_state.agent_status = snap.get("agent_status", {})
    st.session_state.pipeline_stages = snap.get("pipeline_stages", [])
    st.session_state.current_state = snap["state"]
    st.session_state.logs = snap["logs"]

if st.session_state.result:
    st.session_state.result = resolve_report_revision(
        st.session_state.result, data_root=get_project_data_dir())


def report_upgrade_controls(result, *, key):
    """An old session can update its presentation without rerunning the job."""
    result = resolve_report_revision(result, data_root=get_project_data_dir())
    if st.session_state.pop(key + "_updated", False):
        st.success("报告格式已更新；原报告、运行日志和执行结果已保留。")
    if needs_report_upgrade(result):
        st.info("该报告使用旧格式，可从已保存的结果更新实测指标表；不会重新训练或调用模型。")
        if can_regenerate_report(result) and not st.session_state.running:
            if st.button("重新生成报告", key=key):
                try:
                    result = regenerate_report(result, data_root=get_project_data_dir())
                    st.session_state[key + "_updated"] = True
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.warning(f"报告更新失败：{exc}")
        else:
            st.caption("任务未结束或未保留结构化结果，暂无法重新生成；原报告仍可查看和下载。")
    return result


@st.fragment
def render_llm_settings():
    """配置和连接测试局部刷新，反馈直接显示在按钮旁。"""
    with st.expander("LLM API 配置", expanded=False):
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
            "测试 AI 连接", key="test_llm_connection",
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
            st.caption("Mock 模式不调用真实 LLM，连接测试不可用；"
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


def _preview_pdf_input(payload: bytes, mock_mode: bool):
    """Bind preview caching to the parser currently loaded by the frontend."""
    pdf_module = repository_entrypoint.load_pdf_input()
    return _cached_preview_pdf_input(
        payload, mock_mode, pdf_module.PDF_SOURCE_FINGERPRINT, _pdf_module=pdf_module)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_preview_pdf_input(payload: bytes, mock_mode: bool, source_fingerprint: str, _pdf_module):
    """Inspect bytes once per source generation without hashing module objects."""
    pdf_module = _pdf_module
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


def render_paper_input():
    """论文输入变化时刷新页面，同步显示适用的执行和核验选项。"""
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
            st.caption("将取得固定版本并运行：" + selected["label"])
            if selected.get("adapter_id"):
                st.caption("结论限于选定官方方法实验；论文完整基准需要另行适配。")
        elif not st.session_state.mock_mode:
            st.caption("未匹配预设时先定位仓库，运行作者代码需上传 PDF。")
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
                resolution = preview["resolution"]
                if resolution.get("source") == "pdf_author_code_repository":
                    st.success("PDF 原文作者代码声明已匹配实验：" + resolution["discovery_repository_url"])
                    st.caption("作者主仓库与实际训练仓库分别记录；经 ReZero-examples 的 README 定位训练仓库："
                               + resolution["training_repository_url"])
                    with st.expander("作者仓库到训练实验的引用链"):
                        for hop in resolution["repository_relationship"]["hops"]:
                            st.markdown(f"[{hop['repository_url']} · README 第 {hop['line']} 行]"
                                        f"({hop['source_url']}) → [{hop['target_repository_url']}]"
                                        f"({hop['linked_url']})")
                            st.text(hop["quote"])
                else:
                    st.success("PDF 首页已匹配作者仓库：" + selected["repository"]["url"])
                st.markdown(f"[论文]({selected['paper']['url']}) · [作者代码]({selected['repository']['url']})")
                st.caption("将执行：" + selected["label"] + "；结果由实际运行输出给出。")
            else:
                st.caption("从论文正文寻找仓库并尝试运行，实际报错会显示在日志中。")
                if (not st.session_state.mock_mode
                        and st.session_state.get('corpus_paper_select', '无') == '无'):
                    st.caption('按作者 README 直接运行仓库代码，缺包时自动安装并重试。')
            links = preview.get("repository_links", [])
            if links:
                primary = next((link for link in links if link.get("is_author_code")), None)
                if primary:
                    st.success("PDF 原文代码声明链接：" + primary["url"])
                    st.caption(f"来源：第 {primary['page']} 页；将优先于模型猜测和关键词搜索。")
                else:
                    candidate = next((link for link in links if link.get("evidence_type") != "reference"), None)
                    if candidate:
                        st.info("PDF 仓库候选链接：" + candidate["url"])
                st.caption("获取仓库源码后直接安装依赖并运行。")
                with st.expander("PDF 仓库链接提取依据"):
                    for link in links[:12]:
                        kind = {"author_code_statement": "代码公开声明", "reference": "参考文献",
                                "repository_link": "仓库链接"}.get(link["evidence_type"], "仓库链接")
                        st.markdown(f"[第 {link['page']} 页 · {kind}]({link['url']})")
                        if link.get("context"):
                            st.text(link["context"])
            if not preview.get("error"):
                from frontend.repository_fallback import fallback_for_pdf, repository_widget_key
                pdf_hash = (preview.get("pdf_input") or {}).get("sha256", "")
                fallback = fallback_for_pdf(st.session_state.result, pdf_hash)
                if fallback:
                    st.warning("自动仓库确认未通过，可手动补充 GitHub 仓库后重试。")
                    st.caption("未确认原因：" + fallback.get("reason", "缺少足够证据"))
                    previous_identity = ((st.session_state.result or {}).get("data", {}).get("resources") or {}).get("repository_identity") or {}
                    previous_url = (((st.session_state.result or {}).get("data", {}).get("resources") or {}).get("code_repo_url", "")
                                    if previous_identity.get("source") == "user_supplied_pinned_git_llm_review" else "")
                    st.text_input("补充 GitHub 仓库", key=repository_widget_key(pdf_hash), value=previous_url,
                                  placeholder="https://github.com/owner/repository",
                                  disabled=st.session_state.running)
                    st.caption("填写后点击“开始复现”。默认直接获取并尝试运行该仓库；报告记录为用户提供，作者归属未确认。")
    else:
        st.selectbox("真实论文实验", list(PROFILE_LABELS),
                     format_func=lambda key: display_text(PROFILE_LABELS[key]), key="experiment_profile")
        selected = get_profile(st.session_state.experiment_profile)
        st.caption(selected["paper"]["title"])
        st.markdown(f"[论文]({selected['paper']['url']}) · [作者代码]({selected['repository']['url']})")
        if selected.get("adapter_id") in {"siren", "neural_ode"}:
            st.selectbox("实验操作", ["运行实验", "准备实验环境", "仅准备源码和命令"], key="method_action")
            st.caption('自动安装运行依赖，直接执行作者实验。')
            return
        st.checkbox("仅准备代码、数据和命令（不训练）", key="repository_prepare_only")
        st.caption('直接执行作者实验，首次运行会安装隔离依赖。')



# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon" aria-hidden="true">AR</span>
        <div><strong>AutoReproducer</strong><small>RESEARCH WORKSPACE</small></div>
    </div>
    <div class="sidebar-intro">新建复现实验</div>
    <p class="sidebar-help">准备论文，开始新的实验。</p>
    """, unsafe_allow_html=True)

    with st.expander("运行环境", expanded=False):
        st.session_state.mock_mode = st.toggle(
            "Mock 演示模式（无需 API）",
            value=st.session_state.mock_mode,
            help="演示界面与流程，不执行真实论文实验或调用模型服务")

        # Current web inputs use author repositories. Engine availability alone
        # does not make those execution paths support Docker; normalize old sessions.
        st.session_state.use_docker = False
        st.toggle("Docker 沙箱执行（暂未开放）", value=False, disabled=True,
                  help="当前网页的作者仓库与固定预设使用本地受控进程；Docker 执行尚未接通。")
        if st.session_state.mock_mode:
            st.caption("Mock 模式不执行真实代码，无需 Docker")
        else:
            st.caption("作者仓库和固定预设使用本地受控进程，依赖安装在隔离目录；"
                       "依赖隔离不等于容器沙箱。")

        # 预留可选入口；当前版本不执行优化，也不消耗优化预算。
        st.session_state.enable_optimization = False
        st.checkbox("启用智能优化（预留）", key="enable_optimization", disabled=True)
        st.caption("当前所有流程均未开放智能优化；固定预设也只执行所选作者实验。")

    render_llm_settings()
    base_url = st.session_state.llm_base_url
    api_key = st.session_state.llm_api_key
    model_name = st.session_state.llm_model

    # 论文输入
    render_paper_input()
    input_mode = st.session_state.input_mode
    uploaded_file = (st.session_state.get("pdf_uploader")
                     if input_mode == "上传PDF" else None)
    # Reconnecting to a restarted server can leave a DeletedFile marker in
    # widget state while file_uploader itself returns None. It is not a PDF.
    if uploaded_file is not None and not callable(getattr(uploaded_file, "getvalue", None)):
        uploaded_file = None
        st.warning("上传的 PDF 已失效，请重新上传后再开始复现。")
    experiment_profile = (st.session_state.get("experiment_profile")
                          if input_mode == "官方仓库预设" else None)
    manual_repo_url, manual_repo_error = "", ""
    author_execution_only = not st.session_state.mock_mode
    if uploaded_file and not st.session_state.mock_mode:
        from frontend.repository_fallback import manual_repository, repository_widget_key
        uploaded_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        manual_repo_url, manual_repo_error = manual_repository(
            st.session_state.get(repository_widget_key(uploaded_hash), ""))
    method_selected = bool(experiment_profile and get_profile(experiment_profile).get("adapter_id") in {"siren", "neural_ode"})
    requires_api = not experiment_profile

    # 低频选项收纳在折叠区，保留论文输入与启动操作的空间。
    with st.expander("语料对照 · 可选", expanded=False):
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
            requires_api = False
    elif input_mode == "上传PDF" and uploaded_file and not corpus_paper:
        preview = _preview_pdf_input(uploaded_file.getvalue(), st.session_state.mock_mode)
        if preview.get("resolution") and not manual_repo_url:
            # The backend re-reads the immutable upload and rebuilds its evidence.
            # Choosing the reviewed route here also selects the fresh worker when
            # this Streamlit process still owns cached earlier adapters.
            experiment_profile = preview["resolution"]["profile"]
            method_selected = False
            requires_api = False

    if uploaded_file and not st.session_state.mock_mode and not experiment_profile and not corpus_paper:
        author_execution_only = True

    strict_validation = False
    experiment_request = {}
    source_repair = False
    source_repair_auto = False
    source_repair_editable = []
    if (input_mode == "上传PDF" and uploaded_file and not st.session_state.mock_mode
            and not experiment_profile and not corpus_paper):
        requested_method = st.text_input(
            "实验方法 / 模型（可选）", key="pdf_requested_method", max_chars=128,
            disabled=st.session_state.running,
            help="填写论文或作者代码中的名称；留空时自动选择完整实验。不会据此修改作者训练设置。")
        requested_dataset = st.text_input(
            "实验数据集（可选）", key="pdf_requested_dataset", max_chars=128,
            disabled=st.session_state.running)
        experiment_request = {name: text.strip() for name, text in
            (("method", requested_method), ("dataset", requested_dataset)) if text.strip()}
        def _select_pdf_execution_mode(selected, alternative):
            if st.session_state.get(selected):
                st.session_state[alternative] = False

        if "pdf_source_repair" not in st.session_state:
            st.session_state["pdf_source_repair"] = not st.session_state.get("pdf_strict_validation", False)
        if (st.session_state.get("pdf_source_repair")
                and st.session_state.get("pdf_strict_validation")):
            st.session_state["pdf_strict_validation"] = False
        strict_validation = st.checkbox(
            "严格核验：重载模型并独立复算指标", key="pdf_strict_validation",
            disabled=st.session_state.running,
            on_change=_select_pdf_execution_mode,
            args=("pdf_strict_validation", "pdf_source_repair"),
            help="请求在作者代码运行后检查实验协议、重载模型并独立计算指标；暂不支持核验时继续普通运行并记录原因。")
        if strict_validation:
            st.caption("严格核验需要完整模型、数据划分和实验设置证据；不支持时继续普通运行并记录核验不可用，结果以实际检查为准。")
        source_repair = st.checkbox(
            "兼容性源码修复：失败后自动重试", key="pdf_source_repair",
            disabled=st.session_state.running,
            on_change=_select_pdf_execution_mode,
            args=("pdf_source_repair", "pdf_strict_validation"),
            help="按真实报错定位兼容性问题，最多两轮隔离修复和原命令重试；保留补丁与日志，不得改变实验协议。")
        if source_repair:
            strict_validation = False
            editable_text = st.text_area(
                "限定修复文件（可选，每行一个仓库相对路径）",
                key="pdf_source_repair_editable",
                placeholder="src/model.py\ntrain.py",
                disabled=st.session_state.running,
                help="留空时从实际报错自动定位最多 3 个冻结 Python 文件；填写后仅允许修复这些文件，不接受目录或通配符。")
            source_repair_editable = list(dict.fromkeys(
                line.strip() for line in editable_text.splitlines() if line.strip()))
            source_repair_auto = not source_repair_editable
            st.caption("源码修复与严格核验互斥：修复会改变源码，严格核验要求冻结源码。修复不能改变模型、数据划分、训练轮数或评价方法。")

    # 启动 / 重置按钮
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("开始复现", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.running)
    with col2:
        stop_btn = False
        if st.session_state.running:
            stop_btn = st.button("正在停止…" if _task_control_status["cancelling"] else "停止任务",
                                 use_container_width=True, key="stop_pipeline",
                                 disabled=_task_control_status["cancelling"])
        reset_btn = st.button("重置", use_container_width=True,
                              disabled=st.session_state.running)
        if st.session_state.running:
            st.caption("停止会中断本任务的下载、安装和执行；清理完成后可重置，日志与报告保留。")

    # 系统状态
    _sidebar_state_label = ("运行中" if st.session_state.running else
                            {"INIT": "等待开始", "COMPLETED": "已完成", "ERROR": "运行异常"}
                            .get(st.session_state.current_state, st.session_state.current_state))
    st.markdown(
        f'<div class="sidebar-status" data-state="{escape(st.session_state.current_state, quote=True)}"><span>工作台状态</span>'
        f'<strong>{escape(_sidebar_state_label)}</strong></div>',
        unsafe_allow_html=True)


# ========== 事件处理 ==========
if stop_btn:
    if request_cancel(st.session_state.progress_file):
        st.session_state.pipeline_note = "已请求停止，正在结束本任务的子进程并释放环境；清理完成后可重置。"
        st.rerun()
    else:
        st.sidebar.error("无法确认这个旧任务的后台持有者，不能安全停止。请先停止启动此任务的旧服务，再刷新恢复；日志已保留。")


def _save_uploaded_pdf(uploaded_file) -> str:
    """将上传的 PDF 保存为临时文件，返回路径。"""
    suffix = os.path.splitext(uploaded_file.name or "paper.pdf")[1] or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="autorepro_paper_")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


def _new_progress_file() -> str:
    """创建本次复现的进度文件路径（data/runtime/progress_<毫秒>.jsonl）。"""
    runtime_dir = get_data_root() / "runtime"
    os.makedirs(runtime_dir, exist_ok=True)
    return os.path.join(runtime_dir,
                        f"progress_{int(time.time() * 1000)}.jsonl")


# 启动按钮处理：后台线程执行流水线，主线程立即返回并轮询进度
if start_btn and not st.session_state.running:
    pt = st.session_state.paper_title.strip() if input_mode == "论文标题" else ""
    if experiment_profile and st.session_state.mock_mode:
        st.sidebar.error("官方仓库预设需要关闭 Mock 模式，才能执行真实论文代码。")
    elif manual_repo_error:
        st.sidebar.error(manual_repo_error)
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
            st.warning("未填写 API Key：如果上游服务需要鉴权"
                       "（如 DeepSeek/OpenAI），调用会返回 401 错误文本；"
                       "建议先在侧边栏填写 Key 并点击「测试 AI 连接」验证。")
        tmp_pdf = _save_uploaded_pdf(uploaded_file) if uploaded_file else ""
        progress_file = _new_progress_file()
        previous_monitor = {key: st.session_state.get(key) for key in (
            "progress_file", "running", "result", "logs", "agent_status",
            "pipeline_stages", "current_state", "pipeline_note")}
        st.session_state.progress_file = progress_file
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.logs = []
        st.session_state.agent_status = {}
        st.session_state.pipeline_stages = []
        st.session_state.current_state = "INIT"
        st.session_state.pipeline_note = (
            "复现流水线已在后台启动，进度实时刷新中…")
        try:
            run_pipeline_background(
                progress_file,
                paper_title=pt, pdf_path=tmp_pdf,
                **({'author_execution_only': True} if author_execution_only else {}),
                strict_validation=strict_validation,
                experiment_request=experiment_request,
                source_repair=source_repair,
                source_repair_auto=source_repair_auto,
                source_repair_editable=source_repair_editable,
                **({"preferred_repo_url": manual_repo_url} if manual_repo_url else {}),
                corpus_paper=corpus_paper,
                experiment_profile=experiment_profile,
                title_resolution=title_resolution,
                prepare_only=False if title_resolution or tmp_pdf else (st.session_state.get("method_action") == "仅准备源码和命令" if method_selected else
                              bool(st.session_state.get("repository_prepare_only")) if experiment_profile else False),
                prepare_environment=method_selected and st.session_state.get("method_action") == "准备实验环境",
                optimization_mode="off",
                max_candidates=int(st.session_state.get("method_max_candidates", 3)),
                budget_seconds=7200,  # Legacy argument; current execution has no deadline.
                use_llm_review=False, allow_result_summary_review=False,
                model_name=model_name, base_url=base_url,
                api_key=api_key,
                mock_mode=st.session_state.mock_mode,
                enable_optimization=False,
                use_docker=False,
                cleanup_pdf=True)   # 临时 PDF 由后台线程负责删除
        except Exception as exc:
            from src.runtime_preparation import safe_runtime_diagnostic
            message = safe_runtime_diagnostic(exc)
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
            for key, value in previous_monitor.items():
                st.session_state[key] = value
            if tmp_pdf:
                try:
                    Path(tmp_pdf).unlink(missing_ok=True)
                except OSError:
                    pass
            if Path(progress_file).is_file():
                try:
                    ProgressStore(progress_file, reset=False).emit({"type": "error", "error": "任务未启动：" + message})
                except OSError:
                    pass
            st.sidebar.error("任务未启动：" + message)
        else:
            st.session_state.workspace_tabs = "流水线状态"
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
    st.query_params.pop("run", None)
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
_mode_label = "Mock 演示" if st.session_state.mock_mode else "真实运行"
_runtime_label = ("流程演示" if st.session_state.mock_mode else
                  "Docker 沙箱" if st.session_state.use_docker else "本地隔离环境")
st.markdown(f"""
<div class="hero">
    <div class="hero-top">
        <span class="hero-eyebrow"><span class="eyebrow-mark" aria-hidden="true"></span> YOUR RESEARCH WORKSPACE</span>
        <span class="hero-status hero-status-{_hero_status}">{escape(_state_label)}</span>
    </div>
    <div class="hero-content">
        <div class="hero-copy">
            <h1>论文复现工作台</h1>
            <p>从论文到代码，让每一步实验都有据可查。</p>
            <div class="workspace-context"><span>{escape(_mode_label)}</span><span>{escape(_runtime_label)}</span><span>实验全程记录</span></div>
        </div>
        <div class="hero-visual" aria-hidden="true">
            <div class="visual-orbit"></div>
            <div class="paper-sheet paper-sheet-back"></div>
            <div class="paper-sheet"><span class="paper-tag">PAPER</span><div class="paper-line paper-line-title"></div><div class="paper-line"></div><div class="paper-line paper-line-short"></div><div class="paper-chart"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
            <div class="visual-caption"><span class="visual-symbol">↗</span><span>让研究走向实践<small>READ · RUN · RECORD</small></span></div>
        </div>
    </div>
    <div class="hero-steps" aria-label="复现流程"><span><b>01</b> 解析论文</span><span><b>02</b> 准备实验</span><span><b>03</b> 执行实验</span><span><b>04</b> 生成报告</span></div>
</div>
""", unsafe_allow_html=True)

# 新版 Streamlit 支持按所选标签延迟执行；旧版保留显式加载入口。
_poll_initialized = False


@st.fragment(run_every=2 if st.session_state.running and pf else None)
def poll_live_progress():
    # Native Streamlit polling works across tabs without an iframe dependency.
    # A timer only rereads files; it never relaunches a task or fabricates progress.
    global _poll_initialized
    # Always finish the initial page render, even if a busy worker appends
    # between the snapshot read and this fragment. Only timer ticks rerun it.
    if _poll_initialized and pf and (progress_signature(pf) != _progress_signature
                                     or pipeline_task_status(pf) != _task_control_status):
        st.rerun()
    _poll_initialized = True
    if pf:
        if st.button("刷新运行状态", key="refresh_live_progress"):
            st.rerun()
        age = elapsed_seconds((snap or {}).get("updated_at"))
        if st.session_state.running:
            st.caption("监控每 2 秒自动读取后台记录。"
                       + (f"最近一条事件在 {age} 秒前。" if age is not None else "等待第一条后台事件。"))
            for row in (snap or {}).get("pipeline_stages", []):
                if row.get("status") == "running":
                    duration = elapsed_seconds(row.get("started_at"))
                    st.caption("当前阶段：" + display_text(row.get("title", row["id"]))
                               + (f" · 已持续 {duration} 秒" if duration is not None else ""))
            if age is not None and age >= 30:
                st.caption("后台暂未产生新日志，仍等待阶段返回；不能据此判定任务已完成。")
        render_dependency_progress(snap or {}, st_module=st)


poll_live_progress()

if not pf:
    resumable = recent_progress_runs(get_project_data_dir())
    if resumable:
        with st.expander("恢复已有任务监控", expanded=True):
            st.caption("刷新页面后可重新连接已有运行记录；按时间从新到旧排列。")
            selected_run = st.selectbox("运行记录", resumable, key="resume_progress_selection")
            if st.button("恢复监控", key="resume_progress"):
                st.session_state.progress_file = resolve_progress_run(get_project_data_dir(), selected_run)
                st.rerun()

_lazy_tabs = "on_change" in inspect.signature(st.tabs).parameters
_previous_tab = st.session_state.get("workspace_tabs", "")
if _previous_tab and display_text(_previous_tab) != _previous_tab:
    st.session_state.workspace_tabs = display_text(_previous_tab)
_tab_options = {"key": "workspace_tabs", "on_change": "rerun"} if _lazy_tabs else {}
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "流水线状态", "复现报告", "审计日志", "状态机", "历史记录",
], **_tab_options)

# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("""<div class="section-heading"><div><span class="section-kicker">01 / EXPERIMENT WORKFLOW</span>
    <h2>复现流水线</h2><p>从论文解析到报告生成，实时追踪每个阶段。</p></div>
    <span class="section-aside">每一步，清晰可追溯</span></div>""",
                unsafe_allow_html=True)

    # ---------- 后台复现实时进度（轮询进度文件） ----------
    if snap and snap.get("error"):
        st.error(f"后台流水线异常: {snap['error']}")

    stages = st.session_state.pipeline_stages
    execution_context = (snap or {}).get("execution_context") or {}
    active_executions = (snap or {}).get("active_executions") or []
    active_runs = (snap or {}).get("active_runs") or []
    active_workers = (snap or {}).get("active_workers") or []
    pdf_source = (snap or {}).get("pdf_resolution") or {}
    if pdf_source:
        selected = get_profile(pdf_source["profile"])
        st.caption("根据 PDF 首页识别论文：" + pdf_source["title"])
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
        title = display_text(stage.get("title", stage.get("agent", "执行阶段")))
        desc = display_text(stage.get("description", ""))
        agent = display_text(stage.get("agent", ""))
        phase_executions = [context for context in active_executions
                            if context.get("phase_id") == stage["id"]]
        phase_runs = [run for run in active_runs if run.get("phase_id") == stage["id"]]
        # Historical plans may call this role "complete training". Its actual
        # responsibility is code execution, regardless of the paper or commands.
        if agent == "CodeExecutor":
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
            agent += " · CodeExecutor"
            desc = "正在执行优化试验代码，当前轮次完成后继续后续处理。"
        elif stage["id"] == "method_advice" and status == "running" and phase_runs:
            title = "优化试验：执行轮次未结束"
            status_text = "执行轮次未结束"
            agent += " · CodeExecutor"
        if stage.get("completion_pending"):
            status_text = "结束确认中"
        details = []
        if stage.get("started_at"):
            details.append("开始于 " + stage["started_at"][11:19])
        if stage.get("finished_at") and stage.get("completion_confirmed"):
            duration = elapsed_seconds(stage.get("started_at"), stage["finished_at"])
            details.append("结束于 " + stage["finished_at"][11:19]
                           + (f"（{duration} 秒）" if duration is not None else ""))
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
            latest_activity = next((row for row in reversed((snap or {}).get("activities", []))
                                    if row.get("phase_id") == stage["id"]), None)
            if latest_activity:
                details.append("最近动态：" + latest_activity["message"][-500:])
        detail_text = escape(display_text(" · ".join(details))).replace("\r", "").replace("\n", "<br>")
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
        st.progress(progress, text=f"阶段完成: {completed}/{enabled_count}（{task_status}结果以真实命令退出状态为准）")
        st.caption("上方仅按阶段计数，不代表耗时比例；执行轮数和步骤数以实际运行记录为准。")
        with st.expander("各部分结束判定"):
            st.caption("阶段负责人返回结果，且所属执行轮次退出并完成清理，才确认该部分结束。"
                       "步骤间隙、日志中的最终轮数和完成请求都不能代替结束确认。")
            st.table([{"阶段": display_text(stage.get("title", stage["id"])),
                       "结束条件": display_text(stage.get("completion_rule", "负责人返回结果且所属执行全部结束")),
                       "结束已确认": "是" if stage.get("completion_confirmed") else "否"}
                      for stage in stages])
    elif not st.session_state.running and not st.session_state.result:
        st.markdown("""<section class="start-guide">
<div class="guide-intro"><span class="empty-symbol" aria-hidden="true"></span><div><span class="section-kicker">START A NEW EXPERIMENT</span><h3>从一篇论文，开始新的实验</h3><p>在左侧选择输入方式，点击「开始复现」。运行过程与结果会呈现在这里。</p></div></div>
<div class="guide-options"><div class="guide-option"><span class="option-mark">01</span><h4>上传论文 PDF</h4><p>从论文正文提取方法与代码线索。</p><span class="option-note">适合手头已有论文</span></div><div class="guide-option"><span class="option-mark">02</span><h4>输入论文标题</h4><p>用标题查找论文与对应的作者仓库。</p><span class="option-note">适合快速开始探索</span></div><div class="guide-option"><span class="option-mark">03</span><h4>选择官方预设</h4><p>使用已配置的作者实验与运行命令。</p><span class="option-note">适合体验完整流程</span></div></div>
<div class="guide-footnote"><span aria-hidden="true">↖</span> 首次使用？展开左侧「LLM API 配置」，连接模型服务。</div>
</section>""", unsafe_allow_html=True)
    elif st.session_state.running:
        st.info("正在准备实验，阶段状态将在这里更新。")
    if snap:
        render_activity(snap, st_module=st)
    collaboration_messages = (snap or {}).get('agent_messages', [])
    if not collaboration_messages and st.session_state.result:
        result_data = st.session_state.result.get('data') or {}
        collaboration_messages = (result_data.get('agent_collaboration') or {}).get('messages', [])
    if collaboration_messages:
        with st.expander(f"Agent 协作记录（{len(collaboration_messages)} 条）", expanded=False):
            st.caption("查看证据交接、补充请求和校验反馈。协作消息不代表实验已核验通过。")
            for message in collaboration_messages[-40:]:
                st.markdown(f"**{message['sender']} → {message['recipient']}** · {message['kind']}")
                if message.get('reply_to'):
                    st.caption('回复：' + str(message['reply_to']))
                st.json(message.get('content'), expanded=False)
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
        st.markdown("### 运行摘要")
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
            elif validation.get("status") == "reference_not_met":
                st.warning("选定论文实验已完成，独立评估数值未达到固定参考：" + validation.get("reason", "查看指标差异"))
                if (data.get("experiment_spec") or {}).get("adapter_id") == "rezero":
                    accuracy = ((validation.get("metrics_comparison") or {}).get("actual") or {}).get("top1_accuracy_pct")
                    if type(accuracy) in (int, float):
                        st.caption(f"完整 CIFAR-10 测试集 Top-1：{accuracy:.2f}% / 固定验收参考：94.00%。")
                    st.caption("本次代码执行与协议核验已完成；数值未通过不等于代码执行失败。")
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
            elif validation.get('status') == 'author_code_completed':
                st.success('所选仓库代码已运行完成，命令已正常退出。')
                if validation.get('training_observed') is False:
                    st.caption('本次未记录到优化器更新；详见作者程序输出。')
                for warning in validation.get('warnings', []):
                    st.warning(warning)
                comparison = validation.get('author_claim_comparison') or {}
                if comparison.get('comparisons'):
                    st.caption('已对比作者输出与论文宣称，详见报告中的差值和实验设置；本次未进行论文数值审核。')
                else:
                    st.caption('本次完成的是代码运行，论文数值未做验收。')
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
                    st.info("此为历史优化记录；当前版本未开放参数优化，重新运行只执行所选作者实验。")
                elif optimization.get("status") == "suggested":
                    st.info("智能建议已生成，尚未通过训练验证。")
                elif optimization.get("mode") in {"suggest", "validate"}:
                    st.info(optimization.get("reason", "请查看优化记录"))
            elif validation.get("status") == "quality_target_not_met":
                st.warning("方法实验已运行，但未达到预设工程效果门槛。")
            elif validation.get("is_reproduced") is True:
                if (data.get("experiment_spec") or {}).get("adapter_id") == "rezero":
                    st.success("ReZero CIFAR-10 完整实验已完成，独立数值验收通过。")
                    st.caption("本结论限于 ReZero PreActResNet18 的 CIFAR-10 超收敛实验，不覆盖 enwiki8 或整篇论文。")
                else:
                    st.success("完整实验已完成，论文数值验收通过。")
            else:
                st.info("流水线已结束，请查看报告中的复现结论。")
        elif result.get("state") == "ERROR":
            st.error(f"流程出错: {result.get('error', '未知错误')}")
            failure_data = result.get('data') or {}
            attempts = ((failure_data.get('plan_rejection') or {}).get('candidate_attempts')
                        or failure_data.get('planning_trace') or [])
            attempts = [item for item in attempts if item.get('repository_url')]
            analyses = failure_data.get('failure_analyses') or []
            install = (((failure_data.get('environment_preparation') or {})
                        .get('effective_env_config') or {}).get('dependency_install_attempts') or [])
            if attempts or install or analyses:
                with st.expander('查看自动规划与环境修复记录'):
                    if attempts:
                        st.caption('已检查的仓库及阻塞原因')
                        st.dataframe([{'仓库': item.get('repository_url'),
                                       '版本': item.get('revision', ''),
                                       '结果': item.get('planning_status', '获取失败'),
                                       '原因': str(item.get('error', ''))[:600]}
                                      for item in attempts], hide_index=True)
                    if install:
                        st.caption('依赖下载来源及失败类别')
                        st.dataframe([{'来源': item.get('index_url'),
                                       '成功': item.get('success'),
                                       '类别': item.get('failure_kind') or '',
                                       '诊断': str(item.get('diagnostic', ''))[:600]}
                                      for item in install], hide_index=True)
                    if analyses:
                        st.caption('LLM 失败分析与已验证的修复动作')
                        st.dataframe([{'类别': item.get('failure_kind'),
                                       '解释器': item.get('python_version') or '',
                                       '动作': item.get('action'),
                                       '原因': str(item.get('reason') or item.get('analysis_error') or '')[:600]}
                                      for item in analyses], hide_index=True)

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">02 / REPRODUCTION REPORT</span><h2>复现报告</h2><p>查看实验结论、执行证据与验证结果。</p></div></div>', unsafe_allow_html=True)
    source_request = ((st.session_state.result or {}).get("data") or {}).get("source_request") or {}
    if source_request.get("status") == "needs_pdf":
        st.info(source_request.get("next_action") or "请上传对应论文的原始 PDF 后运行作者代码。")
    if st.session_state.result and st.session_state.result.get("data", {}).get("report"):
        st.session_state.result = report_upgrade_controls(st.session_state.result, key="upgrade_current_report")
        report = st.session_state.result["data"]["report"]
        # 深色 IDE 面板渲染已回滚（见 CHANGELOG [2026.09.20-12]）：
        # 面板在真实浏览器里代码不可见，改回原生 Markdown 渲染。
        result = st.session_state.result
        report_path = result.get("report_path") or result["data"].get("report_path")
        render_report(report, report_path, st_module=st)
        report_downloads(result, key="current_report", st_module=st,
                         data_root=get_project_data_dir())
    else:
        st.markdown('<div class="empty-state"><span class="empty-symbol" aria-hidden="true"></span><span class="section-kicker">YOUR NEXT DISCOVERY</span><div class="empty-title">报告将在这里生成</div><p>在左侧提交论文并启动复现。<br>完成后，实验记录与运行结果会整理为可下载的报告。</p></div>', unsafe_allow_html=True)

# ===== Tab 3: 审计日志 =====
with tab3:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">03 / AUDIT TRAIL</span><h2>审计日志</h2><p>按 Agent 和状态筛选，查看每一步的运行记录。</p></div></div>', unsafe_allow_html=True)
    if st.session_state.logs:
        col1, col2 = st.columns(2)
        with col1:
            filter_agent = st.selectbox(
                "按执行角色筛选",
                ["全部"] + sorted(
                    set(l.get("agent", "") for l in st.session_state.logs)),
                key="filter_agent_tab3", format_func=display_text)
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
            status_label = {"SUCCESS": "完成", "ERROR": "错误", "START": "开始",
                            "RUNNING": "运行中", "WARNING": "需注意"}.get(
                                log.get("status", ""), log.get("status", "记录"))
            with st.expander(
                f"{status_label} · {log.get('elapsed_sec', 0):.1f} s · "
                f"{display_text(log.get('agent', '?'))} · {display_text(log.get('action', '?'))}"
            ):
                st.json(log)
    else:
        st.markdown('<div class="empty-state"><span class="empty-symbol empty-symbol-log" aria-hidden="true"></span><span class="section-kicker">EVERY STEP, RECORDED</span><div class="empty-title">暂无运行记录</div><p>启动复现后，这里会记录各 Agent 的执行过程。<br>你可以按执行角色与状态筛选，定位关键步骤。</p></div>', unsafe_allow_html=True)

# ===== Tab 4: 状态机 =====
with tab4:
    st.markdown('<div class="section-heading"><div><span class="section-kicker">04 / STATE TRANSITIONS</span><h2>状态机定义</h2><p>系统通过有限状态机管理 Agent 的流转与异常恢复。</p></div></div>', unsafe_allow_html=True)
    state_info = """
```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> READ_PAPER
    READ_PAPER --> FIND_RESOURCES
    FIND_RESOURCES --> BUILD_ENV
    BUILD_ENV --> EXECUTE_CODE
    EXECUTE_CODE --> GENERATE_REPORT
    GENERATE_REPORT --> COMPLETED
    READ_PAPER --> ERROR
    FIND_RESOURCES --> ERROR
    BUILD_ENV --> ERROR
    EXECUTE_CODE --> ERROR
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
        {"状态": "EXECUTE_CODE", "说明": "运行代码并记录真实输出", "Agent": "CodeExecutor"},
        {"状态": "GENERATE_REPORT", "说明": "生成Markdown复现报告", "Agent": "ReportGenerator"},
        {"状态": "COMPLETED", "说明": "流水线完成", "Agent": "—"},
        {"状态": "ERROR", "说明": "出错状态，可重试", "Agent": "—"},
    ]
    st.table(state_data)
    st.caption("上图展示当前主流程；运行完成以命令退出和进程清理状态为准。")


# ===== Tab 5: 历史记录 =====
def render_history():
    """历史页按需加载，目录统计在本次会话中复用，刷新或清理后更新。"""
    st.markdown('<div class="section-heading"><div><span class="section-kicker">05 / EXPERIMENT ARCHIVE</span><h2>复现历史与存储</h2><p>回顾已运行的会话，并管理实验产物与缓存。</p></div></div>', unsafe_allow_html=True)

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
    if st.session_state.result:
        report_downloads(st.session_state.result, key="history_current", st_module=st,
                         data_root=get_project_data_dir())

# -- 存储仪表板 --
    # -- 资源下载/安装监控 --
    with st.expander("资源下载与安装监控", expanded=True):
        if st.button("刷新资源状态", key="resource_refresh",
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
            st.caption("暂无资源清单。运行一次复现后，依赖和数据集会显示在这里。")
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
    st.markdown("#### 存储占用")
    try:
        storage = snapshot["storage"]
        overview = [
            ("已统计占用", storage.get("total", {}).get("bytes", 0)),
            ("依赖缓存", storage.get("deps", {}).get("bytes", 0)),
            ("代码与数据", sum(storage.get(key, {}).get("bytes", 0)
                               for key in ("repos", "datasets"))),
            ("报告与日志", sum(storage.get(key, {}).get("bytes", 0)
                               for key in ("reports", "logs", "experiment_ledger", "runtime"))),
        ]
        for column, (label, size) in zip(st.columns(4), overview):
            column.metric(label, format_size(size))
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
        with st.expander("查看各目录占用"):
            st.dataframe([
                {"存储类别": label,
                 "文件数": storage.get(key, {}).get("files", 0),
                 "占用": format_size(storage.get(key, {}).get("bytes", 0))}
                for label, key in metrics
            ], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"获取存储统计失败: {e}")

    # -- 一键清理 --
    with st.expander("清理管理"):
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
        st.markdown("**依赖缓存**（跨会话共享，不随历史记录删除）")
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
                  "复用基础环境": d.get('base_environment') or '—',
                  "最后使用": (d["last_used"] or "")[:16].replace("T", " ")}
                 for d in deps_items],
                use_container_width=True, hide_index=True)
            st.caption(f"{len(deps_items)} 个目录 · 合计 {format_size(total_deps)}")
            st.caption('增量环境只保存缺少的包，基础环境按 Python 和 CPU／CUDA 类型隔离。'
                       '仍被引用或正在运行的环境会保留；缓存使用前后和新安装前均自动维护。')
            try:
                maintenance = dependency_maintenance_status()
                policy, latest = maintenance['policy'], maintenance['latest']
                free = format_size(maintenance['free_bytes']) if maintenance['free_bytes'] is not None else '未测量'
                st.caption(f"容量目标 {format_size(policy['quota_bytes'])} · 磁盘余量 {free} · "
                           f"保留最近 {policy['keep_recent']} 个环境 · 闲置 {policy['keep_days']} 天后回收")
                if latest:
                    st.caption('最近维护：' + latest.get('reason', '已完成') +
                               f" 回收环境文件 {format_size(latest.get('freed_bytes', 0))}，" +
                               f"安装临时文件 {format_size(latest.get('temporary_files', {}).get('freed_bytes', 0))}。")
            except (OSError, ValueError) as error:
                st.error(f'读取依赖维护状态失败: {error}')

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
            if st.button(f"删除所选 {len(picked_deps)} 个依赖目录",
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
        st.markdown("**历史会话清理**（危险操作，请谨慎）")
        confirm_clear = st.checkbox(
            "我确认清空全部历史复现会话（实验账本/审计日志/复现报告/runtime 进度）",
            key="confirm_clear_all")
        if st.button("清空全部历史", use_container_width=True,
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
    st.markdown("#### 历史复现会话")
    try:
        sessions = list_sessions()
        if not sessions:
            st.markdown('<div class="empty-state empty-state-compact"><span class="empty-symbol" aria-hidden="true"></span><div class="empty-title">让每次实验都有迹可循</div><p>暂无历史复现记录。完成一次复现后，可在这里回顾报告与运行日志。</p></div>', unsafe_allow_html=True)
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
            if b1.button("全选可见", key="hist_select_all",
                         use_container_width=True, disabled=not visible):
                for s in visible:
                    st.session_state[f"hist_chk_{s['session_id']}"] = True
                st.rerun()
            if b2.button("清除选择", key="hist_clear_sel",
                         use_container_width=True):
                for s in sessions:   # 含被筛选隐藏的，不留幽灵勾选
                    st.session_state[f"hist_chk_{s['session_id']}"] = False
                st.rerun()

            st.caption(f"共 {len(sessions)} 条 · 可见 {len(visible)} 条"
                       f"（勾选左侧方框即可，无需展开）")
            if any(s.get("state") == "RUNNING" for s in visible):
                st.caption("可见列表含 RUNNING 会话：多为异常中断的残留；"
                           "若该复现仍在运行，删除会丢失它的记录"
                           "（被占用的文件自动跳过）。")

            if not visible:
                st.info("没有符合筛选条件的会话。")
            else:
                for sess in visible:
                    sid = sess["session_id"]
                    title = display_text(sess.get("paper_title", "未知论文"))
                    state = sess.get("state", "未知")
                    state_label = {"COMPLETED": "已结束", "ERROR": "异常", "RUNNING": "运行中"}.get(state, state)
                    duration = sess.get("duration_sec", 0)
                    llm_calls = sess.get("llm_calls", 0)
                    log_entries = sess.get("log_entries", 0)
                    report_path = sess.get("report_path", "")

                    c_chk, c_body = st.columns([0.04, 0.96])
                    # 勾选框放在 expander 外：批量删除时不必逐条展开
                    c_chk.checkbox(f"选择 {sid}", key=f"hist_chk_{sid}",
                                   label_visibility="collapsed")
                    with c_body.expander(
                            f"{state_label} · {title[:40] or '无标题'}{'…' if len(title) > 40 else ''} · {sid}"):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("状态", state)
                        m2.metric("耗时", f"{duration:.1f}s")
                        m3.metric("LLM调用", llm_calls)
                        st.caption(f"日志条目: {log_entries}")

                        # 报告下载
                        historical_result = get_session_report_result(sess)
                        if historical_result:
                            historical_result = report_upgrade_controls(historical_result, key=f"upgrade_{sid}")
                            report_downloads(historical_result, key=f"dl_{sid}", st_module=st,
                                             data_root=get_project_data_dir())
                            if st.button("查看报告", key=f"view_report_{sid}", disabled=st.session_state.running):
                                st.session_state.result = historical_result
                                st.session_state.progress_file = None
                                st.session_state.current_state = historical_result.get("state", "COMPLETED")
                                st.session_state.logs = historical_result.get("audit_logs", [])
                                st.session_state["_hist_flash"] = "历史报告已载入，可在复现报告页查看。"
                                st.rerun()
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
                        if st.button("删除本会话", key=f"del_btn_{sid}",
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
                st.markdown("**批量删除所选会话**（危险操作，请谨慎）")
                selected = [
                    s for s in visible
                    if st.session_state.get(f"hist_chk_{s['session_id']}", False)
                ]
                st.caption(f"已选 {len(selected)} 条（仅统计上方可见列表；"
                           f"被筛选隐藏的已勾选会话不会被删除）")

                confirm_batch = st.checkbox(
                    "我确认批量删除以上所选会话（账本/日志/报告/progress 一并删除）",
                    key="confirm_batch_del")
                if st.button(f"删除所选 {len(selected)} 条",
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
    <span><i aria-hidden="true"></i> AutoReproducer <small>研究，从复现开始。</small></span><span>论文复现实验管理 · v0.2.0</span>
</div>
""", unsafe_allow_html=True)
