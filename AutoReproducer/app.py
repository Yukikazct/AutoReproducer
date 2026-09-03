"""AutoReproducer - Streamlit 前端界面"""
import streamlit as st
import json
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator
from src.llm.ollama_client import LLMClient
from src.audit.audit_logger import AuditLogger

# 页面配置
st.set_page_config(
    page_title="AutoReproducer - 论文自动复现系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
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

# 初始化Session状态
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
    st.session_state.paper_title = "AutoReproducer: 基于多智能体协作的论文自动复现与优化系统"
# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/idea.png", width=60)
    st.markdown("## ⚙️ 控制面板")

    # 模式选择
    st.session_state.mock_mode = st.toggle("🧪 Mock模式（无需Ollama）", value=st.session_state.mock_mode,
                          help="启用Mock模式可直接演示，无需连接本地Ollama")

    # Ollama配置
    with st.expander("🔗 Ollama 配置", expanded=not st.session_state.mock_mode):
        model_name = st.text_input("模型名称", value="qwen2.5:7b",
                                   disabled=st.session_state.mock_mode)
        base_url = st.text_input("API地址", value="http://localhost:11434",
                                 disabled=st.session_state.mock_mode)

    # 论文输入
    st.markdown("### 📄 论文输入")
    input_mode = st.radio("输入方式", ["论文标题", "上传PDF"], key="input_mode")

    paper_title = st.session_state.paper_title
    uploaded_file = None

    if input_mode == "论文标题":
        st.session_state.paper_title = st.text_input(
            "论文标题",
            value=st.session_state.paper_title,
            placeholder="输入论文标题...",
            key="paper_title_input"
        )
        paper_title = st.session_state.paper_title
    else:
        uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"], key="pdf_uploader")

    # 启动按钮
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 开始复现", type="primary", use_container_width=True,
                             disabled=st.session_state.running)
    with col2:
        reset_btn = st.button("🔄 重置", use_container_width=True)

    # 系统状态
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    state_colors = {
        "INIT": "⚪", "READ_PAPER": "📖", "FIND_RESOURCES": "🔍",
        "BUILD_ENV": "🔧", "EXECUTE_CODE": "⚡", "VALIDATE": "✅",
        "GENERATE_REPORT": "📝", "COMPLETED": "🎉", "ERROR": "❌"
    }
    st.markdown(f"**当前状态**: {state_colors.get(st.session_state.current_state, '⚪')} "
                f"`{st.session_state.current_state}`")


# ========== 主界面 ==========
st.markdown('<p class="main-title">🔬 AutoReproducer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于多智能体协作的论文自动复现与优化系统</p>',
            unsafe_allow_html=True)

# 标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 流水线状态", "📄 复现报告", "📜 审计日志", "🔍 状态机"
])
# ===== Tab 1: 流水线状态 =====
with tab1:
    st.markdown("### 🏗️ 六步复现流水线")

    # 定义Agent信息
    AGENTS = [
        ("📖 PaperReader", "论文解析", "从PDF中提取结构化信息"),
        ("🔍 ResourceFinder", "资源查找", "定位代码仓库和数据集"),
        ("🔧 EnvBuilder", "环境构建", "自动搭建运行环境"),
        ("⚡ CodeExecutor", "代码执行", "在沙箱中运行论文代码"),
        ("✅ ResultValidator", "结果验证", "比对论文声明值与运行结果"),
        ("📝 ReportGenerator", "报告生成", "生成Markdown复现报告"),
    ]

    # 显示Agent卡片
    cols = st.columns(3)
    for i, (name, title, desc) in enumerate(AGENTS):
        with cols[i % 3]:
            status = st.session_state.agent_status.get(name, "waiting")
            status_icons = {
                "success": "✅",
                "error": "❌",
                "running": "🔄",
                "waiting": "⏳"
            }
            status_colors = {
                "success": "border-left: 4px solid #4CAF50;",
                "error": "border-left: 4px solid #f44336;",
                "running": "border-left: 4px solid #FF9800;",
                "waiting": "border-left: 4px solid #9E9E9E;"
            }
            icon = status_icons.get(status, "⏳")
            border = status_colors.get(status, "")
            st.markdown(f"""
            <div class="agent-card" style="{border}">
                <h4>{icon} {name}</h4>
                <small>{title}</small><br>
                <span style="color: #888;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    # 进度条
    agent_order = ["📖 PaperReader", "🔍 ResourceFinder", "🔧 EnvBuilder",
                   "⚡ CodeExecutor", "✅ ResultValidator", "📝 ReportGenerator"]
    completed = sum(1 for a in agent_order
                    if st.session_state.agent_status.get(a) == "success")
    progress = completed / len(agent_order) if len(agent_order) > 0 else 0
    st.progress(progress, text=f"整体进度: {completed}/{len(agent_order)}")

    # 运行结果展示
    if st.session_state.result:
        result = st.session_state.result
        st.markdown("---")
        st.markdown("### 📊 运行摘要")
        stats = result.get("audit_stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总步骤数", stats.get("total_steps", 0))
        c2.metric("成功", stats.get("success", 0))
        c3.metric("错误", stats.get("errors", 0))
        c4.metric("耗时(秒)", f"{stats.get('duration_sec', 0):.1f}")

        if result.get("state") == "COMPLETED":
            st.success("🎉 复现流程成功完成！")
        elif result.get("state") == "ERROR":
            st.error(f"❌ 流程出错: {result.get('error', '未知错误')}")

# ===== Tab 2: 复现报告 =====
with tab2:
    st.markdown("### 📄 复现报告")
    if st.session_state.result and st.session_state.result.get("data", {}).get("report"):
        report = st.session_state.result["data"]["report"]
        st.markdown(report)
    else:
        st.info("运行复现流程后，这里将显示完整的复现报告。")


# ===== Tab 3: 审计日志 =====
with tab3:
    st.markdown("### 📜 审计日志")
    if st.session_state.logs:
        col1, col2 = st.columns(2)
        with col1:
            filter_agent = st.selectbox(
                "按Agent筛选",
                ["全部"] + list(set(l.get("agent", "") for l in st.session_state.logs)),
                key="filter_agent_tab3"
            )
        with col2:
            filter_status = st.selectbox(
                "按状态筛选",
                ["全部", "SUCCESS", "ERROR", "START", "RUNNING", "WARNING"],
                key="filter_status_tab3"
            )
        filtered_logs = st.session_state.logs
        if filter_agent != "全部":
            filtered_logs = [l for l in filtered_logs if l.get("agent") == filter_agent]
        if filter_status != "全部":
            filtered_logs = [l for l in filtered_logs if l.get("status") == filter_status]

        for log in filtered_logs:
            status_color = {"SUCCESS": "🟢", "ERROR": "🔴", "START": "🟡",
                            "RUNNING": "🔄", "WARNING": "🟠"}.get(log.get("status", ""), "⚪")
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
    st.markdown("系统使用有限状态机（FSM）管理Agent的流转。")

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
        {"状态": "READ_PAPER", "说明": "解析论文PDF，提取结构化信息", "Agent": "PaperReader"},
        {"状态": "FIND_RESOURCES", "说明": "查找代码仓库和数据集", "Agent": "ResourceFinder"},
        {"状态": "BUILD_ENV", "说明": "构建Docker/虚拟环境", "Agent": "EnvBuilder"},
        {"状态": "EXECUTE_CODE", "说明": "在沙箱中执行代码", "Agent": "CodeExecutor"},
        {"状态": "VALIDATE", "说明": "验证结果与论文一致性", "Agent": "ResultValidator"},
        {"状态": "GENERATE_REPORT", "说明": "生成Markdown复现报告", "Agent": "ReportGenerator"},
        {"状态": "COMPLETED", "说明": "流水线完成", "Agent": "—"},
        {"状态": "ERROR", "说明": "出错状态，可重试", "Agent": "—"},
    ]
    st.table(state_data)

# ========== 事件处理 ==========
def run_pipeline(paper_title="", uploaded_file=None):
    """运行完整的复现流水线"""
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.agent_status = {}

    mock_mode = st.session_state.mock_mode

    logger = AuditLogger()
    llm = LLMClient(
        mock_mode=mock_mode,
        model="mock" if mock_mode else "qwen2.5:7b",
        base_url="http://localhost:11434"
    )
    orchestrator = Orchestrator(llm_client=llm, mock_mode=mock_mode, logger=logger)
    st.session_state.orchestrator = orchestrator

    input_data = {"paper_title": paper_title or "AutoReproducer项目方案"}

    agents_info = [
        ("READ_PAPER", "📖 PaperReader", "reader"),
        ("FIND_RESOURCES", "🔍 ResourceFinder", "finder"),
        ("BUILD_ENV", "🔧 EnvBuilder", "builder"),
        ("EXECUTE_CODE", "⚡ CodeExecutor", "executor"),
        ("VALIDATE", "✅ ResultValidator", "validator"),
        ("GENERATE_REPORT", "📝 ReportGenerator", "reporter"),
    ]

    data = {}
    for state_name, display_name, agent_key in agents_info:
        st.session_state.current_state = state_name
        st.session_state.agent_status[display_name] = "running"
        yield

        agent = orchestrator.agents[agent_key]
        try:
            result = agent.run(data)

            if state_name == "READ_PAPER":
                data["paper_info"] = result.get("paper_info", {})
                data["raw_text"] = result.get("raw_text", "")
            elif state_name == "FIND_RESOURCES":
                data["resources"] = result.get("resources", {})
            elif state_name == "BUILD_ENV":
                data["env_config"] = result.get("env_config", {})
            elif state_name == "EXECUTE_CODE":
                data["execution"] = result
            elif state_name == "VALIDATE":
                data["validation"] = result
            elif state_name == "GENERATE_REPORT":
                data["report"] = result.get("report", "")

            st.session_state.agent_status[display_name] = "success"
            st.session_state.logs = logger.get_summary()
            yield

        except Exception as e:
            st.session_state.agent_status[display_name] = "error"
            st.session_state.current_state = "ERROR"
            st.session_state.logs = logger.get_summary()
            yield
            break

    if st.session_state.current_state != "ERROR":
        st.session_state.current_state = "COMPLETED"
        data["audit_summary"] = logger.get_stats()

    st.session_state.result = {
        "state": st.session_state.current_state,
        "error": None,
        "data": data,
        "audit_logs": logger.get_summary(),
        "audit_stats": logger.get_stats()
    }
    st.session_state.logs = logger.get_summary()
    st.session_state.running = False
    yield


# 启动按钮处理
if start_btn:
    pt = st.session_state.paper_title or ""
    if not pt and not uploaded_file:
        st.error("请先输入论文标题或上传PDF文件")
    else:
        with st.spinner("正在执行复现流程..."):
            for _ in run_pipeline(paper_title=pt, uploaded_file=uploaded_file):
                time.sleep(0.3)
        st.rerun()

# 重置按钮处理
if reset_btn:
    st.session_state.orchestrator = None
    st.session_state.result = None
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.current_state = "INIT"
    st.session_state.agent_status = {}
    st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    AutoReproducer v0.1.0 | 基于多智能体协作的论文自动复现与优化系统
</div>
""", unsafe_allow_html=True)
