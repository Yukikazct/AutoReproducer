# AutoReproducer 🧪🔬

基于多智能体协作的论文自动复现与优化系统

## 快速启动

```bash
# 1. 进入项目目录
cd AutoReproducer

# 2. 创建虚拟环境（如有需要）
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动前端
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 功能

**核心流程（六步流水线）：**
1. 📖 **PaperReader** — 解析论文 PDF，提取结构化信息
2. 🔍 **ResourceFinder** — 查找代码仓库和数据集
3. 🔧 **EnvBuilder** — 自动搭建运行环境（Docker/虚拟环境）
4. ⚡ **CodeExecutor** — 在沙箱中运行论文代码
5. ✅ **ResultValidator** — 比对论文声明值与运行结果
6. 📝 **ReportGenerator** — 生成 Markdown 复现报告

**两种模式：**
- **Mock 模式**（默认）— 无需 Ollama，直接演示完整流程
- **真实模式** — 连接本地 Ollama（如 `qwen2.5:7b`），使用真实 LLM

## 项目结构

```
AutoReproducer/
├── app.py                     # Streamlit 前端
├── requirements.txt           # 依赖清单
├── src/
│   ├── orchestrator.py        # 编排器核心（状态机）
│   ├── base_agent.py          # Agent 基类
│   ├── agents/
│   │   ├── paper_reader.py    # 论文解析 Agent
│   │   ├── resource_finder.py # 资源查找 Agent
│   │   ├── env_builder.py     # 环境构建 Agent
│   │   ├── code_executor.py   # 代码执行 Agent
│   │   ├── result_validator.py# 结果验证 Agent
│   │   └── report_generator.py# 报告生成 Agent
│   ├── llm/
│   │   └── ollama_client.py   # Ollama API 客户端
│   └── audit/
│       └── audit_logger.py    # 审计日志记录器
└── data/                      # 日志、报告等
```

## 团队成员分工

- **A - 系统架构师（成员1）**: 整体架构、Orchestrator、Docker 沙箱、模块集成
- **B - Agent 开发工程师（成员2）**: 各 Agent 实现、LLM 集成、Prompt 工程
- **C - 优化与前端工程师（成员3）**: Optimizer、Verifier、Streamlit 前端

## 技术栈

- **前端**: Streamlit
- **核心**: Python 3.12
- **LLM**: Ollama（支持 qwen2.5 / llama3 等）
- **沙箱**: Docker
- **PDF**: PyPDF2 / pdfplumber
