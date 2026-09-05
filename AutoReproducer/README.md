# AutoReproducer 🧪🔬

基于多智能体协作的论文自动复现与优化系统

## 快速启动

```bash
# 1. 进入项目目录
cd AutoReproducer

# 2. 创建虚拟环境（如有需要）
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动前端
streamlit run app.py
```

浏览器访问 http://localhost:8501

> 依赖 Python 3.11+。Docker 真实执行需本地已安装并启动 Docker Desktop（可选，仅真实模式需要）。

## 核心流程（复现 → 验证 → 优化闭环）

**1 个编排器 + 8 个专职 Agent：**

| Agent | 职责 |
|---|---|
| 📖 PaperReader | 解析论文 PDF，提取结构化信息 |
| 🔍 ResourceFinder | 查找代码仓库和数据集 |
| 🔧 EnvBuilder | 生成运行环境配置 / 真实构建 Docker 镜像 |
| ⚡ CodeExecutor | 在本地或 Docker 沙箱中运行代码 |
| ✅ ResultValidator | 比对论文声明值与运行结果 |
| 🛡️ Verifier | Prompt-Free 质量验证（复用各 Agent 系统提示词） |
| 🧪 Optimizer | UCB 预算调度下的智能优化（Keep/Reject） |
| 📝 ReportGenerator | 生成 Markdown 复现 + 优化报告 |

**状态机流转：**

```
INIT → READ_PAPER → FIND_RESOURCES → BUILD_ENV → EXECUTE_CODE → VALIDATE
  → (复现成功) OPTIMIZING → OPTIMIZED → GENERATE_REPORT → COMPLETED
  → (复现失败) GENERATE_REPORT → COMPLETED
```

## 核心创新点

1. **复现-优化一体化闭环** — 复现成功自动触发优化，产出优化报告
2. **预算感知 UCB 调度** — 多臂老虎机算法在优化方向间智能分配预算
3. **Prompt-Free 双层验证** — 复用各 Agent 系统提示词作为质量标准，无需额外验证提示词
4. **语料对照层** — 对接 PaperGuru-Benchmark 23 篇真实论文（依赖 / 复现分）

## 两种模式

- **Mock 模式**（默认）— 无需 Ollama / Docker，直接演示完整流程
- **真实模式** — 连接本地 Ollama（如 `qwen2.5:7b`），可选 Docker 真实执行论文代码

## 项目结构

```
AutoReproducer/
├── app.py                     # Streamlit 前端
├── requirements.txt           # 依赖清单
├── src/
│   ├── orchestrator.py        # 编排器核心（状态机）
│   ├── base_agent.py          # Agent 基类（含 system_prompt）
│   ├── corpus.py              # 语料对照层（PaperBench）
│   ├── agents/
│   │   ├── paper_reader.py    # 论文解析 Agent
│   │   ├── resource_finder.py # 资源查找 Agent
│   │   ├── env_builder.py     # 环境构建 Agent（含真实镜像构建）
│   │   ├── code_executor.py   # 代码执行 Agent（本地/Docker 双轨）
│   │   ├── result_validator.py# 结果验证 Agent
│   │   ├── verifier.py        # 质量验证 Agent（Prompt-Free）
│   │   ├── optimizer.py       # 智能优化 Agent（UCB）
│   │   └── report_generator.py# 报告生成 Agent
│   ├── optimizer/
│   │   └── ucb_scheduler.py   # UCB 多臂老虎机调度器
│   ├── llm/
│   │   └── ollama_client.py   # Ollama API 客户端
│   └── audit/
│       └── audit_logger.py    # 审计日志记录器
└── data/                      # 日志、报告等（已 gitignore）
```

## 团队成员分工

- **A - 系统架构师（成员1）**: 整体架构、Orchestrator、Docker 沙箱、模块集成
- **B - Agent 开发工程师（成员2）**: 各 Agent 实现、LLM 集成、Prompt 工程
- **C - 优化与前端工程师（成员3）**: Optimizer、Verifier、Streamlit 前端

## 技术栈

- **前端**: Streamlit
- **核心**: Python 3.11+
- **LLM**: Ollama（支持 qwen2.5 / llama3 等）
- **沙箱**: Docker（可选，真实执行）
- **PDF**: PyPDF2 / pdfplumber
