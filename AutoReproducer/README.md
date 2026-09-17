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
> docker CLI 自动探测：优先 `DOCKER_PATH` 环境变量 → 系统 PATH → Docker Desktop
> 常见安装目录（`C:\Program Files\Docker\Docker\resources\bin\docker.exe`），
> Windows 上即使 docker 不在 PATH 中也能正常构建/执行。

## 运行测试

```bash
# 完整测试套件（Mock 模式，无需 LLM API/Docker，CI 全绿）
python -m pytest tests/ -v

# 只看端到端集成用例
python -m pytest tests/ -v -k TestEndToEnd
```

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

**每个阶段输出都会经过 Prompt-Free 验证**（Verifier 复用该 Agent 的
`system_prompt` 作为质量标准）；验证未通过时按修正建议触发一次修正重试，
形成「生成 → 验证 → 修正 → 再验证」闭环（预算约束内，`MAX_FIX_RETRIES=1`）。

## 核心创新点

1. **复现-优化一体化闭环** — 复现成功自动触发优化，产出优化报告
2. **预算感知 UCB 调度** — 多臂老虎机算法在优化方向间智能分配预算（预算上限可配，默认 10 次）
3. **Prompt-Free 双层验证** — 复用各 Agent 系统提示词作为质量标准，无需额外验证提示词
4. **可审计完整实验追踪** — 所有步骤的输入输出 / 决策依据写入 `data/logs/` JSONL，
   实验账本（Ledger）写入 `data/experiment_ledger/`，支持 `replay()` 按时间轴回放
5. **语料对照层** — 对接 PaperGuru-Benchmark 23 篇真实论文（依赖 / 复现分）

## 关键机制

| 机制 | 说明 |
|---|---|
| 输入透传 | 支持论文标题 / 上传 PDF / 外部代码三种输入，正确进入数据上下文 |
| 5 轮依赖诊断 | EnvBuilder 循环「探测 → 分类 → 定点修复 → 重验证」，输出逐轮修复报告 |
| smoke + full 双阶段执行 | 先跑轻量 smoke 验证可运行性，再跑完整训练，避免浪费资源 |
| 指标口径统一 | 论文声明 0.85（小数）与运行输出 85.2%（百分数）自动归一化后再比对 |
| 预算统计 | 全程 LLM 调用次数累计，纳入审计统计与报告（方案预算上限 100 次） |
| 修正闭环 | 每步输出经 Prompt-Free 验证，失败按建议修正重试 1 次，全程留痕 |

## 两种模式

- **Mock 模式**（默认）— 无需 LLM API / Docker，直接演示完整流程
- **真实模式** — 调用任意 **OpenAI 兼容的远程 LLM API**（不依赖本地部署；需配置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`），可选 Docker 真实执行论文代码

> **安全提示**：真实模式的代码执行分两种沙箱——**本地子进程**（默认，便捷但
> 无沙箱隔离，运行前会做一道「危险代码静态门」拦截 `subprocess`/`os.system`/
> `eval`/网络外联/递归删除等明显危险调用）与 **Docker 容器**（`use_docker=True`，
> 隔离运行）。执行不可信 LLM 代码请用 Docker。

## LLM API 配置（真实模式）

通过环境变量注入，客户端不绑定具体厂商：

| 环境变量 | 说明 | 示例 |
| --- | --- | --- |
| `LLM_BASE_URL` | OpenAI 兼容端点（含 `/v1` 前缀或网关根地址均可） | `https://api.deepseek.com`、`https://qianfan.baidubce.com/v2`、`https://api.openai.com/v1` |
| `LLM_API_KEY` | API 访问密钥（无鉴权服务可留空） | `sk-xxxx` |
| `LLM_MODEL` | 模型名 | `deepseek-chat`、`ernie-4.0-8k`、`gpt-4o-mini` |
| `LLM_TIMEOUT` | 请求超时秒数（默认 120） | `120` |

```bash
# Linux / macOS
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="sk-xxxx"
export LLM_MODEL="deepseek-chat"

# Windows PowerShell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_API_KEY = "sk-xxxx"
$env:LLM_MODEL = "deepseek-chat"
```

> 未配置 `LLM_BASE_URL` / `LLM_MODEL` 时，真实模式会返回明确错误提示（不会静默回退到本地服务），便于快速定位环境问题。

## DeepSeek 快速接入

项目默认配置已指向 DeepSeek（`https://api.deepseek.com` + `deepseek-chat`），接入只需两步：

1. **获取 API Key**：登录 [platform.deepseek.com](https://platform.deepseek.com) → 「API Keys」→ 创建密钥（按量付费，新用户有赠送额度）。
2. **配置方式二选一**：

   - **网页侧边栏**：运行 `streamlit run app.py` 后，在左侧「LLM API 配置」里填入 API Key 即可（API 地址与模型名已默认填好 DeepSeek 值）；
   - **环境变量**（编程接口 / 命令行场景）：

     ```bash
     # Linux / macOS
     export LLM_BASE_URL="https://api.deepseek.com"
     export LLM_API_KEY="sk-你的key"
     export LLM_MODEL="deepseek-chat"

     # Windows PowerShell
     $env:LLM_BASE_URL = "https://api.deepseek.com"
     $env:LLM_API_KEY = "sk-你的key"
     $env:LLM_MODEL = "deepseek-chat"
     ```

3. **模型名说明**（`LLM_MODEL`）：

| 模型名 | 说明 |
| --- | --- |
| `deepseek-chat` | DeepSeek-V3，通用对话/代码生成，复现与优化默认推荐 |
| `deepseek-reasoner` | DeepSeek-R1，推理更强、响应更慢，适合复杂代码调试场景 |

## 编程接口（无需前端即可运行）

```python
from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient

llm = LLMClient(mock_mode=True)           # Mock 演示；真实模式 mock_mode=False
orch = Orchestrator(llm_client=llm, mock_mode=True, max_trials=6)
result = orch.run({
    "paper_title": "Attention Is All You Need",  # 或 "pdf_path": "paper.pdf"
    # "corpus_paper": "bbox",                    # 可选：语料对照层论文 id
})
print(result["state"])                     # COMPLETED
print(result["data"]["report"])            # Markdown 复现+优化报告
```

真实模式（远程 LLM API + Docker 沙箱执行代码）：

```python
# 前提：已设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量
llm = LLMClient(mock_mode=False)          # 参数缺省时自动读环境变量
orch = Orchestrator(llm_client=llm, mock_mode=False,
                    use_docker=True)      # True: 代码在 Docker 沙箱中执行
result = orch.run({"pdf_path": "paper.pdf"})
```

## 项目结构

```
AutoReproducer/
├── app.py                     # Streamlit 前端
├── requirements.txt           # 依赖清单
├── tests/
│   └── test_architecture.py   # 完整 pytest 套件（单测+端到端）
├── src/
│   ├── orchestrator.py        # 编排器核心（状态机 + 验证闭环 + 预算统计）
│   ├── base_agent.py          # Agent 基类（含 system_prompt、实验账本封装）
│   ├── corpus.py              # 语料对照层（PaperBench）
│   ├── agents/
│   │   ├── paper_reader.py    # 论文解析 Agent（PDF/标题输入透传）
│   │   ├── resource_finder.py # 资源查找 Agent
│   │   ├── env_builder.py     # 环境构建 Agent（5 轮依赖诊断 + 真实镜像构建）
│   │   ├── code_executor.py   # 代码执行 Agent（smoke+full 双阶段，本地/Docker）
│   │   ├── result_validator.py# 结果验证 Agent（指标提取 + 口径归一化）
│   │   ├── verifier.py        # 质量验证 Agent（Prompt-Free）
│   │   ├── optimizer.py       # 智能优化 Agent（UCB 预算调度）
│   │   └── report_generator.py# 报告生成 Agent（含审计/优化/验证记录）
│   ├── optimizer/
│   │   └── ucb_scheduler.py   # UCB 多臂老虎机调度器
│   ├── llm/
│   │   └── llm_client.py   # LLM API 客户端（OpenAI 兼容；Mock 任务精确分发）
│   └── audit/
│       └── audit_logger.py    # 审计日志 + 实验账本（replay 回放）
├── references/
│   └── paperbench/            # 语料对照层数据：PaperBench 23 篇论文复现提交物
└── data/                      # 日志 / 账本 / 报告（已 gitignore）
```

## 团队成员分工

- **A - 系统架构师（成员1）**: 整体架构、Orchestrator、Docker 沙箱、模块集成
- **B - Agent 开发工程师（成员2）**: 各 Agent 实现、LLM 集成、Prompt 工程
- **C - 优化与前端工程师（成员3）**: Optimizer、Verifier、Streamlit 前端

## 技术栈

- **前端**: Streamlit
- **核心**: Python 3.11+
- **LLM**: OpenAI 兼容 API（DeepSeek / 千帆 / OpenAI 等，环境变量配置）
- **沙箱**: Docker（可选，真实执行）
- **PDF**: PyPDF2 / pdfplumber
