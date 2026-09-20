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
| 用量计量（P1-⑫） | plan 级 LLM token / 容器耗时统计：按流水线阶段（READ_PAPER…OPTIMIZING…）分账，审计统计含 llm_calls / tokens / seconds / 容器维度 |
| 确定性验证（P0-②） | 证据链模块：无真实执行证据不判 verified、smoke 上限；实验账本 replay 时间轴回放 |
| 冻结 Spec + Holdout（P1-⑦） | 优化前冻结验收标准（sha256 指纹），只对最终 best 状态做隐藏留出多轮评估，防过拟合验收 |

## 存储管理（三层缓存，存储友好设计）

复现多篇论文 = 代码 + 数据集 + 预训练权重，多篇累加后普通笔记本磁盘（512GB-1TB）很快耗尽。数据集体积远超代码：ImageNet 约 150GB，而 CIFAR-10 仅 170MB。本项目按 **按需懒加载 + 三层存储 + 体积瘦身** 落地：

| 层 | 载体 | 内容 | 生命周期 |
|---|---|---|---|
| **L0 热缓存** | 本地磁盘 `data/` | 当前任务的代码 / 数据子集 / 权重 | 任务完成并归档后清理 |
| **L1 温存储** | 移动硬盘 / NAS / 网盘 | 已复现论文完整快照（zip 归档） | `resource_cli` archive / restore |
| **L2 冷存储** | HuggingFace / ModelScope | 只存清单 + 按 ID 可重现下载 | 不落地 |

### 按需懒加载（ResourceManager，`src/resource_manager.py`）

- `fetch_code` / `fetch_dataset` / `fetch_weights`：只拉当前任务最小集（代码仓库 depth 1 克隆、数据集冒烟子集、权重本地复制或按子路径下载），重复 fetch 幂等复用缓存，绝不重复下载；
- `manifest`：每篇论文的资源清单写入 `data/manifests/<paper_id>.json`（兼容 2026-09-09 存量格式），`cleaned_at` 记录清理时间；
- **L0 配额守护**：`AUTOREPRO_L0_QUOTA_GB` 可配（默认 20GB），超限时 `enforce_quota` 按最近使用（LRU）返回建议归档清单，不自动删除；
- 所有真实网络下载均"尽力而为"：网络不可用时诚实降级为本地合成冒烟集（`dataset_smoke`），在 `state` 字段标注实际状态，绝不静默伪造大文件。

### 隔离依赖安装（P0-3）

真实模式下 `pip install --target data/deps/<sha1(reqs)>` 一次性安装到隔离目录并落 `.ready` 就绪标记，执行时经 **PYTHONPATH 注入** 该目录；同名依赖清单跨论文跨会话只落一份（多论文共享、天然去重），不污染全局 Python。`AUTOREPRO_DEPS_ROOT` 可覆盖根目录。

### 镜像级共享（P1-1，`src/agents/env_builder.py`）

- 底座镜像 `autorepro-base:latest`（`python:3.11-slim` + CPU torch/torchvision/numpy/tqdm + 国内源注入），**一次构建、多论文增量复用**——参考 SWE 领域 SWE-smith 实践（128 仓库共用统一底座镜像，存储/构建时间大幅下降）；
- 论文 Dockerfile `FROM python:*` 自动替换为底座（`_swap_to_base_image`）；底座缺失时按需构建，构建失败自动降级原 Dockerfile 并带 `degraded` 标注，不阻断流水线；
- `AUTOREPRO_PIP_INDEX` 可覆盖 pip 下载源（默认清华镜像，`PIP_FIND_LINKS` 指向阿里云 CPU wheels）。

### 数据集注册表与复现级别数据策略（P1-2，`src/dataset_registry.py`）

15 类常见数据集（MNIST / Fashion-MNIST / CIFAR-10/100 / SVHN / STL-10 / Tiny ImageNet / ImageNet-1k / COCO 2017 / SQuAD v1/v2 / IMDb / AG News / GLUE / WikiText-103 + 零数据合成标记），按 `kind` 决定子集策略：

| kind | 策略 | 示例 |
|---|---|---|
| `torchvision` | 内建数据集，训练代码运行时懒加载，**不预下载** | CIFAR-10（170MB） |
| `full` | 体积可控，直接全量 | SQuAD / AG News（≤120MB） |
| `percent:N` | 超大数据降采样 N% 验证流程趋势（语义为流程验证而非数值复现） | ImageNet（150GB→1%）、COCO（25GB→5%） |
| `synthetic` | 零数据，训练代码运行时合成，体积为 0 | PINN / GAN 类合成数据 |

未知数据集返回 `None` → 自动降级合成冒烟集。注册表同时提供体积预估（`size_gb`）与国内镜像备注，供 fetch 前的选型与配额决策。

### 缓存管理 CLI（P2，`scripts/resource_cli.py`）

| 子命令 | 说明 |
|---|---|
| `status` | 查看 L0 配额占用与按论文统计 |
| `list` / `manifest <pid>` | 列出已登记论文 / 查看单篇资源清单 |
| `archive <pid> [--dest DIR]` | 归档 L0 资源到 L1 zip（默认 `data/archive/`） |
| `restore <zip>` | 从 L1 归档恢复回 L0 |
| `prune [--yes]` | 配额超限建议；`--yes` 先归档 L1 再清理 L0（不丢数据） |
| `quota-check --size-gb N` | 拉取前配额预检：不足则拒绝（退出码 2）并给归档建议 |

```bash
python scripts/resource_cli.py status
python scripts/resource_cli.py archive 2026abcd1234 --dest D:/nas/archive
python scripts/resource_cli.py restore D:/nas/archive/2026abcd1234.zip
python scripts/resource_cli.py prune --yes
python scripts/resource_cli.py quota-check --size-gb 1.5   # 下载前预检
```

### 存储相关环境变量

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `AUTOREPRO_DATA_ROOT` | 缓存根（repos/datasets/manifests/archive/deps） | `<repo>/data` |
| `AUTOREPRO_L0_QUOTA_GB` | L0 热缓存配额 | `20` |
| `AUTOREPRO_DEPS_ROOT` | 隔离依赖安装根 | `data/deps` |
| `AUTOREPRO_PIP_INDEX` | pip 下载源（镜像级共享与注入共用） | 清华镜像 |

## 工程可信度增强（ScholarAgent 融合）

面向「复现结论可信」的工程防线（P0 移植 + P1 增强，参照 ScholarAgent 的
证据链 / 经验库 / 防泄漏理念落地）：

| 机制 | 位置 | 说明 |
|---|---|---|
| 补丁安全策略（P0-①） | `src/safety/patch_policy.py` | 优化补丁禁改数据/权重/评测文件、禁越目录、单文件体积预算，结构化拒绝决策 |
| 快照指纹（P0-④） | `src/safety/workspace_snapshot.py` | 全工作区 SHA-256 指纹 + 运行中篡改检测（不一致即中止恢复） |
| 经验库（P0-③） | `src/experience/experience_store.py` | JSONL 持久化 `validated` 标记、上限截断、summarize/best，供 BeamUCT 播种先验 |
| TrialLedger（P0-⑤） | `src/experience/trial_ledger.py` | Keep/Reject 结构化账本（candidate/评估/理由/restored），供报告与经验库消费 |
| 证据链（P0-②） | `src/evidence/` | 证据注册表（SHA-256 + authentic）、裁决门控（无真实执行证据不判 verified）、Claim/Criterion/Evidence 图 |
| BeamUCT 双层搜索（P1-⑥） | `src/optimizer/beam_uct.py` | 方向级 UCB + 参数树 UCT + Beam top-k，经验库摘要播种先验，方向退役 |
| 冻结 Spec + Holdout（P1-⑦） | `src/optimizer/` | ResearchSpec sha256 冻结验收，只对最终 best 状态隐藏留出多轮评估 |
| 静态依赖解析 + pip 自愈（P1-⑧） | `src/agents/dependency_resolver.py` | AST+requirements 双源解析、stdlib 过滤、运行时缺模块 pip 自愈（≤3 轮） |
| 确定性仓库发现链（P1-⑨） | `src/agents/repo_discovery.py` | 用户URL→PwC→GitHub搜索→curated 四级降级链 + revision pin + 溯源标记 |
| 防泄漏 Benchmark（P1-⑩） | `src/benchmark/leakage_safe.py` | 确定性 hash 切分、隐藏标签私有目录、指标契约冻结后端复算（纯 Python 零依赖） |
| 沙箱加固（P1-⑪） | `src/agents/code_executor.py` | 镜像白名单、cap-drop ALL、no-new-privileges、只读 rootfs+tmpfs、非 root、CPU/mem/pids 限额（随可用性降级） |
| 用量计量（P1-⑫） | `src/audit/audit_logger.py` | plan 级 LLM token 统计（response_metadata 提取）与容器耗时维度，`get_stats()` 汇总 |

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
│   ├── test_architecture.py   # 完整 pytest 套件（单测+端到端）
│   ├── test_env_builder_base.py  # 共享底座镜像（P1-1）
│   ├── test_dataset_registry.py  # 数据集注册表与数据策略（P1-2）
│   ├── test_resource_cli.py      # 三层缓存 CLI（P2）
│   ├── test_orchestrator_storage.py  # 编排器存储钩子（P0-2）
│   ├── test_patch_policy.py       # 补丁安全策略（P0-①）
│   ├── test_workspace_snapshot.py # 快照指纹（P0-④）
│   ├── test_experience_store.py   # 经验库（P0-③）
│   ├── test_trial_ledger.py       # TrialLedger 账本（P0-⑤）
│   ├── test_evidence_graph.py     # 证据链（P0-②）
│   ├── test_beam_uct_fusion.py    # BeamUCT 双层搜索（P1-⑥）
│   ├── test_frozen_spec.py        # 冻结 Spec + Holdout（P1-⑦）
│   ├── test_dependency_resolver.py# 静态依赖解析（P1-⑧）
│   ├── test_repo_discovery.py     # 确定性仓库发现链（P1-⑨）
│   ├── test_leakage_safe_benchmark.py # 防泄漏 Benchmark（P1-⑩）
│   ├── test_sandbox_hardening.py  # 沙箱加固（P1-⑪）
│   ├── test_usage_metering.py     # 用量计量（P1-⑫）
│   └── test_reproduction_core.py  # 复现核心链路
├── scripts/
│   ├── resource_cli.py        # 三层缓存管理 CLI（archive/restore/prune/status…）
│   ├── generate_sample_paper.py
│   └── run_optimization_demo.py
├── src/
│   ├── orchestrator.py        # 编排器核心（状态机 + 验证闭环 + 预算统计 + 存储钩子）
│   ├── resource_manager.py    # 资源懒加载 + L0 缓存/配额/归档（P0-1）
│   ├── dataset_registry.py    # 数据集注册表（别名/体积/子集策略/镜像，P1-2）
│   ├── base_agent.py          # Agent 基类（含 system_prompt、实验账本封装）
│   ├── corpus.py              # 语料对照层（PaperBench）
│   ├── safety/
│   │   ├── patch_policy.py    # 补丁安全策略（P0-①）
│   │   └── workspace_snapshot.py # 快照指纹（P0-④）
│   ├── experience/
│   │   ├── experience_store.py # 经验库（P0-③）
│   │   └── trial_ledger.py     # TrialLedger 账本（P0-⑤）
│   ├── evidence/               # 证据链（P0-②）
│   ├── benchmark/              # 防泄漏 Benchmark（P1-⑩）
│   ├── agents/
│   │   ├── paper_reader.py    # 论文解析 Agent（PDF/标题输入透传）
│   │   ├── resource_finder.py # 资源查找 Agent（确定性发现链 P1-⑨）
│   │   ├── env_builder.py     # 环境构建 Agent（5 轮依赖诊断 + 底座镜像构建）
│   │   ├── dependency_resolver.py # 静态依赖解析 + pip 自愈（P1-⑧）
│   │   ├── repo_discovery.py  # 确定性仓库发现链（P1-⑨）
│   │   ├── code_executor.py   # 代码执行 Agent（smoke+full+沙箱加固 P1-⑪）
│   │   ├── result_validator.py# 结果验证 Agent（指标提取 + 口径归一化）
│   │   ├── verifier.py        # 质量验证 Agent（Prompt-Free）
│   │   ├── optimizer.py       # 智能优化 Agent（UCB 预算调度 + 冻结 Spec P1-⑦）
│   │   └── report_generator.py# 报告生成 Agent（含审计/优化/验证记录）
│   ├── optimizer/
│   │   ├── ucb_scheduler.py   # UCB 多臂老虎机调度器
│   │   └── beam_uct.py        # BeamUCT 双层搜索 + 先验播种（P1-⑥）
│   ├── llm/
│   │   └── llm_client.py   # LLM API 客户端（OpenAI 兼容；Mock 任务精确分发）
│   └── audit/
│       └── audit_logger.py    # 审计日志 + 实验账本 + plan 级用量计量（P1-⑫）
├── references/
│   └── paperbench/            # 语料对照层数据：PaperBench 23 篇论文复现提交物
└── data/                      # L0 热缓存：repos/datasets/manifests/archive/deps
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
