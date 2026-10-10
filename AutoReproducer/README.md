# AutoReproducer 🧪🔬

基于多智能体协作的论文自动复现与优化系统

## 当前可运行实验（2026-10-10）

网页“官方仓库预设”和命令行共用真实执行流程。当前支持以下固定案例，运行前关闭 Mock 与 Docker：

选择预设后可直接运行。系统自动取得固定版本源码和真实数据、检查 Python 与隔离依赖，缺失或损坏时按冻结版本准备和修复；环境检查通过后才开始训练。Windows Store 启动环境会自动切换到健康的标准 CPython 环境。自动恢复保留作者算法、训练参数、真实数据和验收门槛，详见[运行环境与自动准备](docs/cross_platform_runtime.md)。

[10 月 10 日自动恢复验收](docs/automatic_preset_recovery_20261010.md)已完成 DLinear、SIREN 和 Neural ODE 的真实训练；包含缺失依赖、并发排队及前次准备失败后的恢复记录。

| 预设 | 实验与设备 | 已有证据 |
|---|---|---|
| `dlinear_etth1_reference` | ETTh1 多变量 336→96，CPU | 选定论文实验数值复现通过，详见下节 |
| `siren_camera_quick` | cameraman 256×256、500 步，NVIDIA CUDA 12.1 | Windows RTX 4060 上完成全图拟合、独立评价和真实 API 建议；[10 月 8 日快速档记录](docs/siren_quick_result.md) |
| `neural_ode_spiral` | 官方二维螺旋示例、2000 次迭代，CPU | 完成轨迹拟合、独立 SciPy 评价和真实 API 建议；[10 月 8 日方法实验记录](docs/neural_ode_result.md) |

SIREN 与 Neural ODE 的结果为 `method_experiment_completed`、`is_reproduced=null`：完成作者方法实验，不代表论文全部基准或表格数值复现。SIREN 五分钟目标从已准备环境开始计时，首次下载和安装不计入；换设备或网络后需重新实测。

这两个方法预设支持三种优化模式：

| 模式 | 行为 | 结论范围 |
|---|---|---|
| `off`（默认） | 完整训练与独立评价 | 无需建议 API；保留真实基线与图表 |
| `suggest` | 基线后调用真实 API，生成最多 3 条有作者来源的单参数建议 | 建议尚未训练验证，`optimized=False` |
| `validate` | 独立优化协议重训基线，按验证集选择候选，再做两种子留出确认 | 只有两种子均达到事前门槛才标记 `validated_gain`；最多 3 个候选，总预算不超过 7200 秒 |

`validate` 的总预算包含在线分析、官方基线和参数验证，当前调度规则固定预留 40 分钟给最终两种子确认。
因此总预算必须大于 40 分钟，建议使用默认 120 分钟；预算只是上限，完成后会提前结束。
不足的配置会在调用 API 或训练前提示。运行中若预算不足，已完成基线仍保留，优化结果明确标为未完成。

优化扩展中的 SIREN 使用固定 80%/10%/10% 训练、验证和留出像素；Neural ODE 使用新增初值轨迹验证。选优只看验证集，冻结候选后才打开留出评估，不把这些指标混称为官方全图或全轨迹拟合结果。DLinear 保留作者协议，不开放参数优化。已有 [10 月 8 日检查点](docs/implementation_checkpoint_20261008.md)保留 SIREN 优化测量和 Neural ODE 中断记录；[10 月 9 日续跑](docs/resumed_validation_20261009.md)已完成 SIREN 四阶段在线分析及 Neural ODE 两种子留出确认；后者第二种子未达提升门槛，结论为 `tested_no_gain`、`optimized=False`。

从 `AutoReproducer/` 目录直接运行；也可提前准备环境，便于重复实验：

```powershell
# 可选预热：准备源码、数据、隔离依赖和设备；此阶段不训练
python scripts/reproduce_repository.py --profile siren_camera_quick --prepare-environment
python scripts/reproduce_repository.py --profile neural_ode_spiral --prepare-environment

# 真实基线，无需 API
python scripts/reproduce_repository.py --profile siren_camera_quick --optimization off

# 真实基线与建议：从 LLM_API_KEY 读取密钥，交互终端未配置时隐藏输入
python scripts/reproduce_repository.py --profile siren_camera_quick --optimization suggest
python scripts/reproduce_repository.py --profile neural_ode_spiral --optimization suggest

# 独立候选训练与两种子留出确认；每次命令创建新实验
python scripts/reproduce_repository.py --profile neural_ode_spiral --optimization validate --max-candidates 3 --budget-seconds 7200

# 可选：在训练前增加四阶段真实 API 公开来源分析
python scripts/reproduce_repository.py --profile siren_camera_quick --llm-review --optimization suggest
```

`--prepare-only` 只准备源码、数据与执行计划，不安装环境；`--prepare-environment` 还会安装依赖并检查设备。正式运行会自动执行这些准备和健康检查，可直接选择网页“运行实验”；“准备实验环境”用于提前预热。运行结果和原始建议记录保存在本机 `data/runs/<运行编号>/`，不随 Git 分发。

明确指定 `--offline` 时，源码、数据、公开证据和冻结训练依赖只使用可核验本地缓存，缺失或损坏即停止，禁止在线下载安装。运行环境恢复也不会下载新解释器，只使用已有解释器和本地应用 wheel。明确启用建议或在线分析时仍会访问 LLM API。

运行中优化状态为 `running`；正常取消保存 `interrupted` 记录及已有实验结果。Windows 的训练进程受 Job 管理，关闭启动会话（含 `py` 启动器）或强杀主 Python 后会清理训练后代。强杀后，下次方法实验自动识别失去运行锁的未完成记录，也可运行 `python scripts/recover_interrupted_optimizations.py`，为结果与报告补记中断；这不会续训或改写已完成指标。POSIX 的正常 SIGTERM 会清理进程组；SIGKILL 后孤儿训练可能持续至超时，不能将记录恢复视为立即终止所有后代。

Windows 支持直接运行 Python 和标准 CPython 虚拟环境。检测到 Microsoft Store Python、解释器版本不兼容或应用依赖无法导入时，系统会自动发现并验证已有标准环境，或创建项目专用运行环境。64 位 Windows 未发现兼容解释器时，会下载 python.org 官方安装程序、验证数字签名后安装到项目缓存。原启动环境及全局依赖不被改写；后续运行自动健康检查并复用准备结果。

## DLinear 官方仓库复现（2026-10-07：真实 4 + 1 多 Agent 流程通过）

网页默认真实模式。“输入方式 → 官方仓库预设”默认执行 DLinear 的 ETTh1 多变量 **336→96 完整作者实验**：固定源码和真实数据，按最多 10 轮、patience 3 的官方协议训练、加载验证集最优 checkpoint，再测试和独立复算指标。保持 Mock 与 Docker 关闭，使用本地 CPU。默认勾选“使用真实多 Agent 分析论文、仓库和环境”；自行勾选“允许 API 分析本次指标、轮数和核验状态摘要”后，训练后再执行结果解释。取消分析选项时，作者训练本身无需 LLM API。

新运行 `repository_e9408141e3dd44319e854bbdf4a70c0c` 通过生产网页后台入口完成：PaperReader、ResourceFinder、EnvBuilder、Verifier 读取 **16 个真实公开来源**，四阶段各 **1** 次调用且一次通过；训练后 ResultValidator 对已允许的数值摘要调用 **1** 次。真实 `deepseek-chat` API 合计 **5 次、零修正重试**，客户端与审计计数一致。报告分别标注论文原文、作者脚本和代码设置/默认值，作者依赖声明不冒充已核实的原始运行环境。

本次重新训练在第 **7** 轮触发作者早停，最终 **MSE 0.3841444、MAE 0.4047131**，相对论文 Table 2 的 0.375/0.399 分别差 **2.44%/1.43%**，均进入项目事前设置的 **5% 相对容差**。113 个固定源码文件、数据 SHA、实际训练协议、预测产物和独立指标复算均通过，最终状态 `reproduced`、`analysis_status=completed`。该容差是项目验收规则，不是论文阈值；API 解释不覆盖确定性数值判定。结论只覆盖上述一个实验，DLinear 预设保留作者方案，不执行自动模型优化。

查看[新完整报告](data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report.md)、[最终结果](data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/result.json)、[四阶段分析](data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/repository_analysis.json)、[结果解释](data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/result_analysis.json)及[报告和图片 ZIP](data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report_with_figures.zip)。旧版单次 API 基线的[原始报告](data/runs/repository_66efc14890e34cb196a07a7d2bcb78fe/report.md)仍保留，新运行的指标与第 7 轮早停结果相同。完整对照见[复现结果](docs/dlinear_reproduction_result.md)。

2026-10-10 的 Windows 自动恢复验收运行 `repository_b707311a3bb84f9db886af456a3780b5` 从 Store 启动环境自动切换到 `.venv-py312`；GitHub REST 的 403 限额通过官方 issue 网页恢复两个准确作者评论，默认依赖镜像超时后从官方 PyPI 安装相同版本，并通过 Python 3.12 原生包健康检查。四阶段公开分析完成后执行完整作者训练和独立核验，最终 **MSE 0.3841444、MAE 0.4047131**，两项均在原 **5%** 容差内，状态为 `reproduced`。本次公开分析实际调用 **4 次**，未启用结果摘要解释；见[报告](data/runs/repository_b707311a3bb84f9db886af456a3780b5/report.md)与[最终结果](data/runs/repository_b707311a3bb84f9db886af456a3780b5/result.json)。

重跑步骤见 [DLinear 用户测试步骤](docs/dlinear_testing.md)，从 `AutoReproducer/` 目录运行：

```bash
python scripts/reproduce_repository.py --profile dlinear_etth1_reference
# 四阶段公开分析 + 完整训练 + 允许 API 解释结果摘要：
python scripts/reproduce_repository.py --profile dlinear_etth1_reference --llm-review --analysis-mode multi_agent --result-review
```

DLinear 与 Neural ODE 预设使用本地 CPU；SIREN 已适配 Windows/NVIDIA CUDA 并完成上述本机实验。依赖缓存按 Python ABI、OS、架构和依赖清单隔离；DLinear 10 月 7 日历史实测为 macOS ARM64，10 月 10 日自动恢复验收为 Windows AMD64 / Python 3.12。框架和硬件变化可能影响结果，不保证重跑得到逐位相同数值。

Markdown 图片使用同目录的 `<report_stem>_assets/` 相对资产目录。分享时使用网页“⬇️ 下载报告和图片（ZIP）”并保留解压后的目录结构；只复制 `.md` 无法携带图片。预分析拒绝会保留原文诊断并停止训练；每阶段最多一次修正，其他运行的实际调用数可能与本次 5 次不同。

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

> 依赖 Python 3.11+；DLinear 固定预设支持 Python 3.11/3.12，本地 CPU 路径无需 Docker。通用代码执行选择 Docker 时，需本地安装并启动 Docker Desktop。
> docker CLI 自动探测：优先 `DOCKER_PATH` 环境变量 → 系统 PATH → Docker Desktop
> 常见安装目录（`C:\Program Files\Docker\Docker\resources\bin\docker.exe`），
> Windows 上即使 docker 不在 PATH 中也能正常构建/执行。
> 侧边栏会进一步探测**引擎（daemon）是否在跑**——只装 CLI 没启动 Docker Desktop
> 时不会谎报「已就绪」，而是提示原因并自动改用本地隔离执行（启动后点
> 「🔄 重新检测 Docker」即可，无需刷新页面）。

## 运行测试

macOS、Windows 和 Linux 的通用运行稳定性说明见 [跨平台运行与中断恢复](docs/cross_platform_runtime.md)。
CI 配置覆盖三个系统与 Python 3.11/3.12；Windows 另跑 UTF-8 模式。真实论文训练与 API 验收独立于这些回归测试。

```bash
# 安装完整测试依赖（包含真实图片生成测试所需的 matplotlib）
python -m pip install -r requirements-test.txt

# 完整测试套件（无需真实 LLM API；真实实验记录另行保存）
python -m pytest tests/ -v

# Windows 也验证 UTF-8 模式；测试文件读写显式指定 UTF-8
python -X utf8 -m pytest tests/ -q

# 只看端到端集成用例
python -m pytest tests/ -v -k TestEndToEnd
```

GitHub Actions 在 Windows/Linux 与 Python 3.11/3.12 上运行回归；Windows
同时检查系统默认编码和 UTF-8 模式。测试使用模拟 API、小型本地程序与临时缓存，
不需要真实模型密钥或启动论文训练。缺少 matplotlib 的本地环境会明确跳过绘图用例。

## 当前核心流程（复现 → 验证 → 报告）

SIREN 与 Neural ODE 已接入上方三种优化模式，使用有限参数候选和真实训练评估。通用论文流程的旧优化开关仍是预留接口，DLinear 也不做参数优化；UCB/BeamUCT 等既有模块不参与当前方法实验的候选选择。

**1 个编排器 + 8 个专职 Agent：**

| Agent | 职责 |
|---|---|
| 📖 PaperReader | 解析论文 PDF，提取结构化信息 |
| 🔍 ResourceFinder | 查找代码仓库和数据集 |
| 🔧 EnvBuilder | 生成运行环境配置 / 真实构建 Docker 镜像 |
| ⚡ CodeExecutor | 在本地或 Docker 沙箱中运行代码 |
| ✅ ResultValidator | 比对论文声明值与运行结果 |
| 🛡️ Verifier | Prompt-Free 质量验证（复用各 Agent 系统提示词） |
| 🧪 Optimizer | 方法预设的来源建议、有限参数试验与两种子确认；通用流程仍预留 |
| 📝 ReportGenerator | 生成 Markdown 复现报告 |

**状态机流转：**

```
INIT → READ_PAPER → FIND_RESOURCES → BUILD_ENV → EXECUTE_CODE → VALIDATE
  → [方法预设可选：建议 / 候选训练与确认] → GENERATE_REPORT → COMPLETED
```

网页根据实际执行阶段显示进度，Agent 名称只是负责人。启用方法优化时，已完成的基线单独显示“基线执行完成”和“基线核验完成”；后续优化训练在优化卡片显示“代码执行中”和当前轮次。阶段负责人返回结果，且本阶段所属的执行轮次、步骤和后台进程已退出并完成清理，才确认阶段结束。整轮 `execution_run` 从准备到清理持续开放，步骤间隙仍在运行；`done` 或 `error` 提前到达会显示“结束确认中”，不能直接关闭任务。进程异常退出后，只由持有它的进程在确认清理后将未返回结果的执行标为中断，不补写成功。网页的“各部分结束判定”列出每阶段条件与确认状态。

当前执行按执行方提供的独立编号区分轮次与步骤，不从日志计数、论文名称、命令或目录推断状态。总步骤未知时只显示当前步骤，不估算百分比。阶段完成比例既不是耗时比例，也不代表论文数值验收通过，验收结论以 `validation` 为准。执行输出按轮次和步骤分段，使用 240 像素高的滚动窗口；普通脚本模式在步骤返回时更新输出，仓库模式提供实时输出。

### 标题与 PDF 输入的复现路线

直接输入下列完整论文标题，或上传首页标题与之匹配的原始 PDF，即可选择固定作者仓库并运行已核验的实验，不需要手动切换到仓库预设。标题只规范化大小写、空白与外层引号；相似标题、正文或参考文献中的提及不能选择预设。

| 论文 | 作者仓库 | 自动运行与结论范围 |
|---|---|---|
| Neural Ordinary Differential Equations | `rtqichen/torchdiffeq` | 官方螺旋拟合，2000 次迭代；完成作者方法实验，不等同于完整论文基准复现 |
| Are Transformers Effective for Time Series Forecasting? | `cure-lab/LTSF-Linear` | 已适配的 DLinear ETTh1 冻结协议与数值验收 |
| Implicit Neural Representations with Periodic Activation Functions | `vsitzmann/siren` | 已适配的官方单图拟合方法实验 |

PDF 先验证文件与正文，再识别首页标题，保存 SHA-256、页数和字节数；提取为空时尝试另一解析器。缺失、损坏、无法解密或没有可读正文的 PDF 明确停止，不能用空内容生成训练代码。解析依赖缺失则先自动准备兼容运行环境再重试。标题和 PDF 的通用路线也先检查宿主环境，不只仓库预设会自动恢复；Windows Store 等不兼容环境交给受控的兼容进程执行。显式仓库预设、语料对照或外部代码保持其输入优先级。

匹配路线默认在线核对固定作者源码与项目适配证据；可关闭对应复选框。这项审核不宣称已分析论文全文。旧预设的优化、仅准备和结果分享选项不会带入标题或 PDF 输入。其他有效论文继续进入通用发现与代码重建路线：搜索候选仓库不等于核验作者仓库，也不等于执行了该仓库；证据不足的运行只能给出尽力重建或无法验收的结论。详见 [输入与结束判定验收记录](docs/paper_input_completion_20261010.md)。

**每个阶段输出都会经过 Prompt-Free 验证**（Verifier 复用该 Agent 的
`system_prompt` 作为质量标准）；验证未通过时按修正建议触发一次修正重试，
形成「生成 → 验证 → 修正 → 再验证」闭环（预算约束内，`MAX_FIX_RETRIES=1`）。

## 核心创新点

1. **真实实验与确定性核验** — 完整执行后分别检查协议、独立指标复算和数值验收
2. **有限参数优化** — 方法预设使用冻结验证协议和两种子留出确认；UCB 等既有模块仍保留供后续开发
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
- 通用论文流程的下载助手采用“尽力而为”策略：网络不可用时可返回明确标记的 `dataset_smoke` 诊断数据。本文顶部的官方仓库完整实验使用单独的固定来源与哈希校验，在线缺失或坏缓存会按原版本自动恢复；有限重试后仍无法取得真实代码或数据才失败，不替换为合成数据。

### 隔离依赖安装（P0-3）

真实模式下 `pip install --target data/deps/<sha1(reqs)>` 一次性安装到隔离目录并落 `.ready` 就绪标记，执行时经 **PYTHONPATH 注入** 该目录；同名依赖清单跨论文跨会话只落一份（多论文共享、天然去重），不污染全局 Python。`AUTOREPRO_DEPS_ROOT` 可覆盖根目录。

官方预设的依赖缓存进一步按 Python ABI、系统和架构隔离；每次复用都检查固定版本、导入及关键原生运算。就绪标记存在但缓存已损坏时，会加锁重建相同版本；默认公共镜像超时或连接失败时可回退官方 PyPI，准备和健康检查记录随本次运行保存。

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

- **真实模式**（网页默认）— 配置 **OpenAI 兼容的远程 LLM API** 执行分析；官方仓库预设也可关闭分析与优化选项后直接训练。预设按固定配置使用本地 CPU 或 CUDA，不依赖 Docker。
- **Mock 模式**（手动开启）— 无需 LLM API / Docker，演示通用流程；官方仓库预设要求真实模式。

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
