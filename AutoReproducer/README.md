# AutoReproducer

基于多智能体协作的论文自动复现与优化系统

已下载的测试论文及来源清单见 [PDF 样本归档](examples/papers/README.md)，可直接选择原始 PDF 上传测试。清单保留文件 SHA-256、不同论文版本和错误候选说明。

## 从 PDF 运行作者代码

上传 PDF 后直接定位仓库、取得源码、读取 README 和依赖声明、安装环境并运行作者命令。论文匹配审核、Agent 输出审核、实验协议预审、命令引文审核、数据来源审核和论文数值验收已从运行流程移除，前端不再提供审核或严格模式开关。公开 GitHub 仓库不需要登录或先 fork。

模型负责论文阅读、仓库定位和运行命令规划。模型规划不可用时，尝试 README 或启动脚本中的原始 Python 命令；所选入口及本地导入会自动读取。作者程序负责加载数据；文档明确给出的仓库内压缩包仍可先解压。实际缺包时自动安装并重试，最多修复 8 轮；同一缺包反复出现、真实运行错误、取消和超时会保留失败原因。

只有所选命令以退出码 0 结束且所属进程完成清理，才显示代码运行完成。报告保留真实命令、日志、实际指标和已观察到的训练更新；未观察到更新也可记录代码正常结束，不能据此宣称训练或论文数值通过。

**当前未完成**：运行结束后，自动将实测指标与论文或作者宣称对齐并计算差异的环节尚未接入当前流程。报告中的实际指标不代表已经完成数值比较。作者命令和外层后台任务目前仍有超时限制；取消运行时间限制、仅保留报错修复次数上限是后续待办。

2026-10-12 换用 AdaGNN PDF 完成了后台真实运行：作者默认 `python main.py` 自然返回，327 次优化器更新，BlogCatalog 测试准确率 0.8953。范围与证据见 [移除审核后的运行记录](docs/execution_without_reviews_20261012.md)。

1. 按“快速启动”启动网页；仓库定位需要模型时，在侧栏配置 API。
2. 选择“上传PDF”，上传可读取正文的原始论文，点击“开始复现”。
3. 找不到仓库时，在“补充 GitHub 仓库”中填写地址并重试。
4. 在“各部分结束判定”查看命令退出和进程清理状态，完成后下载运行报告。

之前的真实测试与修复记录保留在 [历史独立测试记录](docs/independent_pdf_acceptance.md)。这些历史记录不能证明当前所有 PDF 均可运行。

## 固定预设实验

网页预设与标题匹配入口同样直接运行，已移除在线预审、单独导入审核、训练后协议审核、独立数值验收和参数优化确认环节。系统自动取得源码和真实输入数据、安装隔离依赖，再执行训练命令。单独准备环境仍可使用；环境准备成功不等于训练完成。

| 预设 | 运行内容 |
| --- | --- |
| `dlinear_etth1_reference` | 作者 ETTh1 336→96 训练与自带测试，CPU |
| `siren_camera_quick` | 作者方法全图拟合，500 步，CUDA |
| `neural_ode_spiral` | 官方二维螺旋示例，2000 次迭代，CPU |

```powershell
python scripts/reproduce_repository.py --profile siren_camera_quick --prepare-environment
python scripts/reproduce_repository.py --profile siren_camera_quick --optimization off
python scripts/reproduce_repository.py --profile neural_ode_spiral --optimization off
```

旧的审核和优化请求参数仅作调用兼容，当前运行入口不会据此重新启用审核。以下历史实验记录保留当时的数值和运行证据。

## DLinear 官方仓库复现（2026-10-07：真实 4 + 1 多 Agent 流程通过）

本节记录历史版本及当时的操作选项；当前版本已移除这些审核开关，实际流程以上方“从 PDF 运行作者代码”为准。

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

仓库有一层同名子目录。macOS / Linux 启动前请进入包含 `app.py` 和 `requirements.txt` 的项目目录；Windows PowerShell 命令会自动识别 Git 仓库根目录或内层项目目录。

Windows PowerShell（从 Git 仓库根目录或内层项目目录均可执行）：

```powershell
if (Test-Path .\app.py) {
    $ProjectDir = (Get-Location).Path
} elseif (Test-Path .\AutoReproducer\app.py) {
    $ProjectDir = (Resolve-Path .\AutoReproducer).Path
} else {
    throw "找不到 app.py；请在 Git 仓库根目录或包含 app.py 的项目目录运行。"
}
Set-Location $ProjectDir

# 优先使用 Windows Python Launcher 选择最新 Python 3；没有 Launcher 时使用 PATH 中的 python。
$Launcher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
$PythonArgs = @()
if ($Launcher) {
    $PythonCommand = $Launcher.Source
    $PythonArgs = @("-3")
} else {
    $Launcher = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "找不到 Python。请安装 Python 3.11 或更高版本，并确保 py 或 python 在 PATH 中。"
    }
    $PythonCommand = $Launcher.Source
}
$PythonVersion = & $PythonCommand @PythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $PythonVersion -notmatch '^3\.(1[1-9]|[2-9]\d)$') {
    throw "需要 Python 3.11 或更高版本；检测到 '$PythonVersion'。请用 py -0p 查看已安装版本。"
}

$Venv = ".venv-py$($PythonVersion.Replace('.', ''))"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $PythonCommand @PythonArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败：$Venv" }
}
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败；请检查上方 pip 错误信息后重试。" }

# 启动；保持此终端打开，不需要激活虚拟环境。
& $VenvPython -m streamlit run app.py --server.port 8501
```

命令会自动从 Git 仓库根目录进入内层项目目录，并使用同一个 Python 解释器创建虚拟环境、安装依赖和启动应用；不需要激活环境，也不受 PowerShell 激活脚本执行策略或全局 `streamlit` PATH 的影响。若提示找不到 Python，或版本低于 3.11，可运行 `py -0p` 查看已安装解释器，再安装 Python 3.11 或更高版本并重试。

macOS / Linux：

```bash
cd AutoReproducer
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.port 8501
```

保持启动终端打开，浏览器访问 <http://localhost:8501>；按 `Ctrl+C` 停止服务。若提示端口已占用，先检查已有页面，或把 `--server.port 8501` 改为 `--server.port 8502` 后访问对应地址。若提示找不到 `app.py` 或 `requirements.txt`，检查是否进入了内层项目目录。

> 依赖 Python 3.11+；DLinear 固定预设支持 Python 3.11/3.12，本地 CPU 路径无需 Docker。通用代码执行选择 Docker 时，需本地安装并启动 Docker Desktop。
> docker CLI 自动探测：优先 `DOCKER_PATH` 环境变量 → 系统 PATH → Docker Desktop
> 常见安装目录（`C:\Program Files\Docker\Docker\resources\bin\docker.exe`），
> Windows 上即使 docker 不在 PATH 中也能正常构建/执行。
> 侧边栏会进一步探测**引擎（daemon）是否在跑**——只装 CLI 没启动 Docker Desktop
> 时不会谎报「已就绪」，而是提示原因并自动改用本地隔离执行（启动后点
> 「重新检测 Docker」即可，无需刷新页面）。

## 运行测试

macOS、Windows 和 Linux 的通用运行稳定性说明见 [跨平台运行与中断恢复](docs/cross_platform_runtime.md)。
GitHub Actions 覆盖 Ubuntu、Windows 与 Python 3.11/3.12 的四组配置；Windows 另跑 UTF-8 模式。它验证程序回归，不是部署在线复现服务，也不会自动证明某篇论文训练成功。真实论文训练与 API 验收使用上方单独的运行记录。

移除审核之前，2026-10-12 的本地全量回归为 **2725 项通过、66 项跳过**，跳过项受平台或本地原生依赖条件限制。代码提交 `e47cff3` 的四组 PR CI 均通过，见 [GitHub 测试记录](https://github.com/Yukikazct/AutoReproducer/actions/runs/38231904173)。这不是本次修改的全量回归结果；当前相关测试及待迁移的历史审核测试见 [本次验证记录](docs/execution_without_reviews_20261012.md)。

以下 `python` 指已安装项目依赖的解释器。未激活虚拟环境时，Windows 将它替换为 `.\.venv-py311\Scripts\python.exe`，macOS/Linux 替换为 `.venv/bin/python`，与快速启动使用同一环境。

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

## 当前核心流程（阅读 → 定位 → 安装 → 运行 → 报告）

| Agent | 当前职责 |
| --- | --- |
| PaperReader | 从论文全文、脚注和代码声明中提取仓库位置与实验信息 |
| ResourceFinder | 根据链接、搜索和模型候选定位仓库；允许用户补充地址 |
| EnvBuilder | 从 README、依赖文件及源码导入选择并安装运行依赖 |
| CodeExecutor | 执行命令，遇到实际缺包自动修复并重试，记录退出状态及进程清理 |
| ReportGenerator | 汇总真实运行日志、指标和结束状态 |

```
INIT → READ_PAPER → FIND_RESOURCES → BUILD_ENV → EXECUTE_CODE
  → GENERATE_REPORT → COMPLETED / ERROR
```

RepositoryVerifier、ExperimentProtocol、Verifier 和 ResultValidator 已退出当前编排流程。不存在“审核未通过所以不启动训练”的分支，也不自动执行论文值对比或独立数值验收。相关存量工具和历史记录保留供独立开发使用。

标题输入匹配已知论文时选择相应预设；其余 PDF 从当前上传的正文提取链接并尝试运行。入口不存在、数据下载失败、依赖安装失败及作者代码异常仍是实际运行问题，会显示具体报错。

阶段负责人返回结果且所属轮次、步骤和后台进程退出并完成清理后，才能确认该部分结束。步骤间隙和日志中的最终轮数不代替结束确认；异常退出只记录失败或中断。网页按实际阶段计数显示进度，并在“各部分结束判定”列出结束条件。

## 核心机制

1. 论文全文阅读与仓库位置提取。
2. README 命令规划、隔离依赖缓存和实际缺包修复。
3. 真实命令退出、所属进程清理和运行日志归档。

## 存量工具与历史机制

| 机制 | 说明 |
|---|---|
| 输入透传 | 支持论文标题 / 上传 PDF / 外部代码三种输入，正确进入数据上下文 |
| 5 轮依赖诊断 | EnvBuilder 循环「探测 → 分类 → 定点修复 → 重验证」，输出逐轮修复报告 |
| 执行路线 | 旧代码生成流程采用 smoke + full；仓库路线执行冻结的原始完整命令 |
| 指标口径统一 | 论文声明 0.85（小数）与运行输出 85.2%（百分数）自动归一化后再比对 |
| 用量统计 | 全程 LLM 调用次数累计，纳入审计统计与报告；全文阅读不设总调用预算 |
| 修正闭环 | 先运行模式审查作参考，实际缺包和已支持的数值兼容修复共享最多 3 轮；传统阶段输出最多修正 1 次 |
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

仓库执行现在支持**基础环境与增量包复用**。已有环境包含相同版本的所需包时直接复用；清单增加小包时，由 pip 解析完整版本及传递依赖，复用相同版本的基础包，只将缺少的固定 wheel 安装到增量目录。例如已有 CPU PyTorch、NumPy、SciPy 时，新增 scikit-learn 不再复制整套 PyTorch。完整解析结果、wheel 哈希、基础环境引用和实际导入路径随运行保存。Python 3.11／3.12、CPU／CUDA 和不同版本继续隔离，私有源及复杂源码清单保留独立安装路径。

新安装前自动回收失败残留与闲置环境，按最后使用时间处理容量或空间压力；默认依赖缓存上限为 **20 GiB**、磁盘至少保留 **2 GiB**，超过 30 天未使用的环境可回收。运行期间持有缓存锁，增量目录引用的基础环境也受到保护。空间仍不足时明确提示调整缓存目录。网页“历史记录 → 依赖缓存”可查看各目录的占用、最后使用时间和基础环境引用；不会清理 PDF、数据集或运行证据。详情见 [依赖环境管理](docs/dependency_management.md)。

```powershell
# 仅预览回收计划；加 --yes 执行
python scripts/resource_cli.py deps-prune
python scripts/resource_cli.py deps-prune --yes

# 可选：启动前将新依赖缓存放在空间充足的磁盘
$env:AUTOREPRO_DEPS_ROOT = "E:\AutoReproducer-cache\deps"
$env:AUTOREPRO_DEPS_QUOTA_GB = "20"
$env:AUTOREPRO_DEPS_MIN_FREE_GB = "2"
```

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

## 存量工程模块（当前审核流程已移除）

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

- **真实模式**（网页默认）— 配置 **OpenAI 兼容的远程 LLM API** 执行分析；官方仓库预设直接安装依赖并训练，无需审核 API。预设按固定配置使用本地 CPU 或 CUDA，不依赖 Docker。
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

真实 PDF 仓库执行（远程 LLM API + 本地受控进程，先验证代码可运行）：

```python
# 前提：已设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量
from src.orchestrator import Orchestrator
from src.llm.llm_client import LLMClient

llm = LLMClient(mock_mode=False)          # 参数缺省时自动读环境变量
orch = Orchestrator(llm_client=llm, mock_mode=False,
                    use_docker=False)
result = orch.run({
    "pdf_path": "paper.pdf",
    "author_execution_only": True,
    # "preferred_repo_url": "https://github.com/owner/repo",  # 自动定位失败时补充
})
print(result["state"])
print(result["data"].get("validation"))
```

编程接口须显式传入 `author_execution_only=True` 才选择先运行目标；网页对未注册预设的 PDF 默认勾选该目标。设为 `False` 可请求数值复现。通用作者仓库路线当前使用本地受控进程，Docker 用于其他已支持的代码执行路线。

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
