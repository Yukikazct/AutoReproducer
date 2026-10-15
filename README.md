# AutoReproducer

面向 **PC 桌面浏览器**的论文复现实验工作台。使用 Streamlit 提供界面，由多 Agent 组织论文阅读、代码定位、环境准备和执行，并保存命令、日志、指标与报告。

当前主要入口是**上传论文 PDF 运行作者代码**，也提供固定实验预设和无需 API 的 Mock 演示。

> **当前能力范围**：真实运行显示“代码运行完成”时，表示所选代码正常结束、所属进程完成清理；仅准备源码或环境不代表训练完成。PDF 作者仓库流程会在运行后尝试对比作者输出与论文指标，独立数值验收与自动优化尚未开放。

[快速开始](#快速开始) · [模型配置](#模型配置) · [使用流程](#使用流程) · [固定实验预设](#固定实验预设) · [依赖与文件管理](#依赖与文件管理) · [开发与测试](#开发与测试) · [更多文档](#更多文档)

## 快速开始

### 环境要求

- 推荐 **64 位 CPython 3.12**，也支持 3.11；当前预设运行环境按这两个版本准备。
- 安装 Git；首次取得作者源码、数据和依赖需要联网。
- DLinear、Neural ODE 使用 CPU；SIREN、ReZero 的现有预设需要 NVIDIA GPU 和可用的 CUDA 环境，不能在普通 CPU 或 Apple GPU 上直接运行。
- PDF 作者仓库和固定预设使用本地受控进程，启动时保持 Docker 关闭。

### macOS / Linux

选符合你情况的一段，完整复制执行。本 README 位于仓库根目录（外层）；应用目录是其下包含 `app.py` 的 `AutoReproducer/`。除注明“当前在外层目录”的示例外，后文命令均在应用目录中执行。

**情况 1：已经安装过，当前在 `AutoReproducer` 目录（也适用于这里再次 `cd AutoReproducer` 报错后）。**

```bash
.venv/bin/python -m streamlit run app.py --server.port 8501
```

**情况 2：已经安装过，当前在外层目录。**

```bash
cd AutoReproducer &&
.venv/bin/python -m streamlit run app.py --server.port 8501
```

**情况 3：第一次安装，当前在 `AutoReproducer` 目录。缺少 `.venv/bin/python` 或 `streamlit` 时也用这段。**

```bash
python3.12 -m venv .venv &&
.venv/bin/python -m pip install -r requirements.txt &&
.venv/bin/python -m streamlit run app.py --server.port 8501
```

**情况 4：第一次安装，当前在外层目录。**

```bash
cd AutoReproducer &&
python3.12 -m venv .venv &&
.venv/bin/python -m pip install -r requirements.txt &&
.venv/bin/python -m streamlit run app.py --server.port 8501
```

**情况 5：没有 Python 3.12，已安装 Python 3.11。**

当前在 `AutoReproducer` 目录：

```bash
python3.11 -m venv .venv &&
.venv/bin/python -m pip install -r requirements.txt &&
.venv/bin/python -m streamlit run app.py --server.port 8501
```

当前在外层目录：

```bash
cd AutoReproducer &&
python3.11 -m venv .venv &&
.venv/bin/python -m pip install -r requirements.txt &&
.venv/bin/python -m streamlit run app.py --server.port 8501
```

**情况 6：8501 端口被占用，改用 8502。** 如果 <http://localhost:8501> 已经是本应用，直接使用该页面即可。需要另开服务时执行下面对应的一段。

当前在 `AutoReproducer` 目录：

```bash
.venv/bin/python -m streamlit run app.py --server.port 8502
```

当前在外层目录：

```bash
cd AutoReproducer &&
.venv/bin/python -m streamlit run app.py --server.port 8502
```

改用 8502 后打开 <http://localhost:8502>。Python 3.11 和 3.12 都没有安装时，先安装 [Python 3.12](https://www.python.org/downloads/release/python-31210/)，再执行情况 3 或 4。

### Windows PowerShell

**情况 1：已经安装过，当前在 `AutoReproducer` 目录。**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

**情况 2：已经安装过，当前在外层目录。**

```powershell
Set-Location AutoReproducer -ErrorAction Stop
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

**情况 3：第一次安装，当前在 `AutoReproducer` 目录。缺少虚拟环境或 `streamlit` 时也用这段。**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

**情况 4：第一次安装，当前在外层目录。**

```powershell
Set-Location AutoReproducer -ErrorAction Stop
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

**情况 5：没有 Python 3.12，已安装 Python 3.11。**

当前在 `AutoReproducer` 目录：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

当前在外层目录：

```powershell
Set-Location AutoReproducer -ErrorAction Stop
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

**情况 6：8501 端口被占用，改用 8502。** 如果 <http://localhost:8501> 已经是本应用，直接使用该页面即可。

当前在 `AutoReproducer` 目录：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

当前在外层目录：

```powershell
Set-Location AutoReproducer -ErrorAction Stop
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

PowerShell 安装步骤逐行执行，上一行成功后再执行下一行。改用 8502 后打开 <http://localhost:8502>。

### 打开页面与停止服务

默认打开 <http://localhost:8501>。保持终端打开，按 `Ctrl+C` 停止服务。上面的命令都无需激活虚拟环境。

**第一次体验**：有模型 API 时，先完成下方配置，再上传 PDF；只看界面和流程时，展开侧栏“运行环境”，开启 Mock 演示模式。没有 API 也可以在真实模式下运行固定预设。

## 模型配置

### DeepSeek 实际使用哪个模型

项目通过 **OpenAI 兼容的 Chat Completions API** 调用模型，不在本地部署 DeepSeek。实际请求的 `model` 字段等于当前生效的 `LLM_MODEL`；网页“LLM API 配置”底部会显示生效模型与地址。

需要区分代码默认值和服务商的模型版本：

| 项目 | 当前情况 |
| --- | --- |
| 网页内置 API 地址 | `https://api.deepseek.com` |
| 网页内置模型名 | `deepseek-chat`，仍是代码中的旧默认值 |
| DeepSeek 当前公开模型名 | `deepseek-flash`、`deepseek-v4-pro` |
| 官方模型版本对应关系 | `deepseek-flash` → **DeepSeek-V4.1-Flash**；`deepseek-v4-pro` → **DeepSeek-V4-Pro-0813** |

以上官方模型信息核对于 **2026-10-13**，来源：[DeepSeek 首次调用 API](https://api-docs.deepseek.com/zh-cn/)与[模型说明](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)。服务商可能更新别名对应的模型；旧文档中的 `deepseek-chat = V3`、`deepseek-reasoner = R1` 不能继续作为当前版本说明。

新配置可参考官方当前模型名，例如在侧栏将“模型名称”改为 `deepseek-flash`。**项目尚未对这些新模型完成真实 API 验收**，这里记录的是官方配置参考；代码不会自动把旧默认值替换为新模型。本项目也没有单独设置 DeepSeek 的 `thinking` / `reasoning_effort` 参数，不能把填写模型名等同于配置思考模式。

### 在网页中配置

1. 在 [DeepSeek 开放平台](https://platform.deepseek.com/)创建 API Key。
2. 展开侧栏“LLM API 配置”，填写 API 地址、API Key 和模型名称。
3. 点击“测试 AI 连接”检查响应；这是一次真实 API 请求，会使用服务额度。

也可以连接其他提供兼容 Chat Completions 接口的服务，填写对应的地址、密钥与模型名。地址填写 API 根地址或 `/v1`、`/v2` 前缀地址，**不要填写完整的 `/chat/completions` 地址**；客户端会自行拼接。

### 保存配置

可在应用目录 `AutoReproducer/` 内新建 `.streamlit/secrets.toml`，使用顶层字段：

```toml
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "替换为你的 API Key"
LLM_MODEL = "deepseek-flash"
```

该文件已被 Git 忽略。网页和预设 CLI 会读取它；直接使用 Python 编程接口时，需自行调用 `load_local_llm_settings()` 或传入配置。项目**不会自动加载 `.env` 文件**。

也可在启动服务前设置环境变量。Windows PowerShell：

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_API_KEY = "替换为你的 API Key"
$env:LLM_MODEL = "deepseek-flash"
```

macOS / Linux：

```bash
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="替换为你的 API Key"
export LLM_MODEL="deepseek-flash"
```

网页配置优先级为：**非空界面输入 → 已有环境变量 → 本地 `secrets.toml` → 网页内置默认值**。已有环境变量不会被本地文件覆盖；界面输入留空会回退配置来源。

| 可选环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `LLM_TIMEOUT` | `120` 秒 | 普通模型请求超时 |
| `LLM_MAX_TOKENS` | `8192` | 普通模型请求的单次输出上限，并非服务商模型的最大能力 |

连接测试单独使用 **20 秒超时、64 token 输出上限**，不采用上表设置；思考型模型可能在这种短测试中超时或未返回最终正文。客户端当前读取 `choices[0].message.content`，不展示 `reasoning_content`。

## 使用流程

### 选择输入

| 输入方式 | 适用场景与实际行为 |
| --- | --- |
| 上传 PDF | 推荐的通用入口。提取正文及仓库链接；匹配已支持论文时进入预设，否则尝试定位作者仓库并运行其命令。当前没有扫描件 OCR。 |
| 论文标题 | DLinear、SIREN、Neural ODE 的完整标题可匹配预设；未匹配标题会进入通用分析流程，不能保证获取全文或执行作者代码，建议改为上传 PDF。 |
| 官方仓库预设 | 按固定源码、数据和参数运行下表中的实验，当前无需 LLM API。 |

上传 PDF 后，确认解析提示，点击“开始复现”。如果自动定位仓库失败，页面出现“补充 GitHub 仓库”入口时，可填入地址并重试。手动指定仓库会被记录为用户提供，不代表系统已经确认作者归属。

普通 PDF 流程中，模型用于论文阅读、仓库发现和原作者命令规划。仓库取得后再检查脚本、`python -m` 模块或 README 指明的 Notebook；同一仓库最多四次规划提案、两轮源码补读。当前实现不可运行时，最多检查三个仓库，只跟随论文或作者 README 明确关联的训练实现；手动指定的仓库与版本保持锁定。Notebook 使用选定训练 Python 的 kernel 顺序执行全部代码单元，保存输出副本和失败单元，不跳过失败单元。

作者训练使用与应用分离的隔离 Python。作者 Python 声明或实际请求包的精确 `Requires-Python` 冲突会触发兼容解释器选择；公共 PyPI 下载源依次尝试当前镜像、清华、阿里云、中科大，最后尝试官方源，保留原版本约束。私有源限制和 PyTorch／PyG 的设备专用 wheel 来源保持不变。实际缺包、已支持的 NumPy 兼容错误、作者临时数据目录缺失，以及模型提出并经日志验证的项目推导依赖版本，共享最多 **8 次修复动作**。每种新的执行失败还会触发有限的模型分析：每个仓库最多四轮失败分析，每轮最多两次固定源码补读；相同失败与环境不重复调用。模型只可建议受支持的动作，程序会校验原始 pip 日志、作者版本和固定源码后才重试原命令。失败分析、解释器、下载源和修复结果保存在运行记录中。作者命令、固定预设和外层后台任务不设运行时间上限，等待程序实际退出并完成进程清理；下载、依赖安装和模型请求仍有各自的超时。

这些 Agent 由编排器按阶段顺序执行，使用同一个 LLM 客户端，并非同时运行的独立模型进程。论文阅读、资源发现、入口规划、运行失败分析和完成后的指标比较会按各自证据调用模型；下载、pip、kernel、进程清理和报告落盘由程序执行。这样模型负责判断，实际执行与是否成功由可核验的日志和固定约束决定。

### 查看进度和结果

- **流水线状态**：查看阶段开始/结束时间、当前阶段耗时，以及可按阶段筛选的过程记录（论文读取、源码与数据准备、依赖安装、执行和修复）。阶段内日志产生后立即同步；单阶段完成不代表整个任务完成。
- **复现报告**：查看实测指标、执行记录，并下载 Markdown 或“报告和图片 ZIP”。分享带图片的报告时应保留 ZIP 解压后的目录结构。
- **审计日志 / 状态机**：定位执行步骤、错误和状态变化。
- **历史记录**：回看实验，查看存储概览、资源事件与依赖缓存，并执行清理操作。

运行监控通过 Streamlit 原生机制每 2 秒读取后台记录，并显示最近事件的时间。日志暂时安静时会继续等待，不会把任务标记为完成。可点击“刷新运行状态”手动读取。浏览器地址中的 `run=progress_….jsonl` 仅为运行记录标识，用于刷新后恢复监控；不包含 API Key。旧页面丢失当前任务时，在“恢复已有任务监控”中选择最近的记录并点击“恢复监控”，不会重新训练。代码执行输出单独保留在过程记录下方。

安装依赖时，“依赖安装进度”显示当前软件包、实际下载来源、文件下载量、最近速度、安装用时与时限，以及距最近活动的秒数。只有取得可靠文件总量时才显示当前文件百分比；总量未知或正在安装/复制时会明确说明。刷新或恢复监控会接着显示同一轮安装。已审核的 CPU/CUDA 固定依赖从首轮起使用 1800 秒安装时限，单纯达到总时限不会反复换镜像重下；明确的网络故障仍会尝试备用来源。

旧报告会显示格式更新提示。在“复现报告”或“历史记录”中点击“重新生成报告”，即可用保存的结构化结果补齐实测值；不重新训练、不调用模型，也不改变指标和执行状态。原报告与日志保留，页面、Markdown 和 ZIP 使用更新后的同一版本。新运行会在报告旁保存 `.result.json`，清理进度缓存后仍可重建；仅剩 Markdown、没有结构化结果的旧记录无法自动补回数值。

### 运行模式与边界

网页默认真实模式，Mock 开关位于“运行环境”。Mock 不调用真实 LLM、不训练论文实验，但会执行本地演示脚本；固定预设要求关闭 Mock。

当前 PDF 作者仓库和固定预设使用**本地受控子进程**。依赖目录隔离用于管理 Python 包，不等于容器沙箱。Docker 仅用于其他已支持的通用代码执行路线；开启侧栏 Docker 开关不能让上述作者仓库路线自动容器化。

当前流程已移除论文身份、Agent 输出和实验协议等审核关卡，也不执行独立数值验收与参数优化。旧的审核／优化参数仅作调用兼容，`author_execution_only=False` 不会重新启用数值验收，`--budget-seconds` 不限制代码执行时间。PDF 作者仓库流程在作者命令完成后，使用已配置的模型从原 PDF 中定位对应指标，经引文和实测来源校验后展示数值差值、实验设置与比较限制，并保存 `author_claim_comparison.json`。该比较使用作者输出，尚未独立复算；参考值缺失、设置不一致或模型调用失败均不改变代码完成状态。固定实验预设仍无需 LLM API。**代码运行完成或达到比较参考值，均不代表论文数值复现成功。**

## 固定实验预设

| 预设 ID | 执行内容 | 设备 |
| --- | --- | --- |
| `dlinear_etth1_reference` | ETTh1 多变量 336→96，最多 10 轮、早停 patience 3 | CPU |
| `dlinear_etth1_smoke` | ETTh1 96→96，1 轮快速诊断 | CPU |
| `siren_camera_quick` | 256×256 图像全图拟合，500 步 | NVIDIA CUDA |
| `neural_ode_spiral` | 官方螺旋轨迹示例，2000 次迭代 | CPU |
| `rezero_cifar10_reference` | CIFAR-10，ReZero PreActResNet18，45 轮 | NVIDIA CUDA |

这些预设覆盖所列实验范围；名称中的 `reference` 或“完整实验”不表示当前流程已经完成论文数值验收。CUDA 预设使用固定的 PyTorch CUDA 12.1 依赖，需要兼容驱动和硬件。

以下及后文的 `python` 均指项目虚拟环境解释器：Windows 使用 `.\.venv\Scripts\python.exe`，macOS/Linux 使用 `.venv/bin/python`。

```bash
# CPU 快速诊断
python scripts/reproduce_repository.py --profile dlinear_etth1_smoke

# 完整 DLinear 单项实验
python scripts/reproduce_repository.py --profile dlinear_etth1_reference

# 仅准备源码、数据与命令，不安装论文依赖、不训练
python scripts/reproduce_repository.py --profile dlinear_etth1_reference --prepare-only

# SIREN / Neural ODE 可单独准备环境，不训练
python scripts/reproduce_repository.py --profile siren_camera_quick --prepare-environment

# 使用已有代码、数据及可用安装资源；缺失时失败，不能依赖上次运行的实验环境
python scripts/reproduce_repository.py --profile neural_ode_spiral --offline
```

首次准备环境会下载依赖，环境就绪不代表训练完成。`--prepare-environment` 保存准备结果后也会释放本次环境，下一次运行需要重新安装依赖。作者命令和固定预设不设执行时间上限；更完整的来源与参数见 [预设定义](AutoReproducer/src/repository_profiles.py)、[方法实验定义](AutoReproducer/src/method_profiles.py)和 [ReZero 定义](AutoReproducer/src/rezero_profiles.py)。

## 依赖与文件管理

### 应用依赖与实验依赖

- [requirements.txt](AutoReproducer/requirements.txt) 统一维护网页和主进程依赖；[requirements-test.txt](AutoReproducer/requirements-test.txt) 引用它并增加 pytest、matplotlib。
- PyTorch 等论文依赖按实验安装到托管目录，同一依赖根最多使用一套实验环境及补装包。报告和结构化结果保存后自动释放；切换依赖清单或训练 Python 前先清理上一套环境。
- 作者仓库入口在安装后检查依赖版本和导入；固定预设仍按各自配置决定是否启用健康检查。自动释放不依赖健康检查开关、容量或缓存年龄。历史缓存及单独调用底层执行器的维护策略见 [依赖环境管理](AutoReproducer/docs/dependency_management.md)。

### 数据目录

默认数据根是仓库内的 `AutoReproducer/data/`（应用目录中的 `data/`），不随启动时的工作目录改变。常用目录如下：

代码获取共用按仓库 URL 与完整 commit 区分的 `repository_cache/` 源码缓存。通用资源获取每次导出到独立的 `repos/<paper_id>/workspaces/<运行ID>/repo/`；预设和 PDF 作者执行使用各自的独立运行目录。指定版本无法获取时停止使用该仓库，显式固定版本的通用任务也会停止后续执行；未指定版本时先将远端 HEAD 解析为完整 commit。报告和资源清单记录实际版本、源码归档及逐文件哈希，旧实验的修改和产物不会带入下一次运行。

| 目录 | 内容 |
| --- | --- |
| `runs/` | 作者仓库与预设的运行记录、计划和产物 |
| `reports/`、`logs/`、`experiment_ledger/` | 报告、审计日志与实验账本 |
| `runtime/` | 网页后台进度及运行临时文件 |
| `repos/`、`datasets/`、`manifests/` | 缓存代码、数据和资源清单 |
| `deps/` | 本次实验的隔离依赖与安装临时文件，报告保存后释放 |
| `archive/` | 通过资源 CLI 创建的归档 |

可在启动前设置目录和清理参数：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `AUTOREPRO_DATA_ROOT` | `<应用目录>/data` | 数据根目录 |
| `AUTOREPRO_DEPS_ROOT` | `<数据根>/deps` | 独立依赖缓存目录，可放到另一块磁盘 |
| `AUTOREPRO_L0_QUOTA_GB` | `20` GiB | 资源配额；不等于全部文件的总容量限制 |
| `AUTOREPRO_DEPS_QUOTA_GB` | `20` GiB | 依赖回收的容量目标 |
| `AUTOREPRO_DEPS_MIN_FREE_GB` | `2` GiB | 依赖回收的磁盘剩余空间目标 |

例如 Windows PowerShell：

```powershell
$env:AUTOREPRO_DATA_ROOT = "E:\AutoReproducer-data"
$env:AUTOREPRO_DEPS_ROOT = "E:\AutoReproducer-deps"
```

macOS/Linux 使用 `export AUTOREPRO_DATA_ROOT="/绝对路径"`。修改配置只改变后续读写位置，不会迁移已有文件；需要完整迁移时，先停止任务，再复制数据目录及独立配置的依赖目录。

论文运行环境默认自动释放：同一依赖根的任务串行使用一套实验依赖，报告和结构化结果保存后删除论文包、补装包、托管训练 Python 及其下载缓存；异常和中断也执行清理。启动下一次任务前先清除遗留环境，清理失败则停止创建新环境。应用自身的 Python 与生产依赖保留，报告、图表和日志保留。详见 [依赖环境管理](AutoReproducer/docs/dependency_management.md)。

### 查看与清理

```bash
python scripts/resource_cli.py status
python scripts/resource_cli.py resources --json
python scripts/resource_cli.py deps-prune          # 仅预览依赖回收计划
python scripts/resource_cli.py deps-prune --yes    # 执行依赖回收
```

完整复现任务按上述规则在报告保存后释放环境，不保留最近使用的实验依赖。下面的容量与保留策略用于历史缓存、手动 `deps-prune` 和单独调用底层依赖执行器时的维护。底层执行器取得缓存锁、安装前及安装／执行结束时均会维护，缓存命中也会触发，且不依赖健康检查开关。统计覆盖各 Python／平台目录和增量环境，保护在用环境及完整基础环境引用链。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `AUTOREPRO_DEPS_QUOTA_GB` | 20 | 持久依赖缓存容量目标，GiB |
| `AUTOREPRO_DEPS_MIN_FREE_GB` | 2 | 缓存所在磁盘最低余量，GiB |
| `AUTOREPRO_DEPS_INSTALL_RESERVE_GB` | 2 | 新安装下载、解压和复制的额外空间预留，GiB；这是可配置策略，不是包体积预测 |
| `AUTOREPRO_DEPS_KEEP_DAYS` | 30 | 闲置保留天数 |
| `AUTOREPRO_DEPS_KEEP_RECENT` | 2 | 保护最近使用的就绪环境数；设为 0 可取消这项保护 |

pip 下载、解压和下载缓存统一放在依赖根的 `.pip-temporary/` 所属目录，安装子进程返回并完成清理后释放。单独调用底层执行器时，异常终止的安装临时目录在后续维护中检查归属和锁，满 24 小时后回收；完整复现入口还会在取得独占锁后清理遗留实验环境。不会删除其他应用的系统临时文件。安装前要求磁盘余量达到最低余量与额外安装预留之和，大型 CUDA 环境可调高预留。论文、作者源码、数据、模型、报告和训练日志沿用各自的保留策略。

`deps-prune` 使用同一保留策略，默认仅预览，加 `--yes` 才删除；可用 `--keep-recent 0` 取消最近环境保护。维护记录列出删除／保留原因、文件字节数和维护前后磁盘余量；在用、被引用或最近保留的环境使容量仍超限时会明确记录原因，CLI 返回 2。历史记录页显示容量目标、当前磁盘余量和最近维护结果。资源 CLI 的 `--data-root` 应放在子命令前，例如：

```bash
python scripts/resource_cli.py --data-root E:/AutoReproducer-data status
```

`archive PAPER_ID` / `restore ZIP路径` 用于单篇资源归档与恢复；`prune --yes` 在资源超配额时先归档再清理。**资源归档只包含已登记的代码、数据、本地权重及清单，不包含报告、运行日志或依赖环境。**目录迁移、归档校验和 Windows 文件占用处理见 [文件与存储管理](AutoReproducer/docs/file_management.md)。

## 开发与测试

### 项目结构

```text
AutoReproducer/                  # 仓库根目录（外层）
├── README.md                    # 项目说明
├── LICENSE                      # MIT 许可证
├── .github/workflows/           # 持续集成
└── AutoReproducer/              # 应用目录，命令默认在这里执行
    ├── app.py                   # Streamlit 页面与任务入口
    ├── frontend/                # 页面样式、后台任务、历史与报告展示
    ├── src/
    │   ├── orchestrator.py      # 流程编排与入口分流
    │   ├── agents/              # 阅读、查找、环境、执行、报告等 Agent
    │   ├── llm/                 # OpenAI 兼容 API 客户端与 Mock 响应
    │   ├── repository_profiles.py # 固定预设注册
    │   ├── dependency_cache.py   # 依赖缓存与健康检查实现
    │   ├── dependency_management.py # 依赖清单、复用与回收
    │   ├── project_paths.py      # 统一数据路径
    │   └── resource_manager.py   # 资源下载、清单、归档与恢复
    ├── scripts/                 # 实验与资源 CLI
    ├── tests/                   # pytest 回归测试
    ├── docs/                    # 专题说明与历史实验记录
    ├── examples/papers/         # PDF 样本与来源清单
    ├── requirements.txt         # 应用依赖
    └── requirements-test.txt    # 测试依赖
```

### 运行测试

在与应用相同的虚拟环境中执行：

```bash
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest tests/ -q -ra

# Windows 另验证 UTF-8 模式
python -X utf8 -m pytest tests/ -q -ra
```

[GitHub Actions](.github/workflows/tests.yml) 配置为 Ubuntu / Windows × Python 3.11 / 3.12，Windows 另跑 UTF-8 模式。回归测试无需真实 LLM API；CI 配置和历史通过记录不能代替当前提交的测试结果，也不能证明真实论文训练完成。当前测试已迁移到作者代码执行和报告流程；本地全量结果、Windows 并发修复及真实 PDF 验证状态见 [测试跟进记录](AutoReproducer/docs/test_followup_20261013.md)。

### Python 编程接口

真实 PDF 入口示例，使用与网页相同的本地配置加载器：

```python
from src.local_llm_settings import load_local_llm_settings
from frontend.llm_config import resolve_llm_config
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator

load_local_llm_settings()
llm = LLMClient(mock_mode=False, **resolve_llm_config())
runner = Orchestrator(llm_client=llm, mock_mode=False, use_docker=False)
result = runner.run({"pdf_path": "paper.pdf"})
print(result["state"])
print(result["data"].get("validation"))
```

运行前准备可读取正文的 PDF，并按“模型配置”设置实际使用的服务和模型。该入口同样只报告当前执行结论，不会因省略网页而启用历史数值验收或自动优化。

## 更多文档

| 文档 | 内容 |
| --- | --- |
| [PDF 样本归档](AutoReproducer/examples/papers/README.md) | 论文来源、版本与 SHA-256；收录不代表运行通过 |
| [依赖环境管理](AutoReproducer/docs/dependency_management.md) | 应用清单、缓存复用条件及回收机制 |
| [文件与存储管理](AutoReproducer/docs/file_management.md) | 数据路径、迁移、归档恢复及 Windows 文件规则 |
| [当前执行流程与已知待办](AutoReproducer/docs/execution_without_reviews_20261012.md) | 移除审核后的 AdaGNN 实际运行记录、能力边界与回归情况 |

历史实验资料：[DLinear](AutoReproducer/docs/dlinear_reproduction_result.md) · [SIREN](AutoReproducer/docs/siren_quick_result.md) · [Neural ODE](AutoReproducer/docs/neural_ode_result.md) · [独立 PDF 测试](AutoReproducer/docs/independent_pdf_acceptance.md)。这些文档保留当时的环境、指标和操作步骤；其中的旧审核开关、数值验收和优化结论不代表当前版本功能。

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源，版权归属为 © 2026 Yukikazct。
