# AutoReproducer：基于 GitHub 多文件仓库的论文复现实施方案

原始检查日期：2026-10-03，检查版本：de0fa6a。进展更新：2026-10-07。DLinear单项论文实验已真实复现通过，Neural ODE与通用仓库扩展仍是后续计划。

本文保留原始审查和后续实施方案，并记录现已交付的DLinear完整实验。2026-10-07新运行已通过生产网页后台入口完成四阶段公开分析、固定作者仓库的ETTh1多变量336→96训练/test、独立协议与指标复算和允许的结果摘要解释；没有fork、提交或推送代码。

> 新已验证结果：`repository_e9408141e3dd44319e854bbdf4a70c0c`使用默认`dlinear_etth1_reference`，按作者最多10轮/patience 3协议在第7轮早停；MSE 0.3841443955898285、MAE 0.40471312403678894，相对Table 2的0.375/0.399偏差2.44%/1.43%，通过项目事前5%相对容差（非论文阈值）。16个公开来源、113份固定源码、真实数据、运行协议和独立复算均通过；deepseek-chat四阶段各1次、结果摘要解释1次，共5次真实API调用、零修正重试。数值状态`reproduced`与分析状态`completed`分别记录，结论仅覆盖该一个实验。详见[新完整报告](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report.md)、[报告与图片ZIP](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report_with_figures.zip)、[结果记录](dlinear_reproduction_result.md)及[重跑步骤](dlinear_testing.md)。

旧运行`repository_66efc14890e34cb196a07a7d2bcb78fe`的[单次公开协议API基线报告](../data/runs/repository_66efc14890e34cb196a07a7d2bcb78fe/report.md)保留；新运行重新训练得到相同指标和第7轮早停结果。前期多Agent引用/映射/环境说明被拒绝的记录保留为真实诊断，不将它们写成通过或跳过门控。

## 1. 检查结论与本轮目标

2026-10-03原始审查定位的断点是单文件执行没有贯通“论文→仓库→实验入口→多文件执行→论文指标”。2026-10-07已通过`repository_profiles.py`、`repository_reproduction.py`、`repository_runner.py`、`repository_validation.py`接通DLinear纵向链路，网页与CLI共用`Orchestrator.run`。下表保留原始问题作为通用化改造依据，不代表这些断点仍全部存在于DLinear路径。

下一版本的最小交付定义为：用户选择论文预设或输入 PDF + 已知 GitHub URL，系统获取固定版本的完整仓库，在独立工作区按实验计划准备数据、训练、评估，展示真实日志、曲线、指标和来源。支持两篇已适配论文；陌生仓库先产出候选计划，不能直接宣称普遍支持。

### 1.1 原始检查发现的问题（2026-10-03）

| 优先级 | 原始证据 | 对目标的影响 | 建议动作 |
|---|---|---|---|
| P0 | src/agents/code_executor.py:509 读取 code 字符串；:532 生成代码；:1242、:1684 写入 run.py | 官方仓库的 models、utils、数据加载器和配置没有作为执行对象被使用 | 新增仓库执行路径，保留目录结构、cwd、参数和依赖 |
| P0 | src/orchestrator.py:145 调用资源下载，但执行器未消费 storage.fetched.code.path | “找到并下载仓库”没有转化为“运行该仓库” | 显式传递 RepositorySnapshot、ExecutionPlan 和工作区路径 |
| P0 | frontend/backend_pipeline.py:184 另写阶段循环，未调用 Orchestrator.run | 网页路径没有同步资源下载、镜像构建、验证修正、工作区准备和阶段计量等钩子 | 前后端共用一个编排入口，UI 通过事件订阅进度 |
| P0 | ResultValidator 仍从 stdout 猜指标；失败门只拦部分情况 | 可能产出错误的“复现成功” | 成功门控 + 结构化指标 + 按实验定义指标契约 |
| P1 | paper_reader.py:74 只给 LLM 前 3000 字符，:101 只透传前 2000 字符 | 常漏掉实验表格、附录、超参数和仓库链接 | 按页/章节提取与当前实验相关的信息，保留页码与原文证据 |
| P1 | Reader 输出 code_url，Finder 优先读 code_repo_url；Orchestrator 输入白名单也未透传 preferred_repo_url | 用户明确指定的仓库可能没有进入资源发现优先路径 | 统一仓库字段并为显式 URL 增加贯通测试 |
| P1 | ResourceManager.fetch_code 对已有目录直接复用；revision pin 失败允许降级 HEAD | 不同版本可能复用同一缓存，结果难以比较 | 缓存键含 URL + resolved SHA；论文验证模式禁止静默换版本 |
| P1 | Optimizer 默认哈希模拟；app.py:895 启动后台时未传工作区 | 开启真实 LLM 并不等于真实优化；目前优化目标也是单文件 run.py | 首版真实 Demo 关闭优化，完成复现后再接入多文件优化 |
| P1 | evidence、BeamUCT、TrialLedger 等存在独立模块，但主链路未见对应集成；Spec/holdout 未在默认入口注入 | README 的能力描述超出默认端到端路径 | 文档标注“模块已实现/主流程已接入/真实实验已验证”三种状态 |

### 1.2 实测记录

- Python 3.12.2，执行现有 tests：**691 passed，1 skipped，1 warning**。首次受限运行有 6 个本地 HTTP 测试服务器绑定端口错误；允许回环端口后全量重跑得到上述结果。
- 上述为2026-10-03测试基线。2026-10-07较早批次完整套件为 **913 passed，4 skipped，1 warning**（PyPDF2旧依赖弃用），后续网页/仓库/CLI相关 **70项针对性检查通过**，`git diff --check`通过；这些是保留的历史工程检查。新的四阶段加结果解释共5次真实API、作者训练与独立复算另见本页新运行证据，不能将历史913项作为新批次完整测试数。Neural ODE尚未在本轮训练。
- 新多Agent批次的最终完整回归为 **1138 passed，4 skipped，1 warning**（PyPDF2弃用警告）。该工程回归与新运行的真实API、作者训练和独立指标核验分别记录。
- 额外诊断得到以下反例，记录在 [validator_probes_20261003.json](review/validator_probes_20261003.json)：
  - loss: 1.2e-4 被解析为 1.2。
  - 同时存在首轮 loss=1.0 和最终 loss=0.01 时，冒号形式匹配可能取首轮值。
  - 声明 accuracy 和 f1，只输出匹配的 accuracy，本地比较仍返回 match=true。
  - MSE 声明 0.5、实际 50，被通用百分比归一化误判一致。
  - 使用固定返回 match=true 的测试 LLM 时，退出码 1 但已打印匹配 accuracy 的失败运行，最终可被判 reproduced。这个反例验证了确定性门控缺口，不是对真实 LLM 输出概率的推断。

## 2. 两篇具体的 Demo 论文

| 项目 | 第一篇：DLinear | 第二篇：Neural ODE |
|---|---|---|
| 论文 | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) |
| 作者仓库 | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) |
| 本轮范围 | DLinear 在 ETTh1 的一个预测配置 | 官方二维螺旋 ODE 拟合示例 |
| 入口 | run_longExp.py；参数参考 scripts/EXP-LongForecasting/Linear/etth1.sh | examples/ode_demo.py |
| 多文件调用 | exp → models → layers / data_provider / utils | examples → torchdiffeq 包 → 内部 ODE 求解器 |
| 数据 | ETTh1，约 2.6 MB，官方时间切分 | 示例通过已知微分方程生成轨迹，无外部数据下载 |
| 展示 | 真实值/预测值曲线、MSE、MAE、训练过程 | 真实/学习轨迹、相图、向量场、误差曲线 |
| 可声称的结论 | 已完成作者完整训练、协议核验与独立指标复算；ETTh1 336→96选定实验数值复现通过 | 待适配：官方示例与方法行为复现；不等价于论文全部实验或表格复现 |

选择依据：DLinear 已有本地历史产物；Neural ODE 官方示例输入简单、天然可视化，源码会在没有 CUDA 时选择 CPU。两者都可检验完整仓库执行能力。运行时长须在目标机器实测，不能把“支持 CPU”理解为已保证几分钟收敛。

已核实的版本来源：

- DLinear 历史缓存 commit：0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6。
- torchdiffeq 本次 GitHub 查询的 master commit：657943acefa826ef04c025ebeb1ff5e9d60dc268。运行前按此 SHA 获取并再校验入口文件；本次查看的是查询当时的 master 示例。
- DLinear 历史报告 data/reports/dlinear_cache_verified.md 记录 1 epoch、seq_len=96、pred_len=96，MSE≈0.4090、MAE≈0.4168；这是历史冒烟记录，不能当作本次重跑结果或论文目标。
- 原始审查时历史experiment_profile/execution_plan未贯通主线；本轮已重新接入固定预设、完整仓库执行和证据判定。旧单次API完整基线来自`repository_66efc14890e34cb196a07a7d2bcb78fe`，新四阶段加结果解释的完整结果来自`repository_e9408141e3dd44319e854bbdf4a70c0c`；历史1轮数据保留为旧冒烟记录。

## 3. 从 GitHub fork 到固定版本工作区

**Fork 用来保存你自己的修改和实验分支；clone 用来获取真正执行的完整文件。**自动复现公开仓库不强制要求先 fork。系统应同时支持直接上游 URL 与用户 fork URL；不在每次运行时重复 fork。

### 3.1 人工准备一次 fork（示例操作，尚未执行）

安装并登录 GitHub CLI 后，在你选定的开发目录执行。下列命令会创建 GitHub fork，因此只在你决定开始实施时运行。

```bash
gh auth login
gh repo fork cure-lab/LTSF-Linear --clone=false --remote=false
gh repo fork rtqichen/torchdiffeq --clone=false --remote=false

DEMO_GITHUB_USER="$(gh api user --jq .login)"
mkdir -p demo-repos
cd demo-repos

git clone "https://github.com/${DEMO_GITHUB_USER}/LTSF-Linear.git"
git -C LTSF-Linear remote add upstream https://github.com/cure-lab/LTSF-Linear.git
git -C LTSF-Linear fetch upstream 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
git -C LTSF-Linear switch -c codex/reproduce-etth1 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6

git clone "https://github.com/${DEMO_GITHUB_USER}/torchdiffeq.git"
git -C torchdiffeq remote add upstream https://github.com/rtqichen/torchdiffeq.git
git -C torchdiffeq fetch upstream 657943acefa826ef04c025ebeb1ff5e9d60dc268
git -C torchdiffeq switch -c codex/reproduce-spiral 657943acefa826ef04c025ebeb1ff5e9d60dc268
```

origin 保存你的修改，upstream 用于追踪作者版本。baseline_sha 记录作者 commit；经过兼容修复后另记 candidate_sha 或 patch_sha256。报告必须同时显示这两个版本。更新上游不能静默改变已冻结的实验。

### 3.2 系统运行时的目录模型

```text
data/
  repos/<repo-key>/<commit>/             # 固定版本源码缓存
  datasets/<dataset>/<version>/          # 真实数据缓存与校验和
  runs/<run-id>/
    workspace/repo/                      # 每次运行独立的完整源码副本
    artifacts/                          # 指标、曲线、checkpoint 索引
    logs/<step-id>.stdout.log
    logs/<step-id>.stderr.log
    patches/                            # 每轮修复补丁与决策
    repository.json                     # upstream/fork URL、版本、许可证
    experiment_spec.json                # 论文实验定义与来源
    execution_plan.json                 # 实际命令与依赖顺序
    environment.json                    # Python、平台、锁定依赖、镜像 digest
    result.json
    report.md
```

repo-key 由规范化 URL 生成；run-id 使用 UUID 或时间戳加随机后缀。工作副本排除 .git 凭据和缓存大文件；数据用只读挂载。首次可用复制工作区实现，失败时丢弃本轮副本，避免修改共享缓存。Git submodule / LFS 如果被入口依赖，单独固定版本并准备；首版未支持时返回明确原因。

## 4. 如何“根据论文一步步复现”

### 第一步：把论文目标写成可核验的实验定义

先选论文中的一个具体表格行、图或作者示例，填清：论文版本、页码/表号、方法、数据集版本、切分、输入长度、训练预算、seed、评估命令、指标单位、参考数值和误差容限。未知值保持 null，不由 LLM 猜测补齐。

将 PaperReader 的输出扩展成 ExperimentSpec，并保留原文摘录与定位。本轮只提取两篇选定实验的章节即可，不需要先完成通用 PDF 理解系统。一次人工核对后固定 spec hash；后续训练不能自动改参考答案。

### 第二步：理解完整仓库并建立论文—代码对应表

新增 RepositoryInspector，先读取目录树、README、依赖文件、配置文件、作者脚本与入口参数，再按本地 import 关系补读相关文件。忽略数据、权重、图片、.git 和大文件，不把全仓库一次塞进提示词。

产出 repository_map.json，例如：

| 论文要素 | DLinear 对应文件 | 应核查什么 |
|---|---|---|
| 模型/分解方法 | models/DLinear.py 及其导入模块 | 分解与线性层是否来自官方实现 |
| ETTh1 切分与标准化 | data_provider/data_loader.py | 时间切分边界、scaler 仅使用训练集拟合 |
| 训练和早停 | exp/exp_main.py、utils/tools.py | epoch、损失、早停、checkpoint 选择 |
| 最终指标 | exp/exp_main.py、utils/metrics.py | test split、MSE/MAE 的计算与输出位置 |
| 完整运行参数 | scripts/EXP-LongForecasting/Linear/etth1.sh | 输入窗口、预测长度、学习率、batch size |

首版将这张映射写进两个实验预设。LLM 可解释和补充候选入口，最终执行字段要通过路径存在性、参数和依赖检查。

### 第三步：产出命令计划，保持多文件结构

执行计划由 `src/execution_plan.py` 的 `build_plan` 产出。核心输出已从 code 字符串改为 ExecutionPlan，包含 repo_root、步骤顺序、argv、cwd、环境变量、超时、步骤类型 kind、前置步骤 depends_on、前置产物 requires 与声明产物 artifacts。计划是冻结的 dict，绝不生成 shell 文本；审阅者按阅读顺序编写步骤，依赖只能指向更早的步骤，因而循环依赖不可表达。

最小结构示例（与 `build_plan` 的输出逐字段对应；DLinear 预设走作者合并入口，只用前两步，
`evaluate` 一步是拆分 train/eval 时的写法，SIREN 与 Neural ODE 预设即为该形态。验证级别与
容差属于 `experiment_spec.json`，不在执行计划里）：

```json
{
  "version": 1,
  "mode": "repository",
  "profile": "dlinear_etth1_reference",
  "spec_sha256": "ab91d7f0860a09644ac5c20d5b64ca01188a1bf4a313d47cd4448323efd33b5d",
  "workspace": "/workspace/repo",
  "repo_root": "/workspace/repo",
  "repository": {
    "url": "https://github.com/cure-lab/LTSF-Linear.git",
    "revision": "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
  },
  "dataset": {"name": "ETTh1", "sha256": "f244b507c20580401cd847ba5829a9f53a704e39ee57a59e5c50d39ef0016afa"},
  "limits": {"repair_rounds": 2, "llm_calls": 20, "total_seconds": 2400},
  "steps": [
    {
      "id": "import_check",
      "kind": "check",
      "argv": ["python", "-c", "from exp.exp_main import Exp_Main; import models.DLinear"],
      "cwd": ".",
      "depends_on": [],
      "timeout_s": 60
    },
    {
      "id": "train_and_eval",
      "kind": "train",
      "argv": ["python", "-u", "run_longExp.py", "--is_training", "1", "--model_id", "ETTh1_336_96", "--model", "DLinear", "--data", "ETTh1", "--root_path", "./dataset/", "--data_path", "ETTh1.csv", "--features", "M", "--seq_len", "336", "--pred_len", "96", "--enc_in", "7", "--train_epochs", "10", "--patience", "3", "--batch_size", "32", "--num_workers", "0", "--learning_rate", "0.005", "--itr", "1"],
      "cwd": ".",
      "env": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"},
      "depends_on": ["import_check"],
      "timeout_s": 1800,
      "artifacts": [{"path": "results/*/checkpoint.pth"}, {"path": "results/*/pred.npy"}]
    },
    {
      "id": "evaluate",
      "kind": "eval",
      "argv": ["python", "-u", "evaluate.py", "fit"],
      "cwd": ".",
      "depends_on": ["train_and_eval"],
      "requires": [{"path": "results/ETTh1_336_96/checkpoint.pth"}],
      "timeout_s": 90
    }
  ]
}
```

依赖准备由 EnvBuilder 在这些步骤之前完成。训练和评估如果由作者入口一次完成，就保留该入口；若分 train.py / eval.py，就明确列出两个步骤及 checkpoint 传递。超时值是初始预算建议，需要在目标机器校准。

`requires` 与 `artifacts` 的分工是刻意的：`artifacts` 是这一步跑完后顺带收集的证据，路径可以是通配（作者输出目录常含运行时间戳），匹配不到只记原因，绝不改变该步的结论；`requires` 是下一步启动前要确认存在的产物，必须是具体路径，缺失或为空即阻止该步执行。否则训练“成功退出却没写出 checkpoint”会被读成一次真实评分失败，而它的成因在训练那一步。

执行采用线性停止：第一条必需步骤失败后，其后的步骤一律不启动，登记为 `skipped` 并写明 `blocked_by`。被挡住的步骤计入失败结论，报告里渲染成“未执行（因 X）”，不会因为它没跑就变成可选跳过。`required: false` 只用于 import 探测这类允许失败的步骤，它自身失败不阻断下游，但下游若被别的失败挡住仍算失败。

### 第四步：准备适配平台的依赖和真实数据

- 修改 EnvBuilder：优先处理 requirements、pyproject、environment 文件，生成可追溯环境。环境缓存键至少含源码依赖 hash、Python 版本、OS/架构、CPU/CUDA 类型。
- DLinear原始requirements为torch==1.9.0、README为Python 3.6.9。本轮实际通过Python 3.12.2/macOS ARM64/CPU兼容环境：torch 2.5.1、numpy 1.26.4、pandas 2.2.3、scikit-learn 1.5.2、matplotlib 3.9.2，来源与偏差已记录。当前预设固定CPU；Windows CUDA建议在WSL2中准备匹配环境并另行适配验证，不会自动使用显卡，也不能承诺逐位相同指标。
- torchdiffeq 的 setup.py 声明 torch>=1.5.0、scipy>=1.4.0；示例另用 numpy，可视化用 matplotlib。应安装工作区中的项目源码，使 examples 导入这一版本的本地包。
- 数据缓存必须记录下载来源、SHA-256、大小和切分信息。下载失败返回 data_unavailable；仓库复现模式不自动用合成数据替代真实 ETTh1。
- 安装、下载允许网络；正式实验优先在准备好的容器中执行，固定资源与超时。复用现有 Docker 加固和依赖准备能力，但要扩展为挂载整个仓库。

### 第五步：在整个仓库上执行

新增 RepositoryRunner，底层运行完整 argv，cwd 固定为仓库内相对路径。Python 自己解析 models、utils 和相对文件路径，无须拼接成单文件。

```python
# 接口草案
snapshot = resource_manager.fetch_repository(request.repository)
workspace = workspace_manager.materialize(snapshot, run_id)
repo_map = inspector.inspect(workspace)
spec = reader.build_experiment_spec(request.paper, request.profile)
plan = planner.create(repo_map, spec, request.profile)
environment = env_builder.prepare_repository(workspace, plan)
execution = runner.run(plan, workspace, environment, on_event)
validation = validator.validate(spec, execution)
report = reporter.render(spec, execution, validation)
```

执行器应提供：逐行 stdout/stderr 事件、退出码、实际 argv/cwd、起止时间、超时/取消、资源信息、每步产物清单。缓存/失败后重跑的依据为源码 hash + spec hash + 输入 artifact hash；源码或参数变更后，不复用旧指标。

优先支持 argv 列表，避免 shell 字符串拼接。确实需要官方 shell 脚本时可计划为 ["bash", "scripts/xxx.sh"]，记录它实际执行的子任务。DLinear 官方 etth1.sh 会连续执行四种预测长度，首版只提取一个目标命令，避免无意扩大实验。

### 第六步：按文件和报错定点修复

新增 RepositoryRepairer。循环为：失败步骤 → traceback 文件/行号 → 相关源码/依赖/配置 → 结构化多文件 diff → 校验 → 应用到候选工作区 → 从受影响步骤重跑。

补丁响应格式可为：

```json
{
  "reason": "已定位的兼容性问题与修复理由",
  "patches": [
    {"path": "models/example.py", "base_sha256": "...", "unified_diff": "..."},
    {"path": "train.py", "base_sha256": "...", "unified_diff": "..."}
  ],
  "rerun_from": "import_check"
}
```

复用 PatchPolicy 与 workspace_snapshot，增加真实路径/symlink 检查、base hash 检查、git apply --check 等补丁预检。一次候选的全部文件应原子应用，失败整体回滚。不能只恢复某个 run.py 而留下其他文件改动。

把兼容性修复和算法优化分开：首版仅修路径、依赖/API 兼容、运行参数，不改论文方法、数据切分、指标实现或目标数值。论文评测文件保持只读；若确有评测代码兼容问题，使用预先审查的版本化补丁并记录差异。现有 PatchPolicy 会保护部分配置/数据相关文件，不能为了跑通绕过规则；优先由固定实验预设和环境适配器处理。

每轮保存补丁、理由、改动文件、失败日志和重跑结果。每个任务最多 2 轮自动修复，预算耗尽输出失败报告，保留可继续工作的工作区。

### 第七步：从实验产物验证，而不是从任意文本判成功

建立统一 metrics.json 协议：name、value、unit、split、stage、seed、来源文件、源码/spec hash。DLinear 读取本次 test 阶段产物；Neural ODE 读取最终全轨迹误差。不能把目录中历史 metrics.npy 或第一次训练 loss 当成本次最终结果。

对只能输出文本的仓库，使用实验专属解析器，支持正负数、科学计数法、NaN/Inf 检查，并明确取哪个阶段的哪一行。百分比只按显式单位换算；MSE/MAE 不参与百分比猜测。参考值缺失就不能判断论文数值一致。

结果分层建议：

| 状态 | 所需证据 |
|---|---|
| failed | 必需步骤失败、超时、数据缺失或指标无效；保留具体原因 |
| smoke_passed | 固定版本的真实仓库可运行，退出码正确，输出结构与有限数值有效 |
| experiment_completed | 选定实验按指定配置完成，数据/训练/评估产物齐全 |
| reproduced | 配置、数据切分和评估口径匹配论文目标，所有必需指标满足事前定义的容差 |
| inconclusive | 已完成执行但缺参考值、配置不完全一致或证据不足 |

依据实际证据展示结论：当前DLinear固定单项已通过完整协议和数值核验；缺参考数值的其他实验仍只能显示证据不足。任何必要步骤exit_code非零均不得判reproduced。LLM解析公开协议和解释偏差，不覆盖确定性判定；PaperGuru reproduction_score不能替代论文性能指标。

## 5. 两篇论文的具体执行清单

### 5.1 DLinear：已完成第一个纵向闭环

1. 从 fork 或上游获取固定 SHA 的完整代码。
2. 默认`dlinear_etth1_reference`完整作者实验；`dlinear_etth1_smoke`仅保留为开发诊断。
3. ETTh1 可复用历史已校验数据来源：
   https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv
   本地历史 SHA-256 为 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066。复用前重新校验；不能用历史 manifest 代替文件检查。
4. 将文件挂载到仓库 dataset/ETTh1.csv。先测试 exp.exp_main 与 models.DLinear 的 import。
5. 使用完整输入336、预测96、最多10轮、patience 3、Adam初始lr 0.005和batch 32。本次连续完成7轮后官方早停，加载验证集最优checkpoint再test。
6. Table 2目标MSE 0.375、MAE 0.399；本次实际0.3841443955898285/0.40471312403678894，分别相差2.44%/1.43%，均在项目既有5%相对容差内，该容差非论文阈值。
7. 实际Namespace、固定113份源码、数据SHA、8209/2785/2785窗口、完整训练与唯一checkpoint/预测数组均通过独立协议核验。预测数组shape为(2785,96,7)，独立重建真实测试标签并复算指标，与作者日志在1e-6内一致；报告包含两张真实实验图。
8. 保留原入口seed 2021；[作者明确论文只跑一个seed](https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601)，无需将其他消融的多次实验规则套入本项。
9. 新运行由PaperReader、ResourceFinder、EnvBuilder、Verifier读取16个真实公开来源，逐项引用论文/作者script/default；四阶段各1次且零修正，全部通过后才执行官方训练。允许后ResultValidator用1次API解释MSE/MAE、完成轮数和核验状态摘要，共5次；本地指标与确定性判定不由API替代。原始日志、`repository_analysis.json`、`result_analysis.json`、协议核验与独立指标复算分别落盘；旧单次API基线保留。
10. 网页默认真实模式，官方预设使用本地CPU而不依赖Docker；CLI重跑使用 `--llm-review --analysis-mode multi_agent --result-review`。四阶段可各修正最多一次，报告以真实客户端差量统计调用，不承诺每次都是5次。报告图片存同目录相对companion资产，分享用报告图片ZIP。完整重跑步骤见[dlinear_testing.md](dlinear_testing.md)。

验收已通过：网页/CLI执行官方`run_longExp.py`和跨文件导入，可追溯源码与数据SHA、实际参数、完整协议、独立复算及结果。结论仅为DLinear/ETTh1 336→96选定实验数值复现；Neural ODE、多文件自动修复与陌生仓库通用化仍须后续实现。

### 5.2 Neural ODE：验证第二个仓库可以复用同一执行器

1. 获取上文固定 SHA，安装完整 torchdiffeq 工作副本，准备 torch、scipy、numpy、matplotlib。
2. 先执行一个最小真实步骤（以下是作者现有 CLI，可在已准备环境的仓库根目录运行）：

```bash
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg python examples/ode_demo.py \
  --niters 20 --data_size 100 --batch_time 10 --batch_size 10 --test_freq 10
```

3. Demo 拟合预设可从 200 次迭代开始校准；参考作者默认是 2000 次、data_size=1000。若短跑没有收敛，延长训练并记录预算，不能仅挑选好看的 seed。
4. 图像输出加 --viz，并用 MPLBACKEND=Agg；收集 png/ 目录下本次生成的图。首次适配验证无 GUI 容器下保存行为。
5. 原示例没有 seed 参数；增加一个有限的复现补丁，固定 Python/NumPy/PyTorch 种子，并额外导出初始误差、最终误差、轨迹数组和 JSON。保持网络、优化器、求解器与目标方程不变。
6. 作者 Total Loss 是全条轨迹上的平均绝对误差，这条轨迹也用于采样训练片段；它不是独立留出测试。报告标为“轨迹拟合误差”，不要写“测试泛化精度”。
7. smoke 验证退出成功、loss 有限、产物齐全；方法演示验证轨迹拟合效果与相图。工程误差阈值在预实验之后事前固定，标注自定义标准；没有论文对应数值就不给 reproduced 标签。

验收：无需改主编排器，新增第二个预设与产物解析器即可运行；保留整个 torchdiffeq 包的执行路径。此示例证明多文件执行器可复用，不证明支持所有论文。

## 6. 对当前项目的具体改造位置

| 文件/模块 | 具体改动 | 完成判据 |
|---|---|---|
| src/orchestrator.py | 唯一编排入口；贯通显式 repo URL、revision、profile、mode；提供事件回调；接入仓库计划 | 网页与脚本得到同样的步骤与结论 |
| frontend/backend_pipeline.py | 删除重复业务循环，调用 Orchestrator；只适配 ProgressStore 事件与报告持久化 | 资源下载、修复重试、计量不再分叉 |
| src/resource_manager.py | 固定 SHA 缓存、URL/version 校验、独立运行工作区；保留 provenance | 更换 SHA 不命中旧仓库；pin 失败不能验证成功 |
| src/reproduction/contracts.py（新增） | ReproductionRequest、RepositorySnapshot、ExperimentSpec、ExecutionPlan、StepResult、MetricResult | 输入/输出可序列化并在边界校验 |
| src/reproduction/repository_inspector.py（新增） | 目录/入口/依赖/import 图与论文代码映射 | 两个仓库能得到正确入口和相关文件 |
| src/reproduction/execution_planner.py（新增） | 预设优先生成确定性步骤；未知仓库输出待核查计划 | 计划可落盘和重放，不依赖隐含对话上下文 |
| src/reproduction/repository_runner.py（新增） | 整仓库挂载、argv/cwd、多步依赖、日志流、取消/超时、产物采集 | 跨文件 import、相对路径、train/eval 分离都可运行 |
| src/reproduction/repository_repairer.py（新增） | 错误定位、多文件补丁、原子校验、回滚与重跑 | 失败候选不能污染共享源码或下一候选 |
| src/agents/code_executor.py | 加显式 repository/script 路由；抽取现有容器/依赖工具供复用 | repository 模式绝不自动改走单文件生成 |
| src/agents/env_builder.py | 从仓库声明准备兼容环境，记录偏差并锁定实际依赖 | 全新环境可重建两个案例 |
| src/agents/result_validator.py | 修复已确认误判；接入独立状态、指标契约与产物来源 | 失败/缺指标/无参考值均不能误判成功 |
| src/agents/report_generator.py、app.py | 展示文件树、入口/参数、步骤日志、修复 diff、真实指标、复现级别 | Demo 可解释“运行了什么、依据什么下结论” |
| profiles/（新增） | 两篇论文的 smoke/reference 配置、论文来源、环境适配和解析规则 | 新增同类仓库主要修改预设而非编排器 |

现有 ResearchSpec 偏优化验收，不要用它承载所有运行字段；ExperimentSpec 专门描述论文实验，可以复用其冻结/hash 思路。现有快照、审计、依赖解析和图像采集模块优先复用，避免整体重写。

首版配置使用 JSON 可不增加依赖；若选 YAML，需将 PyYAML 显式加入项目依赖。

## 7. 按依赖推进的实施顺序

以下按一名开发者、已有 Docker/开发环境估算 **10–12 个工作日**；不含网络受阻、环境迁移和长时间参考训练。每阶段先达成验收，再进入后续工作。

| 阶段 | 预计 | 工作 | 可检查交付物 |
|---|---|---|---|
| A：建立人工基线 | 1–2 天 | fork/clone 两个仓库；手工跑 smoke；固定版本、命令、数据与环境 | 两份无需 LLM 的真实运行记录，暴露平台兼容问题 |
| B：单一流程 + 第一个仓库 | 3 天 | 修复验证 P0；统一编排；实现 contracts 与 RepositoryRunner；接入 DLinear 预设 | 网页从选择论文到报告的首个仓库闭环 |
| C：第二仓库 + 文件理解 | 2 天 | 实现 Inspector/Planner 最小版本；接 Neural ODE 与专属解析器 | 不改主循环就能切换第二篇论文 |
| D：有限自修复 | 2 天 | 多文件 patch、运行快照、两轮重试、错误分类、预算/取消 | 可控兼容故障得到最小补丁；不可修复故障有完整报告 |
| E：真实回归与演示 | 2–3 天 | 干净工作区重跑；完善来源/状态/图表；锁定依赖；录制演示 | 两份真实报告、命令清单、5–8 分钟 Demo |

**前三个开发任务建议直接按以下顺序建 Issue：**

1. 统一 run_pipeline_core 与 Orchestrator.run，贯通 repo_url/revision/profile，补前后端一致性测试。
2. 为 RepositoryRunner 实现完整目录挂载与 cwd + argv 执行，用 DLinear 固定命令跑出真实 MSE/MAE。
3. 修正 ResultValidator 误判并接入 metrics.json，使 smoke、完成实验和论文数值复现有明确区分。

A 阶段可以先通过人工命令完成；B 阶段的验证修复与执行器联调同时推进。不要在真实案例跑通之前新增搜索算法或扩展很多 Agent。

## 8. Demo 验收清单

### 必须通过的工程检查

括号内为当前覆盖该条的测试文件；标注「未覆盖」的条目尚未有测试。

- 多文件 fixture：入口导入两个本地模块，读取相对配置，生成产物；路径含空格/中文仍能运行。（`test_repository_runner.py::test_multi_file_fixture_survives_spaces_and_chinese_in_paths`、`::test_multiple_files_import_and_nested_cwd_are_preserved`）
- 支持分离 train/eval 步骤，checkpoint 缺失时停止；必需前置步骤失败时不能执行下游评估。（`test_execution_plan.py::test_method_profiles_split_train_and_eval_into_separate_steps`；`test_repository_runner.py::test_missing_prerequisite_product_blocks_the_consumer_even_after_success`、`::test_failed_prerequisite_blocks_downstream_steps_and_is_a_failed_verdict`、`::test_timeout_halts_dependent_steps`）
- 仓库模式不自动回退为单文件生成：Orchestrator 早期分流、CodeExecutorAgent 硬拒仓库预设、报告把被挡步骤写为「未执行」，三层互不依赖。（`test_repository_no_fallback.py`）
- 执行计划可落盘、可重放，字段与文档示例一致。（`test_execution_plan.py::test_the_documented_plan_example_is_buildable_by_the_real_validator`、`test_repository_runner.py::test_runner_accepts_a_validated_plan_dict_and_runs_it_in_authored_order`）
- repo URL/revision 从前端传到底层；缓存版本或数据 SHA 不符时拒绝复用。（`test_repository_reproduction.py::test_export_uses_fixed_commit_archive_and_leaves_dirty_cache_untouched`、`::test_corrupt_download_never_populates_real_cache_or_synthetic_fallback`）
- 对本次确认的五个验证反例添加回归测试；无指标、NaN/Inf、缺参考值、非零退出码不判成功。（`test_result_validation_gate.py`）
- 多文件补丁有一处不合法时全部不生效；训练失败后恢复上一个完整候选。（未覆盖：第六步 RepositoryRepairer 尚未实现，本批未涉及修复循环）
- 超时/取消能终止当前进程组或容器，不继续启动下一步；保留中间日志和失败报告。（`test_repository_runner.py::test_timeout_keeps_partial_logs_and_stops_descendants`、`::test_timeout_halts_dependent_steps`）
- 同样的 ExperimentSpec 通过网页和脚本运行，得到相同状态与配置记录。（未覆盖：网页路径依赖 streamlit，当前环境未安装）

### 真实案例验收

- DLinear 与 Neural ODE 各从独立工作区完成至少一次真实运行；核心结果可以按保存的命令重跑。
- 展示至少一个真实失败/修复案例，明确改动文件与依据；不人为篡改模型指标来演示成功。
- Demo 全程标注当前是 smoke、方法演示还是论文数值验证；历史运行回放需与现场运行区分。
- 记录真实调用次数、token、运行时间与超时预算；最多 20 次 LLM 调用/任务作为初始 Demo 配置，并在调用边界强制停止。当前各 Agent 共享客户端累计计数的差量逻辑需核对，避免重复累计。
- reference 结果若未达到论文指标，交付可解释的差异报告仍有价值；不将其改称“成功复现”。

## 9. Demo 之后再扩展

第一版先保留已有优化代码，但真实演示不开自动优化。两篇复现稳定后，再选择一个主指标，用真实测量的基线、相同数据切分和冻结评估器做多文件优化。

届时还需修正 Optimizer 对所有结果使用 baseline * (1 + reward) 的处理：MSE 等越小越好的指标不能用同一公式还原结果，实际测量值应直接来自执行器。之后再接入 holdout、TrialLedger 和 BeamUCT，评估它们是否在同等预算下提升成功率。

下一阶段最有价值的指标是：新增第三个仓库需要修改多少核心代码、成功完成多少个事前定义的实验、失败定位耗时，以及从干净环境重放的成功率。

## 10. 本次检查资料

- [测试记录](review/pytest_20261003.txt)
- [验证器反例](review/validator_probes_20261003.json)
- [DLinear 作者仓库](https://github.com/cure-lab/LTSF-Linear)、[ETTh1 作者脚本](https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/scripts/EXP-LongForecasting/Linear/etth1.sh)
- [Neural ODE 作者仓库](https://github.com/rtqichen/torchdiffeq)、[官方示例](https://github.com/rtqichen/torchdiffeq/blob/657943acefa826ef04c025ebeb1ff5e9d60dc268/examples/ode_demo.py)
- 本地原始方案：项目根目录 AutoReproducer-项目方案.pdf；历史实验：data/reports/dlinear_cache_verified.md、reports/pinn_reproduction.md。后者是 PINN 冒烟记录，未证明论文量级的数值复现。
