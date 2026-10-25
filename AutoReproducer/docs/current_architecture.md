# 当前架构与能力边界

本文描述当前代码的执行路径。普通作者执行仍为默认流程；普通真实 PDF 网页默认允许有界兼容性源码修复，可关闭或切换严格核验，两者互斥。历史实验和旧架构图中的结果不能代替当前版本的实际验收，智能优化仍未开放。

当前支持与验收目标是 Windows，应用使用 64 位 CPython 3.11 / 3.12。其他系统的兼容实现仍保留，不作为当前交付承诺。

## 入口与执行路径

网页使用 Streamlit，后台线程负责启动任务并写入 JSONL 进度。需要准备应用运行环境时，后台启动独立 Python worker；普通 PDF 和 ReZero 入口也使用独立 worker 加载本次代码。界面轮询进度，报告和结构化结果写入运行目录及历史归档。

```text
app.py
  → frontend/pipeline_entrypoint.py、repository_entrypoint.py
  → frontend/backend_pipeline.py
  → src/orchestrator.py
      ├─ 固定预设 / 精确匹配的论文标题或 PDF
      │   → RepositoryReproduction
      │       ├─ DLinear → RepositoryRunner
      │       └─ SIREN / Neural ODE / ReZero
      │           → MethodReproduction → RepositoryRunner
      ├─ 普通真实 PDF（未选择语料对照）
      │   → PaperReader → ResourceFinder
      │   → DiscoveredRepositoryReproduction
      │   → 固定仓库版本 → 规划作者命令 → 准备数据与环境
      │   → RepositoryRunner → 命令完成核验 → 论文指标比较
      │   ├─ source_repair=True + 报错自动定位或精确 Python 白名单
      │   │   → 实际失败 → 独立修复副本 → 最多两轮兼容补丁
      │   │   → 原命令/环境重跑 → 补丁来源核对与参考比较
      │   │   （与严格核验互斥；不产生严格成功结论）
      │   └─ 可选 strict_validation=True
      │       → 冻结实验协议与来源核验 → 作者完整实验
      │       → 独立进程重载模型与复算指标 → 所选实验数值判定
      │       （资格证据不足或当前不支持时，记录原因并回到普通作者执行）
      └─ 未匹配预设的真实标题
          → PaperReader → ResourceFinder
          → 保留候选并提示补充原始 PDF，停止执行
  → ReportGenerator → 报告、结构化结果与历史记录
```

普通作者仓库执行依赖原始 PDF 的正文和哈希。标题输入当前不自动获取论文全文；发现仓库后仍需上传 PDF。Mock 演示、显式语料对照和编程接口传入代码的兼容路径另行保留，不构成普通 PDF 作者仓库执行证据；仅运行上传论文时应将语料对照设为“无”。

## 模块职责

| 模块 | 当前职责 |
| --- | --- |
| `app.py`、`frontend/` | 输入、LLM 配置、后台任务、进度、历史、报告展示与下载 |
| `orchestrator.py` | 解析请求并分流至预设、作者仓库或兼容路径 |
| `agents/paper_reader.py`、`paper_reading.py` | 读取 PDF、分批整理全文、保存阅读覆盖情况和原始依据 |
| `pdf_input.py`、`pdf_markdown.py` | 从同一 PDF 字节提取正文并用 MarkItDown 转换；逐页检查内容完整性，保留原文回退及受坐标约束的表格视图 |
| `agent_collaboration.py` | 单次运行内的角色消息、请求与回复关联、收件上下文、脱敏和有界协作记录 |
| `agents/resource_finder.py`、`agents/repo_discovery.py` | 结合论文链接、用户输入及搜索定位仓库候选 |
| `discovered_repository_reproduction.py` | 动态作者仓库的来源固定、计划、数据、环境、执行和结果组织 |
| `author_execution_plan.py`、`author_entrypoints.py` | 从固定源码规划原始 Python 脚本、模块或 Notebook 命令 |
| `author_python.py`、`author_runtime_dependencies.py`、`author_failure_analysis.py` | 选择兼容 Python，依据实际错误进行有限依赖修复与失败分析 |
| `repository_runner.py`、`process_lifecycle.py` | 执行 argv 命令，保存输出和产物，处理退出、中断及所属进程清理 |
| `source_repair.py`、`repository_repairer.py`、`safety/patch_transaction.py`、`source_repair_evidence.py` | 兼容修复：从实际报错派生或手动指定精确白名单、独立副本、运行前 AST 与实际缺失 API 核对、原命令重跑、全树回滚、补丁来源清单及脱敏证据 ZIP |
| `frontend/task_control.py`、`process_lifecycle.py` | 任务取消令牌、独立 worker 与子进程树回收；清理确认后才开放重置，刷新保留状态 |
| `result_diagnosis.py` | 执行完成后的数值或比较证据诊断：确定性观察、引用证据的模型假设及只读核查建议，不改实验或重新训练 |
| `repository_profiles.py`、`method_profiles.py`、`rezero_profiles.py` | 注册固定实验；预设范围由各自源码、数据、参数与设备要求限定 |
| `author_claim_comparison.py` 及相关证据模块 | 从作者输出提取指标，对照原 PDF 并验证引用、统计口径和比较条件 |
| `strict_validation.py`、`agents/experiment_protocol.py`、`discovered_repository_runtime.py` | 可选严格核验：绑定论文与源码协议、在预算内自动读取确切本地 import 并重查冻结依据，核对完整运行、独立模型推理和指标复算证据 |
| `agents/report_generator.py`、`report_revisions.py` | 生成报告，保留来源、执行诊断、指标与历史报告版本 |
| `environment_lifecycle.py`、`resource_manager.py`、`audit/` | 实验依赖释放、资源清单、归档及审计记录 |

这些 Agent 按阶段执行，使用同一运行中的角色收件箱交换证据和反馈。Reader 的来源摘要进入 Finder，Planner 的源码请求交给 SourceReader，PlanValidator 的具体拒绝原因进入下一轮规划；执行失败交给 FailureAnalyst，其结论进入修复提示，PatchValidator 与 CodeExecutor 的校验或重跑结果反馈给下一轮修复。命令、哈希、修复白名单和实验协议仍由确定性规则核验。消息通道本身不调用模型，也不调度并行模型进程。

消息包含发送者、接收者、类型、关联 ID 和消费标记，最终保存为 `agent_collaboration.json`，网页与报告显示协作记录。每次运行独立隔离，最多 128 条、JSON 序列化后总计最多 131072 字符；预算满时停止添加消息，不延长已有规划或修复次数。详见 [Agent 协作与 PDF Markdown](agent_collaboration_and_markdown.md)。

## 当前开放范围

- 网页作者仓库和固定预设使用本地受控子进程与隔离依赖目录。Docker 控件暂未开放，底层执行器的 Docker 支持不等于这些主路径已支持容器执行。
- 原始作者命令不设训练时间上限；严格捕获保留全部原作者步骤，并将这些步骤的 `timeout_s` 设为 `None`。追加的独立模型推理与指标复算步骤单独限制为 300 秒；下载、安装和模型请求保留各自的超时，错误修复有次数上限。缺少 CUDA、依赖或可支持入口时保留具体阻塞原因。
- `author_code_completed` 表示所需命令正常返回且所属进程完成处理。仅准备源码或环境属于准备状态；Mock 属于演示状态。
- 普通 PDF 作者仓库执行完成后尝试比较作者输出与论文宣称。比较保留引用、差值和限制，失败不改变已经保存的命令完成事实；它不替代独立复算和整篇论文验收。
- 编程接口的 `source_repair` 默认 `False`；普通真实 PDF 网页默认勾选有界兼容性修复，可关闭。文件列表留空时启用 `source_repair_auto`，根据最终真实 traceback、固定源码摘要和缺失 API 自动定位最多三个 Python 文件；也可填写 1–32 个已存在的 Python 相对路径，实际每轮修改仍最多三个。不接受目录、通配符、数据、评测或权重文件。标题、Mock、语料对照、外部代码、固定预设和 Notebook 计划不进入该修复路径。
- 显式源码修复在原命令真实失败后只创建一次独立会话，最多两轮、每轮最多三个文件；拒绝也消耗轮数。副本只包含固定作者源码与冻结输入，原工作区、来源与观测器保持独立。候选保留原 argv、cwd、依赖顺序、步骤环境及已准备依赖；在应用前核对 AST 结构、训练/划分/评测敏感名称与会话已观测的真实报错中的缺失 API，再按原命令完整重跑。复制和快照前用元数据检查 4096 文件 / 256 MiB 上限，超限保留 `resource_limit` 诊断；失败或后验校验异常全树回滚，恢复失败立即停止并标记 `rollback_failed`。
- 源码修复对作者环境中的工作区绝对路径做有记录的副本重定位；尚无冻结凭据的动态 `author_runtime` 数据不进入修复。2026-10-21 的 248 项合成机制验证及来源指纹见 [入口证据](evidence/issue20_web_repairs_20261021.json)。2026-10-23 新增真实模型 API 的自动定位、兼容补丁和数值诊断验证，以及下载卡住后的停止与清理机制，见 [紧急修复记录](p0_cancellation_self_repair_20261023.md)。
- 修复成功保存原失败、独立修复计划、逐轮提案 / diff / 日志 / 回滚结果及原/实际源码摘要清单。结果明确标记 `patched_author_repository`、`patched_author_code_execution` 和 `source_modified=true`；仍为 `paper_metrics_verified=false`、`is_reproduced=null`、`training_observed=false`。参考比较使用修复后的来源摘要并标记 `patched_author_output`。此入口与严格核验互斥，细节见 [显式仓库兼容修复](repository_repairs.md)；实现接入不等于网页完整验收或真实论文严格通过。
- `strict_validation` 默认 `False`。网页只对普通真实 PDF、未选预设和语料对照时提供勾选项；旧的审核、结果摘要共享及优化开关不会启用严格核验。显式请求严格核验时，先检查所选实验协议与冻结来源；当前不支持或资格证据不足则记录 `strict_validation.status=unavailable` 和原因，继续普通作者执行及参考比较。
- 普通 PDF 可填写方法 / 模型和数据集名称，作为 `experiment_request` 经网页、后台及独立 worker 透传。名称只用于选择作者已有实验，论文引用、原命令和冻结协议仍分别核验；严格规划不能静默换成另一数据集。
- 捕获选择器支持原变量名、旧 `attributes` / `keys`，以及最多八步的有序 `path`，例如 `{name:'dataset',path:[{attribute:'graph'},{key:'node_feat'}]}`。混合路径绑定原源码的完整 AST 链；运行时只读直接存储字段和内建字典 / 列表 / 元组，不执行表达式、getter 或自定义索引。
- 整数标签 `[N,1]` 只在捕获与独立评估中无损展平成 `[N]`。原始字节、dtype / 字节序、形状及摘要随模型输入归档，独立进程重新核验转换依据；作者训练输入不变。浮点标签、多列标签和不匹配的测试索引仍拒绝。
- 严格协议分析器可自动跟随冻结源码中程序解析出的确切本地 import，并核对读取内容与固定哈希；自动补读受总文件数和字节上限约束，不消耗模型的两轮源码请求额度。证据缺失、预算耗尽或不支持的源码结构会保留诊断，不以模型猜测补齐协议。降级前保存严格拒绝来源包，并在核对物理输入未变后恢复普通规划的原始来源视图，避免严格补读占满普通规划的证据预算。
- 严格数值验收要求原 PDF、固定作者源码、实验协议与执行计划相互绑定，并验证完整训练/测试、原生运行次数、独立新模型预测、独立复算及作者输出的一致性。只有全部证据通过才能标记 `paper_metrics_verified=true`；是否达到冻结论文参考值另由 `is_reproduced` 给出，范围限于 `selected_paper_experiment`。仅勾选、命令退出或打印出相近分数均不满足这些条件。
- 普通执行不启用旧审核关卡；严格路径仅使用其所需的来源、协议与计划检查。智能优化在所有当前入口均关闭。相关 Agent、优化模块和历史记录仍保留，不能仅凭代码文件存在就视为已完成真实论文验收。

当前固定预设包括 DLinear 完整单项实验与快速诊断、SIREN 图像拟合、Neural ODE 螺旋轨迹示例、ReZero CIFAR-10 实验。预设执行完成只说明所选实验范围内的运行结果。

## Windows 运行与验证

在应用目录 `AutoReproducer\` 中用 PowerShell 启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

应用解释器与作者训练解释器分开管理；作者依赖、临时文件和输出保存到配置的数据根，默认是应用目录的 `data\`。解释器切换与环境释放由运行生命周期管理，不删除应用自身虚拟环境。

Windows 验证需要覆盖中文路径、UTF-8 输出、创建与结束所属子进程，以及取消后的日志保存和文件句柄释放。训练退出、所属进程清理及报告归档是不同步骤，不能仅凭界面停止刷新判断完成。严格核验的机制测试和原生算子测试也要与真实 API、新论文的完整训练及数值验收分开记录。

2026-10-21 已完成两次真实模型 API / PDF 的准备回归，使用已有 PyG 回归论文并显式设置 `prepare_only`。两次均在严格协议不可用后继续普通作者命令规划，最终为 `prepared`、未执行训练；第二次已自动读取 8 个确切 import 来源，最终因总源码预算不足记录 `unavailable`。测试代际、源码指纹、原生机制及清理摘要见 [本轮证据](evidence/strict_validation_20261021.json)；它们不构成作者训练完成或新论文严格验收通过证据。

后续 Windows 深路径修复在冻结版本上通过 **165 passed / 7 skipped / 0 failed** 的合并定向验证；其中两项严格合成实验实际使用最长 270 字符的日志路径，分别完成训练、独立进程重载与复算。最终批次 19 份清理记录全部释放，无剩余和错误，详见 [跟进证据](evidence/strict_followup_20261021.json)。[新论文候选审计](strict_candidate_audit_20261021.md) 仍为静态结果，没有新论文训练或严格验收通过记录。

### PR #54 合并后的内部记录路径跟进

[PR #54](https://github.com/Yukikazct/AutoReproducer/pull/54) 已合并，页面展示的 16 条 Windows、Linux、PyG 和 DGL 检查均为 `SUCCESS`。其后的路径修复另以本地专项验证，原先深目录触发 `WinError 206` 的记录仍保留在上一代证据中。

`RepositoryRunner` 当前将内部 records 根目录和 run_dir 统一为 Windows extended-length 路径，日志、启动 gate、环境准备目录与产物副本继承同一格式；作者 workspace、cwd、argv 保持原语义。记录目录的 symlink / junction 会在执行前拒绝，本次未修改系统 LongPaths 设置。这是内部运行记录路径修复，不表示任意深作者目录、依赖缓存或第三方工具均已兼容。

路径及相关运行专项 `runtime_paths_final` 为 **63 passed**；仍在深层临时目录运行的 `native_deep` 为 **3 passed / 30 deselected**，包括两项严格合成实验的重载与独立复算，以及一项严格不可用后的普通作者运行。三项服务的环境清理记录均为 `released`，无剩余依赖或清理错误。后续记录单列于 [strict_followup_20261021.json](evidence/strict_followup_20261021.json)，不替换前一代证据，也不新增真实论文严格验收成功的结论。
