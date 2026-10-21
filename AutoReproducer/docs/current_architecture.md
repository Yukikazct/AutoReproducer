# 当前架构与能力边界

本文描述当前代码的执行路径。普通作者执行仍为默认流程；普通真实 PDF 可额外请求严格核验。历史实验和旧架构图中的结果不能代替当前版本的实际验收，智能优化仍未开放。

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
| `agents/resource_finder.py`、`agents/repo_discovery.py` | 结合论文链接、用户输入及搜索定位仓库候选 |
| `discovered_repository_reproduction.py` | 动态作者仓库的来源固定、计划、数据、环境、执行和结果组织 |
| `author_execution_plan.py`、`author_entrypoints.py` | 从固定源码规划原始 Python 脚本、模块或 Notebook 命令 |
| `author_python.py`、`author_runtime_dependencies.py`、`author_failure_analysis.py` | 选择兼容 Python，依据实际错误进行有限依赖修复与失败分析 |
| `repository_runner.py`、`process_lifecycle.py` | 执行 argv 命令，保存输出和产物，处理退出、中断及所属进程清理 |
| `repository_profiles.py`、`method_profiles.py`、`rezero_profiles.py` | 注册固定实验；预设范围由各自源码、数据、参数与设备要求限定 |
| `author_claim_comparison.py` 及相关证据模块 | 从作者输出提取指标，对照原 PDF 并验证引用、统计口径和比较条件 |
| `strict_validation.py`、`agents/experiment_protocol.py`、`discovered_repository_runtime.py` | 可选严格核验：绑定论文与源码协议、在预算内自动读取确切本地 import 并重查冻结依据，核对完整运行、独立模型推理和指标复算证据 |
| `agents/report_generator.py`、`report_revisions.py` | 生成报告，保留来源、执行诊断、指标与历史报告版本 |
| `environment_lifecycle.py`、`resource_manager.py`、`audit/` | 实验依赖释放、资源清单、归档及审计记录 |

这些 Agent 按阶段执行，并非并行的独立模型进程。模型用于阅读、发现、规划、有限失败分析和信息比较；命令执行、哈希校验、进程清理及结果落盘由程序完成。

## 当前开放范围

- 网页作者仓库和固定预设使用本地受控子进程与隔离依赖目录。Docker 控件暂未开放，底层执行器的 Docker 支持不等于这些主路径已支持容器执行。
- 原始作者命令不设训练时间上限；严格捕获保留全部原作者步骤，并将这些步骤的 `timeout_s` 设为 `None`。追加的独立模型推理与指标复算步骤单独限制为 300 秒；下载、安装和模型请求保留各自的超时，错误修复有次数上限。缺少 CUDA、依赖或可支持入口时保留具体阻塞原因。
- `author_code_completed` 表示所需命令正常返回且所属进程完成处理。仅准备源码或环境属于准备状态；Mock 属于演示状态。
- 普通 PDF 作者仓库执行完成后尝试比较作者输出与论文宣称。比较保留引用、差值和限制，失败不改变已经保存的命令完成事实；它不替代独立复算和整篇论文验收。
- `strict_validation` 默认 `False`。网页只对普通真实 PDF、未选预设和语料对照时提供勾选项；旧的审核、结果摘要共享及优化开关不会启用严格核验。显式请求严格核验时，先检查所选实验协议与冻结来源；当前不支持或资格证据不足则记录 `strict_validation.status=unavailable` 和原因，继续普通作者执行及参考比较。
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
