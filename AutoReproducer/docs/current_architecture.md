# 当前架构与能力边界

本文描述当前代码的执行路径。历史实验和旧架构图中的审核、独立数值验收与优化结果，不代表这些功能已在当前主流程开放。

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
| `agents/report_generator.py`、`report_revisions.py` | 生成报告，保留来源、执行诊断、指标与历史报告版本 |
| `environment_lifecycle.py`、`resource_manager.py`、`audit/` | 实验依赖释放、资源清单、归档及审计记录 |

这些 Agent 按阶段执行，并非并行的独立模型进程。模型用于阅读、发现、规划、有限失败分析和信息比较；命令执行、哈希校验、进程清理及结果落盘由程序完成。

## 当前开放范围

- 网页作者仓库和固定预设使用本地受控子进程与隔离依赖目录。Docker 控件暂未开放，底层执行器的 Docker 支持不等于这些主路径已支持容器执行。
- 原始作者命令不设训练时间上限；下载、安装和模型请求保留各自的超时，错误修复有次数上限。缺少 CUDA、依赖或可支持入口时保留具体阻塞原因。
- `author_code_completed` 表示所需命令正常返回且所属进程完成处理。仅准备源码或环境属于准备状态；Mock 属于演示状态。
- 普通 PDF 作者仓库执行完成后尝试比较作者输出与论文宣称。比较保留引用、差值和限制，失败不改变已经保存的命令完成事实；它不替代独立复算和整篇论文验收。
- 论文身份、阶段输出及实验协议的旧审核关卡当前不启用；智能优化在所有当前入口均关闭。相关 Agent、优化模块和历史记录仍保留，不能仅凭代码文件存在就视为主流程已交付。

当前固定预设包括 DLinear 完整单项实验与快速诊断、SIREN 图像拟合、Neural ODE 螺旋轨迹示例、ReZero CIFAR-10 实验。预设执行完成只说明所选实验范围内的运行结果。

## Windows 运行与验证

在应用目录 `AutoReproducer\` 中用 PowerShell 启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

应用解释器与作者训练解释器分开管理；作者依赖、临时文件和输出保存到配置的数据根，默认是应用目录的 `data\`。解释器切换与环境释放由运行生命周期管理，不删除应用自身虚拟环境。

Windows 验证需要覆盖中文路径、UTF-8 输出、创建与结束所属子进程，以及取消后的日志保存和文件句柄释放。训练退出、所属进程清理及报告归档是不同步骤，不能仅凭界面停止刷新判断完成。当前回归测试与具体 Windows 机器上的真实 API / 训练验收应分别记录；本文不声明已在本轮完成 Windows 实机训练。
