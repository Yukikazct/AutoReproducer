# Agent 协作与 PDF Markdown

本文描述 2026-10-25 的实现，关联 #60（PDF 转 Markdown）、#61（失败修复）、#59（停止与恢复）及 #20（有界事务修复）。这里的测试证明消息和证据处理机制，不表示整篇论文已完成独立数值复现。

## PDF 数值读取

应用依赖包含 `markitdown[pdf]>=0.1.9,<0.2`。普通安装与自动运行环境准备使用同一份 `requirements.txt`，旧环境缺少该依赖时会进入已有的隔离环境准备流程。

1. 从原始文件读取不可变字节，保留 PDF SHA-256、页码、标题和仓库链接。原有 PyPDF2 / pdfplumber 提取与数字排版恢复仍用于建立完整原文。
2. 将每页交给 MarkItDown 的公开 PDF 转换器。格式已知，因此不初始化通用格式识别模型或其线程池。转换采用本地流，不加载插件，不调用 OCR 或额外模型服务。数字的原始写法、出现次数和顺序、可读字符顺序、数学符号与脚注位置及解码异常都要核对；不通过就保留整页原文。数字相同但 `±`、不等号、幂符号或脚注消失，也会拒绝。
3. 科学论文双栏排版可能被 MarkItDown 默认的表单识别错误合并。对于含表格标题、至少三条等宽横线、单层表头和可解析数值单元格的表格，程序按 PDF 坐标划分列，将 HTML 表格交给 MarkItDown 转成 Markdown，并附在完整原页之后。跨列文字、合并单元格、缺失字符、数字变化及不支持的结构会拒绝。补充视图与原文属于同一证据，不计作重复实验或独立来源。
4. Reader、规划来源包和数值引用共用同一份规范页文本及摘要，引用偏移不会落在另一版提取结果上。表格比较支持 Markdown 分隔符、明确空白左上角和分组行；数据集列及方法变体仍须精确匹配。带脚注的目标数值保留未知，不能忽略脚注条件后直接确认。

运行目录保存：

| 文件 | 内容 |
| --- | --- |
| `paper_markdown.md` | 逐页文本，以 PDF 页码标记；可能混合转换页和完整原文回退页 |
| `paper_markdown.json` | 引擎版本、PDF 与文本摘要、逐页结果、回退原因、表格坐标与摘要 |

报告页面提供“下载论文 Markdown”，报告 ZIP 同时携带 Markdown 与转换依据。网页使用下载按钮，离线 Markdown 保留相对附件链接与摘要标记；格式和收尾元数据不会显示成正文。导出时重新核对原 PDF 身份及附件摘要，再复制为报告旁的固定命名附件；附件被改动、缺失或指向其他路径时拒绝打包。重建报告沿用同一份已保存的来源。

转换限额为 30 MiB PDF、200 页和 100 万转换字符。超限、缺少依赖或普通转换错误不会截断原论文；取消与进程终止继续传播给所属 worker。扫描件 OCR、任意复杂多层表头和跨页合并表格尚未实现。Markdown 转换成功本身不是数值正确或复现成功的证明。

## 角色之间实际传递的信息

| 发送者 → 接收者 | 内容如何被使用 |
| --- | --- |
| PaperReader → ResourceFinder | 论文来源和仓库线索进入发现提示 |
| AuthorPlanner → SourceReader → AuthorPlanner | 请求确切源码路径，只返回固定哈希匹配的来源；补读进入下一轮提示 |
| PlanValidator → AuthorPlanner | 原始拒绝原因反馈给下一轮提案，不增加规划次数 |
| ExperimentProtocol → AuthorPlanner → ProtocolReviewer | 可选严格路径消费冻结协议上下文，保留原有独立审查；语义拒绝后终止，不新增重试或修改协议 |
| AuthorPlanner → EnvBuilder / CodeExecutor | 已验证计划的摘要；消费时与实际权威计划核对，不能由消息替换命令 |
| CodeExecutor → FailureAnalyst | 真实失败和原命令约束；取消、超时或清理未确认时不发起分析 |
| FailureAnalyst → RepositoryRepairer / EnvBuilder | 源码检查结论作为假设进入提示，依赖或补丁仍需确定性验证 |
| PatchValidator / CodeExecutor → RepositoryRepairer | 校验拒绝或重跑错误进入下一轮修复，失败先回滚 |
| 执行结果 → ReportGenerator | 消费最终结果上下文并附上协作轨迹 |

协作使用进程内、单次运行隔离的 `CollaborationSession`。它提供 `send`、`reply`、`context`、`snapshot`、`persist`，不创建无限对话或独立常驻模型。消息存储最多 128 条、总计 128 KiB，单条最多 8192 字符，收件上下文最多 16 KiB。已知 API 凭据、授权头和常见密钥字段先脱敏，再进入事件、日志和产物。

网页的“Agent 协作记录”展示消息内容，历史结果与报告保存同一轨迹；收到消息不会令执行阶段自动完成。报告用短摘要和消息序号展示交接及回复关系，完整负载保存在 `agent_collaboration.json`，避免长摘要和哈希撑宽页面。异常中止仍保留已发生的交流，分析被打断时记录 `analysis_interrupted`，不向修复器发布未完成的分析结论。

## 修复与恢复边界

源码修复保持独立副本、两轮上限、原始命令重跑、哈希与 AST 检查及全树回滚。协作反馈不允许改变数据、训练次数、评测协议或数值容差。退出码 124 / 130、取消、超时和 `cleanup_confirmed=false` 都阻止修复；候选即使带有成功标记，也不能绕过这些停止条件。

Windows owner 退出、Job 活跃计数为零或 Job 消失均不足以单独证明后代退出。未确认清理保留 `cleanup_unconfirmed` 与依赖根阻断，恢复时重新取得锁并核对任务身份。原工作区、用户报告与生成的实验产物应与本轮代码提交分开保存。

## 验证入口

`tests/test_pdf_markdown.py` 包含真实 MarkItDown 与仓库内 SGC、SSGC PDF 的回归：核验 SGC 第 6 页的 Cora、Citeseer、Pubmed 均值和标准差，以及 SSGC 第 10 页的 R52 数值，并拒绝邻近方法或数据集的错误数值。这些样本经过原 PDF 页面视觉核对；不调用模型，也不运行训练。

`test_agent_collaboration.py`、`test_agent_collaboration_integration.py`、`test_author_plan_collaboration.py` 和 `test_repair_collaboration.py` 验证消息消费、来源约束、拒绝反馈、真实本地命令重跑、取消和预算。全量测试使用显式的项目 `tests/test_*.py` 列表，避免收集数据目录中下载的作者仓库测试。
