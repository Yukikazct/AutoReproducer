# CHANGELOG

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。
版本号采用 `YYYY.MM.DD-<序号>`（按发布批次日期的扁平版本）。

---

## [2026.09.17-1] - 2026-09-17

### 修复（安全加固，Batch 3）

- **本地执行危险代码静态门**：`src/agents/code_executor.py` 本地模式（无沙箱）执行
  LLM 代码前新增 `_dangerous_constructs()` 静态扫描，拦截明显危险调用——
  `subprocess`/`os.system`/`os.popen`/`os.spawn*`/`pty`（命令执行）、`eval`/`exec`/
  `__import__`（动态执行）、`socket`/`requests`/`urllib`/`http.client`/`ftplib`/
  `smtplib`/`paramiko`/`httpx`/`aiohttp`（网络外联）、`shutil.rmtree`（递归删除）；
  - `run()` 语法门之后命中即 `_not_runnable`（exit_code=-5，下游判「无法验证」）；
  - `_execute_code_local` 开头兜底拦截（exit_code=-6 / `danger_blocked`），覆盖
    Optimizer 真实执行（`execute_in_workspace` 绕过 `run()`）；
  - 正则兜底、非正式沙箱；生产复现不可信代码请用 Docker。
- **patch_policy 收紧**：`src/safety/patch_policy.py`
  - `classify` 的 protected 目录前缀由「只查首段」改为「任意层级命中」
    （`src/data/x` 与 `data/x` 一样判 protected）；
  - 新增 `DEFAULT_PROTECTED_SUFFIXES`（`.pem`/`.key`/`.p12`/`.pfx`/`.crt`）与
    密钥文件（`.env`/`id_rsa`/`credentials.*` 等）到 protected_files。
- **明确执行边界**：`_execute_code_local` 首次执行记 WARNING 审计日志
  （「本地模式无沙箱隔离…请 use_docker=True」）；README「两种模式」补安全提示。

### 测试

- 新增 `tests/test_code_executor_safety.py`（28 例：危险命中/良性不误报/run 短路/
  本地兜底拦截/良性仍执行）。
- 扩展 `tests/test_safety.py`（2 例：嵌套受保护目录、密钥文件拦截）。
- 全量 **163 passed**。

---

## [2026.09.10-3] - 2026-09-10

### 修复（前端可观测性 / 诚实性，Batch 2）

- **前端不再卡「运行中」**：`app.py` 原来只在 `snap["result"]` 为真时清
  `st.session_state.running`，但后台线程兜底崩溃时只写 `error` 事件、不写
  `result`，导致「开始复现」按钮永久禁用。改为用 `snap["running"]`
  （`ProgressStore.read_snapshot` 在 `done` 与 `error` 两种终态都置 False）
  作为终态判据。
- **历史列表不再恒显「未知 / 0 / 0」**：
  - 写侧 `frontend/backend_pipeline.py::run_pipeline_core` 收尾处（COMPLETED 与
    ERROR 都执行）新增一条 `FINISH` 终态 ledger 记录，携带
    `result.state / duration_sec / llm_calls`（复用 `logger.get_stats()`）；
  - 读侧 `frontend/history_manager.py::list_sessions` 由「只读首条」改为
    「读全量记录」：标题取首个含 `outputs.title` 的记录，终态取末条 `result`，
    缺失时回退 `RUNNING`/0/0。
- **异常不再被吞**：`backend_pipeline.py` 三处 `except Exception as e:` 原先把
  `e` 抓了不用、`result["error"]` 硬编码 `None`；现改为 `error_msg` 累积真实异常
  文本并 `logger.log(...ERROR...)` 留审计，随 `done` 事件透出，前端
  `result.state==ERROR` 分支即可展示原因。
- **审计日志去重**：`logger.get_summary()` 返回全量 entries，原来每阶段结束后
  全量重发导致进度文件同一条日志重复出现；新增 `_emit_new_logs()` 用 `emitted`
  游标只发增量。

### 测试

- `tests/test_history_manager.py` 新增 3 例：真实结构 ledger（标题首条 / 终态
  末条）、终态 ERROR 回填、末条缺失终态回退 RUNNING。
- `tests/test_backend_pipeline.py` 新增 2 例：进度文件无重复审计日志；阶段异常
  时 `result["error"]` 非 None 且 ledger 末条存在 FINISH 记录。
- 全量 **131 passed, 1 skipped**。

---

## [2026.09.10-4] - 2026-09-10

### 修复（环境依赖 + 诚实降级）

- **补齐 PDF 解析依赖**：此前环境缺 PyPDF2 / pdfplumber，`PaperReader._extract_text`
  恒返回空文本，上传 PDF 会退化成「未知标题 → 信息不足 → 无法验证」。现安装
  PyPDF2 3.0.1 / pdfplumber 0.11.10（`requirements.txt` 已列出，见下）。
- **`PaperReader._fallback_extract` 诚实说明降级原因**：原先把所有本地降级都
  写成「未知（LLM 不可用时本地降级提取）」，误导用户以为是 LLM 挂了；实为
  「PDF 文本提取失败」（缺依赖 / 扫描件）。现按「正文为空」/「LLM 解析失败」两
  种情况分别标注，`method` 仍以「未知」开头，`_judge_insufficient` 的判据不受影响。
- **`requirements.txt` 中文注释改为 ASCII**：Windows GBK 环境下 `pip install -r
  requirements.txt` 会因 UTF-8 中文注释触发 `UnicodeDecodeError` 直接失败；
  改为英文注释后可在任意 locale 下安装。

### 测试

- 全量 **133 passed**（含前端 AppTest 冒烟测试，此前因缺 streamlit 被跳过）。

---

## [2026.09.10-2] - 2026-09-10

### 修复（复现核心闭环，Batch 1）

- **`_sanitize_code` 抹掉全部缩进（根因修复）**：`src/agents/code_executor.py`
  语法兜底分支用 `.strip()` 清行，把**所有前导缩进一并删掉**，使「LLM 输出
  被截断」这个真实原因被伪装成整份代码的 `IndentationError`。改为只清理
  「行首行号 + 行尾空白」（新增 `_LINE_NO_RE`，行号后最多吃掉一个分隔空白，
  其余空白是原本的缩进），并在「去行号后不再像代码行」时保留原行避免误删。
- **语法门 + 再生成**：`src/agents/code_executor.py` 新增 `_produce_code` /
  `_regenerate_code`。清洗后的代码必须能 `compile`，不通过时按
  「截断 / 语法错误」把失败原因回灌给 LLM 重新生成（`MAX_CODE_REGEN=2` 次），
  仍不可编译则短路为**未运行**（`_not_runnable` / `exit_code=-5`），
  **绝不把残码送进沙箱**。
- **信息不足不再编造代码**：`src/agents/code_executor.py::_info_insufficient`
  在论文缺方法/数据集/指标时短路为「无法运行」，不再生成无关占位代码
  （此前会用 CIFAR-10 CNN 去复现线性回归）；生成提示词新增
  `# INSUFFICIENT_INFO` 约定，明令禁止用无关数据集充数。
- **PaperReader 标题-only 诚实降级**：`src/agents/paper_reader.py`
  - 删除编造的占位摘要（`"摘要: 这是关于《X》的论文,包含方法、实验与指标声明。"`），
    改为如实标注「未获取到论文正文」并在提示词中要求推断不出时返回
    `insufficient_info: true`；
  - 修正 `if paper_title and "：" not in parsed.get("title", "")` 的自相矛盾条件，
    用户传入的标题一律优先；
  - 透传 `insufficient_info` / `info_sufficient` 给下游 Agent。
- **ResultValidator 区分「无法运行」与「复现失败」**：`src/agents/result_validator.py`
  - 新增三态 `status`：`not_runnable`（`is_reproduced=None`，代码没跑起来）/
    `not_reproduced` / `reproduced`；未运行时跳过 LLM 比对，省预算；
  - `mse` 与 `rmse` 不再混键（`_METRIC_PATTERNS` 拆分 + `\b` 词边界，
    修复 `"rmse: 1.2"` 被 mse 分支抢先命中）；
  - 声明了但输出中提取不到的指标不再静默跳过，记入 `missing_metrics`
    并在报告「无法比对的指标」中列出；
  - 声明指标一个都没对上时不再判为「复现成功」。
- **报告如实渲染三态**：`src/agents/report_generator.py` 执行状态显示
  「⚠️ 未运行」及原因，验证状态显示「⚠️ 无法验证（代码未运行）」；
  新增 `_fmt` 容忍非数值 confidence（此前 `f"{0.1:.2f}"` 对字符串会崩溃）。
- **优化跳过原因更准确**：`src/orchestrator.py`、`frontend/backend_pipeline.py`
  区分「代码未能运行，无法优化」与「复现未成功,跳过优化」。

### 变更

- **LLM 截断诊断**：`src/llm/llm_client.py` 记录 `last_finish_reason`
  （`choices[0].finish_reason`），便于判断截断发生在 API 侧（`length`）
  还是清洗侧；CodeExecutor 在触发再生成时把该值写入审计日志。

### 测试

- 新增 `tests/test_reproduction_core.py`（21 用例：缩进保留、行号剥离不丢缩进、
  截断判定、再生成闭环、语法门短路、信息不足短路、PaperReader 诚实降级、
  Validator 三态、mse/rmse 拆键、缺失指标上报）。
- 全量 **126 passed, 1 skipped**。

---

## [2026.09.10-1] - 2026-09-10

### 新增

- **复现历史记录 Tab（前端 Tab5）**：`frontend/history_manager.py`
  - 存储仪表板：按目录统计 experiment_ledger / logs / runtime / reports / optimization_demo / pinn-output 的文件数与占用空间；
  - 历史会话列表：从 ledger / logs / runtime / reports 反向聚合每次复现会话（标题、状态、耗时、LLM 调用数、日志条目数）；
  - 单会话详情：查看该次复现的审计日志片段（最近 5 条）；
  - 一键清理：按保留天数清理过期 `data/runtime/` 实时进度文件（单次约 500KB-700KB）；
  - 报告下载按钮：历史报告与本次报告均可在页面直接下载。
- **复现报告落盘**：`frontend/backend_pipeline.py`
  - 报告生成后自动写入 `data/reports/{论文标题}_{session_id}.md` 并返回 `report_path`；
  - 文件名时间戳统一复用 `logger.session_id`，与 ledger 文件名（`YYYYMMDD_HHMMSS`）严格一致，保证历史模块可回链。
- **历史记录模块单元测试**：`tests/test_history_manager.py`（18 用例：会话聚合、报告匹配、存储统计、过期清理、详情读取、格式化、文件名解析）。
- **外部项目参考资源**：`references/`（详见 `references/README.md`）
  - `references/paperbench/`：PaperGuru-Benchmark 23 篇论文复现评测基准（`aggregate-final.json`、逐篇 Δ 对比、评级协议 README）+ **全部 23 篇论文完整提交物**（含 pinn 真实复现样例与 semantic-self-consistency 最高分样例等）。
  - `references/researchstudio-idea/`：Microsoft ResearchStudio-Idea 说明、`.env.template`、六源并发论文检索脚本（arXiv/DBLP/OpenAlex/OpenReview/Semantic Scholar/Crossref）与 idea 质量评测数据；
  - `references/researchstudio-reel/`：论文→海报/视频/博客交付流水线说明（paper2reel skill 定义）。

### 修复

- **LLM 生成代码被截断（关键修复）**：`src/llm/llm_client.py` payload 增加 `max_tokens: 8192`（此前默认值导致长代码在数百行处硬截断，输出不完整）。
- **代码提取正则漏掉方法体行**：`src/agents/code_executor.py` 扩展 `_CODE_LINE_START` 正则，增加函数调用（`super().__init__()`）、索引访问（`self.net[0]`）、属性访问（`model.forward`）三种行首模式，避免缩进的方法体/调用语句被误丢弃。
  - Orchestrator 与 backend_pipeline 的 `audit_summary` / `audit_stats` 键名不匹配（两处），统一为 `audit_stats`；
  - `audit_stats` 注入时机晚于报告生成，前置到报告渲染前；
  - backend_pipeline 未把各 Agent `llm_calls` 增量写入 logger，补齐累加。mock 端到端验证报告统计恢复正常（LLM 调用 35）。
- **历史报告无法回链**：报告文件名原采用写盘时刻时间戳，与 session_id（流水开始时刻）不一致导致列表匹配失败；改用 `logger.session_id` 后与 ledger 对齐。
- **CodeExecutor 缺失依赖直接失败**：`src/agents/code_executor.py` 新增 `_ensure_local_deps()`，执行前按 requirements 自动 pip 安装（进程内幂等缓存），失败以 `exit_code=-4` 阻断并给出可读原因（此前真实作业因缺 matplotlib 在第 4 步崩溃）。
- **Mock 模式依赖 numpy**：`src/llm/llm_client.py` 去除 numpy，改为纯标准库实现。
- **残留 Ollama 用例与命名不一致**：删除 Ollama 测试用例，统一 `llm_client.py` 命名（9 处引用 + README 同步）。

### 变更

- **LLM 配置**：弃用 Ollama，全面切换 OpenAI 兼容远程 API（默认 `https://api.deepseek.com` 根地址，自动补全 `/v1/chat/completions`；`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_TIMEOUT` 环境变量注入，前端页内填写优先）。
- **A3 Optimizer 安全网**：`src/safety/patch_policy.py` + `workspace_snapshot.py`，实验前快照 + 出界 patch 拦截（14 单测）。
- **Optimizer 真实执行闭环**：`src/optimizer/real_simulator.py` + orchestrator 工作区注入 + `tests/test_optimizer_real.py`；`scripts/run_optimization_demo.py` 真实跑通 2 案例（+5.63% / +9.76%），报告见 `reports/optimization_demo.md`（Docker 本机不可用，采用降级验收）。
- **前端 API 配置校验**：`frontend/llm_config.py`（`resolve_llm_config` / `config_missing` / `test_llm_connection`）+ 侧边栏"测试 AI 连接"按钮 + 配置缺失提示。
- **后台复现进度实时展示**：`frontend/backend_pipeline.py` 的 ProgressStore（JSONL 事件流）+ 前端每 2 秒轮询展示 8 Agent 状态（依赖 `streamlit-autorefresh`）。
- **样本论文**：`samples/paper/minimal_linear_regression.pdf`（OLS 线性回归，仅 numpy，MSE≈0.09，seed=42，端到端 `is_reproduced=true`）；生成脚本 `scripts/generate_sample_paper.py`（依赖 fpdf2）。

### 测试

- 全量 **107 passed**（含 history_manager 18、backend_pipeline 7、code_executor_deps 5、optimizer_real 7、safety 14、llm_config 10、app_smoke 2、architecture 等）。

### 待办 / 已知项

- **样本论文缺失（阻塞）**：`samples/paper/minimal_linear_regression.pdf` 已丢失（目录为空），导致真实模式端到端复现无法输入有效论文；需重新生成或用户提供可访问的 PDF 样本。
- **端到端真实复现尚未跑通（阻塞）**：API Key 已验证可用、审计链路已修复，但因样本论文缺失，PaperReader 退而生成占位信息，LLM 输出无关代码（CIFAR-10 CNN 而非线性回归），复现失败。需先恢复样本论文再验证。
- **`.gitignore` 与 `.env` 支持待用户确认**：API Key 目前仅驻内存，无密钥文件进入仓库；但缺少 `.gitignore` 防止误提交，也缺少 `.env` 文件便于本地配置管理。
- **`data/runtime/` 自动清理策略缺失**：实时进度文件单次 500KB-700KB 且持续累积，目前仅靠前端的"一键清理"手动操作；如需策略化（启动时自动清理 N 天前文件）可后续实现。
- **PDF 解析依赖未确认**：真实 PDF 提取依赖 `PyPDF2`（或 `pdfplumber`），当前环境未确认是否已安装；若缺失会导致 PaperReader 回退到标题-only 模式，信息严重不全。
- **CodeGenerator 与 CodeExecutor 职责耦合**：同一 Agent 既生成代码又执行代码，prompt 过于简略（仅方法名/指标/数据集），容易因输入信息不足生成无关占位代码；长期建议拆分为独立的 CodeGeneratorAgent，接收完整论文结构化信息（方法细节、算法步骤、超参数）后再生成。

---

## [2026.09.10-0] - 2026-09-10（追溯）

### 变更

- 项目初始多智能体架构：8 Agent（理解 → 拆解 → 依赖识别 → 代码生成 → 执行 → 验证 → 报告 → 审计）流水线。
- LLM 调用审计链路：`src/audit/audit_logger.py`（ledger 落盘 `data/experiment_ledger/`、日志落盘 `data/logs/`）。
- Docker 真实执行（`ea7fb73`）与 pinn 复现报告（`b310fc5`）。