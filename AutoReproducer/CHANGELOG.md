# CHANGELOG

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。
版本号采用 `YYYY.MM.DD-<序号>`（按发布批次日期的扁平版本）。

---

## [2026.09.20-1] - 2026-09-20

### 修复（代码生成完整性：截断 -> 续写拼接）

针对「LLM 生成的复现代码不完整、跑不起来」。

- **续写拼接取代"从头重写"（主修复）**：`src/agents/code_executor.py`
  - 新增 `_continue_code()` / `_stitch()`：代码没写完时，把已写部分的尾部
    交给模型**从断点续写**，再按"末行由续写重写"的约定拼接（含重复行去重），
    每轮都过语法门；最多 `MAX_CODE_CONTINUE=3` 轮。
  - 旧逻辑（`_regenerate_code`）是拿**同一个 prompt** 从头重写，若模型本就
    写不完，重写只会再写不完一次；续写把总长度累加，是唯一能真正"写完"的路径。
  - 续写仍不完整时才回落到从头重生成（`MAX_CODE_REGEN=2`）作为最后手段。
  - `_needs_continuation()` 判据：①`finish_reason == "length"`（API 侧确定
    截断信号）；②语法错误且尾部悬挂/括号未闭合；③能编译但结构上不做任何事。
- **不再空烧预算**：模型复述已有内容（`more in code`）或续写后语法错误一字未变
  时立即停止续写并转入重生成，避免把重复内容越拼越长。
- **清洗层保真 + 丢弃可见**：`_sanitize_code_ex()`（`_sanitize_code` 保留为
  兼容包装）
  - **有围栏就整块原样取用**（只做 compile 校验，一个字符都不改）——保真路径；
  - 无围栏/围栏内不可编译才降级到逐行过滤，且**统计并上报丢弃行数**
    （`_record_sanitize` 写 WARNING），不再让"丢行"静默发生；
  - 收窄中文丢弃规则（`_looks_like_prose`）：只丢"含中文且**无任何代码特征
    字符**"的纯叙述行。旧实现按"含中文就丢"，会把 `print(f"准确率: {acc}")`
    这类**合法语句**删掉，而删完代码往往仍能编译，于是残缺被静默放过。
- **修掉一处 `.strip()` 吃缩进（与 Batch 1 根因同源）**：`_sanitize_code_ex`
  开头原为 `text = raw.strip()`，会把**首行缩进一并剥掉**。续写片段的首行
  天然可能就是缩进行（如 `    print(...)`），被剥成顶格后拼接即
  `IndentationError`。改为 `raw.strip("\n").rstrip()`，只去首尾空行与行尾空白。
- **结构完整性门**：新增 `_structurally_complete()`（AST 解析），拒绝
  "只有 def/class/import、既无顶层调用也无 `__main__` 守卫"的脚本——
  这类片段能通过 `compile()` 却不会产出任何结果，是纯语法门的盲区。
- **提示词纠正**：`_generate_code_prompt`
  - 删掉「生成一段**简短的**训练代码」——原文直接诱导模型写短；
  - 规则 3 反转：由"不要用围栏"改为"**用单个 ```python 围栏**包起来"，
    使清洗层的保真路径真正生效（此前围栏被禁，每次生成都被迫走有损的逐行过滤）；
  - 明确**允许**中文出现在字符串字面量/注释中（只禁围栏外叙述段落）；
  - 增加续写契约（写不完请在完整语句边界停下）与完整流程要求。

### 修复（Verifier 入参截断导致的误判）

- `src/agents/verifier.py` 原以 `str(output)[:2000]` 截断待验证输出。一份
  3616 字符的复现代码被从中间切断后，验证器看到半截代码即**误报**
  「code 字段被截断」——截断其实发生在验证器自己的入参上；同时 2000 字符
  之后的真实缺陷验证器完全看不到。现新增 `VERIFY_INPUT_LIMIT=8000` 与
  `_render_output()`：放大窗口，且真超限时**明确标注**"是入参被截断、
  不代表内容缺失，请勿仅因内容在此处结束就判定不完整"。

### 变更

- **`max_tokens` 参数化**：`src/llm/llm_client.py` 新增构造参数与环境变量
  `LLM_MAX_TOKENS`（默认 `DEFAULT_MAX_TOKENS=8192`），并新增 `is_truncated()`
  （`finish_reason == "length"`）作为确定性截断信号。
  **注意**：这是安全阀而非本次缺陷的主因——见下方「证据」。

### 证据与局限（重要）

- 排查确认：`data/` 中 **2026-09-16 之后没有真实 LLM 运行**，全部是 Mock
  模式的 Dummy 演示（每步 0.0~0.3s，执行的是 `_MOCK_TASKS["code_executor"]`
  的硬编码脚本）。因此"生成仍不完整"无法从近期数据直接验证。
- 全库仅有一条 `finish_reason` 记录，值为 **`stop`**，无任何 `length`。
  即**没有证据表明截断源于撞 `max_tokens` 上限**；观测到的形态更像
  「模型自己没写完/写错」（尾部悬挂运算符、括号未闭合、中间缺 token）。
  故本轮以**续写**为主解（对"模型没写完"同样有效），`max_tokens` 参数化
  仅作安全阀保留。
- 旧报告里"生成代码"代码块在 ~800 字符处断掉是**旧版展示层**静默截断
  （`_clip` 已改为 6000 字符上限 + 明确标注 + `*_execution.txt` 附件）；
  对照 `experiment_ledger` 中 EXECUTE_CODE 的 `inputs.code` 才是完整代码。
  **判断"生成是否完整"必须以 ledger 为准，不能看旧报告的代码块。**

### 测试

- 新增 `tests/test_code_generation_completeness.py`（24 例：续写拼接/接缝去重/
  首行缩进保留/`finish_reason=length` 触发/复述即停/围栏保真/中文 print 不丢/
  叙述行丢弃计数/结构完整性门/提示词断言）。
- 新增 `tests/test_verifier_input_limit.py`（5 例：短输出原样、3600 字符代码
  不被砍在 2000、超限时标注而非静默截断）。
- 改写 `tests/test_reproduction_core.py` 中 3 例：原断言绑定"从头重生成"
  旧策略，现更新为"续写拼接"新契约。
- 全量 **547 passed**。

### 待验证（需真实模式）

本轮修复**尚未经真实 LLM 运行验证**。验收方式：用真实模式跑一篇**非 Dummy**
且信息完整（有方法/数据集/指标）的论文，在 `data/experiment_ledger/` 的
EXECUTE_CODE 记录里确认生成代码完整、可编译、能跑出指标输出。

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

## [2026.09.16-1] - 2026-09-16

### 新增（ScholarAgent 融合：工程可信度）

基于参考项目 ScholarAgent 的迁移融合（P0 五项全部落地，P1 七项推进至 ⑫ 完成）。

- **P0-① 补丁安全策略（`src/safety/patch_policy.py`）**：禁改目录段模式、禁改扩展名（数据/权重）、文件名正则、单文件体积预算、结构化拒绝决策；优化补丁越界即拒。
- **P0-④ 快照指纹（`src/safety/workspace_snapshot.py`）**：全工作区 SHA-256 fingerprint，运行中篡改检测（指纹不一致 → 中止恢复），实验前快照。
- **P0-③ 经验库（`src/experience/experience_store.py`）**：JSONL 持久化（`validated` 标记）、上限截断、`summarize`/`best`、线程安全；路径接入 `AUTOREPRO_DATA_ROOT/experience`。
- **P0-⑤ TrialLedger 账本（`src/experience/trial_ledger.py`）**：Keep/Reject 结构化记录（candidate/评估/理由/restored/apply 文件），供报告与经验库消费。
- **P0-② 证据链模块（`src/evidence/`）**：`build_evidence_registry`（SHA-256 + authentic 判定）、`normalize_findings`（确定性裁决门控：无真实执行证据不判 verified、smoke 天花板）、`build_graph`（Claim/Criterion/Evidence + graph_sha256），纯函数无 LLM。
- **P1-⑥ BeamUCT 双层搜索+先验播种（`src/optimizer/beam_uct.py`）**：方向级 UCB + 参数树 UCT + Beam top-k；`direction_priors`/`seed_from_summary`/`seed_from_store` 经验库播种；`ucb_scheduler` 增 seed/retire/retired。
- **P1-⑦ 冻结 Spec + 隐藏 Holdout**：ResearchSpec sha256 冻结验收，只对最终 best 状态做隐藏留出多轮评估，优化结果按 spec 判定。
- **P1-⑧ 静态依赖解析 + pip 自愈（`src/agents/dependency_resolver.py`）**：AST+requirements 双源解析、stdlib 过滤、py39 归一、运行时缺模块 pip 自愈（≤3 轮，本地隔离目录 / Docker 累积重跑）。
- **P1-⑨ 确定性仓库发现链（`src/agents/repo_discovery.py`）**：用户URL→PwC→GitHub 搜索→curated 回退四级降级链 + 离线开关；`fetch_code` revision pin + `.autorepro-repo-source.json` 溯源标记。
- **P1-⑩ 防泄漏 Benchmark 评测（`src/benchmark/leakage_safe.py`）**：确定性 hash 切分（60/20/20）、隐藏标签私有目录、指标契约冻结后端复算；`dataset_registry` 增 `benchmark_hint`；`ResourceManager.prepare_leakage_safe_benchmark`（纯 Python 零依赖）。
- **P1-⑪ Docker 沙箱加固（`src/agents/code_executor.py`）**：镜像白名单（拒非官方镜像 `exit_code=-5`）、cap-drop ALL + no-new-privileges + 只读 rootfs + tmpfs + 非 root + CPU/mem/pids 限额，随 Docker 可用性三级降级（level 0→1→2）；加固时 pip 走 `--target /tmp/site-packages` + PYTHONPATH 注入；`AUTOREPRO_DOCKER_IMAGE_ALLOWLIST`/`AUTOREPRO_DOCKER_HARDEN` 可配。
- **P1-⑫ 用量计量增强（`src/audit/audit_logger.py` + `src/llm/llm_client.py`）**：plan 级 LLM token 统计（`begin_plan`/`end_plan` 界定流水线阶段，`extract_token_usage` 兼容 standard/别名/推算三种 usage 形态，`usage_hook` 直连归账）+ 容器耗时维度（`record_sandbox_exec`，docker 执行 try/finally 墙钟计量）；`get_stats()` 增 `plans`/`usage` 汇总（llm_calls/tokens/llm_seconds/container_exec_calls/container_exec_seconds/models）；Orchestrator 各阶段接线（含异常路径出栈、兼容外部 mock LLM）。

### 测试

- 新增 **266 项** 用例：补丁安全（P0-①）23 项、快照指纹（P0-④）14 项、经验库（P0-③）16 项、TrialLedger（P0-⑤）14 项、证据链（P0-②）17 项、BeamUCT 融合（P1-⑥）24 项、冻结 Spec/Holdout（P1-⑦）24 项、依赖解析自愈（P1-⑧）23 项、仓库发现链（P1-⑨）37 项、防泄漏 Benchmark（P1-⑩）38 项、沙箱加固（P1-⑪）13 项、用量计量（P1-⑫）23 项。
- 全量 **478 passed, 1 skipped**（真实模式 21s 全绿）。

---

## [2026.09.16-0] - 2026-09-16

### 新增（存储管理：按需懒加载 + 三层缓存 + 体积瘦身）

解决「多篇论文复现累加后磁盘耗尽」：ImageNet 约 150GB vs CIFAR-10 仅 170MB，
按需懒加载只拉当前任务最小集，任务完成归档后清理 L0。

- **P0-1 ResourceManager（`src/resource_manager.py`）**：`fetch_code` /
  `fetch_dataset` / `fetch_weights` 懒加载（git depth-1 克隆、冒烟子集、
  本地/URL/HF 子路径权重），重复 fetch 幂等；manifest 生成/读取兼容
  2026-09-09 存量格式；cleanup 记 `cleaned_at`；archive/restore 对接 L1
  温存储（zip 内路径 `repos/<pid>/...` 与 restore 对齐）；L0 配额守护
  `AUTOREPRO_L0_QUOTA_GB`（默认 20GB），超限按 LRU 给建议不自动删除。
- **P0-2 编排器存储钩子（`src/orchestrator.py`）**：`paper_id`（corpus 键或
  sha1(title)[:12]）注入数据上下文；FIND_RESOURCES 后自动 fetch、
  COMPLETED 前生成 manifest + 统计；fetch 失败仅告警不阻断流水线。
- **P0-3 隔离依赖安装（`src/agents/code_executor.py`）**：真实模式
  `pip install --target data/deps/<sha1(reqs)>` 一次性安装 + `.ready`
  磁盘就绪标记，执行时经 PYTHONPATH 注入隔离目录；同名依赖清单
  跨论文只落一份天然去重，不污染全局 Python（`AUTOREPRO_DEPS_ROOT`
  可覆盖）。此前真实模式无条件重装 torch（2GB+ 被 timeout 杀）的问题解除。
- **P1-1 共享底座镜像（`src/agents/env_builder.py`）**：`autorepro-base`
  （python:3.11-slim + CPU torch/torchvision/numpy/tqdm + 国内源注入），
  `build_base_image` / `ensure_base_image` 一次构建多论文复用；论文
  Dockerfile `FROM python:*` 自动替换为底座，底座缺失按需构建、失败
  自动降级原 Dockerfile（`degraded` 标注不阻断）。
- **P1-2 数据集注册表（`src/dataset_registry.py`）**：15 类常见数据集的
  别名归一化 / 体积预估 / 子集策略（torchvision 内建懒加载、full、
  percent:N 降采样、synthetic 零数据）/ 国内镜像备注 / 下载入口
  （url: / hf: / script:）；ResourceManager 默认启用，`fetch_dataset`
  按 kind 决策，未知或失败诚实降级合成冒烟集并注明非数值复现。
- **P2 缓存管理 CLI（`scripts/resource_cli.py`）**：`status` / `list` /
  `manifest` / `archive` / `restore` / `prune` / `quota-check` 七子命令
  （argparse 纯净实现）；`prune --yes` 先归档 L1 再清理 L0（不丢数据）；
  `quota-check` 拉取前预检，不足拒绝（退出码 2）并按 LRU 给释放建议；
  `enforce_quota` 新增 `excess_bytes` 支持"预计超限"投影。

### 测试

- 新增 68 用例：存储钩子 8（test_orchestrator_storage）、底座镜像 13
  （test_env_builder_base）、注册表策略 18（test_dataset_registry）、
  CLI 12（test_resource_cli）、隔离依赖与既有资源管理 19 回归。
- 全量 **201 passed, 1 skipped**（修复前 300s 卡死 → 21s 全绿）。

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