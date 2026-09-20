# CHANGELOG

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。
版本号采用 `YYYY.MM.DD-<序号>`（按发布批次日期的扁平版本）。

---

## [2026.09.20-13] - 2026-09-20

### 修复（Docker 沙箱把 /tmp 挂成 noexec，C 扩展全部加载失败）

用户启动 Docker Desktop 后首次真正走加固沙箱，报：

```
ImportError: /tmp/site-packages/numpy/_core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so:
failed to map segment from shared object
```

**根因**：加固参数 `--tmpfs /tmp:rw,nosuid,size=256m` 没写 `exec`，而
Docker `--tmpfs` 的默认挂载选项是 `rw,nosuid,nodev,noexec`（本机实测
`mount` 输出确认）。加固模式又恰好把依赖 `pip --target` 装进
`/tmp/site-packages` —— C 扩展的 `.so` 需要 `mmap(PROT_EXEC)`，noexec 下
直接 EPERM。**凡是带 C 扩展的包（numpy/torch/scipy…）都装得进去、导不进来**，
用户看到的是一长串 numpy 安装建议，像是论文代码或环境坏了。

本机真实容器实测（Docker 29.7.2，`--read-only` + 非 root + 同一组加固参数）：

- 往 /tmp 拷二进制执行 → `Permission denied`（126）；
- 装 numpy 后 import → 复现上述报错，`PY-RC=1`；
- **同一条命令把挂载选项改成 `rw,exec,nosuid,size=256m` → `numpy ok 2.4.6`，`PY-RC=0`**。

**为什么此前没暴露**：引擎一直没启动，加固沙箱这条路在这台机器上从未真正
跑过（[2026.09.20-11] 修的正是「引擎没起却假装就绪」）。引擎一启动，第一个
带 C 扩展的依赖就撞上。

**改法**（`src/agents/code_executor.py::_sandbox_args`）：
`--tmpfs /tmp:rw,exec,nosuid,nodev,size=256m`。保留 `nosuid`/`nodev`：要挡的
是 setuid 与设备节点，不是「执行刚装进来的库」——容器里跑的本来就是不可信
代码，它本来就要被执行，这点上不构成新的攻击面。

**兜底**：`_HARDEN_INCOMPATIBLE_HINTS` 增加 `"failed to map segment"`——成因
已修，但别的机器/别的 Docker 版本仍可能给出 noexec 的 tmpfs，命中即按既有
降级链退到 level 2（无 tmpfs，`pip --target` 落到可执行的可写层）继续跑，
而不是把基础设施问题报成「论文代码失败」。

**验证**：走**生产代码路径**（`CodeExecutorAgent._execute_code_docker`，
`image_tag=python:3.11-slim` + `requirements_txt=numpy>=1.24`）真实跑容器：
`success=True, exit=0, sandbox={level: 0, degraded: False}, stdout="numpy ok 2.4.6"`
——完整加固、零降级。

**测试**（`tests/test_sandbox_hardening.py` +2）：`test_tmpfs_mount_allows_exec`
锁死挂载选项（判据按逗号切分取成员——`"exec" in "noexec"` 是子串为真，
用字符串 `in` 判断会恰好把这个 bug 判成「有 exec」）；
`test_degrade_on_tmpfs_noexec` 覆盖他机 noexec 时逐级降级到 level 2 跑通。
`test_full_hardening_present` 增补 exec 断言。

---

## [2026.09.20-12] - 2026-09-20

### 回滚 / 修复（深色 IDE 面板回滚；docker 输出在中文 Windows 上丢输出）

**1. 回滚深色 IDE 面板**（用户反馈：「回滚掉 ide 修改吧 我现在都看不到代码了」）

[2026.09.20-10] 引入 `frontend/markdown_render.py`，把报告里的围栏代码块
渲染成自绘深色面板。**它只被 AppTest 断言过 HTML 字符串、并在独立预览页里
看过，从未在真实 Streamlit 页面里打开验证**——用户打开页面看到的是代码
不可见。自绘 HTML 这条路到此为止：原生 `st.markdown` 至少是「能看见」的。

删除 `frontend/markdown_render.py`、`tests/test_markdown_render.py`、
`tests/test_app_report_tab.py`，以及 `app.py` 里的 `.autorepro-code*` /
`.autorepro-plain` / 行号 / 滚动条 CSS 段与 `render_markdown` 导入，Tab 2
恢复 `st.markdown(report)`。

**保留**同批次里的「不再截断」（那是用户明确提的「能不能不截断啊」，与
面板渲染是两件独立的事），排版代价回到「长输出会把页面拉长」，接受。

**2. docker 路径的 GBK 解码丢输出**（回滚验证时全量测试暴露）

```
PytestUnhandledThreadExceptionWarning: Exception in thread Thread-4 (_readerthread)
UnicodeDecodeError: 'gbk' codec can't decode byte 0xaf in position 4124
```

[2026.09.20-3] 已修过同一根因，但当时只覆盖了**本地执行**路径的
`subprocess.run`：`env_builder` 的 `docker build` / `docker images`、
`code_executor` 的 `docker run`、`base_agent` 的引擎探测都还是
`text=True` 不带 encoding —— 中文 Windows 上按 GBK 解码 UTF-8 的 docker
输出，非 GBK 字节让 reader 线程抛 `UnicodeDecodeError`、`stdout` 变成
**None**。两处后果：

- `env_builder`：`result.stdout[-300:]` 抛 `TypeError: 'NoneType' object is
  not subscriptable`，被兜底 `except Exception` 吞掉 → 用户读到的构建失败
  原因是这行 TypeError，而不是 docker 真正报的错；
- `code_executor`：None 直接进 `final["stdout"]`，`report_generator` 的
  `"\n".join(lines)` 崩（`dict.get("stdout", "无输出")` 对「键存在且值为
  None」不生效）→ 报告页整块渲染不出来。

**改法**：四处 docker 调用补 `encoding="utf-8", errors="replace"`；
`stdout`/`stderr` 统一 `or ""` 兜底；`report_generator` 新增 `_txt()` 并
把三个会崩的 join 位点（`requirements_txt` / `code` / `final["stdout"]`）
改为经它取出。本机 Docker Desktop 此刻是启用的（用户已启动），这条路径
不再是纸面问题。

**3. 一个靠机器状态才通过的测试**

`test_docker_not_required_for_mock` 断言「无 Docker 时构建诚实失败」，却
没构造「无 Docker」—— 它靠**本机 Docker 恰好不可用**才通过。引擎一启动，
降级路径真去 `docker build`（本机 python:3.11-slim 已在本地，几秒构建成功），
断言翻车，全量耗时也从 48s 涨到 307s。改为显式
`monkeypatch.setattr(agent, "_resolve_docker_cmd", lambda: None)`，
并删掉文件里从未被引用的 `_NoDocker` 辅助类。

**测试**：`test_exec_output_encoding.py` +1（docker run 捕获必须声明
UTF-8）；`test_env_builder_base.py` +2（docker build 捕获声明 UTF-8、
stdout 为 None 时报的是构建失败而非 TypeError）；`test_code_generation_
completeness.py` +1（stdout/requirements 为 None 时报告仍渲染）。

---

## [2026.09.20-11] - 2026-09-20

### 修复（Docker「已就绪」是假的：CLI 在 PATH ≠ 引擎在跑）

用户反馈真实模式跑复现时报
`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`。

**根因**：`app.py` 的就绪判定是 `shutil.which("docker") is not None`——
只证明 **CLI 二进制在 PATH 上**，不证明 **Docker Desktop 的引擎在跑**。
本机实测正是这个组合（CLI 29.7.2 在 PATH、引擎未启动），于是侧边栏显示
「✅ Docker 已就绪」，`use_docker=True` 传进流水线，`docker run` 秒失败，
最终用户读到的是一坨 npipe 原始报错（报告已改为全文内嵌，这条错误整段
糊在脸上）。执行层也没有任何兜底：`_execute_code_docker` 返回
`success=False` 就结束，不存在「引擎不可用时降级」的路径。

**探测（`src/base_agent.py::BaseAgent.docker_engine_available`）**：

- 跑 `docker version --format {{.Server.Version}}`，只有引擎在线才有输出；
- **为什么不用 `docker info`**：本机实测引擎未启动时 `docker version`
  **188ms** 即失败返回，而 `docker info` 要 **20.7s** 才返回——后者会让
  Streamlit 每轮重跑冻住 20 秒。探测命令的选择本身是性能决策，已用
  测试钉死（断言实际执行的 argv 就是 `version --format ...`）；
- 复用既有 `_resolve_docker_cmd` 探测链（`DOCKER_PATH` → PATH → 常见
  安装目录），顺带修掉一处旧不一致：README 承诺的「Docker Desktop 常见
  安装目录自动探测」此前在 UI 层从未生效（UI 只看 PATH）；
- 失败原因分三类人话：「未安装」/「探测超时」/「引擎未启动」，
  可直接展示给用户。

**UI（`app.py` 侧边栏）**：真实模式才探测（Mock 不执行代码、用不上
Docker）；结果缓存进 `docker_probe`，引擎不可用时**把开关拉回关闭**并
禁用（显示开着却跑不了是最坏的组合），文案改为「⚠️ {原因}，将使用本地
隔离执行」；新增「🔄 重新检测 Docker」按钮（启动 Docker Desktop 后一键
刷新，无需刷新页面）。

**执行层守卫**（漏网时给人话而非 npipe 报错）：

- `code_executor._execute_code_docker`：进沙箱前探测，不可用直接返回
  `exit_code=EXIT_DOCKER_DAEMON_DOWN(-4)` + 「请启动 Docker Desktop 后
  重试；或在侧边栏关闭「Docker 沙箱执行」改用本地隔离执行」。位置在
  **镜像白名单之后**——镜像否决是关于镜像本身的安全判定，不该被「引擎
  没起」掩盖；`-4` 经测试确认不与既有 `-1/-2/-3/-5/-6` 语义冲突；
- `env_builder.build_image` 与收口点 `_build_dockerfile`：引擎不可用即
  诚实报错，不再先跑一次必然失败的底座探测、打一条误导性的「降级为
  python slim」日志、最后把 npipe 报错写进构建结果。

**不做**：daemon 中途挂掉时静默改走本地执行。用户显式勾了沙箱，降级必须
由用户拍板（沙箱语义不能让基础设施故障悄悄改掉）。

**测试（644 → 659）**：新增 `tests/test_app_docker_gate.py`（5 例，AppTest
驱动真实 app.py：不谎报就绪 / 开关禁用并强制关闭 / 可用时报就绪 /
重新检测真的重新探测 / Mock 不探测）；`test_architecture.py` +5 例覆盖探测
函数本身（命令锁定、daemon 未起、exit 0 但无版本号、超时、无 CLI）；
`test_sandbox_hardening.py` +3 例（exit code 不冲突、引擎不可用零调用且
报错可操作、镜像否决不被引擎状态掩盖）；`test_env_builder_base.py` +2 例。
另给 4 个既有测试文件加引擎打桩夹具——真实探测会让用例结果随本机
Docker 状态漂移，`test_usage_metering` 还会因多出一条 `subprocess` 调用
记录撞坏 `len(calls) == 1` 断言。

---

## [2026.09.20-10] - 2026-09-20

<!-- 本条的「深色 IDE 面板」部分已在 [2026.09.20-12] 回滚（面板在真实浏览器
里代码不可见）；「全文不再截断」部分保留有效。 -->

### 新增 / 修复（报告代码块：深色 IDE 面板 + 全文不再截断）

用户两条反馈指向同一处（Tab 2 复现报告）：

1. **「把代码显示改成 ide 那种」**——报告由 `st.markdown(report)` 整体渲染，
   围栏代码块只是灰底 `<pre>`：无高亮、无行号、长输出看不出边界；
2. **「输出过长，此处仅截断展示前 6000 / 10627 字符…能不能不截断啊」**——
   `_clip` 把 code/stdout/stderr 砍到 6000/6000/3000 字符，实测一次
   10627 字符的输出在报告里只剩前 6000，读报告的人拿到的是半截内容。

**不再截断**（`src/agents/report_generator.py`）：删掉 `_clip` 与三个
`*_SHOW_LIMIT`，代码/执行输出/错误输出**全文内嵌**。截图里那句话不再出现。
排版代价由前端消化——代码块渲染成**固定最大高度 + 内部滚动**的面板，
多长都不会把页面撑开。`*_execution.txt` 附件保留（改为「可直接下载原文的
纯文本旁路」，不再承担「补全被截断内容」的职责）。

**深色 IDE 面板**（新文件 `frontend/markdown_render.py`）：

- `split_fenced_blocks` 逐行状态机把报告切成「文本段 + 代码段」：结束围栏
  要求反引号数 ≥ 开围栏（```` 裹 ``` 不会被拦腰截断），**未闭合围栏把剩余
  内容整段当代码**——宁可多渲染，绝不丢内容；4 空格缩进不被误判为围栏；
  代码段内容原样保留（不 `strip`，缩进是代码语义）；
- `code_block_html` 走**服务端 pygments**（monokai + 行号 + 语言标题栏 +
  溢出滚动）。无语言标签的执行输出走纯文本、**不加行号**——终端里报错行号
  不该被平移；
- pygments 默认转义全部内容 → LLM 生成的内容无法注入 HTML（有测试钉住）；
- `render_markdown`：文本段仍走 `st.markdown`，代码段走面板；仅当确实存在
  代码段时才下发样式表。

**为什么不用原生 `st.code`**（它自带高亮和行号）：浅色主题下 token 颜色是
react-syntax-highlighter 写死的内联样式，CSS 改不出稳定的深色面板；也拿不到
「标题栏 + 滚动区」这套套壳结构。

**踩到并修掉的坑**：pygments 只给 token 打**类名**（`<span class="kn">`），
颜色在主题样式表里，**必须显式注入**——不注入的话满屏 span 一个颜色都不变，
看起来「有高亮」其实等于没有。这是最难自查的一步，已用测试钉住
（`test_pygments_css_defines_token_colors`），并由 AppTest 断言页面上真的
下发了样式表。另外 monokai 自带面板底色，优先级与本项目外壳相同且位置更靠
后，已显式压成透明，否则标题栏与代码区会出现两种深色。

**测试**（全量 **644 passed**，新增 23 例）：

- `tests/test_markdown_render.py`（新，19 例）：切分器（交错/语言标签/
  缩进围栏/未闭合/长围栏/CRLF/真实报告形状重建不丢行）、面板 HTML
  （高亮+行号、输出块无行号、未知语言退化、**注入转义**、max-height、
  一万字符不截断）、样式表（定义了 token 颜色、只作用于代码块、压掉
  monokai 底色）；
- `tests/test_code_generation_completeness.py` +1：8000 字符代码 +
  20000 字符输出进报告，**完整包含末行**且无「截断」字样；
- `tests/test_app_report_tab.py`（新，3 例，AppTest 真跑 app.py）：
  代码块确实渲染成面板（而不是退回裸 Markdown）、正文仍是 Markdown、
  500 行输出在页面上完整出现。

---

## [2026.09.20-9] - 2026-09-20

### 修复 / 新增（依赖缓存：看得见、认得出、单独清）

接上一条的用户追问：「依赖缓存应该和之前的数据一起被删除吗？」

**不该**——`data/deps/` 是按依赖清单内容哈希寻址的**跨会话共享**热缓存，
删掉后下次执行同一依赖要重新下载安装（numpy+matplotlib 那档就是 123 MB），
把它并进「清空历史」等于让用户点一次「清空」赔一次重装。但也不能不管：
它**不随历史记录增长，只随论文依赖增长**，且改造前**没有任何元数据**，
230 MB 摊在那儿连自己装的是什么都说不出。

于是分三件事：

**① 目录自带元数据**（`src/agents/code_executor.py`）：安装成功后在隔离
目录写 `meta.json`（`kind` / `installed_at` / `last_used` / `requirements`
/ `normalized` / `module`）+ 一份 `requirements.txt`。文件写到目录**里面**
是刻意的——目录整体删掉元数据跟着走，不会出现「清单还在、目录没了」的
对不上的状态。写失败只 `pass`：元数据是可观测性，不能影响安装与执行。
命中缓存时 `touch_deps_meta` 只刷新 `last_used`，不动 `installed_at`。

**② 清单归一后再哈希**（同上）：`reqs_digest` 改为对
`normalize_requirements(reqs)` 取 sha1。归一只折叠**写法**差异——去空行、
去纯注释行、去首尾空白、保序去重、按包名排序（大小写不敏感）——不改变
pip 的解析结果。例外：清单里含 `-r` / `-c` / `--index-url` 这类**选项行**
时顺序有意义（选项只对其后的行生效），此时只去重去空白、**不排序**。

实测本机就有两个内容等价、各 **53.48 MB / 1553 文件**的 numpy 目录
（`50410a7b4faa3413`、`9e50c492c254da58`），只因清单文本不同（行序/多一行
注释）就各存一份完整依赖。

**③ 独立的清理入口**（`frontend/history_manager.py` + `app.py` Tab 5）：
新增 `list_deps_cache` / `delete_deps_cache` / `cleanup_deps_cache`，
在「🧹 清理管理」里加一块「📦 依赖缓存」，**不进「清空历史」**：

- 列出每个目录的 名称 / 类型（依赖清单 · 自愈补装 · 旧目录）/ 包含的包 /
  占用 / 最后使用（旧目录回退到目录 mtime，否则永远清不掉）；
- 按目录多选删除，同样需要勾确认（沿用「危险操作」模式）；
- 「保留最近 N 天」按 `last_used` 清冷缓存——这是将来真要腾空间的唯一入口。

`history_manager` 刻意**不 import `src.*`**（保持纯 stdlib，前端加载更轻），
故 `_DEPS_META_NAME` 两边各写一份常量，用一条一致性测试钉住，而不是跨层 import。

**测试**（全量 **621 passed**，新增 16 例）：

- `tests/test_code_executor_deps.py` +5：归一折叠写法差异、内容不同不折叠、
  选项行不排序、等价清单只落一份目录且命中时刷新 `last_used`（并锁死
  `installed_at` 不被改写）、heal 目录元数据记 `module`；
- `tests/test_history_manager.py` +9：读元数据/包名（**测试当场抓到包名带了
  版本号**，改成从右切一次）、旧目录回退 mtime 且类型为 legacy、按
  `last_used` 倒序、根目录不存在返回空、只删点名的、未知名字返回 `(0,0)`、
  **拒绝路径穿越/嵌套路径**（`../outside`、绝对路径、`inside/../inside`）、
  冷热清理只删冷的、跨模块常量一致；
- `tests/test_app_history_tab.py` +2：面板列出缓存目录、**未选/未确认时
  删除按钮禁用、确认后只删点名的那个且确认框自动复位**（AppTest 真跑 app.py）。

---

## [2026.09.20-8] - 2026-09-20

### 新增（历史记录：批量删除 + 筛选）

用户实测前端后要求「删除历史复现记录」。排查发现**单条删除早已存在**
（每个会话展开后的「🗑️ 删除本会话」，`delete_session` 完备且有单测），
但历史里躺着 **332 条会话**（大量 Dummy / 异常中断的 RUNNING 残留），
一条条展开→勾选→删除基本不可用。经确认需求是**列表上直接多选、一次删掉**，
外加筛选把测试残留找出来。

**后端**（`frontend/history_manager.py`）：

- 新增 `delete_sessions(session_ids) -> (删除数, 释放字节)`：输入会话 id
  保序去重，取各自 `_related_files` 的并集后再一次性 `_unlink_files`；
  空列表/全部未知会话返回 `(0, 0)`。复用既有 `_related_files` /
  `_unlink_files`，不新增扫描逻辑；
- `_session_id_from_progress` 拆出带缓存的包装：按「路径 + mtime + 大小」
  缓存解析结果（后两者只作缓存键；progress 是追加写的，内容一变键即失效，
  不会返回过期结果）。原来每处理一个会话都要重扫一遍全部 progress 文件，
  332 条会话约 10 万次开文件。实测全量扫描 **0.39s**。

**前端**（`app.py` Tab 5）：

- 筛选行：状态（全部 / COMPLETED / ERROR / RUNNING / 未知 / 其他，
  「其他」兜住状态机中间态）+ 标题子串（不区分大小写、纯 `in`、无正则）；
- 每行勾选框放在 **expander 之外**（窄列 0.04 + 内容列 0.96），批量删除
  不必逐条展开；
- 「☑️ 全选可见」/「▫️ 清除选择」，已选条数实时显示；
- 可见列表含 RUNNING 时给出提示：多为异常中断残留，但若该复现仍在运行，
  删除会丢记录（占用中的文件自动跳过）；
- 底部批量删除沿用既有「危险操作」模式：加粗标题 + 确认勾选 +
  `type="secondary"` 按钮，未勾确认或未选中时禁用。

**三条 Streamlit 运行时语义**（源码级核实，决定了上面的写法）：

1. **widget 实例化之后不能再改它的 session_state**（抛
   `StreamlitAPIException`）——所以「全选/清除」按钮必须放在勾选框**之前**，
   删除后的确认框复位必须延后到下一轮、在 checkbox 创建前执行；
2. **勾选值在 widget 实例化时才写入 session_state**——所以「已选 N 条」与
   删除目标必须在**渲染循环之后**统计，放前面会滞后一次交互；
3. `RerunException` 继承自 `BaseException`，因此 try/except 不会吞掉
   `st.rerun()`。

删除目标恒为 **可见 ∩ 已勾选**：被筛选隐藏的会话即使之前勾过也不会被删
（AppTest 已锁死这条性质）。

### 顺带修复（删除后的提示被 rerun 吃掉）

原代码一律是 `st.success(...)` 紧跟 `st.rerun()`——**当次运行的提示会被
新一次运行整棵树丢弃，用户看不到任何反馈**。AppTest 驱动时表现为
`success` 列表为空，确认是真丢。四处（runtime 清理 / 清空全部 / 单条删除 /
批量删除）统一改为写 `_hist_flash` 到 session_state，下一次运行在 Tab 顶部
弹出。

**测试**：`tests/test_history_manager.py` 新增 5 例（多删、不误伤、
空/未知、去重、progress 按内容关联）；新增 `tests/test_app_history_tab.py`
6 例，用 Streamlit `AppTest` **真的跑起 app.py、真的点按钮**——覆盖
勾选框渲染、筛选收窄、未确认时按钮禁用、**勾选后被筛选隐藏的会话不被删**、
提示跨 rerun 存活且确认框复位、全选可见只勾可见。全量 **605 passed**
（新增 11 例）。

### 待用户执行

真实 `data/` 里有 332 条会话，批量删除后共可释放 **23.57 MB / 754 个文件**。
dry-run 只列了清单，**删除动作没有执行**，留给用户在页面上操作。

---

## [2026.09.20-7] - 2026-09-20

### 修复（相对工作区把脚本路径拼成两遍）

接上一条：为了确认真实优化的 trial 为什么全是 `rejected`，读了一次真实
trial 的 `detail` —— `stdout_tail` 为空、`metric=None`、`reward_basis={}`，
而 `exit_code` 是 **2**（`NOENT`）。stderr 里是：

```
python: can't open file 'D:\...\data\_e2e_ws\data\_e2e_ws\run.py':
[Errno 2] No such file or directory
```

**根因**：`_execute_code_local` 把脚本以
`[python, os.path.join(workdir, "run.py")]` 启动，**同时** `cwd=workdir`
（[code_executor.py:770](AutoReproducer/src/agents/code_executor.py#L770)）。
`workdir` 是相对路径（`workspace_dir="data/_e2e_ws"` 由调用方传入）时，
相对路径的脚本参数会被这个 cwd **再解析一次**，于是去找
`data/_e2e_ws/data/_e2e_ws/run.py`。

这条 bug 与真实优化是**同一处**：只有走真实执行才会传入工作区目录，
所以它一直被哈希模拟掩盖着——模拟不执行任何代码，自然不暴露。

危险之处在于**表象具有误导性**：`exit_code=2` + 空 stdout 看起来就像
"生成的代码/补丁跑不起来"，会被误判成代码质量问题（正是用户反复反馈
的那一类），而实际是执行器自己的路径拼接错。

**改法**：`_execute_code_local` 与 `_execute_code_docker` 在使用外部传入的
`workdir` 时统一 `os.path.abspath()` 一次（Docker 侧同样是 `-v` 挂载被相对
路径坑），临时目录分支本就走绝对路径，不受影响。

**验证**：修复前踩坑的工作区（`data/_e2e_ws`，numpy 依赖）现在真实执行
`success=True, exit_code=0`，stdout 正常产出指标；补齐后的真实优化闭环
首次跑通——trial 能拿到真实指标与 `reward_basis`。

**测试**：`tests/test_code_executor_deps.py` 新增
`test_relative_workdir_is_not_double_joined`。用例把工作区建在**仓库内**
（`tempfile.mkdtemp(dir=os.getcwd())`）——pytest 的 `tmp_path` 落在 C: 盘、
仓库在 D: 盘，跨盘构不出相对路径，`os.path.relpath` 会直接抛
`ValueError: path is on mount 'C:', start on mount 'D:'`。用完即删。

---

## [2026.09.20-6] - 2026-09-20

### 修复（模拟优化的数字被当成实测结果展示）

复用样例论文做端到端验证时发现：真实模式下，**优化阶段根本没真跑**，
但报告把它当实测结果呈现。

```
## 6. 智能优化
- **优化状态**: ✅ 已优化
- **改进幅度**: 1.80%
- **最优结果**: 0.0908056
```

**根因**：真实优化执行器只在 `Orchestrator.workspace_dir` 存在时才注入
（[orchestrator.py:83](AutoReproducer/src/orchestrator.py#L83)），而
**`app.py` 调用 `run_pipeline_background` 时从不传 `workspace_dir`**
（[app.py:626](AutoReproducer/app.py#L626)）——于是前端跑起来时
`workspace_dir` 恒为 `None`，`RealSimulator` 永不注入，优化**永远**走
哈希模拟 `OptimizerAgent._simulate_trial`。

那个模拟函数是拿**方向名的 md5** 当"改进潜力"（
[optimizer.py:270](AutoReproducer/src/agents/optimizer.py#L270)）：

```python
h = int(hashlib.md5(arm.encode("utf-8")).hexdigest()[:8], 16)
return round((((h % 10000) / 10000.0) - 0.35) * 0.15, 4)
```

与生成的代码、与指标方向**都无关**。实测那一跑里：

| 方向 | 模拟结果 | 相对基线 0.0892 | 判定 |
|---|---|---|---|
| 岭回归/Lasso 替代 OLS | 0.096764（更差） | +8.48% | ✅ Keep |
| QR/SVD/Cholesky 求解 | 0.090806（更差） | +1.80% | ✅ Keep |

MSE 越小越好，两条都**变差**却都被判为"改进"并 Keep——因为模拟奖励是
哈希值，压根不知道指标方向。

**改法**（`src/agents/report_generator.py`）：把模拟与实测在报告里分开。

- 优化状态按记录里的 `detail.type` 区分：全为 `ucb_mock` 时显示
  **`⚠️ 已优化（模拟）`** 并加一条醒目说明——"以下改进幅度/最优结果由
  方向名哈希模拟得出，不是跑出来的实测值，不能作为代码改进效果的依据"；
- 「改进幅度」「最优结果」两处数值各带 **`（模拟）`** 后缀；
- 尝试记录表新增**「依据」**列：`真实执行` / `⚠️ 哈希模拟`；
- `real_exec` 记录不受影响，仍显示 `✅ 已优化`。

**测试**：`tests/test_optimizer_real.py` 新增
`TestReportDisclosesSimulatedOptimization` 3 例（模拟须标注、数值须带
后缀、真实执行不得被误标）。

### 顺带修复（真实优化的奖励方向会悄悄退化）

排查上面那条时发现 `RealSimulator` 里有同一族的键匹配 bug：`bind_paper`
把论文声明的**原始大小写**键存进 `_metric_key`（
[real_simulator.py:78](AutoReproducer/src/optimizer/real_simulator.py#L78)，
`next(iter({"MSE": ...}))` → `"MSE"`），而 `_compute_reward` 拿它去
`in metrics` 比对，`metrics` 的键是 `_extract_metrics` 提取的小写 `mse`
→ **永远不匹配** → 静默退回"取与基线绝对值最接近的指标"这条启发式，
连奖励方向（`_LOWER_IS_BETTER`）都跟着按启发式选中的键来定。

后果是**该 Keep 的被 Reject、该 Reject 的被 Keep**，例如论文声明 `F1`、
输出同时有 `f1_score: 0.86` 与 `accuracy: 0.84` 时，可能与 `accuracy`
比、按"越大越好"判。

**改法**：

- `bind_paper` 存归一化键（复用 `src/metric_keys.py`，与判定层同源）；
- `_compute_reward` 按归一化键匹配，命中即用声明指标；
- 新增 `_reward_basis()`，把「用了哪个键 / 键从哪来（论文声明 or 启发式）
  / 方向」写进 trial 的 `detail.reward_basis`；
- 报告尝试记录表的「依据」列据此补注：`真实执行 · mse↓`，启发式选键时
  标 `真实执行 · loss↓，启发式选键`。

`_compute_reward` 的返回值由 `(reward, metric)` 改为
`(reward, metric, basis)`（私有方法，仅一处调用）。

**测试**：`tests/test_optimizer_real.py` 新增 `TestRewardMetricResolution`
5 例 —— `MSE`/`mse` 须匹配上、MSE 变大必须得负奖励、`Accuracy` 这类
越大越好的指标方向不变、启发式选键须如实标注、trial detail 须带
`reward_basis`；另在报告族补 2 例（真实执行须显示指标与方向、启发式
选键须标注）。

> **未修（需决策）**：`app.py` 不传 `workspace_dir` 意味着**前端永远
> 不做真实优化**。真正的修法是让 app 为每个会话分配工作区并传入，
> 但那会让优化阶段真实重跑 `max_trials` 次（每次一次 LLM 调用 + 一次
> 沙箱执行），显著改变运行时长与成本，且与待排期的「Batch 5 可编辑
> 代码面板」共用同一套工作区生命周期——单独拿出来做决策更稳妥。
> 本轮只保证：**要么真跑，要么说清楚没真跑**。

---

## [2026.09.20-5] - 2026-09-20

### 修复（指标键大小写不匹配导致复现结果误判 —— 真实模式实测发现）

第三条由真实模式实测暴露的缺陷。pip 修好后代码第一次真正跑起来，输出：

```
MSE: 0.0869           <- 实际运行
论文声明: {'MSE': 0.0892}
→ status: not_reproduced, is_reproduced: False, confidence: 1.0
```

打眼一看是"没复现"，但**两者只差 2.6%**，而判定阈值是 5%。

**根因**：`_extract_metrics` 把输出里的指标名一律归一成小写（`MSE` →
`mse`），而 `paper_metrics` 直接来自论文解析结果、保留原大小写。随后
`_local_compare` 用 `akey == key` 做**精确字符串比对**，`"MSE" != "mse"`
于是判定"论文声明了但运行输出未提取到该指标"，`matched == 0` 直接把
`match_all` 置 False，**veto 掉 LLM 的 match=true**。

比误判本身更糟的是**理由写反了**：报告里写"运行输出未提取到该指标"，
而实际上值就在 stdout 里、也成功提取到了——只是键名大小写不同。

**改法**：

- 新增 `src/metric_keys.py::norm_metric_key()`：键归一为「小写 + 空白/
  连字符折叠成下划线」。`MSE`/`mse`/`F1 Score`/`f1-score` 等写法差异
  不再构成"不同指标"；归一化**不合并真正不同的指标**（`rmse` ≠ `mse`）；
  归一键冲突时取首次出现值（`setdefault`），保证确定性；
- 判定层（`_local_compare`）与展示层（`report_generator` 的指标对比表）
  **共用同一条规则**——此前报告表格也按精确键名配对，会把 `MSE` 与
  `mse` 排成两行、各缺一半，看起来同样像"没跑出来"；
- 报告新增「指标差异」小节，把逐项差异（声明 X vs 实际 Y，相对差异 Z%）
  如实列出。原先只给"成功/失败"结论而不给差异依据，用户无从判断判定
  是否合理。

**顺带修掉一处自相矛盾的汇报**：`confidence` 原先无条件透传 LLM 的置信度，
于是"LLM 说 match=true/置信度 1.0、本地数值说不匹配"这种情况会输出
**「未复现 + 置信度 1.0」**——读起来像"确定没复现"，实际是两个判据打架。
现在两判据结论相反时：

- 在 `validation.verdict_sources` 如实记下 `{llm, local}` 各自的结论；
- `differences` 追一条"判据分歧"说明；
- 置信度按**最弱一环**取 `min(模型置信度, 本地置信度)`。

`_local_compare` 原先在一次 `run()` 里被调用两遍（兜底分支一次、取交集
一次），现降为一次。

**测试**：`tests/test_reproduction_core.py` 新增两族 10 例 ——
`TestValidatorKeyNormalization`（大小写/分隔符变体、真实 E2E 那一跑的
形状、`rmse`/`mse` 不得混同）与 `TestValidatorVerdictHonesty`（分歧须
上报且降置信度、一致时不改动模型置信度）。

### 修复续（判定标准未写进 prompt，导致 LLM 判定随机横跳）

上一条修完后重跑真实模式：本地数值判 **匹配**（2.6% < 5%），但 LLM 判
**不匹配**，两者取交集仍报未复现。翻账本发现真正的问题：

| 运行时刻 | 输入（完全相同） | LLM 判定 |
|---|---|---|
| 12:58:56 | 声明 0.0892 / 实际 0.0869 | `false` |
| 13:00:34 | 同上 | `true` |
| 13:01:16 | 同上 | `true` |
| 13:02:34 | 同上 | `false` |

**同样的数字，LLM 在 true/false 之间反复横跳。** 根因是比对 prompt
**只给了输出格式、没给判定标准**——没有阈值、没说键名大小写不算差异、
没说声明值出现在输出里时不能当运行结果，模型只能凭感觉判。而最终判定
是「LLM ∧ 本地」的交集，一次随机的 false 就能否决掉正确的结论。

**改法**（`result_validator.py`）：把判据显式写进 prompt，与本地规则同源：

1. 以"提取到的实际指标"为准；输出里若同时出现论文声明值（复现脚本自己
   打印了一行对照），不得当成运行结果；
2. 指标名大小写/分隔符差异不构成不同指标；
3. 逐项算相对差异，`≤ 5%` 视为一致（阈值直接取 `_TOLERANCE` 插值，
   不写死数字，改阈值时 prompt 跟着走）；`> 5%` 不一致；声明了但确实
   没跑出来视为不一致；
4. `match=true` 当且仅当**所有**声明指标都一致；
5. 分析里必须给出每个指标的相对差异百分比，不许只说"接近"。

**实测验证**：固定输入（声明 0.0892 / 实际 0.0869）重复调真实 API 5 次，
改 prompt 前是 `false/true/true/false` 横跳，改后 **5/5 稳定判为复现成功**，
且模型开始显式列出 `相对差异 = |0.0869-0.0892|/0.0892 = 2.6%` 并引用规则条款。

---

## [2026.09.20-4] - 2026-09-20

### 修复（隔离依赖安装被站点级 pip 配置阻断 —— 真实模式实测发现）

第二条由真实模式实测暴露的缺陷。上一个提交修好解码崩溃后，真实流水线
第一次跑到了「装依赖」这一步，随即失败：

```
exit_code: -4
依赖安装失败(exit=1): ERROR: Can not combine '--user' and '--target'
```

**根因**：本机 Python 安装带了一份 **site 级 `pip.ini`**，里面写死
`install.user = yes`（`python -m pip config debug` 可见 `site: install.user: yes`）。
隔离安装用的是 `pip install --target data/deps/<hash>/`，而站点配置又强制
追加 `--user`，两者互斥，pip 直接报错退出。

影响面比它看起来大：`--user` 是**配置**而非命令行传入，所以这份配置存在时，
**本机真实模式的依赖安装 100% 失败**，进而所有真实模式的代码执行都跑不起来
（表现为恒定的 `not_runnable`），且失败被缓存进 `_INSTALLED_DEPS`，
看起来像"论文依赖有问题"。

**改法**（`src/agents/code_executor.py`）：

- 新增 `_pip_env()`：pip 子进程环境注入 `PIP_USER=0`（环境变量优先级
  高于配置文件）并钉 `PYTHONIOENCODING=utf-8`；
- `_ensure_local_deps` / `_heal_install_local` 两处 pip 命令行显式加
  `--no-user`，与上面的环境变量形成双保险；
- 两处调用点从"继承 `os.environ`"改为显式传 `env=self._pip_env()`。

两种修法都已在本机实测：`--no-user` → exit=0；`PIP_USER=0` → exit=0。

**测试**：`tests/test_code_executor_deps.py` 新增 3 例 —— 隔离安装命令行
必须含 `--no-user`、pip 子进程环境必须 `PIP_USER=0`、自愈补装走同一套参数。

---

## [2026.09.20-3] - 2026-09-20

### 修复（本地执行输出解码崩溃 —— 真实模式实测发现）

**这是本轮唯一由真实 API 实测暴露的缺陷**，首次真实模式运行即崩：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa2 ...
TypeError: 'NoneType' object is not subscriptable
  at code_executor.py: "stdout_tail": full.get("stdout", "")[-300:]
```

**根因**：`_run_local_script` 用 `subprocess.run(..., text=True)` 且**未指定
encoding**，父进程按系统 locale（Windows 中文 = GBK）解码捕获到的输出；
而子进程的标准流编码受 `PYTHONIOENCODING` 影响。两者不一致时 reader 线程抛
`UnicodeDecodeError`，`CompletedProcess.stdout` 变成 **None**；随后
`full.get("stdout", "")[-300:]` 崩溃——**key 存在且值为 None 时，`dict.get`
的默认值不生效**。

触发条件很常见：真实 LLM 生成的复现脚本会打印中文（"训练集 MSE: ..."），
而本轮提示词又明确**允许中文字符串**，等于把这个坑踩实了。

**改法**（`src/agents/code_executor.py`）：
- `_exec_env()` 设 `PYTHONIOENCODING=utf-8`——把子进程标准流钉死为 UTF-8，
  与父进程解码一致，不再依赖系统 locale；
- `_run_local_script` 显式 `encoding="utf-8", errors="replace"`；
- 两处 pip（依赖预装 / 运行时自愈）同样补 `encoding`——pip 输出也可能含
  非 GBK 字节；
- `stdout`/`stderr` 统一 `or ""` 兜底，杜绝 None 流向下游；
- 三处会崩的下标切片改为 None 安全：`code_executor` 的
  `smoke.get('stderr','')[:120]`、`full.get("stdout","")[-300:]`，
  以及 `src/agents/env_builder.py` 的 `build.get("stderr","")[-300:]`。

### 真实模式实测结果（首次端到端）

用真实 DeepSeek API 跑 numpy 线性回归复现：

| 观测项 | 结果 |
|---|---|
| LLM 调用次数 | **1**（一次生成即完整，未触发续写/重生成） |
| `finish_reason` | **`stop`** |
| 代码规模 | 2381 字符 / 101 行 |
| `sanitize_stats` | `{prose_dropped: 0, code_dropped: 0}` —— 清洗**一个字都没丢** |
| 语法门 / 结构门 | 通过 / 通过 |
| 执行 | `exit_code=0`，正常打印指标（含大量中文，全部保留） |

**续写机制专项验证**（人为把 127 行脚本截断到 76 行）：模型续写首行恰好是
被截断的末行（遵守契约），拼接后 104 行、可编译、结构完整、末行 `main()`，
重写行仅出现 1 次（去重生效）。

**旁证**：该次生成**未使用代码围栏**（模型没遵守提示词里的围栏要求），
但 `prose_dropped=0 / code_dropped=0`——说明保真清洗的"只丢确定是叙述的行"
这条改法本身就能兜住无围栏输出，不必依赖模型守约。

### 测试

- 新增 `tests/test_exec_output_encoding.py`（5 例：中文输出被完整捕获且为
  `str`、emoji 等非 GBK 字符不丢、stdout/stderr 永不为 None、
  `_exec_env` 钉死 UTF-8、整条 `run()` 不因中文输出崩溃）。
- 全量 **563 passed**。

---

## [2026.09.20-2] - 2026-09-20

### 修复（清洗层静默篡改合法代码——「代码不完整」的真正残余通道）

上一轮（`05a4ab3`）给清洗层加了"丢弃行计数 + WARNING"，但**只记日志不补救**，
复查发现真正的静默篡改仍在，且已实测复现：

```
输入:  a = 1                       输出:  a = 1
       *x, y = [1, 2, 3]      ->          print(y)
       print(y)                          可编译、结构完整、无重试
执行:  NameError: name 'y' is not defined     （沙箱跑起来才炸）
```

```
docstring 内容被篡改:
       """说明                          """说明
       *   ) paren               ->          """
       *   * bullet                         可编译、能跑，字符串内容被悄悄改写
       """
```

**根因**：`_CODE_LINE_START` 是一份"**看起来像代码行**"的正则**白名单**，
清洗层用它**决定丢弃**。白名单天然不完整——实测 `)`、`]`、`}`、`{`、
`) as f:`、`*args,`、`**kwargs):`、`*x, y = ...`、`~mask`、`%`、`!`、`;`、
`|`、`\` 等**合法 Python 行首**全部被误删；而**删掉中间若干行后剩下的代码
往往仍能编译**，语法门与结构完整性门双双放行，于是残缺无痕通过。

**改法**（`src/agents/code_executor.py`）：

- **清洗改为三级，保真优先**：
  1. 有围栏且可编译 → 整块原样返回（不变）；
  2. 否则**只丢"确定是叙述"的行**（`_drop_prose_only`，复用显式判据
     `_looks_like_prose`），其余**一律保留**——不再靠"像不像代码"猜；
  3. 保留版仍编译不过，才启用白名单过滤兜底（`_filter_code_lines`），
     并把丢弃行**单独计数**为 `code_dropped`（此前这一步完全不计数）。
- **返回值区分两类丢弃**：`(code, {"prose_dropped": n, "code_dropped": m})`。
  `prose_dropped` 是预期行为（剥叙述），`code_dropped` 才是"可能把代码洗残了"。
- **`code_dropped > 0` 触发补救**：`_needs_continuation()` 把它作为第一判据，
  让模型重写，而不是带着缺损继续跑。
- **`_stitch` 收敛对模型约定的依赖**：原**无条件丢弃 head 末行**，模型只要
  不遵守"先重写末行"的约定就会静默吞掉一行；现新增 `_line_incomplete()`
  （数括号 + 悬挂尾字符），**只在末行确实没写完时**才丢弃。`:` 不算悬挂——
  `for i in range(3):` 是完整行，丢掉它会毁掉循环。
- **报告如实展示清洗记录**：`report_generator` 在「代码执行」段落显示
  「⚠️ 清洗丢弃：N 行疑似代码行」，让"代码是否被清洗动过"可见。

### 修复（优化补丁路径缺语法门）

`src/optimizer/real_simulator.py` 的补丁生成**完全没有语法门**，且
`_PATCH_PROMPT` 至今仍**禁止代码围栏**（必走有损过滤）：补丁生成后直接
写盘 + 执行，运行期才炸 SyntaxError → 一次 trial 白烧，残码还落在了工作区。

- `_PATCH_PROMPT` 与生成 prompt 对齐：要求单个 ```python 围栏、要求完整脚本、
  允许中文字符串；
- 写盘前加语法门（复用 `executor._syntax_error`）：不通过则该次 trial 直接
  `status="rejected"`，**不写盘、不执行**。

### 测试

- `tests/test_code_generation_completeness.py` 扩至 34 例，新增：
  - **三条实测静默篡改的回归锁定**（星号解包、docstring 内容、多行调用收尾
    括号）——清洗后必须一字不少；
  - `code_dropped` 触发补救、`prose_dropped` 不触发；
  - `_line_incomplete` 判定（悬挂运算符/少右括号为真，块首行与完整语句为假）；
  - `_stitch` 保留完整末行（模型未守约时不吞行）；
  - 报告三态展示（代码丢弃告警 / 仅叙述不告警 / 无统计不打扰）。
- `tests/test_optimizer_real.py` 新增：补丁不可编译时必须被拦下且
  **工作区未被写入**。
- 全量 **555 passed**。

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