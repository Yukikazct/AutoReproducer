# 跨平台运行与中断恢复

仓库执行与方法任务共用自动准备、隔离缓存和中断管理，保留论文训练配置与硬件要求。SIREN 的 CUDA 预设仍需要 NVIDIA 环境。

## 自动准备与恢复（2026-10-10）

网页选择“运行实验”或直接运行 `scripts/reproduce_repository.py` 后，系统先检查运行环境、固定作者源码、数据和隔离依赖，全部通过后才开始训练。`--prepare-environment` 是可选的提前预热操作；`--prepare-only` 仍只生成源码、数据与计划，不安装训练依赖。

- **Python 与应用依赖**：检测 Microsoft Store 启动环境、非标准或不兼容解释器、缺失及无法导入的应用依赖，自动寻找健康的 64 位 CPython 3.11/3.12 环境；需要时创建项目专用虚拟环境。64 位 Windows 没有兼容解释器时，下载 python.org 官方安装程序并验证 Python Software Foundation 数字签名后安装到项目缓存。其他系统需要已有兼容 CPython。已有启动环境和全局依赖不被改写。
- **固定源码与数据**：源码缓存需要通过来源、完整 commit SHA 和 `git archive` 读取检查；坏对象在线重新获取相同 SHA，并保留原坏缓存。真实数据按固定字节数和 SHA-256 校验，坏缓存在线重新下载、验证后原子替换。临时传输失败最多尝试 3 次，失败不会变成合成数据或历史训练结果。
- **公开作者证据**：DLinear 作者评论 REST 请求遇限额或网络失败时，自动读取同一个官方 issue 网页内嵌的原始 Markdown，核验仓库、issue、comment ID/URL 和作者成员身份。原 HTML、派生评论和哈希分别保留，离线复用时重新提取核验；正文缺失或身份不符会停止分析。可选 `AUTOREPRO_GITHUB_TOKEN`、`GITHUB_TOKEN` 或 `GH_TOKEN` 只发送给固定 GitHub REST URL，不进入分析 packet、日志或报告。
- **训练依赖**：每次运行检查冻结版本、导入及关键原生运算，损坏或上次未安装完整的独立依赖缓存会自动重建。默认公共镜像连接失败或超时时，最多回退官方 PyPI 一次，保留相同版本约束；明确配置的私有来源继续使用其既定策略。安装和健康检查结果记录在运行环境证据中。

明确指定 `--offline` 时，源码、数据及公开证据不下载，缺失或无法核验即停止；运行环境恢复不下载新解释器，应用依赖只从本地 wheel 准备。冻结训练依赖必须已有健康兼容缓存，缺失或损坏时停止，禁止在线安装或重建。明确启用在线分析或优化建议仍访问 LLM API。自动恢复有有限次数和超时，外部来源持续不可用或硬件不满足时保留具体失败证据。

Windows Store 启动环境的完整 DLinear 验收已通过：运行 `repository_b707311a3bb84f9db886af456a3780b5` 自动切换到 `.venv-py312`，恢复 GitHub REST 403 的作者证据，依赖镜像超时后从官方源安装并通过 Python 3.12 原生健康检查。完整作者训练、协议核验与独立复算得到 MSE **0.3841444**、MAE **0.4047131**，均在原 5% 容差内，状态 `reproduced`。证据见[报告](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/report.md)和[最终结果](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/result.json)；这一验收覆盖该项 Windows CPU 实验。

## 运行状态与取消

- 仓库步骤取消时退出码为 `130`，`cancelled=true`、`timed_out=false`，保留已有 stdout/stderr 及声明产物；后续步骤标记为中断未执行。
- 方法任务在命令行捕获 Ctrl+C、POSIX SIGTERM、Windows Ctrl+Break 及启动进程退出；正常取消生成带有中断标识的 `result.json`、`report.md` 和 `run_status.json`。网页后台线程不安装进程级信号处理器。
- macOS/Linux 终止独立进程组；Windows 使用 Job 管理实验后代，兼容清理路径也可使用 `taskkill /T /F`。运行环境准备和切换后的训练子进程均接受取消及超时管理。
- 优化记录使用 `running`、`interrupted`、`budget_insufficient`、`budget_exhausted` 等明确状态。预留确认预算不足不表示时间已经耗尽。中断后已完成基线和候选仍保留，但不据部分确认宣称优化有效。
- 依赖缓存仍按 Python ABI、系统和处理器架构隔离；运行期间持有缓存锁，取消后可重新获取锁。

强制终止进程不能保证它在退出前生成报告。对此使用下面的显式恢复命令，不把未完成记录当成成功，也不自动续跑训练。

## 检查强制终止后的记录

在项目目录、激活虚拟环境后，两端使用相同命令：

```text
python scripts/recover_run.py data/runs/repository_<运行编号>
```

命令通过操作系统文件锁检查是否仍有活动持有者。任务仍在运行则只返回 `running`；持有者已退出时，将未完成的 `running` 记录标为 `interrupted`，原始文件保存到 `recovery_originals/`，恢复说明保存到 `recovery.json`。完整试验、指标及原始 API 建议不改写。如果最终 `result.json` 已经写入，则优先依据其真实终态恢复状态。

旧版运行没有 `run_status.json` 和持有者锁，命令返回 `unknown_owner`，保留原记录供人工核对。恢复操作只整理状态，不承诺训练断点续跑。

## 安装与测试

推荐 Python 3.11 或 3.12。macOS/Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest tests/ -q -ra
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.venv\Scripts\python.exe -m pytest tests/ -q -ra
.venv\Scripts\python.exe -X utf8 -m pytest tests/ -q -ra
```

测试无需真实 API Key 或训练依赖。部分 API 连接测试会在 `127.0.0.1` 上启动模拟服务，需要允许监听本地端口。无 Windows 符号链接权限（WinError 1314）时，相关测试明确记为跳过，不能解读为该能力已验证。

普通 Python 程序不会提前导入 matplotlib；实际使用 matplotlib 时才安装中文字体回退，避免字体扫描耗尽普通短任务的超时预算。

## 验证记录（2026-10-09）

本机环境：macOS ARM64、Python 3.12.2。专项回归首轮 112 项通过，包括真实小型进程的取消、启动进程退出、强制终止及中文绘图。全量回归 **1321 passed，1 warning**（101.66 秒）；警告为既有 PyPDF2 弃用提醒。结果保存在本机 `data/test-results-macos-20261009.xml`，不随仓库分发。

全量检查后，补充了实际预算耗尽边界与 Windows 控制台进程组隔离：方法/恢复相关 56 项、执行器/恢复相关 54 项针对性回归均通过；最后新增的操作系统取消信号实测也通过（macOS 使用 SIGTERM，Windows CI 对应 Ctrl+Break）。这些检查有重叠，不与全量数量相加。

CI 已配置 Linux / Windows / macOS × Python 3.11 / 3.12，Windows 另外验证 UTF-8 模式。上述为 10 月 9 日推送前的本地验证记录，该阶段没有进行真实训练和外部 API 验收；10 月 10 日 Windows DLinear 的真实自动恢复验收见本页前节。远端平台回归仍以对应提交的 CI 结果为准。
