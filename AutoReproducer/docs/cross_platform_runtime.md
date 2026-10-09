# 跨平台运行与中断恢复

本轮面向 macOS / Windows / Linux 的通用仓库执行与方法任务稳定性，不改变论文训练配置或硬件要求。SIREN 的 CUDA 预设仍需要 NVIDIA 环境；真实训练和在线 API 验收已按用户要求暂停。

## 运行状态与取消

- 仓库步骤取消时退出码为 `130`，`cancelled=true`、`timed_out=false`，保留已有 stdout/stderr 及声明产物；后续步骤标记为中断未执行。
- 方法任务在命令行捕获 Ctrl+C、POSIX SIGTERM、Windows Ctrl+Break 及启动进程退出；正常取消生成带有中断标识的 `result.json`、`report.md` 和 `run_status.json`。网页后台线程不安装进程级信号处理器。
- macOS/Linux 终止独立进程组；Windows 使用 `taskkill /T /F`。驱动突然退出时，独立监督进程检测持有者退出并终止实验进程树。
- 优化记录使用 `running`、`interrupted`、`insufficient_budget`、`budget_exhausted` 等明确状态。预留确认预算不足不表示时间已经耗尽。中断后已完成基线和候选仍保留，但不据部分确认宣称优化有效。
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

CI 已配置 Linux / Windows / macOS × Python 3.11 / 3.12，Windows 另外验证 UTF-8 模式。上述为推送前的本地验证记录；Windows 等远端平台的验收以对应提交的 CI 结果为准。本轮没有进行真实训练和外部 API 验收。
