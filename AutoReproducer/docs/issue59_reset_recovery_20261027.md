# Issue #59：卡死任务停止后恢复重置

2026-10-27 检查 [issue #59](https://github.com/Yukikazct/AutoReproducer/issues/59) 时，该 issue 仍为 OPEN，正文和评论为空。已合并的 PR #62、#63 修复了 Windows 的停止按钮、所属 Job 回收与 PDF 解析中断，但最新主分支 `c2c75a7` 的 macOS/Linux 强制停止路径仍无条件写入 `cleanup_unconfirmed`。恢复入口拒绝这种记录，因此任务及同一依赖目录的新任务一直被阻止。

## 本次修复

- 独立 worker 拥有私有 POSIX session；应用管理的嵌套安装器和仓库监督进程使用各自的 process group，并继承该 session。停止时可以找到已经被重新收养的嵌套进程组。
- 保留尚未被 wait/reap 的 worker PID，避免会话标识被复用；macOS 通过 kqueue、Linux 通过 waitid/WNOWAIT 观察退出。
- 强制停止时冻结会话成员，确认成员和进程组稳定后终止，再确认所有非僵尸成员退出。只有成功返回这一回收结果，才释放环境、结束活动作用域并允许重置。Linux 使用 `/proc`，无需安装 `ps`；macOS 使用进程表及 getsid。
- 所有权初始化、进程查询或回收失败继续阻止重启，不允许降级为仅清理外层进程组后宣布成功。正常退出保留原退出码，日志和已有报告继续保留。
- 新 backend 会隔离加载支持该合同的 runtime，避免 Streamlit 缓存旧 runner 后出现不认识 `contain_session` 参数的问题。正在执行的旧模块不会被原地 reload。
- 原先只在 Windows 执行的静默安装器、原生阻塞 PDF 解析测试扩展到 POSIX；同步修复 Streamlit 1.63 的测试 fragment 存储接口兼容性，仍实际重入同一 fragment 两次。

## 验证

| 环境 | 结果 | 覆盖范围 |
| --- | --- | --- |
| macOS ARM64 / Python 3.12.2 | 338 passed，12 skipped | 停止、重置重启、真实卡死解析器和安装器、嵌套进程组、孤儿进程组、环境及所有权生命周期、旧模块升级；跳过项为 Windows 专项 |
| Linux / Python 3.11，现有 `python:3.11-slim` 容器 | 53 passed，3 skipped | POSIX 回收、正常与异常退出、停止任务、隔离 runtime；跳过 Windows 锁文件专项 |
| 静态与环境检查 | 通过 | `git diff --check`、`pip check`；五份原有未跟踪用户文件 SHA-256 保持一致 |

Linux 容器禁用网络，只读挂载现有纯 Python 测试依赖；这组结果不代表完整应用或原生训练依赖验收。macOS 本地环境补齐了最新生产清单已要求的 `markitdown[pdf]`。测试使用规范化的 `/private/tmp` 临时根，避免 macOS `/var`、`/tmp` 软链接触发训练环境的目录保护。

macOS 回归命令（在 `AutoReproducer/` 内）：

```sh
TMPDIR=/private/tmp .venv/bin/python -m pytest \
  tests/test_posix_process_scope.py tests/test_owned_session_runtime.py \
  tests/test_task_cancellation.py tests/test_app_interactions.py \
  tests/test_safe_runtime_preparation.py tests/test_repository_runner.py \
  tests/test_process_lifecycle.py tests/test_preset_bootstrap_imports.py \
  tests/test_live_pipeline_entrypoint.py tests/test_preset_runtime_delegation.py \
  tests/test_nested_execution_cleanup.py tests/test_environment_lifecycle.py \
  tests/test_training_environment_lifecycle.py tests/test_run_lifecycle.py \
  tests/test_repository_run_lifecycle.py tests/test_runtime_entrypoint.py \
  -q -ra --tb=short
```

汇总证据：[issue59_reset_recovery_20261027.json](evidence/issue59_reset_recovery_20261027.json)。以上测试组存在交集，不应相加作为互不重复的测试总数。本轮未运行全套回归或 Windows 实机测试。

## 验收边界与交付

本次确认默认网页本地任务的“卡死 → 停止 → 进程退出 → 清理 → 重置 → 再次启动”流程。Docker 模式的外部容器清理、没有所有权元数据的旧任务，以及作者程序主动调用 setsid 逃离应用管理会话的行为未在本轮验收。会话机制用于管理受信任的运行命令，不构成恶意程序隔离；不能确认回收的记录仍保留保护门禁。

修复分支为 `codex/issue59-posix-reset-20261027`，基于主分支 `c2c75a7` 建立，通过关联 #59 的 PR 提交审查。此处记录的是本地和容器验证；远端持续集成结果以 PR 检查为准。
