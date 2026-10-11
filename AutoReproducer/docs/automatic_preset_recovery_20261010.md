# 预设自动恢复验收（2026-10-10）

本次修复让网页和 CLI 在执行预设前自动准备兼容 Python、固定源码、真实数据和隔离依赖。缺失或损坏的准备资源按原版本恢复，训练参数、设备要求、作者协议和指标门槛保持不变。

## 修复内容

- GitHub REST 限额时，从固定官方 issue 网页读取同一作者评论的原始 Markdown；验证定位、作者身份与缓存哈希。
- Store Python、应用依赖缺失或导入失败时，自动切换到经过验证的标准 CPython 3.11/3.12，继续同一进度记录。
- 网页每轮检查后台入口的能力版本。Streamlit 保留旧模块时，在独立命名空间加载当前入口，保证新任务进入自动环境准备，同时保留已有执行线程的全局对象。
- 依赖缓存每次检查版本、导入和原生运算；损坏或安装未完成时在线按相同依赖清单重建。默认公共镜像失效时回退官方源。
- 自动准备阶段最多等待缓存锁 1800 秒，等待和安装不占训练预算；显式离线时只使用健康兼容缓存。
- 预设安装器和健康检查由独立进程树管理，超时先清理后代再重试，保留有上限的诊断输出。CUDA 官方安装最多 1800 秒；公共镜像首次尝试仍为 300 秒。
- LLM 密钥通过私有管道交给网页 worker；训练子进程不继承 LLM/GitHub 凭据。CLI 在切换到非交互 worker 前处理隐藏输入。

## 真实验收

均从原 Store `.venv` 启动，自动切换 `.venv-py312`，没有先执行手动环境准备命令。

以下五项为独立入口验收，当时未覆盖已经运行数小时的 Streamlit 服务。持续运行服务的实际启动事件验收见下一节。

| 预设与场景 | 结果 | 本地证据 |
|---|---|---|
| DLinear：新 Python ABI 的依赖缓存缺失，GitHub REST 403，公共镜像超时 | 四阶段公开分析共 4 次；完整作者训练、协议核验、独立复算通过。MSE 0.3841444、MAE 0.4047131，均在原 5% 相对容差内 | [结果](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/result.json) |
| DLinear：另一项预设正在准备依赖，已有健康缓存 | 自动排队后完整训练通过；零安装，缓存健康检查通过，指标与上次一致 | [结果](../data/runs/repository_9f8cc9ee773848939618397da59b4014/result.json) |
| Neural ODE：CPU 依赖缓存缺失 | 自动安装并完成 2000 次迭代；质量、协议、独立复算通过。MAE 0.4700458、RMSE 0.5628648 | [结果](../data/runs/repository_eb7de6cd10ce4b9b89961b9bdf71bf55/result.json) |
| SIREN：前次 CUDA 安装未完成，没有就绪标记 | 自动重建并完成 CUDA 500 步；质量、协议、独立复算通过。MSE 0.0001835671、PSNR 37.3621 dB | [结果](../data/runs/repository_ee82363faa4b4eaea092ea93f813aea9/result.json) |
| SIREN：明确离线，复用健康 CUDA 缓存 | 零安装，健康检查和 500 步训练通过，指标与上次一致；总耗时 38.531 秒 | [结果](../data/runs/repository_cb633ab5d99e4f42aba15c7b9623d27e/result.json) |

DLinear 的结论只覆盖选定的 ETTh1 336→96 论文实验。SIREN 与 Neural ODE 为 `method_experiment_completed`、`is_reproduced=null`，表示完成官方方法实验，未宣称论文表格数值复现。

Neural ODE 首次调用总耗时 280.781 秒，其中依赖准备 162.312 秒、训练 98.765 秒；SIREN 重测总耗时 206.469 秒，其中依赖准备 171.438 秒、训练 14.109 秒。两次均关闭公开 API 分析、结果摘要解释和优化建议，未向 LLM 发送本次实验结果。SIREN 的本次基线成功不替代包含真实建议 API 的五分钟完整流程验收。

## 持续运行网页服务验收

用户在 18:19–18:20 的三次运行仍然失败：[最近一次](../data/runs/repository_4c5caf4447ac434a866133ca93a04810/result.json)。进度包含新的依赖准备阶段，却完全没有 `prepare_runtime`，最后触发 Store Python 的执行拦截。健康新进程加载磁盘入口时能正确交接，原服务 PID 16544 则从 15:40:36 持续运行，保留了不包含交接逻辑的旧后台入口。

修复后没有重启或终止该服务。直接连接 `localhost:8501` 的 Streamlit 应用 WebSocket，从服务返回的控件 ID 设置 DLinear 完整实验、关闭 Mock/Docker/两项 API 分析选项，并提交实际“开始复现”事件。三次均进入 `prepare_runtime`、使用 `.venv-py312`、完成真实训练，得到 `validation.status=reproduced`，MSE 0.3841443956、MAE 0.4047131240；API 调用均为零。

第二次结果：[完整记录](../data/runs/repository_501c5f8c17324f4683fb38415820f800/result.json)。最终验收：[完整记录](../data/runs/repository_baee9c6c44bd44c2981547e7c306f6e6/result.json)、[进度](../data/runtime/progress_1791628254973.jsonl)、[服务验收摘要](../data/runtime/live_web_acceptance_20261010.json)。最终刷新同一测试会话时，服务实际返回“完整实验已完成，论文数值验收通过！”；没有 Store 执行拦截，流水线状态为任务完成。

这是现有网页服务器的实际事件与页面输出验收。当前工具没有可用浏览器控制通道，未宣称操作或刷新用户原有的浏览器会话；原失败报告保留在历史记录中。

## 保留的失败记录

修复前的并发 DLinear 运行在等待依赖缓存锁 60 秒后停止：[原始失败](../data/runs/repository_e5d0f33210c5488c86b9d34e90ca8367/result.json)。该记录虽然流程状态为 `COMPLETED`，其 `validation.result_level=failed`；流程结束不能视作验收成功。

首次 SIREN 在两轮安装各达到 300 秒上限后停止，尚未训练：[原始失败](../data/runs/repository_6ab549b122514b91ace26f9706e1a4ee/result.json)。Windows 虚拟环境启动器的后代超时清理和 CUDA 官方安装预算随后修正，真实重测记录单独保存。

失败记录未删除，成功结果来自后续实际训练。运行产物保存在本机 `data/`，不随 Git 分发。

## 回归验证

最终完整测试 **1679 passed、1 skipped、0 failed**，耗时 144.57 秒。唯一跳过项为 Windows 不适用的 POSIX SIGTERM 行为；图像生成测试已实际执行。唯一警告为既有 PyPDF2 弃用提醒。JUnit 保存在本机 `data/runtime_final_pytest.xml`。

覆盖固定输入重试与完整性校验、评论恢复及离线重验、损坏依赖重建、缺应用依赖时的 worker 交接、实际安装器/健康检查子孙超时清理、并发缓存等待、离线禁止联网安装、密钥传递与训练环境隔离、取消和中断记录。测试数量不与此前专项检查相加。

持续运行网页入口修复后，已有页面、后台交接和导入专项 **58 passed**；新增旧入口缓存专项 **8 passed**。新增测试覆盖旧模块隔离升级、并发初始化、失败清理，以及真实 AppTest 启动按钮经实际后台线程进入安全 worker，校验 DLinear 运行与 SIREN 环境准备参数。此轮未重跑上述完整测试，66 项专项与此前完整测试分别报告。
