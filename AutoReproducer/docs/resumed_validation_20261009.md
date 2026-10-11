# 2026-10-09 续跑验收

接续 [10 月 8 日检查点](implementation_checkpoint_20261008.md) 与 #21、#23、#24。旧实验和原始建议未改写；本次创建独立工作区，重新完成真实训练、API 调用和独立评价。可分发的数值摘要见 [JSON](benchmarks/resumed_validation_20261009.json)，完整数组、检查点、原始建议及日志仍保存在本机 `data/runs/`。

当前运行行为已在 2026-10-10 更新：Store 启动环境会自动切换到标准 CPython，正式预设自动准备和修复缺失环境，详见[自动准备与恢复](cross_platform_runtime.md)。下文 Store 拒绝与环境准备检查保留为 10 月 9 日版本的实测历史，不作为当前操作步骤。

## Neural ODE：确认完成，但未验证稳定提升

运行 `repository_8a04f219d3b44f4db8be053b9d4d0e69`，Windows x64、Python 3.11.9、固定 torchdiffeq commit `657943acefa826ef04c025ebeb1ff5e9d60dc268`、PyTorch 2.5.1 CPU。总耗时 689.360 秒，真实 `deepseek-chat` 调用 1 次。

先完成官方 2000 次迭代拟合，独立 SciPy 复算 MAE **0.4700458448**、RMSE **0.5628648218**，状态 `method_experiment_completed`、`is_reproduced=null`。随后按独立的新增初值协议重训优化基线；训练初值为 `[2,0]`，验证初值为 `[1.5,0]` 和 `[0,1.5]`，两条留出初值为 `[1.75,0]` 和 `[0,1.75]`。所有轨迹在 0–25 区间取 1000 个时间点，以原坐标尺度计算误差。

种子 2021 的优化基线验证 MAE 为 **0.3432119692**。候选只使用验证集比较：

| 单参数候选 | 验证 MAE | 训练与验证耗时 |
|---|---:|---:|
| learning_rate = 0.0003 | 0.2690935735 | 75.375 秒 |
| learning_rate = 0.003 | 0.4746722805 | 93.031 秒 |
| batch_time = 20 | 0.0253652106 | 119.860 秒 |

冻结 `batch_time=20` 后，完成种子 2022 的基线和候选重训，再打开两个种子的留出评价：

| 种子 | 基线留出 MAE | 候选留出 MAE | 事前至少 1% 降低门槛 |
|---|---:|---:|---|
| 2021 | 0.4148808465 | 0.0791055637 | 通过 |
| 2022 | 0.0392793241 | 0.0612942997 | 未通过，误差增大 |

最终 **`tested_no_gain`、`optimized=False`**。验证集改善不足以证明稳定优化，不能因一个种子较好宣称已验证提升，也不使用留出结果继续调参。本次完成昨天缺失的第二种子和全部留出确认；原中断运行 `repository_5eb1e232c85947f5b7b36b21f484fcf3` 保留原样。

## SIREN：四阶段在线分析及真实训练通过

运行 `repository_7485b1a793704487b69b68b14ff0a05e` 使用固定 SIREN commit `4df34baee3f0f9c8f351630992c1fe1f69114b5f` 与已准备的 CUDA 12.1 环境。Reader、Finder、Builder、Verifier 四阶段各调用真实 API 一次，引用校验全部通过；随后完整训练 500 步并独立复算。总耗时 **37.531 秒**，全图拟合 MSE **0.0001835671**、PSNR **37.3620524 dB**。

这是固定图像的官方方法实验，未在本次运行开启参数优化，不属于论文全部基准数值复现。首次验收尝试 `repository_e126add1d180496f95e34e8774589566` 在四次在线分析成功后，因架构字段为空而未找到已准备环境，正确停止在训练之前；修复 Windows 架构识别后重新执行获得上述完整结果。失败记录保留，没有被替换为成功记录。

## 协议与中断修复

- #24：真实建议使用实际优化基线参数与冻结协议，摘要明确 `validation`、指标尺度和验证定义；SIREN 标明 80/10/10 像素切分。原始响应、未修改的建议快照、基线与协议哈希都保存；提示词不含留出指标。
- 本次 Neural ODE 真实响应明确引用新增初值验证 MAE，候选验证计划保留“候选冻结后才评估 holdout，且不继续调参”。三条原始建议仍为 `untested`，后续实验状态没有回写到历史快照。
- #23：运行中、取消、预算不足和真实超时分别保存；强杀后的残留记录可通过运行锁识别，并同步标记结果与报告。Windows 新进程先暂停、纳入 Job 后才恢复执行，阻止普通启动器提前派生子进程。
- 补测确认 Microsoft Store Python 的虚拟环境通过系统代理启动实际解释器，实际进程不属于启动器的 Job。对此配置在执行前明确拒绝，不用部分生效的退出保护冒充完整进程管理；标准 CPython 虚拟环境和本次真实实验使用的直接 Python 启动路径属于支持范围。

## 重跑

先按 README 准备对应环境，在 `AutoReproducer/` 目录运行。API key 使用环境变量或隐藏输入，不写入命令行参数或产物。

```powershell
python scripts/reproduce_repository.py --profile siren_camera_quick --offline --llm-review
python scripts/reproduce_repository.py --profile neural_ode_spiral --offline --optimization validate --max-candidates 3 --budget-seconds 7200
```

每次命令创建新实验；本次实测数值与时间不保证在其他硬件逐位相同。

## 工程回归与交付状态

最终使用本机 Python 3.11.9 直接启动测试，绘图库等测试依赖从本任务专用目录加载，不修改系统安装。两套完整回归均覆盖真实图表、Windows 会话退出、主进程强杀、候选和确认中断及缓存锁恢复：

| 模式 | 结果 | 耗时 |
|---|---|---:|
| Windows 默认区域编码，`PYTHONUTF8=0` | 1312 passed，1 skipped | 255.22 秒 |
| Windows UTF-8，`-X utf8` | 1312 passed，1 skipped | 254.91 秒 |

唯一跳过项是 POSIX SIGTERM 测试，不适用于 Windows；两套各有一条既有 PyPDF2 弃用警告。JUnit 记录保存在本机 `data/reports/regression_20261009_final_locale.xml` 与 `regression_20261009_final_utf8.xml`。

额外使用本机标准 Python 3.12 创建独立虚拟环境，核验启动器及实际解释器均属于同一 Job；此项进程归属检查不等于 Python 3.12 全量回归。Microsoft Store 虚拟环境的实际拒绝路径也已检查，确认依赖安装与训练均未启动。

代码与记录仅保存在本地，未推送或创建 PR。Windows/Linux × Python 3.11/3.12 远端 CI 尚未执行；先交由本地测试，后续再继续该项验收。不能将本机结果当作远端矩阵通过。
