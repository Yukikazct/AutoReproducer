# 2026-10-08 实施检查点

按用户要求，提交已完成工作并停止实验。本轮代码仅提交到本地，尚未推送；不据部分交付关闭 issue。

## 已完成与实测

- 适配接口：`94fe92b`。
- SIREN 快速档与真实建议：`73a5875`。三个独立工作区总耗时 28.844 / 28.391 / 28.500 秒，PSNR 均为 37.362052 dB；见 [快速档验收](siren_quick_result.md)。
- Neural ODE 官方方法实验：`79e7a46`。完整 2000 次迭代，独立 MAE 0.4700458448，总耗时 85.078 秒；见 [真实实验报告](neural_ode_result.md)。
- 有限参数搜索及两种子留出确认：`59c000d`。
- 网页、报告结论与 CLI 边界修正：`eb112b9`。

这些案例使用 `method_experiment_completed`、`is_reproduced=null`，不宣称整篇论文数值复现。

## SIREN 优化实测

运行 `repository_66c617897d734a5b9f11eeb820b57bcb`，总耗时 160.344 秒，真实 API 调用一次。使用种子 1729 划分 80%/10%/10% 像素，重新训练基线；以下结果属于像素留出协议，与官方全图拟合指标分别报告。

优化基线验证 PSNR 为 34.211680 dB。三个单参数候选全部保留：

| 候选 | 验证 PSNR | 试验耗时 | 结果 |
|---|---:|---:|---|
| 学习率 0.00005 | 31.708047 dB | 21.062 秒 | 未超过基线 |
| 首层频率 15 | 32.518561 dB | 21.078 秒 | 未超过基线 |
| 首层频率 60 | 35.714247 dB | 21.188 秒 | 验证集选定 |

选定首层频率 60 后，完成第二种子的基线和候选重训，再进行留出确认：

| 种子 | 基线留出 PSNR | 候选留出 PSNR | 提升 |
|---|---:|---:|---:|
| 2021 | 34.431862 dB | 35.912437 dB | +1.480575 dB |
| 2022 | 34.269290 dB | 35.905840 dB | +1.636550 dB |

两种子均超过事前 0.1 dB 门槛，测量结果为 `validated_gain`。结论限于这张图片的固定像素划分。建议提示词存在协议上下文问题（[#24](https://github.com/Yukikazct/AutoReproducer/issues/24)）；保留原始建议，不能将其文字解释当作正确的实验协议说明。

## Neural ODE 优化已中断

运行 `repository_5eb1e232c85947f5b7b36b21f484fcf3` 的三个候选比较已完成。新增初值验证集基线 MAE 为 0.3432119692，候选如下：

| 候选 | 验证 MAE | 试验耗时 |
|---|---:|---:|
| 学习率 0.003 | 0.4746722805 | 87.469 秒 |
| 学习率 0.0003 | 0.2690935735 | 72.734 秒 |
| batch_time 20 | 0.0253652106 | 117.094 秒 |

验证集选定 batch_time 20，第二种子基线已完成，但候选训练被停止；两种子留出确认未完成，不能宣称优化有效。实验进程树已终止，保留已有产物；本地记录人工补记为 `interrupted`、`optimized=False`，没有补造最终运行结果。中断状态及子进程退出问题见 [#23](https://github.com/Yukikazct/AutoReproducer/issues/23)。

## 检查与剩余工作

收尾针对性回归：85 passed，11.62 秒；`git diff --check` 通过。

```powershell
python -m pytest tests/test_method_optimization.py tests/test_method_experiments.py tests/test_app_interactions.py tests/test_reproduce_repository_cli.py tests/test_repository_analysis_report.py tests/test_report_renderer.py tests/test_app_report_tab.py -q
```

待后续继续：修复 #23/#24、完成 Neural ODE 留出确认、本轮新增功能的 Windows 默认编码和 UTF-8 全量回归、Windows/Linux × Python 3.11/3.12 CI 验收、真实四阶段在线分析验收，以及同步 README 和原实施计划。CI 配置已存在，但本轮未推送或触发远端检查。

使用已准备环境及本机 API 配置，可重新开始独立优化运行：

```powershell
python scripts/reproduce_repository.py --profile siren_camera_quick --offline --optimization validate --max-candidates 3 --budget-seconds 7200
python scripts/reproduce_repository.py --profile neural_ode_spiral --offline --optimization validate --max-candidates 3 --budget-seconds 7200
```

每条命令创建新实验，不会续跑这次中断的工作区。原始 checkpoint、数组和详细记录保存在本机 `data/runs/`，不随 Git 分发。
