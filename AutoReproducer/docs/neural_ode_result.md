# Neural ODE：官方螺旋拟合真实实验

2026-10-08，Windows、Python 3.11、i9-14900HX CPU，从固定作者仓库完成 2000 次训练、独立 SciPy 评估、轨迹/相图/向量场及一次真实 DeepSeek 建议调用。

| 项目 | 实测结果 |
|---|---|
| 运行编号 | `repository_63937713d8f74cce897e57f092729935` |
| 基线流程耗时 | 80.516 秒 |
| 含真实 API 建议的总耗时 | 85.078 秒 |
| 训练耗时 | 69.906 秒 |
| 独立复算轨迹 MAE | 0.4700458448 |
| 独立复算轨迹 RMSE | 0.5628648218 |
| 训练函数调用次数 | 113314 |
| 训练期间及最终轨迹评估函数调用次数 | 248986 |
| 真实 API 调用 | 1 |

## 冻结协议与结论

- 论文：[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)。
- [官方示例固定版本](https://github.com/rtqichen/torchdiffeq/blob/657943acefa826ef04c025ebeb1ff5e9d60dc268/examples/ode_demo.py)。模型 `ODEFunc` 从固定源文件逐字提取，运行完整作者 torchdiffeq 包。
- 目标方程 `dy/dt = y³ @ [[-0.1,2],[-2,-0.1]]`，初值 `[2,0]`，时间 0 至 25，共 1000 个点。
- 原始网络 2→50→2、Tanh、输入立方，RMSprop、学习率 0.001、batch_time=10、batch_size=20，2000 次迭代。固定项目种子 2021。
- 求解器实际为 dopri5，rtol=1e-7、atol=1e-9；不依赖作者脚本中未贯通训练调用的 `--method` 参数。
- 独立评估使用双精度 SciPy DOP853，rtol=1e-10、atol=1e-12，重新生成参考轨迹。

结果为 `method_experiment_completed`、`is_reproduced=null`。这是作者方法示例，不是论文 MNIST 等完整基准，也没有对应论文表格的数值通过结论。

模型学到了螺旋形状，但后段轨迹存在明显相位偏差；训练中的全轨迹误差也有波动。这里保留最终第 2000 次迭代的结果，没有挑选更好看的历史 checkpoint。API 提出的学习率或训练片段长度调整仍须独立验证，不能用建议文字宣称已改善。

## 重跑

```powershell
# 可选：提前完成环境准备和检查
python scripts/reproduce_repository.py --profile neural_ode_spiral --prepare-environment
# 直接运行也会自动准备和检查环境
python scripts/reproduce_repository.py --profile neural_ode_spiral --optimization suggest
```

API 使用本机配置的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。省略优化选项即可不调用 API。默认运行自动取得固定作者源码、准备冻结依赖并做原生健康检查；源码及健康兼容依赖缓存已有时可明确添加 `--offline`，离线缺失或损坏即停止，禁止在线下载安装。原始报告、checkpoint、数组和核验结果位于本机 `data/runs/<运行编号>/`；这些运行产物不随 Git 分发。上方实测仍为 10 月 8 日记录。

新增初值的验证/留出评估属于单独的优化扩展，固定初值、种子和验收门槛，结论不与上述全轨迹拟合误差混用。
