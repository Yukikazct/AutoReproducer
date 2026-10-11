# SIREN 五分钟快速档：本机真实验收

2026-10-08，在 Windows、Python 3.11、RTX 4060 Laptop GPU 8GB 上，从三个独立工作区重新训练，完成独立指标复算、图表、报告和真实 DeepSeek 建议。环境提前准备，计时包含每次源码导出及核验、设备导入、500 步训练、评估和一次建议 API 调用。

| 运行 | 总耗时 | PSNR | MSE | 真实 API 调用 | 建议状态 |
|---|---:|---:|---:|---:|---|
| `repository_40d042b874f64105876ccc0020f8f340` | 28.844 秒 | 37.362052 dB | 0.000183567063 | 1 | suggested |
| `repository_d3fe395c1c3941129285777e49239bf8` | 28.391 秒 | 37.362052 dB | 0.000183567063 | 1 | suggested |
| `repository_53f3cb8258574604889e572c08664ae0` | 28.500 秒 | 37.362052 dB | 0.000183567063 | 1 | suggested |

三次均满足 300 秒与 PSNR ≥25 dB 的事前工程门槛。固定种子 2021；这是重复运行与耗时验收，不是跨随机种子的泛化结论。原始运行目录保存在本机 `data/runs/`，不随 Git 分发；机器可读摘要见 [验收记录](benchmarks/siren_quick_20261008.json)。

## 实验范围

- 论文：[Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)。
- 官方仓库固定为 `vsitzmann/siren@4df34baee3f0f9c8f351630992c1fe1f69114b5f`。
- 模型从官方 `explore_siren.ipynb` 第 3 个代码单元（JSON 下标）逐字提取；固定 notebook 的 LF 内容哈希和导出文件实际字节哈希均记录。
- 输入为 scikit-image 0.16.2 的 cameraman，SHA256 `361a6d56d22ee52289cd308d5461d090e06a56cb36007d8dfc3226cbe8aaa5db`；使用作者的 PIL/torchvision 预处理缩放至 256×256。
- 保留官方初始化、256 宽度、`hidden_layers=3`、首层/隐藏层频率 30、Adam、学习率 0.0001、500 步。
- 现代兼容环境：PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.26.4、Pillow 10.4.0。CUDA 实际可用且依赖从独立缓存导入。
- 适配增加固定种子、结构化参数、checkpoint、预测数组和日志；将训练中反复交互绘图移至末尾独立评估。

PSNR/MSE 衡量本张训练图像的拟合。结果为 `method_experiment_completed`、`is_reproduced=null`；25 dB 不是论文公布的验收阈值，不能据此宣称整篇论文数值复现。

## 重跑

在应用目录可直接运行快速档，系统会先自动准备和检查环境。也可提前预热，单独完成源码、数据、依赖及设备检查：

```powershell
python scripts/reproduce_repository.py --profile siren_camera_quick --prepare-environment
```

使用已在本机配置的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 执行快速档：

```powershell
python scripts/reproduce_repository.py --profile siren_camera_quick --optimization suggest
```

不需要 API 时省略 `--optimization suggest`，仍会产生真实基线与图表。`--prepare-environment` 是可选预热，`--prepare-only` 仍仅准备源码、数据与命令。已有可核验源码、数据及健康兼容依赖缓存时可明确添加 `--offline`；离线缓存缺失或损坏时停止，禁止在线下载安装。

网页选择本预设，可直接选择“运行实验”和“真实基线 + 智能建议”，或先用“准备实验环境”预热。完整模式可额外启用运行前四阶段在线分析；其 API 等待不属于上方 10 月 8 日快速档验收结果。

## 建议与失败行为

建议仅可改变事前允许的一个学习率或首层频率参数，每条包含假设、成本、验证方法和可逐字核对的作者来源。所有建议均为尚未验证；`optimized=False`。真实优化需要独立训练/验证/留出协议及两种子确认。

建议 API 以子进程强制限时 45 秒，失败保留基线。正式运行自动安装缺失的冻结依赖、检查原生包并重建坏缓存，准备阶段的依赖缓存锁最多等待 60 秒；默认公共镜像不可用时回退官方源，参数与版本约束不变。训练超时不会降低分辨率、步数或复用历史产物。上方三次耗时属于 10 月 8 日已准备环境的记录；其他设备与网络条件须重新实测，首次安装不在五分钟承诺内。
