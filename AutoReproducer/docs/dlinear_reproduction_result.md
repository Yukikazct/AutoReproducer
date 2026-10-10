# DLinear / ETTh1 336→96：真实复现与汇报材料

2026-10-07 已从网页使用的生产后台入口完成一次新的真实运行：**5 次真实 DeepSeek API 分析 → 作者完整训练 → 最终测试 → 独立协议核验与指标复算 → 带图报告**。Mock 关闭，使用真实作者源码和 ETTh1 数据。最终状态为 `reproduced`，LLM 分析状态为 `completed`。

2026-10-10 增加 Windows 自动恢复验收：`repository_b707311a3bb84f9db886af456a3780b5` 从 Store 启动环境自动切换到标准 `.venv-py312`，GitHub REST 403 的评论通过同一官方 issue 原文恢复，依赖镜像超时后从官方 PyPI 安装原冻结版本并通过原生健康检查。四阶段公开分析实际调用 4 次，完整 CPU 训练、协议核验及独立复算通过，MSE **0.3841444**、MAE **0.4047131**，两项均在原 5% 容差内，状态 `reproduced`。本次未启用结果摘要解释；见[报告](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/report.md)和[结果](../data/runs/repository_b707311a3bb84f9db886af456a3780b5/result.json)。以下五次 API 调用与 macOS 环境细节保留为 10 月 7 日的历史记录。

范围为 AAAI 2023 论文 **Are Transformers Effective for Time Series Forecasting?** 的 Table 2 中 **DLinear、ETTh1、多变量、输入 336 步、预测 96 步**这一项。结论不扩展为整篇论文全部实验已经复现。

## 汇报结论

| 指标 | 论文 Table 2 | 本次真实测试与独立复算 | 相对差异 |
|---|---:|---:|---:|
| MSE | 0.375 | 0.3841443955898285 | 2.44% |
| MAE | 0.399 | 0.40471312403678894 | 1.43% |

两个指标满足项目事前采用的 **5% 相对容差**。这是项目验收规则，**不是论文公布的阈值**，没有按本次结果调整。源码、数据、完整协议和独立复算同时通过后才给出本项数值复现结论；API 的文字解释不决定通过与否。

适合汇报的表述：

> 系统用固定作者源码和真实 ETTh1 数据完成 DLinear 多变量 336→96 实验。五个分析任务均实际调用 API，训练按官方早停完成；从预测数组和原始数据独立复算得到 MSE 0.38414、MAE 0.40471，相对论文偏差为 2.44% 和 1.43%，满足事前项目容差。结论限定为该实验，环境和初始化版本差异已记录。

## 方法与实际协议

DLinear 用窗口 25 的移动平均把输入拆为趋势与剩余分量，再分别通过线性层从 336 步映射至 96 步，两路相加形成预测。本项含 7 个变量，使用作者默认共享权重。执行的是完整官方仓库的 `run_longExp.py`。

| 设置 | 实际值 | 依据 |
|---|---|---|
| 方法 / 数据 / 预测任务 | DLinear / ETTh1 / 多变量 96 步 | 论文 Table 2、作者脚本 |
| 输入窗口 / 分解窗口 | 336 / 25 | 论文 Appendix B.2、作者模型 |
| batch / 初始学习率 | 32 / 0.005 | 作者 `etth1.sh` |
| 最大轮数 / patience | 10 / 3 | 作者入口默认值 |
| seed / 重复次数 | 2021 / 1 | 作者入口固定 seed、脚本 `itr=1` |
| 优化器 / 调度 | Adam / 作者 `type1` | 固定作者训练代码 |
| checkpoint 选择 | 验证集损失最优 | 作者训练实现 |
| 运行设备 | 本地 CPU，`num_workers=0` | 实际执行记录 |

论文参数、作者脚本设置和代码默认值分别标注。作者“论文只跑一个 seed”的说明本身不能证明原论文的 seed 数值就是 2021。

本次连续训练 **epoch 1–7**，每轮 256 个 batch；第 7 轮出现 `EarlyStopping counter: 3 out of 3` 和 `Early stopping`。验证集最优 checkpoint 来自第 4 轮，作者入口加载它后完成最终测试。官方早停结束属于完整训练协议。

真实 CSV 有 17420 行、2589657 字节。前 8640 行训练、验证至 11520 行、测试至 14400 行，标准化只拟合训练部分。实际训练 / 验证 / 测试窗口为 **8209 / 2785 / 2785**；最终预测 shape 为 **`(2785, 96, 7)`**。独立重建真实测试标签后复算的 MSE/MAE 与作者最终日志在 `1e-6` 内一致。

## 五次真实 API 调用

| 角色 | 实际工作 | 调用 / 返回 / 引用接受 | 总 tokens |
|---|---|---|---:|
| PaperReader | 阅读论文表头、目标行、作者脚本及默认值，抽取协议并区分来源 | 1 / 完成 / 通过 | 25409 |
| ResourceFinder | 核对固定作者仓库与模型、切分、训练、指标定义、命令入口 | 1 / 完成 / 通过 | 27315 |
| EnvBuilder | 核对原始依赖，区分文档声明与未执行的兼容建议 | 1 / 完成 / 通过 | 27624 |
| Verifier | 独立审查前三份分析与公开原文是否一致 | 1 / 完成 / 通过 | 28887 |
| ResultValidator | 分析获授权的指标摘要，核对差异计算并解释限制 | 1 / 完成 / 通过 | 8125 |

合计 **5 次真实调用、117360 tokens、零引用修正重试**，计数来自实际客户端差量。输入包含 **16 份真实公开来源**；接受的引用均核验为对应来源的原文。前四次未发送本地训练结果；第五次只发送 MSE/MAE、完成轮数、协议与独立复算状态及指标派生差值，不发送日志、路径、CSV 或 checkpoint。

LLM 不改变冻结命令、作者源码、参考指标或容差。训练执行、指标计算和报告生成由程序完成，无需为了增加调用数而额外请求模型。

## 原始记录与报告

本次运行 `repository_e9408141e3dd44319e854bbdf4a70c0c` 于 2026-10-07 16:10:26 启动，约 38.1 秒完成。已复用校验通过的公开资料与依赖缓存，时间不包含首次下载和安装。所有必需执行步骤退出码为 0。

以下 `data/` 链接指向本地运行产物，不随代码仓库分发；按重跑步骤可在自己的运行目录生成对应报告和核验记录。

- [完整带图报告](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report.md)
- [报告和图片 ZIP](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/report_with_figures.zip)
- [最终结果](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/result.json)
- [四阶段真实分析](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/repository_analysis.json)
- [真实结果摘要分析](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/result_analysis.json)
- [独立协议核验](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/protocol_verification.json)
- [独立指标复算](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/independent_metrics.json)
- [训练与最终测试日志](../data/runs/repository_e9408141e3dd44319e854bbdf4a70c0c/.autorepro_repository_runs/35f69af4c7064a6791810136f08de701/train_and_eval.stdout.log)

报告包含真实预测对照图和训练 / 验证损失曲线。预测图固定展示第一个测试窗口的 OT 通道，整体指标计算全部测试窗口与 7 个变量。浏览器已实际显示两张图；文件携带使用 ZIP 或同时复制 Markdown 的图片目录。

[旧单次公开协议解析基线](../data/runs/repository_66efc14890e34cb196a07a7d2bcb78fe/report.md)仍保留。本次在新工作副本重新训练并完成多 Agent 分析；固定 seed、同一环境得到相同数值，没有拿旧预测产物作为新训练。

## 版本、环境和限制

源码固定为 `cure-lab/LTSF-Linear@0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`，训练后 113 个源码文件的哈希全部一致。数据 SHA-256：

```text
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066
```

本次实际环境为 macOS 26.5.1 ARM64、Python 3.12.2、PyTorch 2.5.1、NumPy 1.26.4、Pandas 2.2.3、scikit-learn 1.5.2、Matplotlib 3.9.2。作者文档声明 Python 3.6.9、`torch==1.9.0`；本次是现代 CPU 兼容环境，不声称与论文原实验环境完全相同。

[作者单 seed 说明](https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601)指出设备与 PyTorch 可能造成结果差异；[初始化版本说明](https://github.com/cure-lab/LTSF-Linear/issues/39#issuecomment-1398345611)指出后续模型与早期版本有差别。本次保留所选版本，没有调整初始化、挑选 seed 或重定义目标追平表格；不能确定小幅偏差的具体原因。

未提供多 seed 方差、其他预测长度、其他数据集或其他模型的结果。当前仓库预设使用本地 CPU，不依赖 Docker。Windows CPU 已完成上述 10 月 10 日完整验收；DLinear CUDA 尚未实测，硬件或框架变化后须按同样数据与协议重新验收。

## 重跑与汇报顺序

从 `AutoReproducer/` 运行：

```bash
python scripts/reproduce_repository.py --profile dlinear_etth1_reference --llm-review --result-review
```

密钥通过已有配置或隐藏输入读取，运行时自动准备和健康检查环境；资源已有可核验缓存时可明确添加 `--offline`。详细操作见 [重跑步骤](dlinear_testing.md)。网页默认真实模式，选择官方仓库完整实验、保留多 Agent 分析、允许指标摘要分析后启动，使用同一生产后台。

汇报可依次展示论文目标行和 DLinear 原理、参数来源、五个真实分析任务、训练与早停日志、独立复算、指标对照和两张图，最后说明范围与环境限制。

该次实验后的工程完整回归为 **1138 passed、4 skipped、1 warning**；包含后续阶段状态与报告图片修复的提交前完整回归为 **1187 passed、4 skipped、1 warning**（既有 PyPDF2 弃用提示）。真实训练与产物提供论文单项复现证据，工程回归验证程序行为。
