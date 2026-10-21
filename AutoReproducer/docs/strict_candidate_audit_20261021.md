# 严格独立验收候选只读审计（2026-10-21）

本轮目标是寻找现有严格捕获能力可以直接支持的、尚未参与产品修复或回归的真实论文实验。审计结论：**尚未确认可以直接启动的新候选；没有完成新的论文严格验收。** 本轮仅查询公开仓库元数据、读取固定版本源码和论文正文，没有安装候选依赖、执行作者程序、注册候选或修改作者协议。本报告本身是本轮审计的唯一文件产物。

GitHub stars 是优先筛选和排序的信号，不能证明论文结果正确、代码完全可复现或不存在学术问题。下文 stars 为本次查询快照，可能随时间变化。

## 重点候选：tunedGNN

论文：**Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification**，NeurIPS 2024 Datasets and Benchmarks Track。

| 项目 | 已核实证据 |
| --- | --- |
| 官方仓库 | [LUOyk1999/tunedGNN](https://github.com/LUOyk1999/tunedGNN)，本次查询 189 stars |
| 固定代码 revision | `23f9604e8b13a9a6d3faa2f691cd844006979153` |
| 正式论文 PDF | [NeurIPS 正式版本，20 页](https://proceedings.neurips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf) |
| PDF SHA256 | `1756d6d6c976e34fe668340551145f135de2c32a7cd16e5e046e4686ee9a4fb6` |
| 拟评估的有限范围 | GCN* 在 Cora 上的一个完整论文实验，不代表论文所有数据集与方法 |
| 论文目标值 | p5 Table 2：GCN* / Cora，准确率 `85.10 ± 0.67%` |
| 当前登记状态 | 本轮查询的 `paper_acceptance_registry.json` 中没有该论文；本轮未新增登记 |

PDF 从上述正式地址读取并在内存中计算散列和提取正文，未保存候选 PDF 文件。论文 p5、p16 明确报告 **5 次不同初始化的独立运行**；p16 明确使用验证集表现最高的模型对应的测试准确率。p17 Table 8 的 Cora 参数与固定源码命令一致。

### 作者原命令

来源为固定版本 [medium_graph/run_gnn.sh 第 23 行](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/run_gnn.sh#L23)，工作目录为 `medium_graph`：

```bash
python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
```

`$1` 是作者脚本的设备分配参数。以 GPU 0 调用原脚本时，该行中的设备参数展开为 `--device 0`。这只是设备编号的原生参数展开，不改变运行次数、训练轮数、划分或模型选择。该单行是作者完整实验脚本中的一个实验，不能据此声称执行了整个脚本。

### 完整协议与计算预算

| 项目 | 作者协议及来源 |
| --- | --- |
| 数据 | Cora：2708 节点、1433 特征、7 类，论文 p4 Table 1；通过作者的 Planetoid loader 读取 |
| 划分 | 每类训练 20 个，共 140 个；验证 500 个；测试 1000 个；由原始 `class_rand_splits` 生成，其余节点仍在完整图中 |
| 随机性 | 所有运行开始前只调用一次 `fix_seed(123)`，随后生成一次划分；5 次运行复用该划分，每次执行原始 `reset_parameters()`，连续消耗同一随机数流 |
| 模型 | 3 层 GCN，隐藏维度 512，dropout 0.7，无归一化、无残差；论文 p17 Table 8 |
| 优化器 | Adam，学习率 0.001，weight decay `5e-4` |
| 完成规则 | 原生 5 次运行，每次固定 500 epochs，每个 epoch 一次优化器更新；共 2500 次完整图优化更新，没有提前缩短训练 |
| 选模 | `result[1] > best_val` 时更新 `best_val` 和 `best_test = result[2]`；验证值相同时保留先前选择 |
| 汇总 | 每次最高验证准确率对应的测试准确率，最终输出 `Final Test` 的均值及样本标准差；不能使用日志中的 `Highest Test` 代替 |
| 设备 | 论文 p16 描述原作者使用 8×RTX3090 工作站；本条命令只选择一个设备。源码有 CUDA 可用性判断、CPU fallback 和原生 `--cpu` 参数 |
| 本机预算判断 | 固定工作量为 5×500 次完整图更新；未安装或执行，实际耗时、峰值内存和 RTX4060 8GB 的运行兼容性均未实测，不能声称已经通过 |

不得把原始随机性替换成 seed 123 至 127 的五次重新播种，也不得减少运行次数、轮数或测试样本。论文报告 5 次初始化，源码展示了作者实现该协议的具体方式。

固定源码证据：

- [main.py：一次播种、一次划分及完整训练循环](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/main.py)
- [parse.py：参数默认值与模型构造](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/parse.py)
- [data_utils.py：class_rand_splits 与准确率](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/data_utils.py)
- [eval.py：完整图前向及划分评估](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/eval.py)
- [logger.py：验证选模及 Final Test 汇总](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/logger.py)
- [model.py：完整模型 forward(x, edge_index)](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/model.py)

### 当前两项通用捕获缺口

原始 [main.py 第 109 行](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/medium_graph/main.py#L109) 调用：

```python
out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
```

拟捕获的完整值分别为 `model`、`dataset.graph['node_feat']`、`dataset.graph['edge_index']`、`dataset.label` 和 `split_idx['test']`。现有机制有以下明确边界：

1. **stored attribute 与固定字典键的混合选择。** 当前 selector 分别支持 `attributes` 或 `keys`，不能直接表达 `dataset` 的原始存储字段 `graph` 后再取固定键 `node_feat` / `edge_index`。不能用表达式求值、动态属性访问或新写作者变量绕过。
2. **单列整数标签。** 作者在 `main.py` 将标签统一为 `[N, 1]`，loss 和评估按作者代码执行 `squeeze(1)`；当前严格捕获的 `select` 要求一维整数标签。仅增加混合 selector 仍不能完成该实验的严格捕获。

原生 `run` 重复循环、固定优化次数和验证选择在结构上与现有 native-series 方向相近；这不等于完整计划、语义核验或运行验证已经通过。两项缺口之外仍可能存在尚未经过实际规划和执行验证的条件。

本轮不为 tunedGNN 修改捕获实现后，再把 tunedGNN 声称为未参与开发的独立验收。若未来使用这些候选的具体问题推动产品修复，必须保留该历史、遵守候选退役规则，并重新选择未参与修复的论文；不能通过更换别名或 PDF 版本重新计为独立样本。

## 其他候选的简要排除记录

这些记录是静态筛选，不代表作者程序已经失败。涉及现有捕获能力的结论仅说明本轮无法确认可以直接严格核验；普通作者执行与严格论文复现结论仍须分开。

| 候选及固定来源 | 本轮结论 |
| --- | --- |
| FAGCN、BernNet、GloGNN | 当前 registry 均为 `regression`，已有开发或回归历史，排除新的独立验收。以本项目登记记录为准，不能因为代码容易运行而复用为独立样本。 |
| [Set Transformer，691 stars，73432c640ac78140496d6738416c54d32c686d65](https://github.com/juho-lee/set_transformer/tree/73432c640ac78140496d6738416c54d32c686d65) | 官方 README 明确分类实验需要无法公开分享的 `ModelNet40_cloud.h5`，作者使用 8×Tesla P40；不能把仓库中的回归演示当成论文分类实验。 |
| [NAGphormer，137 stars，dd709f935d42796adfec74a40443e316626f5433](https://github.com/JHL-HUST/NAGphormer/tree/dd709f935d42796adfec74a40443e316626f5433) | 标签、完整预处理特征和测试索引可见，但原始单脚本有早停，实际优化次数可变；当前 single capture 要求固定次数，不能把原始 epoch 循环冒充完整运行的重复循环。论文正确标识为 [arXiv:2206.04910v4](https://arxiv.org/pdf/2206.04910v4)，不能使用先前隔离的其他 arXiv 编号。 |
| [SuperGAT，160 stars，1a3220c7c7d706248f67edad6d84540a646f0904](https://github.com/dongkwan-kim/SuperGAT/tree/1a3220c7c7d706248f67edad6d84540a646f0904) | 原生 7 次运行，但作者锁定 Python 3.7.6 / Torch 1.4 / CUDA 10.0 / PyG 1.4.3；本轮未确认完整测试输入在捕获作用域中的可选取性及原生依赖兼容性，不能登记为可直接启动。 |
| [OrderedGNN，48 stars，80c375896efa5ee2ae6f52b735c70753e3243f4e](https://github.com/LUMIA-Group/OrderedGNN/tree/80c375896efa5ee2ae6f52b735c70753e3243f4e) | 作者原入口使用 W&B sweep/agent；完整模型返回 `encode_values` 字典，作者后续取 `['x']`。当前严格捕获要求二维 logits tensor；不能自行替换入口或只捕获一个下游层来宣称完整模型核验。 |
| [GraphENS，86 stars，67833317348446e4464a1c4106553bbc1f477896](https://github.com/JoonHyung-Park/GraphENS/tree/67833317348446e4464a1c4106553bbc1f477896) | 原生 5×2000 epochs、一维标签和验证选模的结构接近现有能力，但方法训练注册 backward hooks，现有模型复制会保留它们，TorchScript trace 明确拒绝此类模型；前向还需要作者字面量 `None` 作为第三参数，现有 forward selectors 不能直接表达。本轮不通过关闭方法选项或新增导出修补来取得该候选的独立验收。 |
| [ReNode，57 stars，feffff6f2b502e5eb8440c6b30410dc01903e3df](https://github.com/victorchen96/ReNode/tree/feffff6f2b502e5eb8440c6b30410dc01903e3df) | 作者主结果是 weighted / macro F1，当前严格分类准确率核验不能覆盖；准确率成功不能替代论文原指标。 |

以上所有候选本轮均未安装、运行或注册。没有新的真实论文严格验收成功，也没有以工程机制测试、静态阅读、API 可用性或作者日志输出替代该结论。
