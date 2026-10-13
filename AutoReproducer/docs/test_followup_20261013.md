# 2026-10-13 测试跟进

## Windows 并发清单读取

主分支 GitHub Actions 运行 `38349367975` 的 Windows / Python 3.12 / 系统编码任务，在 `test_concurrent_manifest_readers_see_complete_documents` 中读取到了 `None`；同一运行的 UTF-8 任务及其他矩阵任务通过。

清单写入使用临时文件和原子替换，但 Windows 在替换期间仍可能拒绝读取文件。此前读取方法将此异常解释为“没有清单”。本次让保存、单个读取和列表读取共同使用按清单分配的进程间文件锁，原子替换仍保留。新增真实子进程持有写锁的测试，确认读取等待写入完成后返回新清单。

清单锁修复的本地 Python 3.12 / UTF-8 全量回归：**2945 passed, 70 skipped**，耗时 312.88 秒。[GitHub Actions 38360989627](https://github.com/Yukikazct/AutoReproducer/actions/runs/38360989627) 已通过 Ubuntu / Windows × Python 3.11 / 3.12 的全部任务，包括 Windows 系统编码与 UTF-8 回归。这个远端结果对应 `a0eeffd`，不包含下面的数据目录修复。

跳过项包括未配置的可选原生依赖与不适用的进程行为，不能算作这些环境的通过证据。

## 旧报告实际升级

历史会话 `20261012_221705`（DAGNN / Towards Deeper Graph Neural Networks）已从保存结果更新到格式 2，显示 loss **0.6258** 和 accuracy **0.845**。

- 原报告继续保留，文本 SHA-256 为 `cb4f8d0cb091b3bf1fc85746d2798c65ef4f01a8c98be599715feef49e6ec39e`，与升级清单记录的原始摘要一致。
- 新报告使用独立的 `.report-revisions` 文件；重新读取历史会话解析到这个版本。
- 禁止进程启动和模型调用的验证仍能读取、重复调用更新入口；原报告、运行结果、进度及审计日志摘要不变。
- 全量回归包含旧会话、重启恢复、Markdown / ZIP 一致性及归档结果重建测试。

关联 #37。报告显示实测值不代表完成论文数值比较。

## 真实 API 与新 PDF

使用用户授权的 DeepSeek 官方 API，`https://api.deepseek.com/v1` / `deepseek-chat`，项目连接测试返回 `OK`。密钥通过隐藏的交互输入传入，仅保留在进程内存。

新论文选为 *Graph Structure Learning for Robust Graph Neural Networks*，PDF 来源 `https://arxiv.org/pdf/2005.10203`。下载发生在源代码冻结之后；PDF SHA-256 为 `32191b55ab5fd8fbafc5db5591233f44ca256181fb79d825b33a4c24a1367657`，1,230,859 字节。产品源文件摘要为 `bf61fa80498664aef261b8444e01cae51fe3c7db2c98074221d6050071a2f72c`，包含本次清单锁修复。

测试只向网页后端入口提交 PDF，没有传入仓库 URL 或手工运行命令。任务 `progress_1791875237497.jsonl` 自动定位到 `ChandlerBang/Pro-GNN` 并启动 `train.py`，但未进入训练：作者使用的 `Dataset(root='/tmp/', ...)` 在 Windows 上没有对应目录。`urllib.request.urlretrieve` 已取得响应，却在保存 `/tmp/cora.npz` 时抛出 `FileNotFoundError`；DeepRobust 随后将它包装成一般的下载失败。

实际失败后，系统现在检查异常链和作者调用位置，将缺失的 `/tmp/` 数据目录映射到本任务的 `_autorepro_capture/author_data`，再重试同一作者命令。仅改变执行时的数据路径字面量，原仓库文件、模型、训练参数和实际下载内容不变。失败日志、路径修复记录及完成报告均保留。它与依赖、NumPy 修复共用最多 8 次报错修复机会，不新增训练时间上限。

路径修复及相关回归为 **76 passed, 7 skipped**；依赖就绪后再次执行路径、NumPy 与报错重试测试为 **27 passed**，其中完整服务集成测试实际经历缺失目录报错、重试和 40 次训练更新。第一次扩展全量为 **2951 passed, 70 skipped**。Pro-GNN 已退役为回归样本，没有将它标记为完整训练成功。

## 修复后换用 DGC PDF 独立测试

使用未参与前述修复的论文 *Dissecting the Diffusion Process in Linear Graph Convolutional Networks*。冻结 125 个产品 Python 文件后，从 `https://arxiv.org/pdf/2102.10739` 获取 PDF：980,587 字节，SHA-256 `295dae352e820b0db6817ca5b6ec5f338bb27c74c39960f4eb519aec2842307e`。产品源文件摘要为 `2600dae1acc0ac2d4b934e5f1eec716b068a4bee0fc21206433791267070e654`，包含数据目录修复。测试前、测试后及证据采集时均确认产品源文件未变。

仍只向网页后端提交 PDF，使用真实 DeepSeek API，未提供仓库 URL 或手工命令。系统从 PDF 第 1 页作者代码声明中提取 `https://github.com/yifeiwang77/DGC`，取得固定提交 `f91b35a0d4388660e250d575d4c1b8fef41d7039`，按作者启动脚本选择 Cora 实验，在 CPU 上实际执行：

```text
python main.py --dataset cora --T 5.27 --K 250 --lr 0.156 --weight_decay 3.544e-5 --no-cuda
```

- 作者代码完成内置的 10 次运行，观察到 **1000 次优化器更新**，退出码 **0**，执行耗时 **35.234 秒**，整条流程约 **186 秒**。
- 原始输出：`Dataset: cora Test accuracy of 10 runs: mean 0.83280, std 0.00040`。最后一次运行 accuracy 为 `0.8330`，生成报告的实测值列显示 `0.833`。
- 状态 `COMPLETED` / `author_code_completed`；作者命令自然返回、训练更新及所属进程结束记录均确认。任务 `done=true`，活动 worker、运行任务和执行进程均为 **0**。
- 无 NumPy 或数据路径兼容修改应用于这篇论文。作者代码运行成功不代表系统已经实现论文数值比较，`paper_metrics_verified=false`、`is_reproduced=null` 保持明确。

冻结源代码的最终全量回归：**2951 passed, 70 skipped**，327.31 秒。上述 27 项定向回归另行确认了数据目录报错后的真实服务重试和训练；测试组存在重叠，不相加计数。

脱敏证据：[dgc_holdout_20261013_02.json](evidence/dgc_holdout_20261013_02.json)；[源文件冻结记录](evidence/dgc_holdout_20261013_02_source_freeze.json)；[可供手工上传的测试 PDF](../test_papers/dgc_2102.10739_holdout_20261013.pdf)。本地报告：`data/reports/Dissecting_the_Diffusion_Proce_20261013_201059.md`。

上述 DGC 成功结果对应当时的冻结版本。浏览器工具随后恢复可用，真实网页测试发现以下新问题；DGC 因产品继续修改而退役，不作为最新版本的独立验收。

## 浏览器上传与作者启动配置

重启服务后，Streamlit 上传组件可能保留 `DeletedFile`，旧页面会对它调用 `getvalue()` 并崩溃。现在检测失效上传对象，显示重新上传提示，点击开始不会创建后台任务；真实 AppTest 回归通过。

实际通过网页上传 *EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks*（arXiv `2205.13892`），自动从 PDF 定位 `Leirunlin/EvenNet`。模型已给出作者 `src/main_common.py` 命令，但四次规划均因输出目录 `../logs` 被拒绝。这个目录从 `src/` 解析后仍在整个仓库内：现在按仓库边界判断，允许此类作者输出路径。备用启动脚本提取同时支持有限字面量 `for` 循环的首个作者值以及子目录入口，保留完整 Python 调用和作者训练参数，不执行 shell。EvenNet 未训练成功，已退役。

随后冻结产品，实际通过网页上传 *Hyperbolic Graph Convolutional Neural Networks*（arXiv `1910.12933`），自动定位 `HazyResearch/hgcn` 并启动作者程序。程序在训练前因 `KeyError: 'DATAPATH'` 失败：README 要求加载 `set_env.sh`，原流程遗漏了它的环境声明。新实现读取已固定摘要的作者环境脚本，解析字面量、`$(pwd)` 和已声明变量引用，将数据及模块路径带入子进程，并保留来源文件、摘要和原文。作者 `PYTHONPATH` 与已准备的依赖目录合并。无法解析的声明会记录提示；不执行 shell、环境激活或未知命令，不将主机密钥用于变量展开。HGCN 未训练成功，已退役。

这批定向回归为 **128 passed, 8 skipped**，包括真实服务加载作者环境配置、40 次优化器更新和自然退出。此前环境配置修复前的全量结果为 **2952 passed, 70 skipped**，不能代替最新代码的全量验证。

#30 的 ChebNetII 历史问题尚未重新实跑，保留该 issue。运行后论文数值比较仍未实现。

## 最新独立网页验收：DiGCN

修复后换用未参与修复的 *Digraph Inception Convolutional Networks*，冻结 126 个产品 Python 文件，摘要 `253fbfdb7e06b32e98da96245c7b9a3992959266eefdf01cafc927c146e74d79`。从 NeurIPS 官方下载 PDF，SHA-256 为 `1fd2ec4b7b04b7735f53c5ddf5f438fb88ea6bbb7a5817d8da7605ecbb967eff`。通过真实浏览器选择“上传 PDF”并点击开始，使用真实 DeepSeek API，没有手工提供仓库或命令。

系统从 PDF 第 2 页作者声明提取 `flyingtango/DiGCN`，固定源码 `0b87d1917493b32683f3f04c00fa9453eb884ec6`，使用仓库自带完整 Cora-ML 数据，从 `code/` 执行 `python digcn.py --gpu-no 0 --dataset cora_ml`。保留作者默认 20 次运行、1000 轮上限和原生提前停止规则。

- 真实 CUDA 训练观察到 **10,049 次优化器更新**，作者命令自然退出，退出码 **0**，执行耗时 **60.094 秒**；流水线统计耗时 **298.05 秒**。
- 原始输出：`Val Acc: 0.8055, Val Loss: 0.6815, Test Accuracy: 0.7719 ± 0.0044, Duration: 2.4038`。网页格式 2 报告的实测值表显示验证准确率 `0.8055`、验证损失 `0.6815`、测试准确率 `0.7719`。
- 网页训练期间保持整体“运行中”，逐阶段显示事件；结束后 `done=true`，活动 worker / run / execution 均为 **0**。
- 测试前后及证据采集时确认冻结源文件未变；这篇论文未应用 NumPy 或临时数据目录兼容修改。
- 最新代码全量回归：**2961 passed, 70 skipped**，564.51 秒。跳过项不作为已验证环境；定向测试与全量有重叠，不能相加。

证据：[独立测试记录](evidence/digcn_holdout_20261013_05.json)、[源文件冻结](evidence/digcn_holdout_20261013_05_source_freeze.json)、[完成页面](evidence/digcn_completed_20261013.jpg)、[实测值页面](evidence/digcn_metrics_20261013.jpg)、[可上传 PDF](../test_papers/digcn_holdout_20261013_05.pdf)。确认范围是作者代码实际完成，`paper_metrics_verified=false`、`is_reproduced=null`；论文数值比较尚未接入。
