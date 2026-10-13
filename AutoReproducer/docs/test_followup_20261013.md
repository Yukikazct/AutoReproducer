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

当前 Computer Use 工具返回空浏览器清单，因此本轮尚未验证真实浏览器上传；后端任务不能记作浏览器验收。#30 的 ChebNetII 历史问题尚未重新实跑，保留该 issue。
