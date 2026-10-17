# PDF 作者运行与比较：2026-10-17 实施记录

基线为已合并且常规 CI 全绿的 `main@0f3fd59`。本轮目标是 #27 的真实网页运行与有效比较；独立模型重载/指标复算由 #41 跟踪。`paper_metrics_verified=false`、`is_reproduced=null` 继续准确表示当前能力。

## 当前能力和剩余事项

| Issue | 模块/入口已实现 | 已有真实证据 | 本轮剩余事项 |
| --- | --- | --- | --- |
| #20 | 多文件事务、两轮显式修复、完整回滚；`run_with_repairs` API | 原子拒绝、取消、受保护文件和回滚错误回归 | 网页显式入口未接入，后续阶段完成 |
| #27 | PDF 仓库发现、原命令、有限错误修复、指标比较、网页/MD/ZIP | DGC 比较回归；NodeFormer、DIFFormer 完整执行 | 冻结当前修复后，用新 PDF 完成全流程验收 |
| #30 | bundled 数据提示、旧 PyG/NumPy 兼容 | 指定 Cora 十次自然结束、5316 次更新；raw/processed 前后摘要与输入张量核对通过 | 精确命令回归完成；独立模型重载、指标复算和论文协议完整核对仍待完成，issue 保持开放 |
| #34 | 原生 DGL 1.1.2 构造器及 raw/processed、mask、文件变化观测 | CCA-SSG 2050 次更新，原生 Cora 输入记录不变 | 与独立官方来源及论文协议核对，仍未完成 |
| #41 | 训练解释器依赖健康、原生检查、输入观测、统计口径及引用验证 | CPU/PyG、DGL、pip 24.0 及真实作者日志 | 复用已有捕获能力，接入完整协议和独立指标复算 |
| #42 | 尚无安装包验收 | 无 | 后续阶段，不以启动器代替复现能力验收 |

历史维护证据见 [10 月 16 日记录](maintenance_20261016.md)。此前成功记录仅证明对应冻结版本与实验，不追溯作为最新版本的新论文验收。

## 统一台账和冻结规则

[论文验收台账](paper_acceptance_registry.json)是论文身份、PDF 哈希、历史试验及独立候选资格的唯一当前清单。原 PDF 来源清单继续保留下载回执及原始状态，历史记录不覆盖。另核对并合并 11 份历史运行下载回执；其中错误下载的氢化物 PDF 按实际标题单独隔离，不能作为 NodeFormer 证据。初始迁移包含 50 个身份；补录曾尝试并停止的 GraphCON，再登记本轮三篇候选及一份错误下载的隔离身份后共 55 个身份。GAT/H2GCN 尚未执行，其 TensorFlow 环境未验证且尚无本轮固定作者提交，不能仅凭候选状态认定本轮可用。

冻结脚本和浏览器脚本共同读取台账。相同标题别名、同一 arXiv 的不同版本和重复 PDF 哈希不能创建不同身份；未知、冲突、已退役或已有试验历史的候选均拒绝独立验收。新冻结包含产品 Python/CSS 文件集合、执行/验收脚本、依赖清单及已安装发行包版本；开始和结束均核对，新增/删除文件、Git 提交及运行环境变化均拒绝独立验收。实际导出的仓库和执行计划也必须匹配登记的作者提交。

```text
python scripts/freeze_holdout_source.py --tag <trial> --paper-id <id>
python scripts/verify_browser_pdf_holdout.py --pdf <pdf> --repo <repo> --freeze <freeze> --tag <trial> --paper-id <id> --require-independent --require-comparison
```

凭据由原有隐藏输入提交。旧调用保持回归用途；旧冻结或未启用独立检查的记录不能宣称新独立验收。`reference_only` 可以满足比较功能验收，但必须保留设置差异；`not_comparable`、空比较和 unavailable 不通过。有效比较及参考值达标都不代表独立数值复现。

## 实施顺序与证据

1. 统一台账和脚本，更新 issue 的剩余事项。
2. 固定 ChebNetII 提交与用户指定 Cora 原命令，完成全部十次、原始早停和模型选择规则的回归；必要修复先完成测试。
3. Windows/Ubuntu 原生专项：CPython 3.12/PyG 2.6.1 与 CPython 3.11/DGL 1.1.2。实际原生运算预检，所选测试非空且无跳过，保存安装/健康/JUnit 证据。
4. 固定三篇未参与修复且满足本机支持条件的候选顺序，全部修复和 CI 通过后冻结产品，逐篇做真实网页/API 验收。
5. 修改产品会使试验论文退役；保留失败记录，修复后换下一篇。三篇均未通过时交付阻塞清单、保持 #27 开放。

作者完整训练不设默认截止时间。不得缩短轮次、减少重复次数或更换种子追求通过。原始未跟踪文件及运行证据保留。

当前新增台账/浏览器/JUnit 检查相关测试：**63 passed**，包括安装环境变化、作者提交变化、CSS 变化、仅供参考的设置说明，以及下载内容与保存的报告/结构化记录不一致。新增用例先复现失败，再修复校验。

原生 CI [39007889018](https://github.com/Yukikazct/AutoReproducer/actions/runs/39007889018) 四个任务全部通过：两平台的 PyG 各 **51 passed**，DGL 各 **14 passed**，均无跳过；实际原生运算、全部安装版本和 JUnit 已归档到工作流制品，[精简证据](evidence/native_cpu_ci_20261017.json)记录结果。

## 训练前固定的新论文顺序

三篇 PDF 的身份、哈希及作者提交已在任何候选训练前写入[来源回执](../examples/papers/candidates_20261017/manifest.json)和[静态筛选记录](evidence/pdf_candidates_preflight_20261017.json)。固定顺序为 **GCNH → GREAD → Critical Look Heterophily**，不按实测成绩改选。完整依赖是否能安装、自动规划及比较是否通过，均由后续真实网页试验检验；静态筛选不等于作者程序已经运行。

GCNH 在 PDF 第 4 页给出作者仓库和 Table I，包含公开 Cornell/Cora 原始数据及十份划分；作者代码通过 CUDA 可用性判断保留 CPU 分支。GREAD 的 CPU 分支明确，但 pykeops 可能需要 Windows 原生编译器，作为已记录的安装风险保留。Critical Look 的 `--device` 及 Dataset 的默认 CPU 参数提供作者支持的 CPU 路径，保留十次运行和 1000 步；分类 accuracy 与其他数据集的 ROC AUC 必须分别比较。DropGNN 因原 Python 3.8 环境不在当前声明支持范围内，在训练前排除。

## #30：Cora 精确命令回归完成

作者提交 `ded6c18cbe9673234071031767d17826ad632aca`，工作目录 `main/`，完整原命令：

```text
python training.py --net ChebNetII --dataset Cora --K 10 --prop_lr 0.001 --prop_wd 0.05 --dprate 0.8 --semi_rnd True
```

最后完整轮次固定产品源码于 `558049c`，十次均保留作者固定种子、最高 1000 轮、early stopping 200 和最低验证损失的模型选择规则。合计 **5316 次优化器更新**，自然退出 0。十份 mask 均为 **140/500/1000**，7 类各 20 个训练节点，互不重叠。测试均值 **82.4400**；作者打印的 **±1.1700** 是 **95% bootstrap 置信区间半宽**，不能当作标准差。

实际输入来自作者自带的 Cora raw 文件；PyG processed 文件由原加载器在首轮尝试中生成。最后十次执行前后记录全部 **13 个 raw/processed 及目录附带文件**，完整集合、大小及 SHA-256 一致。作者加载器只读重载后，`x/y/edge_index` 的张量摘要与十次训练输入全部一致；这仅核对输入对应关系，不是独立模型或指标复算。

历史尝试全部保留：首次缺少 matplotlib、随后十次训练后的 NumPy ragged 汇总失败、现有源码约束兼容修复后的自然成功，以及补齐 processed 前后摘要后的完整十次成功。修复只作用于经过校验的最终统计表达式，原作者 122 个版本内文件摘要不变。没有缩短轮次、减少次数或调整种子。

结束后所属作者进程为零；移除本次额外测试依赖 **205,986,665 字节**，标准应用 CPython 和专用原生测试 fixture 保留。完整命令、规划、全部 stdout/stderr、逐次结果、mask 摘要和清理回执见[结构化证据](evidence/chebnetii_cora_regression_20261017.json)及[完整日志归档](evidence/chebnetii_cora_regression_20261017_logs.zip)。这次固定命令回归没有重新做自动 PDF 规划或独立数值验收；`paper_metrics_verified=false`、`is_reproduced=null`。

执行、比较、环境清理及进程完成契约检查在专用原生环境实测为 **186 passed / 0 skipped**（此前普通环境为 173 passed / 13 skipped）。冻结提交 `5ef21fe` 的普通 CI [39009944558](https://github.com/Yukikazct/AutoReproducer/actions/runs/39009944558) 和原生 CI [39009944559](https://github.com/Yukikazct/AutoReproducer/actions/runs/39009944559) 全部通过，见[冻结前 CI 记录](evidence/pdf_acceptance_ci_20261017.json)。

## 第一篇：GCNH 因报告一致性缺陷退役

在 `5ef21fe` 上从网页上传 GCNH，使用真实 API，冻结摘要 `7e7b896c43123dc3526eb3674048c4a7c16a70e0ca0ecd5125b874cf3bc4fed3`。自动发现并固定作者 `311ebe34d4cd58fdc6279d803e212f4c1ea5aaf0`，原样执行 `experiments/main_table.sh` 的 Cornell 命令：

```text
python main.py --dataset cornell --epochs 300 --nlayers 1 --verbose False --model GCNH --dropout 0.25 --nhid 16 --batch_size 50
```

全部十份划分自然结束，**6000 次优化器更新**，原始平均准确率 **0.8648648648648649**；作者未打印标准差，保持缺失。24 个 Cornell 输入文件前后大小和摘要一致。与 PDF 第 4 页 Table I 的 86.49% 得到有效 `reference_only` 比较，差值约 **−0.003514 个百分点**，同时展示环境与实验设置差异；不代表独立数值复现。依赖首轮安装成功，结束后完整释放，活动 worker/run/execution 全为零。

**正式验收未通过。** Markdown 与 ZIP 已一致，但下载内容比运行目录的 `report.md` 多了最后补入的运行环境说明和完整输出附件；刷新检查尚未到达。因此不能关闭 #27，GCNH 已登记为回归样本。原始失败报告、结果和输入保持不变，见[失败记录](evidence/gcnh_pdf_trial_20261017.json)及[完整证据归档](evidence/gcnh_pdf_trial_20261017_logs.zip)。

修复将最终网页报告、附件、已校验图片和结构化结果同步到运行目录；保存失败明确返回错误，并使历史终态一致，保留作者训练完成的原始事实。ZIP 同时携带报告声明的完整输出附件。新增无图/有图及 ZIP 附件回归先复现失败，修复后相关检查 **167 passed / 4 skipped**；跳过的是缺少可选 matplotlib 的绘图用例，不作为原生能力证据。真实 GCNH 保存结果的副本回放已确认报告、结构化记录及 ZIP 完整输出附件一致，未重训且没有修改原始证据；该回放仍只是回归。修复提交并通过 CI 后重新冻结，按既定顺序进入 **GREAD**。

## 第二篇：GREAD 因入口身份校验缺陷退役

报告修复后的 `9357bd7` [普通 CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39013932764) 与[原生 CPU CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39013932782) 全部通过，PR 检查亦全绿。PyG 两平台各 **51 passed**，DGL 各 **14 passed**，零跳过；[完整版本和 JUnit 摘要](evidence/pdf_acceptance_report_fix_ci_20261017.json)已保存。

GREAD 真实网页上传与 API 连接成功，冻结该提交的 152 个源码文件，摘要 `fccddd17550ae9a9bf80a5c71519efcec7145385b5cc8efde55dd5e44d9946b2`。发现了正确作者提交 `f97ec01a17bb9cba1da2b89192b83cd7a25a250b`，但**在规划阶段失败，没有安装作者训练依赖或执行训练，也没有论文比较**。两次模型提案均正确给出 `cwd=src` 和原命令 `python run_GNN.py --dataset=Cora --use_best_params`，却因显式入口填入仓库路径 `src/run_GNN.py` 而被字符串相等检查拒绝；备用 README 命令提取随后在根目录寻找该脚本，仍失败。

修复先解析原始 argv 和 cwd，再核对显式入口是否是同一个已固定作者文件；命令、工作目录及全部参数保持不变，指向其他文件或改变入口类型仍拒绝。三条等价路径回归先复现失败，修复后通过。使用真实被拒提案做的只读规划回放已通过，并确认原始运行文件没有改变；这不是训练或独立 PDF 通过。GREAD 已转为回归样本，活动 worker/run/execution 全为零，清理无残留；[失败与回放记录](evidence/gread_pdf_trial_20261017.json)及[完整原始证据](evidence/gread_pdf_trial_20261017_logs.zip)已保存。pykeops 安装等原先风险尚未走到，不能据此声称环境已验证。

修复的原生执行与规划检查 **145 passed / 0 skipped**，退役身份与浏览器门槛检查 **60 passed / 0 skipped**；两组不相加。29 个固定作者文件的摘要全部未变。失败运行因没有已接受的仓库元数据，其正式成功路径的 `source_workspace_retained` 检查为 false；磁盘原始仓库已另外逐文件核对存在且未变，其余清理检查均通过，不能把该失败运行视为验收通过。

修复提交并通过 CI 后将再次冻结，按固定顺序进入最后一篇 **Critical Look Heterophily**。新 PDF 验收结果随实际完成追加；本文件不是 #27 完成声明。
