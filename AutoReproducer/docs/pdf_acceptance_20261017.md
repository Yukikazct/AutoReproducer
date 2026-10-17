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

执行、比较、环境清理及进程完成契约检查另有 **173 passed / 13 skipped**；这些原生用例由专用环境继续实测，不能把跳过当作原生能力通过。

新 PDF 验收结果随实际完成追加；本文件不是 #27 完成声明。
