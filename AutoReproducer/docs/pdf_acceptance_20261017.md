# PDF 作者运行与比较：2026-10-17 实施记录

基线为已合并且常规 CI 全绿的 `main@0f3fd59`。本轮目标是 #27 的真实网页运行与有效比较；独立模型重载/指标复算由 #41 跟踪。`paper_metrics_verified=false`、`is_reproduced=null` 继续准确表示当前能力。

**本轮三篇预定候选均未通过独立 PDF 全流程验收，#27 保持开放。** 已完成统一台账、#30 精确 Cora 回归、持续原生 CPU CI，并修复三次试验发现的产品缺陷。三篇失败是独立验收的阶段交付点，修复工作继续；本记录保留失败阶段、回归证据和后续阻塞，不能作为独立 PDF 通过声明。

用户指出三篇失败后不应停止修复，已于 17:17（北京时间）继续工作。接下来完成退役论文的完整真实网页回归，遇到新缺陷继续补失败测试、修复及推送；每次实际运行冻结产品代码，旧失败结果和样本资格保留。任何新独立候选都须另行预登记身份、顺序与筛选依据，不能补入已结束的旧三篇顺序。

## 当前能力和剩余事项

| Issue | 模块/入口已实现 | 已有真实证据 | 本轮剩余事项 |
| --- | --- | --- | --- |
| #20 | 多文件事务、两轮显式修复、完整回滚；`run_with_repairs` API | 原子拒绝、取消、受保护文件和回滚错误回归 | 网页显式入口未接入，后续阶段完成 |
| #27 | PDF 仓库发现、原命令、有限错误修复、指标比较、网页/MD/ZIP；本轮修复报告同步、嵌套入口、断字链接排序 | GCNH 十次执行及有效 reference_only 比较；三篇正式试验均失败，修复后仅作回归 | 未参与修复的新 PDF 在重新冻结版本上完成全部门槛；本轮无通过样本 |
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

初始新增台账/浏览器/JUnit 检查相关测试为 **63 passed**，包括安装环境变化、作者提交变化、CSS 变化、仅供参考的设置说明，以及下载内容与保存的报告/结构化记录不一致。新增用例先复现失败，再修复校验；后续检查按修复批次记录，不将重叠测试组相加。

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

随后在 `37d8fd6` 上，从真实 Chrome 历史页恢复原 GCNH 会话，展示比较表，下载 Markdown、ZIP 和完整输出附件；全部内容与保存的报告、结构化结果一致，没有新建任务，原始报告及结果未变。见[历史恢复证据](evidence/gcnh_history_restore_20261017_logs.zip)。这补充了网页历史入口的回归证据，不能改变原冻结试验失败及论文已退役的事实。

## 第二篇：GREAD 因入口身份校验缺陷退役

报告修复后的 `9357bd7` [普通 CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39013932764) 与[原生 CPU CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39013932782) 全部通过，PR 检查亦全绿。PyG 两平台各 **51 passed**，DGL 各 **14 passed**，零跳过；[完整版本和 JUnit 摘要](evidence/pdf_acceptance_report_fix_ci_20261017.json)已保存。

GREAD 真实网页上传与 API 连接成功，冻结该提交的 152 个源码文件，摘要 `fccddd17550ae9a9bf80a5c71519efcec7145385b5cc8efde55dd5e44d9946b2`。发现了正确作者提交 `f97ec01a17bb9cba1da2b89192b83cd7a25a250b`，但**在规划阶段失败，没有安装作者训练依赖或执行训练，也没有论文比较**。两次模型提案均正确给出 `cwd=src` 和原命令 `python run_GNN.py --dataset=Cora --use_best_params`，却因显式入口填入仓库路径 `src/run_GNN.py` 而被字符串相等检查拒绝；备用 README 命令提取随后在根目录寻找该脚本，仍失败。

修复先解析原始 argv 和 cwd，再核对显式入口是否是同一个已固定作者文件；命令、工作目录及全部参数保持不变，指向其他文件或改变入口类型仍拒绝。三条等价路径回归先复现失败，修复后通过。使用真实被拒提案做的只读规划回放已通过，并确认原始运行文件没有改变；这不是训练或独立 PDF 通过。GREAD 已转为回归样本，活动 worker/run/execution 全为零，清理无残留；[失败与回放记录](evidence/gread_pdf_trial_20261017.json)及[完整原始证据](evidence/gread_pdf_trial_20261017_logs.zip)已保存。pykeops 安装等原先风险尚未走到，不能据此声称环境已验证。

修复的原生执行与规划检查 **145 passed / 0 skipped**，退役身份与浏览器门槛检查 **60 passed / 0 skipped**；两组不相加。29 个固定作者文件的摘要全部未变。失败运行因没有已接受的仓库元数据，其正式成功路径的 `source_workspace_retained` 检查为 false；磁盘原始仓库已另外逐文件核对存在且未变，其余清理检查均通过，不能把该失败运行视为验收通过。

入口修复 `37d8fd6` 的[常规 CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39016059962)及[原生 CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/39016059944)均为四项成功，PR 对应八项检查也全部通过。两平台 PyG 各 **51 passed**，DGL 各 **14 passed**，无跳过，实际运算及依赖/JUnit 摘要见[第三次冻结前 CI 记录](evidence/pdf_acceptance_entrypoint_fix_ci_20261017.json)。在该提交重新冻结后，按既定顺序尝试最后一篇。

## 第三篇：Critical Look 因 PDF 链接排序缺陷退役

冻结 `37d8fd64b213e42fb127207ac7b80440207e0b0d` 的 152 个源码文件，摘要 `14664b048b7f23a371d524ee8d9a37c202de8a2227638b63438da2bda5ca8d1c`，同时启用独立样本与有效比较门槛。真实 API 连接测试通过，随后从 Chrome 上传原 PDF。PDF SHA-256 为 `9025d1a5a4258bc77dadc31528a68f7abbe6c08c3b9abf82ecb644e00811bd14`，预先登记的作者提交为 `a431395582e929d88271309716bea4fe24ce6318`。

**失败发生在上传预览，任务尚未提交。** PyPDF2 将第 1 页正文提取为 `https://github.com/yandex-research/heterophilous-gr aphs .`，同页 URI 注释则包含完整 `heterophilous-graphs` 链接。两者都带作者代码声明，排序却优先显示截断的正文候选；浏览器等待完整链接超时。没有作者依赖安装、训练更新、执行计划或指标比较，也没有可导出的本次实验报告。修改产品之前已再次核对原冻结证据一致，活动 worker/run/execution 全为零。

新增内部空格/换行用例先得到 **2 failed / 5 passed**。修复仅在同页、同证据类型且正文明确包含剩余后缀、只有一个候选 URI 时，优先采用完整的作者 URI；保留截断正文、原始 URL、页码和上下文。跨页、无对应后缀、长词误匹配、参考文献及歧义候选不能据此升级。PDF 解析、路由、预览和流水线相关检查 **176 passed / 0 skipped**；将此论文退役后的身份/浏览器门槛检查 **60 passed / 0 skipped**。

仍在运行的旧服务缓存了修复前预览；保留该回归尝试后，重启本轮专用服务，真实 Chrome 再次上传原 PDF，已显示完整作者链接且保留原正文证据。未点击开始、未创建新任务，PDF 和原进度记录的摘要不变。该验证仅证明预览修复，Critical Look 已登记为回归样本。见[失败与回归记录](evidence/critical_look_pdf_trial_20261017.json)及[17 文件原始冻结、截图和测试归档](evidence/critical_look_pdf_trial_20261017_logs.zip)。

## 本轮结论与阻塞

### 继续修复：运行中 PDF 预览缓存

重新排查确认：即使解析器已按源码指纹加载新模块，网页最外层的 `st.cache_data` 仍会直接返回同一 PDF 字节的旧预览，绕过模块加载。新增 AppTest 先复现 **1 failed / 0 skipped**；修复将模块加载移至无缓存包装层，并把解析源码指纹纳入缓存键，同次调用绑定实际模块。相同源码仍命中缓存，源码变化后重新提取，旧活动解析器的全局状态保持不变，不创建任务或调用模型。

修复后的 PDF 解析、路由、真实上传组件、入口加载、台账及浏览器门槛相关检查 **237 passed / 0 skipped**。这是预览缓存回归，后续完整作者运行另行记录；[失败与成功 JUnit](evidence/pdf_preview_cache_regression_20261017_logs.zip)保留本次检查。

### 继续修复：README 原命令回退

GREAD 的原模型提案已修复，但模型不可用时的回退仍只识别独占 `cd` 行，忽略作者 README 的 `navigate into src` 目录说明；识别目录后还会误选 `--kwargs` 占位示例。新增目录回归先得到 **16 failed / 2 passed**，占位命令回归先得到 **1 failed / 1 passed**。修复只采用作者明确给出的目录并记录原文依据，通过现有工作区边界校验；没有目录证据时不猜同名脚本位置。跳过当前入口未声明的 `--kwargs` 示例，保留后续完整作者命令；无关脚本的参数声明不能替当前入口背书。

最终新增检查 **25 passed / 0 skipped**，包括目录/脚本 symlink、越界目录、重复脚本、完整参数、无默认训练截止及原文件不变；相邻规划、解释器和环境检查在 CPU fixture 中 **93 passed / 0 skipped**。另一次应用环境综合检查 **222 passed / 15 skipped**，缺原生依赖的跳过不作为能力证明，各组不相加。[JUnit 归档](evidence/readme_author_command_regression_20261017_logs.zip)保留失败和修复后的记录。原 GREAD 来源包只读回放已从 README 选中完整作者命令；真实网页训练继续另行验证。

### 继续修复：原生依赖、解释器与终态回执

`68a125e` 的[常规及原生 CI](evidence/pdf_acceptance_readme_fix_ci_20261017.json)全部通过，原生四项零跳过。随后冻结该提交，从真实 Chrome 上传 Critical Look，真实 API 完成发现、规划并执行作者 ResNet/roman-empire 原命令。主机实际有可用 NVIDIA GPU，自动选择作者支持的 `--device cuda:0` 有硬件及源码依据；没有手工改计划。作者入口首先报告缺少 DGL，依赖修复安装 DGL 2.2.1 后，健康检查缺少对应 Torch 2.5.1 的 GraphBolt DLL。相同 wheel 干净重装一次仍失败，**训练更新为零**，不能声称作者训练或比较完成。[原始失败证据](evidence/critical_look_gpu_regression_20261017_logs.zip)已保存，终态后重新核对冻结源码一致，全部所属作用域归零，依赖和训练解释器实际释放。

据此补齐两项修复。CPU 精确 `dgl==1.1.2` 使用本阶段已验证的 CPython 3.11，并与作者 Python 声明取交集；运行时才发现 DGL 时也先释放旧环境，再选择兼容解释器重试完整原命令。该约束标记为项目已验证组合，不能当作 DGL 上游 Requires-Python。冲突声明在下载和安装前失败；保留 patch 版本约束，不对 CUDA 或其他 DGL 版本套用。失败先行及修复后 **100 passed / 0 skipped**，相邻 CPU 原生服务检查 **95 passed / 0 skipped**，见[解释器修复证据](evidence/dgl_native_python_repair_20261017_logs.zip)。

GraphBolt 的缺库错误仅在安装元数据、Torch 版本、安全 CSV RECORD 和实际其他版本 DLL 共同证明目标 ABI 未包含时分类为 `native_abi_incompatible`，停止重复下载相同环境。目标已声明却文件遗失、RECORD 缺失或证据不足仍保留原有一次损坏缓存重建。依赖安装与健康检查 **102 passed / 0 skipped**，见[ABI 分类证据](evidence/graphbolt_native_abi_repair_20261017_logs.zip)。此修复不会自动换作者设备或声称 Windows CUDA DGL 已支持。

本次浏览器还捕获到清理回执发布竞态：child 已发出 done，supervisor 先关闭 worker，再写最终清理回执，短暂向网页公开了旧回执。实际清理成功，但两项一致性检查失败，原结果保持原样。修复先持久化全部最终回执再关闭 worker；取消、异常退出也保留报告和一致回执，并同步热加载接口版本。四项竞态、一项缓存升级、两项中断回归均先复现失败；最终相邻检查 **175 passed / 0 skipped**，见[清理发布证据](evidence/cleanup_publication_repair_20261017_logs.zip)。三项修复整合后的六文件检查 **192 passed / 0 skipped**，各组覆盖重叠，不相加。完整 CPU 网页回归及新的独立 PDF 验收仍需实际执行。

| 固定顺序 | 作者执行 | 有效比较 | 独立验收失败阶段 | 修复后的证据范围 |
| --- | --- | --- | --- | --- |
| 1. GCNH | 十份划分、6000 次更新、自然退出 0 | reference_only，含 PDF 引用与设置差异 | 运行目录与发布报告不一致，正式刷新检查尚未到达 | 保存结果副本回放、真实历史页恢复与下载一致性；回归 |
| 2. GREAD | 未执行 | 无 | 规划器拒绝同一嵌套作者入口的路径表达 | 原始被拒提案只读回放、145 项规划/原生检查；回归 |
| 3. Critical Look | 未提交任务 | 无 | 上传预览优先显示断字截断链接 | 原 PDF 提取和真实网页上传预览；回归 |

原批次三篇都已退役，原候选顺序及筛选回执保持不变。新批次已在训练前另行登记 **PyGSD → Polynormer → BN-Pool**，见[独立批次清单](../examples/papers/candidates_20261017_batch2/manifest.json)和[增量登记回执](../examples/papers/candidates_20261017_batch2/registration_receipt.json)；并非向原三篇批次追加第四篇，也未依据实测成绩改选。三篇均未训练或参与产品修复，只通过身份与静态来源筛选，不能宣称依赖、自动规划或有效比较已通过。全部修复及 CI 通过后重新冻结，按该顺序作真实网页独立验收；不能复用退役论文补作独立通过。当前仍缺少一篇在固定产品上贯通上传、发现、完整作者执行、有效比较、刷新/历史、Markdown/ZIP 和结构化结果一致性的新样本。

GREAD 的 pykeops 安装、完整作者训练和比较均未实测；Critical Look 已进入执行链路，但 GPU 回归失败于依赖健康检查、没有训练更新。这些事实不能写成环境已验证或数值不达标。#30 的原命令回归已完成，但独立重载、指标复算与完整论文协议核验仍未完成，issue 保持开放。#41 在 #27 门槛满足后再推进单一受支持实验，#20 网页修复入口及 #42 Windows 安装包继续留在后续阶段。

所有实际运行结束后均记录所属作用域归零和依赖清理状态；原始数据、源码、失败结果和日志保留。原有四个未跟踪文件的 SHA-256 与工作前记录一致。最终提交的远端常规/原生 CI 及剩余事项统一在 [PR #50](https://github.com/Yukikazct/AutoReproducer/pull/50) 跟踪。**#27 不关闭；作者执行完成、比较完成和独立数值复现继续分别表述，`paper_metrics_verified=false`、`is_reproduced=null`。**
