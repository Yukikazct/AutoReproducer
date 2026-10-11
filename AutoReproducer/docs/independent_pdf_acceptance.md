# 独立新 PDF 验收规则

## 2026-10-11 收尾：ChebNetII 规划失败，暂停新 PDF 验收

按用户要求暂停继续寻找论文和训练，提交已验证修复，并将剩余问题提交为 GitHub issue。本轮尚无未参与修复的新 PDF 完整复现成功记录。

最后一次真实网页测试使用《Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited》。产品代码冻结为 `b4e164bd63ea44d5b55a897496836462770f012f`，冻结时间 `2026-10-11T14:18:36.834573+00:00`；随后从 <https://arxiv.org/pdf/2202.03580> 下载 20 页、713,026 字节的 PDF，SHA-256 为 `c32e1ebcd886611ff66293f211e7f1b9e1ffebc6668a4b853de3bc0d2f357e37`。

全文阅读、资源定位审查、作者仓库身份确认及实验协议选择通过。仓库为 <https://github.com/ivam-he/ChebNetII>，固定 commit `ded6c18cbe9673234071031767d17826ad632aca`。选定 Table 6 的 ChebNetII/Cora 半监督随机划分实验：每类 20 个训练节点、500 个验证节点、1,000 个测试节点，保留作者全部 10 次运行、原始早停与验证损失选择规则。作者命令在 `main/` 执行：

```text
python training.py --net ChebNetII --dataset Cora --K 10 --prop_lr 0.001 --prop_wd 0.05 --dprate 0.8 --semi_rnd True
```

规划器两次选择 `main/dataset_loader.py:DataLoader` 的 `author_loader` 准备路径，确定性检查均返回：

```text
No validated author experiment plan: AuthorLoaderUnsupported: Loader module has an unreviewed top-level import
```

该模块还包含其它数据集的导入和类定义，超出当前纯加载器准备路径的支持范围；固定仓库却已包含 `main/data/cora/cora/raw/ind.cora.{x,y,tx,ty,allx,ally,graph,test.index}` 的全部 8 个原始文件。收尾修复向规划模型提供从真实文件清单推导的完整 bundled 数据选项，并在不支持的加载器反馈中指出这些路径。原加载器是否使用这些文件、实验划分和数量仍须原始源码审核，最终 bundled 文件仍逐个核对固定 SHA-256。该提示修复只完成机械回归，未再次调用真实 API 或训练，不能宣称 ChebNetII 已解决。

本地失败任务为 `progress_1791728348815.jsonl`，实验目录为 `data/runs/repository_37defa7ddb6a426ca2135a3e050123a1/`；原始失败提案、协议、来源包及结果保留。网页终态为 ERROR，实验状态为 failed，终态活动 worker 和实验进程为零。训练、独立评估和数值验收均未开始。运行文件未上传；本记录仅公开必要摘要。

收尾检查：使用本地标准 Python 3.11 执行 bundled 数据规划、通用计划、实验协议、作者加载器和加载准备链路测试，共 **159 passed**；`git diff --check` 通过。这些回归不代替真实论文复现验收。

后续需验证真实规划是否正确使用作者已有数据、完善所选加载路径及其实际依赖支持，再冻结代码并换未参与修复的新 PDF 完成全部原生运行、独立模型重载和数值验收。ChebNetII 已转为回归样本。

## 验收规则及先前记录

最终验收必须使用未参与当前实现修复的新论文 PDF。DeepXDE、ReZero 和 DLinear 已用于开发或排查，只能作为回归样本，不能证明通用流程通过最终验收。

2026-10-11 的 DAGNN 论文《Towards Deeper Graph Neural Networks》在 `b9d5219` 冻结后获取，原 PDF 为 <https://arxiv.org/pdf/2007.09296>，7,333,332 字节，SHA-256 为 `221f0885de41ffc6732da7b540d1bfbca8972da04a125ff0c8f6b16a3604c81b`。真实网页任务 `progress_1791725239608.jsonl` 阅读全部 57,978 个提取字符，阅读与资源审查、作者仓库 `divelab/DeeperGNN` 的身份确认通过，固定 commit `4d22a7d37800b0acbb5701bc341da2c6b5452bd2`。实验协议模型认为作者只保留验证选中的测试标量、未保存权重，直接返回 insufficient；当前运行机制已支持在原始比较分支完成赋值后保留实际权重，但判别器没有正确使用这一能力。原失败完整保留于 `repository_324d906b394f4241acbe44b80bf6305d`，未训练，终态无活动 worker 或执行进程。后续通过程序从实际 AST 列出支持的原生选择点，把源码、分支、作用域和哈希提供给判别器；不能自动覆盖负向协议或声称运行完成。该 PDF 已转为回归样本，修复后的最终验收必须更换论文。

2026-10-11 的 JacobiConv 论文《How Powerful are Spectral Graph Neural Networks》在 `0b2a771` 冻结后获取。v1 未包含可提取的 GitHub 链接，停在上传预览，未创建后台任务；保持产品不变后取得 v2，SHA-256 为 `8a2020af117fd55687eb70c427baf5cd3ebf02ba9458836f7e6b3e120a2d3ba7`。真实网页任务 `progress_1791721398954.jsonl` 完整阅读 77,788 个字符，阅读与资源审查、固定作者仓库 `GraphPKU/JacobiConv` 核验通过，实验协议阶段因 README 的 `--dataset $dataset` 有限枚举模板未被命令校验支持而失败。训练未启动，终态活动 worker 和执行进程均为零。后续修复只允许绑定到同一入口命令及其紧邻明确枚举的字面值，不放宽协议、重复次数或指标验收。该论文已退役，下一次验收必须更换论文。

命令模板修复的 99 项相关测试通过。对上述退役样本保存的真实提案进行程序回归，模板校验通过后，协议仍拒绝 `lr1`：作者通过按数据集索引的配置字典向原始函数传入训练设置，而当前参数编译仅支持 argparse 字段。该后续问题单独保留于 `command_template_regression_followup.json`，未启动训练；不能把模板修复或此回归说成完整协议通过。

1. 完成通用实现与回归验证，记录执行代码的 Git 版本和文件 SHA-256。
2. 冻结后获取新的公开论文 PDF，记录下载地址、时间、字节数和 SHA-256。
3. 从真实网页上传 PDF 并启动；从原文链接提取候选，由 Agent 联合核对实际 PDF 和固定 Git 仓库的 README、论文引用与目录，确认作者实现关系，不手工指定该论文预设、命令或参考指标。
4. 系统核验并固定作者仓库版本，从原始论文和源码确定完整实验、依赖、真实数据、参数及数值验收规则。
5. 执行作者训练入口，等待实际进程与子进程完成；检查优化器更新、完整测试划分和运行产物。独立进程重载实际模型，重新推理并计算指标。
6. 同时满足执行、协议、独立指标及事前数值规则才算成功；单元测试、冒烟、源码发现或生成替代算法不算成功。结论只覆盖选定实验。

若新 PDF 的测试暴露问题且需要修改代码，该 PDF 立即转为回归样本。修复与重新冻结后必须换另一篇 PDF，直到新样本在冻结代码上直接完成复现。保留每次失败、修改、代码版本和新样本记录，不覆盖失败结果，不更换种子或放宽阈值来追逐通过结果。

运行记录中的完成状态与数值结论分别保存。正在执行的训练、评估或后台进程存在时，网页应保持运行状态；代码执行完成但数值未达到要求时，应显示实验已完成、数值未通过。

2026-10-11 的 GGCN 独立样本《Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks》于 `bc20605` 冻结后获取，PDF SHA-256 为 `e9e78e0ba131c39a0143e73114bad846d21224586f444534ba84cf1737dbb177`。真实网页上传后，全文阅读及审查、作者仓库确认通过；实验协议阶段失败，训练未启动，终态无活动 worker 或实验进程。原表格的 71.14 被首个解析器提取为 `71:14`，数值核验无法匹配；后续来源包还独立重解析了 PDF。已改为按字体乱码尝试完整第二解析器，并在阅读与实验判别间共享同一份分页原文。该 PDF 已退役，解析回归不是完整复现成功证据。

2026-10-11 的首个独立样本《Simplifying Graph Convolutional Networks》在网页预览阶段失败：PDF 将编号脚注和分栏正文粘连，正文的 `Github1` 未关联到同页的正确 URI annotation。已加入通用的同页编号关联：要求明确代码声明、相同编号和唯一仓库目标，排除引用、第三方和歧义。76 项相关测试通过，原 PDF 仅用于提取回归验证，未进入最终复现验收。

第二个独立样本《Simple Spectral Graph Convolution》从 UNSW 公开存档取得，PDF SHA-256 为 `d00c9f89f873b11e7201bf249bb76ad698f0a2f1a6046a43a96064d8188d04a0`，在提交 `cbef881` 冻结之后通过网页上传。作者仓库 `allenhaozhu/SSGC` 发现和固定成功；训练前的计划引用因纯空白差异被拒绝，未执行训练。修复仅允许空白序列等价，最终保存匹配到的原文及模型原提案；字符、数值、页码和语义审查仍严格核验。46 项规划测试和 17 项服务/路由测试通过。该 PDF 已转为回归样本，不能作为最终成功证据；其论文十次运行均值也不能用一次训练代替。

第三个独立样本《Adaptive Universal Generalized PageRank Graph Neural Network》于提交 `82d1b5e` 冻结后取得，原 PDF 为 <https://arxiv.org/pdf/2006.07988>，SHA-256 为 `ce84494f19fd6d74d1c2d19a9c4b5acfa5542a297cbcdd3755e5270bfcd2a942`。网页预览阶段未识别纯文本编号脚注：原文 `Our implementation is available online.1` 对应 `1https://github.com/jianhao2016/GPRGNN`，文件没有 URI annotation。已补齐无注释的行首编号链接和句号后的编号声明关联，并保持同页、唯一目标及非引用要求。83 项 PDF 相关测试通过，原文提取回归确认作者链接优先于第 6 页的第三方仓库。该样本未训练，已退役为回归样本。

第四个独立样本《Improving Graph Neural Networks with Simple Architecture Design》在通用 CPU/CUDA 原生多次运行能力通过 103 项相关测试、提交 `87a47f0` 冻结后取得。原 PDF 为 <https://arxiv.org/pdf/2105.07634>，SHA-256 为 `0485a517bd4359d844c6dcad74bd8186ae1ebcdeb23110ca1b1191a29c7bd6dc`。真实浏览器上传后，第一页摘要的 `Source code at https://github.com/sunilkmaurya/FSGNN` 被提取为普通仓库链接，未确认作者代码声明，训练未启动。补齐直接来源标签的识别：必须紧邻 GitHub 链接，仍排除引用、第三方、基线和不相关链接。该 PDF 已退役，下一次最终验收必须使用另一篇论文。

第五个独立样本《Simple and Deep Graph Convolutional Networks》于 `0d848e8` 冻结后取得，原 PDF 为 <https://arxiv.org/pdf/2007.02133>，SHA-256 为 `fda616230e48642bbda2218f82af71fa7e3de5e14e98dfa8c8a065cf451287a4`。真实网页上传后成功拉取 `chennnM/GCNII`，固定 commit `ca91f5686c4cd09cc1c6f98431a5d5b7e36acc92`；LLM 联合核对 PDF、README 与目录，论文身份、作者实现和方法匹配均通过。完整实验规划失败：提案把普通 Cora 数据文件路径作为 `source=dataset` 的可读取 NPZ 划分摘要引用，确定性检查拒绝；训练未启动。运行终态无活动 worker 或实验进程，失败记录保留于 `repository_f2ef8546d6174e6a8f7740b9ac96e989`。该论文不算最终成功，后续修复后必须更换样本。用户补充仓库功能解决仓库来源兜底，不替代此规划问题的修复。

第六个候选《Interpreting and Unifying Graph Neural Networks with An Optimization Framework》于 `82e0643` 冻结后从公开仓库取得，PDF SHA-256 为 `862f95d01b672cbbe41f0455f70196438c18b1dbfb29206016351b35c49fda05`。网页上传后，原文只提取到基线 DGL、DropEdge 链接，没有作者仓库链接，预览检查未满足，未启动后台复现任务。该候选不构成最终验收证据。实际查看了首页和第 9 页的表格/脚注，没有把基线仓库当作作者仓库。

第七个独立样本《DropEdge: Towards Deep Graph Convolutional Networks on Node Classification》于 `ce0f92d` 冻结后获取，原 PDF 为 <https://arxiv.org/pdf/1907.10903>，SHA-256 为 `0855a55c3d125dd50705f3f36b03083a468ace85a3a9ec8425f8b2362debe3f7`。真实网页上传创建任务 `progress_1791705971697.jsonl`；原文首页和 URI annotation 均给出 `https://github.com/DropEdge/DropEdge`。阅读后处理因词法规则未识别复数 “Codes are released on” 而把已提取的真实候选清空，两轮阅读审查均拒绝。任务明确 ERROR，未训练，终态活动 worker 和实验进程均为零。已修复候选与作者身份混淆，并补齐复数声明/基线排除：105 项相关回归及退役 PDF 的真实 API 阅读、资源定位审查通过。该 PDF 已退役，后续完整验收必须换另一篇论文。
