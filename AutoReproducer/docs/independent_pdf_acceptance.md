# 独立新 PDF 验收规则

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
