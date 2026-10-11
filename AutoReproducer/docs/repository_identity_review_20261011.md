# PDF 与 Git 仓库联合核验

通用 PDF 路线的作者实现确认现由 `RepositoryVerifierAgent` 执行。系统重新读取实际上传的 PDF，提取真实候选链接，拉取该地址的 Git 仓库并固定完整 commit。Agent 阅读首页、链接所在页、README/CITATION 和文件目录，分别判断论文身份、作者实现关系和方法匹配。

模型选择程序提取的来源片段编号。程序重新校验页码、文件、原文及 PDF/源码哈希，再将结论绑定到固定 Git 版本；不能使用猜测 URL、参考文献链接、模拟模型响应或调用者提供的确认布尔值。最多检查三个原文候选，保留每次实际来源、判断和失败原因。单次引用抄写错误允许一次修复，第三方、无关或证据不足的语义结论不会被反复提示改成通过。

2026-10-11 使用 `deepseek-chat` 的真实 API 回归：

| PDF | 实际仓库 | 结果 |
| --- | --- | --- |
| Simple Spectral Graph Convolution | allenhaozhu/SSGC 的既有固定 Git 快照，重新校验源码哈希 | author_repository，三个检查均通过 |
| Improving Graph Neural Networks with Simple Architecture Design | 在线拉取 graphdml-uiuc-jlu/geom-gcn，commit `1124af17444d7fd09686504ec46caa3f39f4f632` | third_party；确认只是数据划分来源，不允许启动作者实验 |

上述 PDF 已用于此前排查，仅属于仓库关联的正反例回归。它们不构成独立新 PDF 的完整实验验收，也不足以证明在广泛论文上的识别准确率提升。最终成功仍须使用另一篇论文，通过真实网页上传、完整作者实验、独立评估和事前数值标准。
