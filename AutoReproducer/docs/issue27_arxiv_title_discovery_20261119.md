# #27：有 arXiv 编号、无代码链接的 PDF 仓库发现

2026-11-19，在当前工作树上，用 [ResNet 原始 arXiv PDF v1](https://arxiv.org/pdf/1512.03385v1) 实测。PDF 共 12 页、819,383 字节，SHA-256 `1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a`。从第一页提取标题 `Deep Residual Learning for Image Recognition`，从 PDF 来源取得编号 `1512.03385v1`；全文文本和 URI 注释未提取到 GitHub 仓库链接。

| 阶段 | 可观察结果 |
| --- | --- |
| 修复前 | Hugging Face 论文索引超时；GitHub 用 `1512.03385v1` 检索，选中 `Shahip2016/ResNet`，候选中没有作者仓库。 |
| 修复后 | 论文索引仍使用编号；超时后以 `"Deep Residual Learning for Image Recognition" in:readme` 搜索 GitHub。`KaimingHe/deep-residual-networks` 排第 1，实测 6,777 星，作为未核验候选交给下一阶段。 |
| 固定源码核对 | 通过项目的仓库导出器固定到 `a7026cb6d478e131b765b898c312e25f9f6dc031`，仓库归档 SHA-256 `30c0cfcb0daf942020b57c3252540b9ebeb98f98db4d1a8aa65d95000d1f22ee`。README SHA-256 `57c86bfb74e810a844ed79137930e4c45b9d60ef60db38f9ddb3a1f80ec1efc8`，四位作者姓名、论文标题及 arXiv 编号都和原文一致。 |

补丁也防止论文索引返回的相邻编号挡住 GitHub 回退：匹配论文编号时忽略版本后缀，但不能把不同编号关联的仓库当作当前论文候选。显式指定的仓库或 PDF 作者代码声明仍优先；标题结果仍按标题校验相关性。

首次定向回归有 3 个用例失败，修复后通过；PDF 输入、仓库发现和候选交接等相关回归合计 **428 passed**。真实回放使用原始 PDF、在线搜索和固定 Git 归档，直接向 ResourceFinder 提供已核对的论文元信息，不调用模型为仓库提名。未运行网页或语义核验 Agent；原仓库提供模型与配置、将训练代码指向第三方重实现，本次没有运行作者训练或论文数值验收。#27 的完整 PDF 端到端验收仍未完成。
