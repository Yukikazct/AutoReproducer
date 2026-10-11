# PDF 提取与 GitHub 仓库定位修复（2026-10-10）

## 当前流程的失败原因

用户上传的 DeepXDE PDF 与原始 `data/papers/deepxde_1907.04502.pdf` 相同：21 页，1,118,278 字节，SHA-256 为 `d17f9bf6b8f346baaa52b6ec81223c53dd3da6899324114c72c7e8c3432d9368`。

保留的失败记录 `data/runtime/progress_1791641433654.jsonl` 显示：

1. PDF 已成功提取 58,694 字符，文件读取本身没有失败。
2. 第 2 页脚注明确说明源码公开，但 URL 在 PDF 提取文本中断行为：

   ```text
   Source code is published under the Apache License, Version 2.0 on GitHub. https://github.
   com/lululxvi/deepxde
   ```

3. URL 在全文偏移 7,644 处，超出 Reader 给模型的前 3,000 字符。旧正则只接受连续 `github.com`，因此 `extracted_code_urls=[]`；Finder 的普通 URL 正则只得到无效的 `https://github.`。
4. 原文链接丢失后，PwC 查询超时，GitHub 词面搜索选中了第三方 `yuan666-G/deepxde`。来源核验两次都拒绝了这个候选。
5. 原任务于 22:14:53 结束并完成工作进程清理，最终 `not_runnable` / `failed`，`repository_executed=False`；没有成功执行作者实验。

标题输入的 `progress_1791641311248.jsonl` 也失败，但本次先修复 PDF 原文到仓库的路径；没有给 DeepXDE 添加标题映射或专用实验预设。

## 修复后的路径

| 环节 | 行为与依据 |
|---|---|
| 文件验证 | 从同一字节快照提取文字、超链接、页数及 SHA-256；坏 PDF 仍在模型请求之前拒绝 |
| 链接提取 | 扫描所有页面和 URI 注释，支持裸 `github.com`、明确断行、子路径、`.git` 与结尾标点；隐藏字符不会破坏原文偏移 |
| 来源分类 | 区分代码公开声明、普通仓库链接和参考文献；保留原始 URL、页码、来源和上下文 |
| Reader | 将完整文档中提取的仓库证据加入模型输入；PDF 的 `code_url` 由真实代码声明确定，模型猜测不能改写 |
| 上下文交接 | 保留结构化证据与 PDF 哈希；有逐页证据时不重新解释扁平全文，不制造第 1 页来源 |
| Finder | 用户指定优先，其次论文代码声明和正文链接，再走模型元数据与搜索；参考文献只保留为候选证据 |
| 页面与报告 | 上传预览立即展示原文代码链接和第几页；结果报告记录所选证据及 PDF 哈希 |

`paper_linked` 表示有本篇论文链接依据，`is_official` 仍由独立身份核验确认；找到仓库不等于已运行仓库或复现论文。仅有引用仓库的模型候选仍为 `candidate_unverified`。

## 实际验证

原始 DeepXDE PDF 提取出第 2 页文字与 URI 注释两条独立来源，均为 `https://github.com/lululxvi/deepxde`，来源类型为代码公开声明。

真实 Reader → Finder → Verifier 使用现有 API 配置完成 3 次调用，Reader 没有降级，Finder 选择作者仓库，质量核验通过。验证器仍对数据源列表形式和置信度提出非阻塞建议；没有执行环境安装、代码生成或训练。因为有原文链接，不调用 PwC/GitHub 关键词搜索。证据保存在本机忽略目录的 `real_pdf_repository_acceptance_20261010.json` 和 `pdf_repository_discovery_report_20261010.md`。

真实 Web 上传预览通过现有 `localhost:8501` 应用协议验证。新验收会话显示：

- `PDF 原文代码声明链接：https://github.com/lululxvi/deepxde`。
- 来源“第 2 页”，以及可点击的代码公开声明证据。
- 未点击“开始复现”或 API 连接按钮；未读取密钥控件，未启动训练。

证据为 `live_pdf_repository_preview_acceptance_20261010.json`。前后服务均为 PID `16544`、启动时间 `2026-10-10 15:40:36`；没有重启服务或操作用户原有浏览器会话。PDF 预览入口可以隔离加载新版解析器，不修改旧读者正在使用的模块全局变量。

## 回归验证

覆盖真实二进制 PDF 的断行脚注、只有超链接的仓库、参考文献、Unicode 隐藏字符、原文偏移、模型错误候选、页码与哈希交接、报告以及上传页面预览。最终源码稳定后的全套回归 **1994 通过、1 跳过**，耗时 148.30 秒，仅保留 PyPDF2 已有弃用警告。命令：`.venv-py312/Scripts/python.exe -X utf8 -m pytest tests -q --junitxml=data/runtime/pdf_repository_discovery_tests_20261010.xml`。`git diff --check` 通过。
