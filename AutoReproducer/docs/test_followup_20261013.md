# 2026-10-13 测试跟进

## Windows 并发清单读取

主分支 GitHub Actions 运行 `38349367975` 的 Windows / Python 3.12 / 系统编码任务，在 `test_concurrent_manifest_readers_see_complete_documents` 中读取到了 `None`；同一运行的 UTF-8 任务及其他矩阵任务通过。

清单写入使用临时文件和原子替换，但 Windows 在替换期间仍可能拒绝读取文件。此前读取方法将此异常解释为“没有清单”。本次让保存、单个读取和列表读取共同使用按清单分配的进程间文件锁，原子替换仍保留。新增真实子进程持有写锁的测试，确认读取等待写入完成后返回新清单。

本地 Python 3.12 / UTF-8 全量回归：**2945 passed, 70 skipped**，耗时 312.88 秒。跳过项包括未配置的可选原生依赖与不适用的进程行为，不能算作这些环境的通过证据。远端矩阵仍需验证本次提交。

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

测试只向网页后端入口提交 PDF，没有传入仓库 URL 或手工运行命令。任务 `progress_1791875237497.jsonl` 已进入依赖安装，尚无最终执行结论。当前 Computer Use 工具返回空浏览器清单，因此本轮尚未验证真实浏览器上传；后端任务不能记作浏览器验收。
