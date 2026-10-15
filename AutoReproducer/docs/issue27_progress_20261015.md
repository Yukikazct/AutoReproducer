# #27 指标比较与 R-FCN 实测记录（2026-10-15）

## 当前结论

指标提取、证据配对、报告统计信息及真实 PDF 路线的安装进度转发已修复。DGC 完成了一次有有效数值比较的真实 API / 浏览器回归；用户随后指定先测试新上传的 R-FCN，因此独立论文结项验收尚未完成，#27 保持开放。独立重载、指标复算和完整协议核验仍属于 #41。

“从论文复现中学到了什么”的[子 issue 完整草稿](issue27_lessons_subissue_draft.md)已经保存。本轮没有实现该总结功能，也没有把未通过验收的运行绑定为已成功样本；正式 GitHub 父子关系待前置验收通过后创建。

## 产品修改

- 最终汇总优先于逐次结果，识别明确的 mean/std、±、重复次数，保存 `aggregation`、`runs`、`std` 和原输出位置。结构化输出继续优先；缺失信息保持未知，明确的置信区间不当作标准差。非指标的 `latency mean ...` 日志不会中断其他指标提取。
- LLM 同时读取论文原始表格、列标题、指标定义、固定作者命令和源码默认参数。程序校验数据列、指标和方法引用，均值与标准差均须来自对应数据集的同一表格单元，实测统计字段不能被模型覆盖。只规范化 `(ours)` 等作者署名，不删除真实方法变体。
- 首次提取后最多一次针对性纠错；无效配对单独保存，同次响应中的有效结果保留。不可比和失败原因进入保存的报告；5% 相对误差仍为项目参考规则。
- 报告展示双方的标准差、次数和原始引文。网页、历史、Markdown、ZIP 继续读取保存的同一结果；作者命令完成与论文数值验收保持各自含义。
- 真实 PDF 后端和通用编排器现在转发安装心跳。修复前直接安装器测试能看到进度，真实 PDF 路线的事件白名单却丢弃了它。
- 规划器向 LLM 返回实际存在的同名入口路径，由模型纠正工作目录；README 没有逐步命令时也可补读训练入口、参数解析器和数据加载代码。界面阶段名改为“结合论文和源码规划作者命令”。
- Python 2 源码不再使输出目录分析器抛出未分类的异常：兼容性诊断会回传模型并保留 `runtime_unsupported`、路径、行号与源摘要。候选仓库的规划失败依据分别归档。

## 保存日志回归与 DGC 浏览器运行

[真实保存日志的重新解析结果](evidence/issue27_saved_metrics_replay_20261015.json)：

| 样本 | 最终实测值 | 统计证据 |
| --- | --- | --- |
| 原 DGC 保存日志 | 0.83280 ± 0.00040 | 明确 10 次均值；不再采用最后一次 0.8330 |
| 原 DiGCN 保存日志 | 0.7719 ± 0.0044 | 该输出行未说明次数，解析器保留未知；验证指标独立保留 |

DGC 浏览器回归 `issue27_dgc_20261015_03`：

- 产品冻结摘要 `b69f13f93eff5c4b99461d8c029bf444a14fe1296fafa7fc8e2d5d306eeca865`，作者提交 `f91b35a0d4388660e250d575d4c1b8fef41d7039`。
- 只通过浏览器上传原 PDF，真实 DeepSeek API，自行规划作者 Cora 命令。原生 10 次、每次 100 轮，共 1000 次优化器更新，正常退出。
- 此次随机运行实际均值 **0.83290**、标准差 **0.00030**，论文 Cora 行为 **83.3 ± 0.0% / 10 次均值**。差值为 **−0.01 个百分点**。没有将保存日志的 0.83280 硬编码到新运行。
- `paper_metrics_verified=false`、`is_reproduced=null`；作者输出比较不代表独立数值复现。
- 首次实时页 ZIP 保存返回 `Download.save_as: canceled`。随后使用新浏览器会话恢复同一记录，Markdown 和 ZIP 下载成功，ZIP 内报告逐字节一致；刷新后的比较表相同，没有新建任务。
- 清理释放 **1,523,415,699 字节**；训练 Python、依赖及临时下载已移除，报告、结构化归档与清理回执保留；终态活动进程全为零。

[DGC 完整精简证据](evidence/issue27_dgc_comparison_20261015_03.json) · [恢复后的网页对比表](evidence/issue27_dgc_comparison_20261015.jpg)。该运行保存的真实 API 提案另经当前提交 [62d9132 重新校验](evidence/issue27_dgc_revalidated_62d9132.json)通过；这次重新校验未训练、未调用 API。该浏览器运行早于后续入口诊断修复，属于回归证据，不能代替最终产品版本的独立新论文验收。

此前两次 DGC 比较未通过，分别暴露 `(ours)` 方法名对应和误把方法名填入指标名的问题；修复后才得到上述有效对比。GraphCON 未正确定位原作者子目录入口，回退至参考实现后被停止，未列为成功训练或验收样本。

## 用户指定的 R-FCN

输入为 `examples/papers/pdf/NIPS-2016-r-fcn-object-detection-via-region-based-fully-convolutional-networks-Paper.pdf`，9 页、645,670 字节，SHA-256 为 `374eb111e85610c20646b165aa6ed37ba7fa8bfcc6df0f068c086ba58ef11d2a`。

原文第 1 页指向 `daijifeng001/r-fcn`，固定提交 `94797e0e8d15998a9ab0a76cbac3281ad907f04a`。作者 README 链接的 Python 实现为 `Orpine/py-R-FCN`，固定提交 `290dba7fad47f1ec74e2d1675e72cca6efb6cc14`。

第一次真实上传到达 Python 实现后，在读取旧语法时抛出 `Missing parentheses in call to 'print'`；训练未开始。该样本用于修复兼容性诊断，后续作为回归样本，不作为修复后的独立 holdout。

第二次真实浏览器/API 流程使用冻结摘要 `2d348800d06678ab67c19b920b04ef407b9ae634c81a551c7c43c92385b4ccce`：

1. LLM 为原 MATLAB 库选择了真实的 ResNet-50 OHEM/RPN `.m` 脚本；当前 Python 执行器不能执行该入口。
2. LLM 为作者链接的 Python 版规划了完整 ResNet-50 / VOC / 110000 次迭代命令；Python 2 源码的兼容性问题被正确回传和归档，没有再使规划器直接崩溃。
3. 自动回退至 `iahmedmaher/r-fcn` 的固定源码，依赖首轮安装成功，随后在 `torchvision.models.resnet.model_urls` 导入处退出码 1。这个仓库的 README 明确标记 WIP，入口只做一次前向计算，即使成功退出也不能算完整 R-FCN 训练。
4. 此次没有训练指标、没有 mAP，没有论文数值复现结论。不能将第三方 WIP 当作原作者实现通过。
5. 报告生成后释放 **1,434,044,119 字节**，无剩余依赖或训练解释器；活动 worker/run/execution 均为 0。浏览器恢复、Markdown/ZIP、保存归档的报告文本一致，刷新保留失败状态且没有新建任务。

[首次失败](evidence/issue27_rfcn_browser_20261015_01.json) · [第二次完整精简证据](evidence/issue27_rfcn_browser_20261015_02.json) · [产品冻结文件](evidence/issue27_rfcn_20261015_02_source_freeze.json)。冻结文件中的 Git 提交是冻结时的基线；当时尚未提交的产品文件以逐文件 SHA-256 为准。

R-FCN 摘要的 **83.6%** 是 ResNet-101、多尺度训练及 VOC+COCO 的结果；VOC 07+12、单尺度、ResNet-101 的论文参考为 **79.5%**。这些是论文参考，均不是本机测量值，不可混用。

## 原生运行环境的实际检查

用户要求继续实测兼容环境后，启动本机已安装的 Docker Desktop。复用已有 `python:3.11-slim` 镜像、使用共享环境锁串行执行 GPU 检查，Linux 容器实际识别到 RTX 4060 Laptop、8188 MiB、驱动 595.97；退出码 0。未下载新镜像，测试容器已释放，原有镜像保留。这证明 GPU 可见，不证明 Caffe 已编译或 R-FCN 已训练。

[GPU 透传证据](evidence/issue27_rfcn_gpu_probe_20261015_01.json) · [容器内下载来源检查](evidence/issue27_rfcn_native_network_20261015_01.json)。旧 CUDA Docker Hub 查询遇到连接重置/超时，未据此断言镜像不存在。Python、NVIDIA 和固定 Caffe 源码入口实际可连接，Debian 官方镜像域名出现解析失败，因此原生构建测试使用有签名校验的 Debian 镜像。

固定 Microsoft Caffe 提交 `1a2be8ecf9ba318d516d79187845e90ac6e73197` 已完成 CUDA 12.8 构建和 GPU 初始化，PSROI Pooling GPU 前向健康检查退出码 0、输出校验和 49。此为独立合成输入的原生环境检查，未接入产品自动训练路线，也未开始完整作者训练。[R-FCN 完整测试说明](rfcn_preflight_20261015.md)逐条保存了五次检查、测试器修正及来源限制。作者 OneDrive 权重入口当前超时，匿名 API 返回 401；从公开镜像取得的权重已保存 SHA-256，未与作者失效入口逐字节核对。

## 测试范围

- Windows / CPython 3.12.10 / UTF-8 最终产品全量：`python -X utf8 -m pytest tests/ -q --tb=short`，**3219 passed、87 skipped、0 failed**，566.57 秒，仅有既有 PyPDF2 弃用警告。
- 包含规划纠错、旧语法诊断、指标、比较、报告与服务的相关定向组合：**143 passed、15 skipped**，63.28 秒。组合与全量重叠，不相加。
- 提交 `62d9132` 的 [CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/38721145017) 四个矩阵任务全部成功：Ubuntu / Windows × Python 3.11 / 3.12，包含 Windows 系统编码与 UTF-8。
- 跳过项涉及未配置的专用 CPU/CUDA 原生缓存、Notebook、matplotlib/torch、POSIX 信号语义，不能当作这些环境通过。具体数量和原因见[测试摘要](evidence/issue27_regression_20261015.json)。
- 一次未指定 `tests/` 的 pytest 命令误收集了下载缓存中的上游 Caffe 测试；按项目测试目录重新运行全量后得到上面的通过结果，没有执行上游训练作为项目测试。
- 原生测试缓存改为显式配置，避免测试借用实时运行后会释放的生产依赖。应用虚拟环境和用户原有文件保持保留。

独立新 PDF 的有效指标对比仍未通过，因此不关闭 #27，也不将本轮标记为整篇论文复现完成。浏览器测试工具目录 `.tmp/browser-tools` 的既有删除操作曾被自动审批拒绝（`blocked by policy`），该测试工具目录仍保留；运行时清理结果指受管理的训练解释器、依赖和临时下载。


原生检查和宿主临时下载的最终释放证据见[清理审计](evidence/issue27_rfcn_final_cleanup_20261015.json)；实验性原生健康检查没有改变原 R-FCN 失败运行的结论。
