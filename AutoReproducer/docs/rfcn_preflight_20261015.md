# R-FCN 新上传 PDF 的真实测试（2026-10-15）

## 结论与范围

已进行真实浏览器上传、真实模型规划、作者仓库与链接实现检查，以及隔离 Linux/CUDA 原生环境测试。**完整 R-FCN 训练尚未开始，没有实测 mAP，不能标记论文复现成功。**

固定 Microsoft Caffe 已编译、链接并在 RTX 4060 上初始化 GPU；PSROI Pooling 的独立 GPU 前向健康检查通过。它与完整作者训练分别记录。

| 检查 | 结果 |
| --- | --- |
| PDF 与原作者链接 | 9 页，PDF SHA-256 `374eb111e85610c20646b165aa6ed37ba7fa8bfcc6df0f068c086ba58ef11d2a`；第 1 页链接 `daijifeng001/r-fcn` |
| 当前 GPU | RTX 4060 Laptop，8188 MiB，驱动 595.97；Linux 容器实际识别成功 |
| 产品自动路线 | 原 MATLAB 入口和链接 Python 2 入口暂不受当前 Python 3 执行器支持 |
| 第三方回退 | WIP 仓库在旧 torchvision API 导入处失败；该入口没有完整训练循环，不能作为作者实验通过 |
| 固定 Caffe GPU 构建 | 构建及 GPU 初始化均退出码 0，实际识别 SM 8.9 / RTX 4060 |
| PSROI Pooling GPU 健康检查 | 退出码 0；常数输入的预期校验和 49，实际两处输出均为 49；独立合成输入，不作为论文实验 |
| 论文训练及 mAP | 尚未完成；数值复现结论保持未知 |

## LLM 实际规划了什么

模型先选择原 MATLAB 库的 ResNet-50 OHEM/RPN 脚本，然后跟随作者 README 中的链接检查 `Orpine/py-R-FCN`。其规划的完整原始命令为：

```text
python tools/train_net.py --gpu 0 \
  --solver models/pascal_voc/ResNet-50/rfcn_end2end/solver.prototxt \
  --weights data/imagenet_models/ResNet-50-model.caffemodel \
  --imdb voc_0712_trainval --iters 110000 \
  --cfg experiments/cfgs/rfcn_end2end.yml
```

该命令与作者的 `experiments/scripts/rfcn_end2end.sh` 对应。Python 2 的 `print` 语法在当前产品分析器中不受支持，错误发生在训练之前。[浏览器/API 运行证据](evidence/issue27_rfcn_browser_20261015_02.json)保留了逐个仓库的提案、失败和最终执行输出。

当前模型能够结合源码规划命令；执行器支持的语言、原生依赖构建与进程管理仍须由程序实现。已修复子目录入口反馈、旧语法错误分类及其归档，代码提交 [62d9132](https://github.com/Yukikazct/AutoReproducer/commit/62d91326882cbf0f487a005d3cb33fc60dc09ef0)。

## 固定来源与原生环境

| 对象 | 固定版本或来源 |
| --- | --- |
| PDF 原链接实现 | `daijifeng001/r-fcn@94797e0e8d15998a9ab0a76cbac3281ad907f04a` |
| 作者链接的 Python 实现 | `Orpine/py-R-FCN@290dba7fad47f1ec74e2d1675e72cca6efb6cc14` |
| Python 实现要求的 Caffe | `microsoft/caffe@1a2be8ecf9ba318d516d79187845e90ac6e73197` |
| 原生测试基础系统 | 官方 Ubuntu Base 22.04.5 amd64；已核对官方 HTTPS SHA256SUMS |
| CUDA | NVIDIA 仓库 CUDA 12.8，编译目标 SM 8.9 |

原生检查逐步遇到了 Debian 13 对旧 NVIDIA 签名密钥的拒绝、Docker 旧构建器的大镜像层解压错误，以及旧 Caffe 构建配置缺失 HDF5 高层库链接项。测试改用 Ubuntu 22.04、隔离的 BuildKit 和显式链接库；保留软件包签名验证，作者模型源码和训练参数未作更改。该原生测试尚未接入产品的自动训练路线。

[原生测试原始记录](evidence/issue27_rfcn_native_build_20261015_05.json) · [独立输出核验](evidence/issue27_rfcn_native_verified_20261015.json) · [测试配方与源码快照](evidence/rfcn_native_20261015/manifest.json)。

补充健康检查的两个测试器问题单独保留：首次误用要求权重参数的 Caffe `test` 入口，退出发生在执行算子之前；修正调用后，程序实际退出码 0 并输出 `checksum = 49`，初始检查器却匹配了其他版本的 `Test score #0` 日志格式。原始结果中的校验失败标记未被覆盖，独立核验按该固定版本的真实输出确认通过。

该网络无可学习参数；传入的分类权重只满足旧 CLI 的必填要求，日志显示其层被忽略。`49 = 2 × 7 × 7 × 0.5` 是常数输入的 GPU 算子校验和，不是检测准确率或 mAP。该构建关闭 cuDNN，且 `BUILD_python=OFF`、`BUILD_python_layer=OFF`，只验证原生 Caffe/GPU 前向路径，未验证 Python 2 训练、完整检测网络、OHEM 反向过程或模型精度。

继续完整复现仍需 Python 2/pycaffe 及作者扩展、全量真实 VOC 数据、原始 110000 次训练与完整评估，以及产品中受控旧运行环境的执行接入。

## 模型输入与数据

作者 OneDrive 原地址当前连接超时，匿名文件 API 返回 401。历史 Microsoft/MXNet 下载地址有返回 HTML、证书主机名不匹配等问题，这些响应未当作权重使用。

从 DeepDetect 项目公开下载脚本指向的镜像取得了 ResNet-50 Caffe 权重：

- 大小 **102,462,397 字节**，二进制网络名 `ResNet-50`。
- SHA-256：`44ee2b08816cede2b7aaa047888df07dcab52f73399aa1c8bef05a17bfdd4888`。
- [来源与文件证据](evidence/issue27_rfcn_model_acquired_20261015.json)。尚未与作者失效入口逐字节核对。
- 完整 VOC 07+12 训练集和 VOC07 测试集仍需取得并校验；此前只确认备用下载入口能返回 tar 数据，没有把它当作已经下载完成的数据集。

本次规划的是 ResNet-50，不能拿论文摘要中 ResNet-101、VOC+COCO、多尺度训练的 **83.6%** 直接作为其参考；也不能把环境健康检查的数值当作检测 mAP。

## 测试与归档

产品修改通过 Windows / CPython 3.12.10 / UTF-8 全量回归：**3219 passed，87 skipped**。同提交的 [CI](https://github.com/Yukikazct/AutoReproducer/actions/runs/38721145017) 四个系统/Python 矩阵任务全部通过。跳过项及证据摘要见[测试记录](evidence/issue27_regression_20261015.json)。

[浏览器失败报告截图](evidence/issue27_rfcn_failure_report_20261015.jpg)。该报告已经验证刷新恢复、Markdown/ZIP 与结构化归档的报告文本一致，未重新创建任务。该次报告后释放 **1,434,044,119 字节**，训练解释器、依赖、所属进程均已清理，原始报告和源码保留。

#27 的独立新论文有效指标比较仍未通过，因此保持开放；独立模型重载、指标复算及协议核验由 #41 跟踪。“从复现中学到了什么”的[子 issue 草稿](issue27_lessons_subissue_draft.md)已保存，正式关联和成功运行绑定待前置验收完成。


## 最终清理

[独立清理审计](evidence/issue27_rfcn_final_cleanup_20261015.json)确认：新增原生镜像、测试容器、独立构建器缓存卷均为 0；训练解释器目录不存在，运行依赖为空。原有 `python:3.11-slim`、`python:3.12-slim`、`hello-world` 镜像保留。此次启动的 Docker Desktop 和 8530 测试服务已停止，用户原有 8515 服务保持。

官方 Ubuntu 基础包和临时 CA 副本共 30,056,734 字节已清理；原论文、源码、运行日志、报告和镜像模型输入作为证据保留。测试用 `.tmp/browser-tools` 的先前删除请求被自动审批拒绝，返回 `blocked by policy`，该工具目录保留，不计作已释放的训练依赖。

保存结果可用 `evidence/rfcn_native_20261015/verify_record.py` 只读复核；它不调用模型、不安装依赖、不启动容器、不改写原结果。
