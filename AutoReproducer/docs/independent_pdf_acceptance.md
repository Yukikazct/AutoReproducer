> 2026-10-21 当前状态：ChebNetII / Cora 已在修复后的同一冻结版本完成 PDF/API 规划、十轮原训练、独立模型预测、论文数值比较和报告生成，正式终态 COMPLETED；详见[严格回归记录](chebnetii_strict_regression_20261021.md)及[统一论文台账](paper_acceptance_registry.json)。它仍是已参与修复的回归样本，#27 的未使用新论文验收尚未完成。下文保留历史方案/证据；其中“未实现”不自动代表当前代码状态。2026-10-17 的实施记录见[原文](pdf_acceptance_20261017.md)。

# 独立新 PDF 验收规则

已下载的原始 PDF 现已归档到 [论文样本目录](../examples/papers/README.md)，[来源清单](../examples/papers/manifest.json)记录各文件的 SHA-256、页数和下载依据；目录收录只表示保存测试资料，验收结论以本记录为准。

## 2026-10-12 继续验收：先稳定运行原作者代码

用户恢复修复任务，并明确当前阶段先做到论文提供的 GitHub 仓库代码能稳定运行。PR #32 已加入显式 `author_code_execution` 目标；严格论文数值路线仍保留。下面的数值验收规则与失败记录保持为原目标的历史证据，不作为当前作者代码执行目标的必需前置条件。

本轮使用真实网页上传、真实 API 仓库实现核验、固定原始源码与所选原始命令，检查原生环境、实际训练更新、各必需命令退出码和进程清理。用户随后明确放松执行前审查：README/requirements 优先提取依赖，实际缺包时安装并重试同一命令，全任务最多修复三轮；阅读、资源发现及计划的质量审查保留为参考，实际仓库与论文内容仍核对。数据暂时无法完整证明时允许作者程序自行加载，并记录未核验；作者批处理脚本中的单次原始训练可单独执行。网页结束时必须没有活动 worker、实验或执行进程，并明确显示“所选仓库代码已运行完成”和“论文数值尚未核验”。`author_code_completed` 的 `is_reproduced` 必须为 null，不能伪造论文数值或独立指标通过。

用户允许先准备 10 篇，因此本轮可以预先下载候选 PDF。每次试验开始前冻结产品源文件 SHA-256 和 Git 版本，记录 PDF 来源、获取时间、字节数、页数及哈希。真实测试暴露问题后，若修改产品代码，该论文即转为回归样本；再次冻结后换未参与修复的候选验收。准备候选、工程回归和论文数值验收分别记录。本节已记录 CCA-SSG 新 PDF 的真实网页作者命令完成，论文数值仍未独立核验。

首个新样本 BernNet 于 `49beb9a` 的源文件指纹 `7ba227152b630efacfb0678dadcc993aa1980cafe925e9aac0c36afae417d36f` 上测试，PDF 第一页成功显示 `https://github.com/ivam-he/BernNet` 的原文代码声明。但上传仅触发 `render_paper_input` 局部刷新，区域外的新执行目标复选框尚未渲染，实际网页无法在启动前选择目标。未创建后台实验，也未开始训练。修复将复选框移入上传区域，按 PDF SHA-256 保存选择，并由启动处理读取；该 PDF 转为回归样本，下一次使用 DGI。

DGI 在源文件指纹 `f56b89b39349491981fd09eca17bd35ae9ff7b480b1b85f184838a332a6d42d2` 上通过真实上传并创建 `progress_1791775506944.jsonl`。本地旧凭据实际返回 HTTP 401，阅读器将 API 错误当成无效 JSON，再尝试本地提取和一次内容修正，最终显示“论文解析未通过”。没有开始仓库训练，终态活动 worker、实验及执行进程均为零。修复让阅读 API 错误直接保留 HTTP/网络原因并停止内容修正，诊断消息遮蔽凭据；下一篇使用已测试通过的新凭据，从正常网页密码输入框提交。DGI 的失败记录不改写为成功。

PairNorm 在源文件指纹 `fa2fe964c31119ef91564b4f428ac2723aee6556b845943779dbeed93102441c` 上创建 `progress_1791775810241.jsonl`，阅读器正确直接报告 HTTP 401，未开始训练，终态活动 worker、实验及执行进程均为零。新凭据通过项目连接函数测试成功，但浏览器自动填入密码并按 Tab 后，Streamlit 尚未提交输入，任务仍使用旧值。随后独立浏览器诊断确认：密码框视觉值虽然已经改变，连接按钮仍返回旧凭据的 401；按 Enter 提交后网页连接成功。只修改验收脚本的输入提交与连接前置核对，产品指纹保持不变。GIN 首次连接前置检查未通过，未上传或启动实验；该前置失败保留，之后用新标签 `author_gin_20261012_04b` 从网页成功测试连接并上传 PDF，创建 `progress_1791776402193.jsonl`。不能把 API 连通或全文阅读通过当作作者训练完成。

GIN 的 `progress_1791776402193.jsonl` 使用新凭据通过完整阅读和作者仓库确认，固定 `weihua916/powerful-gnns` 的 `9a2ce8ac3e99278307093a464a95caf0fb04b602`。随后规划两次将真实文件路径 `dataset.zip` 填入 `evidence_id`，终态明确 ERROR，未开始训练，所有活动计数为零。原 README 还要求 `unzip dataset.zip` 后执行完整默认 `python main.py`，此前准备器只校验 ZIP 文件而未支持此原始准备步骤。修复为库存文件提供可选证据编号，并支持按原 README 指定目录解压完整固定 ZIP：程序核对压缩包及每个成员的大小与哈希，拒绝覆盖作者源码、重复路径、目录穿越、链接和执行元数据，保存解压数据来源，执行前后再次核验。提示同时明确方法需要 PDF 与仓库证据、兼容依赖由程序推导。GIN 转为回归样本；下一次网页验收换 GRAND。

GRAND 在产品指纹 `761f5111c9e6bda2cbec5fd6ce1d906fcb94296a9980e99cec9fa0bf35b71ac5` 上创建 `progress_1791777090482.jsonl`。全文阅读、仓库 `Grand20/grand` 核验、原始完整 Cora 命令规划和 8 个 bundled 原始数据文件校验通过，随后开始准备 CUDA 依赖。规划器照搬作者机器的 `--cuda_device 2`，但实际本机只有逻辑 GPU 0；训练尚未开始时确认不兼容，停止已核对所属关系的测试 worker。持有者确认清理，终态 ERROR，worker、环境准备进程和实验全部退出，活动计数为零，未宣称作者代码成功。修复向规划器提供实际设备信息，只有原整数 CLI 字段仅用于 `torch.cuda.set_device` 时才允许选择本机可用序号；训练参数、数据和种子不能据此改变。相同结构校验提前拒绝不存在的 GPU，避免先下载依赖。GRAND 转为回归样本，下一篇换 GCA。

GIN 的静态回归还发现原源码 `sys.path.append("models/")` 对应的 `models/mlp.py` 被误当作第三方包 `mlp`。补齐作者原始顶层字面量 `sys.path.append/insert` 的路径识别，并保持源码顺序和仓库边界；不创建替代模块。机械回归已经能固定 162 个 ZIP 成员、147,765,791 个真实解压字节并保留原 `python main.py`；这不是再次训练或最终验收。

GCA 在产品指纹 `536ba8a9ae7dbf2907237658d05f0df84951dfc92d15e9335843dfc10ad94666` 上完成真实网页连接测试、PDF 上传、全文阅读和 `CRIPAC-DIG/GCA` 作者仓库核验，固定 `0f17497f3ae89718ed610c671a5b9f56033d4d51`。规划两次被 `Author argparse declaration requires unsupported dynamic/action semantics: save_split` 拒绝：原参数是 `type=str, nargs='?'` 的可选输出路径，当前完整 CLI 编译器尚不支持。真实源代码还包含动态添加超参数、WikiCS/OGB 数据分支及用户目录加载路径，不能把单独放行可选参数说成整个仓库已可执行。终态 ERROR，活动计数全部为零，未准备数据或开始训练，失败目录为 `repository_958dd49f052f4e15b8fbf19e447675eb`。随后增加运行时数据路径，让原程序自行解析复杂 CLI 并调用原始 PyG 构造器；GCA 转为回归样本，未重新作为新 PDF 验收。

显式作者执行目标支持 `author_runtime` 数据声明。声明绑定原文件、函数、框架类和真实数据名称；观察器保留原构造参数与变换，记录完整 raw/processed 文件的大小、SHA-256、样本数及训练前加载时序。数据阶段在构造器尚未执行时保持等待，只有作者命令自然返回、真实数据证据通过且所属进程全部结束后才能报告作者执行完成。论文数值目标不使用这条放宽路线。

GRACE 的真实网页测试通过阅读与 `CRIPAC-DIG/GRACE` 作者核验，固定 `b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`，随后规划因原 `get_dataset` 定义在 `if __name__ == '__main__'` 分支而被仅查顶层的函数解析器拒绝。目录为 `repository_3a2a34fb5b05488a9b4d5c71419f285e`；终态 ERROR，未开始训练，所有活动计数为零。修复按 AST 查找唯一函数作用域，并补齐原 `yaml` 导入对应的 PyYAML 兼容依赖；GRACE 转为回归样本。

BGRL 在产品指纹 `dd6fdb07f8e3cc3645d3fa7c0210d67f3b687acd683333bd149ff61252684694` 上开始真实网页测试，标签 `author_bgrl_20261012_08`，提交文件 `progress_1791779654581.jsonl`。只上传 PDF，不手动覆盖仓库或训练设置；测试脚本的预期仓库参数仅用于事后核对。作者仓库 `nerdslab/bgrl` 核验与下载通过，固定 `60f9f19ad0598f9163ad70ebbde3e7297760901e`。规划前三次因原解释器 token `python3` 被拒绝，第四次改为 `python` 后，真实字典工厂选择的 `datasets.Coauthor` 未被识别。终态 ERROR，所有活动计数为零，未开始训练；目录为 `repository_b27b66bf20e34f6bb83b8f40514090a4`。

修复支持原 `python3`、`.cfg` 证据、限定的原始框架导入别名与字面量工厂选择、absl/TensorBoard 依赖识别。`sys.exit(0)` 可自然结束作者命令，仍必须验证实际训练更新、真实数据及进程清理；非零或字符串退出错误不生成成功回执。相关回归 116 项通过，依赖映射与数据补充回归 37 项通过；BGRL 真实原提案的结构回归通过，但没有重跑为最终验收。

GATv2 的标签为 `author_gatv2_20261012_09`，冻结产品指纹 `be1fc329388313e1061ddd095385c8f5a77bf3283b36cbb2c833482feff250f7`。网页上传预览优先选择正文断行截出的 `how_attentive_are_`，虽然同页 URI 注释完整指向 `how_attentive_are_gats`；预览前置核对失败，未提交后台任务。修复只在同页注释、相同证据类别及可见精确续行共同确认时去掉截断候选，保留没有确认的不同地址。105 项 PDF 相关回归通过，GATv2 转为回归样本，没有训练成功记录。

AirGNN《Graph Neural Networks with Adaptive Residual》从 NeurIPS 官方公开页面取得，14 页、516,244 字节，SHA-256 为 `394cb32a33e4e84e554649fc6728d918d836e02ad5dd7f218627ef235c432311`。首页原文与 URI 注释都明确指向 `https://github.com/lxiaorui/AirGNN`。冻结产品指纹 `12bfe6a1c4f6c7a0efc5cdefddbd85d7120533b0365e6198c03eda49d4f6889a`，标签 `author_airgnn_20261012_10`，进度 `progress_1791780898583.jsonl`。作者核验与源码下载通过，固定 `4178134a29f7c642e630e2bebbc6d230a472cc8f`；模型四次选外层 `prepare_data`，实际 Planetoid 构造器在 `get_planetoid_dataset`，终态 ERROR，所有活动计数为零，未开始训练。目录 `repository_f2bae40cd1b3428cbe2732c16e1868df`。先前下载的 medspaCy 错误候选不是这篇论文，仍保留在拒绝样本目录。

修复向模型列出程序实际发现的原始数据构造器文件、函数、类、源码哈希与原句；模块作用域不再包含未执行的函数体。结构反馈明确给出正确构造器作用域，不能把外层包装函数当成调用点。原仓库没有依赖声明时允许 `requirements.author=[]`，实际依赖仍从原始导入推导。补齐已遇到的 DeepRobust/OGB 依赖及匹配的 torch/torchvision 版本；未知运行兼容性仍以实际安装和执行检查。构造器与原生执行回归 38 项通过，依赖和数据回归 44 项通过，额外图依赖回归 7 项通过。AirGNN 原提案经正确函数及无虚构依赖声明修正后通过结构回归，不作为最终新 PDF 成功。

下一篇 ElasticGNN《Elastic Graph Neural Networks》从 PMLR 官方公开页面取得，13 页、1,218,486 字节，SHA-256 为 `ea07251e70cc27d74bd4320b0d5193f232bc5c0667c7f729d09afdcfbb59c7b6`。首页 URI 注释指向 `https://github.com/lxiaorui/ElasticGNN`。标签 `author_elasticgnn_20261012_11`，冻结产品指纹 `13dbf2c589ab206bfec326357a7bf3c412ee097b802295dc9a4fe13d319d6322`。真实网页任务 `progress_1791781325289.jsonl` 通过全文阅读、仓库核验、完整原命令规划与独立审查。随后固定依赖在默认镜像和官方 PyPI 两次均触发 300 秒安装超时，训练尚未开始，终态 ERROR，活动 worker、实验和执行计数均为零。不能把 pip 的宿主包冲突提示直接当成超时根因。失败缓存未写入 `.ready`。该论文转为回归样本。

后续修复对已固定 CPU torch wheel、匹配官方索引且启用自动准备的环境给予 1800 秒官方安装限时，默认镜像仍为 300 秒；私有源和普通安装策略保持原有有限预算。代码审查同时修复作者多 split 重复构造数据的最终核验：每项声明必须有训练前完整真实数据基线，后续原始构造允许发生在已有训练更新之后，所有构造记录的名称、样本数和完整文件哈希仍逐条检查。61 项针对性回归通过；这不是新 PDF 运行成功证据。

新凭据的项目真实连接测试与网页连接测试均通过，随后真实后台的全文阅读、资源审查和仓库确认也成功；凭据未写入验收文件或命令行。当前论文流程的失败属于规划/运行支持问题，不能再归因于新 API Key。运行时数据、规划与设备回归最新为 105 passed、3 skipped；运行时作者执行与服务回归为 42 passed，使用真实 CPU 依赖缓存，包含 ZIP 准备后 40 次真实优化器更新。这些工程回归不代替新论文验收。

新增的 15 份原始候选 PDF 已归档到 [候选目录](../examples/papers/candidates_20261012/README.md)，总计 29,941,970 字节，逐份核对下载清单的字节数和 SHA-256。其中 11 份原文提取到预期仓库链接；其余 4 份保留为尚未提取作者链接的候选。候选清单分别记录回归退役、鉴权失败、进行中与尚未执行状态。

### 放松审核与按报错补依赖的后续排查

AdaGNN 正文 PDF（10 页，1,256,662 字节，SHA-256 `ad4780e85914beaec0be6f7233eaa2126bbbf2160d0e18c16a8f55770ccc1ab5`）通过阅读和仓库身份核验，固定 `yushundong/AdaGNN` 的 `711a614f26e654185e38d8120678aecc3a9cb44d`，但将原仓库根目录的 `adagnn.utils` 误判为外部包，未训练。标签 `author_adagnn_20261012_12b`，源指纹 `c60f068243ce4590943411813566801cbf2fcad142f7123894ba118b6e6bb5a2`。修复只注册与真实根模块对应的仓库 namespace，保留作者源码；实际 CPU fixture 完成 40 次更新。误下载的 44 页 AdaGNN 幻灯片不列作论文。

GMNN 两版正文均没有 GitHub URL；模型已找到 `DeepGraphLearning/GMNN`，但 `author_gmnn_20261012_13b` 被 PDF 链接硬条件拒绝，没有开始训练。随后允许执行目标使用在线发现候选，将真实 README 与 PDF 联合核验；不伪造 PDF 链接或作者归属。两版原 PDF 均归档，哈希分别为 `de914534f60a62ade79541ae215adad5a3081885afbaf36e79857eaa31c2cc35`、`705fc0af7b846b12b6b19d2aece4751251ce40523125eb9f6932a3e1ae648363`。

FAGCN 标签 `author_fagcn_20261012_14`、源指纹 `675c2c4ef836cb998e5062469568b245d51da24334001d6e0e692594af210ff6`。模型选择 `alexfanjn/FAGCN_PyG`，并正确判定同论文匹配实现、作者归属未确认；论文身份和方法引文通过，但多放在 `checks` 内的结论说明触发字段集合严格相等检查。修复允许附加元数据及嵌套结论说明，仍要求实际来源引用和必需检查；未训练。

APPNP 标签 `author_appnp_20261012_15`、源指纹 `74fc1ba7f6830b9df07f4dfe37db524719b7420ab5c994f1e2d98f5abc0e8d7a`。原文给出项目主页 `kdd.in.tum.de/ppnp`，审查器两次要求它进入只接收 GitHub URL 的字段，阅读阶段停止，未训练。修复允许完整原文阅读完成后，将此类审查意见作为执行目标的参考，继续发现与核验实际 Git 仓库。数值路线及不完整阅读仍拒绝。

MVGRL 标签 `author_mvgrl_20261012_16`、源指纹 `32103abc22d266f0c6d471e52817c6731fcde5f389405da766048a9060f9f260`。核验通过的 `hengruizhang98/mvgrl` README 明确列出 `python main.py --dataname MUTAG --epochs 20`，但全文中的撇号导致整体 shell 解析失败，四次规划均拒绝 `--epochs`。改为逐条解析原始命令，并支持作者 Python argv 字面量，不因字段存在就允许模型编造短训练参数；未训练。

Graph U-Nets 标签 `author_graphunets_20261012_17`、源指纹 `087d015d5a78f7af60ea233ea1d599b371b98b2efee5009bc408d3abd2ea0a3e`。仓库身份审核返回无效 JSON，未保存原响应，也没有格式重试；未训练。修复给予一次格式重试，保留无效响应与错误，不能将语义否定改成通过；API 错误则保留实际诊断并遮蔽凭据。

SGFormer 标签 `author_sgformer_20261012_18`、源指纹 `b9505aa5f7603f145e749ca3122491008b2f3b6ccc417a148765b1604ecbefa4`。仓库确认通过，但准备阶段还留有一处旧的 `semantic_review.accepted` 硬门槛，参考审查仍阻止训练。修复让准备与执行都遵循同一执行目标规则，并加入端到端的负面审查 fixture：实际训练完成 40 次更新，审查仍记录 `accepted=false`。原论文退役，未训练。

以上论文在产品修复后均退役为回归样本，终态 ERROR，活动 worker、实验与执行进程均为零，没有改写成成功。后续换 GloGNN 新 PDF，在冻结源指纹 `ea90a7a9f75699bc3ed5784960460651f9d3fd9778dae5f23dddd5d138ec15a3` 上做真实网页测试。

GloGNN《Finding Global Homophily in Graph Neural Networks When Meeting Heterophily》从 arXiv 取得，15 页，1,129,828 字节，SHA-256 `0eba6207e064e6c87ccc1a921321cf061aa0589dc2d6e995ba73fa87a8c94b0a`。真实网页标签 `author_glognn_20261012_19`、任务 `progress_1791786472076.jsonl` 核验并固定 `RecklessRonan/GloGNN`，按原始 small-scale Chameleon split 0 命令执行。CPU torch、NumPy、SciPy、NetworkX 安装和原生检查通过，完整原始数据校验通过。实际作者入口在 `load_data_new -> normalize -> np.power(rowsum, -1)` 报 `ValueError: Integers to negative integer powers are not allowed.`，尚未发生训练更新。这不是缺少模块错误，不能靠盲目安装包宣称修复。产品源仍保持冻结，未重跑该论文追逐成功。终态 ERROR，所属 worker、实验和执行进程全部结束；此样本不算代码完成。精选机器可读记录见 [试验摘要](author_execution_trials_20261012.json)。

按错误补依赖机制还通过了独立的真实进程测试：初次 `python train.py` 确实报 `No module named 'boltons'`，系统按声明安装 `boltons==24.1.0`，保留失败 invocation，重试相同 argv 并自然成功退出。源码哈希未变；整个测试约 5 秒。这是依赖修复功能证据，不是论文复现成功证据。补装环境在数据准备与训练之间继承，任务共享三轮预算；相同包重复失败、取消、超时及普通程序错误会如实停止，不宣称所有异常都能自动修复。

### Windows 类型兼容修复与 CCA-SSG 新样本通过

GloGNN 真实 Chameleon 特征为 `uint8`，形状 `2277 × 2325`，当前 Windows 下行和为 `uint32`。同样的值改为 `uint64` 后，NumPy 1.26.4 自动提升为浮点并正常求倒数。作者声明的 [SciPy 1.5.4 get_sum_dtype](https://github.com/scipy/scipy/blob/v1.5.4/scipy/sparse/sputils.py) 同样使用平台相关的 `np.uint`；[NumPy 1.19.2 文档](https://github.com/numpy/numpy/blob/v1.19.2/numpy/core/code_generators/ufunc_docstrings.py) 也明确限制整数负幂。这说明该源码未显式指定浮点归一化，会受平台整数类型行为影响；不能据此否定论文方法，也未声称完整安装旧依赖实际跑过。

显式作者执行目标现允许在真实错误、固定入口、唯一倒数调用、训练更新尚为零、整数值可被 float64 精确表示的条件下，记录源码 SHA、位置、原表达式及失败 invocation，在下一轮执行中把该调用编译为 `numpy.float_power`。作者仓库文件、其他 AST 节点、原始 argv、训练参数及输入数据保持不变；报告明确标注兼容修复。补依赖与类型修复共享全任务三轮预算。真实服务回归先报原错误，记录修复后完成 40 次优化更新，固定源码哈希及适配器清单再次通过；该 fixture 不是论文验收。GloGNN 自产品修复后退役，没有重跑它追逐最终成功。

修复后更换全新 CCA-SSG PDF，冻结产品指纹 `1b6f1a3af95f8136b6f0aa382b3feb4c73bee239ce0d4c70a064a32456f29879`。真实网页上传、全文阅读、作者仓库核验、兼容依赖准备及原始 Cora 命令均通过，完成 50 轮训练和 2000 轮线性评估，共 **2050 次优化更新**，正常退出码 0。未补包重试、未使用类型修复、未改写作者源码或缩短原始命令。网页完成时所属 worker、实验及执行进程全部为零，产品与仓库源码哈希再次核对通过。作者程序报告准确率 0.8170；完整数据来源及论文数值尚未独立核验。详见 [验收说明](ccassg_author_execution_20261012.md)及[机器记录](ccassg_author_execution_20261012.json)。本轮候选归档包含 23 份 PDF，大小与 SHA-256 均逐份核对。

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
