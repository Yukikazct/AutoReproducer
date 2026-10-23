# 显式仓库兼容修复

普通真实 PDF 的网页入口默认启用“兼容性源码修复：失败后自动重试”，可手动关闭。文件列表留空时，真实执行失败后按 traceback 自动定位最多三个固定作者源码文件；也可填写精确 Python 文件路径来限定修复范围。标题、Mock、语料对照、外部代码和固定预设不使用此入口。直接调用服务的默认值仍为关闭。

后台线程与独立 worker 传递 `source_repair`、`source_repair_auto` 和 `source_repair_editable`，由 `src/source_repair.py` 衔接作者执行、事务补丁与结果核对。源码修复与 `strict_validation` 互斥：网页切换选项时关闭另一项，服务请求同时开启时拒绝执行。

## 白名单与独立副本

自动模式只接受实际末次异常中的缺失属性或意外关键字参数，从 traceback 中选择包含该 API 名称的 Python 源码。每个文件必须属于冻结源码、哈希未变且通过保护策略；选择依据和拒绝原因写入 `auto_selection`。无法定位受支持问题时记录 `not_applicable`，不调用补丁模型。自动模式不会因此获得修改整个仓库的权限。

白名单必须列出 1–32 个固定作者源码中已存在的 `.py` 文件，使用精确的仓库相对路径；不接受目录、通配符、重复路径或仓库外路径。数据、评测、划分、权重及其他受保护文件不能因进入白名单而获得修改权限。每轮候选仍最多修改三个文件，不支持新增、删除或移动文件。

只有原命令实际执行并失败时才请求源码补丁。安装失败、未运行、正常退出、取消、超时和 Notebook 计划不触发此修复。服务保留原失败结果，从固定作者源码及已冻结输入建立独立的 `repair_workspace`，不复制失败运行的临时产物。原作者工作区、来源证据、原冻结计划及观测器保持独立；共享缓存和用户仓库不作为修复目标。

一次任务只进入一次源码修复会话，最多两轮；拒绝的提案也消耗一轮。空补丁表示当前约束下无法修复，会提前停止。已发生的原失败通过 `initial_execution` 传入修复器，不为取得相同报错再执行一遍。候选从头重跑全部原作者命令，保留原 argv、cwd、依赖顺序和步骤环境，使用实际已准备的解释器及依赖环境；模型不能修改命令、依赖或环境。

修复副本不插入严格训练观测器，也不从命令正常退出推断已经观测到训练更新。

README 环境中指向原工作区的绝对路径（例如 `PYTHONPATH`、数据根目录）会等价重定位到修复副本，并写入 `environment_relocations`；不会让候选继续导入原副本模块或写入原输入。无法隔离的外部绝对路径会拒绝。原失败运行尚无完整冻结凭据的 `author_runtime` 动态数据暂不进入源码修复，保留原失败和 `not_applicable` 原因，不在副本中静默重新下载。

## 候选、重跑和恢复

模型获得本次实际错误、原命令和有界的真实源码上下文。响应包含修复理由与一至三个完整 UTF-8 文件替换，每个文件必须携带其 `path`、精确 `base_sha256` 和 `content`。未提供给本次模型的文件不能修改。

每个候选的全部文件先统一检查路径与链接、保护规则、重复目标、文件大小、原文件摘要和 Python 语法。执行前还拒绝已有代码检查器能够识别的命令执行、动态执行和网络外联等危险构造。该检查器提供确定性拦截，不构成操作系统沙箱。

网页服务在保存提案和 diff 后、应用补丁前继续核对 AST：变量引用、字面量、运算、控制流、import、函数参数和训练/划分/评测敏感名称必须保留，只允许同结构的 API、方法或关键字名称兼容修改。被替换的原名称还必须出现在本次会话实际观测的缺失属性或意外关键字参数诊断中；第二轮累积原失败及最新失败的依据，因为回滚后仍需重新应用首轮修复。结构或报错依据不符时记录 `rejected`，不应用、不执行该候选。这是范围受限的兼容规则，不能证明任意程序的语义等价。

副本创建前与事务快照前通过 `stat()` 检查工作区规模，上限为 **4096 个文件、256 MiB**。超过上限记录 `resource_limit` 并保留诊断；复制前发现超限时不读取文件正文、不复制、不请求补丁。修复器也在源码上下文和快照边界复查预算。快照保存在内存中，因此大型数据、模型或缓存超限时不能进入此路径；正常成功的原执行不受修复快照预算影响。

检查通过后保存整个工作区快照，再统一应用候选，并清除受影响模块的旧字节码。候选失败、取消、执行器异常或应用中途出错时恢复快照，包括其他被运行过程改动或删除的文件，并删除新产物。执行成功后还会核对受保护输入、原 PDF 和冻结计划、实际源码与批准补丁的精确摘要，并重新检查兼容结构；校验异常同样回滚。恢复失败立即停止并标记 `rollback_failed`，不继续下一候选。

## 证据与结果边界

原运行保存在 `original_execution.json`；`source_repair.json` 记录授权白名单、最多轮数、原工作区、修复副本、尝试与诊断。修复计划单独保存为 `source_repair_execution_plan.json`。每轮脱敏理由、提案、diff、完整命令日志位置、执行结果和回滚记录保存在工作区外的 `.autorepro_repository_repairs/<id>/`，摘要为 `repairs.json` 和 `repair_report.md`；失败候选日志不会被回滚删除。内部修复记录使用 Windows extended-length 路径并拒绝 symlink / junction。

成功候选额外保存 `source_repair_manifest.json`，绑定原固定源码、实际源码、冻结输入、实验摘要、独立修复计划和逐文件 `path`、`base_sha256`、`sha256`。后续完成核对会重新检查该清单摘要、实际文件和完整原命令的退出结果。API key 从提案和报告中脱敏，检测到敏感内容的源码不发送给模型。

服务还尝试生成 `source_repair_evidence.zip`，汇集上述记录、原模型响应和原运行、各轮运行的完整文本日志；附件内 `evidence_manifest.json` 记录每个成员的原件摘要、归档摘要、字节数与是否脱敏。归档输入上限为 64 MiB，失败记录 `evidence_error`，完整原件继续保留在运行目录。报告旁的证据附件和网页报告下载核对附件 SHA256，下载包包含已绑定的 ZIP；不能仅靠报告中的任意本地链接加入文件。复制准备期间的失败或取消也写入终态；附件通过原子替换写入，避免沿已有硬链接改写外部文件。

命令完成时使用 `execution_source=patched_author_repository`、`reproduction_scope=patched_author_code_execution` 和 `source_modified=true` 标明补丁来源；最多给出 `author_code_completed`。此路径保留 `paper_metrics_verified=false`、`is_reproduced=null`、`training_observed=false`，不计作原始源码严格验收或独立数值复现。后续论文参考比较使用修复副本及实际源码摘要，并以 `basis=patched_author_output` 记录依据和修复清单摘要。

最终冻结源码专项 **248 项通过，零失败、零跳过**，包含 Streamlit 控件及实际后台→服务→所属子进程→报告附件的合成机制测试，详见 [来源绑定记录](evidence/issue20_web_repairs_20261021.json)。模型回复及网络来源为 fixture；真实模型 API / 外部论文的完整网页验收需另行记录，机制测试不构成 #41 或论文数值通过证据。

## 底层接口

`RepositoryRunner.run_with_repairs()` 也可由程序显式调用。调用方必须先导出独立、可丢弃的工作区，并明确列出可修改源码；带 `.git` 的目录、共享源码缓存、应用目录和用户原始仓库不能作为候选工作区。开始、应用和回滚时，工作区所属进程必须已结束。

```python
from src.llm.llm_client import LLMClient
from src.repository_runner import RepositoryRunner

runner = RepositoryRunner()
result = runner.run_with_repairs(
    workspace,
    original_steps,
    original_environment,
    llm=LLMClient(),  # 从已有环境配置读取 API 设置
    editable=["models/network.py", "train.py"],
    max_rounds=2,
)
```

可选 `initial_execution` 复用已发生的真实失败；`validate_patches(workspace, files, failed_execution)` 在应用前校验整组候选，`verify_candidate(workspace, files, candidate_execution)` 在候选成功且受保护输入检查通过后、提交前校验。两个 hook 接收补丁记录的深拷贝。AST 与实际缺失 API 校验由网页服务通过前验 hook 接入；底层 API 调用方应自行提供适合其固定协议的校验，不能把未设置 hook 的普通成功当作语义或论文数值保证。
