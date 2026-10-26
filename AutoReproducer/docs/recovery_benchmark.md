# 智能恢复的任务级评测

`src.benchmark.recovery` 比较冻结的当前流水线 A、增强单智能体 B、增强协作 C。A 保持冻结版本原有策略，不施加新增模型预算；B/C 使用相同增强产品版本、模型、初始证据、工具及配置。模式差异只有角色组织与协作。

本模块提供注册校验、冻结、隔离试验回调和结果汇总。它没有内置 12 个真实新案例，也没有运行 3 篇新论文。通过本模块单元测试不表示完成这些实测。

## 注册与冻结

Manifest 使用 `version: 1`、`repetitions: 3`，包含以下字段：

| 字段 | 要求 |
| --- | --- |
| `arms` | A/B/C 分别登记 `product_revision`（40 位 Git SHA）及 `product_source_sha256`；B/C 相同 |
| `model` | 固定实际模型标识；不存 API key |
| `tools` | `source_context`、`runtime_metadata`、`artifact_contract`、`author_contract` 的固定子集 |
| `budget` | `max_model_requests`、`max_tokens`、`max_probes`、`max_source_repair_rounds`、`max_repair_actions` |
| `cases` | 下方的预登记案例列表 |

默认 `max_model_requests: null`、`max_tokens: null`，允许根据实际需要调用。也可显式设置正整数上限。探针/源码补丁/修复动作上限分别为 4/2/8，不能在角色切换或新错误时重置。模型上下文容量另行配置，不能把 1M 上下文当作累计用量或价格。

每个案例要求：

```json
{
  "id": "example-blocker",
  "repository_url": "https://github.com/owner/repository",
  "repository_revision": "<40位真实固定版本>",
  "source_files": {"train.py": "<文件真实SHA-256>"},
  "split": "holdout",
  "origin": "natural",
  "category": "runtime_environment",
  "unused_for_tuning": true,
  "goal": "提前固定的完成目标",
  "oracle": {
    "kind": "author_contract",
    "reference": "可信评分器读取的原有测试或参考记录位置",
    "sha256": "<该评分依据的真实SHA-256>"
  },
  "initial_evidence": {"diagnostic": "原始阻塞及公开初始证据"},
  "initial_evidence_sha256": "<initial_evidence规范JSON摘要>"
}
```

这只是结构示例，尖括号值不能通过校验。摘要由 `src.holdout_registry.digest` 计算。`source_files` 登记每次试验所需的完整初始文件集合；试验只复制这些文件，不把上次试验产物混入新工作区。调用者应从登记的作者版本准备来源目录，并核对登记文件摘要。Oracle 位置和原始评分内容不传给智能体。

`split` 仅允许 `dev`、`holdout`；同仓库的大小写和 `.git` URL 别名不能跨两组。历史开发样本登记为 dev。`origin` 仅允许 `natural`、`injected`；故障注入单独统计，不充当真实新案例。`category` 预登记为 `runtime_environment`、`local_import`、`api_compatibility`、`result_evidence` 之一，正式案例集必须覆盖四类。未参与调试声明和源码摘要是可审计身份约束，不能凭声明证明实际未泄漏；案例选择与使用历史仍需真实保存。

```powershell
python scripts/evaluate_recovery.py validate --manifest recovery_manifest.json
python scripts/evaluate_recovery.py freeze --manifest recovery_manifest.json --output recovery_frozen.json
python scripts/evaluate_recovery.py aggregate --manifest recovery_frozen.json --results trials.jsonl --output recovery_summary.json
python scripts/evaluate_recovery.py run --manifest recovery_frozen.json --source-roots recovery_sources.json --product-roots recovery_products.json --adapter my_adapter:run_trial --oracle my_adapter:evaluate_trial --output-dir recovery_trials
```

冻结/汇总输出文件必须不存在。`validate` 不训练、不调用模型、不执行验收。冻结后不得修改目标、Oracle、案例、模型或配置再把原试验混入新结论。修改产品后应重新冻结；用于修复产品的案例转开发回归，补充新的 holdout。

## 试验执行接口

```python
run_benchmark(
    frozen_manifest,
    source_roots={"example-blocker": prepared_author_source},
    output_dir=new_output_directory,
    trial_runner=trusted_trial_runner,
    oracle_evaluator=trusted_oracle_evaluator,
    product_roots={"A": frozen_baseline_app, "B": frozen_enhanced_app, "C": frozen_enhanced_app},
)
```

Runner 的关键字参数为 `case, arm, repeat, workspace, environment, context, configuration`。每次获得新的目录和空上下文；公开案例不含 oracle。B/C 配置相同，仅 `mode` 为 `single` 或 `collaborative`。A 的 `mode` 是 `baseline`、`budget_policy` 为 `frozen_baseline`，应由可信适配器调用冻结的实际旧策略。回调负责加载配置指定的产品版本，使用受管进程执行，不能复用前次依赖根、模型对话或修复结果。评测器不会执行模型生成的 Shell 指令。

`recovery_sources.json` 是 case id 到已准备作者来源目录的映射；`recovery_products.json` 是 A/B/C 到应用根目录（含 `app.py`）的映射。`run` CLI 只导入操作者显式提供的可信 `module:function`，不从模型文本生成插件名。适配器需通过正常 Python 导入路径提供；它可以调用固定旧版本子进程或当前增强入口，评分器应独立核查其产物。

`verify_product_identity` 使用现有 holdout 的 `product_hashes` 和实际 `git rev-parse HEAD`，执行前及每次试验前后重新验证来源。输出目录必须位于产品 checkout 之外。Python API 可省略 `product_roots` 以运行开发夹具，但该记录标为未验证，不能满足正式验收；CLI `run` 要求显式提供两个冻结版本的实际目录。

每个案例重复 3 次；每轮三组顺序由冻结摘要确定性打乱。输出逐次追加 `trials.jsonl`，中断仍保留已完成记录。取消、超时或清理未确认时，保存当前失败及不完整摘要后停止后续调度；回调异常也不能被当作已确认清理。来源哈希在执行前后检查；作者来源目录不能作为输出目录，也不应由回调修改。

Runner 返回公开结果及计量：

```python
{
    "claimed_success": False,
    "human_interventions": 0,
    "blocker": "execution_failed",
    "usage": {
        "model_requests": 5, "tokens_charged": 30000,
        "probes": 2, "source_repair_rounds": 1, "repair_actions": 1,
        "prompt_tokens": None, "completion_tokens": None, "invalid_retries": 1, "cost": None,
        "execution_retries": 2, "same_failure_retries": 1,
    },
}
```

实际模型用量缺失保持 `null`，不把未知费用当零。`tokens_charged` 可以来自控制器的保守预算计量，与供应商实际用量分开。显式设置上限时，未知或超过限制的计量使正式评测不完整。

`invalid_retries` 记录模型格式纠错请求；`execution_retries` 记录首个实际作者执行之后的重跑；`same_failure_retries` 记录相邻实际执行在相同步骤、源码位置和终末诊断处再次失败。环境准备拒绝、未启动作者进程的调用不计重跑。修复副本的绝对根路径不产生新错误身份。正在执行的调用被外部中断、没有完整结果时，后两项保持未知；历史回执没有这些字段时也不补零。汇总拒绝同错误次数超过实际重跑次数、格式纠错次数超过模型请求数的回执。

Oracle 回调独立获得 `case, result, workspace, environment`，必须返回显式布尔值 `oracle_passed, protocol_preserved, cleanup_confirmed`，可增加可审计的评分摘要。它负责检查真实执行、冻结设置和进程清理，不能把模型自述成功当作评分。原始评分摘要、摘要哈希与 `trial_id` 绑定保存到每条记录；不要返回隐藏标签、凭据或其他不应出现在报告的私有内容。

Oracle 通过、协议保持、清理确认且没有人工介入，才算自主完成。不能完成或静默失败均算需要接管。评分器异常时保留已有成功宣称和用量，不擦除不受证据支持的成功信号；该记录不能完成验收。自动阶段不为降低询问次数而隐藏阻塞。评分人员事后核验属于评测成本，不能计为试验内智能体自主能力。

## 可执行默认适配器

`src.benchmark.recovery_adapter.RecoveryAdapter` 已将上述接口接到真实产品代码，提供四类预登记路径：Python API 兼容、本地导入、受限的缺失依赖恢复和结果产物调查。每类都有下述可执行边界，超出边界不会由模型扩大权限。它比较固定计划下的作者恢复组件，**不代表完整 PDF 发现与规划流程的端到端 A/B/C 评测**；支持四类路径不等于完成自然案例覆盖。

A 在独立受管子进程中导入真实旧 checkout 的 `RepositoryRunner`、`analyze_author_failure` 和 `source_repair.repair_author_failure`。B/C 导入增强 checkout 的同一生产恢复链，再分别使用 single/collaborative `RecoverySession`。加载来源检查拒绝任何来自其他 checkout 的 `src.*` 模块。每次创建新作者 venv、数据根和依赖根，清除宿主 `PYTHONPATH`、用户 site 和提供商凭据；API key 只经环境读取或内存参数进入 stdin 管道。父进程确认 worker 及其后代清理后，才进入独立评分。作者完整命令不附加 60 秒上限；venv 准备与独立断言各最多 60 秒。

适配器复用生产的 traceback 自动圈定、兼容 AST 约束、固定输入/解释器/原始命令复查及失败回滚。新错误不创建新会话。B/C 的固定工具子集也按同一配置限制。普通作者命令先执行；无需前置模型批准。`adapter_receipt.json` 记录产品来源、实际调用计量、清理及 driver 摘要，不包含 API key 或私有 Oracle。

可在 Python 中创建 `RecoveryAdapter(product_roots=..., oracle_paths=..., product_python=..., author_python=..., llm_config=...)`，把 `adapter.run_trial` 和 `adapter.evaluate_trial` 传给 `run_benchmark`。两个 Python 路径分别指向具备应用依赖的协调解释器和创建独立作者 venv 的基础解释器；不会复用后者的 site-packages。依赖清单首版仅接受公开包的精确 `==` 版本，不接受 URL、索引参数、markers 或任意安装命令。

CLI 也可直接使用默认回调。操作者准备 `adapter.json`：

```json
{
  "product_roots": {"A": "<旧应用根>", "B": "<增强应用根>", "C": "<增强应用根>"},
  "oracle_paths": {"example-blocker": "<私有评分JSON绝对路径>"},
  "product_python": "<应用Python绝对路径>",
  "author_python": "<作者基础Python绝对路径>",
  "llm_config": {"base_url": "https://api.deepseek.com", "timeout": 120}
}
```

配置文件不得写入 `api_key`，密钥使用当前进程的 `LLM_API_KEY`。模型标识来自冻结 manifest。随后执行：

```powershell
$env:AUTOREPRO_RECOVERY_ADAPTER_CONFIG = 'C:\trusted\adapter.json'
python scripts/evaluate_recovery.py run --manifest recovery_frozen.json --source-roots recovery_sources.json --product-roots recovery_products.json --adapter src.benchmark.recovery_adapter:run_trial --oracle src.benchmark.recovery_adapter:evaluate_trial --output-dir recovery_trials
```

适配器与固定 bootstrap 在 factory 创建时计算摘要，每次启动前后复查。增强源码的产品摘要包含适配器行为实现；固定 bootstrap 只加载这份源码。运行期间修改适配器或 bootstrap 会停止执行。默认适配器的 holdout 执行要求将 `adapter.driver_sha256` 写入 manifest 顶层 `adapter_driver_sha256` 后再冻结；配置、逐条 `execution_metadata` 和离线聚合都会核验该身份，不允许混合不同 driver 的回执。两份源码摘要相同的 checkout 不能分别冒充 A/B。

可执行案例的 `initial_evidence` 需要额外提供：

```json
{
  "adapter_version": 1,
  "plan": {
    "repository": {"url": "https://github.com/owner/repository", "revision": "<与案例相同的SHA>"},
    "steps": [{"id": "train", "kind": "train", "argv": ["python", "train.py"], "cwd": ".", "timeout_s": null}],
    "requirements": {"author": [], "compatibility": []}
  },
  "packet": {"file_inventory": {"train.py": {"sha256": "<固定文件摘要>", "text_available": true}}},
  "environment": {"requirements_txt": ""},
  "editable": ["train.py"],
  "initial_failure": {"error_type": "AttributeError", "diagnostic_contains": "missing_api"}
}
```

`packet` 必须准确对应案例完整源码清单。`editable` 仅是外层上限；实际修改范围仍由当前真实 traceback 派生。每轮先重新执行原命令，不能把事先填写的日志冒充当前失败。脚本 argv 仅允许 `python`/`python3` 加仓库内固定 `.py` 路径及原始参数，禁止 Shell、`-c`、`-m` 和案例自带环境覆盖。

独立评分 JSON 的完整字节摘要登记在 `case.oracle.sha256`。Oracle 及其路径不传给模型 worker。支持两种固定方式：

- `oracle.kind: trusted_stdout_v1`：私有 JSON 包含 `{"version":1,"stdout_sha256":"<可信原始参考输出摘要>"}`；与实际末步作者 stdout 的 UTF-8 字节摘要比较，预先固定换行与统计口径。
- `oracle.kind: original_assertions_v1`：私有 JSON 包含 `{"version":1,"scripts":[{"path":"original_test.py","sha256":"<原始测试摘要>"}]}`。路径相对评分 JSON 所在目录，不能来自候选目录。原始测试在模型 worker 退出后，使用本轮作者 venv 在单独受管子进程执行；断言优化被禁用。该测试的依赖必须在原作者环境中声明。

预登记参考必须来源于作者原有测试或可信参考输出。适配器不生成 expected，也不将候选自己生成的测试当作独立验证。API 拼写补丁运行通过，只能证明登记契约与原命令恢复；数值行为等价仍保持 `unknown`。

`local_import` 案例使用同样的冻结结构，`initial_failure.error_type` 为 `ModuleNotFoundError`，`diagnostic_contains` 登记实际缺失模块，`editable` 必须为空。各组调用自己 checkout 内的 `run_with_dependency_repair` 与 `repair_local_import`：仅在真实终末 traceback、导入语句、唯一冻结模块以及包边界均匹配时，恢复仓库内原有导入根并重跑完全相同的命令。不会安装同名外部包、替换解释器、修改源码或放宽原实验设置。恢复器请求及实际运行后的完整环境控制均需与初始环境一致；只允许原有依赖安装/缓存遥测变化和已证明的 `repository_pythonpath`。

同名模块有歧义或缺少来源时，增强组可以使用原固定检查工具补查，然后停止或交给已有分析者；模型不能自行挑选导入根。B/C 的检查、角色与后续错误共享原会话和动作额度。每个已恢复根在回执中保留真实失败与源码摘要，评分器会独立重新推导根列表；原始断言也在相同、经过重新核验的导入路径中运行。最简单的唯一根案例复用 A/B/C 已有的确定性能力，三组都恢复并不能证明 AI 增益。

### 依赖环境恢复

`runtime_environment` 在原命令实际产生顶层 `ModuleNotFoundError` 后，调用各自冻结产品的依赖恢复循环。`initial_evidence.runtime_environment` 为 `version: 1`，其中 `allowed_repairs` 提前列出缺失 `module`、精确 `requirement` 和完整精确版本 `additions`。每次依赖增加必须由真实 traceback、冻结作者源码或依赖声明支持，最终依赖集合由独立评分器重新核对。版本替换、原生兼容和解释器切换尚不支持，不能用这一条路径宣称覆盖这些能力。

自然案例仅接受公开包的精确版本；`development_wheels` 只供 `dev/injected` 的离线测试使用，必须作为冻结来源文件登记路径与摘要。wheel 元数据、安装文件及字节由评分器核验，不允许拿测试 wheel 冒充自然依赖。环境恢复不修改源码、原命令、数据或实验设置。

### 结果产物调查

`result_evidence` 要求作者原命令已经完整成功，再调查实际产生的单个、不超过 2 MiB 的平铺 JSON 指标摘要。`editable` 必须为空。`initial_evidence.result_contract` 显式登记 `version: 1`、`artifact`、`format: metric_summary_json`、`reference`、`relative_tolerance` 和 `absolute_tolerance`；未知值保留 null，已知字段不能与冻结实验协议冲突。

`reference` 可包括 `metric/value/unit/method/dataset/split/aggregation/runs`。增强组必须选择实际检查工具并在结束决策中引用对应产物证据，工具只公开聚合指标及实验元数据。评分器在模型 worker 退出后独立重读原始产物，核对实际判断、原始字节、实验对应、单位、统计方式及显式容差；不执行隐含单位换算。私有 `oracle.kind: result_evidence_v1` 登记预期判断与必须发现的问题码；接受未知结论还必须显式登记 `allow_unknown` 和完整的 `expected_unknown_fields`。

本路径的“完成”表示正确处理预登记的结果疑点；发现数值差距或无效证据也可以是正确处理，不能据此称为论文数值达标。缺参考时只有保持未知的结论可能通过对应 Oracle。A 保留旧产品已有的、证据完整的比较能力；普通诊断建议不自动成为成功。B/C 和 A 均不修改产物、实验协议、训练参数或验收容差。

### 调查计量

增强组的 `adapter_receipt.json` 另有 `usage.roles` 和 `collaboration`：仅保留各任务实际请求/用量、审查阶段/角色/候选摘要/动作/证据编号/安全原因，以及审查、补查和不同补丁否决数。它们不含 prompt 或补丁正文，不增加模型调用，也不能替代独立成功 Oracle；外部取消时同样进入受身份绑定的计量 checkpoint。

每次模型请求前后写入仅含计量的 `recovery_meter.json`，绑定随机试验 nonce 与配置摘要；探针和补丁状态变化也更新它。外部取消/超时杀死 worker 后，父进程只在清理确认后回收匹配的 checkpoint。进行中的请求不虚构 token，用量保持未知。Oracle 取消、超时、基础设施异常都停止后续调度；正常原始断言失败作为候选失败保留。

`development_responses` 仅是受信测试配置，可提供按任务分组的固定 JSON 回复；factory 硬性禁止在 natural 或 holdout 案例使用它。开发演示的实际模型 token、费用为未知，不把 scripted 计量当作真实模型收益。

2026-10-26 开发验证：适配器、benchmark 与 holdout registry 联合测试为 **94 passed**。其中一个明确登记为 `dev/injected` 的 API 拼写案例经默认 CLI 完成 A/B/C 各 3 次隔离执行：三组均 3/3 恢复、错误成功判定为 0，固定回复请求总数分别为 6/15/15。A 实际加载 `e0fb5a8` 旧源码，B/C 为增强源码快照。该样例只验证执行与评分接线，未观察到恢复率差异，不构成模型或协作收益证据；12 个真实新阻塞案例与 3 篇完整论文验收仍未完成。

## 汇总和接受门槛

`aggregate_results(frozen_manifest, receipts)` 检查输入身份、配置、Oracle 评分记录、独立工作区/环境/上下文身份和重复记录。哈希约束绑定身份，不是评分器签名；离线汇总仍依赖可信评分器产出的真实记录，不能凭手填布尔值证明复现。

只有自然 holdout 案例至少 12 个、至少 6 个仓库、全部 A/B/C 各 3 次试验齐全、实际产品版本经过核对、无取消/超时/未清理记录，且 B/C 限额计量通过，`evaluation_complete` 才为真。缺失试验保留分母并列出，不允许删除失败试验后计算成绩。开发样本和注入样本另列。

每个案例至少 2/3 次自主完成才算该组解决此案例；不能把 108 次运行视为 108 个独立样本，同仓库案例也相关。验收目标为 C 相比 A 至少净增 2 个案例、人工接管需求相对降低 30%，错误成功判定和协议违规均为零。A 没有接管需求时，降低比例不可计算，不能据此声称下降。

C−A 与 C−B 分别汇总。C 不优于 B 时，建议候选为增强单智能体；即使 C 多解决一个案例，也只描述样本内观察，不证明稳定的总体协作收益。案例评测结果不等于论文总体复现率，3 篇新论文完整选定实验及独立复算仍需另行验收。
