"""Orchestrator - 编排器核心，管理 Agent 的状态机流转。

职责（对齐方案 4.1 / 4.4）：
1. 任务分解与流程控制：INIT -> READ_PAPER -> FIND_RESOURCES -> BUILD_ENV
   -> EXECUTE_CODE -> VALIDATE -> (OPTIMIZING -> OPTIMIZED) -> GENERATE_REPORT
   -> COMPLETED / ERROR；
2. Prompt-Free 验证闭环：每步输出经 Verifier 校验，未通过时按修正建议
   触发一次修正重试（预算内），形成「生成->验证->修正->再验证」；
3. 预算统计：汇总 LLM 调用次数，纳入审计统计与报告。
"""
from typing import Any, Callable, Dict, Optional
from pathlib import Path
from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.agents.paper_reader import PaperReaderAgent
from src.agents.resource_finder import ResourceFinderAgent
from src.agents.env_builder import EnvBuilderAgent
from src.agents.execution_planner import ExecutionPlannerAgent
from src.agents.code_executor import CodeExecutorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.verifier import VerifierAgent
from src.agents.optimizer import OptimizerAgent
from src.optimizer.real_simulator import RealSimulator
from src.resource_manager import ResourceManager
from src.experiment_profiles import resolve_profile, stage_result, PROFILES

# 每步验证失败时最多触发的修正重试次数（预算约束）
MAX_FIX_RETRIES = 1

# EnvBuilder 真实构建时使用的镜像名；成功构建后通过 env_config.image_tag
# 透传给 CodeExecutor，让复现代码在本镜像内运行（含论文依赖）
DEFAULT_IMAGE_TAG = "autorepro-env"

# FSM 状态 -> UI 展示名（progress_cb 通知用；空串表示不归属某个 Agent 卡片）
STAGE_DISPLAY = {
    "READ_PAPER": "PaperReader",
    "FIND_RESOURCES": "ResourceFinder",
    "BUILD_ENV": "EnvBuilder",
    "PLAN_EXECUTION": "ExecutionPlanner",
    "EXECUTE_CODE": "CodeExecutor",
    "VALIDATE": "ResultValidator",
    "OPTIMIZING": "Optimizer",
    "OPTIMIZED": "Optimizer",
    "GENERATE_REPORT": "ReportGenerator",
    "COMPLETED": "",
    "ERROR": "",
}


class Orchestrator:
    """编排器 - 管理复现->验证->优化->报告 流水线的状态机流转。"""

    # 状态定义
    STATES = [
        "INIT", "READ_PAPER", "FIND_RESOURCES", "BUILD_ENV",
        "PLAN_EXECUTION", "EXECUTE_CODE", "VALIDATE", "OPTIMIZING",
        "OPTIMIZED", "GENERATE_REPORT", "COMPLETED", "ERROR",
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 mock_mode: bool = True, logger: Optional[AuditLogger] = None,
                 max_trials: int = 10, use_docker: bool = False,
                 workspace_dir: Optional[str] = None,
                 resource_manager: Optional[ResourceManager] = None,
                 progress_cb: Optional[Callable[[str, str, str], None]] = None):
        self.state = "INIT"
        self.logger = logger or AuditLogger()
        # UI 进度回调 (state_name, display_name, status)；None 时静默跳过。
        # 使 UI 路径与程序化路径共用同一 FSM 驱动器，消除双驱动器漂移
        # （此前 backend_pipeline 自持一份流水线循环，漏掉了资源拉取、
        # 镜像构建与验证修正闭环）。
        self.progress_cb = progress_cb
        self.llm = llm_client or LLMClient(mock_mode=mock_mode)
        # P1-⑫ 用量计量：把 LLM 调用的 token/耗时通过 hook 归入当前 plan
        # （AuditLogger 按流水线阶段的 begin_plan/end_plan 界定 plan 边界）
        # 兼容外部注入的 mock/脚本 LLM（无 usage_hook 属性时静默跳过）。
        if hasattr(self.llm, "usage_hook"):
            self.llm.usage_hook = self.logger.record_llm_usage
        self.max_trials = max_trials
        self.mock_mode = mock_mode
        self.use_docker = use_docker and not mock_mode
        # 优化工作区:提供时启用 Optimizer 真实执行闭环(补丁 -> 白名单 ->
        # 快照 -> 重跑 -> 真实指标 -> Keep/Reject);缺省仅 Mock 使用模拟。
        self.workspace_dir = workspace_dir
        # L0 热缓存资源管理（三层存储）：FIND_RESOURCES 后按需懒加载，
        # COMPLETED 前落盘 manifest 与统计；可注入以隔离数据根（测试）。
        self.resource_manager = resource_manager or ResourceManager(
            logger=self.logger, mock_mode=mock_mode)
        self.resource_manager.mock_mode = mock_mode

        # 初始化所有 Agent
        self.agents: Dict[str, Any] = {
            "reader": PaperReaderAgent(self.llm, self.logger),
            "finder": ResourceFinderAgent(self.llm, self.logger,
                                          offline=mock_mode),
            "builder": EnvBuilderAgent(self.llm, self.logger),
            "planner": ExecutionPlannerAgent(self.llm, self.logger),
            "executor": CodeExecutorAgent(self.llm, self.logger,
                                          use_docker=self.use_docker,
                                          mock_mode=mock_mode),
            "validator": ResultValidatorAgent(self.llm, self.logger),
            "verifier": VerifierAgent(self.llm, self.logger),
            "optimizer": OptimizerAgent(self.llm, self.logger,
                                        max_trials=self.max_trials),
            "reporter": ReportGeneratorAgent(self.logger),
        }
        # 真实优化闭环:注入真实执行器(替代默认哈希模拟)
        if self.workspace_dir:
            self.agents["optimizer"].simulator = RealSimulator(
                llm=self.llm, executor=self.agents["executor"],
                workspace_dir=self.workspace_dir, logger=self.logger)

        self.data: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def run(self, input_data: dict) -> dict:
        """执行完整的复现流程（复现 -> 验证 -> 优化 -> 报告）。

        input_data 支持:
          - "paper_title": 论文标题（字符串输入方式）
          - "pdf_path": 论文 PDF 路径（上传/本地文件）
          - "code": 可选，外部提供的真实复现代码
          - "corpus_paper": 可选，PaperGuru-Benchmark 论文 id（语料对照层）
        """
        # Explicit PDF/code inputs retain the generic pipeline. Only title-only
        # input automatically selects a demo adapter.
        auto_title = (input_data.get("paper_title") or "") if not (
            input_data.get("pdf_path") or input_data.get("code")) else ""
        profile = ("" if input_data.get("use_llm_pipeline") else
                   resolve_profile(auto_title,
                                   input_data.get("experiment_profile") or ""))
        input_data = dict(input_data)
        if profile:
            input_data["paper_title"] = PROFILES[profile]["title"]
        self.logger.log("Orchestrator", "start_pipeline", "START",
                        "开始自动复现流水线", input_data)

        # 透传用户输入（修复：此前 paper_title/pdf_path 未进入数据上下文）
        self.data = {
            "paper_title": input_data.get("paper_title", "") or "",
            "pdf_path": input_data.get("pdf_path", "") or "",
            "code": input_data.get("code", "") or "",
            "corpus_paper": input_data.get("corpus_paper"),
            # 用户显式指定仓库 URL：FIND_RESOURCES 里最高优先（发现链第一层）
            "preferred_repo_url": input_data.get("preferred_repo_url", "") or "",
            "repro_mode": input_data.get("repro_mode", "") or "",
            "experiment_profile": profile,
            "mock_mode": self.mock_mode,
            "verifications": [],
            "fix_records": [],
        }

        # 论文稳定 ID（三层存储索引）：corpus 语料键 / sha1(title) 前 12 位
        paper_id = self.resource_manager.paper_id_for(
            self.data.get("paper_title", ""),
            corpus_key=self.data.get("corpus_paper") or "")
        if profile:
            paper_id = profile
        self.data["paper_id"] = paper_id
        self.data["storage"] = {"paper_id": paper_id,
                                "repro_level": input_data.get(
                                    "repro_level", "smoke") or "smoke"}

        # 复现阶段状态机流转
        pipeline = [
            ("READ_PAPER", self.agents["reader"]),
            ("FIND_RESOURCES", self.agents["finder"]),
            ("BUILD_ENV", self.agents["builder"]),
            ("PLAN_EXECUTION", self.agents["planner"]),
            ("EXECUTE_CODE", self.agents["executor"]),
            ("VALIDATE", self.agents["validator"]),
        ]

        for state_name, agent in pipeline:
            self.state = state_name
            # P1-⑫ 用量计量：阶段级 plan（enter/exit 界定，失败也出栈）
            self.logger.begin_plan(state_name)
            self.logger.log("Orchestrator", f"enter_{state_name}", "RUNNING",
                            f"进入阶段: {state_name}")
            self._notify(state_name, "running")
            try:
                result = stage_result(state_name, self.data)
                if result is None:
                    result = agent.run(self.data)
                self._merge_result(state_name, result)
                self._accumulate_llm_calls(result)

                # 三层存储：FIND_RESOURCES 后按需懒加载代码/数据集/权重
                # 到 L0（失败不阻断流水线，仅告警）
                if state_name == "FIND_RESOURCES":
                    self._fetch_resources()

                # Docker 真实模式：BUILD_ENV 产出配置后即真实构建镜像，
                # 成功把 image_tag 透传给 EXECUTE_CODE；失败不阻断（slim 降级）
                if (state_name == "BUILD_ENV" and self.use_docker
                        and self.data.get("env_config", {}).get("build_required", True)):
                    build_res = self.agents["builder"].build_image(
                        self.data.get("env_config", {}), tag=DEFAULT_IMAGE_TAG)
                    if build_res.get("success"):
                        self.data.setdefault("env_config", {})[
                            "image_tag"] = build_res.get("tag", DEFAULT_IMAGE_TAG)
                        self.logger.log("Orchestrator", "build_image", "SUCCESS",
                                        f"镜像就绪: {build_res.get('tag')}")
                    else:
                        self.logger.log(
                            "Orchestrator", "build_image", "WARNING",
                            "镜像构建失败,降级为 python:3.11-slim + 注入依赖",
                            {"error": build_res.get("error") or
                                      (build_res.get("stderr") or "")[-300:]})

                # 多代码单元管理：执行计划落盘 data/plans/<paper_id>.json，
                # 供历史审计、归档恢复与后续复用（失败仅告警，不阻断）
                if state_name == "PLAN_EXECUTION":
                    self._save_plan(result)

                # 真实优化工作区：把复现代码物化到磁盘，
                # 供 Optimizer 真实执行器快照/补丁/重跑
                if state_name == "EXECUTE_CODE" and self.workspace_dir:
                    code = (result or {}).get("code", "") or \
                        self.data.get("execution", {}).get("code", "")
                    # 计划模式的 code 是入口脚本原文（多为 .sh），物化成
                    # workspace/run.py 没有意义（见 _optimization_plan_mode_reason）
                    if (result or {}).get("execution_mode") == "plan":
                        self.logger.log(
                            "Orchestrator", "materialize_workspace", "WARNING",
                            "官方代码计划模式：入口脚本不物化为优化工作区")
                    else:
                        self._materialize_workspace(code)

                if state_name == "EXECUTE_CODE" and profile and not self.mock_mode:
                    self._save_profile_evidence()

                # Prompt-Free 验证 + 修正闭环
                self._verify_step(state_name, agent, result)

                self.logger.log("Orchestrator", f"exit_{state_name}", "SUCCESS",
                                f"完成阶段: {state_name}")
                self.logger.end_plan(state_name)
                self._notify(state_name, "success")
            except Exception as e:
                self.logger.end_plan(state_name)
                self._notify(state_name, "error")
                self._fail(state_name, str(e))
                break

        # 优化阶段：仅在复现成功后触发
        plan_mode_reason = self._optimization_plan_mode_reason()
        if self.state != "ERROR":
            if profile and not self.mock_mode:
                self.data["optimization"] = {"optimized": False,
                    "reason": "CPU 真实冒烟仅验证流程，未核对论文数值，不执行优化"}
                self._notify("OPTIMIZING", "waiting")
            elif plan_mode_reason:
                # 官方代码计划模式：优化闭环不适用（见原因说明），
                # 不物化入口脚本原文、不产出误导性的失败 trial。
                self.data["optimization"] = {"optimized": False,
                                             "reason": plan_mode_reason}
                self._notify("OPTIMIZING", "waiting")
            elif not self.mock_mode and not self.workspace_dir:
                self.data["optimization"] = {"optimized": False,
                    "reason": "未配置真实优化执行器，已跳过（真实模式不使用模拟优化）"}
                self._notify("OPTIMIZING", "waiting")
            elif self.data.get("validation", {}).get("is_reproduced"):
                self.state = "OPTIMIZING"
                self.logger.begin_plan("OPTIMIZING")
                self.logger.log("Orchestrator", "enter_OPTIMIZING", "RUNNING",
                                "进入优化阶段")
                self._notify("OPTIMIZING", "running")
                try:
                    # 真实优化:把论文指标键绑定到执行器(奖励方向与对齐依据)
                    sim = getattr(self.agents["optimizer"], "simulator", None)
                    if isinstance(sim, RealSimulator):
                        sim.bind_paper(self.data.get("paper_info") or {})
                    opt_result = self.agents["optimizer"].run(self.data)
                    self.data["optimization"] = opt_result
                    self._accumulate_llm_calls(opt_result)
                    self.state = "OPTIMIZED"
                    self.logger.log("Orchestrator", "exit_OPTIMIZING", "SUCCESS",
                                    "优化阶段完成")
                    self.logger.end_plan("OPTIMIZING")
                    self._notify("OPTIMIZED", "success")
                except Exception as e:
                    self.logger.end_plan("OPTIMIZING")
                    self._notify("OPTIMIZING", "error")
                    self._fail("OPTIMIZING", str(e))
            else:
                self.data["optimization"] = {
                    "optimized": False,
                    "reason": self._optimization_skip_reason()}
                self._notify("OPTIMIZING", "waiting")

        # 报告生成（合并复现 + 优化）
        if self.state != "ERROR":
            self.state = "GENERATE_REPORT"
            self.logger.begin_plan("GENERATE_REPORT")
            self.data["audit_stats"] = self.logger.get_stats()
            self._notify("GENERATE_REPORT", "running")
            try:
                self.data["report"] = self.agents["reporter"].run(self.data) \
                    .get("report", "")
                self.logger.end_plan("GENERATE_REPORT")
                self._notify("GENERATE_REPORT", "success")
            except Exception as e:
                self.logger.end_plan("GENERATE_REPORT")
                self._notify("GENERATE_REPORT", "error")
                self._fail("GENERATE_REPORT", str(e))

        if self.state != "ERROR":
            # 三层存储：COMPLETED 前落盘资源 manifest 与存储统计
            self._finalize_storage()
            self.state = "COMPLETED"
            self.data["audit_stats"] = self.logger.get_stats()
            self.logger.log("Orchestrator", "finish_pipeline", "SUCCESS",
                            "流水线完成", self.data.get("audit_stats"))
        self._notify(self.state, "success")

        return self.get_result()

    # ---------------- 内部流程 ----------------

    def _notify(self, state_name: str, status: str) -> None:
        """向 UI 进度回调发送 (state, display_name, status) 事件。

        未注入 progress_cb（程序化调用/测试）时静默跳过。
        """
        if self.progress_cb is None:
            return
        display = STAGE_DISPLAY.get(state_name, state_name)
        try:
            self.progress_cb(state_name, display, status)
        except Exception:
            pass  # UI 进度事件失败不得影响流水线本身

    def _merge_result(self, state_name: str, result: dict) -> None:
        """将 Agent 输出合并进数据上下文。"""
        if state_name == "READ_PAPER":
            self.data["paper_info"] = result.get("paper_info", {})
            self.data["raw_text"] = result.get("raw_text", "")
        elif state_name == "FIND_RESOURCES":
            self.data["resources"] = result.get("resources", {})
        elif state_name == "BUILD_ENV":
            self.data["env_config"] = result.get("env_config", {})
        elif state_name == "PLAN_EXECUTION":
            self.data["execution_plan"] = result.get("execution_plan", {})
        elif state_name == "EXECUTE_CODE":
            self.data["execution"] = result
        elif state_name == "VALIDATE":
            self.data["validation"] = result

    def _optimization_skip_reason(self) -> str:
        """优化未触发的原因（区分"复现失败"/"压根没跑起来"/"跑了但信息不足"）。"""
        validation = self.data.get("validation", {}) or {}
        if validation.get("status") == "not_runnable":
            return f"代码未能运行，无法优化（{validation.get('reason', '未运行')}）"
        if validation.get("status") == "best_effort":
            return "代码为尽力而为的占位实现（论文信息不足），无法作为优化基线"
        return "复现未成功,跳过优化"

    def _optimization_plan_mode_reason(self) -> str:
        """官方代码计划模式下不跑真实优化时给出的原因（无则返回 ""）。

        优化器的真实闭环是「把单个文件当补丁对象，改写后 `python run.py`
        重跑」。官方代码不满足这个前提：`execution.code` 是入口脚本**原文**
        （多为 .sh，也可能是依赖 --root_path 的 run.py），把它单独物化成
        workspace/run.py 会丢掉整个仓库与数据准备步骤，每次 trial 都必然
        ImportError / FileNotFoundError——跑出来的只有一堆失败记录，
        还容易被读成"优化试过但没效果"。因此显式跳过并说明原因，
        而不是产出一串看似真实的失败 trial。
        """
        execution = self.data.get("execution", {}) or {}
        if execution.get("execution_mode") != "plan":
            return ""
        return ("官方代码按执行计划运行（execution_mode=plan）：优化闭环的"
                "「单文件补丁 + 重跑」不适用于含数据准备与多段调用的官方"
                "入口脚本，已跳过真实优化（复现结论不受影响）")

    # ---------------- 三层存储：懒加载与 manifest ----------------

    def _fetch_resources(self) -> None:
        """FIND_RESOURCES 后按需懒加载代码/数据集/权重到 L0。

        只拉当前任务最小集（代码仓库 + smoke 级数据集子集）；'未找到'
        等占位引用归一化为空。任何 fetch 异常仅记录 WARNING，不阻断流水线。
        """
        rm = self.resource_manager
        resources = self.data.get("resources", {}) or {}
        paper_id = self.data.get("paper_id", "")
        # P1-⑨: 确定性发现链选中的仓库为权威主线，LLM 猜测仅兜底
        discovery = resources.get("repo_discovery") or {}
        code_url = self._clean_ref(
            discovery.get("selected_repo")
            or resources.get("code_repo_url", ""))
        pinned_revision = discovery.get("pinned_revision") or ""
        dataset_name = self._clean_ref(resources.get("dataset_url", ""))
        weights_ref = self._clean_ref(
            resources.get("weights_url") or resources.get("weights_ref", ""))
        # 复现模式: ResourceFinder 的决策优先（考虑显式 full/内存），
        # 无决策时回退 storage.repro_level（默认 smoke）
        repro_mode = resources.get("repro_mode") or {}
        level = (repro_mode.get("effective_mode")
                 or (self.data.get("storage") or {}).get(
                     "repro_level", "smoke")) or "smoke"
        try:
            # 多代码单元拉取：code_units 存在时逐单元 clone 到
            # data/repos/<paper_id>/<unit_id>/；否则退回单仓库 fetch_code
            # （向后兼容）。main 单元的路径透传给 EnvBuilder 的静态依赖
            # 分析（env_builder 已支持 input_data["repo_path"]），
            # 让复现环境吃上真实仓库的 import 清单。
            units = resources.get("code_units") or []
            if units:
                unit_infos = rm.fetch_units(paper_id, units)
                main_info = next(
                    (u for u in unit_infos
                     if u.get("unit_id") == "main" and u.get("path")),
                    unit_infos[0] if unit_infos else {})
                fetched = {
                    "code": main_info,
                    "units": unit_infos,
                    "dataset": rm.fetch_dataset(paper_id, dataset_name,
                                                level=level),
                    "weights": rm.fetch_weights(paper_id, weights_ref),
                }
                repo_path = main_info.get("path", "")
                if repo_path:
                    self.data["repo_path"] = repo_path
            else:
                fetched = {
                    "code": rm.fetch_code(paper_id, code_url,
                                          revision=pinned_revision),
                    "dataset": rm.fetch_dataset(paper_id, dataset_name,
                                                level=level),
                    "weights": rm.fetch_weights(paper_id, weights_ref),
                }
            storage = self.data.setdefault("storage", {})
            storage["fetched"] = fetched
        except Exception as exc:
            self.logger.log("Orchestrator", "fetch_resources", "WARNING",
                            f"资源拉取失败，不阻断流水线: {str(exc)[-200:]}")

    def _save_profile_evidence(self) -> None:
        """Persist the actual execution records for user inspection."""
        import json
        root = self.resource_manager.data_root / "runs" / self.logger.session_id
        root.mkdir(parents=True, exist_ok=True)
        execution = self.data.get("execution") or {}
        execution["evidence_dir"] = str(root)
        (root / "execution.json").write_text(
            json.dumps(execution, ensure_ascii=False, indent=2), encoding="utf-8")
        for stream in ("stdout", "stderr"):
            (root / f"{stream}.txt").write_text(
                (execution.get("final") or {}).get(stream) or "", encoding="utf-8")

    def _save_plan(self, result: dict) -> None:
        """执行计划落盘 data/plans/<paper_id>.json（失败仅告警）。"""
        plan = (result or {}).get("execution_plan") or {}
        paper_id = self.data.get("paper_id", "")
        if not plan or not paper_id:
            return
        plan = dict(plan)
        plan.setdefault("paper_id", paper_id)
        try:
            path = self.resource_manager.save_plan(plan)
            self.data.setdefault("storage", {})["plan_path"] = path
            self.logger.log("Orchestrator", "save_plan", "SUCCESS",
                            f"执行计划已落盘: {path}")
        except Exception as exc:
            self.logger.log("Orchestrator", "save_plan", "WARNING",
                            f"执行计划落盘失败: {str(exc)[-200:]}")

    def _finalize_storage(self) -> None:
        """COMPLETED 前落盘资源 manifest 与存储统计。

        依赖 ResourceManager 的存量兼容格式（paper_id / created_at /
        resources / paper_title / code_url / dataset_name）；失败仅告警，
        不影响报告生成。
        """
        rm = self.resource_manager
        resources = self.data.get("resources", {}) or {}
        paper_id = self.data.get("paper_id", "")
        try:
            manifest = rm.build_manifest(
                paper_id=paper_id,
                paper_title=self.data.get("paper_title", ""),
                code_url=self._clean_ref(
                    resources.get("code_repo_url", "")),
                dataset_name=self._clean_ref(
                    resources.get("dataset_url", "")),
                weights_ref=self._clean_ref(
                    resources.get("weights_url") or
                    resources.get("weights_ref", "")))
            path = rm.save_manifest(manifest)
            stats = rm.stats()
            storage = self.data.setdefault("storage", {})
            storage["manifest"] = manifest
            storage["manifest_path"] = path
            storage["stats"] = stats
            self.logger.log("Orchestrator", "finalize_storage", "SUCCESS",
                            f"资源清单已落盘: {path}")
        except Exception as exc:
            self.logger.log("Orchestrator", "finalize_storage", "WARNING",
                            f"资源清单落盘失败: {str(exc)[-200:]}")

    @staticmethod
    def _clean_ref(ref: str) -> str:
        """'未找到' 等占位文本归一化为空引用。"""
        if not ref:
            return ""
        if ref.strip().lower() in ("未找到", "未知", "无", "none", "n/a",
                                   "null", "nan"):
            return ""
        return ref.strip()

    def _materialize_workspace(self, code: str) -> None:
        """把复现代码写入优化工作区（run.py），供真实优化闭环使用。"""
        if not code or not code.strip() or not self.workspace_dir:
            self.logger.log("Orchestrator", "materialize_workspace", "WARNING",
                            "无可用代码或未配置工作区,不落盘")
            return
        ws = Path(self.workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "run.py").write_text(code, encoding="utf-8")
        self.logger.log("Orchestrator", "materialize_workspace", "SUCCESS",
                        f"复现代码已物化到工作区: {ws / 'run.py'}")

    def _verify_step(self, state_name: str, agent, result: dict) -> None:
        """Prompt-Free 验证某步输出；未通过时按修正建议触发一次修正重试。"""
        if self.data.get("experiment_profile") and not self.mock_mode:
            # Fixed inputs use local checks and must not trigger another
            # training job through the LLM correction loop.
            fetched = (self.data.get("storage") or {}).get("fetched") or {}
            checks = {
                "READ_PAPER": bool((result.get("paper_info") or {}).get("title")),
                "FIND_RESOURCES": (fetched.get("code", {}).get("state") in ("cloned", "cached")
                                   and fetched.get("dataset", {}).get("state") in ("downloaded", "cached")),
                "BUILD_ENV": bool((result.get("env_config") or {}).get("requirements_txt")),
                "PLAN_EXECUTION": bool((result.get("execution_plan") or {}).get("steps")),
                "EXECUTE_CODE": result.get("success") is True,
                "VALIDATE": result.get("status") == "smoke_verified",
            }
            passed = checks.get(state_name, False)
            self.data.setdefault("verifications", []).append({
                "state": state_name, "agent": agent.name,
                "source": "deterministic_profile",
                "pass": passed,
                "issues": [] if passed else [f"{state_name} 未通过本地检查，请查看阶段输出"],
                "llm_calls": 0,
            })
            return
        verifier = self.agents["verifier"]
        verif = verifier.run({
            "agent_name": agent.name,
            "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
            "output": result,
        })
        self._accumulate_llm_calls(verif)
        self.data.setdefault("verifications", []).append(
            {"state": state_name, "agent": agent.name, **verif})

        if not verif.get("pass", False):
            self.logger.log("Verifier", state_name, "WARNING",
                            f"{agent.name} 输出未通过质量验证",
                            {"issues": verif.get("issues", [])})
            # 修正闭环：预算内重试一次（生成->验证->修正->再验证）
            retries = 0
            while retries < MAX_FIX_RETRIES and not verif.get("pass", False):
                retries += 1
                suggestions = verif.get("fix_suggestions", []) or []
                self.logger.log("Verifier", f"fix_{state_name}", "RUNNING",
                                f"第 {retries} 次修正: {agent.name}",
                                {"suggestions": suggestions})
                fixed_result = agent.run(self.data)
                self._merge_result(state_name, fixed_result)
                self._accumulate_llm_calls(fixed_result)
                verif = verifier.run({
                    "agent_name": agent.name,
                    "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
                    "output": fixed_result,
                })
                self._accumulate_llm_calls(verif)
                self.data.setdefault("verifications", []).append(
                    {"state": state_name, "agent": agent.name,
                     "round": retries + 1, **verif})
                self.data.setdefault("fix_records", []).append({
                    "state": state_name,
                    "agent": agent.name,
                    "round": retries,
                    "issues": verif.get("issues", []),
                    "suggestions": suggestions,
                })
                if not verif.get("pass", False):
                    self.logger.log("Verifier", f"fix_{state_name}", "WARNING",
                                    f"{agent.name} 修正后仍未通过验证")

    def _fail(self, stage: str, message: str) -> None:
        """进入 ERROR 状态并记录日志。"""
        self.state = "ERROR"
        self.error = message
        self.logger.log("Orchestrator", stage, "ERROR", f"阶段失败: {message}")

    def _accumulate_llm_calls(self, result: dict) -> None:
        """将 Agent / Verifier 输出的 llm_calls 增量累计进全局预算统计。"""
        calls = int((result or {}).get("llm_calls", 0) or 0)
        if calls:
            self.data["total_llm_calls"] = (
                self.data.get("total_llm_calls", 0) + calls)
            self.logger.add_llm_calls(calls)

    def get_result(self) -> dict:
        """获取最终结果。"""
        return {
            "state": self.state,
            "error": self.error,
            "data": self.data,
            "audit_logs": self.logger.get_summary(),
            "audit_stats": self.logger.get_stats(),
        }

    def get_state_machine(self) -> list:
        """返回状态机定义。"""
        return [{"state": s, "transitions": self._get_transitions(s)}
                for s in self.STATES]

    def _get_transitions(self, state: str) -> list:
        """获取状态的合法转移。"""
        transitions = {
            "INIT": ["READ_PAPER"],
            "READ_PAPER": ["FIND_RESOURCES", "ERROR"],
            "FIND_RESOURCES": ["BUILD_ENV", "ERROR"],
            "BUILD_ENV": ["PLAN_EXECUTION", "ERROR"],
            "PLAN_EXECUTION": ["EXECUTE_CODE", "ERROR"],
            "EXECUTE_CODE": ["VALIDATE", "ERROR"],
            "VALIDATE": ["OPTIMIZING", "GENERATE_REPORT", "ERROR"],
            "OPTIMIZING": ["OPTIMIZED", "ERROR"],
            "OPTIMIZED": ["GENERATE_REPORT"],
            "GENERATE_REPORT": ["COMPLETED", "ERROR"],
            "COMPLETED": [],
            "ERROR": ["INIT"],
        }
        return transitions.get(state, [])
