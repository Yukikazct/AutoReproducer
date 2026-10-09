"""Orchestrator - 编排器核心，管理 Agent 的状态机流转。

职责（对齐方案 4.1 / 4.4）：
1. 任务分解与流程控制：INIT -> READ_PAPER -> FIND_RESOURCES -> BUILD_ENV
   -> EXECUTE_CODE -> VALIDATE -> GENERATE_REPORT
   -> COMPLETED / ERROR；
2. Prompt-Free 验证闭环：每步输出经 Verifier 校验，未通过时按修正建议
   触发一次修正重试（预算内），形成「生成->验证->修正->再验证」；
3. 预算统计：汇总 LLM 调用次数，纳入审计统计与报告。
"""
from typing import Any, Dict, Optional
from pathlib import Path
from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.agents.paper_reader import PaperReaderAgent
from src.agents.resource_finder import ResourceFinderAgent
from src.agents.env_builder import EnvBuilderAgent
from src.agents.code_executor import CodeExecutorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.verifier import VerifierAgent
from src.agents.optimizer import OptimizerAgent
from src.resource_manager import ResourceManager

# 每步验证失败时最多触发的修正重试次数（预算约束）
MAX_FIX_RETRIES = 1

# EnvBuilder 真实构建时使用的镜像名；成功构建后通过 env_config.image_tag
# 透传给 CodeExecutor，让复现代码在本镜像内运行（含论文依赖）
DEFAULT_IMAGE_TAG = "autorepro-env"

_MAIN_PHASE_IDS = {"READ_PAPER": "generate_reader", "FIND_RESOURCES": "find_resources",
                   "BUILD_ENV": "build_environment", "EXECUTE_CODE": "execute_code",
                   "VALIDATE": "validate_result"}
_VERIFY_PHASE_IDS = {"READ_PAPER": "verify_reader", "FIND_RESOURCES": "verify_finder",
                     "BUILD_ENV": "verify_builder", "EXECUTE_CODE": "verify_executor",
                     "VALIDATE": "verify_validator"}


class Orchestrator:
    """编排复现、验证与报告；优化参数保留供未来实现。"""

    # 状态定义
    STATES = [
        "INIT", "READ_PAPER", "FIND_RESOURCES", "BUILD_ENV",
        "EXECUTE_CODE", "VALIDATE", "OPTIMIZING", "OPTIMIZED",
        "GENERATE_REPORT", "COMPLETED", "ERROR",
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 mock_mode: bool = True, logger: Optional[AuditLogger] = None,
                 max_trials: int = 10, use_docker: bool = False,
                 workspace_dir: Optional[str] = None,
                 resource_manager: Optional[ResourceManager] = None,
                 enable_optimization: bool = False):
        self.state = "INIT"
        self.mock_mode = mock_mode
        self.on_event = None
        self.logger = logger or AuditLogger()
        self.llm = llm_client or LLMClient(mock_mode=mock_mode)
        # P1-⑫ 用量计量：把 LLM 调用的 token/耗时通过 hook 归入当前 plan
        # （AuditLogger 按流水线阶段的 begin_plan/end_plan 界定 plan 边界）
        # 兼容外部注入的 mock/脚本 LLM（无 usage_hook 属性时静默跳过）。
        if hasattr(self.llm, "usage_hook"):
            self.llm.usage_hook = self.logger.record_llm_usage
        self.max_trials = max_trials
        self.use_docker = use_docker
        # 预留未来优化接口；当前版本即使请求启用也不执行优化。
        self.enable_optimization = enable_optimization is True
        self.workspace_dir = workspace_dir
        # L0 热缓存资源管理（三层存储）：FIND_RESOURCES 后按需懒加载，
        # COMPLETED 前落盘 manifest 与统计；可注入以隔离数据根（测试）。
        self.resource_manager = resource_manager or ResourceManager(
            logger=self.logger)

        # 初始化所有 Agent
        self.agents: Dict[str, Any] = {
            "reader": PaperReaderAgent(self.llm, self.logger),
            "finder": ResourceFinderAgent(self.llm, self.logger,
                                          offline=mock_mode),
            "builder": EnvBuilderAgent(self.llm, self.logger),
            "executor": CodeExecutorAgent(self.llm, self.logger,
                                          use_docker=use_docker,
                                          mock_mode=mock_mode),
            "validator": ResultValidatorAgent(self.llm, self.logger),
            "verifier": VerifierAgent(self.llm, self.logger),
            "optimizer": OptimizerAgent(self.llm, self.logger,
                                        max_trials=self.max_trials),
            "reporter": ReportGeneratorAgent(self.logger),
        }
        self.agents["executor"].on_event = self._emit_execution_event
        self.data: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def run(self, input_data: dict, on_event=None) -> dict:
        """执行复现、验证与报告；智能优化接口当前尚未开放。

        input_data 支持:
          - "paper_title": 论文标题（字符串输入方式）
          - "pdf_path": 论文 PDF 路径（上传/本地文件）
          - "code": 可选，外部提供的真实复现代码
          - "corpus_paper": 可选，PaperGuru-Benchmark 论文 id（语料对照层）
          - "enable_optimization": 可选布尔值，缺省 False；True 也记录为未开放并跳过
        """
        self.on_event = on_event
        self.state, self.error = "INIT", None
        if input_data.get("experiment_profile"):
            if self.mock_mode:
                self.data = {}
                self._fail("INIT", "官方仓库预设需要关闭Mock模式；不会用模拟结果冒充真实实验")
            else:
                from src.repository_reproduction import RepositoryReproduction
                result = RepositoryReproduction(
                    self.resource_manager.data_root, self.logger, self.use_docker,
                    llm=self.llm).run(
                        ({**input_data, "enable_optimization": self.enable_optimization}
                         if "enable_optimization" not in input_data and self.enable_optimization
                         else input_data), on_event=on_event)
                self.state, self.data, self.error = result["state"], result["data"], result["error"]
            self._emit_state(self.state, "", "error" if self.error else "success")
            return self.get_result()
        self.logger.log("Orchestrator", "start_pipeline", "START",
                        "开始自动复现流水线", input_data)

        # 透传用户输入（修复：此前 paper_title/pdf_path 未进入数据上下文）
        self.data = {
            "paper_title": input_data.get("paper_title", "") or "",
            "pdf_path": input_data.get("pdf_path", "") or "",
            "code": input_data.get("code", "") or "",
            "corpus_paper": input_data.get("corpus_paper"),
            "preferred_repo_url": input_data.get("preferred_repo_url", ""),
            "code_repo_url": input_data.get("code_repo_url", ""),
            "verifications": [],
            "fix_records": [],
            "optimization": self._reserved_optimization(input_data),
        }

        # 论文稳定 ID（三层存储索引）：corpus 语料键 / sha1(title) 前 12 位
        paper_id = self.resource_manager.paper_id_for(
            self.data.get("paper_title", ""),
            corpus_key=self.data.get("corpus_paper") or "")
        self.data["paper_id"] = paper_id
        self.data["storage"] = {"paper_id": paper_id,
                                "repro_level": input_data.get(
                                    "repro_level", "smoke") or "smoke"}

        # 复现阶段状态机流转
        pipeline = [
            ("READ_PAPER", self.agents["reader"]),
            ("FIND_RESOURCES", self.agents["finder"]),
            ("BUILD_ENV", self.agents["builder"]),
            ("EXECUTE_CODE", self.agents["executor"]),
            ("VALIDATE", self.agents["validator"]),
        ]
        if self.on_event:
            self.on_event({"type": "pipeline_plan", "pipeline": "generic",
                           "stages": self._generic_plan(pipeline)})

        for state_name, agent in pipeline:
            self.state = state_name
            self._emit_state(state_name, agent.name, "running", phase_id=_MAIN_PHASE_IDS[state_name], attempt=1)
            main_completed = False
            # P1-⑫ 用量计量：阶段级 plan（enter/exit 界定，失败也出栈）
            self.logger.begin_plan(state_name)
            self.logger.log("Orchestrator", f"enter_{state_name}", "RUNNING",
                            f"进入阶段: {state_name}")
            try:
                result = agent.run(self.data)
                self._merge_result(state_name, result)
                self._accumulate_llm_calls(result)

                # 三层存储：FIND_RESOURCES 后按需懒加载代码/数据集/权重
                # 到 L0（失败不阻断流水线，仅告警）
                if state_name == "FIND_RESOURCES":
                    self._fetch_resources()

                # Docker 真实模式：BUILD_ENV 产出配置后即真实构建镜像，
                # 成功把 image_tag 透传给 EXECUTE_CODE；失败不阻断（slim 降级）
                if state_name == "BUILD_ENV" and self.use_docker:
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

                # Main work and its quality review have independent stage rows.
                self._emit_agent_completion(state_name, agent, result, attempt=1)
                main_completed = True
                # Prompt-Free 验证 + 修正闭环
                self._verify_step(state_name, agent, result)

                self.logger.log("Orchestrator", f"exit_{state_name}", "SUCCESS",
                                f"完成阶段: {state_name}")
                self.logger.end_plan(state_name)
            except Exception as e:
                self.logger.end_plan(state_name)
                if not main_completed:
                    self._emit_state(state_name, agent.name, "error", phase_id=_MAIN_PHASE_IDS[state_name],
                                     attempt=1, reason=str(e), outcome="exception")
                self._fail(state_name, str(e))
                break

        # 优化仅保留未来参数接口：成功、失败或显式请求均不启动优化。
        if self.state != "ERROR":
            self.logger.log("Orchestrator", "skip_OPTIMIZING", "SKIP",
                            self.data["optimization"]["reason"], self.data["optimization"])
            self._emit_state(self.state, "Optimizer", "skipped", phase_id="reserve_optimization",
                             reason=self.data["optimization"]["reason"], outcome="not_implemented")

        # 报告生成（合并复现 + 优化）
        if self.state != "ERROR":
            self.state = "GENERATE_REPORT"
            self._emit_state(self.state, "ReportGenerator", "running", phase_id="generate_report", attempt=1)
            self.logger.begin_plan("GENERATE_REPORT")
            self.data["audit_stats"] = self.logger.get_stats()
            try:
                self.data["report"] = self.agents["reporter"].run(self.data) \
                    .get("report", "")
                self.logger.end_plan("GENERATE_REPORT")
                self._emit_state(self.state, "ReportGenerator", "success", phase_id="generate_report", attempt=1)
            except Exception as e:
                self.logger.end_plan("GENERATE_REPORT")
                self._emit_state(self.state, "ReportGenerator", "error", phase_id="generate_report", attempt=1, reason=str(e))
                self._fail("GENERATE_REPORT", str(e))

        if self.state != "ERROR":
            # 三层存储：COMPLETED 前落盘资源 manifest 与存储统计
            self._finalize_storage()
            self.state = "COMPLETED"
            self.data["audit_stats"] = self.logger.get_stats()
            self.logger.log("Orchestrator", "finish_pipeline", "SUCCESS",
                            "流水线完成", self.data.get("audit_stats"))

        self._emit_state(self.state, "", "error" if self.error else "success")
        return self.get_result()

    def _emit_execution_event(self, event):
        if self.on_event and event.get("type") in {"execution_step", "execution_output"}:
            self.on_event({**event, "phase_id": _MAIN_PHASE_IDS["EXECUTE_CODE"]})

    def _emit_state(self, state, agent, status, *, phase_id=None, attempt=None, reason=None, outcome=None):
        if self.on_event:
            event = {"type": "state", "state": state, "agent": agent, "status": status}
            event.update({key: value for key, value in {"phase_id": phase_id, "attempt": attempt,
                          "reason": reason, "outcome": outcome}.items() if value is not None})
            self.on_event(event)

    # ---------------- 内部流程 ----------------

    def _generic_plan(self, pipeline):
        titles = {"READ_PAPER": "论文解析", "FIND_RESOURCES": "资源定位",
                  "BUILD_ENV": "环境分析与配置", "EXECUTE_CODE": "代码执行",
                  "VALIDATE": "论文数值比对"}
        stages = []
        for state, agent in pipeline:
            stages += [{"id": _MAIN_PHASE_IDS[state], "agent": agent.name, "title": titles[state],
                        "description": "执行本阶段工作；质量审查单独记录。", "status": "waiting"},
                       {"id": _VERIFY_PHASE_IDS[state], "agent": "Verifier", "title": "核验" + titles[state],
                        "description": "审查本阶段输出，按实际轮次记录修正和最终结果。", "status": "waiting"}]
        stages += [{"id": "reserve_optimization", "agent": "Optimizer", "title": "智能优化（预留接口）",
                    "description": "当前版本尚未开放。", "status": "skipped"},
                   {"id": "generate_report", "agent": "ReportGenerator", "title": "报告生成",
                    "description": "汇总真实执行与独立核验结果。", "status": "waiting"}]
        return stages

    def _emit_agent_completion(self, state_name, agent, result, attempt):
        status, reason, outcome = "success", None, "completed"
        if state_name == "EXECUTE_CODE":
            final = (result or {}).get("final") or {}
            if (result.get("success") is False or result.get("not_runnable")
                    or final.get("success") is False or final.get("timed_out") or final.get("cancelled")
                    or final.get("exit_code") not in (None, 0)):
                status, outcome = "error", "execution_failed"
                reason = result.get("reason") or final.get("reason") or final.get("stderr") or "执行输出表明本阶段未成功。"
        elif state_name == "VALIDATE":
            outcome = result.get("status") or (
                "reproduced" if result.get("is_reproduced") is True else "inconclusive")
            # A completed comparison may reject the paper's numbers. Keep that
            # conclusion separate from failure to execute the validation stage.
            if (result.get("success") is False or result.get("result_level") == "failed"
                    or outcome == "execution_failed"):
                status = "error"
            reason = result.get("reason")
            if not reason and result.get("is_reproduced") is not True:
                reason = "最终数值未通过确定性复现核验。"
        self._emit_state(state_name, agent.name, status, phase_id=_MAIN_PHASE_IDS[state_name],
                         attempt=attempt, reason=str(reason)[:600] if reason else None, outcome=outcome)

    def _reserved_optimization(self, input_data: dict) -> dict:
        requested = input_data.get("enable_optimization", self.enable_optimization) is True
        return {"optimized": False, "requested": requested, "available": False,
                "status": "not_implemented" if requested else "disabled",
                "reason": ("智能优化接口已预留，当前版本尚未开放。" if requested
                           else "智能优化未启用；当前版本仅保留未来接口。")}

    def _merge_result(self, state_name: str, result: dict) -> None:
        """将 Agent 输出合并进数据上下文。"""
        if state_name == "READ_PAPER":
            self.data["paper_info"] = result.get("paper_info", {})
            self.data["raw_text"] = result.get("raw_text", "")
        elif state_name == "FIND_RESOURCES":
            self.data["resources"] = result.get("resources", {})
        elif state_name == "BUILD_ENV":
            self.data["env_config"] = result.get("env_config", {})
        elif state_name == "EXECUTE_CODE":
            self.data["execution"] = result
            if result.get("effective_env_config"):
                self.data["env_config"] = result["effective_env_config"]
        elif state_name == "VALIDATE":
            self.data["validation"] = result

    def _optimization_skip_reason(self) -> str:
        """优化未触发的原因（区分"复现失败"/"压根没跑起来"/"跑了但信息不足"）。"""
        validation = self.data.get("validation", {}) or {}
        if validation.get("status") == "not_runnable":
            return f"代码未能运行，无法优化（{validation.get('reason', '未运行')}）"
        if validation.get("status") == "best_effort":
            return "代码为尽力而为的占位实现（论文信息不足），无法作为优化基线"
        if validation.get("status") == "no_reference_metrics":
            return "论文未声明参考指标数值，无法确认复现基线，跳过优化"
        if validation.get("status") in {"execution_failed", "execution_incomplete", "invalid_metrics"}:
            return f"执行或指标证据未通过核验，跳过优化（{validation.get('reason', '证据无效')}）"
        if validation.get("status") == "smoke_passed":
            return "仅冒烟通过，尚无完整实验基线，跳过优化"
        return "复现未成功,跳过优化"

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
        # Mock demonstrations must remain offline even for curated known titles.
        if self.mock_mode:
            code_url = ""
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
            fetched = {
                "code": rm.fetch_code(paper_id, code_url,
                                      revision=pinned_revision),
                "dataset": rm.fetch_dataset(paper_id, dataset_name,
                                            level=level),
                "weights": rm.fetch_weights(paper_id, weights_ref),
            }
            self.data.setdefault("storage", {})["fetched"] = {
                k: {"path": v.get("path", ""), "state": v.get("state", "")}
                for k, v in fetched.items()}
        except Exception as exc:
            self.logger.log("Orchestrator", "fetch_resources", "WARNING",
                            f"资源拉取失败，不阻断流水线: {str(exc)[-200:]}")

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
        verifier = self.agents["verifier"]
        def review(output, attempt):
            self._emit_state(state_name, "Verifier", "running", phase_id=_VERIFY_PHASE_IDS[state_name], attempt=attempt)
            try:
                reviewed = verifier.run({
                    "agent_name": agent.name,
                    "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
                    "output": output,
                })
                accepted = bool(reviewed.get("pass"))
                issues = reviewed.get("issues") or []
                reason = None if accepted else (
                    "；".join(str(issue) for issue in issues[:3])[:600]
                    if isinstance(issues, list) else str(issues)[:600])
                self._accumulate_llm_calls(reviewed)
            except Exception as exc:
                self._emit_state(state_name, "Verifier", "error", phase_id=_VERIFY_PHASE_IDS[state_name],
                                 attempt=attempt, reason=str(exc), outcome="exception")
                raise
            self._emit_state(state_name, "Verifier", "success" if accepted else "error",
                             phase_id=_VERIFY_PHASE_IDS[state_name], attempt=attempt,
                             reason=reason or (None if accepted else "质量审查未通过。"),
                             outcome="accepted" if accepted else "rejected")
            return reviewed

        verif = review(result, 1)
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
                self._emit_state(state_name, agent.name, "running", phase_id=_MAIN_PHASE_IDS[state_name], attempt=retries + 1)
                try:
                    fixed_result = agent.run(self.data)
                    self._merge_result(state_name, fixed_result)
                    self._accumulate_llm_calls(fixed_result)
                    self._emit_agent_completion(state_name, agent, fixed_result, attempt=retries + 1)
                except Exception as exc:
                    self._emit_state(state_name, agent.name, "error", phase_id=_MAIN_PHASE_IDS[state_name],
                                     attempt=retries + 1, reason=str(exc), outcome="exception")
                    raise
                verif = review(fixed_result, retries + 1)
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
            "BUILD_ENV": ["EXECUTE_CODE", "ERROR"],
            "EXECUTE_CODE": ["VALIDATE", "ERROR"],
            "VALIDATE": ["GENERATE_REPORT", "ERROR"],
            "OPTIMIZING": ["OPTIMIZED", "ERROR"],
            "OPTIMIZED": ["GENERATE_REPORT"],
            "GENERATE_REPORT": ["COMPLETED", "ERROR"],
            "COMPLETED": [],
            "ERROR": ["INIT"],
        }
        return transitions.get(state, [])
