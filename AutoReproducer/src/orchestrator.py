"""Orchestrator - 编排器核心，管理Agent的状态机流转"""
from typing import Optional
from src.audit.audit_logger import AuditLogger
from src.llm.ollama_client import LLMClient
from src.agents.paper_reader import PaperReaderAgent
from src.agents.resource_finder import ResourceFinderAgent
from src.agents.env_builder import EnvBuilderAgent
from src.agents.code_executor import CodeExecutorAgent
from src.agents.result_validator import ResultValidatorAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.verifier import VerifierAgent
from src.agents.optimizer import OptimizerAgent


class Orchestrator:
    """编排器 - 管理六步流水线的状态机流转"""

    # 状态定义
    STATES = [
        "INIT",
        "READ_PAPER",
        "FIND_RESOURCES",
        "BUILD_ENV",
        "EXECUTE_CODE",
        "VALIDATE",
        "OPTIMIZING",
        "OPTIMIZED",
        "GENERATE_REPORT",
        "COMPLETED",
        "ERROR"
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 mock_mode: bool = True, logger: Optional[AuditLogger] = None):
        self.state = "INIT"
        self.logger = logger or AuditLogger()
        self.llm = llm_client or LLMClient(mock_mode=mock_mode)

        # 初始化所有Agent
        self.agents = {
            "reader": PaperReaderAgent(self.llm, self.logger),
            "finder": ResourceFinderAgent(self.llm, self.logger),
            "builder": EnvBuilderAgent(self.llm, self.logger),
            "executor": CodeExecutorAgent(self.llm, self.logger),
            "validator": ResultValidatorAgent(self.llm, self.logger),
            "verifier": VerifierAgent(self.llm, self.logger),
            "optimizer": OptimizerAgent(self.llm, self.logger),
            "reporter": ReportGeneratorAgent(self.logger),
        }

        self.data = {}
        self.error = None

    def run(self, input_data: dict) -> dict:
        """执行完整的复现流程(复现 -> 验证 -> 优化 -> 报告)"""
        self.logger.log("Orchestrator", "start_pipeline", "START",
                        "开始自动复现流水线", input_data)

        # 语料对照层:可选的真实论文锚点
        self.data["corpus_paper"] = input_data.get("corpus_paper")

        # 复现阶段状态机流转(优化在 VALIDATE 之后按需触发)
        pipeline = [
            ("READ_PAPER", self.agents["reader"]),
            ("FIND_RESOURCES", self.agents["finder"]),
            ("BUILD_ENV", self.agents["builder"]),
            ("EXECUTE_CODE", self.agents["executor"]),
            ("VALIDATE", self.agents["validator"]),
        ]

        for state_name, agent in pipeline:
            self.state = state_name
            self.logger.log("Orchestrator", f"enter_{state_name}", "RUNNING",
                           f"进入阶段: {state_name}")

            try:
                result = agent.run(self.data)

                # 合并结果到数据上下文
                if state_name == "READ_PAPER":
                    self.data["paper_info"] = result.get("paper_info", {})
                    self.data["raw_text"] = result.get("raw_text", "")
                elif state_name == "FIND_RESOURCES":
                    self.data["resources"] = result.get("resources", {})
                elif state_name == "BUILD_ENV":
                    self.data["env_config"] = result.get("env_config", {})
                elif state_name == "EXECUTE_CODE":
                    self.data["execution"] = result
                elif state_name == "VALIDATE":
                    self.data["validation"] = result

                # Prompt-Free 验证:复用该 Agent 系统提示词检查输出质量
                self._verify_step(state_name, agent, result)

                # 记录LLM调用次数
                llm_calls = result.get("llm_calls", 0)
                if llm_calls:
                    self.data.setdefault("total_llm_calls", 0)
                    self.data["total_llm_calls"] += llm_calls

                self.logger.log("Orchestrator", f"exit_{state_name}", "SUCCESS",
                               f"完成阶段: {state_name}")

            except Exception as e:
                self.state = "ERROR"
                self.error = str(e)
                self.logger.log("Orchestrator", state_name, "ERROR",
                               f"阶段失败: {e}")
                break

        # 优化阶段:仅在复现成功后触发
        if self.state != "ERROR":
            if self.data.get("validation", {}).get("is_reproduced"):
                self.state = "OPTIMIZING"
                self.logger.log("Orchestrator", "enter_OPTIMIZING", "RUNNING",
                               "进入优化阶段")
                try:
                    self.data["optimization"] = self.agents["optimizer"].run(self.data)
                    self.state = "OPTIMIZED"
                    self.logger.log("Orchestrator", "exit_OPTIMIZING", "SUCCESS",
                                   "优化阶段完成")
                except Exception as e:
                    self.state = "ERROR"
                    self.error = str(e)
                    self.logger.log("Orchestrator", "OPTIMIZING", "ERROR",
                                   f"优化失败: {e}")
            else:
                self.data["optimization"] = {"optimized": False,
                                             "reason": "复现未成功,跳过优化"}

        # 报告生成(合并复现 + 优化)
        if self.state != "ERROR":
            self.state = "GENERATE_REPORT"
            self.data["audit_summary"] = self.logger.get_stats()
            try:
                self.data["report"] = self.agents["reporter"].run(self.data).get("report", "")
            except Exception as e:
                self.state = "ERROR"
                self.error = str(e)
                self.logger.log("Orchestrator", "GENERATE_REPORT", "ERROR",
                               f"报告生成失败: {e}")

        if self.state != "ERROR":
            self.state = "COMPLETED"
            self.data["audit_summary"] = self.logger.get_stats()
            self.logger.log("Orchestrator", "finish_pipeline", "SUCCESS",
                           "流水线完成", self.data.get("audit_summary"))

        return self.get_result()

    def _verify_step(self, state_name: str, agent, result: dict):
        """Prompt-Free 验证某一步的输出质量,结果记入 data["verifications"]。"""
        verifier = self.agents["verifier"]
        verif = verifier.run({
            "agent_name": agent.name,
            "system_prompt": getattr(agent, "system_prompt", "") or agent.name,
            "output": result,
        })
        self.data.setdefault("verifications", []).append(
            {"state": state_name, "agent": agent.name, **verif})
        if not verif.get("pass", False):
            self.logger.log("Verifier", state_name, "WARNING",
                           f"{agent.name} 输出未通过质量验证", verif)

    def get_result(self) -> dict:
        """获取最终结果"""
        return {
            "state": self.state,
            "error": self.error,
            "data": self.data,
            "audit_logs": self.logger.get_summary(),
            "audit_stats": self.logger.get_stats()
        }

    def get_state_machine(self) -> list:
        """返回状态机定义"""
        return [
            {"state": s, "transitions": self._get_transitions(s)}
            for s in self.STATES
        ]

    def _get_transitions(self, state: str) -> list:
        """获取状态的合法转移"""
        transitions = {
            "INIT": ["READ_PAPER"],
            "READ_PAPER": ["FIND_RESOURCES", "ERROR"],
            "FIND_RESOURCES": ["BUILD_ENV", "ERROR"],
            "BUILD_ENV": ["EXECUTE_CODE", "ERROR"],
            "EXECUTE_CODE": ["VALIDATE", "ERROR"],
            "VALIDATE": ["OPTIMIZING", "GENERATE_REPORT", "ERROR"],
            "OPTIMIZING": ["OPTIMIZED", "ERROR"],
            "OPTIMIZED": ["GENERATE_REPORT"],
            "GENERATE_REPORT": ["COMPLETED", "ERROR"],
            "COMPLETED": [],
            "ERROR": ["INIT"],
        }
        return transitions.get(state, [])