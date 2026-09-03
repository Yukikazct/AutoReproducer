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
            "reporter": ReportGeneratorAgent(self.logger),
        }

        self.data = {}
        self.error = None

    def run(self, input_data: dict) -> dict:
        """执行完整的复现流程"""
        self.logger.log("Orchestrator", "start_pipeline", "START",
                        "开始自动复现流水线", input_data)

        # 状态机流转
        pipeline = [
            ("READ_PAPER", self.agents["reader"]),
            ("FIND_RESOURCES", self.agents["finder"]),
            ("BUILD_ENV", self.agents["builder"]),
            ("EXECUTE_CODE", self.agents["executor"]),
            ("VALIDATE", self.agents["validator"]),
            ("GENERATE_REPORT", self.agents["reporter"]),
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
                elif state_name == "GENERATE_REPORT":
                    self.data["report"] = result.get("report", "")

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

        if self.state != "ERROR":
            self.state = "COMPLETED"
            self.data["audit_summary"] = self.logger.get_stats()
            self.logger.log("Orchestrator", "finish_pipeline", "SUCCESS",
                           "流水线完成", self.data.get("audit_summary"))

        return self.get_result()

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
            "VALIDATE": ["GENERATE_REPORT", "ERROR"],
            "GENERATE_REPORT": ["COMPLETED", "ERROR"],
            "COMPLETED": [],
            "ERROR": ["INIT"],
        }
        return transitions.get(state, [])