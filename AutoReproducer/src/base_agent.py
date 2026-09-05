"""Agent基类 - 所有Agent的抽象基类"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from src.audit.audit_logger import AuditLogger


class BaseAgent(ABC):
    """所有Agent的抽象基类"""

    # 本 Agent 的质量标准(系统提示词)。供 Verifier 做 Prompt-Free 验证时复用,
    # 无需额外手写验证提示词。
    system_prompt = ""

    def __init__(self, name: str, logger: Optional[AuditLogger] = None):
        self.name = name
        self.logger = logger or AuditLogger()

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """执行Agent的核心逻辑"""
        pass

    def log(self, action: str, status: str, detail: str, data: dict = None):
        """记录日志"""
        if self.logger:
            self.logger.log(self.name, action, status, detail, data)

    def __repr__(self) -> str:
        return f"Agent({self.name})"