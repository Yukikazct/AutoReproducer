"""EnvBuilderAgent - 环境构建Agent，自动搭建运行环境"""
import json
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient


class EnvBuilderAgent(BaseAgent):
    """根据论文依赖自动构建运行环境（Docker/虚拟环境）"""

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("EnvBuilder", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """构建运行环境
        input_data: {"paper_info": dict, "resources": dict}
        """
        self.log("build_env", "START", "开始构建运行环境", input_data)

        paper_info = input_data.get("paper_info", {})
        resources = input_data.get("resources", {})

        deps = paper_info.get("dependencies", [])

        # 用LLM生成依赖列表和Dockerfile
        prompt = f"""根据论文信息生成运行环境配置。
论文方法: {paper_info.get('method', '未知')}
已有依赖: {deps}

返回JSON格式:
{{
    "required_packages": ["包名>=版本"],
    "python_version": "3.12",
    "dockerfile": "完整的Dockerfile内容",
    "setup_commands": ["环境搭建命令列表"],
    "estimated_disk_gb": 5.0
}}
"""
        llm_result = self.llm.chat(prompt)
        try:
            parsed = json.loads(llm_result)
        except json.JSONDecodeError:
            parsed = {
                "required_packages": deps if deps else ["torch>=2.0.0"],
                "python_version": "3.12",
                "dockerfile": "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt",
                "setup_commands": ["pip install -r requirements.txt"],
                "estimated_disk_gb": 3.0
            }

        # 生成requirements.txt
        requirements = "\n".join(parsed.get("required_packages", []))
        dockerfile = parsed.get("dockerfile", "")

        env_config = {
            "requirements_txt": requirements,
            "dockerfile": dockerfile,
            "python_version": parsed.get("python_version", "3.12"),
            "setup_commands": parsed.get("setup_commands", []),
            "estimated_disk_gb": parsed.get("estimated_disk_gb", 3.0)
        }

        self.log("build_env", "SUCCESS",
                 f"环境配置生成完成，包含 {len(parsed.get('required_packages', []))} 个依赖",
                 {"packages": parsed.get("required_packages", [])})

        return {
            "env_config": env_config,
            "llm_calls": self.llm.get_call_count()
        }