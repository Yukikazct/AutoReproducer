"""CodeExecutorAgent - 代码执行Agent，在沙箱中运行论文代码"""
import json
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient


class CodeExecutorAgent(BaseAgent):
    """在Docker或本地沙箱中执行论文代码"""

    system_prompt = "在沙箱中安全执行论文代码,输出运行日志、数值结果与退出码"

    def __init__(self, llm_client: LLMClient, logger=None, use_docker: bool = False):
        super().__init__("CodeExecutor", logger)
        self.llm = llm_client
        self.use_docker = use_docker

    def run(self, input_data: dict) -> dict:
        """执行论文代码
        input_data: {"paper_info": dict, "env_config": dict, "resources": dict}
        """
        self.log("execute_code", "START", "开始执行代码", input_data)

        paper_info = input_data.get("paper_info", {})

        # 优先使用外部传入的真实代码(如论文复现脚本),否则 LLM 生成
        code = input_data.get("code", "")
        if not code:
            prompt = f"""根据论文信息生成一段简短的训练代码用于复现实验。
论文方法: {paper_info.get('method', '未知')}
指标: {paper_info.get('metrics', {})}
数据集: {paper_info.get('dataset', '未知')}

请生成Python代码，包含完整的训练和评估流程，并在最后打印关键指标。
代码要简洁但完整。
"""
            code = self.llm.chat(prompt)

        # 执行代码(本地或 Docker)
        execution_result = self._execute_code(code)

        self.log("execute_code",
                 "SUCCESS" if execution_result["success"] else "ERROR",
                 f"代码执行{'成功' if execution_result['success'] else '失败'}",
                 {"stdout": execution_result["stdout"][:500],
                  "stderr": execution_result["stderr"][:500],
                  "exit_code": execution_result.get("exit_code", -1)})

        return {
            "code": code,
            "execution": execution_result,
            "llm_calls": self.llm.get_call_count()
        }

    def _execute_code(self, code: str) -> dict:
        """执行代码:本地子进程或 Docker 容器"""
        if self.use_docker:
            return self._execute_code_docker(code)
        return self._execute_code_local(code)

    def _execute_code_local(self, code: str) -> dict:
        """在本地临时文件中执行代码"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "执行超时(30s)", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -2}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _execute_code_docker(self, code: str, image: str = "python:3.12-slim") -> dict:
        """在 Docker 容器中执行代码(挂载临时目录,隔离运行)"""
        workdir = tempfile.mkdtemp(prefix="autorepro_")
        script = os.path.join(workdir, "run.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)

        # Windows 路径转成正斜杠,Docker Desktop 自动映射驱动器
        mount = workdir.replace("\\", "/")
        try:
            result = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{mount}:/app", "-w", "/app",
                 image, "python", "run.py"],
                capture_output=True, text=True, timeout=120
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "执行超时(120s)", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -2}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)