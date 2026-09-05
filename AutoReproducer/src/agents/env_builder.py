"""EnvBuilderAgent - 环境构建Agent，自动搭建运行环境"""
import json
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient


class EnvBuilderAgent(BaseAgent):
    """根据论文依赖自动构建运行环境（Docker/虚拟环境）"""

    system_prompt = "根据论文依赖生成可执行的运行环境配置(Dockerfile + requirements + 搭建命令)"

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

        # 语料对照层:若指定真实论文,依赖改用其 requirements.txt
        corpus_paper = input_data.get("corpus_paper") or paper_info.get("corpus_paper")
        corpus_deps = []
        if corpus_paper:
            from src.corpus import get_requirements
            real_reqs = get_requirements(corpus_paper)
            if real_reqs:
                corpus_deps = [ln.strip() for ln in real_reqs.splitlines()
                               if ln.strip() and not ln.strip().startswith("#")]
        if corpus_deps:
            deps = corpus_deps

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

        # 语料对照层:若指定真实论文,依赖直接用其 requirements.txt
        if corpus_deps:
            parsed["required_packages"] = corpus_deps

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

    def build_image(self, env_config: dict, tag: str = "autorepro-env") -> dict:
        """用生成的 Dockerfile + requirements 真实构建 Docker 镜像"""
        import subprocess
        import tempfile
        import os
        import shutil

        dockerfile = env_config.get("dockerfile", "")
        if not dockerfile:
            return {"success": False, "error": "无 Dockerfile,无法构建"}

        build_dir = tempfile.mkdtemp(prefix="autorepro_env_")
        with open(os.path.join(build_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile)
        reqs = env_config.get("requirements_txt", "")
        if reqs:
            with open(os.path.join(build_dir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write(reqs)

        self.log("build_image", "START", f"构建镜像 {tag}")
        try:
            result = subprocess.run(
                ["docker", "build", "-t", tag, build_dir],
                capture_output=True, text=True, timeout=600
            )
            ok = result.returncode == 0
            self.log("build_image", "SUCCESS" if ok else "ERROR",
                     f"镜像构建{'成功' if ok else '失败'}: {tag}",
                     {"stdout": result.stdout[-300:], "stderr": result.stderr[-300:]})
            return {"success": ok, "tag": tag,
                    "stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "构建超时(600s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)