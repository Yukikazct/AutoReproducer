"""Ollama LLM客户端 - 封装的Ollama API调用"""
import json
import urllib.request
import urllib.error
from typing import Optional


class LLMClient:
    """Ollama API客户端，支持Mock模式和真实模式"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 mock_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mock_mode = mock_mode
        self.call_count = 0

    def chat(self, prompt: str, system_prompt: str = "",
             temperature: float = 0.3) -> str:
        """发送聊天请求到Ollama"""
        self.call_count += 1

        if self.mock_mode:
            return self._mock_response(prompt)

        return self._real_chat(prompt, system_prompt, temperature)

    def _real_chat(self, prompt: str, system_prompt: str,
                   temperature: float) -> str:
        """真实的Ollama API调用"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "temperature": temperature
        }
        if system_prompt:
            payload["messages"].append({
                "role": "system",
                "content": system_prompt
            })
        payload["messages"].append({
            "role": "user",
            "content": prompt
        })

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            return f"[Ollama API Error: {e}]"

    def _mock_response(self, prompt: str) -> str:
        """Mock模式返回模拟数据(按提示词关键特征精确分发,避免关键词误命中)"""
        # Verifier: 质量验证
        if "质量验证器" in prompt or "待验证输出" in prompt:
            return json.dumps({"pass": True, "issues": [],
                               "fix_suggestions": [], "confidence": 0.9})
        # ResultValidator: 比对验证
        if "比对论文声明" in prompt or "提取到的实际指标" in prompt:
            return json.dumps({"match": True, "differences": [],
                               "confidence": 0.85,
                               "analysis": "运行结果与论文声明基本一致(复现成功)"})
        # ResourceFinder: 资源定位
        if "代码仓库" in prompt and "推测" in prompt:
            return json.dumps({"code_repo_url": "https://github.com/example/repo",
                               "alternative_repos": [],
                               "dataset_url": "https://example.com/dataset",
                               "confidence": 0.7})
        # EnvBuilder: 环境配置
        if "运行环境配置" in prompt or "Dockerfile" in prompt:
            return json.dumps({
                "required_packages": ["torch>=2.0.0", "torchvision>=0.15.0",
                                      "numpy>=1.24.0", "tqdm>=4.65.0"],
                "python_version": "3.12",
                "dockerfile": "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt",
                "setup_commands": ["pip install -r requirements.txt"],
                "estimated_disk_gb": 3.0
            })
        # CodeExecutor: 生成训练代码
        if "训练代码" in prompt:
            return "print('Training complete. Test accuracy: 85.2%')"
        # Optimizer: 优化建议
        if "优化建议" in prompt or "optimize" in prompt.lower():
            return json.dumps({
                "suggestions": [
                    "将学习率从0.01降至0.005",
                    "增加Batch Normalization层",
                    "使用AdamW优化器替代SGD"
                ]
            })
        # PaperReader: 结构化解析
        if "提取结构化信息" in prompt or "论文内容" in prompt:
            return json.dumps({
                "title": "AutoReproducer: 基于多智能体协作的论文自动复现与优化系统",
                "authors": ["AutoReproducer Team"],
                "method": "多智能体协作框架，包含1个Orchestrator和7个Agent",
                "dependencies": ["Python 3.12", "PyTorch", "Streamlit"],
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "dataset": "CIFAR-10",
                "code_url": "https://github.com/example/autoreproducer"
            })
        return f"[Mock Response] 收到请求: {prompt[:50]}..."

    def get_call_count(self) -> int:
        return self.call_count