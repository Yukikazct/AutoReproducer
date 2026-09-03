"""ResourceFinderAgent - 资源查找Agent，定位代码仓库和数据集"""
import json
import re
from src.base_agent import BaseAgent
from src.llm.ollama_client import LLMClient


class ResourceFinderAgent(BaseAgent):
    """从论文信息中查找代码仓库、数据集和相关资源"""

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("ResourceFinder", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """查找论文相关资源
        input_data: {"paper_info": dict, "raw_text": str}
        """
        self.log("find_resources", "START", "开始查找资源", input_data)

        paper_info = input_data.get("paper_info", {})
        raw_text = input_data.get("raw_text", "")

        # 1. 从文本中提取GitHub链接
        urls = self._extract_urls(raw_text)
        github_urls = [u for u in urls if "github.com" in u]

        # 2. 用LLM推测代码仓库
        prompt = f"""根据论文信息，推测最可能的代码仓库URL和使用的数据集。
论文标题: {paper_info.get('title', '未知')}
方法: {paper_info.get('method', '未知')}

返回JSON格式:
{{
    "code_repo_url": "最可能的GitHub URL或'未找到'",
    "alternative_repos": ["备用仓库1", "备用仓库2"],
    "dataset_url": "数据集URL或'未找到'",
    "confidence": 0.0-1.0
}}
"""
        llm_result = self.llm.chat(prompt)
        try:
            parsed = json.loads(llm_result)
        except json.JSONDecodeError:
            parsed = {
                "code_repo_url": github_urls[0] if github_urls else "未找到",
                "alternative_repos": [],
                "dataset_url": "未找到",
                "confidence": 0.7 if github_urls else 0.3
            }

        # 合并提取的URL
        parsed["extracted_urls"] = urls[:10]
        parsed["github_urls"] = github_urls

        self.log("find_resources", "SUCCESS",
                 f"找到 {len(github_urls)} 个GitHub仓库",
                 {"github_urls": github_urls, "confidence": parsed.get("confidence")})

        return {
            "resources": parsed,
            "llm_calls": self.llm.get_call_count()
        }

    def _extract_urls(self, text: str) -> list:
        """从文本中提取URL"""
        url_pattern = r'https?://[^\s\)\]}"]+'
        return list(set(re.findall(url_pattern, text)))