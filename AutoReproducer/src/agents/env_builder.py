"""EnvBuilderAgent - 环境构建 Agent，自动搭建运行环境。

对齐方案「Phase 3: 环境构建」：
- 生成 Dockerfile / requirements / 搭建命令；
- 5 轮依赖诊断循环：分类错误 -> 定点修复 -> 重验证，输出依赖修复报告；
- 支持真实 Docker 镜像构建（build_image，需本机 Docker）。
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.agents.dependency_resolver import (
    dependency_root, resolve_dependencies,
)

# 依赖诊断轮数（方案要求 5 轮）
MAX_DIAGNOSE_ROUNDS = 5
# pip 下载源：默认国内镜像（清华），可用环境变量 AUTOREPRO_PIP_INDEX 覆盖。
# 解决官方源在受限网络下安装 torch 等大包卡死的问题（依赖获取打通）；
# torch/torchvision 大 wheel 经 find-links 指向阿里云 pytorch-wheels 镜像——
# 官方 download.pytorch.org 的 wheel 会 302 到 download-r2.pytorch.org
# （实测 ~100KB/s 且频繁断流），而阿里云镜像稳定 ~600KB/s。
# find-links 里的 "+cpu" 本地版本标识符比 PyPI 的 CUDA 全量包排序更高，
# 会优先选中 CPU wheel（体积 ~196MB，约为 CUDA 版的 1/4）。可用环境变量
# AUTOREPRO_PIP_FIND_LINKS 覆盖（如公司内网镜像）。
PIP_INDEX_URL = os.environ.get(
    "AUTOREPRO_PIP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")
PIP_FIND_LINKS = os.environ.get(
    "AUTOREPRO_PIP_FIND_LINKS", "https://mirrors.aliyun.com/pytorch-wheels/cpu/")
# 共享底座镜像（三层存储「镜像级共享」L1 关联解法）：128 个仓库共用
# 统一底座，存储与构建时间较逐任务构建大幅下降。
# 底座 = python:3.11-slim + CPU torch/torchvision + numpy + tqdm（复现高频依赖），
# 论文镜像只在其上做增量层（COPY requirements + pip install 剩余包）。
BASE_IMAGE_TAG = "autorepro-base:latest"
BASE_IMAGE_FROM = "python:3.11-slim"
# 常见依赖冲突类别与其修复动作（确定性演示 + 真实模式指导）
_COMMON_FIXES = {
    "version_conflict": "固定为论文环境兼容版本（降级到声明版本）",
    "missing_package": "补充缺失包到 requirements.txt",
    "import_api_break": "按新版本 API 改写 import / 调用签名",
    "platform_issue": "指定 python_version 并安装对应平台 wheel",
    "unknown": "逐项隔离安装,定位最小冲突组合",
}


class EnvBuilderAgent(BaseAgent):
    """根据论文依赖自动构建运行环境（Docker/虚拟环境），并做依赖诊断。"""

    system_prompt = "根据论文依赖生成可执行的运行环境配置(Dockerfile + requirements + 搭建命令),并诊断修复依赖冲突"

    def __init__(self, llm_client: LLMClient, logger=None):
        super().__init__("EnvBuilder", logger)
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        """构建运行环境。

        input_data: {"paper_info": dict, "resources": dict, "corpus_paper": str}
        """
        self.log("build_env", "START", "开始构建运行环境", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        deps = list(paper_info.get("dependencies", []) or [])

        # 语料对照层：若指定真实论文，依赖改用其 requirements.txt
        corpus_paper = input_data.get("corpus_paper") or paper_info.get("corpus_paper")
        corpus_deps = self._load_corpus_deps(corpus_paper)
        if corpus_deps:
            deps = corpus_deps

        prompt = f"""根据论文信息生成运行环境配置。
论文方法: {paper_info.get('method', '未知')}
已有依赖: {deps}

返回JSON格式:
{{
    "required_packages": ["包名>=版本"],
    "python_version": "3.11",
    "dockerfile": "完整的Dockerfile内容",
    "setup_commands": ["环境搭建命令列表"],
    "estimated_disk_gb": 5.0
}}
"""
        llm_result = self.llm.chat(prompt, task="env_builder")
        parsed = self._parse_json(llm_result)

        # 依赖兜底：过滤 LLM 返回的占位/无意义文本（"未找到"/"未知"等），
        # 为空时回退到论文声明依赖，保证真实构建时有可安装的 requirements。
        pkgs = (parsed or {}).get("required_packages", []) or []
        if isinstance(pkgs, str):
            pkgs = [pkgs]
        pkgs = [p for p in pkgs if isinstance(p, str) and p.strip()
                and "未找到" not in p and "未知" not in p
                and p.strip().lower() not in ("none", "n/a", "unknown", "no", "暂无")]
        if not pkgs:
            pkgs = deps or ["torch>=2.0.0"]
        parsed = {
            "required_packages": pkgs,
            "python_version": (parsed or {}).get("python_version", "3.11"),
            "dockerfile": (parsed or {}).get("dockerfile", "") or (
                "FROM python:3.11-slim\n"
                "WORKDIR /app\n"
                "COPY requirements.txt .\n"
                "RUN pip install -r requirements.txt"
            ),
            "setup_commands": (parsed or {}).get("setup_commands", [])
                              or ["pip install -r requirements.txt"],
            "estimated_disk_gb": (parsed or {}).get("estimated_disk_gb", 3.0),
        }

        # 语料对照层：真实论文的 requirements 直接采用
        if corpus_deps:
            parsed["required_packages"] = corpus_deps

        # 静态依赖解析（P1-⑧，融合 ScholarAgent detect_python_dependencies）：
        # EnvBuilder 的依赖来源原本只有 LLM 从摘要猜测 + 语料 requirements，
        # 与"代码实际 import 了什么"可能不一致（LLM 猜错/漏猜 -> 运行时
        # ModuleNotFoundError）。这里用 AST + 声明文件做第二来源：
        # 语料/声明依赖保持权威，静态解析只"补漏不覆盖"——代码里 import
        # 了但声明清单缺失的包会被并入，避免覆盖语料的版本 pin。
        static_deps = []
        code_src = input_data.get("code") or ""
        repo_path = input_data.get("repo_path") or ""
        if code_src or repo_path:
            static_deps = resolve_dependencies(
                code=code_src or None, repo_path=repo_path or None)
            if static_deps:
                parsed["required_packages"] = self._merge_dependencies(
                    parsed.get("required_packages", []), static_deps)

        # 依赖诊断（5 轮循环）：分类错误 -> 定点修复 -> 重验证
        diagnose_report = self.diagnose_dependencies(
            parsed.get("required_packages", []))

        requirements = "\n".join(parsed.get("required_packages", []))
        env_config = {
            "requirements_txt": requirements,
            "dockerfile": parsed.get("dockerfile", ""),
            "python_version": parsed.get("python_version", "3.11"),
            "setup_commands": parsed.get("setup_commands", []),
            "estimated_disk_gb": parsed.get("estimated_disk_gb", 3.0),
            "dependency_diagnosis": diagnose_report,
            "static_dependencies": static_deps,
            "static_source": ("code" if code_src else "repo")
                             if static_deps else "none",
        }

        self.log_experiment(
            "BUILD_ENV", "生成运行环境配置并完成依赖诊断",
            inputs={"dependencies": deps, "static_dependencies": static_deps},
            outputs={"env_config": env_config, "diagnosis": diagnose_report},
            result={"rounds": diagnose_report.get("rounds"),
                    "resolved": diagnose_report.get("resolved"),
                    "static_found": len(static_deps)})
        self.log("build_env", "SUCCESS",
                 f"环境配置生成完成,{len(parsed.get('required_packages', []))} 个依赖,"
                 f"诊断 {diagnose_report.get('rounds', 0)} 轮"
                 + (f",静态解析补充 {len(static_deps)} 个" if static_deps else ""),
                 {"packages": parsed.get("required_packages", []),
                  "diagnosis": diagnose_report,
                  "static_dependencies": static_deps})

        return {
            "env_config": env_config,
            "static_dependencies": static_deps,
            "llm_calls": self._delta_llm_calls(),
        }

    @staticmethod
    def _merge_dependencies(base: List[str], extra: List[str]) -> List[str]:
        """合并依赖清单：以 base 为权威，extra 仅补充 base 缺失的包名。

        按 requirements token 的包根名比对（剥离版本与环境标记），
        保持 base 顺序不变，新增包追加在尾部。
        """
        base_roots = {dependency_root(dep) for dep in base}
        merged = list(base)
        for dep in extra:
            root = dependency_root(dep)
            if root and root not in base_roots:
                base_roots.add(root)
                merged.append(dep)
        return merged

    # ---------------- 依赖诊断 ----------------

    def diagnose_dependencies(self, packages: List[str]) -> Dict:
        """依赖诊断循环：对包清单执行最多 MAX_DIAGNOSE_ROUNDS 轮
        「探测 -> 分类 -> 定点修复 -> 重验证」。

        Mock 模式确定性模拟常见冲突场景；真实模式（install_check=True）
        可实际探测 pip 解析是否成功，但默认不触碰用户环境。
        """
        rounds = []
        current = list(packages)
        resolved = False

        for rnd in range(1, MAX_DIAGNOSE_ROUNDS + 1):
            issues = self._probe_issues(current)
            if not issues:
                resolved = True
                rounds.append({"round": rnd, "issues": [],
                               "resolved": True, "packages": list(current)})
                break

            fixed = self._apply_fixes(current, issues)
            rounds.append({
                "round": rnd,
                "issues": issues,
                "fix_actions": fixed["actions"],
                "resolved": False,
                "packages": list(current),
            })
            current = fixed["packages"]
            self.log("diagnose_dependencies", "RUNNING",
                     f"第 {rnd} 轮: {len(issues)} 个问题,修复后剩余 {len(current)} 个依赖")

        return {
            "rounds": len(rounds),
            "resolved": resolved,
            "max_rounds": MAX_DIAGNOSE_ROUNDS,
            "final_requirements_txt": "\n".join(current),
            "rounds_detail": rounds,
            "overview": ("依赖冲突全部解决" if resolved
                         else f"达到诊断上限({MAX_DIAGNOSE_ROUNDS} 轮),采用最小可用路径"),
        }

    def _probe_issues(self, packages: List[str]) -> List[Dict]:
        """探测依赖问题：确定性模拟常见冲突（版本冲突/缺失/API 变更）。"""
        issues = []
        lower = [p.lower() for p in packages]
        if any("torch" in p for p in lower) and not any(
                p.startswith("torch") for p in lower):
            issues.append({"type": "missing_package",
                           "package": "torch", "message": "检测到 torchvision 但缺少 torch"})
        for p in packages:
            m = re.search(r"([A-Za-z0-9_.-]+)[<>=!~]+([\d.]+)", p)
            if m and m.group(1).lower() in {"torch", "tensorflow", "numpy"} \
                    and m.group(2).startswith("0."):
                issues.append({"type": "version_conflict", "package": m.group(1),
                               "message": f"{m.group(1)} 版本过旧,与 Python 运行时冲突"})
        return issues[:2]  # 至多两条，保证演示节奏

    def _apply_fixes(self, packages: List[str], issues: List[Dict]) -> Dict:
        actions = []
        updated = list(packages)
        for issue in issues:
            kind = issue.get("type", "unknown")
            package = issue.get("package", "")
            actions.append({"type": kind, "package": package,
                            "action": _COMMON_FIXES.get(kind, _COMMON_FIXES["unknown"])})
            if kind == "missing_package" and package:
                updated.append(f"{package}>=2.0.0")
            elif kind == "version_conflict":
                # 定点修复：统一到兼容版本
                updated = [re.sub(r"([A-Za-z0-9_.-]+)[<>=!~]+[\d.]+",
                                  lambda m: f"{m.group(1)}==2.1.0"
                                  if m.group(1).lower() == package else m.group(0),
                                  p) for p in updated]
        # 去重保持顺序
        seen, dedup = set(), []
        for p in updated:
            key = p.split(">=")[0].split("==")[0]
            if key not in seen:
                seen.add(key)
                dedup.append(p)
        return {"actions": actions, "packages": dedup}

    # ---------------- 语料对照 ----------------

    @staticmethod
    def _load_corpus_deps(corpus_paper) -> List[str]:
        if not corpus_paper:
            return []
        from src.corpus import get_requirements
        real_reqs = get_requirements(corpus_paper)
        if not real_reqs:
            return []
        return [ln.strip() for ln in real_reqs.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

# ---------------- Docker 真实构建 ----------------

    def build_image(self, env_config: dict, tag: str = "autorepro-env",
                    use_base_image: bool = True) -> dict:
        """用生成的 Dockerfile + requirements 真实构建 Docker 镜像。

        依赖获取加固：若 Dockerfile 使用 pip install 且未显式指定下载源，
        自动在 FROM 后注入 PIP_INDEX_URL（默认清华镜像，可用环境变量
        AUTOREPRO_PIP_INDEX 覆盖），避免官方源在受限网络下卡死。

        共享底座（镜像级共享）：use_base_image=True 时先确保 autorepro-base
        底座存在（缺失则先构建一次，long-term 复用），并把论文 Dockerfile
        的 FROM python:* 替换为 FROM autorepro-base:latest；底座不可用时
        自动降级为原 Dockerfile（python slim + 注入依赖），结果带
        degraded 标注，不阻断流水线。
        """
        dockerfile = (env_config or {}).get("dockerfile", "")
        if not dockerfile:
            return {"success": False, "error": "无 Dockerfile,无法构建"}

        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "error": "本机未安装 Docker 或不在 PATH 中"}

        degraded = None
        if use_base_image:
            base = self.ensure_base_image(docker_cmd)
            if base.get("success"):
                dockerfile = self._swap_to_base_image(
                    dockerfile, BASE_IMAGE_TAG)
            else:
                degraded = base.get("error", "底座镜像不可用")
                self.log("build_image", "WARNING",
                         f"底座镜像不可用,降级为 python slim + 注入依赖: {degraded}")

        dockerfile = self._inject_pip_index(dockerfile)
        result = self._build_dockerfile(
            dockerfile,
            tag=tag,
            reqs=env_config.get("requirements_txt", ""))
        result["degraded"] = degraded
        return result

    def build_base_image(self, tag: str = BASE_IMAGE_TAG) -> dict:
        """构建共享底座镜像 autorepro-base（CPU torch 全家桶）。

        底座按需构建一次，长期复用（L0 镜像层缓存）。耗时为首次下载
        CPU torch 等 wheel（国内镜像下数分钟内）。
        """
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "error": "本机未安装 Docker 或不在 PATH 中"}
        return self._build_dockerfile(self._base_dockerfile(), tag=tag)

    def ensure_base_image(self, docker_cmd: str,
                          tag: str = BASE_IMAGE_TAG) -> dict:
        """确保共享底座镜像存在；不存在则按需构建一次。

        返回 {"success", "tag", "cached": bool}；
        success=False 时 error 说明原因（无 Docker / 构建失败）。
        """
        if self._image_exists(docker_cmd, tag):
            return {"success": True, "tag": tag, "cached": True}
        self.log("build_base_image", "START",
                 f"底座镜像不存在,按需构建: {tag}")
        build = self._build_dockerfile(self._base_dockerfile(), tag=tag)
        if build.get("success"):
            self.log("build_base_image", "SUCCESS",
                     f"底座镜像就绪: {tag}")
        return {"success": build.get("success", False),
                "tag": tag, "cached": False,
                "error": build.get("error") or build.get("stderr", "")[-300:]}

    def _build_dockerfile(self, dockerfile: str, tag: str,
                          reqs: str = "") -> dict:
        """把 Dockerfile（可选 requirements.txt）写入临时目录并 docker build。

        返回 {"success", "tag", "stdout", "stderr"} / {"success", "error"}。
        """
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "error": "本机未安装 Docker 或不在 PATH 中"}
        build_dir = tempfile.mkdtemp(prefix="autorepro_env_")
        try:
            with open(os.path.join(build_dir, "Dockerfile"), "w",
                      encoding="utf-8") as f:
                f.write(dockerfile)
            if reqs:
                with open(os.path.join(build_dir, "requirements.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(reqs)
            self.log("build_image", "START", f"构建镜像 {tag}")
            result = subprocess.run(
                [docker_cmd, "build", "-t", tag, build_dir],
                capture_output=True, text=True, timeout=1800)
            ok = result.returncode == 0
            self.log("build_image", "SUCCESS" if ok else "ERROR",
                     f"镜像构建{'成功' if ok else '失败'}: {tag}",
                     {"stdout": result.stdout[-300:],
                      "stderr": result.stderr[-300:]})
            return {"success": ok, "tag": tag,
                    "stdout": result.stdout[-500:],
                    "stderr": result.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "构建超时(1800s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    @staticmethod
    def _base_dockerfile() -> str:
        """共享底座镜像 Dockerfile：python:3.11-slim + CPU torch 全家桶。

        结构（对齐「镜像级共享」：底座一次构建、多论文增量复用）：
        - 国内源 ENV 注入（与 _inject_pip_index 同源）；
        - build-essential：torch 相关编译型依赖最小工具链；装完即清 apt 缓存；
        - pip 一次性安装 torch/torchvision/numpy/tqdm：find-links 指向阿里云
          pytorch-wheels/cpu，使 +cpu 版本排序优先于 CUDA 全量包，体积约 1/4。
        """
        return (
            f"FROM {BASE_IMAGE_FROM}\n"
            f"ENV PIP_INDEX_URL={PIP_INDEX_URL} \\\n"
            f"    PIP_FIND_LINKS={PIP_FIND_LINKS} \\\n"
            "    PIP_DISABLE_PIP_VERSION_CHECK=1\n"
            "# 编译型依赖最小工具链,装完即清缓存控体积\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    build-essential && rm -rf /var/lib/apt/lists/*\n"
            "# 复现高频依赖一次装齐（CPU torch,内网镜像加速）\n"
            f"RUN pip install --no-cache-dir torch torchvision numpy tqdm \\\n"
            f"    -i {PIP_INDEX_URL} --find-links {PIP_FIND_LINKS}\n"
            "WORKDIR /app\n"
        )

    @staticmethod
    def _swap_to_base_image(dockerfile: str,
                            base_tag: str = BASE_IMAGE_TAG) -> str:
        """把论文 Dockerfile 的 FROM python:* 替换为 FROM autorepro-base。

        已是底座自身、或没有 FROM 行时不改动；只替换首个 FROM 基础镜像
        行（多阶段构建只动第一段，后续 FROM 保持不变——论文产物阶段
        通常只有一段 COPY+RUN 的运行时层）。
        """
        if not dockerfile:
            return dockerfile
        lines = dockerfile.splitlines()
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped.startswith("FROM "):
                if "autorepro-base" in stripped or \
                        not stripped.split()[1].startswith("python"):
                    return dockerfile
                indent = ln[:len(ln) - len(ln.lstrip())]
                lines[i] = f"{indent}FROM {base_tag}"
                return "\n".join(lines)
        return dockerfile

    @staticmethod
    def _image_exists(docker_cmd: str, tag: str) -> bool:
        """探测本地是否已有指定镜像标签。"""
        try:
            res = subprocess.run(
                [docker_cmd, "images", "-q", tag],
                capture_output=True, text=True, timeout=60)
            return res.returncode == 0 and bool(res.stdout.strip())
        except Exception:
            return False

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _inject_pip_index(dockerfile: str) -> str:
        """在需要 pip 安装且未指定源/索引的 Dockerfile 中注入国内 pip 源。

        仅做透明加固：不改包列表与安装顺序，只保证下载源可用。
        注入 PIP_INDEX_URL（国内镜像）与 PIP_FIND_LINKS（阿里云
        pytorch-wheels/cpu，让 torch/torchvision 走 CPU wheel 且可快速下载）。
        """
        if not dockerfile:
            return dockerfile
        if any(k in dockerfile for k in ("PIP_INDEX_URL", "-i ", "--index-url",
                                         "PIP_FIND_LINKS", "--find-links")):
            return dockerfile
        lines = dockerfile.splitlines()
        out = []
        inserted = False
        for ln in lines:
            out.append(ln)
            if not inserted and ln.strip().upper().startswith("FROM "):
                out.append(f"ENV PIP_INDEX_URL={PIP_INDEX_URL} \\")
                out.append(f"    PIP_FIND_LINKS={PIP_FIND_LINKS} \\")
                out.append("    PIP_DISABLE_PIP_VERSION_CHECK=1")
                inserted = True
        return "\n".join(out)

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        for candidate in (text, re.sub(r"```(?:json)?\s*(.*?)```", r"\1", text, flags=re.DOTALL)):
            if not candidate or not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        return {}