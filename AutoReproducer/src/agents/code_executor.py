"""CodeExecutorAgent - 代码执行 Agent，在沙箱中运行论文代码。

对齐方案「Phase 4: 代码执行」：
- 两阶段执行：smoke test（短时冒烟，快速暴露环境问题）-> full run（完整运行）；
- 捕获标准输出、错误日志、退出码；
- 支持本地子进程（隔离临时目录 + 超时）与 Docker 容器两种沙箱；
- 本地模式执行前按 env_config 依赖清单自动 pip 安装（幂等缓存 +
  独立超时 + 失败诊断），修复"EnvBuilder 给出依赖但本地执行器直接运行
  导致 ModuleNotFoundError"缺陷——复现环境与执行环境现在保持一致；
- 执行前语法门：清洗后的代码必须能 compile，不通过则针对"截断/语法
  错误"再生成（限次），仍不可编译则诚实短路为"未运行"，绝不把残码
  送进沙箱——避免把"代码被截断"掩盖成沙箱里的 IndentationError；
- 信息不足时不生成针对性代码，短路为"无法运行"，交由 ResultValidator
  判定为"无法验证"而非"复现失败"。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional
from src.base_agent import BaseAgent
from src.llm.llm_client import LLMClient
from src.agents.env_builder import PIP_INDEX_URL, PIP_FIND_LINKS

LOCAL_TIMEOUT_SMOKE = 10
LOCAL_TIMEOUT_FULL = 60
DOCKER_TIMEOUT_SMOKE = 30
DOCKER_TIMEOUT_FULL = 300
# 本地依赖安装超时（numpy/matplotlib/torch 等大包需要更长时间）
LOCAL_PIP_TIMEOUT = 300
# 进程内依赖安装结果缓存：依赖清单文本 -> ""(已就绪) 或 失败诊断文本。
# smoke/full/多次优化重跑共用一个进程，只对同一清单安装一次；
# 失败也缓存，避免反复重装浪费时间。
_INSTALLED_DEPS: Dict[str, str] = {}

# 代码不可编译时的再生成次数上限（LLM 输出被截断是常见故障）
MAX_CODE_REGEN = 2
# 信息不足时 LLM 应按约定返回的标记行（整份"代码"只有这一行注释）
_INSUFFICIENT_INFO_MARK = "# INSUFFICIENT_INFO"
# 语法错误信息中提示"输出被截断"的特征词
_TRUNCATION_HINTS = (
    "unexpected eof", "eof in multi-line", "unterminated",
    "was never closed", "unexpected end of",
)
# 末尾行以这些字符结尾 -> 语句明显没写完（截断的典型特征）
_TRUNCATION_TAIL_CHARS = "=*+-([{,:\\"
# 判定"未知/占位"论文信息用的空值模式（与 PaperReader._UNKNOWN_RE 判据一致）
_UNKNOWN_RE = re.compile(
    r"^\s*(|未知.*|未找到|无|n/?a|none|null)\s*$", re.IGNORECASE)
# Exit code：代码在进入沙箱前就被拦下（信息不足/语法错误）
EXIT_NOT_RUNNABLE = -5
# Exit code：本地执行被危险代码静态门拦下（命令执行/动态执行/网络/递归删除）
EXIT_DANGER_BLOCKED = -6

# 本地无沙箱执行前的危险代码静态门：命中即拒绝执行。高信号、对「复现
# 训练脚本」低误报；是正则兜底而非正式沙箱，生产复现不可信代码请用 Docker。
_DANGEROUS_PATTERNS = (
    (re.compile(r"\bsubprocess\b"), "subprocess 进程/命令执行"),
    (re.compile(r"\bos\.system\b"), "os.system 命令执行"),
    (re.compile(r"\bos\.popen\b"), "os.popen 命令执行"),
    (re.compile(r"\bos\.spawn\w*\b"), "os.spawn* 进程创建"),
    (re.compile(r"\bpty\b"), "pty 终端"),
    (re.compile(r"\beval\s*\("), "eval 动态执行"),
    (re.compile(r"\bexec\s*\("), "exec 动态执行"),
    (re.compile(r"\b__import__\s*\("), "__import__ 动态导入"),
    (re.compile(r"\bsocket\b"), "socket 网络"),
    (re.compile(r"\brequests\b"), "requests 网络外联"),
    (re.compile(r"\burllib\b"), "urllib 网络外联"),
    (re.compile(r"\bhttp\.client\b"), "http.client 网络"),
    (re.compile(r"\bftplib\b"), "ftplib 网络"),
    (re.compile(r"\bsmtplib\b"), "smtplib 邮件外发"),
    (re.compile(r"\bparamiko\b"), "paramiko SSH"),
    (re.compile(r"\bhttpx\b"), "httpx 网络外联"),
    (re.compile(r"\baiohttp\b"), "aiohttp 网络外联"),
    (re.compile(r"\bshutil\.rmtree\b"), "shutil.rmtree 递归删除"),
)

# markdown 代码块围栏（可能带 python 语言标注）
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# 行首行号（"1 def f(x):" 这类带行号转储）：数字前空白保留，数字后
# 最多吃掉一个分隔空白，剩下的空白是原本的缩进，必须留给代码。
_LINE_NO_RE = re.compile(r"^(\s*)\d+[ \t]?")
# 行首残留的围栏/引号残片
_FENCE_LEFT = re.compile(r"^\s*(```+|>>>|\.\.\.)\s*", re.MULTILINE)
# 判定"看起来像 Python 代码行"的行首（\w 会匹配中文,故全部用 ASCII 白名单）
# 覆盖：import/from/def/class/if/for/while/try/except/with/return/print/raise/
# pass/break/continue/del/assert/global/nonlocal/yield/match/case/lambda/
# 装饰器@/注释#/赋值= / 函数调用()/索引访问[]/属性访问. / 数字
_CODE_LINE_START = re.compile(
    r"^\s*(?:"
    r"import\s|from\s|def\s|class\s|if\s|elif\s|else\s*:|for\s|while\s|"
    r"try\s*:|except\s|finally\s*:|with\s|return\s|print\s*\(|raise\s|"
    r"pass\s*$|break\s*$|continue\s*$|del\s|assert\s|global\s|nonlocal\s|"
    r"yield\s|match\s|case\s|lambda\s|@|#|"
    r"[A-Za-z_][A-Za-z0-9_.]*\s*=|"                    # 赋值
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\(|"                   # 函数调用 super().__init__()
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\[|"                   # 索引 self.net[0]
    r"[A-Za-z_][A-Za-z0-9_.]*\s*\."                    # 属性访问 self.net.forward
    r"|[A-Za-z_\[\(\"']|[\d+\-.]"                      # 兜底：字母/括号/引号/数字开头
    r")")


class CodeExecutorAgent(BaseAgent):
    """在 Docker 或本地沙箱中执行论文代码。"""

    system_prompt = "在沙箱中安全执行论文代码,输出运行日志、数值结果与退出码"

    def __init__(self, llm_client: LLMClient, logger=None, use_docker: bool = False):
        super().__init__("CodeExecutor", logger)
        self.llm = llm_client
        self.use_docker = use_docker

    def run(self, input_data: dict) -> dict:
        """执行论文代码（smoke test + full run）。

        input_data: {"paper_info", "env_config", "resources", "code"(可选)}
        """
        self.log("execute_code", "START", "开始执行代码", input_data)

        paper_info = input_data.get("paper_info", {}) or {}
        env_config = input_data.get("env_config", {}) or {}
        code = input_data.get("code", "") or ""
        self.env_config = env_config  # 供执行阶段选择镜像/依赖

        if code:
            # 外部提供的真实复现代码：只做清洗，不走生成/再生成
            code = self._sanitize_code(code)
        else:
            # 信息不足时不生成针对性代码，诚实短路（下游判"无法验证"）
            if self._info_insufficient(paper_info):
                return self._not_runnable(
                    "论文信息不足（缺少方法/数据集/指标），无法生成"
                    "针对性复现代码；请提供完整 PDF 或更完整的摘要", code="")
            code = self._produce_code(paper_info)

        # 语法门：不可编译的代码绝不进沙箱——残码在沙箱里会被报成
        # IndentationError 之类，掩盖"输出被截断"这个真实原因。
        syntax_error = self._syntax_error(code)
        if syntax_error:
            return self._not_runnable(
                f"代码存在语法错误，未执行: {syntax_error}", code=code)

        # 危险代码静态门：本地执行无沙箱，拒绝明显危险的调用（命令执行/
        # 动态执行/网络外联/递归删除）。Docker 已是隔离沙箱，不必拦。
        if not self.use_docker:
            danger = self._dangerous_constructs(code)
            if danger:
                return self._not_runnable(
                    f"代码含危险调用，已拒绝执行: {danger}", code=code)

        smoke = self._execute_code(code, stage="smoke")
        if not smoke["success"]:
            # smoke 失败：不浪费预算跑 full，返回诊断信息
            result = {"stages": [{"stage": "smoke", **smoke}],
                      "success": False, "final": smoke,
                      "code": code}
            self.log_experiment(
                "EXECUTE_CODE", "smoke test 失败,终止 full run",
                inputs={"code": code}, outputs=smoke,
                result={"success": False})
            self.log("execute_code", "ERROR",
                     f"smoke test 失败: {smoke.get('stderr', '')[:120]}",
                     {"stage": "smoke", "exit_code": smoke.get("exit_code")})
            return {**result, "llm_calls": self._delta_llm_calls()}

        full = self._execute_code(code, stage="full")
        stages = [{"stage": "smoke", **smoke}, {"stage": "full", **full}]
        result = {"stages": stages, "success": full["success"],
                  "final": full, "code": code}

        self.log_experiment(
            "EXECUTE_CODE", "完成 smoke + full 两阶段执行",
            inputs={"code": code},
            outputs={"smoke": smoke, "full": full},
            result={"success": full["success"]},
        )
        self.log("execute_code",
                 "SUCCESS" if full["success"] else "ERROR",
                 f"代码执行{'成功' if full['success'] else '失败'} "
                 f"(smoke 通过, full {'通过' if full['success'] else '失败'})",
                 {"smoke_exit": smoke.get("exit_code"),
                  "full_exit": full.get("exit_code"),
                  "stdout_tail": full.get("stdout", "")[-300:]})

        return {**result, "llm_calls": self._delta_llm_calls()}

    # ---------------- 代码生成 ----------------

    def _produce_code(self, paper_info: Dict) -> str:
        """生成复现代码；语法不通过时针对"截断/语法错误"再生成（限次）。

        返回仍可能是不可编译的代码——由调用方 run() 的语法门统一判定并
        短路为"未运行"，此处只负责"多试几次"，不负责掩盖失败。
        """
        code = self._sanitize_code(self._generate_code(paper_info))
        for attempt in range(1, MAX_CODE_REGEN + 1):
            err = self._syntax_error(code)
            if err is None:
                return code
            reason = ("疑似输出被截断" if self._looks_truncated(code, err)
                      else "语法错误")
            finish = getattr(self.llm, "last_finish_reason", "")
            self.log("generate_code", "WARNING",
                     f"生成代码不可编译（{reason}，第 {attempt} 次）: {err}"
                     + (f" [finish_reason={finish}]" if finish else ""))
            code = self._sanitize_code(
                self._regenerate_code(paper_info, err, attempt))
        return code

    def _generate_code_prompt(self, paper_info: Dict) -> str:
        return f"""根据论文信息生成一段简短的训练代码用于复现实验。
论文方法: {paper_info.get('method', '未知')}
指标: {paper_info.get('metrics', {})}
数据集: {paper_info.get('dataset', '未知')}

【关键输出约束 - 必须严格遵守】
1. 只输出一份可直接运行的 Python 脚本（完整训练+评估流程,最后打印关键指标）；
2. 输出的每一行都必须是合法 Python 代码,严禁出现任何解释性文字、中文叙述、
   说明语句或自然语言段落；
3. 不要使用 markdown 代码块围栏(``` 或 ```python)包裹输出,不要输出围栏标记；
4. 如需注释仅使用以 # 开头的 Python 注释；
5. 第一行直接开始写代码,不要有开场白；
6. 必须输出完整脚本,不要在函数/循环中途停止,每一行都要写完整；
7. 若上面的论文方法/数据集确实是未知的占位值,无法据此写出针对性代码,
   则只输出一行 `{_INSUFFICIENT_INFO_MARK}` 并停止,严禁用无关数据集
   (如 CIFAR-10/IMDB)编造一个与本论文无关的模型来充数。
"""

    def _generate_code(self, paper_info: Dict) -> str:
        return self.llm.chat(self._generate_code_prompt(paper_info),
                             task="code_executor")

    def _regenerate_code(self, paper_info: Dict, err: str, attempt: int) -> str:
        """再生成：把上一次的失败原因回灌给 LLM，要求输出完整脚本。"""
        prompt = self._generate_code_prompt(paper_info) + f"""
【上一次输出不可用 - 第 {attempt} 次重试】
上一次生成的代码无法通过编译，原因: {err}
这通常意味着输出被截断了。请重新输出一份**完整**的 Python 脚本：
每个函数体/循环体都要有正确的缩进，最后一行必须是完整语句。
"""
        return self.llm.chat(prompt, task="code_executor")

    def _sanitize_code(self, raw: str) -> str:
        """将 LLM 原始输出清洗为可执行的纯净 Python 代码。

        兜底处理两类常见污染：
        1. markdown 代码块围栏包裹(```python ... ```)；
        2. 代码块外/代码中的中文叙述行("为了...""假设我们使用..."等自然语言)。
        清洗后产物为纯代码文本;仍含语法错误时如实保留,由执行阶段判定。
        """
        if not raw or not raw.strip():
            return raw or ""
        text = raw.strip()

        # 1) 提取最长的 markdown 代码块（若被围栏包裹）
        fenced = _CODE_FENCE.findall(text)
        if fenced:
            text = max(fenced, key=len).strip()
            # 直接尝试编译——代码块内应只有纯代码，保持缩进
            try:
                compile(text, "<generated>", "exec")
                return text
            except SyntaxError:
                pass  # 可能混入了叙述行，继续向下清理

        # 2) 逐行剥离叙述行,只保留代码行与代码内空行
        cleaned = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if not stripped:
                if cleaned and cleaned[-1].strip():
                    cleaned.append(ln)
                continue
            # 中文叙述行一律丢弃（含中文且非 # 注释）
            if re.search(r"[\u4e00-\u9fff]", stripped) and not stripped.startswith("#"):
                continue
            if _CODE_LINE_START.match(stripped):
                cleaned.append(ln)
        code = "\n".join(cleaned).strip("\n")

        # 3) 语法兜底: 若整体不可编译,再去掉围栏残片/行首行号后重新过滤
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError:
            code = _FENCE_LEFT.sub("", code)
            lines = []
            for ln in code.splitlines():
                # 只清掉行首行号与行尾空白,保留前导缩进——缩进一旦被抹掉,
                # 函数体/循环体会整体塌陷,把"输出被截断"这个真实原因
                # 伪装成一个更难定位的 IndentationError。
                fixed = _LINE_NO_RE.sub(r"\1", ln).rstrip()
                # 去行号后不像代码行（如续行 "  2)"）时保留原行,避免误删
                if not _CODE_LINE_START.match(fixed) \
                        and _CODE_LINE_START.match(ln.rstrip()):
                    fixed = ln.rstrip()
                ln = fixed
                if not ln.strip():
                    continue
                if (_CODE_LINE_START.match(ln)
                        and not (re.search(r"[\u4e00-\u9fff]", ln)
                                 and not ln.startswith("#"))):
                    lines.append(ln)
            code = "\n".join(lines)
        return code

    # ---------------- 执行前检查 ----------------

    @staticmethod
    def _dangerous_constructs(code: str) -> Optional[str]:
        """扫描代码中的危险调用，命中返回可读原因，否则 None。

        仅针对本地无沙箱执行（_execute_code_local）：本地模式以完整用户权限
        运行 LLM 代码，明显危险的调用（命令执行/动态执行/网络外联/递归删除）
        一律拒绝。正则兜底，非正式沙箱；生产复现不可信代码请用 Docker。
        """
        if not code:
            return None
        for pattern, label in _DANGEROUS_PATTERNS:
            if pattern.search(code):
                return label
        return None

    @staticmethod
    def _syntax_error(code: str) -> Optional[str]:
        """编译检查：语法错误返回可读信息，通过则返回 None。"""
        if not code or not code.strip():
            return "代码为空"
        try:
            compile(code, "<generated>", "exec")
            return None
        except SyntaxError as e:
            return f"{e.msg} (line {e.lineno})"
        except ValueError as e:      # 源码含空字节等
            return str(e)

    @staticmethod
    def _looks_truncated(code: str, err: str) -> bool:
        """判断语法错误是否更像"输出被截断"而非"模型写错了语法"。"""
        if any(h in err.lower() for h in _TRUNCATION_HINTS):
            return True
        tail = next((ln.strip() for ln in reversed(code.splitlines())
                     if ln.strip()), "")
        return bool(tail) and tail[-1] in _TRUNCATION_TAIL_CHARS

    @staticmethod
    def _info_insufficient(paper_info: Dict) -> bool:
        """论文结构化信息是否不足以生成针对性复现代码。

        判据：解析层显式标记 info_sufficient=False / insufficient_info，
        或方法与数据集双双缺失/为占位值。缺方法必不足以写代码；缺数据集
        但给出了声明指标时仍可尝试（例如纯数学/合成数据的方法）。
        """
        if paper_info.get("info_sufficient") is False:
            return True
        if paper_info.get("insufficient_info") is True:
            return True
        method = str(paper_info.get("method", "") or "")
        dataset = str(paper_info.get("dataset", "") or "")
        metrics = paper_info.get("metrics") or {}
        return bool(_UNKNOWN_RE.match(method) and _UNKNOWN_RE.match(dataset)
                    and not metrics)

    def _not_runnable(self, reason: str, code: str) -> dict:
        """代码未进入执行阶段（信息不足/语法错误）时的统一返回。

        与"跑了但失败"区分：exit_code=EXIT_NOT_RUNNABLE 且带 not_runnable
        标记，供 ResultValidator 判为"无法验证"而非"复现失败"。
        """
        stage = {"stage": "precheck", "success": False, "stdout": "",
                 "stderr": reason, "exit_code": EXIT_NOT_RUNNABLE,
                 "not_runnable": True}
        self.log_experiment(
            "EXECUTE_CODE", "代码未通过执行前检查,未进入沙箱",
            inputs={"code": code}, outputs=stage, result={"success": False})
        self.log("execute_code", "ERROR", f"代码未运行: {reason}",
                 {"exit_code": EXIT_NOT_RUNNABLE, "not_runnable": True})
        return {"stages": [stage], "success": False, "final": stage,
                "code": code, "not_runnable": True, "reason": reason,
                "llm_calls": self._delta_llm_calls()}

    # ---------------- 执行 ----------------

    def _execute_code(self, code: str, stage: str,
                      workdir: Optional[str] = None) -> Dict:
        """执行代码：本地子进程或 Docker 容器，按阶段使用不同超时。

        workdir: 指定执行目录时在目标目录执行且不清理（生命周期由调用方
        管理，如优化器真实执行配合快照回滚）；缺省时使用临时目录（用完删除）。
        """
        if self.use_docker:
            return self._execute_code_docker(code, stage, workdir=workdir)
        return self._execute_code_local(code, stage, workdir=workdir)

    def execute_in_workspace(self, code: str, workdir: str,
                             stage: str = "full") -> Dict:
        """在指定工作区目录中执行代码（真实优化闭环用）。

        与 _execute_code 的区别：工作目录由调用方提供且执行后保留
        （不清理），配合 src.safety.workspace_snapshot 完成
        "补丁 -> 真实重跑 -> 快照回滚"的安全优化闭环。
        """
        return self._execute_code(code, stage, workdir=workdir)

    def _execute_code_local(self, code: str, stage: str,
                            workdir: Optional[str] = None) -> Dict:
        """在本地执行代码：临时目录（不指定 workdir）或目标目录执行。

        执行前按 env_config 依赖清单自动安装依赖（_ensure_local_deps），
        依赖安装失败时直接返回失败诊断，不浪费脚本执行预算。
        """
        # 危险代码静态门（兜底）：Optimizer 真实执行 execute_in_workspace
        # 绕过 run() 直接进这里，仍需拦截危险调用。
        danger = self._dangerous_constructs(code)
        if danger:
            return {"success": False, "stdout": "",
                    "stderr": f"拒绝执行(危险代码): {danger}",
                    "exit_code": EXIT_DANGER_BLOCKED, "danger_blocked": True}

        # 本地无沙箱边界提示（一次性，避免 smoke/full/优化重跑反复刷屏）
        if not getattr(self, "_warned_unsandboxed", False):
            self._warned_unsandboxed = True
            self.log("execute_local", "WARNING",
                     "本地模式无沙箱隔离，仅用于演示/可信代码；"
                     "生产复现请 use_docker=True")

        cleanup = workdir is None
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="autorepro_exec_")
        else:
            os.makedirs(workdir, exist_ok=True)

        # 依赖预装：缺失依赖时运行必然失败，先安装再执行
        deps_err = self._ensure_local_deps(workdir)
        if deps_err:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)
            return {"success": False, "stdout": "",
                    "stderr": deps_err, "exit_code": -4,
                    "deps_prepared": False}

        script = os.path.join(workdir, "run.py")
        timeout = LOCAL_TIMEOUT_SMOKE if stage == "smoke" else LOCAL_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=timeout,
                cwd=workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "deps_prepared": True,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1,
                    "deps_prepared": True}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2, "deps_prepared": True}
        finally:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    def _ensure_local_deps(self, workdir: str) -> Optional[str]:
        """确保本地执行环境已安装论文依赖；None 表示就绪，否则返回诊断文本。

        依赖来源与 Docker 路径一致：优先 env_config.requirements_txt，
        否则回退 required_packages。安装走 `pip install`（国内镜像 +
        find-links，与 EnvBuilder 同源），成功/失败均缓存到进程级
        _INSTALLED_DEPS，避免 smoke/full/优化重跑重复安装。
        """
        env_config = getattr(self, "env_config", None) or {}
        reqs = (env_config.get("requirements_txt") or "").strip()
        if not reqs:
            pkgs = env_config.get("required_packages") or []
            if isinstance(pkgs, list):
                reqs = "\n".join(str(p) for p in pkgs if p).strip()
        if not reqs:
            return None

        key = reqs
        if key in _INSTALLED_DEPS:
            return _INSTALLED_DEPS[key] or None

        req_file = os.path.join(workdir, "requirements.txt")
        with open(req_file, "w", encoding="utf-8") as f:
            f.write(reqs)
        self.log("install_deps", "RUNNING",
                 f"按依赖清单安装环境依赖: {reqs[:120]}...")

        cmd = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check", "-q",
               "-i", PIP_INDEX_URL]
        if PIP_FIND_LINKS:
            cmd += ["--find-links", PIP_FIND_LINKS]
        cmd += ["-r", req_file]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=LOCAL_PIP_TIMEOUT)
            if res.returncode == 0:
                _INSTALLED_DEPS[key] = ""
                self.log("install_deps", "SUCCESS",
                         f"环境依赖安装完成: {reqs[:120]}...")
                return None
            detail = (res.stderr or res.stdout or "").strip()[-800:]
            _INSTALLED_DEPS[key] = (
                f"依赖安装失败(exit={res.returncode}), 无法在本地环境执行: "
                f"{detail}\n依赖清单: {reqs[:200]}...")
        except subprocess.TimeoutExpired:
            _INSTALLED_DEPS[key] = (
                f"依赖安装超时({LOCAL_PIP_TIMEOUT}s), 无法在本地环境执行: "
                f"{reqs[:200]}...")
        except Exception as e:      # 连失败原因都拿不到（如 pip 自身异常）
            _INSTALLED_DEPS[key] = f"依赖安装异常: {e}"
        self.log("install_deps", "ERROR", _INSTALLED_DEPS[key][:200])
        return _INSTALLED_DEPS[key]

    def _execute_code_docker(self, code: str, stage: str,
                             workdir: Optional[str] = None) -> Dict:
        """在 Docker 容器中执行代码（挂载临时目录或指定目录，隔离运行）。

        镜像选择：优先使用 env_config.image_tag（如流水线 EnvBuilder 已构建的
        autorepro-env 镜像，内含 requirements 依赖）；否则退回 python:3.11-slim，
        并把 env_config 中的 requirements 注入容器临时安装后执行。
        """
        docker_cmd = self._resolve_docker_cmd()
        if docker_cmd is None:
            return {"success": False, "stdout": "",
                    "stderr": "本机未安装 Docker 或不在 PATH 中", "exit_code": -3}

        env_config = getattr(self, "env_config", None) or {}
        image = env_config.get("image_tag") or "python:3.11-slim"
        reqs = (env_config.get("requirements_txt") or "").strip()
        if not reqs:
            pkgs = env_config.get("required_packages") or []
            if isinstance(pkgs, list):
                reqs = "\n".join(str(p) for p in pkgs if p).strip()

        cleanup = workdir is None
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="autorepro_docker_")
        else:
            os.makedirs(workdir, exist_ok=True)
        script = os.path.join(workdir, "run.py")
        timeout = DOCKER_TIMEOUT_SMOKE if stage == "smoke" else DOCKER_TIMEOUT_FULL
        try:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            mount = workdir.replace("\\", "/")
            cmd = [docker_cmd, "run", "--rm",
                   "-v", f"{mount}:/app", "-w", "/app"]
            if image == "python:3.11-slim" and reqs:
                with open(os.path.join(workdir, "requirements.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(reqs)
                runner = ["sh", "-c",
                          f"pip install -i {PIP_INDEX_URL} "
                          f"--find-links {PIP_FIND_LINKS} "
                          "-r /app/requirements.txt -q && python run.py"]
            else:
                runner = ["python", "run.py"]
            cmd += [image] + runner
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"执行超时({timeout}s, {stage})", "exit_code": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "exit_code": -2}
        finally:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    # ---------------- 内部工具 ----------------

    def _delta_llm_calls(self) -> int:
        total = self.llm.get_call_count()
        delta = total - getattr(self, "_last_call_count", 0)
        self._last_call_count = total
        return max(delta, 0)

    def extract_result_files(self, result: Dict) -> List[str]:
        """从执行产物中收集数值结果/文件（对齐方案的输出采集）。"""
        files = []
        for artifact in ("stdout", "stderr"):
            text = result.get(artifact, "") or ""
            for line in text.splitlines():
                if "=" in line and any(ch.isdigit() for ch in line):
                    files.append(line.strip())
        return files