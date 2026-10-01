"""复现核心闭环回归测试（对应缺陷修复：缩进丢失 / 截断 / 占位代码 / 三态验证）。

覆盖：
1. `_sanitize_code` 语法兜底不再抹掉缩进（修复"整份代码塌陷成
   IndentationError"的根因）；
2. 语法门 + 再生成：截断的代码被拦下、触发重试、仍不可编译时诚实
   短路为"未运行"，绝不把残码送进沙箱；
3. 信息不足时改为**尽力而为生成并执行**（`best_effort` 标注）：报告里不再
   出现"未运行"、也不判为复现成功；模型两次都不给代码时落本地兜底脚本；
4. PaperReader 标题-only 不再编造占位摘要，透传 insufficient_info；
5. ResultValidator 三态：无法运行 / 复现失败 / 复现成功，且 mse 与
   rmse 不再混键、缺失指标不再静默跳过。

运行: python -m pytest tests/test_reproduction_core.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent, EXIT_NOT_RUNNABLE, MAX_CODE_REGEN,
)
from src.agents.paper_reader import PaperReaderAgent  # noqa: E402
from src.agents.result_validator import (  # noqa: E402
    ResultValidatorAgent, _TOLERANCE,
)
from src.llm.llm_client import LLMClient  # noqa: E402


# ---------------- 测试替身 ----------------

class _ScriptedLLM:
    """按顺序返回预置响应的假 LLM，用于验证"再生成"路径。

    最后一条响应会被重复返回（模拟"重试后仍然一样"的模型）。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.call_count = 0
        self.last_finish_reason = ""

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        self.prompts.append(prompt)
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[idx]

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        self.call_count = 0


@pytest.fixture(autouse=True)
def _clear_deps_cache():
    ce_mod._INSTALLED_DEPS.clear()
    yield
    ce_mod._INSTALLED_DEPS.clear()


TRUNCATED_CODE = (
    "import numpy as np\n"
    "def forward(X, w, b):\n"
    "    z = np.dot(X, w) + b\n"
    "    return z\n"
    "def adam_update(w, m_w, beta1=0.9):\n"
    "    m_w = beta1 * \n"
)
COMPLETE_CODE = (
    "import math\n"
    "def train(epochs=3):\n"
    "    best = 0.0\n"
    "    for _ in range(epochs):\n"
    "        best = best + 0.1\n"
    "    print('accuracy=%.2f' % best)\n"
    "    return best\n"
    "train()\n"
)
# 续写场景的两段：PART1 停在半个表达式上（不可编译），PART2 按续写约定
# **先重写 PART1 的最后一行**再往下写——拼接后应得到一份完整可编译的脚本。
_CONT_PART1 = (
    "import math\n"
    "def train(epochs=3):\n"
    "    best = 0.0\n"
    "    for _ in range(epochs):\n"
    "        best = best + 0.1\n"
    "    print('accuracy=%.2f' % best\n"
)
_CONT_PART2 = (
    "    print('accuracy=%.2f' % best)\n"
    "    return best\n"
    "train()\n"
)


# ============================================================
# 1. _sanitize_code：语法兜底不抹缩进
# ============================================================

class TestSanitizeKeepsIndentation:

    def _agent(self):
        return CodeExecutorAgent(LLMClient(mock_mode=True))

    def test_fallback_path_keeps_indentation(self):
        """不可编译时走兜底分支，函数体缩进必须原样保留。"""
        raw = ("为了复现该论文，我们假设如下。\n"
               "```python\n"
               "def f(x):\n"
               "    return x + 1\n"
               "```\n")
        # 人为制造不可编译（尾部截断），强制走第 3 步兜底
        broken = raw.replace("```\n", "").replace("```python\n", "")
        broken = broken + "def g(y):\n"
        out = self._agent()._sanitize_code(broken)
        assert "def f(x):" in out
        assert "\n    return x + 1" in out, out
        assert "为了" not in out

    def test_line_numbers_stripped_without_losing_indent(self):
        raw = "1 def f(x):\n2     return x\n"
        out = self._agent()._sanitize_code(raw)
        assert out == "def f(x):\n    return x"

    def test_truncated_code_still_reports_real_error(self):
        """截断代码清洗后仍是"未写完"，而非被伪装成 IndentationError。"""
        agent = self._agent()
        out = agent._sanitize_code(TRUNCATED_CODE)
        err = agent._syntax_error(out)
        assert err is not None
        assert agent._looks_truncated(out, err) is True


# ============================================================
# 2. 语法门 + 再生成
# ============================================================

class TestSyntaxGateAndRegeneration:

    def _agent(self, llm):
        return CodeExecutorAgent(llm)

    def test_continuation_completes_truncated_output(self):
        """截断 -> 续写拼接 -> 拼出完整脚本并成功执行。

        这是「代码生成不完整」的主修复路径：模型第一段没写完，不从头重写，
        而是让它从断点续写，拼接后得到完整代码。
        """
        llm = _ScriptedLLM([_CONT_PART1, _CONT_PART2])
        agent = self._agent(llm)
        result = agent.run({"paper_info": {"method": "线性回归",
                                           "dataset": "合成数据"}})
        assert result["success"] is True, result.get("reason")
        assert "accuracy=0.30" in result["final"]["stdout"]
        assert llm.call_count == 2          # 初次生成 + 1 轮续写
        # 拼接后是完整脚本，不是两段的堆叠
        assert result["code"].count("def train(epochs=3):") == 1

    def test_continuation_prompt_asks_to_resume(self):
        """续写 prompt 必须要求"接着写"并重写末行，而不是重写整份。"""
        llm = _ScriptedLLM([_CONT_PART1, _CONT_PART2])
        self._agent(llm).run({"paper_info": {"method": "线性回归",
                                             "dataset": "合成数据"}})
        cont_prompt = llm.prompts[1]
        assert "继续" in cont_prompt
        assert "把上面最后一行完整地重写一遍" in cont_prompt
        assert "不要重写开头" in cont_prompt

    def test_persistent_syntax_error_short_circuits_without_running(self,
                                                                   monkeypatch):
        """续写+重生成都拿不到可用代码 -> 诚实短路为"未运行"，不进沙箱。"""
        def _boom(*a, **kw):
            raise AssertionError("语法错误的代码不得进入沙箱执行")

        monkeypatch.setattr(ce_mod.subprocess, "run", _boom)

        llm = _ScriptedLLM([TRUNCATED_CODE])       # 每次都返回同一份残码
        agent = self._agent(llm)
        result = agent.run({"paper_info": {"method": "线性回归",
                                           "dataset": "合成数据"}})

        assert result["success"] is False
        assert result["not_runnable"] is True
        assert result["final"]["exit_code"] == EXIT_NOT_RUNNABLE
        assert result["final"]["stage"] == "precheck"
        # 初次 + 1 轮续写（发现模型在复述即刻停止，不空烧预算）
        #      + MAX_CODE_REGEN 次从头重生成
        assert llm.call_count == 1 + 1 + MAX_CODE_REGEN

    def test_external_code_is_not_regenerated(self):
        """调用方传入的真实复现代码只清洗，不触发再生成。"""
        llm = _ScriptedLLM([COMPLETE_CODE])
        agent = self._agent(llm)
        result = agent.run({"code": COMPLETE_CODE, "paper_info": {}})
        assert result["success"] is True
        assert llm.call_count == 0


class TestExecutionRepair:

    def _agent(self, llm):
        return CodeExecutorAgent(llm)

    def test_runtime_error_is_repaired_and_rerun(self):
        broken = "print(missing_value)\n"
        fixed = "print('repaired')\n"
        llm = _ScriptedLLM([fixed])
        result = CodeExecutorAgent(llm).run({
            "code": broken,
            "paper_info": {"method": "演示方法", "dataset": "合成数据"},
        })

        assert result["success"] is True, result
        assert result["code"] == fixed.strip()
        assert result["final"]["stdout"].strip() == "repaired"
        assert len(result["repair_attempts"]) == 1
        assert result["repair_attempts"][0]["diagnosis"]["line"] == 1
        assert "NameError" in result["repair_attempts"][0]["diagnosis"]["message"]
        assert "失败位置" in llm.prompts[0]

    def test_unrepairable_timeout_does_not_call_llm(self, monkeypatch):
        agent = CodeExecutorAgent(_ScriptedLLM(["print('must not use')"]))

        monkeypatch.setattr(
            agent, "_execute_code",
            lambda code, stage: {
                "success": False, "stdout": "", "stderr": "执行超时(10s, smoke)",
                "exit_code": -1,
            },
        )
        result = agent.run({"code": "print('x')", "paper_info": {}})

        assert result["success"] is False
        assert result["repair_attempts"][0]["status"] == "stopped"
        assert agent.llm.call_count == 0

    def test_repair_result_is_checked_for_dangerous_code(self):
        llm = _ScriptedLLM(["import os\nos.system('echo unsafe')\n"])
        result = CodeExecutorAgent(llm).run({
            "code": "print(missing_value)\n", "paper_info": {},
        })

        assert result["success"] is False
        assert result["repair_attempts"][0]["status"] == "stopped"
        assert "危险调用" in result["repair_attempts"][0]["diagnosis"]["repair_error"]

    def test_external_broken_code_reports_syntax_error(self):
        llm = _ScriptedLLM([COMPLETE_CODE])
        agent = self._agent(llm)
        result = agent.run({"code": "def f(:\n  pass\n"})
        assert result["not_runnable"] is True
        assert llm.call_count == 0


# ============================================================
# 3. 信息不足：尽力而为生成并执行（best_effort）
# ============================================================

class TestInsufficientInfoBestEffort:
    """用户要求「一定要尝试生成代码并允许运行」——这条线不许再交白卷。

    此前的行为是"信息不足 → 不生成 → 未运行"（报告里 0 字符 + exit_code=-5）。
    """

    def test_missing_method_and_dataset_still_generates_and_runs(self):
        llm = _ScriptedLLM([COMPLETE_CODE])
        agent = CodeExecutorAgent(llm)
        result = agent.run({"paper_info": {"method": "",
                                           "dataset": "未知",
                                           "metrics": {}}})

        assert not result.get("not_runnable")     # 不再短路"未运行"
        assert result["best_effort"] is True
        assert result["success"] is True
        assert len(result["code"]) > 0
        assert llm.call_count == 1                # 真的让模型生成了一次
        # 真进了沙箱：阶段是 smoke+full，而不是被拦在 precheck（旧行为只有一个
        # precheck 阶段 + exit_code=-5）
        assert [st["stage"] for st in result["stages"]] == ["smoke", "full"]

    def test_insufficient_flag_from_paper_reader_still_generates(self):
        llm = _ScriptedLLM([COMPLETE_CODE])
        agent = CodeExecutorAgent(llm)
        result = agent.run({"paper_info": {"insufficient_info": True,
                                           "method": "某种方法",
                                           "dataset": "某数据集"}})
        assert not result.get("not_runnable")
        assert result["best_effort"] is True
        assert result["success"] is True
        assert llm.call_count == 1

    def test_sufficient_info_is_not_best_effort(self):
        llm = _ScriptedLLM([COMPLETE_CODE])
        agent = CodeExecutorAgent(llm)
        result = agent.run({"paper_info": {"method": "梯度下降",
                                           "dataset": "二维二分类"}})
        assert result["success"] is True
        assert result["best_effort"] is False
        assert result["best_effort_reason"] == ""
        assert result["fallback_used"] is False

    def test_external_code_is_never_best_effort(self):
        """调用方给的真实代码不因论文信息不足被改判——它不是我们的占位实现。"""
        agent = CodeExecutorAgent(_ScriptedLLM([COMPLETE_CODE]))
        result = agent.run({"code": COMPLETE_CODE,
                            "paper_info": {"insufficient_info": True}})
        assert result["best_effort"] is False
        assert result["success"] is True


class TestPlaceholderRecovery:
    """模型"拒答"（只回占位标记/空）时的定向重试与本地兜底。"""

    def test_marker_triggers_one_targeted_retry_then_real_code(self):
        llm = _ScriptedLLM([ce_mod._INSUFFICIENT_INFO_MARK, COMPLETE_CODE])
        result = CodeExecutorAgent(llm).run(
            {"paper_info": {"method": "", "dataset": "", "metrics": {}}})

        # 1 次初生成 + MAX_MARK_RETRY 次定向重试；旧实现这条路径会走到
        # 1+3+2 次空烧（占位标记被当成"没写完"反复追问）
        assert llm.call_count == 1 + ce_mod.MAX_MARK_RETRY
        assert result["fallback_used"] is False
        assert result["success"] is True
        assert "上一次只回复了占位标记" in llm.prompts[-1]

    def test_marker_twice_falls_back_to_local_script(self):
        llm = _ScriptedLLM([ce_mod._INSUFFICIENT_INFO_MARK])
        result = CodeExecutorAgent(llm).run(
            {"paper_info": {"method": "", "dataset": "", "metrics": {}}})

        assert llm.call_count == 1 + ce_mod.MAX_MARK_RETRY   # 严格锁预算
        assert result["fallback_used"] is True
        assert result["code"] == ce_mod._BEST_EFFORT_SCRIPT
        assert result["success"] is True                     # 兜底脚本真的跑通
        assert "不是论文结论" in result["final"]["stdout"]
        assert len(result["assumptions"]) >= 3

    def test_empty_response_falls_back(self):
        llm = _ScriptedLLM([""])
        result = CodeExecutorAgent(llm).run({"paper_info": {}})
        assert result["fallback_used"] is True
        assert result["success"] is True

    def test_fallback_script_passes_all_gates(self):
        """兜底脚本本身必须过语法门/危险门/结构完整性，否则它自己就被拦了。"""
        agent = CodeExecutorAgent(_ScriptedLLM([COMPLETE_CODE]))
        bs = ce_mod._BEST_EFFORT_SCRIPT
        assert agent._syntax_error(bs) is None
        assert agent._dangerous_constructs(bs) is None
        assert CodeExecutorAgent._structurally_complete(bs) is True
        assert CodeExecutorAgent._is_placeholder_code(bs) is False

    def test_fallback_output_has_no_extractable_metrics(self):
        """兜底脚本的输出不能被抽成"实测指标"——否则会喂出假的复现结论。"""
        llm = _ScriptedLLM([ce_mod._INSUFFICIENT_INFO_MARK])
        result = CodeExecutorAgent(llm).run({"paper_info": {}})
        stdout = result["final"]["stdout"]
        assert "占位复现脚本" in stdout
        validator = ResultValidatorAgent(LLMClient(mock_mode=True))
        assert validator._extract_metrics(stdout) == {}


# ============================================================
# 4. PaperReader 标题-only 诚实降级
# ============================================================

class _EchoTitleLLM:
    """返回"什么都没有推断出来"的论文 JSON，模拟标题-only 场景。"""

    def __init__(self):
        self.prompts = []
        self.call_count = 0

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        self.prompts.append(prompt)
        self.call_count += 1
        return ('{"title": "某篇论文", "authors": [], "method": "", '
                '"dependencies": [], "metrics": {}, "dataset": "", '
                '"code_url": "未找到", "insufficient_info": true}')

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        self.call_count = 0


class TestPaperReaderHonestDegradation:

    def test_no_fabricated_abstract_in_prompt(self):
        llm = _EchoTitleLLM()
        agent = PaperReaderAgent(llm)
        agent.run({"paper_title": "Some Paper Title"})
        prompt = llm.prompts[0]
        assert "Some Paper Title" in prompt
        assert "包含方法、实验与指标声明" not in prompt   # 旧占位摘要已删除
        assert "未获取到论文正文" in prompt              # 如实标注只有标题

    def test_insufficient_flag_propagates(self):
        agent = PaperReaderAgent(_EchoTitleLLM())
        result = agent.run({"paper_title": "Some Paper Title"})
        info = result["paper_info"]
        assert info["insufficient_info"] is True
        assert info["info_sufficient"] is False
        assert info["title"] == "Some Paper Title"     # 用户标题优先

    def test_no_information_loss_when_title_has_colon(self):
        agent = PaperReaderAgent(_EchoTitleLLM())
        title = "机器学习上机实验10：梯度下降"
        result = agent.run({"paper_title": title})
        assert result["paper_info"]["title"] == title


# ============================================================
# 5. ResultValidator 三态
# ============================================================

def _validator():
    return ResultValidatorAgent(LLMClient(mock_mode=True))


def _stub_llm(match: bool, confidence: float) -> LLMClient:
    """固定判定的假 LLM：把"模型侧结论"与"本地数值结论"解耦，便于测分歧。"""
    class _Stub(LLMClient):
        def chat(self, prompt, **kw):        # type: ignore[override]
            return json.dumps({"match": match, "differences": [],
                               "confidence": confidence, "analysis": "stub"})

    return _Stub(mock_mode=True)


def _ran_execution(stdout: str) -> dict:
    """构造"确实跑起来了"的 execution（结构与 CodeExecutor 输出一致）。"""
    full = {"stage": "full", "success": True, "stdout": stdout,
            "stderr": "", "exit_code": 0}
    return {"success": True, "stages": [full], "final": full}


class TestValidatorThreeStates:

    def test_not_runnable_reports_cannot_verify(self):
        agent = _validator()
        execution = {"not_runnable": True, "reason": "论文信息不足",
                     "success": False, "stages": [], "final": {}}
        result = agent.run({"paper_info": {"metrics": {"accuracy": 0.85}},
                            "execution": execution})
        assert result["is_reproduced"] is None
        assert result["status"] == "not_runnable"
        assert "未能运行" in result["validation"]["analysis"]
        # 未运行 -> 不调 LLM 比对，省预算
        assert result["llm_calls"] == 0

    def test_failed_execution_is_not_runnable(self):
        """语法错误/依赖失败：没有 stdout 的失败 = 没跑起来。"""
        agent = _validator()
        execution = {"success": False,
                     "final": {"stage": "smoke", "success": False,
                               "stdout": "", "stderr": "IndentationError: ...",
                               "exit_code": 1},
                     "stages": [{"stage": "smoke", "success": False}]}
        result = agent.run({"paper_info": {"metrics": {"accuracy": 0.85}},
                            "execution": execution})
        assert result["status"] == "not_runnable"
        assert "IndentationError" in result["reason"]

    _ran = staticmethod(_ran_execution)

    def test_ran_but_mismatched_is_not_reproduced(self):
        agent = _validator()
        result = agent.run({"paper_info": {"metrics": {"accuracy": 0.85}},
                            "execution": self._ran("accuracy: 0.42")})
        assert result["is_reproduced"] is False
        assert result["status"] == "not_reproduced"

    def test_matched_is_reproduced(self):
        agent = _validator()
        result = agent.run({"paper_info": {"metrics": {"accuracy": 0.85}},
                            "execution": self._ran("Test accuracy: 85.2%")})
        assert result["is_reproduced"] is True
        assert result["status"] == "reproduced"


class TestValidatorBestEffort:
    """第四态：代码确实跑了，但论文信息不足、代码是占位实现——既不判成功也不判失败。"""

    def test_best_effort_is_not_reported_as_reproduced(self):
        """占位脚本即便打印出可抽取的数值，也不能被读成"复现成功"。

        不拦的话 `_local_compare` 对"无论文声明指标"是乐观判定（跑出数值即
        match=True）——报告会显示假的 ✅ 成功，而且会真去触发优化。
        """
        agent = _validator()
        execution = {**_ran_execution("accuracy: 0.85"),
                     "best_effort": True,
                     "best_effort_reason": "论文未提供可用的方法/数据集/声明指标"}
        result = agent.run({"paper_info": {"insufficient_info": True,
                                           "metrics": {}},
                            "execution": execution})

        assert result["status"] == "best_effort"
        assert result["is_reproduced"] is None
        assert result["llm_calls"] == 0            # 跳过 LLM 比对，省预算
        assert "占位" in result["reason"]
        # 证据仍留：报告要显示"确实跑出了什么"，只是标注为不可核对
        assert result["metrics_comparison"]["actual"] == {"accuracy": 0.85}

    def test_insufficient_paper_info_alone_marks_best_effort(self):
        """execution 没带标注（旧数据/旁路调用）时，论文侧的信息不足也足以判定。"""
        agent = _validator()
        result = agent.run({"paper_info": {"insufficient_info": True},
                            "execution": _ran_execution("loss: 0.1")})
        assert result["status"] == "best_effort"

    def test_not_runnable_takes_precedence_over_best_effort(self):
        """真没跑 > 跑了但不可核对：两者同时为真时报"未运行"。"""
        agent = _validator()
        execution = {"not_runnable": True, "reason": "语法错误",
                     "best_effort": True, "success": False,
                     "stages": [], "final": {}}
        result = agent.run({"paper_info": {"insufficient_info": True},
                            "execution": execution})
        assert result["status"] == "not_runnable"


class TestValidatorMetricDetails:

    def test_rmse_not_folded_into_mse(self):
        m = _validator()._extract_metrics("rmse: 1.234, mse: 1.5")
        assert m["rmse"] == pytest.approx(1.234)
        assert m["mse"] == pytest.approx(1.5)

    def test_missing_metric_is_reported_not_silently_skipped(self):
        cmp = _validator()._local_compare(
            {"accuracy": 0.85, "f1_score": 0.80}, {"accuracy": 0.85})
        assert cmp["match"] is True              # 部分匹配仍算通过
        assert cmp["missing_metrics"]            # 但缺失项要如实记录
        assert "f1_score" in cmp["missing_metrics"][0]
        assert any("f1_score" in d for d in cmp["differences"])

    def test_no_overlapping_metric_is_not_reproduced(self):
        """声明指标一个都没对上 -> 不得判为复现成功。"""
        cmp = _validator()._local_compare(
            {"reproduction_score": 0.85}, {"accuracy": 0.85})
        assert cmp["match"] is False


class TestValidatorKeyNormalization:
    """键名大小写/分隔符差异不得导致误判（真实模式实测）。

    实测：论文声明 `{"MSE": 0.0892}`，脚本打印 `MSE: 0.0869`，提取成键
    `mse`。原先的精确比对判"未提取到该指标"，把 2.6% 的差异（阈值 5%）
    误报成未复现，且理由写反。
    """

    def test_uppercase_declared_key_matches_lowercase_actual(self):
        agent = _validator()
        actual = agent._extract_metrics("学到的偏置 b: 0.5012\nMSE: 0.0869\n")
        assert actual == {"mse": pytest.approx(0.0869)}
        cmp = agent._local_compare({"MSE": 0.0892}, actual)
        assert cmp["match"] is True, cmp["differences"]
        assert not cmp["missing_metrics"]
        assert "2.6%" in cmp["differences"][0]

    def test_real_e2e_shape_verdict_is_reproduced(self):
        """端到端复现真实模式那一跑的判定结果。"""
        stdout = ("数据形状: X=(1000, 2), y=(1000,)\n"
                  "学到的权重 w: [ 1.50833009 -1.99485473]\n"
                  "学到的偏置 b: 0.5012\nMSE: 0.0869\n")
        agent = ResultValidatorAgent(_stub_llm(match=True, confidence=0.9))
        result = agent.run({"paper_info": {"metrics": {"MSE": 0.0892}},
                            "execution": _ran_execution(stdout)})
        assert result["status"] == "reproduced"
        assert result["is_reproduced"] is True
        assert result["metrics_comparison"]["actual"]["mse"] == pytest.approx(0.0869)
        assert result["confidence"] == pytest.approx(0.9)

    @pytest.mark.parametrize("declared", ["f1_score", "F1_score", "f1-score",
                                          "F1 Score", " f1_score "])
    def test_separator_and_case_variants_all_match(self, declared):
        cmp = _validator()._local_compare({declared: 0.80}, {"f1_score": 0.80})
        assert cmp["match"] is True, cmp["differences"]

    def test_distinct_metrics_are_not_conflated(self):
        """归一化不得把不同指标混为一谈（rmse 与 mse 必须各自比对）。"""
        cmp = _validator()._local_compare({"rmse": 0.9, "mse": 0.9},
                                          {"rmse": 0.9, "mse": 0.1})
        assert cmp["match"] is False
        assert any(d.startswith("mse:") for d in cmp["differences"])


class TestValidatorVerdictHonesty:
    """两个判据结论相反时，汇报必须自洽（不能报"未复现 且 置信度 1.0"）。"""

    @staticmethod
    def _run_with_llm(llm_match: bool, conf: float, stdout: str,
                      paper_metrics: dict):
        agent = ResultValidatorAgent(_stub_llm(llm_match, conf))
        return agent.run({"paper_info": {"metrics": paper_metrics},
                          "execution": _ran_execution(stdout)})

    def test_disagreement_is_reported_and_confidence_lowered(self):
        """LLM 说匹配、本地数值说不匹配 -> 以交集为准，且写明分歧。"""
        result = self._run_with_llm(llm_match=True, conf=1.0,
                                    stdout="accuracy: 0.42", paper_metrics={"accuracy": 0.85})
        assert result["is_reproduced"] is False
        assert result["confidence"] <= 0.4, "分歧时不得沿用模型的高置信度"
        assert result["validation"]["verdict_sources"] == {"llm": True, "local": False}
        assert any("判据分歧" in d for d in result["validation"]["differences"])

    def test_agreement_keeps_llm_confidence(self):
        result = self._run_with_llm(llm_match=True, conf=0.9,
                                    stdout="Test accuracy: 85.2%",
                                    paper_metrics={"accuracy": 0.85})
        assert result["is_reproduced"] is True
        assert result["confidence"] == pytest.approx(0.9)
        assert not any("判据分歧" in d for d in result["validation"]["differences"])


class TestValidatorPromptCarriesCriteria:
    """比对准则必须写进 prompt——否则模型凭感觉判，同样的数字会随机横跳。

    实测：声明 0.0892 / 实际 0.0869 固定输入下，模型在 true/false 之间
    来回变（账本里 12:58 false、13:00 true、13:01 true、13:02 false）；
    补上判据后重复 5 次稳定判为一致。这里锁住判据不被重新删掉。
    """

    @staticmethod
    def _prompt() -> str:
        captured = {}

        class _Capture(LLMClient):
            def chat(self, prompt, **kw):        # type: ignore[override]
                captured["prompt"] = prompt
                return json.dumps({"match": True, "confidence": 0.5,
                                   "differences": [], "analysis": ""})

        ResultValidatorAgent(_Capture(mock_mode=True)).run(
            {"paper_info": {"metrics": {"MSE": 0.0892}},
             "execution": _ran_execution("MSE: 0.0869\n")})
        return captured["prompt"]

    def test_tolerance_ratio_is_stated(self):
        """阈值直接由 _TOLERANCE 插值，改阈值时 prompt 跟着走。"""
        assert f"{_TOLERANCE:.0%}" in self._prompt()

    def test_states_all_declared_metrics_must_match(self):
        prompt = self._prompt()
        assert "所有声明指标都一致" in prompt

    def test_warns_declared_value_in_stdout_is_not_the_result(self):
        """复现脚本常自己打印一行论文声明值做对照，不得被当成运行结果。"""
        assert "不得" in self._prompt()

    def test_states_case_insensitive_metric_names(self):
        assert "大小写" in self._prompt()
