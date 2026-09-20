"""代码生成完整性测试：截断续写拼接 / 保真清洗 / 结构完整性门。

对应缺陷：「LLM 生成的复现代码不完整，跑不起来」。修复思路是
**截断时续写拼接**（突破单次调用 max_tokens 上限）而不是用同一个 prompt
从头重写（重写只会再撞一次），并让清洗层不再静默吞代码。

覆盖：
1. 续写拼接：首段截断 -> 续写 -> 拼出完整可编译脚本；接缝去重不产生重复行；
2. 截断信号：finish_reason=length（is_truncated）触发续写；
3. 不空转：模型复述已有内容 / 错误没改善时立即停止续写，不白烧预算；
4. 清洗保真：带围栏的完整脚本原样返回（缩进、中文 print 都不丢）；
5. 清洗不再误删含中文的合法代码行，但纯叙述行仍会被丢弃；
6. 清洗丢弃行数被如实统计上报（不再静默）；
7. 结构完整性门：只有 def/import、不做任何事的脚本被判为不完整；
8. 提示词不再诱导写短、要求围栏、允许中文字符串。

运行: python -m pytest tests/test_code_generation_completeness.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents.code_executor as ce_mod  # noqa: E402
from src.agents.code_executor import (  # noqa: E402
    CodeExecutorAgent,
    MAX_CODE_CONTINUE,
    MAX_CODE_REGEN,
)
from src.llm.llm_client import LLMClient  # noqa: E402


class _ScriptedLLM:
    """按顺序返回预置响应的假 LLM；最后一条会被重复返回。"""

    def __init__(self, responses, finish_reason=""):
        self.responses = list(responses)
        self.prompts = []
        self.call_count = 0
        self.last_finish_reason = finish_reason

    def chat(self, prompt, system_prompt="", temperature=0.3, task=""):
        self.prompts.append(prompt)
        self.call_count += 1
        return self.responses[min(self.call_count - 1, len(self.responses) - 1)]

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self) -> None:
        self.call_count = 0

    def is_truncated(self) -> bool:
        return self.last_finish_reason == "length"


@pytest.fixture(autouse=True)
def _clear_deps_cache():
    ce_mod._INSTALLED_DEPS.clear()
    yield
    ce_mod._INSTALLED_DEPS.clear()


def _agent(llm):
    return CodeExecutorAgent(llm)


# 第一段停在半个表达式（未闭合括号），第二段按续写约定先重写末行再往下写
_PART1 = (
    "import math\n"
    "def train(epochs=3):\n"
    "    best = 0.0\n"
    "    for _ in range(epochs):\n"
    "        best = best + 0.1\n"
    "    print('accuracy=%.2f' % best\n"
)
_PART2 = (
    "    print('accuracy=%.2f' % best)\n"
    "    return best\n"
    "train()\n"
)

_COMPLETE = (
    "import math\n"
    "def train(epochs=3):\n"
    "    best = 0.0\n"
    "    for _ in range(epochs):\n"
    "        best = best + 0.1\n"
    "    print('accuracy=%.2f' % best)\n"
    "    return best\n"
    "train()\n"
)

# 只有定义、没有任何顶层调用：能编译但跑起来什么也不产出
_INERT_ONLY = (
    "import math\n"
    "def train(epochs=3):\n"
    "    return 0.3\n"
)


# ============================================================
# 1. 续写拼接
# ============================================================

class TestContinuationStitching:

    def test_truncated_output_is_completed_by_continuation(self):
        llm = _ScriptedLLM([_PART1, _PART2])
        result = _agent(llm).run({"paper_info": {"method": "线性回归",
                                                 "dataset": "合成数据"}})
        assert result["success"] is True, result.get("reason")
        assert "accuracy=0.30" in result["final"]["stdout"]
        assert llm.call_count == 2          # 初次 + 1 轮续写

    def test_stitch_does_not_duplicate_seam_line(self):
        """接缝处只保留续写重写后的那一行，不出现重复/半行。"""
        llm = _ScriptedLLM([_PART1, _PART2])
        result = _agent(llm).run({"paper_info": {"method": "线性回归",
                                                 "dataset": "合成数据"}})
        code = result["code"]
        assert code.count("print('accuracy=%.2f' % best)") == 1
        assert code.count("def train(epochs=3):") == 1
        assert "print('accuracy=%.2f' % best\n" not in code   # 半句已被替换

    def test_stitch_dedupes_repeated_context(self):
        """续写若多复述了上文若干行，去重后不产生重复。"""
        head = "def f():\n    a = 1\n    b = 2\n"
        tail = "    a = 1\n    b = 2\n    return a + b\n"
        out = CodeExecutorAgent._stitch(head, tail)
        assert out.count("    a = 1") == 1
        assert out.count("    b = 2") == 1
        assert "return a + b" in out

    def test_continuation_preserves_first_line_indentation(self):
        """续写片段首行是缩进行时，拼接后缩进不能被吃掉。"""
        code, stats = _agent(_ScriptedLLM([""]))._sanitize_code_ex(_PART2)
        assert code.startswith("    print("), repr(code)
        assert stats["code_dropped"] == 0

    def test_stitch_keeps_complete_last_line(self):
        """模型没按约定重写末行时，head 的完整末行不能被吞掉。"""
        head = "def f():\n    total = 0\n"
        tail = "    return total\n"
        out = CodeExecutorAgent._stitch(head, tail)
        assert "    total = 0" in out
        assert "    return total" in out

    def test_stitch_drops_dangling_last_line(self):
        """末行确实没写完（悬挂运算符/未闭合括号）时，才交给续写重写。"""
        assert CodeExecutorAgent._line_incomplete("    m_w = beta1 *") is True
        assert CodeExecutorAgent._line_incomplete(
            "    print('%s' % best") is True          # 少右括号
        # 完整的块首行 / 普通语句不算没写完
        assert CodeExecutorAgent._line_incomplete("    for i in range(3):") is False
        assert CodeExecutorAgent._line_incomplete("    x = 1") is False


# ============================================================
# 2. 截断信号 / 不空转
# ============================================================

class TestTruncationSignals:

    def test_finish_reason_length_triggers_continuation(self):
        """finish_reason=length 是 API 给出的确定性截断信号即便代码能编译。"""
        llm = _ScriptedLLM([_COMPLETE, _PART2], finish_reason="length")
        agent = _agent(llm)
        assert agent._needs_continuation(
            agent._sanitize_code(_COMPLETE)) is True

    def test_stop_finish_reason_and_compilable_code_needs_no_continuation(self):
        llm = _ScriptedLLM([_COMPLETE], finish_reason="stop")
        agent = _agent(llm)
        assert agent._needs_continuation(
            agent._sanitize_code(_COMPLETE)) is False

    def test_repeating_model_does_not_burn_budget(self):
        """模型每次返回同一份残码 -> 一轮续写后立刻停，不空转到上限。"""
        llm = _ScriptedLLM([_PART1])        # 永远返回第一段
        result = _agent(llm).run({"paper_info": {"method": "线性回归",
                                                 "dataset": "合成数据"}})
        assert result["not_runnable"] is True
        # 初次 + 1 轮续写（发现复述即停）+ MAX_CODE_REGEN 次重生成
        assert llm.call_count == 1 + 1 + MAX_CODE_REGEN
        assert llm.call_count < 1 + MAX_CODE_CONTINUE + MAX_CODE_REGEN


# ============================================================
# 3. 清洗保真 + 丢弃可见
# ============================================================

class TestSanitizeIsLossless:

    def _agent(self):
        return CodeExecutorAgent(LLMClient(mock_mode=True))

    def test_fenced_script_returned_verbatim(self):
        """带围栏的完整脚本原样返回——缩进与中文 print 都不丢。"""
        body = ("import math\n"
                "def f(x):\n"
                "    print(f\"准确率: {x:.4f}\")\n"
                "    return x\n")
        raw = f"这是说明文字，应该被忽略。\n```python\n{body}```\n"
        code, stats = self._agent()._sanitize_code_ex(raw)
        assert code == body.strip("\n")
        assert stats == {"prose_dropped": 0, "code_dropped": 0}
        assert "准确率" in code

    def test_chinese_print_is_not_dropped(self):
        """无围栏时，含中文的合法代码行不能被当成叙述行删掉。"""
        raw = ("import math\n"
               "def f(x):\n"
               "    print(f\"准确率: {x:.4f}\")\n"
               "    return x\n"
               "f(0.85)\n")
        code, stats = self._agent()._sanitize_code_ex(raw)
        assert "准确率" in code
        assert stats["code_dropped"] == 0

    def test_prose_lines_are_dropped_and_counted(self):
        """纯中文叙述行仍会被丢弃，且计入 prose_dropped。"""
        raw = ("为了复现该论文，我们采用如下配置。\n"
               "import math\n"
               "下面是训练函数：\n"
               "def f(x):\n"
               "    return x\n"
               "f(1)\n")
        code, stats = self._agent()._sanitize_code_ex(raw)
        assert "为了复现" not in code
        assert "下面是训练函数" not in code
        assert stats["prose_dropped"] == 2
        assert stats["code_dropped"] == 0       # 叙述不算"洗残代码"
        assert "import math" in code

    def test_dropped_code_lines_are_reported_not_silent(self):
        """丢掉疑似代码行时会写 WARNING，不再静默。"""
        agent = self._agent()
        agent._record_sanitize("测试", "import math\n",
                               {"prose_dropped": 0, "code_dropped": 3})
        entries = [e for e in agent.logger.get_summary()
                   if e.get("action") == "sanitize_code"]
        assert entries and entries[-1]["status"] == "WARNING"
        assert entries[-1]["data"]["code_dropped"] == 3


# ---- 实测过的三条"静默篡改"回归用例：内容必须一字不少 ----

class TestNoSilentLineDeletion:
    """这些输入里的合法代码行曾被白名单静默删掉，且删完仍能编译。

    典型后果：`*x, y = [1, 2, 3]` 被删 -> 沙箱里 NameError；
    docstring 内容被删 -> 字符串被悄悄改写。
    """

    @staticmethod
    def _clean(raw):
        return CodeExecutorAgent(LLMClient(mock_mode=True)) \
            ._sanitize_code_ex(raw)

    def test_star_unpacking_line_survives(self):
        raw = "a = 1\n*x, y = [1, 2, 3]\nprint(y)\n"
        code, stats = self._clean(raw)
        assert code == raw.strip("\n")
        assert stats["code_dropped"] == 0

    def test_docstring_content_survives(self):
        raw = ('def f():\n'
               '    """说明\n'
               '    ) paren\n'
               '    * bullet\n'
               '    """\n'
               '    return 1\n'
               'f()\n')
        code, stats = self._clean(raw)
        assert ") paren" in code
        assert "* bullet" in code
        assert stats["code_dropped"] == 0

    def test_multiline_call_closing_bracket_survives(self):
        raw = ("def load(a, b):\n"
               "    return a + b\n"
               "data = load(\n"
               "    1,\n"
               "    2,\n"
               ")\n"
               "print(data)\n")
        code, stats = self._clean(raw)
        assert "\n)\n" in code
        assert stats["code_dropped"] == 0

    def test_deleted_code_line_triggers_remediation(self):
        """若保真清洗后仍编译不过、只能丢行，必须触发补救。"""
        agent = CodeExecutorAgent(LLMClient(mock_mode=True))
        stats = {"prose_dropped": 0, "code_dropped": 2}
        assert agent._needs_continuation("print(1)\n", stats) is True

    def test_prose_only_drop_does_not_trigger_remediation(self):
        """只剥叙述行是预期行为，不该触发续写、也不该告警。"""
        agent = CodeExecutorAgent(LLMClient(mock_mode=True))
        stats = {"prose_dropped": 5, "code_dropped": 0}
        assert agent._needs_continuation("print(1)\n", stats) is False
        agent._record_sanitize("测试", "print(1)\n", stats)
        entries = [e for e in agent.logger.get_summary()
                   if e.get("action") == "sanitize_code"]
        assert entries and entries[-1]["status"] == "RUNNING"


# ============================================================
# 6. 报告如实展示清洗记录
# ============================================================

class TestReportSurfacesSanitize:

    @staticmethod
    def _report(execution: dict) -> str:
        from src.agents.report_generator import ReportGeneratorAgent
        return ReportGeneratorAgent().run(
            {"execution": execution, "paper_info": {}})["report"]

    def test_code_drop_is_flagged_in_report(self):
        report = self._report({
            "code": "a = 1\n", "stages": [], "final": {},
            "sanitize_stats": {"prose_dropped": 2, "code_dropped": 3},
        })
        assert "⚠️ 清洗丢弃" in report
        assert "3 行疑似代码行" in report

    def test_prose_only_drop_is_not_flagged_as_danger(self):
        report = self._report({
            "code": "a = 1\n", "stages": [], "final": {},
            "sanitize_stats": {"prose_dropped": 2, "code_dropped": 0},
        })
        assert "⚠️ 清洗丢弃" not in report
        assert "叙述行 2 行" in report

    def test_no_sanitize_stats_no_noise(self):
        report = self._report({"code": "a = 1\n", "stages": [], "final": {}})
        assert "清洗丢弃" not in report

    def test_report_embeds_full_output_without_truncation(self):
        """报告全文内嵌代码与执行输出，不截断。

        此前 `_clip` 把 code/stdout/stderr 砍到 6000/6000/3000 字符，实测
        一次 10627 字符的输出在报告里只剩前 6000——读报告的人拿到半截内容。
        代码块现已由前端渲染成可滚动面板，长度不再有排版代价。
        """
        code = "# 生成代码\n" + "\n".join(
            f"print('step {i}')" for i in range(400))          # ≈ 8000 字符
        stdout = "\n".join(f"result[{i}] = {i * 1.5:.3f}"
                           for i in range(900))                # ≈ 20000 字符
        report = self._report({
            "code": code, "stages": [], "final": {"stdout": stdout},
        })

        assert code.splitlines()[-1] in report       # 代码末行必须在
        assert stdout.splitlines()[-1] in report     # 输出末行必须在
        assert "截断" not in report
        assert len(report) > len(code) + len(stdout)


# ============================================================
# 7. 报告如实展示验证结论的依据
# ============================================================

class TestReportSurfacesVerdictBasis:

    @staticmethod
    def _report(validation: dict) -> str:
        from src.agents.report_generator import ReportGeneratorAgent
        return ReportGeneratorAgent().run(
            {"validation": validation, "execution": {}, "paper_info": {}})["report"]

    def test_metric_differences_are_listed(self):
        """只给"失败"结论、不给逐项差异，用户无从判断判定是否合理。"""
        report = self._report({
            "status": "not_reproduced", "is_reproduced": False,
            "confidence": 0.4,
            "validation": {
                "analysis": "本地数值比对完成",
                "differences": ["MSE: 声明 0.0892 vs 实际 0.0869 "
                                "(归一化后相对差异 2.6%)"],
            },
            "metrics_comparison": {"paper": {"MSE": 0.0892},
                                   "actual": {"mse": 0.0869}},
        })
        assert "指标差异" in report
        assert "2.6%" in report

    def test_metric_table_pairs_case_variant_keys(self):
        """论文声明 MSE、运行输出 mse 是同一个指标，必须排成一行。

        按精确键名配对会排成两行、各缺一半，看起来像"没跑出来"——
        与判定层误报未复现是同一个 bug 的两种表现。
        """
        report = self._report({
            "status": "reproduced", "is_reproduced": True, "confidence": 0.9,
            "validation": {"analysis": "ok", "differences": []},
            "metrics_comparison": {"paper": {"MSE": 0.0892},
                                   "actual": {"mse": 0.0869}},
        })
        rows = [ln for ln in report.splitlines()
                if ln.startswith("| ") and ("0.0892" in ln or "0.0869" in ln)]
        assert len(rows) == 1, f"同一指标应只占一行，实际:\n{report}"
        assert "0.0892" in rows[0] and "0.0869" in rows[0]

    def test_unmatched_metrics_still_shown_as_na(self):
        report = self._report({
            "status": "not_reproduced", "is_reproduced": False,
            "confidence": 0.4,
            "validation": {"analysis": "缺指标",
                           "missing_metrics": ["f1_score: 声明 0.8，未提取到"]},
            "metrics_comparison": {"paper": {"accuracy": 0.85, "f1_score": 0.8},
                                   "actual": {"accuracy": 0.85}},
        })
        assert "无法比对的指标" in report
        assert "| f1_score | 0.8 | N/A |" in report


# ============================================================
# 4. 结构完整性门
# ============================================================

class TestStructuralCompleteness:

    def test_defs_only_script_is_incomplete(self):
        assert CodeExecutorAgent._structurally_complete(_INERT_ONLY) is False

    def test_imports_only_script_is_incomplete(self):
        assert CodeExecutorAgent._structurally_complete("import math\n") is False

    def test_docstring_only_script_is_incomplete(self):
        assert CodeExecutorAgent._structurally_complete('"""说明"""\n') is False

    @pytest.mark.parametrize("code", [
        _COMPLETE,
        "def f():\n    return 1\n\n\nf()\n",
        "if __name__ == '__main__':\n    print('ok')\n",
        "print('Training complete. Test accuracy: 85.2%')\n",
    ])
    def test_executable_scripts_are_complete(self, code):
        assert CodeExecutorAgent._structurally_complete(code) is True

    def test_inert_script_triggers_continuation(self):
        """只有定义没有执行的脚本，应被判为"还没写完"从而触发续写。"""
        llm = _ScriptedLLM([_INERT_ONLY, ""], finish_reason="stop")
        agent = CodeExecutorAgent(llm)
        assert agent._needs_continuation(_INERT_ONLY) is True


# ============================================================
# 5. 提示词
# ============================================================

class TestPrompts:

    def _prompt(self):
        return CodeExecutorAgent(LLMClient(mock_mode=True)) \
            ._generate_code_prompt({"method": "线性回归",
                                    "dataset": "合成数据",
                                    "metrics": {"accuracy": 0.85}})

    def test_prompt_no_longer_asks_for_short_code(self):
        """旧提示词开头是「生成一段**简短的**训练代码」，直接诱导模型写短。

        "简短"现在只允许出现在**否定**语境里（"不要为了简短而省略…"），
        不能再作为要求出现。
        """
        prompt = self._prompt()
        assert "生成一段简短的训练代码" not in prompt
        assert "不要为了简短而省略" in prompt
        assert "完整" in prompt

    def test_prompt_requires_fence(self):
        prompt = self._prompt()
        assert "围栏" in prompt
        assert "```python" in prompt

    def test_prompt_allows_chinese_in_string_literals(self):
        prompt = self._prompt()
        assert "字符串字面量" in prompt

    def test_prompt_states_required_pipeline(self):
        prompt = self._prompt()
        for stage in ("数据加载", "训练循环", "评估", "打印"):
            assert stage in prompt

    def test_continuation_prompt_mentions_resume_contract(self):
        llm = _ScriptedLLM([_PART1, _PART2])
        agent = CodeExecutorAgent(llm)
        agent._continue_code({"method": "线性回归", "dataset": "合成数据"},
                             _PART1)
        prompt = llm.prompts[0]
        assert "从断点继续往下写" in prompt
        assert "把上面最后一行完整地重写一遍" in prompt
        assert "不要重写开头" in prompt
