"""Shared budget boundaries for the frozen method optimization protocol."""
import math


CONFIRMATION_RESERVE_SECONDS = 2400
MAX_OPTIMIZATION_SECONDS = 7200


def optimization_budget_error(seconds) -> str:
    """Return a user-facing error, or an empty string for a usable total budget."""
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 0 < seconds <= MAX_OPTIMIZATION_SECONDS or not math.isfinite(seconds)):
        return "优化总预算必须是大于 0 且不超过 7200 秒（120 分钟）的有限数值"
    if seconds <= CONFIRMATION_RESERVE_SECONDS:
        return ("优化总预算必须大于 40 分钟：当前验证流程需为确认阶段预留 40 分钟，"
                "总预算还包含在线分析、基线训练和候选试验。建议设置为 120 分钟；"
                "当前预算无法启动完整优化验证。")
    return ""
