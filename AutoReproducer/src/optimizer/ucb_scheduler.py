"""UCB(Upper Confidence Bound)预算感知调度器"""
import math


class UCBScheduler:
    """多臂老虎机调度器,在预算内平衡探索(Exploration)与利用(Exploitation)。

    每个"臂"代表一个优化方向,维护其拉取次数 n 与平均收益 Q。
    选择公式: UCB = Q + c * sqrt(ln N / n_i);未拉过的臂优先(强制探索)。

    融合增强（P1-⑥）:
    - seed(): 注入历史先验(伪拉取次数 + 先验均值),配合 ExperienceStore 在
      启动时播种,让搜索不必从零开始;
    - retire(): 退役一个方向(如参数树已穷尽),后续不再被选中。
    """

    def __init__(self, arms, budget, c=1.0):
        self.arms = {arm: {"n": 0, "Q": 0.0} for arm in arms}
        self.budget = budget
        self.c = c
        self.total_pulls = 0
        self.history = []
        self._retired = set()

    def select_arm(self):
        """按 UCB 公式选出下一轮尝试方向;预算耗尽或全部退役返回 None。"""
        if self.is_exhausted():
            return None

        alive = [a for a in self.arms if a not in self._retired]
        if not alive:
            return None

        unexplored = [a for a in alive if self.arms[a]["n"] == 0]
        if unexplored:
            return unexplored[0]

        n_total = self.total_pulls

        def ucb_score(arm):
            s = self.arms[arm]
            return s["Q"] + self.c * math.sqrt(math.log(n_total) / s["n"])

        return max(alive, key=ucb_score)

    def seed(self, arm, count, mean):
        """注入历史先验:伪拉取次数 + 先验均值(不消耗本轮预算)。

        count>0 会消除该臂的"未探索"状态并压缩其探索项,
        使搜索开局就偏向经验证明有效的方向。
        """
        if arm not in self.arms:
            self.arms[arm] = {"n": 0, "Q": 0.0}
        s = self.arms[arm]
        s["n"] = max(s["n"], int(count))
        s["Q"] = float(mean)

    def retire(self, arm):
        """退役一个方向:不再参与 select_arm(如参数树已穷尽)。"""
        self._retired.add(arm)

    def retired(self):
        """已退役的方向列表(排序稳定)。"""
        return sorted(self._retired)

    def update(self, arm, reward):
        """记录一次尝试结果,增量更新平均收益。"""
        s = self.arms[arm]
        s["n"] += 1
        s["Q"] += (reward - s["Q"]) / s["n"]
        self.total_pulls += 1
        self.history.append({"arm": arm, "reward": reward})

    def is_exhausted(self):
        return self.total_pulls >= self.budget

    def remaining(self):
        return max(0, self.budget - self.total_pulls)

    def trace(self):
        """返回调度轨迹,供报告与前端展示。"""
        return {
            "total_pulls": self.total_pulls,
            "budget": self.budget,
            "remaining": self.remaining(),
            "arms": {a: {"n": s["n"], "Q": round(s["Q"], 4)}
                     for a, s in self.arms.items()},
            "history": self.history,
        }
