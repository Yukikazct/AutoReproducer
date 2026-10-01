"""代码单元（CodeUnit）——多代码单元管理的资源层数据模型。

背景：不少论文的实现不是单个仓库，而是**多个代码块最后整体调用**——
主实现仓库 + 统一模型库（如 thuml/Time-Series-Library）+ benchmark 框架 +
数据准备脚本 + 独立数据集仓库。此前系统只支持单仓库（且仅存档不执行），
iTransformer（thuml/iTransformer + Time-Series-Library + 网盘数据集）
即因此复现失败。

CodeUnit 把「一篇论文的一组代码来源」显式建模：每个单元有自己的
unit_id（目录名）、角色（main/library/benchmark/scripts/dataset/
alternative）、URL、revision pin 与溯源信息，落盘到
data/repos/<paper_id>/<unit_id>/，由 ResourceManager.fetch_units
逐单元克隆（复用 fetch_code 的幂等缓存与溯源标记）。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 单元角色：main=主实现仓库；library=被主实现依赖/并入的统一库；
# benchmark=评测框架；scripts=独立脚本仓库；dataset=数据集所在仓库；
# alternative=备选实现（主仓库不可用时兜底）
UNIT_ROLES = ("main", "library", "benchmark", "scripts", "dataset",
              "alternative")

# unit_id 只允许安全字符（目录名，防路径穿越）
_UNIT_ID_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


@dataclass
class CodeUnit:
    unit_id: str = ""          # "main" / "lib_0" / "alt_0" / "dataset" ...
    role: str = "main"         # UNIT_ROLES 之一
    url: str = ""              # git URL（数据单元可为普通 URL）
    local_path: str = ""       # data/repos/<paper_id>/<unit_id>/（fetch 后填充）
    revision: str = ""         # pin（commit sha / tag），尽力而为
    source: str = ""           # user_preference / papers_with_code /
                               # github_search / curated_fallback / llm_inferred
    fetch_state: str = ""      # cloned / cached / placeholder-skip /
                               # clone-failed / not_fetched
    provenance: Dict = field(default_factory=dict)   # {repo_url, commit, acquisition, revision}
    notes: str = ""            # 选中理由 / 提供哪个入口脚本

    def to_dict(self) -> Dict:
        return {
            "unit_id": self.unit_id,
            "role": self.role,
            "url": self.url,
            "local_path": self.local_path,
            "revision": self.revision,
            "source": self.source,
            "fetch_state": self.fetch_state,
            "provenance": dict(self.provenance),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CodeUnit":
        unit = cls()
        for key in ("unit_id", "role", "url", "local_path", "revision",
                    "source", "fetch_state", "notes"):
            if key in data:
                setattr(unit, key, str(data.get(key) or ""))
        if isinstance(data.get("provenance"), dict):
            unit.provenance = dict(data["provenance"])
        return unit


def sanitize_unit_id(raw: str) -> str:
    """unit_id 清洗：非法字符替换为 '_'；空串/'.'/'..' 返回空（拒绝克隆）。"""
    if not raw:
        return ""
    cleaned = _UNIT_ID_RE.sub("_", str(raw).strip())
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned


def units_from_discovery(discovery: Dict, max_extra: int = 2) -> List[CodeUnit]:
    """把 repo_discovery 的发现结果转成 CodeUnit 列表。

    selected_repo -> unit_id "main"（role 优先取候选携带的 role，缺省 main）；
    其余带 URL 的可信候选按序取前 max_extra 个作为 alternative/library。
    """
    units: List[CodeUnit] = []
    selected = (discovery.get("selected_repo") or "").strip()
    if selected:
        units.append(CodeUnit(
            unit_id="main",
            role="main",
            url=selected,
            revision=discovery.get("pinned_revision") or "",
            source="user_preference" if discovery.get("discovery_chain")
            and discovery["discovery_chain"][0] == "user_preference"
            else "discovery",
        ))
    seen = {selected}
    for cand in (discovery.get("candidates") or []):
        urls = cand.get("repo_urls") or []
        if not urls:
            continue
        url = urls[0]
        if url in seen:
            continue
        seen.add(url)
        role = cand.get("role") or (
            "alternative" if not selected else "alternative")
        units.append(CodeUnit(
            unit_id=("main" if not units and not selected else
                     next_unit_id(units, "alt")),
            role=role,
            url=url,
            source=cand.get("source") or "discovery",
        ))
        if len(units) > (1 if selected else 0) + max_extra:
            break
    return units


def dedupe_units(units: List[CodeUnit]) -> List[CodeUnit]:
    """按 URL 去重（保序）；空 URL 丢弃。"""
    out: List[CodeUnit] = []
    seen: set = set()
    for unit in units:
        url = (unit.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(unit)
    return out


def next_unit_id(units: List[CodeUnit], prefix: str) -> str:
    """按 prefix 生成下一个不冲突的 unit_id（lib_0 / lib_1 ...）。"""
    index = 0
    existing = {u.unit_id for u in units}
    while f"{prefix}_{index}" in existing:
        index += 1
    return f"{prefix}_{index}"
