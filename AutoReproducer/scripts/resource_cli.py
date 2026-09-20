#!/usr/bin/env python3
"""resource_cli.py - 三层缓存命令化（P2：L0/L1 存储管理 CLI）。

对 ResourceManager 的缓存管理能力做命令行封装，覆盖方案
「三层存储」的可操作层：

  status       查看 L0 配额占用与按论文统计（热缓存健康度）
  list         列出已登记论文（manifest）及资源路径/体积
  manifest     查看单篇论文 manifest（含懒加载资源状态）
  archive      归档论文 L0 资源到 L1 温存储（zip，默认 data/archive/）
  restore      从 L1 归档 zip 恢复回 L0（repos/datasets/manifests）
  prune        配额超限时给出建议归档清单；
               --yes 时先归档到 L1 再清理 L0（归档后清理，不丢数据）
  quota-check  拉取前配额预检：剩余配额不足时拒绝（退出码 2）并
               给出建议归档清单——供编排器/脚本在 fetch 前调用

退出码：0 成功；1 资源不存在/参数错误；2 配额超限（quota-check）。

数据根与配额可被环境变量覆盖（与 ResourceManager 同源）：
  AUTOREPRO_DATA_ROOT    缓存根（默认 <repo>/data）
  AUTOREPRO_L0_QUOTA_GB  L0 配额 GB（默认 20）

示例：
  python scripts/resource_cli.py status
  python scripts/resource_cli.py list
  python scripts/resource_cli.py manifest 2026abcd1234
  python scripts/resource_cli.py archive 2026abcd1234 --dest D:/nas/archive
  python scripts/resource_cli.py restore D:/nas/archive/2026abcd1234.zip
  python scripts/resource_cli.py prune --yes
  python scripts/resource_cli.py quota-check --size-gb 1.5
"""
import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.resource_manager import (  # noqa: E402
    ResourceManager, _fmt_bytes)


def _manager(args) -> ResourceManager:
    quota = None
    if getattr(args, "quota_gb", None):
        quota = int(float(args.quota_gb) * 1024 ** 3)
    return ResourceManager(data_root=args.data_root, quota_bytes=quota)


def _timestamp(ts) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, TypeError):
        return "-"


# ---------------- 子命令 ----------------

def cmd_status(args) -> int:
    mgr = _manager(args)
    usage = mgr.quota_usage()
    print(f"L0 热缓存占用: {usage['human']} / {usage['quota_human']} "
          f"({usage['percent']}%)")
    for name, root_info in usage["per_root"].items():
        print(f"  {name:<10} {root_info['human']:>10}")
    stats = mgr.stats()
    print(f"已登记论文: {stats['manifest_count']} 篇")
    for p in stats["papers"]:
        extra = f" | 已清理 {p['cleaned_at'][:10]}" if p.get("cleaned_at") else ""
        print(f"  {p['paper_id']}  {(p['title'] or '')[:40]:<40} "
              f"code {_fmt_bytes(p['code_bytes'])}  "
              f"data {_fmt_bytes(p['dataset_bytes'])}{extra}")
    return 0


def cmd_list(args) -> int:
    mgr = _manager(args)
    manifests = mgr.list_manifests()
    if not manifests:
        print("暂无已登记论文（data/manifests 为空）")
        return 0
    for m in manifests:
        res = m.get("resources", {})
        print(f"{m['paper_id']}  {(m.get('paper_title') or '')[:50]}")
        print(f"  code   : {res.get('code') or '-'}")
        print(f"  dataset: {res.get('dataset') or '-'}")
        print(f"  weights: {res.get('weights') or '-'}")
        print(f"  created: {m.get('created_at', '')[:19]}")
    return 0


def cmd_manifest(args) -> int:
    mgr = _manager(args)
    manifest = mgr.get_manifest(args.paper_id)
    if manifest is None:
        print(f"manifest 不存在: {args.paper_id}", file=sys.stderr)
        return 1
    import json
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_archive(args) -> int:
    mgr = _manager(args)
    dest = args.dest or str(mgr.archive_root)
    result = mgr.archive(args.paper_id, dest_dir=dest)
    bytes_ = result.get("bytes", 0)
    print(f"已归档 -> {result['archive']}  ({_fmt_bytes(bytes_)}, "
          f"{result['entries']} 个条目)")
    if not result["entries"]:
        print("警告: 该论文 L0 无任何资源（已清理或从未拉取）",
              file=sys.stderr)
        return 1
    return 0


def cmd_restore(args) -> int:
    mgr = _manager(args)
    result = mgr.restore(args.archive, paper_id=args.paper_id)
    if not result.get("ok"):
        print(result.get("detail", "恢复失败"), file=sys.stderr)
        return 1
    print(f"已恢复 {len(result['restored'])} 个文件")
    if result.get("manifest"):
        print(f"manifest: {result['manifest'].get('paper_id', '?')}")
    return 0


def cmd_prune(args) -> int:
    mgr = _manager(args)
    usage = mgr.quota_usage()
    if usage["bytes"] <= mgr.quota_bytes:
        print(f"配额未超限（{usage['human']} / {usage['quota_human']}），无需清理")
        return 0
    suggestions = mgr.enforce_quota(dry_run=not args.yes)
    if not suggestions:
        print(f"配额超限（{usage['human']} / {usage['quota_human']}），"
              "但无可用归档候选")
        return 2
    for s in suggestions:
        print(f"  [{s['paper_id']}] "
              f"{s['base']:<10} {s['human']:>10}  "
              f"最近使用 {_timestamp(s['last_used'])}  {s['reason']}")
    if not args.yes:
        print("以上为建议清单（dry_run）。确认执行请加 --yes："
              "将先归档到 L1 再清理 L0（不丢数据）")
        return 2
    # --yes：按 paper_id 去重，先归档 L1 再清理 L0
    done = {}
    archives = []
    for s in suggestions:
        pid = s["paper_id"]
        if pid in done:
            continue
        done[pid] = True
        arc = mgr.archive(pid)
        if arc.get("entries"):
            archives.append(arc["archive"])
        mgr.cleanup(pid, keep_manifest=True)
        print(f"已归档并清理: {pid} -> {arc['archive']}")
    print(f"配额腾出检查: {mgr.quota_usage()['human']} "
          f"/ {mgr.quota_usage()['quota_human']}")
    return 0


def cmd_quota_check(args) -> int:
    """拉取前配额预检：模拟下载 size_gb 后是否超限；超限则拒绝。"""
    mgr = _manager(args)
    usage = mgr.quota_usage()
    need = float(args.size_gb) * 1024 ** 3
    after = usage["bytes"] + need
    if after <= mgr.quota_bytes:
        print(f"OK: 当前 {usage['human']}，预计增长 {_fmt_bytes(need)}，"
              f"合计 {_fmt_bytes(after)} <= {usage['quota_human']}")
        return 0
    print(f"配额不足（拒绝下载）: 当前 {usage['human']}，+{_fmt_bytes(need)}"
          f" -> {_fmt_bytes(after)} > {usage['quota_human']}",
          file=sys.stderr)
    # 预计超额：即使当前未超限也给出按 LRU 排列的建议
    remaining = max(mgr.quota_bytes - usage["bytes"], 0)
    projected_excess = int(need - remaining)
    suggestions = mgr.enforce_quota(dry_run=True,
                                    excess_bytes=projected_excess)
    if suggestions:
        print("建议先归档:", file=sys.stderr)
        for s in suggestions[:10]:
            print(f"  {s['paper_id']} {s['base']} {s['human']} "
                  f"({_timestamp(s['last_used'])})", file=sys.stderr)
    return 2


# ---------------- 主入口 ----------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resource_cli",
        description="AutoReproducer 三层缓存管理 CLI（L0/L1）")
    parser.add_argument("--data-root", default=None,
                        help="缓存根（默认 $AUTOREPRO_DATA_ROOT 或 <repo>/data）")
    parser.add_argument("--quota-gb", default=None,
                        help="L0 配额 GB（默认 20 或 $AUTOREPRO_L0_QUOTA_GB）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看 L0 配额占用与论文统计")
    sub.add_parser("list", help="列出已登记论文及资源状态")

    p_manifest = sub.add_parser("manifest", help="查看单篇论文 manifest")
    p_manifest.add_argument("paper_id")

    p_archive = sub.add_parser("archive", help="归档论文 L0 到 L1 zip")
    p_archive.add_argument("paper_id")
    p_archive.add_argument("--dest", default=None,
                           help="L1 目录（默认 data/archive/）")

    p_restore = sub.add_parser("restore", help="从 L1 zip 恢复回 L0")
    p_restore.add_argument("archive", help="归档 zip 路径")
    p_restore.add_argument("--paper-id", default=None)

    p_prune = sub.add_parser("prune", help="配额超限建议/执行归档清理")
    p_prune.add_argument("--yes", action="store_true",
                         help="确认执行：先归档 L1 再清理 L0")

    p_check = sub.add_parser("quota-check",
                             help="拉取前配额预检（超限拒绝，退出码 2）")
    p_check.add_argument("--size-gb", type=float, required=True,
                         help="预计下载体积 GB（可用数据集注册表 size_gb）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "status": cmd_status, "list": cmd_list,
        "manifest": cmd_manifest, "archive": cmd_archive,
        "restore": cmd_restore, "prune": cmd_prune,
        "quota-check": cmd_quota_check,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())