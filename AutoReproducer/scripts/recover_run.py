"""Identify and mark interrupted method runs after their owning process exits."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_lifecycle import recover_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="需要检查的 data/runs/repository_* 目录")
    args = parser.parse_args(argv)
    result = recover_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "unknown_owner" else 0


if __name__ == "__main__":
    raise SystemExit(main())
