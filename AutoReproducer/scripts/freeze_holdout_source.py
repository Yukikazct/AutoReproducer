"""Freeze a product generation and, optionally, an unused registry paper."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.holdout_registry import build_freeze, write_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--paper-id", help="Unused paper identity in the shared acceptance registry")
    args = parser.parse_args()
    record = build_freeze(ROOT, paper_id=args.paper_id)
    path = write_freeze(ROOT, args.tag, record)
    print(json.dumps({key: value for key, value in record.items() if key != "files"}, ensure_ascii=False))
    print(path)


if __name__ == "__main__":
    main()
