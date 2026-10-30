"""Render and bundle an existing terminal result, without API calls or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.author_claim_comparison import _redact_text
from src.report_export import export_reproduction_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Saved terminal result.json or report.result.json')
    parser.add_argument('--output-dir', required=True, help='New output directory; must not already exist')
    args = parser.parse_args(argv)
    try:
        receipt = export_reproduction_report(args.result, args.output_dir)
    except (OSError, ValueError, TypeError) as exc:
        print('报告导出失败: ' + _redact_text(str(exc), ''), file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
