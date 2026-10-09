"""Mark abandoned optimization records interrupted without restarting training."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization_recovery import recover_interrupted_optimizations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(os.environ.get(
        "AUTOREPRO_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "runs")
    args = parser.parse_args(argv)
    for path in recover_interrupted_optimizations(args.runs_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
