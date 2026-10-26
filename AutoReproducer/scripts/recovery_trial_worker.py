"""Fixed isolated benchmark bootstrap; all behavior lives in hashed src code."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    driver = Path(__file__).resolve().parents[1] / "src" / "benchmark" / "recovery_adapter.py"
    runpy.run_path(str(driver))["worker_main"]()
