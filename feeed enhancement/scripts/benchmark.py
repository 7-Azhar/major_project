"""Standalone CLI script for running AURA Video Enhancement Engine benchmarks."""

import logging
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.benchmark import BenchmarkRunner
from app.config import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    config = parse_args()
    print("=" * 60)
    print("           AURA ENGINE PHASE 0 BENCHMARK SUITE")
    print("=" * 60)
    runner = BenchmarkRunner(config=config)
    report = runner.run()

    json_path = report.get("json_report_path", "")
    txt_path = report.get("txt_report_path", "")
    print(f"\n[SUCCESS] Benchmark run complete.")
    print(f"Artifact JSON: {json_path}")
    print(f"Artifact TXT:  {txt_path}")


if __name__ == "__main__":
    main()
