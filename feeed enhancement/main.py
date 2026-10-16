"""Main entry point for AURA Video Enhancement Engine Phase 0."""

import logging
import sys
from app.benchmark import BenchmarkRunner
from app.config import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    config = parse_args()
    runner = BenchmarkRunner(config=config)
    runner.run()


if __name__ == "__main__":
    main()
