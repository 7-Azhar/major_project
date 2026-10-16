"""Video processing pipeline interface and Identity baseline processor for AURA."""

import abc
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseProcessor(abc.ABC):
    """Abstract base class for all frame processing and enhancement modules."""

    @abc.abstractmethod
    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process an incoming video frame.

        Args:
            frame: Input image array (BGR format, uint8)
            previous_frames: Optional list of preceding frames for temporal processors
            timestamps: Optional list of timestamps matching historical & current frame

        Returns:
            Tuple of (processed_frame, metadata_dict)
        """
        pass


class IdentityProcessor(BaseProcessor):
    """
    Baseline identity processor (Baseline 0).
    Performs identity pass-through: input frame -> unchanged frame -> output.
    Used for baseline performance benchmarking.
    """

    def __init__(self):
        super().__init__()

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Passes input frame directly to output without modification."""
        start_time = time.perf_counter()

        output_frame = frame

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "IdentityProcessor",
            "processing_time_sec": elapsed_time,
            "shape": frame.shape,
            "dtype": str(frame.dtype),
        }

        return output_frame, metadata


if __name__ == "__main__":
    import logging
    import sys
    from app.benchmark import BenchmarkRunner
    from app.config import parse_args

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = parse_args()
    runner = BenchmarkRunner(config=config, processor=IdentityProcessor())
    runner.run()
