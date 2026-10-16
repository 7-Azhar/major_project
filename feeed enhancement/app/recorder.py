"""Video stream recording module for AURA Engine."""

import logging
import os
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Manages OpenCV VideoWriter with automatic codec fallback and error checking."""

    CODEC_FALLBACKS = ["mp4v", "avc1", "XVID", "MJPG"]

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        enabled: bool = True,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 30.0
        self.enabled = enabled

        self.writer: Optional[cv2.VideoWriter] = None
        self.active_codec: Optional[str] = None
        self.frames_recorded: int = 0

    def start(self) -> bool:
        """Initialize VideoWriter using available codec."""
        if not self.enabled:
            logger.info("Recording is disabled.")
            return False

        if self.width <= 0 or self.height <= 0:
            logger.error(f"Invalid recording dimensions: {self.width}x{self.height}")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        for codec in self.CODEC_FALLBACKS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )

            if writer.isOpened():
                self.writer = writer
                self.active_codec = codec
                logger.info(
                    f"VideoRecorder initialized successfully. Path: '{self.output_path}', Codec: '{codec}', FPS: {self.fps:.2f}, Res: {self.width}x{self.height}"
                )
                return True
            else:
                writer.release()
                logger.warning(f"Codec '{codec}' failed for writer initialization. Trying next fallback...")

        logger.error(f"Failed to initialize VideoWriter with any fallback codec for path '{self.output_path}'.")
        self.enabled = False
        return False

    def write(self, frame: np.ndarray) -> bool:
        """Write a processed frame to disk."""
        if not self.enabled or self.writer is None or not self.writer.isOpened():
            return False

        if frame is None or frame.size == 0:
            return False

        # Validate shape match
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            self.writer.write(frame)
            self.frames_recorded += 1
            return True
        except Exception as e:
            logger.error(f"Error writing frame to video file: {e}")
            return False

    def release(self):
        """Finalize video file and release writer resources."""
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info(f"VideoRecorder closed. Total frames recorded: {self.frames_recorded}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
