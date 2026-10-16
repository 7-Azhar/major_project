"""Video capture layer for AURA Video Enhancement Engine.

Provides low-latency, non-blocking frame ingestion from video files, webcams,
and OpenCV capture devices with precise per-frame microsecond timestamps.
"""

import logging
import queue
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameCapture:
    """Threaded OpenCV video capture manager with bounded buffer queue."""

    def __init__(
        self,
        source: Union[int, str],
        requested_width: int = 0,
        requested_height: int = 0,
        requested_fps: float = 0.0,
        queue_size: int = 2,
        loop: bool = False,
    ):
        self.source = source
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.requested_fps = requested_fps
        self.queue_size = max(1, queue_size)
        self.loop = loop

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)

        self.width: int = 0
        self.height: int = 0
        self.reported_fps: float = 0.0

        self.frames_received: int = 0
        self.dropped_frames: int = 0
        self.is_eos: bool = False
        self._is_opened: bool = False

    def start(self) -> "FrameCapture":
        """Initialize capture device and start non-blocking ingestion thread."""
        logger.info(f"Opening video source: {self.source}")

        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap or not self.cap.isOpened():
            # Fallback attempt without backend flag if DSHOW failed on Windows
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            self._is_opened = False
            logger.error(f"Failed to open video source: {self.source}")
            raise ValueError(f"Could not open video source: {self.source}")

        # Set requested resolution & FPS if specified
        if self.requested_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # Retrieve actual parameters
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.reported_fps < 0 or np.isnan(self.reported_fps):
            self.reported_fps = 0.0

        self._is_opened = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="AuraCaptureThread")
        self._thread.start()
        logger.info(
            f"Capture started. Dimensions: {self.width}x{self.height}, Reported FPS: {self.reported_fps:.2f}"
        )
        return self

    @property
    def is_opened(self) -> bool:
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def _capture_loop(self):
        """Worker thread loop to read frames continuously and place in queue."""
        frame_counter = 0

        while not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            capture_timestamp = time.perf_counter()

            if not ret or frame is None or frame.size == 0:
                if self.loop and isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    capture_timestamp = time.perf_counter()

            if not ret or frame is None or frame.size == 0:
                logger.info("End of stream or empty frame received.")
                self.is_eos = True
                break

            frame_counter += 1
            self.frames_received = frame_counter

            item = (frame_counter, capture_timestamp, frame)

            # If bounded queue is full, discard the oldest frame to maintain low-latency freshness
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self.dropped_frames += 1

        self.is_eos = True

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[int], float, Optional[np.ndarray]]:
        """
        Fetch the next available frame from the queue.

        Returns:
            (success, frame_id, capture_timestamp, frame)
        """
        if not self._is_opened:
            return False, None, 0.0, None

        try:
            frame_id, capture_timestamp, frame = self._queue.get(timeout=timeout)
            return True, frame_id, capture_timestamp, frame
        except queue.Empty:
            if self.is_eos:
                return False, None, 0.0, None
            return False, None, 0.0, None

    def stop(self):
        """Stop capture thread and release hardware resources."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self._is_opened = False
        logger.info("Capture thread stopped and resources released.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
