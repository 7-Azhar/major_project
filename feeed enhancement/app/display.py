"""Display layer for rendering live feed with minimal overlay metrics HUD."""

import logging
from typing import Any, Dict, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamDisplay:
    """Manages OpenCV window rendering with HUD overlay statistics."""

    def __init__(self, window_name: str = "AURA Video Engine Phase 0", disabled: bool = False):
        self.window_name = window_name
        self.disabled = disabled
        self._window_created = False

    def render(
        self,
        frame: np.ndarray,
        metrics_summary: Dict[str, Any],
        raw_frame: Optional[np.ndarray] = None,
        side_by_side: bool = False,
    ) -> bool:
        """
        Draw HUD overlay and render frame to window.

        Args:
            frame: Processed/Enhanced frame array
            metrics_summary: Metrics dict snapshot
            raw_frame: Original un-enhanced raw frame (required for side-by-side view)
            side_by_side: If True, renders raw frame (left) next to enhanced frame (right)

        Returns:
            bool: True if application should continue, False if user pressed 'q' or ESC to quit.
        """
        if self.disabled or frame is None or frame.size == 0:
            return True

        processor_name = metrics_summary.get("processor", "IdentityProcessor")
        fps = metrics_summary.get("observed_output_fps", 0.0)
        frame_time = metrics_summary.get("avg_frame_time_ms", 0.0)
        res = metrics_summary.get("resolution", f"{frame.shape[1]}x{frame.shape[0]}")
        dropped = metrics_summary.get("frames_dropped", 0)

        if side_by_side and raw_frame is not None and raw_frame.size > 0:
            # Ensure raw frame matches dimensions
            if raw_frame.shape[:2] != frame.shape[:2]:
                raw_frame = cv2.resize(raw_frame, (frame.shape[1], frame.shape[0]))

            left_side = raw_frame.copy()
            right_side = frame.copy()

            # Left Overlay (Raw)
            cv2.rectangle(left_side, (10, 10), (250, 45), (0, 0, 0), -1)
            cv2.putText(
                left_side,
                "RAW INPUT",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            # Right Overlay (Enhanced)
            cv2.rectangle(right_side, (10, 10), (360, 95), (0, 0, 0), -1)
            cv2.putText(
                right_side,
                f"ENHANCED: {processor_name}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                right_side,
                f"FPS: {fps:.1f} | Latency: {frame_time:.2f} ms",
                (20, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                right_side,
                f"Res: {res} | Dropped: {dropped}",
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Combine horizontally
            display_frame = np.hstack((left_side, right_side))
        else:
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (360, 120), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 10), (360, 120), (0, 255, 128), 1)

            cv2.putText(
                display_frame,
                f"AURA | Processor: {processor_name}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Latency: {frame_time:.2f} ms",
                (20, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Res: {res} | Dropped: {dropped}",
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Press 'q' or 'ESC' to exit",
                (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        try:
            cv2.imshow(self.window_name, display_frame)
            self._window_created = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
        except Exception as e:
            logger.warning(f"Display window failed or not supported in this environment: {e}")
            self.disabled = True

        return True

    def close(self):
        """Destroy OpenCV window."""
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_created = False
