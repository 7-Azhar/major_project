"""Benchmark runner engine for AURA Video Enhancement Engine."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from app.capture import FrameCapture
from app.config import EngineConfig
from app.display import StreamDisplay
from app.enhancers import get_enhancer_by_name
from app.metrics import MetricsTracker
from app.pipeline import BaseProcessor, IdentityProcessor
from app.quality import evaluate_quality_pair
from app.recorder import VideoRecorder

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Executes streaming session and collects structured benchmark metrics."""

    def __init__(self, config: EngineConfig, processor: Optional[BaseProcessor] = None):
        self.config = config
        if processor is not None:
            self.processor = processor
        else:
            if self.config.processor == "clahe":
                self.processor = get_enhancer_by_name("clahe", clip_limit=self.config.clahe_clip)
            elif self.config.processor == "unsharp":
                self.processor = get_enhancer_by_name("unsharp", amount=self.config.unsharp_amount)
            else:
                self.processor = get_enhancer_by_name(self.config.processor)

        logger.info(f"BenchmarkRunner using processor: {self.processor.__class__.__name__}")

    def run(self) -> Dict[str, Any]:
        """
        Runs video streaming loop, measures performance, and exports reports.

        Returns:
            Dict containing complete performance report dictionary.
        """
        logger.info(f"Starting AURA Engine execution. Source: {self.config.source}")

        capture = FrameCapture(
            source=self.config.source,
            requested_width=self.config.requested_width,
            requested_height=self.config.requested_height,
            requested_fps=self.config.requested_fps,
            queue_size=self.config.queue_size,
            loop=self.config.loop,
        )

        try:
            capture.start()
        except Exception as e:
            logger.error(f"Failed starting capture: {e}")
            raise

        metrics = MetricsTracker(resolution_w=capture.width, resolution_h=capture.height)
        display = StreamDisplay(disabled=self.config.no_display)

        recorder: Optional[VideoRecorder] = None
        if self.config.record:
            rec_filename = (
                self.config.record_filename
                if self.config.record_filename
                else f"aura_recorded_{int(time.time())}.mp4"
            )
            rec_path = os.path.join(self.config.output_dir, rec_filename)
            recorder = VideoRecorder(
                output_path=rec_path,
                width=capture.width,
                height=capture.height,
                fps=capture.reported_fps if capture.reported_fps > 0 else 30.0,
                enabled=True,
            )
            recorder.start()

        metrics.start()
        start_wall_clock = time.perf_counter()

        logger.info("Entering main streaming loop...")
        try:
            while capture.is_opened and not capture.is_eos:
                # Check benchmark duration condition
                if (
                    self.config.benchmark_duration > 0
                    and (time.perf_counter() - start_wall_clock) >= self.config.benchmark_duration
                ):
                    logger.info(f"Reached benchmark duration target of {self.config.benchmark_duration}s.")
                    break

                # Check max frames condition
                if (
                    self.config.max_frames > 0
                    and metrics.frames_processed >= self.config.max_frames
                ):
                    logger.info(f"Reached max frames target of {self.config.max_frames}.")
                    break

                # Fetch next frame
                success, frame_id, capture_timestamp, frame = capture.read(timeout=0.5)
                if not success or frame is None:
                    if capture.is_eos:
                        break
                    continue

                # Process frame
                proc_start = time.perf_counter()
                processed_frame, proc_meta = self.processor.process(frame)
                proc_elapsed = time.perf_counter() - proc_start

                # Update metrics
                metrics.update_capture_counts(
                    frames_received=capture.frames_received,
                    dropped_frames=capture.dropped_frames,
                )

                # Record frame if recorder enabled
                if recorder and recorder.enabled:
                    recorder.write(processed_frame)

                # Render display
                stats_snapshot = metrics.get_current_stats()
                stats_snapshot["processor"] = self.processor.__class__.__name__

                should_continue = display.render(
                    frame=processed_frame,
                    metrics_summary=stats_snapshot,
                    raw_frame=frame,
                    side_by_side=self.config.side_by_side,
                )

                metrics.record_frame(
                    processing_time_sec=proc_elapsed,
                    timestamp_sec=capture_timestamp,
                    displayed=not self.config.no_display,
                )

                if not should_continue:
                    logger.info("User requested exit via display HUD ('q'/ESC).")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            metrics.stop()
            capture.stop()
            display.close()
            if recorder:
                recorder.release()

        # Generate report
        report_text = metrics.generate_report_text()
        report_data = metrics.get_current_stats()

        logger.info("\n" + report_text)

        # Export benchmark artifacts
        timestamp_str = int(time.time())
        json_path = os.path.join(self.config.output_dir, f"benchmark_report_{timestamp_str}.json")
        txt_path = os.path.join(self.config.output_dir, f"benchmark_report_{timestamp_str}.txt")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Saved benchmark report JSON: {json_path}")
            logger.info(f"Saved benchmark report TXT:  {txt_path}")
        except Exception as e:
            logger.error(f"Failed writing benchmark reports to disk: {e}")

        report_data["json_report_path"] = json_path
        report_data["txt_report_path"] = txt_path
        return report_data
