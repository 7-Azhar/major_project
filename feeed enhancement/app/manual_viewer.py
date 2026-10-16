"""Manual Real-Time Video Enhancement Viewer for AURA Engine — Phase 4A.

Provides interactive side-by-side, split-view, and full-frame comparison
of raw video against classical and neural enhancement pipelines.
"""

import os
import sys

from app.runtime import configure_runtime
configure_runtime()

import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.enhancers import (
    BilateralDenoiseEnhancer,
    ClaheEnhancer,
    CompositeEnhancer,
    FpvHybridEnhancer,
    FpvRestorationEnhancer,
    FpvRestorationV2Enhancer,
    IdentityProcessor,
    UnsharpMaskEnhancer,
    ZeroDceEnhancer,
)
from app.pipeline import BaseProcessor
from app.recorder import VideoRecorder

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class ZeroDceOnnxCUDAEnhancer(BaseProcessor):
    """Wrapper for Zero-DCE FP16 ONNX model with CUDA Execution Provider."""

    def __init__(self, onnx_path: Optional[str] = None):
        super().__init__()
        self.onnx_path = onnx_path
        self.session = None
        self.provider = "CPU"
        self.fp16 = True
        self.input_h = 720
        self.input_w = 1280
        self._fallback = None

        if HAS_ORT:
            if self.onnx_path is None or not os.path.exists(self.onnx_path):
                candidates = [
                    "outputs/models/zerodce_fp16_c4_1280x720.onnx",
                    "outputs/models/zerodce_fp16_c4_640x360.onnx",
                    "outputs/models/zerodce_fp16_c4.onnx",
                    "outputs/models/zerodce_fp16.onnx",
                    "outputs/models/zerodce.onnx",
                ]
                for c in candidates:
                    if os.path.exists(c):
                        self.onnx_path = c
                        break

            if self.onnx_path and os.path.exists(self.onnx_path):
                try:
                    so = ort.SessionOptions()
                    so.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    self.session = ort.InferenceSession(
                        self.onnx_path,
                        sess_options=so,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    active = self.session.get_providers()
                    self.provider = active[0]

                    # Parse input shape if static
                    inp = self.session.get_inputs()[0]
                    shape = inp.shape
                    if len(shape) == 4:
                        if isinstance(shape[2], int) and shape[2] > 0:
                            self.input_h = shape[2]
                        if isinstance(shape[3], int) and shape[3] > 0:
                            self.input_w = shape[3]
                except Exception as e:
                    logger.warning(f"Could not load ONNX session: {e}")

    @property
    def fallback(self):
        if self._fallback is None:
            self._fallback = ZeroDceEnhancer(fp16=True, num_curves=4)
        return self._fallback

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.perf_counter()

        if self.session is not None:
            h, w = frame.shape[:2]
            # Resize to expected ONNX input size if needed
            if h != self.input_h or w != self.input_w:
                frame_in = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_in = frame

            rgb = frame_in[:, :, ::-1].astype(np.float16 if self.fp16 else np.float32) / 255.0
            tensor_in = np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[np.newaxis, :, :, :]
            )
            try:
                ort_out = self.session.run(None, {"input": tensor_in})[0]
                out = np.clip(ort_out[0].transpose(1, 2, 0), 0, 1)
                enhanced = (out * 255).astype(np.uint8)[:, :, ::-1]

                if h != self.input_h or w != self.input_w:
                    enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)

                t1 = time.perf_counter()
                meta = {
                    "processor": f"Zero-DCE FP16 ({self.provider})",
                    "latency_ms": (t1 - t0) * 1000,
                    "provider": self.provider,
                }
                return np.ascontiguousarray(enhanced), meta
            except Exception as e:
                logger.warning(f"ONNX inference failed: {e}, using PyTorch fallback")

        # PyTorch / analytical fallback
        enhanced, meta = self.fallback.process(frame)
        t1 = time.perf_counter()
        meta["processor"] = "Zero-DCE (PyTorch Fallback)"
        meta["latency_ms"] = (t1 - t0) * 1000
        return enhanced, meta


class ManualViewerApp:
    """
    Interactive Real-Time Video Enhancement Viewer for Manual Evaluation.
    """

    ENHANCER_MAP = {
        "1": ("Identity / Raw", IdentityProcessor),
        "2": ("CLAHE", ClaheEnhancer),
        "3": ("Unsharp Mask", UnsharpMaskEnhancer),
        "4": ("Bilateral Denoise", BilateralDenoiseEnhancer),
        "5": ("FPV Restoration Baseline", FpvRestorationEnhancer),
        "6": ("Zero-DCE FP16", ZeroDceOnnxCUDAEnhancer),
        "7": ("FPV Hybrid", FpvHybridEnhancer),
        "8": ("FPV Restoration v2", FpvRestorationV2Enhancer),
    }

    MODE_MAP = {
        "original": "Original",
        "enhanced": "Enhanced",
        "side_by_side": "Side-by-Side",
        "split_view": "Split View",
    }

    def __init__(
        self,
        source: Union[int, str],
        initial_enhancer_key: str = "6",
        initial_mode: str = "side_by_side",
        window_name: str = "AURA Manual Enhancement Viewer",
        output_dir: str = "outputs/manual_eval",
        headless: bool = False,
        max_frames: int = 0,
    ):
        self.source = source
        self.window_name = window_name
        self.output_dir = output_dir
        self.headless = headless
        self.max_frames = max_frames

        # State
        self.current_enhancer_key = initial_enhancer_key if initial_enhancer_key in self.ENHANCER_MAP else "6"
        self.current_mode = initial_mode if initial_mode in self.MODE_MAP else "side_by_side"
        self.paused = False
        self.hud_enabled = True
        self.split_ratio = 0.5  # 0.1 to 0.9

        # Cache initialized enhancers
        self.enhancer_instances: Dict[str, Any] = {}
        self._init_enhancers()

        # Metrics & Session Tracking
        self.processing_latencies: List[float] = []
        self.frame_times: List[float] = []
        self.evaluated_enhancers: set = {self.ENHANCER_MAP[self.current_enhancer_key][0]}
        self.evaluated_modes: set = {self.current_mode}

        # Video Recording
        self.recorder: Optional[VideoRecorder] = None
        self.recording_active = False

        # Capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.source_width = 0
        self.source_height = 0
        self.source_fps = 30.0

    def _init_enhancers(self):
        """Pre-instantiate enhancers for instant switching."""
        logger.info("Initializing enhancement processors...")
        for key, (label, cls) in self.ENHANCER_MAP.items():
            try:
                self.enhancer_instances[key] = cls()
                logger.info(f"  [{key}] {label} initialized.")
            except Exception as e:
                logger.warning(f"  [{key}] {label} initialization failed: {e}")
                self.enhancer_instances[key] = IdentityProcessor()

    def get_current_enhancer(self) -> Tuple[str, Any]:
        label, _ = self.ENHANCER_MAP.get(self.current_enhancer_key, ("Identity", IdentityProcessor))
        instance = self.enhancer_instances.get(self.current_enhancer_key, IdentityProcessor())
        return label, instance

    def _compose_display_frame(
        self, raw_frame: np.ndarray, enhanced_frame: np.ndarray
    ) -> np.ndarray:
        """Compose current display layout based on mode."""
        # Ensure dimensions match
        if raw_frame.shape[:2] != enhanced_frame.shape[:2]:
            enhanced_frame = cv2.resize(enhanced_frame, (raw_frame.shape[1], raw_frame.shape[0]))

        h, w = raw_frame.shape[:2]

        if self.current_mode == "original":
            return raw_frame.copy()

        elif self.current_mode == "enhanced":
            return enhanced_frame.copy()

        elif self.current_mode == "side_by_side":
            # Left: Raw, Right: Enhanced
            left = raw_frame.copy()
            right = enhanced_frame.copy()

            # Add header tags to sub-panels
            cv2.rectangle(left, (10, 10), (140, 35), (0, 0, 0), -1)
            cv2.putText(left, "ORIGINAL", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

            cv2.rectangle(right, (10, 10), (140, 35), (0, 0, 0), -1)
            cv2.putText(right, "ENHANCED", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

            return np.hstack([left, right])

        elif self.current_mode == "split_view":
            split_x = int(w * self.split_ratio)
            composed = enhanced_frame.copy()
            composed[:, :split_x] = raw_frame[:, :split_x]

            # Vertical divider line
            cv2.line(composed, (split_x, 0), (split_x, h), (0, 255, 255), 2)
            cv2.circle(composed, (split_x, h // 2), 6, (0, 255, 255), -1)

            # Labels on left and right of split line
            if split_x > 100:
                cv2.putText(composed, "ORIGINAL", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            if w - split_x > 100:
                cv2.putText(composed, "ENHANCED", (split_x + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

            return composed

        return raw_frame.copy()

    def _draw_hud_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        frame_idx: int,
    ):
        """Draw minimal HUD stats overlay."""
        if not self.hud_enabled:
            return

        enhancer_name, _ = self.get_current_enhancer()
        mode_name = self.MODE_MAP.get(self.current_mode, self.current_mode)
        h, w = frame.shape[:2]

        # Compact HUD box in top-left
        box_w = min(380, w - 20)
        box_h = 105
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (15, 15, 15), -1)
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 255, 128), 1)

        # Title
        cv2.putText(frame, "AURA Manual Viewer", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)

        # Mode & Enhancer
        split_str = f" ({int(self.split_ratio*100)}%)" if self.current_mode == "split_view" else ""
        cv2.putText(frame, f"Mode: {mode_name}{split_str} | Enhancer: {enhancer_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Performance stats
        cv2.putText(frame, f"FPS: {fps:.1f} | Processor Latency: {latency_ms:.1f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Resolution & Frame count
        total_str = f"/{self.total_frames}" if self.total_frames > 0 else ""
        rec_str = " | REC🔴" if self.recording_active else ""
        pause_str = " | PAUSED ⏸" if self.paused else ""
        cv2.putText(frame, f"Res: {self.source_width}x{self.source_height} | Frame: {frame_idx}{total_str}{pause_str}{rec_str}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom help prompt
        cv2.putText(frame, "[1-8]: Enhancer | [S/E/O/V]: Mode | [Space]: Pause | [C]: Rec | [Q]: Quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        """Main viewer event loop."""
        logger.info(f"Opening video source: {self.source}")
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap or not self.cap.isOpened():
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            print(f"ERROR: Could not open video source '{self.source}'", flush=True)
            return

        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.source_fps <= 0 or np.isnan(self.source_fps):
            self.source_fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\n" + "=" * 70, flush=True)
        print("AURA Phase 4A — Manual Video Enhancement Viewer", flush=True)
        print(f"Source: {self.source} ({self.source_width}x{self.source_height} @ {self.source_fps:.1f} FPS)", flush=True)
        print("Controls: [1-6] Enhancer | [S/E/O/V] View Mode | [Space] Pause | [C] Record | [Q] Quit", flush=True)
        print("=" * 70 + "\n", flush=True)

        if not self.headless:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, min(1280, max(640, self.source_width)), min(720, max(360, self.source_height)))
            except Exception as e:
                logger.warning(f"Could not create OpenCV window: {e}. Running in headless mode.")
                self.headless = True

        frame_idx = 0
        last_time = time.perf_counter()
        raw_frame = None

        while True:
            t_loop_start = time.perf_counter()

            if not self.paused or raw_frame is None:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    if isinstance(self.source, str) and not self.headless:
                        # Loop video file
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret or frame is None or frame.size == 0:
                        print("\nEnd of video stream.", flush=True)
                        break
                raw_frame = frame
                frame_idx += 1

            if self.max_frames > 0 and frame_idx > self.max_frames:
                print(f"\nReached max frames limit ({self.max_frames}). Exiting loop.", flush=True)
                break

            # Apply current enhancement
            enhancer_name, enhancer = self.get_current_enhancer()
            self.evaluated_enhancers.add(enhancer_name)
            self.evaluated_modes.add(self.current_mode)

            t_proc_start = time.perf_counter()
            if hasattr(enhancer, "process"):
                enhanced_frame, meta = enhancer.process(raw_frame)
            else:
                enhanced_frame = raw_frame.copy()
                meta = {}
            t_proc_end = time.perf_counter()
            proc_latency_ms = (t_proc_end - t_proc_start) * 1000
            self.processing_latencies.append(proc_latency_ms)

            # Compose display layout
            display_frame = self._compose_display_frame(raw_frame, enhanced_frame)

            # Record if recording active
            if self.recording_active and self.recorder:
                self.recorder.write(display_frame)

            # Instantaneous FPS calculation
            t_loop_end = time.perf_counter()
            frame_dt = t_loop_end - t_loop_start
            self.frame_times.append(frame_dt)
            fps = 1.0 / frame_dt if frame_dt > 0 else 0.0

            # Draw HUD
            hud_frame = display_frame.copy()
            self._draw_hud_overlay(hud_frame, fps, proc_latency_ms, frame_idx)

            if self.headless:
                if frame_idx % 20 == 0:
                    print(f"  Frame {frame_idx} | Mode: {self.current_mode} | Enhancer: {enhancer_name} | Latency: {proc_latency_ms:.1f} ms", flush=True)
                continue

            # Render frame to window
            cv2.imshow(self.window_name, hud_frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1 if not self.paused else 30) & 0xFF
            if key == 27 or key == ord("q") or key == ord("Q"):
                print("\nQuit requested by user.", flush=True)
                break

            # Handle Controls
            self._handle_key(key, frame_idx)

        # Cleanup & Save session summary
        self.stop()

    def _handle_key(self, key: int, current_frame_idx: int):
        """Process keyboard shortcuts."""
        if key == 255 or key == 0:
            return

        char = chr(key).lower() if 0 <= key < 128 else ""

        # Enhancer Switching (1-6)
        if char in self.ENHANCER_MAP:
            self.current_enhancer_key = char
            label, _ = self.get_current_enhancer()
            print(f"  -> Switched Enhancer to [{char}]: {label}", flush=True)
            if self.current_mode == "original":
                self.current_mode = "enhanced"

        # Mode Switching
        elif char == "s":
            self.current_mode = "side_by_side"
            print("  -> View Mode: Side-by-Side", flush=True)
        elif char == "e":
            self.current_mode = "enhanced"
            print("  -> View Mode: Enhanced Only", flush=True)
        elif char == "o":
            self.current_mode = "original"
            print("  -> View Mode: Original Only", flush=True)
        elif char == "v":
            self.current_mode = "split_view"
            print("  -> View Mode: Split View", flush=True)

        # Pause / Resume
        elif key == 32:  # Space
            self.paused = not self.paused
            print(f"  -> Playback {'PAUSED' if self.paused else 'RESUMED'}", flush=True)

        # Frame Stepping
        elif char in (".", "n") and self.cap:
            # Step forward
            self.paused = True
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 1)
            print(f"  -> Step Forward (Frame {pos+1})", flush=True)
        elif char in (",", "p") and self.cap:
            # Step backward
            self.paused = True
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_pos = max(0, pos - 2)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print(f"  -> Step Backward (Frame {new_pos})", flush=True)

        # Restart Video
        elif char == "r" and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.paused = False
            print("  -> Video Restarted", flush=True)

        # Recording Toggle
        elif char == "c":
            self._toggle_recording()

        # HUD Toggle
        elif char == "h":
            self.hud_enabled = not self.hud_enabled
            print(f"  -> HUD Overlay {'ENABLED' if self.hud_enabled else 'DISABLED'}", flush=True)

        # Split Ratio Adjustment
        elif char == "[":
            self.split_ratio = max(0.1, self.split_ratio - 0.05)
            print(f"  -> Split Ratio: {int(self.split_ratio*100)}%", flush=True)
        elif char == "]":
            self.split_ratio = min(0.9, self.split_ratio + 0.05)
            print(f"  -> Split Ratio: {int(self.split_ratio*100)}%", flush=True)

    def _toggle_recording(self):
        """Start or stop recording stream."""
        if self.recording_active:
            if self.recorder:
                self.recorder.release()
                self.recorder = None
            self.recording_active = False
            print("  -> Recording STOPPED.", flush=True)
        else:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_path = os.path.join("outputs", "recordings", f"manual_eval_{timestamp_str}.mp4")
            # Width depend on side-by-side or standard
            w = self.source_width * 2 if self.current_mode == "side_by_side" else self.source_width
            h = self.source_height
            self.recorder = VideoRecorder(rec_path, width=w, height=h, fps=self.source_fps)
            if self.recorder.start():
                self.recording_active = True
                print(f"  -> Recording STARTED: {rec_path}", flush=True)
            else:
                self.recorder = None
                print("  -> Recording initialization failed.", flush=True)

    def stop(self):
        """Release resources and write session report."""
        if self.recording_active and self.recorder:
            self.recorder.release()
            self.recorder = None

        if self.cap:
            self.cap.release()
            self.cap = None

        if not self.headless:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

        # Save session evaluation log JSON
        self._save_session_summary()

    def _save_session_summary(self):
        """Save manual evaluation session report."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.output_dir, f"manual_eval_{timestamp_str}.json")

        lats = self.processing_latencies if self.processing_latencies else [0.0]
        fts = self.frame_times if self.frame_times else [0.033]

        mean_fps = float(1.0 / np.mean(fts)) if np.mean(fts) > 0 else 0.0
        median_lat = float(np.median(lats))
        p95_lat = float(np.percentile(lats, 95))

        enhancer_label, _ = self.get_current_enhancer()

        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": str(self.source),
            "source_filename": os.path.basename(str(self.source)),
            "last_enhancement_method": enhancer_label,
            "evaluated_enhancers": sorted(list(self.evaluated_enhancers)),
            "evaluated_modes": sorted(list(self.evaluated_modes)),
            "source_resolution": f"{self.source_width}x{self.source_height}",
            "output_resolution": f"{self.source_width}x{self.source_height}",
            "average_fps": round(mean_fps, 2),
            "median_processing_latency_ms": round(median_lat, 2),
            "p95_processing_latency_ms": round(p95_lat, 2),
            "frame_count": len(self.processing_latencies),
            "dropped_frames": 0,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved manual evaluation log: {log_path}", flush=True)
        print(f"Summary: Evaluated {len(self.processing_latencies)} frames | Avg FPS: {mean_fps:.1f} | Median Latency: {median_lat:.2f} ms", flush=True)
