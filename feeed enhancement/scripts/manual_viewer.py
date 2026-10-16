"""CLI launcher for AURA Phase 4A Manual Real-Time Video Enhancement Viewer.

Usage:
  python -m scripts.manual_viewer [--source <path>] [--enhancer <1-8>] [--mode <mode>]

Examples:
  python -m scripts.manual_viewer
  python -m scripts.manual_viewer --source samples/fpv/my_flight.mp4
  python -m scripts.manual_viewer --source 0
"""

import os
import sys

from app.runtime import configure_runtime
configure_runtime()

import argparse
import glob

from app.manual_viewer import ManualViewerApp


def find_default_source() -> str:
    """Locate a default video source for evaluation."""
    # 1. Search samples/fpv/ for user-provided recordings
    fpv_dir = os.path.join("samples", "fpv")
    if os.path.isdir(fpv_dir):
        exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        matches = []
        for ext in exts:
            matches.extend(glob.glob(os.path.join(fpv_dir, ext)))
        if matches:
            print(f"Found user FPV recording: {matches[0]}", flush=True)
            return matches[0]

    # 2. Fall back to project sample videos
    fallback_1 = os.path.join("samples", "real_fpv_degraded.mp4")
    if os.path.exists(fallback_1):
        return fallback_1

    fallback_2 = os.path.join("samples", "test_synthetic.mp4")
    if os.path.exists(fallback_2):
        return fallback_2

    raise FileNotFoundError(
        "No video source found. Please place video files in 'samples/fpv/' or specify --source <path>."
    )


def main():
    parser = argparse.ArgumentParser(
        description="AURA Phase 4A — Manual Real-Time Video Enhancement Viewer"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Path to video file (.mp4, .avi, .mov) or webcam index (e.g. 0). Defaults to samples/fpv/",
    )
    parser.add_argument(
        "--enhancer",
        type=str,
        default="6",
        help="Initial enhancer key [1=Raw, 2=CLAHE, 3=Unsharp, 4=Bilateral, 5=FPV Restoration, 6=Zero-DCE, 7=FPV Hybrid, 8=FPV Restoration v2] (default: 6)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="side_by_side",
        choices=["original", "enhanced", "side_by_side", "split_view"],
        help="Initial display mode (default: side_by_side)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV window (for automated testing)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit maximum frames to process (0 = unlimited)",
    )

    args = parser.parse_args()

    # Determine source
    source_val = args.source
    if not source_val:
        source_val = find_default_source()
    elif source_val.isdigit():
        source_val = int(source_val)

    # Initialize and launch app
    app = ManualViewerApp(
        source=source_val,
        initial_enhancer_key=args.enhancer,
        initial_mode=args.mode,
        headless=args.headless,
        max_frames=args.max_frames,
    )

    app.run()


if __name__ == "__main__":
    main()
