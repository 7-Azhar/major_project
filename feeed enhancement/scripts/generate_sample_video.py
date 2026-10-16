"""Utility script to generate a synthetic test video file for AURA benchmarking."""

import os
import sys
import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_synthetic_video(
    output_path: str = "samples/test_synthetic.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 60,
    duration_sec: int = 5,
):
    """Generates a synthetic MP4 video with moving shapes and frame numbers."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not initialize VideoWriter for path: {output_path}")

    print(f"Generating synthetic video: {output_path} ({width}x{height} @ {fps}fps, {duration_sec}s)...")

    for i in range(total_frames):
        # Create dark blue background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (40, 20, 15)

        # Draw grid
        for x in range(0, width, 100):
            cv2.line(frame, (x, 0), (x, height), (60, 40, 30), 1)
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (60, 40, 30), 1)

        # Moving circle
        cx = int((width / 2) + (width / 3) * np.sin(2 * np.pi * i / fps))
        cy = int((height / 2) + (height / 4) * np.cos(2 * np.pi * i / fps))
        cv2.circle(frame, (cx, cy), 40, (0, 255, 255), -1)

        # Rotating rectangle
        rx = int((width / 2) - (width / 4) * np.cos(2 * np.pi * i / fps))
        ry = int((height / 2) - (height / 3) * np.sin(2 * np.pi * i / fps))
        cv2.rectangle(frame, (rx - 30, ry - 30), (rx + 30, ry + 30), (255, 0, 128), -1)

        # Text HUD
        cv2.putText(
            frame,
            f"AURA Synthetic Test Stream - Frame {i + 1}/{total_frames}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Timestamp: {i / fps:.3f}s",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print(f"Synthetic video successfully saved to: {output_path}")


if __name__ == "__main__":
    generate_synthetic_video()
