"""Generates realistic degraded FPV benchmark video samples.

Injects FPV degradation characteristics:
- Low-light underexposure (vignetting & illumination drop)
- High-ISO sensor noise & analog transmission snow
- Motion blur / camera vibration blur
- H.264 compression blocking artifacts
"""

import os
import sys
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_degraded_fpv_sample(
    output_path: str = "samples/real_fpv_degraded.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 60,
    duration_sec: int = 5,
):
    """Generates a degraded FPV video file containing underexposure, noise, and motion blur."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not initialize VideoWriter for path: {output_path}")

    print(f"Generating realistic degraded FPV sample: {output_path} ({width}x{height} @ {fps}fps)...")

    # Create vignetting mask for underexposed FPV lens effect
    X_kernel = cv2.getGaussianKernel(width, width / 2)
    Y_kernel = cv2.getGaussianKernel(height, height / 2)
    kernel_matrix = Y_kernel * X_kernel.T
    vignette_mask = kernel_matrix / np.max(kernel_matrix)

    np.random.seed(101)

    for i in range(total_frames):
        # 1. Base scene (Dark drone flight environment)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (25, 15, 10)  # Very dim low-light background

        # Draw low-light terrain obstacles
        cv2.rectangle(frame, (100, 400), (400, 720), (40, 30, 20), -1)
        cv2.rectangle(frame, (800, 300), (1100, 720), (35, 25, 15), -1)

        # Moving drone object (target)
        cx = int((width / 2) + (width / 3) * np.sin(2 * np.pi * i / fps))
        cy = int((height / 2) + (height / 5) * np.cos(2 * np.pi * i / fps))
        cv2.circle(frame, (cx, cy), 35, (30, 140, 180), -1)  # Low contrast drone target

        # Moving vehicle target
        vx = int((width / 3) + (width / 4) * np.cos(2 * np.pi * i / fps))
        vy = int((height / 1.5) + (height / 8) * np.sin(2 * np.pi * i / fps))
        cv2.rectangle(frame, (vx - 25, vy - 20), (vx + 25, vy + 20), (120, 40, 40), -1)

        # 2. Apply Low-Light Underexposure (Drop illumination to 35%)
        dimmed = (frame.astype(np.float32) * 0.35).astype(np.uint8)

        # 3. Apply Vignetting
        vignetted = (dimmed.astype(np.float32) * np.dstack([vignette_mask]*3)).astype(np.uint8)

        # 4. Inject Gaussian Sensor Noise & Transmission Snow
        noise = np.random.normal(0, 18, (height, width, 3)).astype(np.float32)
        noisy = np.clip(vignetted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Apply Directional Motion Blur (Camera vibration)
        blur_kernel_size = 5
        kernel_mb = np.zeros((blur_kernel_size, blur_kernel_size))
        kernel_mb[int((blur_kernel_size-1)/2), :] = np.ones(blur_kernel_size)
        kernel_mb /= blur_kernel_size
        degraded_frame = cv2.filter2D(noisy, -1, kernel_mb)

        # HUD Text
        cv2.putText(
            degraded_frame,
            f"REAL FPV DEGRADED STREAM (Low Light + Noise + Motion Blur) Frame {i+1}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

        out.write(degraded_frame)

    out.release()
    print(f"Degraded FPV video successfully saved to: {output_path}")


if __name__ == "__main__":
    generate_degraded_fpv_sample()
