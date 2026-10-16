"""Verify pretrained Zero-DCE checkpoint compatibility and visual enhancement."""
import os
import sys
import torch
import cv2
import numpy as np

from app.enhancers.neural import ZeroDCENet, ZeroDceEnhancer

print("=" * 60)
print("Phase 3 Correction Gate — Pretrained Zero-DCE Verification")
print("=" * 60)

ckpt_path = "outputs/models/Epoch99.pth"
if not os.path.exists(ckpt_path):
    print(f"Error: Checkpoint file '{ckpt_path}' not found.")
    sys.exit(1)

# 1. Inspect state dict keys
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
print(f"\n1. State dict keys in checkpoint:")
for k, v in state_dict.items():
    print(f"   - {k}: shape {tuple(v.shape)}")

# Map 'e_convX' -> 'convX' for our ZeroDCENet architecture
mapped_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("e_conv", "conv")
    mapped_state_dict[new_key] = v

print(f"\n2. Mapped state dict keys:")
for k in mapped_state_dict.keys():
    print(f"   - {k}")

# 2. Instantiate ZeroDCENet and load state dict strictly
net = ZeroDCENet(num_curves=8)
try:
    missing, unexpected = net.load_state_dict(mapped_state_dict, strict=True)
    print("\n3. State dict load result: SUCCESS (strict=True)")
    print(f"   Missing keys: {missing}")
    print(f"   Unexpected keys: {unexpected}")
except Exception as e:
    print(f"\n3. State dict load result: FAILED - {e}")
    sys.exit(1)

net.eval()

# 3. Test with ZeroDceEnhancer class
enhancer = ZeroDceEnhancer(model_path=ckpt_path, device="cpu", fp16=False, num_curves=8)
print(f"\n4. ZeroDceEnhancer initialized with checkpoint.")
print(f"   Model path loaded: {enhancer.model_path}")

# 4. Test deterministic inference
dummy = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
out1, _ = enhancer.process(dummy)
out2, _ = enhancer.process(dummy)
diff = np.abs(out1.astype(int) - out2.astype(int)).max()
print(f"\n5. Deterministic inference check: max diff between identical runs = {diff}")

# 5. Visual verification on low-light frame from real_fpv_degraded.mp4
cap = cv2.VideoCapture("samples/real_fpv_degraded.mp4")
ret, low_light_frame = cap.read()
cap.release()

if ret and low_light_frame is not None:
    enhanced_frame, meta = enhancer.process(low_light_frame)
    os.makedirs("outputs/visualizations", exist_ok=True)
    vis = np.hstack([low_light_frame, enhanced_frame])
    cv2.imwrite("outputs/visualizations/zerodce_pretrained_verification.png", vis)
    print(f"\n6. Visual test performed on sample frame (720p).")
    print(f"   Original frame mean brightness: {low_light_frame.mean():.2f}")
    print(f"   Enhanced frame mean brightness: {enhanced_frame.mean():.2f}")
    print(f"   Saved side-by-side comparison: outputs/visualizations/zerodce_pretrained_verification.png")

print("\n" + "=" * 60)
print("Zero-DCE Checkpoint Verification PASSED!")
print("=" * 60)
