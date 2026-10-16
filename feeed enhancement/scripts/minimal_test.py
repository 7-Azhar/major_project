"""Minimal 3-frame timing test for ZeroDceEnhancer."""
import os
import sys
import time

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import torch
from app.enhancers.neural import ZeroDceEnhancer

def main():
    print("1. Initializing ZeroDceEnhancer with Epoch99.pth...")
    t_init0 = time.perf_counter()
    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)
    t_init1 = time.perf_counter()
    print(f"   Initialization took: {(t_init1 - t_init0)*1000:.2f} ms")

    print("2. Generating 720p (1280x720) sample frame...")
    np.random.seed(42)
    frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    print("3. Running 3 sequential inference passes...")
    for i in range(3):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(frame)
        t1 = time.perf_counter()
        with torch.no_grad():
            tensor_out = enhancer._forward_neural(tensor_in)
        t2 = time.perf_counter()
        out_frame = enhancer.postprocess(tensor_out)
        t3 = time.perf_counter()

        prep = (t1 - t0) * 1000
        fwd = (t2 - t1) * 1000
        post = (t3 - t2) * 1000
        total = (t3 - t0) * 1000
        print(f"   Frame {i+1}: Pre={prep:.2f}ms | Forward={fwd:.2f}ms | Post={post:.2f}ms | Total={total:.2f}ms")

    print("Done!")

if __name__ == "__main__":
    main()
