"""Download and verify pretrained Zero-DCE checkpoint."""
import urllib.request
import os
import sys

os.makedirs("outputs/models", exist_ok=True)

# Try Hugging Face mirror first
urls = [
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth",
    "https://huggingface.co/IanNathaniel/Zero-DCE/raw/main/Epoch99.pth",
    "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth",
]
dest = "outputs/models/Epoch99.pth"

print(f"Downloading pretrained Zero-DCE checkpoint...")

success = False
for url in urls:
    print(f"Trying URL: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        size = os.path.getsize(dest)
        if size > 1000:
            print(f"  Downloaded successfully: {size:,} bytes ({size/1024:.1f} KB)")
            success = True
            break
        else:
            print(f"  Downloaded file too small ({size} bytes), trying next...")
    except Exception as e:
        print(f"  Failed: {e}")

if not success:
    print("All download attempts failed.")
    sys.exit(1)

# Verify it's a valid PyTorch checkpoint
import torch

print("\nLoading checkpoint...")
state_dict = torch.load(dest, map_location="cpu", weights_only=True)

if isinstance(state_dict, dict):
    print(f"  Type: dict with {len(state_dict)} keys")
    print(f"  Keys:")
    for key in sorted(state_dict.keys()):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"    {key}: {shape}")

    total_params = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
    print(f"\n  Total parameters in checkpoint: {total_params:,}")
else:
    print(f"  Unexpected type: {type(state_dict)}")
    print(f"  Content: {state_dict}")
