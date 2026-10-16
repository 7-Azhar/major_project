"""Search Windows C: drive for cuDNN DLLs."""
import os

print("Searching C: drive for cuDNN DLLs (cudnn64_*.dll or cudnn*.dll)...")
matches = []
search_roots = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\tools",
    r"C:\cuda",
    r"C:\Users\azhar",
]

for root_dir in search_roots:
    if os.path.exists(root_dir):
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().startswith("cudnn") and f.lower().endswith(".dll"):
                    full = os.path.join(root, f)
                    matches.append(full)
                    print(f"Found: {full}")

if not matches:
    print("NO cuDNN DLLs found anywhere in searched directories!")
