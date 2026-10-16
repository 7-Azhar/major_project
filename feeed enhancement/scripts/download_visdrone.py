"""Download VisDrone2019-DET validation set."""
import os
import sys
import zipfile
import urllib.request

VISDRONE_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
DEST_DIR = os.path.join("data", "visdrone")
ZIP_PATH = os.path.join(DEST_DIR, "VisDrone2019-DET-val.zip")


def download_with_progress(url: str, dest: str):
    """Download a file with progress reporting."""
    print(f"Downloading: {url}", flush=True)
    print(f"Destination: {dest}", flush=True)

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print(flush=True)


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    # Check if already extracted
    images_dir = os.path.join(DEST_DIR, "VisDrone2019-DET-val", "images")
    annot_dir = os.path.join(DEST_DIR, "VisDrone2019-DET-val", "annotations")

    if os.path.isdir(images_dir) and os.path.isdir(annot_dir):
        n_img = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        n_ann = len([f for f in os.listdir(annot_dir) if f.endswith(".txt")])
        if n_img >= 548 and n_ann >= 548:
            print(f"VisDrone val set already present: {n_img} images, {n_ann} annotations", flush=True)
            return

    # Download
    if not os.path.exists(ZIP_PATH):
        download_with_progress(VISDRONE_VAL_URL, ZIP_PATH)
    else:
        print(f"ZIP already exists: {ZIP_PATH}", flush=True)

    # Extract
    print("Extracting...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DEST_DIR)
    print("Extraction complete.", flush=True)

    # Validate
    n_img = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    n_ann = len([f for f in os.listdir(annot_dir) if f.endswith(".txt")])
    print(f"Validation: {n_img} images, {n_ann} annotations", flush=True)

    if n_img >= 500 and n_ann >= 500:
        print("Download and extraction successful!", flush=True)
    else:
        print(f"WARNING: Expected ~548 images and annotations, got {n_img}/{n_ann}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
