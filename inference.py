from pathlib import Path
import json
from glob import glob
import numpy as np
import openslide
import tifffile
import random

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def run():
    interface_key = get_interface_key()

    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "bulk-rna-seq-bladder-cancer",
            "chimera-clinical-data-of-bladder-cancer-recurrence",
            "tissue-mask",
        ): interf0_handler,
    }[interface_key]

    return handler()


def interf0_handler():
    # Load images
    tissue_mask = load_slide_image(INPUT_PATH / "images/tissue-mask")
    wsi = load_slide_image(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi")

    # Load JSONs
    rna = load_json_file(INPUT_PATH / "bulk-rna-seq-bladder-cancer.json")
    clinical = load_json_file(INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json")

    # Debug
    print("=+=" * 10)
    print("Tissue mask shape:", tissue_mask.shape)
    print("WSI shape:", wsi.shape)
    print("RNA keys:", list(rna.keys()))
    print("Clinical keys:", list(clinical.keys()))
    print("=+=" * 10)

    _show_torch_cuda_info()

    # Dummy prediction
    output = round(random.uniform(0, 80), 1)
    write_json_file(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", output)

    return 0


def get_interface_key():
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(location):
    with open(location, "r") as f:
        return json.load(f)


def write_json_file(location, content):
    with open(location, "w") as f:
        json.dump(content, f, indent=4)


def load_slide_image(location):
    files = glob(str(location / "*"))
    if not files:
        raise FileNotFoundError(f"No image found in {location}")
    file = files[0]
    print(f"Loading image: {file}")

    ext = Path(file).suffix.lower()
    if ext in [".svs", ".tif", ".ndpi", ".mrxs"]:
        try:
            slide = openslide.OpenSlide(file)
            thumbnail = slide.get_thumbnail((1024, 1024))
            return np.array(thumbnail)
        except Exception as e:
            print(f"OpenSlide failed: {e}")

    try:
        image = tifffile.imread(file)
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image {file} using OpenSlide or tifffile: {e}")


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Torch CUDA available:", (avail := torch.cuda.is_available()))
    if avail:
        print("Device count:", torch.cuda.device_count())
        device = torch.cuda.current_device()
        print("Device properties:", torch.cuda.get_device_properties(device))
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
