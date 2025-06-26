from pathlib import Path
import json
from glob import glob
import numpy as np
import openslide
import tifffile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    tissue_mask = load_slide_image(INPUT_PATH / "images/tissue-mask")
    wsi_slide = load_slide(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi")

    rna = load_json_file(INPUT_PATH / "bulk-rna-seq-bladder-cancer.json")
    clinical = load_json_file(INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json")

    _show_torch_cuda_info()

    patch_features_with_coords = extract_patch_features(wsi_slide)

    print(f"✅ Final patch features shape: {patch_features_with_coords.shape}")

    output = round(random.uniform(0, 80), 1)
    write_json_file(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", output)

    return 0


def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k_new = k.replace("norm.1", "norm1").replace("norm.2", "norm2")\
                 .replace("conv.1", "conv1").replace("conv.2", "conv2")
        new_state_dict[k_new] = v
    return new_state_dict


def extract_patch_features(slide, patch_size=224, stride=224, max_patches=512):
    width, height = slide.dimensions
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = models.densenet121(weights=None)
    state_dict = torch.load("resources/densenet121-a639ec97.pth", map_location="cpu")
    state_dict = rename_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()

    features = []
    coords = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    feat = model(patch_tensor).cpu().numpy()
                features.append(feat[0])
                coords.append([x, y])
            except Exception as e:
                print(f"⚠️ Skipping patch at ({x}, {y}): {e}")
            if len(features) >= max_patches:
                break
        if len(features) >= max_patches:
            break

    features = np.array(features)
    coords = np.array(coords)

    assert features.shape[0] == coords.shape[0], "Mismatch between features and coordinates count!"
    return np.concatenate([features, coords], axis=1)


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


def load_slide(location):
    files = glob(str(location / "*"))
    if not files:
        raise FileNotFoundError(f"No slide image found in {location}")
    return openslide.OpenSlide(files[0])


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Torch CUDA available:", (avail := torch.cuda.is_available()))
    if avail:
        print("Device count:", torch.cuda.device_count())
        device = torch.cuda.current_device()
        print("Device properties:", torch.cuda.get_device_properties(device))
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
