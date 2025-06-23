from pathlib import Path
import json
import torch
import openslide
import numpy as np
import os
from torch import nn, Tensor
from typing import Dict, Union

def encode_clinical(cd: Dict[str, Union[str, int, float]]) -> Tensor:
    """
    Encodes clinical categorical and numerical features into a flat tensor.

    Args:
        cd (Dict[str, Union[str, int, float]]): Clinical dictionary loaded from _CD.json.

    Returns:
        Tensor: A 1D tensor containing encoded clinical features.
    """
    cat_map = {
        "sex": {"Male": 0, "Female": 1},
        "smoking": {"No": 0, "Yes": 1},
        "tumor": {"Primary": 0, "Recurrence": 1},
        "stage": {"TaHG": 0, "T1HG": 1, "T2HG": 2},
        "substage": {"T1m": 0, "T1e": 1},
        "grade": {"G2": 0, "G3": 1},
        "reTUR": {"No": 0, "Yes": 1},
        "LVI": {"No": 0, "Yes": 1},
        "variant": {"UCC": 0, "UCC + Variant": 1},
        "EORTC": {"High risk": 0, "Highest risk": 1},
        "BRS": {"BRS1": 0, "BRS2": 1, "BRS3": 2}
    }

    # Safe numeric encoding
    age = float(cd.get("age", 0.0))
    instills = int(cd.get("no_instillations", -1))

    categorical = []
    for key, mapping in cat_map.items():
        value = cd.get(key, None)
        if value not in mapping:
            print(f"⚠️ Warning: Unexpected value `{value}` for key `{key}` — defaulting to 0.")
            categorical.append(0)
        else:
            categorical.append(mapping[value])

    return torch.tensor([age, instills] + categorical, dtype=torch.float)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim, proj_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.attn = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, x):  # [B, N, D]
        x_proj = self.proj(x)
        weights = self.attn(x_proj).squeeze(-1)  # [B, N]
        attn_weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * x_proj, dim=1)  # [B, proj_dim]
        return pooled


class BaselineModel(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=13,
                 hist_proj_dim=512, hist_feat_dim=256, rna_feat_dim=512, clinical_feat_dim=64):
        super().__init__()

        self.attn_pool = AttentionPooling(hist_dim, hist_proj_dim)

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hist_feat_dim),
            nn.ReLU()
        )

        self.rna_bottleneck = nn.Sequential(
            nn.Linear(rna_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )

        self.rna_encoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, rna_feat_dim),
            nn.ReLU()
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, clinical_feat_dim)
        )

        fusion_input_dim = hist_feat_dim + rna_feat_dim + clinical_feat_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, hist_patches, rna_vec, clinical_vec):
        pooled_hist = self.attn_pool(hist_patches)
        hist_repr = self.hist_encoder(pooled_hist)

        rna_bottled = self.rna_bottleneck(rna_vec)
        rna_repr = self.rna_encoder(rna_bottled)

        clinical_repr = self.clinical_encoder(clinical_vec)

        fused = torch.cat([hist_repr, rna_repr, clinical_repr], dim=1)
        return self.fusion(fused).squeeze(-1)



INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")  # model.pth is here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_interface_key():
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))

def load_json_file(location):
    with open(location, "r") as f:
        return json.load(f)

def write_json_file(location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def extract_patch_features(slide_path, mask_path, patch_size=224, max_patches=512):
    slide = openslide.OpenSlide(slide_path)
    mask = openslide.OpenSlide(mask_path)

    slide_w, slide_h = slide.dimensions
    downsample = int(slide.level_downsamples[0])

    sampled_patches = []
    sampled_coords = []

    step = patch_size * 4  # Downsampled sampling for speed
    count = 0

    for y in range(0, slide_h, step):
        for x in range(0, slide_w, step):
            if count >= max_patches:
                break
            mask_region = mask.read_region((x, y), 0, (patch_size, patch_size)).convert("L")
            mask_np = np.array(mask_region)

            if np.mean(mask_np) > 15:  # Tissue threshold
                region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                region_np = np.array(region).astype(np.float32) / 255.0
                region_np = region_np.transpose(2, 0, 1)
                sampled_patches.append(region_np)
                sampled_coords.append([x / slide_w, y / slide_h])
                count += 1

    if count == 0:
        sampled_patches.append(np.zeros((3, patch_size, patch_size)))
        sampled_coords.append([0.0, 0.0])

    feats = torch.tensor(np.stack(sampled_patches), dtype=torch.float)
    coords = torch.tensor(sampled_coords, dtype=torch.float)

    # Flatten features + coords to match training
    B, C, H, W = feats.shape
    feats = feats.view(B, -1)
    hist_patches = torch.cat([feats, coords], dim=1)  # [B, C*H*W + 2]
    return hist_patches

def run():
    interface_key = get_interface_key()

    if interface_key != (
        "bladder-cancer-tissue-biopsy-whole-slide-image",
        "bulk-rna-seq-bladder-cancer",
        "chimera-clinical-data-of-bladder-cancer-recurrence",
        "tissue-mask",
    ):
        raise ValueError(f"Unsupported interface: {interface_key}")

    wsi_path = list((INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi").glob("*"))[0]
    mask_path = list((INPUT_PATH / "images/tissue-mask").glob("*"))[0]
    rna_path = INPUT_PATH / "bulk-rna-seq-bladder-cancer.json"
    clinical_path = INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"

    hist_tensor = extract_patch_features(str(wsi_path), str(mask_path)).to(device).unsqueeze(0)

    rna_json = load_json_file(rna_path)
    rna_vec = torch.tensor(list(rna_json.values()), dtype=torch.float).unsqueeze(0).to(device)

    clinical_json = load_json_file(clinical_path)
    clinical_vec = encode_clinical(clinical_json).unsqueeze(0).to(device)

    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(RESOURCE_PATH / "bs_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(hist_tensor, rna_vec, clinical_vec)
        likelihood = round(output.item(), 2)

    write_json_file(
        OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json",
        likelihood
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(run())
