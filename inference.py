# -------------------- FULL MATCHED INFERENCE SCRIPT --------------------
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

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_clinical(cd):
    """
    One-hot encodes clinical categorical features and appends normalized numerical features.

    Args:
        cd (Dict[str, Union[str, int, float]]): Clinical dictionary loaded from _CD.json.

    Returns:
        Tensor: A 1D tensor containing one-hot encoded clinical features.
    """
    cat_map = {
        "sex": ["Male", "Female"],
        "smoking": ["No", "Yes"],
        "tumor": ["Primary", "Recurrence"],
        "stage": ["TaHG", "T1HG", "T2HG"],
        "substage": ["T1m", "T1e"],
        "grade": ["G2", "G3"],
        "reTUR": ["No", "Yes"],
        "LVI": ["No", "Yes"],
        "variant": ["UCC", "UCC + Variant"],
        "EORTC": ["High risk", "Highest risk"],
        "BRS": ["BRS1", "BRS2", "BRS3"]
    }

    # One-hot encoding categorical variables
    one_hot = []
    for key, options in cat_map.items():
        vec = [0] * len(options)
        value = cd.get(key, None)
        if value in options:
            vec[options.index(value)] = 1
        else:
            vec[0] = 1  # Default to first class if unknown
            # print(f"⚠️ Warning: Unexpected value `{value}` for key `{key}` — defaulting to {options[0]}.")
        one_hot.extend(vec)

    # Append numerical features (you can normalize if needed)
    age = float(cd.get("age", 0.0))
    instills = float(cd.get("no_instillations", -1))
    numerical = [age, instills]

    return torch.tensor(numerical + one_hot, dtype=torch.float)

class TopKAttentionPooling(nn.Module):
    def __init__(self, dim, k=10):
        super().__init__()
        self.k = k
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        weights = self.attn(x).squeeze(-1)
        topk = torch.topk(weights, self.k, dim=0).indices
        return x[topk].mean(dim=0)

class ExpertNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class GatingNet(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

class MoMoEDeepSurv(nn.Module):
    def __init__(self, full_dim, splits, hidden_dim=224):
        super().__init__()
        self.splits = splits
        self.experts = nn.ModuleList([
            ExpertNet(s, hidden_dim) for s in splits
        ])
        self.gate = GatingNet(full_dim, len(splits))

    def forward(self, x):
        h_dim, r_dim, c_dim = self.splits
        hist = x[:, :h_dim]
        rna = x[:, h_dim:h_dim + r_dim]
        clin = x[:, -c_dim:]
        expert_inputs = [hist, rna, clin]
        expert_outputs = [e(inp).squeeze(-1) for e, inp in zip(self.experts, expert_inputs)]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights = self.gate(x)
        out = (expert_outputs * gate_weights).sum(dim=1, keepdim=True)
        return out

def build_model(input_dim, splits, hidden_dim=224, lr=1e-3):
    net = MoMoEDeepSurv(input_dim, splits, hidden_dim)
    model = CoxPH(net, optimizer=torch.optim.Adam)
    model.optimizer.set_lr(lr)
    return model

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
    state_dict = torch.load("resources/densenet121-a639ec97.pth", map_location=device)
    model.load_state_dict(rename_state_dict_keys(state_dict))
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()

    features, coords = [], []
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

def interf0_handler():
    tissue_mask = load_slide_image(INPUT_PATH / "images/tissue-mask")
    wsi_slide = load_slide(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi")
    rna = load_json_file(INPUT_PATH / "bulk-rna-seq-bladder-cancer.json")
    clinical = load_json_file(INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json")
    _show_torch_cuda_info()

    patch_features_with_coords = extract_patch_features(wsi_slide)
    print(f"✅ Final patch features shape: {patch_features_with_coords.shape}")

    hist_patches = torch.tensor(patch_features_with_coords, dtype=torch.float32).unsqueeze(0).to(device)
    pooled_feat = TopKAttentionPooling(dim=1024).to(device)(hist_patches.squeeze(0)[:, :-2])

    rna_vec = torch.tensor(np.array(list(rna.values())), dtype=torch.float32).to(device)
    clinical_vec = encode_clinical(clinical).to(device)
    x = torch.cat([pooled_feat, rna_vec, clinical_vec]).unsqueeze(0).to(device)

    h_dim = pooled_feat.shape[0]
    r_dim = rna_vec.shape[0]
    c_dim = clinical_vec.shape[0]
    input_dim = h_dim + r_dim + c_dim
    splits = (h_dim, r_dim, c_dim)

    model = build_model(input_dim=input_dim, splits=splits)
    state_dict = torch.load("resources/momoe_fold5.pt", map_location=device)
    model.net.load_state_dict(state_dict)
    model.net.to(device)
    model.net.eval()

    with torch.no_grad():
        risk_score = model.net(x)
        likelihood = float(risk_score.item())

    write_json_file(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", round(likelihood, 1))
    return 0

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

def get_interface_key():
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))

def load_json_file(location):
    with open(location, "r") as f:
        return json.load(f)

def write_json_file(location, content):
    with open(location, "w") as f:
        json.dump(content, f, indent=4)

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
