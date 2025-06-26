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

def encode_clinical(cd):
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
            # print(f"⚠️ Warning: Unexpected value `{value}` for key `{key}` — defaulting to 0.")
            categorical.append(0)
        else:
            categorical.append(mapping[value])

    return torch.tensor([age, instills] + categorical, dtype=torch.float)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim, proj_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.attn = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, x):
        x_proj = self.proj(x)
        weights = self.attn(x_proj).squeeze(-1)
        attn_weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * x_proj, dim=1)
        return pooled

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class BaselineEncoder(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=13,
                 hist_proj_dim=1024, hist_feat_dim=512,
                 rna_feat_dim=1024, clinical_feat_dim=128):
        super().__init__()
        self.attn_pool = AttentionPooling(hist_dim, hist_proj_dim)

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_proj_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            ResidualBlock(1024),
            nn.Linear(1024, hist_feat_dim),
            nn.ReLU()
        )

        self.rna_bottleneck = nn.Sequential(
            nn.Linear(rna_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.rna_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            ResidualBlock(1024),
            nn.Linear(1024, rna_feat_dim),
            nn.ReLU()
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, clinical_feat_dim)
        )

        self.out_dim = hist_feat_dim + rna_feat_dim + clinical_feat_dim

    def forward(self, hist_patches, rna_vec, clinical_vec):
        pooled_hist = self.attn_pool(hist_patches)
        hist_repr = self.hist_encoder(pooled_hist)
        rna_bottled = self.rna_bottleneck(rna_vec)
        rna_repr = self.rna_encoder(rna_bottled)
        clinical_repr = self.clinical_encoder(clinical_vec)
        return torch.cat([hist_repr, rna_repr, clinical_repr], dim=1)

class ProgressionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # logits for 2 classes
        )

    def forward(self, x):
        return self.classifier(x)

class ProgressionEmbedding(nn.Module):
    def __init__(self, num_classes=2, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, class_logits):
        class_probs = torch.softmax(class_logits, dim=1)
        pred_class = class_probs.argmax(dim=1)
        return self.embedding(pred_class)

class SurvivalRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            ResidualBlock(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)

class TwoStageModel(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=13):
        super().__init__()
        self.encoder = BaselineEncoder(hist_dim, rna_dim, clinical_dim)
        self.classifier = ProgressionClassifier(self.encoder.out_dim)
        self.prog_embed = ProgressionEmbedding()
        self.survival_regressor = SurvivalRegressor(self.encoder.out_dim + 32)

    def forward(self, hist_patches, rna_vec, clinical_vec):
        features = self.encoder(hist_patches, rna_vec, clinical_vec)
        class_logits = self.classifier(features)
        prog_embedding = self.prog_embed(class_logits)
        surv_input = torch.cat([features, prog_embedding], dim=1)
        risk_score = self.survival_regressor(surv_input)
        return class_logits, risk_score

def train_model_two_stage(model, train_loader, val_loader, fold_idx, writer, 
                          lambda_cls=1.0, lambda_surv=1.0, patience=10, max_epochs=100, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, total_iters=5), CosineAnnealingLR(optimizer, T_max=max_epochs - 5)],
        milestones=[5]
    )
    bce_loss = nn.CrossEntropyLoss()
    cox_loss = CoxPHLoss()

    best_val_cindex = 0
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/fold_{fold_idx + 1}_best_2stage.pth"

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        skipped_batches = 0

        for batch in train_loader:
            hist = batch['hist_patches'].to(device)
            rna = batch['rna_vec'].to(device)
            clinical = batch['clinical_vec'].to(device)
            time = batch['time'].to(device).float()
            event = batch['event'].to(device).float()
            progression = batch['event'].long().to(device)

            optimizer.zero_grad()
            class_logits, risk_score = model(hist, rna, clinical)
            risk_score = torch.clamp(risk_score, min=-30.0, max=30.0)

            loss_cls = bce_loss(class_logits, progression)
            loss_surv = cox_loss(risk_score, time, event)
            loss = lambda_cls * loss_cls + lambda_surv * loss_surv

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader) - skipped_batches, 1)

        # Validation
        model.eval()
        val_preds, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist_patches'].to(device)
                rna = batch['rna_vec'].to(device)
                clinical = batch['clinical_vec'].to(device)
                _, risk_score = model(hist, rna, clinical)
                val_preds.extend(risk_score.cpu().numpy())
                val_times.extend(batch['time'].cpu().numpy())
                val_events.extend(batch['event'].cpu().numpy())

        val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)], dtype=[('event', '?'), ('time', '<f8')])
        val_cindex = concordance_index_censored(val_struct['event'], val_struct['time'], -np.array(val_preds))[0]

        writer.add_scalar(f"Fold{fold_idx}/TrainLoss", avg_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/ValCIndex", val_cindex, epoch)
        print(f"Epoch {epoch + 1:03d} | Fold {fold_idx + 1} | Loss: {avg_loss:.4f} | C-Index: {val_cindex:.4f}")

        scheduler.step()
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} for fold {fold_idx + 1}")
                break

    return best_val_cindex



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

    hist_patches = torch.tensor(patch_features_with_coords, dtype=torch.float32).unsqueeze(0).to(device)

    rna_vec = torch.tensor(np.array(list(rna.values())), dtype=torch.float32).unsqueeze(0).to(device)
    clinical_vec = encode_clinical(clinical).unsqueeze(0).to(device)

    model = TwoStageModel().to(device)
    model.load_state_dict(torch.load("resources/bs_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        class_logits, risk_score = model(hist_patches, rna_vec, clinical_vec)
        likelihood = float(torch.sigmoid(class_logits)[0][1].cpu()) * 100

    write_json_file(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", round(likelihood, 1))

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
    state_dict = torch.load("resources/densenet121-a639ec97.pth", map_location=device)
    state_dict = rename_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
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
