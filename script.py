#!/usr/bin/env python3
"""
Pipeline with patch-level test-time augmentations:
- For each patch, apply rotations (90/180/270) + horizontal & vertical flips
- Extract Virchow2 features for each augmented version
- Average features across augmentations before building graph
"""

import os
import sys
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import tifffile

import timm
from timm.layers import SwiGLUPacked

from fastcluster import linkage_vector
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist

from torch_geometric.data import Data, Batch
from torchvision import transforms

# ---------------------------
# Configuration
# ---------------------------
PATCH_SIZE = 560
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EDGE_DIST = 3000
MIN_CLUSTERS = 10
DEFAULT_NUM_WORKERS = 1
Image.MAX_IMAGE_PIXELS = None
TMP_FEATURE_ROOT = Path("wsi_features_cache")

# ---------------------------
# Patch transform
# ---------------------------
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Augmentations: rotations + flips
AUGS = [
    lambda x: x,  # identity
    lambda x: x.rotate(90),
    lambda x: x.rotate(180),
    lambda x: x.rotate(270),
    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
]

# ---------------------------
# File discovery
# ---------------------------
def find_wsi_and_mask_in_patient_dir(patient_dir: Path):
    wsi_candidates = [p for p in patient_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".mrxs", ".svs", ".ndpi", ".mha")]
    wsi_candidates = sorted(set(wsi_candidates))
    wsi_path = wsi_candidates[0] if wsi_candidates else None

    mask_candidates = [p for p in patient_dir.iterdir() if "mask" in p.stem.lower()]
    mask_candidates = sorted(set(mask_candidates))
    mask_path = mask_candidates[0] if mask_candidates else None

    return wsi_path, mask_path

# ---------------------------
# Patch extraction
# ---------------------------
def extract_patches_from_mask(wsi_path: str, mask_path: str, patch_size=PATCH_SIZE):
    from tiatoolbox.wsicore.wsireader import WSIReader
    wsi = WSIReader.open(wsi_path)
    mask_np = tifffile.imread(mask_path)

    if mask_np.ndim == 3:
        mask_np = (mask_np[..., 0] > 0).astype(np.uint8) * 255
    elif mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    coords = []
    patches = []
    mask_h, mask_w = mask_np.shape[:2]

    for y in range(0, mask_h - patch_size + 1, patch_size):
        for x in range(0, mask_w - patch_size + 1, patch_size):
            patch_mask = mask_np[y:y+patch_size, x:x+patch_size]
            if patch_mask.size == 0 or patch_mask.mean() == 0:
                continue
            try:
                patch_img = wsi.read_region(location=(x, y), level=0, size=(patch_size, patch_size))
            except Exception:
                continue
            if not isinstance(patch_img, Image.Image):
                patch_img = Image.fromarray(patch_img)

            patches.append(patch_img)
            coords.append((x, y))

    return patches, np.array(coords, dtype=np.float32)

# ---------------------------
# Virchow2 feature extraction with augmentations
# ---------------------------
def extract_virchow2_features(wsi_path: str, mask_path: str, output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    ).to(DEVICE).eval()

    patches, coords = extract_patches_from_mask(wsi_path, mask_path, patch_size=PATCH_SIZE)
    if len(patches) == 0:
        raise RuntimeError(f"No tissue patches found for {wsi_path}")

    # Build augmented dataset: shape = (N_patches * N_augs, C, H, W)
    aug_tensors = []
    for patch in patches:
        for aug in AUGS:
            aug_img = aug(patch)
            aug_tensors.append(base_transform(aug_img))
    aug_tensors = torch.stack(aug_tensors)  # (N * A, C, H, W)

    # Process in batches
    features_list = []
    with torch.no_grad():
        for i in range(0, len(aug_tensors), BATCH_SIZE):
            batch = aug_tensors[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                feats = model.forward_features(batch)
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            elif feats.ndim > 2:
                feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu())

    # Concatenate all augmented features
    all_feats = torch.cat(features_list, dim=0)  # (N * A, D)

    # Reshape back to (N_patches, N_augs, D)
    n_patches = len(patches)
    n_augs = len(AUGS)
    all_feats = all_feats.view(n_patches, n_augs, -1)

    # Average augmentations → (N_patches, D)
    patch_features = all_feats.mean(dim=1)

    # Save
    torch.save(patch_features, outdir / "slide_features.pt")
    np.save(outdir / "slide_coords.npy", coords.astype(np.float32))

    return patch_features, coords

# ---------------------------
# Graph construction
# ---------------------------
def build_pyg_graph_from_patches(patch_features, patch_coords, max_dist=MAX_EDGE_DIST, device='cpu'):
    features_np = patch_features.cpu().numpy()
    coords_np = patch_coords.astype(np.float32)
    fused = np.concatenate([coords_np, features_np], axis=1)

    n_clusters = max(MIN_CLUSTERS, int(len(fused) / 20))
    Z = linkage_vector(fused, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    node_coords, node_features = [], []
    for k in range(1, n_clusters + 1):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        node_coords.append(coords_np[idxs].mean(axis=0))
        node_features.append(features_np[idxs].mean(axis=0))

    node_coords = np.vstack(node_coords)
    node_features = np.vstack(node_features)

    dist_matrix = cdist(node_coords, node_coords)
    edge_list = [(i, j) for i in range(len(node_coords)) for j in range(len(node_coords)) if i != j and dist_matrix[i, j] <= max_dist]
    if len(edge_list) == 0:
        edge_list = [(0, 0)]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(node_features, dtype=torch.float32).to(device),
                edge_index=edge_index.to(device),
                pos=torch.tensor(node_coords, dtype=torch.float32).to(device))

# ---------------------------
# Patient processing
# ---------------------------
def process_patient(patient_dir: Path, label_root: Path, tmp_features_root: Path, max_dist=MAX_EDGE_DIST, device=DEVICE):
    pid = patient_dir.name
    try:
        wsi_path, mask_path = find_wsi_and_mask_in_patient_dir(patient_dir)
        if wsi_path is None or mask_path is None:
            return None

        slide_out_dir = tmp_features_root / pid
        feat_file = slide_out_dir / "slide_features.pt"
        coords_file = slide_out_dir / "slide_coords.npy"

        if feat_file.exists() and coords_file.exists():
            patch_feats = torch.load(str(feat_file))
            coords_np = np.load(str(coords_file))
        else:
            patch_feats, coords_np = extract_virchow2_features(str(wsi_path), str(mask_path), str(slide_out_dir))

        graph = build_pyg_graph_from_patches(patch_feats, coords_np, max_dist=max_dist, device=device)

        json_path = patient_dir / f"{pid}_CD.json"
        if not json_path.exists():
            return None

        with open(json_path, 'r') as f:
            meta = json.load(f)

        time_val = meta.get("Time_to_prog_or_FUend", None)
        progression_val = meta.get("progression", None)
        if time_val is None or progression_val is None:
            return None

        graph.y = torch.tensor([time_val, progression_val], dtype=torch.float32).view(1, 2)
        graph.pid = pid
        return graph
    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        return None

# ---------------------------
# Build batch
# ---------------------------
def build_batch_from_directory(data_root: str, label_root: str = None,
                               tmp_features_root: str = None,
                               max_dist: float = MAX_EDGE_DIST,
                               device: str = DEVICE,
                               num_workers: int = DEFAULT_NUM_WORKERS) -> Batch:
    data_root = Path(data_root)
    label_root = Path(label_root) if label_root else data_root
    tmp_features_root = Path(tmp_features_root) if tmp_features_root else TMP_FEATURE_ROOT
    tmp_features_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    graphs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_patient, p, label_root, tmp_features_root, max_dist, device): p for p in patient_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Graph Construction"):
            result = fut.result()
            if result is not None:
                graphs.append(result)

    y_tensor = torch.cat([g.y for g in graphs], dim=0)
    batch = Batch.from_data_list(graphs)
    batch.y = y_tensor
    batch.pids = [g.pid for g in graphs]
    return batch

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_pkl", type=str, required=True)
    parser.add_argument("--label_root", type=str, default=None)
    parser.add_argument("--tmp_features_root", type=str, default=str(TMP_FEATURE_ROOT))
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    batch = build_batch_from_directory(
        data_root=args.data_root,
        label_root=args.label_root,
        tmp_features_root=args.tmp_features_root,
        num_workers=args.num_workers
    )

    with open(args.output_pkl, "wb") as f:
        pickle.dump(batch, f)
    print(f"[INFO] Saved Batch with {batch.num_graphs} graphs and {batch.num_nodes} nodes to {args.output_pkl}")
