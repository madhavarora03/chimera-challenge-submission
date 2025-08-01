import os
import torch
import numpy as np
import pickle
import json
from fastcluster import linkage_vector
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_pyg_graph_from_patches(patch_features: torch.Tensor,
                                  patch_coords: np.ndarray,
                                  max_dist: float = 1500,
                                  device: str = 'cpu') -> Data:
    features_np = patch_features.cpu().numpy()
    coords_np = patch_coords.astype(np.float32)
    fused_features = np.concatenate([coords_np, features_np], axis=1)

    n_clusters = max(10, int(len(fused_features) / 20))
    Z = linkage_vector(fused_features, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    node_coords, node_features = [], []
    for k in range(1, n_clusters + 1):
        idxs = np.where(labels == k)[0]
        cluster_coords = coords_np[idxs]
        cluster_feats = features_np[idxs]
        gc = np.mean(cluster_coords, axis=0)
        hf = np.mean(cluster_feats, axis=0)
        node_coords.append(gc)
        node_features.append(hf)

    node_coords = torch.tensor(np.vstack(node_coords), dtype=torch.float32)
    node_features = torch.tensor(np.vstack(node_features), dtype=torch.float32)

    dist_matrix = cdist(node_coords.numpy(), node_coords.numpy())
    edge_list = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if dist_matrix[i, j] <= max_dist:
                edge_list.append((i, j))
                edge_list.append((j, i))

    if len(edge_list) == 0:
        for i in range(n_clusters - 1):
            edge_list.append((i, i + 1))
            edge_list.append((i + 1, i))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(
        x=node_features.to(device),
        edge_index=edge_index.to(device),
        pos=node_coords.to(device)
    )


def process_patient(pid, features_dir, coords_dir, label_dir, max_dist, device):
    feat_path = os.path.join(features_dir, f"{pid}_HE.pt")
    coord_path = os.path.join(coords_dir, f"{pid}_HE.npy")
    json_path = os.path.join(label_dir, pid, f"{pid}_CD.json")

    if not os.path.exists(coord_path) or not os.path.exists(json_path):
        print(f"[WARN] Missing data for {pid}, skipping.")
        return None

    try:
        patch_feats = torch.load(feat_path)
        coords_struct = np.load(coord_path)
        patch_coords = np.stack([coords_struct["x"], coords_struct["y"]], axis=1).astype(np.float32)
        graph = build_pyg_graph_from_patches(patch_feats, patch_coords, max_dist=max_dist, device=device)

        with open(json_path, 'r') as f:
            meta = json.load(f)

        time_val = meta.get("Time_to_prog_or_FUend", None)
        progression_val = meta.get("progression", None)

        if time_val is None or not isinstance(time_val, (int, float)):
            print(f"[WARN] Invalid Time_to_prog_or_FUend for {pid}, skipping.")
            return None
        if progression_val is None or not isinstance(progression_val, int):
            print(f"[WARN] Invalid progression for {pid}, skipping.")
            return None

        # Ensure y is of shape (2,) not scalar or (2, 1)
        graph.y = torch.tensor([time_val, progression_val], dtype=torch.float32).view(1, 2)
        graph.pid = pid
        return graph
    except Exception as e:
        print(f"[ERROR] Failed processing {pid}: {e}")
        return None


def build_batch_from_directory(input_dir: str, max_dist: float = 1500, device: str = 'cpu',
                               num_workers: int = 8, label_dir: str = None) -> Batch:
    features_dir = os.path.join(input_dir, 'features')
    coords_dir = os.path.join(input_dir, 'coordinates')

    pids = sorted([f.replace('_HE.pt', '') for f in os.listdir(features_dir) if f.endswith('.pt')])
    print(f"[INFO] Found {len(pids)} patients. Processing in {num_workers} threads...")

    graphs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_patient, pid, features_dir, coords_dir, label_dir, max_dist, device)
            for pid in pids
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Graph Construction"):
            result = future.result()
            if result is not None:
                graphs.append(result)

    # Stack the y values from each graph: (N, 2)
    y_tensor = torch.cat([g.y for g in graphs], dim=0)

    batch = Batch.from_data_list(graphs, follow_batch=[])
    batch.y = y_tensor  # Attach batched y manually
    batch.pids = [g.pid for g in graphs]
    return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a PyG Batch object from patch features/coords.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with 'features/' and 'coordinates/' subdirs.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory with patient folders and *_CD.json files.")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output filename (.pkl) to store Batch object.")
    parser.add_argument("--max_dist", type=float, default=1500, help="Max distance for edge creation.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to store graph data on.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel threads to use.")
    args = parser.parse_args()

    batch = build_batch_from_directory(
        input_dir=args.input_dir,
        max_dist=args.max_dist,
        device=args.device,
        num_workers=args.num_workers,
        label_dir=args.label_dir
    )

    print(f"[INFO] Saving Batch with {batch.num_graphs} graphs and {batch.num_nodes} nodes to {args.output_pkl}")
    with open(args.output_pkl, "wb") as f:
        pickle.dump(batch, f)

    print(f"[INFO] Example pids in batch: {batch.pids[:5]}")
    print(f"[INFO] Final batch.y shape: {batch.y.shape}")
