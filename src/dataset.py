import json
import os
from os import PathLike
from typing import List, Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils import encode_clinical, chimera_collate_fn


class ChimeraDataset(Dataset):
    """
    Chimera Challenge Dataset with random sampling of patch features per patient.

    Each sample includes:
    - Randomly sampled (or padded) histopathology patch features with coordinates
    - RNA expression vector
    - Clinical metadata
    - Survival labels

    Args:
        patient_ids (List[str])
        features_dir (str | PathLike)
        coords_dir (str | PathLike)
        data_dir (str | PathLike)
        max_patches (int): Number of patches to sample per patient (default=512)
    """

    def __init__(
            self,
            patient_ids: List[str],
            features_dir: str | PathLike,
            coords_dir: str | PathLike,
            data_dir: str | PathLike,
            max_patches: int = 512
    ) -> None:
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.coords_dir = coords_dir
        self.data_dir = data_dir
        self.max_patches = max_patches

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor, float, int]]:
        pid = self.patient_ids[idx]

        # Load features and coordinates
        feats = torch.load(os.path.join(self.features_dir, f"{pid}_HE.pt"))  # [N, D]
        coords_np = np.load(os.path.join(self.coords_dir, f"{pid}_HE.npy"))
        coords = (np.stack([coords_np['x'], coords_np['y']], axis=1) if coords_np.dtype.fields
                  else coords_np[:, :2]).astype(np.float32)
        coords = torch.from_numpy(coords)  # [N, 2]

        hist_patches = torch.cat([feats, coords], dim=1)  # [N, D+2]
        num_patches = hist_patches.shape[0]

        if num_patches >= self.max_patches:
            indices = torch.randperm(num_patches)[:self.max_patches]
            hist_sampled = hist_patches[indices]
        else:
            pad_size = self.max_patches - num_patches
            pad_tensor = torch.zeros(pad_size, hist_patches.size(1))
            hist_sampled = torch.cat([hist_patches, pad_tensor], dim=0)

        # RNA vector
        with open(os.path.join(self.data_dir, pid, f"{pid}_RNA.json"), 'r') as f:
            rna_vec = torch.tensor(list(json.load(f).values()), dtype=torch.float)

        # Clinical
        with open(os.path.join(self.data_dir, pid, f"{pid}_CD.json"), 'r') as f:
            cd = json.load(f)
        clinical_vec = encode_clinical(cd)
        y_time = float(cd["time_to_HG_recur_or_FUend"])
        y_event = int(cd["progression"])

        return {
            "pid": pid,
            "hist_patches": hist_sampled,
            "rna_vec": rna_vec,
            "clinical_vec": clinical_vec,
            "time": y_time,
            "event": y_event,
        }


def test():
    patient_ids = sorted([
        pid for pid in os.listdir("data")
        if pid not in [".gitkeep", "task3_quality_control.csv"]
    ])

    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data",
        max_patches=1048576
    )

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=chimera_collate_fn, shuffle=False)
    batch = next(iter(dataloader))

    print("✅ ChimeraDataset Test Output:")
    print("=" * 40)
    print(f"Patient ID: {batch['pid'][0]}")
    print(f"Histopathology patches: {batch['hist_patches'].shape} | dtype: {batch['hist_patches'].dtype}")
    print(f"RNA vector: {batch['rna_vec'].shape} | dtype: {batch['rna_vec'].dtype}")
    print(f"Clinical vector: {batch['clinical_vec'].shape} | dtype: {batch['clinical_vec'].dtype}")
    print(f"Time to event: {batch['time'][0].item()}")
    print(f"Progression event: {batch['event'][0].item()}")
    print("=" * 40)


if __name__ == "__main__":
    test()
