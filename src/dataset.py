import json
import os
from os import PathLike
from typing import List, Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils import encode_clinical


class ChimeraDataset(Dataset):
    """
    PyTorch Dataset for loading multimodal patient data for the Chimera Challenge Task 3, with pre-extracted features.

    Each sample includes:
    - Histopathology patch features and spatial coordinates
    - RNA expression vector
    - Clinical metadata
    - Survival outcome labels (time to recurrence and progression status)

    Args:
        patient_ids (List[str]): List of patient identifiers.
        features_dir (str | PathLike): Directory path containing patch-level histopathology features (*.pt).
        coords_dir (str | PathLike): Directory path containing patch spatial coordinates (*.npy).
        data_dir (str | PathLike): Root directory containing subfolders for each patient with *_RNA.json and *_CD.json.
    """

    def __init__(
            self,
            patient_ids: List[str],
            features_dir: str | PathLike,
            coords_dir: str | PathLike,
            data_dir: str | PathLike
    ) -> None:
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.coords_dir = coords_dir
        self.data_dir = data_dir

    def __len__(self) -> int:
        """
        Returns:
            int: Number of patients in the dataset.
        """
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor, float, int]]:
        """
        Loads a single patient sample with all associated modalities.

        Args:
            idx (int): Index of the patient in the list.

        Returns:
            Dict[str, Union[str, Tensor, float, int]]: A dictionary containing:
                - "pid": Patient ID
                - "hist_feats": Tensor of shape [N, D] (patch features)
                - "hist_coords": Tensor of shape [N, 2] (x, y coordinates)
                - "rna_vec": RNA expression vector
                - "clinical_vec": Encoded clinical features
                - "time": Time to recurrence or follow-up
                - "event": Progression event flag (1 or 0)
        """
        pid = str(self.patient_ids[idx])

        # Load histopathology features
        hist_feats_path = os.path.join(self.features_dir, f"{pid}_HE.pt")
        hist_feats = torch.load(hist_feats_path)  # [N_patches, feature_dim]

        # Load coordinates
        hist_coords_path = os.path.join(self.coords_dir, f"{pid}_HE.npy")
        hist_coords_raw = np.load(hist_coords_path)

        # If structured array, extract only x and y coordinates
        if hist_coords_raw.dtype.fields is not None:
            selected_fields = ["x", "y"]
            hist_coords = np.stack([
                hist_coords_raw[field].astype(np.float32)
                for field in selected_fields
            ], axis=1)
        else:
            hist_coords = hist_coords_raw.astype(np.float32)

        hist_coords = torch.from_numpy(hist_coords)  # [num_patches, 3]

        # Load RNA expression vector
        rna_path = os.path.join(self.data_dir, pid, f"{pid}_RNA.json")
        with open(rna_path, 'r') as f:
            rna_vec = torch.tensor(list(json.load(f).values()), dtype=torch.float)

        # Load clinical data
        cd_path = os.path.join(self.data_dir, pid, f"{pid}_CD.json")
        with open(cd_path, 'r') as f:
            cd = json.load(f)
        clinical_vec = encode_clinical(cd)

        # Labels
        y_time = float(cd["time_to_HG_recur_or_FUend"])
        y_event = int(cd["progression"])

        return {
            "pid": pid,
            "hist_feats": hist_feats,
            "hist_coords": hist_coords,
            "rna_vec": rna_vec,
            "clinical_vec": clinical_vec,
            "time": y_time,
            "event": y_event,
        }


def test():
    patient_ids = ["3A_001"]
    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get first batch
    batch = next(iter(dataloader))

    print("✅ ChimeraDataset Test Output:")
    print("=" * 40)

    print(f"Patient ID: {batch['pid'][0]}")
    print(f"Histopathology features: {batch['hist_feats'].shape} | dtype: {batch['hist_feats'].dtype}")
    print(f"Histopathology coordinates: {batch['hist_coords'].shape} | dtype: {batch['hist_coords'].dtype}")
    print(f"RNA vector: {batch['rna_vec'].shape} | dtype: {batch['rna_vec'].dtype}")
    print(f"Clinical vector: {batch['clinical_vec'].shape} | dtype: {batch['clinical_vec'].dtype}")
    print(f"Time to event: {batch['time'].item()}")
    print(f"Progression event: {batch['event'].item()}")

    print("=" * 40)


if __name__ == "__main__":
    test()
