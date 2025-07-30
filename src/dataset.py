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
    Chimera Dataset that loads precomputed graph embeddings per patient
    and concatenates them with RNA expression and clinical features.

    Each sample includes:
    - Precomputed GNN embedding
    - RNA expression vector (filtered to selected genes)
    - Clinical metadata
    - Survival labels
    """

    def __init__(
        self,
        patient_ids: List[str],
        embeddings_dir: str | PathLike,
        data_dir: str | PathLike,
        gene_list_file: str,
    ) -> None:
        self.patient_ids = patient_ids
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir

        # Load gene filter list
        with open(gene_list_file, 'r') as f:
            self.selected_genes = [line.strip() for line in f if line.strip()]
        self.selected_genes_set = set(self.selected_genes)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor, float, int]]:
        pid = self.patient_ids[idx]

        # Load precomputed graph embedding
        emb_path = os.path.join(self.embeddings_dir, f"{pid}.pt")
        gnn_emb = torch.load(emb_path).squeeze(0)  # shape: [D]

        # Load RNA vector (filtered to selected genes)
        with open(os.path.join(self.data_dir, pid, f"{pid}_RNA.json"), 'r') as f:
            rna_dict = json.load(f)
        filtered_rna = [rna_dict[gene] for gene in self.selected_genes if gene in rna_dict]
        rna_vec = torch.tensor(filtered_rna, dtype=torch.float)

        # Load clinical features
        with open(os.path.join(self.data_dir, pid, f"{pid}_CD.json"), 'r') as f:
            cd = json.load(f)
        clinical_vec = encode_clinical(cd)

        # Time and event
        y_time = float(cd["time_to_HG_recur_or_FUend"])
        y_event = int(cd["progression"])

        # Final input vector
        combined_vec = torch.cat([gnn_emb, rna_vec, clinical_vec], dim=0)

        return {
            "pid": pid,
            "input_vec": combined_vec,
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
        embeddings_dir="embeddings",
        data_dir="data",
        gene_list_file="top_2000_variable_genes.txt",
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    print("✅ ChimeraDataset Test Output (w/ embeddings):")
    print("=" * 40)
    print(f"Patient ID: {batch['pid'][0]}")
    print(f"Input vector shape: {batch['input_vec'].shape} | dtype: {batch['input_vec'].dtype}")
    print(f"Time to event: {batch['time'][0].item()}")
    print(f"Progression event: {batch['event'][0].item()}")
    print("=" * 40)


if __name__ == "__main__":
    test()
