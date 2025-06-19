from typing import Dict, Union

import torch
import os
import numpy as np
from torch.backends import cudnn
import random
from torch import Tensor

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)



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

def chimera_collate_fn(batch):
    batch_dict = {
        "pid": [item["pid"] for item in batch],
        "hist_patches": torch.stack([item["hist_patches"] for item in batch]),  # [B, N, D]
        "rna_vec": torch.stack([item["rna_vec"] for item in batch]),
        "clinical_vec": torch.stack([item["clinical_vec"] for item in batch]),
        "time": torch.tensor([item["time"] for item in batch], dtype=torch.float),
        "event": torch.tensor([item["event"] for item in batch], dtype=torch.long),
    }
    return batch_dict
