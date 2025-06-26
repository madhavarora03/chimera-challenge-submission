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
