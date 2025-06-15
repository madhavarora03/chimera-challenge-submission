from typing import Dict, Union

import torch
from torch import Tensor


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

    numeric = [float(cd["age"]), int(cd.get("no_instillations", -1))]
    categorical = [cat_map[key][cd[key]] for key in cat_map]
    return torch.tensor(numeric + categorical, dtype=torch.float)
