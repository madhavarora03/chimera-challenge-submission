import json
from pathlib import Path
import pandas as pd
import numpy as np
from harmony import harmonize
import torch

# --------------------------------------------
# Step 1: Load RNA-seq JSON files
# --------------------------------------------
base_dir = Path("data")  # 🔁 Your input RNA JSON directory

rna_data = {}
for patient_dir in base_dir.iterdir():
    if patient_dir.is_dir():
        json_file = patient_dir / f"{patient_dir.name}_RNA.json"
        if json_file.exists():
            with open(json_file) as f:
                rna_data[patient_dir.name] = json.load(f)

# --------------------------------------------
# Step 2: Convert to DataFrame
# --------------------------------------------
df = pd.DataFrame.from_dict(rna_data, orient='index')
df = df.apply(pd.to_numeric, errors='coerce')  # ensure numeric
df = df.dropna(axis=1, how='any')              # drop genes with any NaNs

print("Raw RNA-seq shape:", df.shape)  # (patients, genes)

# --------------------------------------------
# Step 3: Extract batch info from patient IDs
# --------------------------------------------
df_meta = pd.DataFrame({
    'batch': df.index.str.slice(0, 2)  # e.g. '3A', '3B'
}, index=df.index)

# --------------------------------------------
# Step 4: Apply Harmony for batch correction
# --------------------------------------------
X_harmony = harmonize(df.to_numpy(), df_meta, batch_key='batch')

# --------------------------------------------
# Step 5: Save corrected JSON files
# --------------------------------------------
df_corrected = pd.DataFrame(X_harmony, index=df.index, columns=df.columns)

output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

for patient_id in df_corrected.index:
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    patient_data = df_corrected.loc[patient_id].to_dict()
    json_path = patient_dir / f"{patient_id}_RNA.json"

    with open(json_path, "w") as f:
        json.dump(patient_data, f, indent=2)

print("✅ Batch-corrected RNA JSON files saved to:", output_dir)
