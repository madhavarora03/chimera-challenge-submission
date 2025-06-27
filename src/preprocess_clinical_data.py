import os
import json
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import MinMaxScaler

def normalize_and_update_age(json_root_dir):
    records = []
    all_keys = set()
    patient_to_file = {}

    # Step 1: Load and flatten all _CD.json files
    for root, dirs, files in os.walk(json_root_dir):
        for file in files:
            if file.endswith('_CD.json'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        flat_data = json_normalize(data)
                        patient_id = os.path.basename(root)
                        flat_data['patient_id'] = patient_id
                        records.append(flat_data)
                        all_keys.update(flat_data.columns)
                        patient_to_file[patient_id] = path
                except json.JSONDecodeError:
                    print(f"❌ Skipping invalid JSON: {path}")

    if not records:
        print("⚠️ No valid CD JSON files found.")
        return

    df = pd.concat(records, ignore_index=True)

    for key in all_keys:
        if key not in df.columns:
            df[key] = pd.NA

    # Step 2: Normalize age using MinMaxScaler
    if 'age' not in df.columns:
        print("⚠️ 'age' column not found, skipping normalization.")
        return

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    scaler = MinMaxScaler()
    df['age_normalized'] = scaler.fit_transform(df[['age']])

    # Step 3: Update original JSON files with normalized age
    updated_count = 0
    for i, row in df.iterrows():
        patient_id = row['patient_id']
        normalized_age = row['age_normalized']

        if pd.isna(normalized_age) or patient_id not in patient_to_file:
            print(f"⚠️ Skipping patient {patient_id} (invalid or missing age)")
            continue

        json_path = patient_to_file[patient_id]
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            data['age'] = float(normalized_age)

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

            updated_count += 1
        except Exception as e:
            print(f"❌ Error updating {json_path}: {e}")

    print(f"✅ Updated 'age' in {updated_count} JSON files.")

# === Usage ===
json_root_dir = "data"
normalize_and_update_age(json_root_dir)
