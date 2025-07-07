import os
import numpy as np
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from dataset import ChimeraDataset
from utils import chimera_collate_fn, seed_everything

# -------------------- Attention DeepSurv --------------------

class AttentionDeepSurv(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for 4 heads"
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        attn_output, _ = self.attn(x, x, x)
        return self.mlp(attn_output.squeeze(1))

# -------------------- Feature Pooling --------------------

class TopKAttentionPooling(nn.Module):
    def __init__(self, dim, k=10):
        super().__init__()
        self.k = k
        self.attn = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = self.attn(x).squeeze(-1)
        topk = torch.topk(weights, self.k, dim=0).indices
        selected = x[topk]
        return selected.mean(dim=0)

def extract_patient_features(dataset: ChimeraDataset, device='cpu') -> List[dict]:
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=chimera_collate_fn)
    pooling = TopKAttentionPooling(dim=1024).to(device)
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            pid = batch['pid'][0]
            hist = batch['hist_patches'].squeeze(0).to(device)
            patch_feats = hist[:, :-2]
            pooled_feat = pooling(patch_feats)

            rna = batch['rna_vec'].squeeze(0).cpu().numpy()
            clin = batch['clinical_vec'].squeeze(0).cpu().numpy()
            time = float(batch['time'][0])
            event = int(batch['event'][0])

            final_feature = np.concatenate([pooled_feat.cpu().numpy(), rna, clin])
            samples.append({
                "pid": pid, "x": final_feature,
                "time": time, "event": event
            })
    return samples

# -------------------- Training Function --------------------

def train_attention_deepsurv(samples: List[dict], patient_ids: List[str], best_params, device='cpu'):
    X = np.stack([s['x'] for s in samples])
    y_struct = np.array([(bool(s['event']), s['time']) for s in samples],
                        dtype=[("event", "?"), ("time", "<f8")])

    cohorts = [0 if pid.startswith("3A") else 1 for pid in patient_ids]
    strat_labels = [f"{c}_{int(s['time'])}" for c, s in zip(cohorts, samples)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    os.makedirs("checkpoints", exist_ok=True)
    cindices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels)):
        print(f"\n🌀 Fold {fold + 1} -------------------------")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_struct[train_idx], y_struct[val_idx]

        scaler = StandardScaler()
        X_train_scaled = np.ascontiguousarray(scaler.fit_transform(X_train))
        X_val_scaled = np.ascontiguousarray(scaler.transform(X_val))

        durations_train = y_train["time"].copy()
        events_train = y_train["event"].copy()
        durations_val = y_val["time"].copy()
        events_val = y_val["event"].copy()

        net = AttentionDeepSurv(X.shape[1], best_params['hidden_dim'], best_params['dropout'])
        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(best_params['lr'])

        model.fit(X_train_scaled, (durations_train, events_train),
                  batch_size=64,
                  callbacks=[tt.callbacks.EarlyStopping(patience=10)],
                  val_data=(X_val_scaled, (durations_val, events_val)),
                  verbose=False)

        model.compute_baseline_hazards()

        model_path = f"checkpoints/attn_deepsurv_fold_{fold + 1}.pt"
        torch.save(model.net.state_dict(), model_path)
        print(f"💾 Saved model checkpoint: {model_path}")

        surv = model.predict_surv_df(X_val_scaled)
        ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
        cindex = ev.concordance_td()
        print(f"✅ Fold {fold + 1} C-Index: {cindex:.4f}")
        cindices.append(cindex)

    print(f"\n🔥 Mean C-Index (Attention DeepSurv): {np.mean(cindices):.4f}")

# -------------------- Main --------------------

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = {
        "hidden_dim": 224,
        "dropout": 0.36573746975198096,
        "lr": 0.00020292719266061987,
    }

    patient_ids = sorted([pid for pid in os.listdir("data") if pid not in [".gitkeep", "task3_quality_control.csv"]])
    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data",
        max_patches=1024,
    )

    print("📦 Extracting patient features...")
    samples = extract_patient_features(dataset, device=device)

    print("📊 Training Attention DeepSurv (CoxPH) model...")
    train_attention_deepsurv(samples, patient_ids=patient_ids, best_params=best_params, device=device)

if __name__ == "__main__":
    main()