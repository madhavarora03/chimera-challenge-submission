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

# -------------------- Attention Pooling --------------------
class TopKAttentionPooling(nn.Module):
    def __init__(self, dim, k=10):
        super().__init__()
        self.k = k
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        weights = self.attn(x).squeeze(-1)
        topk = torch.topk(weights, self.k, dim=0).indices
        return x[topk].mean(dim=0)

# -------------------- Feature Extraction --------------------
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

            x = np.concatenate([pooled_feat.cpu().numpy(), rna, clin])
            samples.append({"pid": pid, "x": x, "time": time, "event": event})
    return samples

# -------------------- Model Components --------------------
class ExpertNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class GatingNet(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

class MoMoEDeepSurv(nn.Module):
    def __init__(self, full_dim, splits, hidden_dim=128):
        super().__init__()
        self.splits = splits  # (hist_dim, rna_dim, clin_dim)
        self.experts = nn.ModuleList([
            ExpertNet(s, hidden_dim) for s in splits
        ])
        self.gate = GatingNet(full_dim, len(splits))

    def forward(self, x):
        h_dim, r_dim, c_dim = self.splits
        hist = x[:, :h_dim]
        rna = x[:, h_dim:h_dim + r_dim]
        clin = x[:, -c_dim:]

        expert_inputs = [hist, rna, clin]
        expert_outputs = [e(inp).squeeze(-1) for e, inp in zip(self.experts, expert_inputs)]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts)

        gate_weights = self.gate(x)  # (B, num_experts)
        out = (expert_outputs * gate_weights).sum(dim=1, keepdim=True)
        return out

def build_model(input_dim, splits, hidden_dim=128, lr=1e-3):
    net = MoMoEDeepSurv(input_dim, splits, hidden_dim)
    model = CoxPH(net, optimizer=torch.optim.Adam)
    model.optimizer.set_lr(lr)
    return model

# -------------------- Training Pipeline --------------------
def train_pipeline(samples, patient_ids, modality_splits, device='cpu'):
    X = np.stack([s['x'] for s in samples])
    y_struct = np.array([(bool(s['event']), s['time']) for s in samples],
                        dtype=[("event","?"),("time","<f8")])

    scaler = StandardScaler()
    X_scaled = np.ascontiguousarray(scaler.fit_transform(X))

    cohorts = [0 if pid.startswith("3A") else 1 for pid in patient_ids]
    strat_labels = [f"{int(e)}_{c}" for e, c in zip(y_struct['event'].astype(int), cohorts)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = build_model(X_scaled.shape[1], modality_splits, hidden_dim=224, lr=2e-4)
    cindices = []
    os.makedirs("checkpoints", exist_ok=True)

    for fold, (tr, val) in enumerate(skf.split(X_scaled, strat_labels)):
        print(f"\n🌀 Fold {fold+1}")
        X_train = np.ascontiguousarray(X_scaled[tr])
        X_val = np.ascontiguousarray(X_scaled[val])
        y_train = y_struct[tr].copy()
        y_val = y_struct[val].copy()

        model.fit(
            X_train,
            (np.ascontiguousarray(y_train['time']), np.ascontiguousarray(y_train['event'])),
            batch_size=64,
            callbacks=[tt.callbacks.EarlyStopping(patience=10)],
            val_data=(X_val, (
                np.ascontiguousarray(y_val['time']),
                np.ascontiguousarray(y_val['event'])
            )),
            verbose=False
        )
        model.compute_baseline_hazards()

        path = f"checkpoints/momoe_fold{fold+1}.pt"
        torch.save(model.net.state_dict(), path)
        print("💾 Saved:", path)

        surv = model.predict_surv_df(X_val)
        ev = EvalSurv(surv, y_val['time'], y_val['event'], censor_surv='km')
        ci = ev.concordance_td()
        print("✅ Fold C-Index:", round(ci, 4))
        cindices.append(ci)

    print("\n🔥 Mean C-Index:", round(np.mean(cindices), 4))

# -------------------- Main --------------------
def main():
    seed_everything(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_ids = sorted([pid for pid in os.listdir("data") if pid not in [".gitkeep", "task3_quality_control.csv"]])
    np.random.shuffle(patient_ids)

    dataset = ChimeraDataset(patient_ids, "features/features", "features/coordinates", "data", max_patches=1024)

    print("📦 Extracting features...")
    samples = extract_patient_features(dataset, device=device)

    # compute modality splits (e.g., hist=1024, rna=1000, clin=16)
    hist_dim = 1024
    total_feat_dim = samples[0]['x'].shape[0]
    rna_clin_dim = total_feat_dim - hist_dim
    clin_dim = 26  # adjust based on your clinical vector size
    rna_dim = rna_clin_dim - clin_dim
    splits = (hist_dim, rna_dim, clin_dim)

    print("🚀 Training MoMoE-DeepSurv...")
    train_pipeline(samples, patient_ids, splits, device=device)

if __name__ == "__main__":
    main()
