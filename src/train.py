import os
import numpy as np
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
        self.attn = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

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
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)

class ExpertNet(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.res1 = ResidualBlock(dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return self.out(x)

class GatingNet(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, num_experts), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)

class MoEDeepSurv(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([ExpertNet(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = GatingNet(dim, num_experts)

    def forward(self, x):
        ex_out = torch.stack([e(x).squeeze(-1) for e in self.experts], dim=1)
        gw = self.gate(x)
        return (ex_out * gw).sum(dim=1, keepdim=True)

def build_model(input_dim, hidden_dim=128, num_experts=3, lr=1e-3):
    net = MoEDeepSurv(input_dim, hidden_dim, num_experts)
    model = CoxPH(net, optimizer=torch.optim.Adam)
    model.optimizer.set_lr(lr)
    return model

# -------------------- Training Pipeline --------------------
def train_pipeline(samples, patient_ids, device='cpu'):
    X = np.stack([s['x'] for s in samples])
    y_struct = np.array([(bool(s['event']), s['time']) for s in samples],
                        dtype=[("event","?"),("time","<f8")])
    y_event = y_struct['event'].astype(int)

    scaler = StandardScaler()
    X_scaled = np.ascontiguousarray(scaler.fit_transform(X))

    # 🪵 Step 1: Train Random Forest for event prediction
    print("🌲 Training RF for event prediction...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_event)
    preds = rf.predict(X_scaled)
    idx = np.where(preds == 1)[0]
    print(f"✅ {len(idx)} patients selected.")

    X_sel = X_scaled[idx]
    y_sel = y_struct[idx]

    # 🧠 Stratified 5-Fold CV
    strat_labels = [int(e)*10000 + int(t) for e,t in zip(y_sel['event'], y_sel['time'])]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = build_model(X_sel.shape[1], hidden_dim=224, num_experts=3, lr=2e-4)
    cindices = []
    os.makedirs("checkpoints", exist_ok=True)

    for fold, (tr, val) in enumerate(skf.split(X_sel, strat_labels)):
        print(f"\n🌀 Fold {fold+1}")
        X_train = np.ascontiguousarray(X_sel[tr])
        X_val = np.ascontiguousarray(X_sel[val])
        y_train = y_sel[tr].copy()
        y_val = y_sel[val].copy()

        # Fix for non-contiguous structured array slices
        y_train_time = np.ascontiguousarray(y_train['time'])
        y_train_event = np.ascontiguousarray(y_train['event'])
        y_val_time = np.ascontiguousarray(y_val['time'])
        y_val_event = np.ascontiguousarray(y_val['event'])

        model.fit(X_train, (y_train_time, y_train_event),
                  batch_size=64,
                  callbacks=[tt.callbacks.EarlyStopping(patience=10)],
                  val_data=(X_val, (y_val_time, y_val_event)),
                  verbose=False)
        model.compute_baseline_hazards()

        path = f"checkpoints/ressurv_moe_fold{fold+1}.pt"
        torch.save(model.net.state_dict(), path)
        print("💾 Saved:", path)

        surv = model.predict_surv_df(X_val)
        ev = EvalSurv(surv, y_val_time, y_val_event, censor_surv='km')
        ci = ev.concordance_td()
        print("✅ Fold C-Index:", round(ci, 4))
        cindices.append(ci)

    print("\n🔥 Mean C-Index:", round(np.mean(cindices), 4))

# -------------------- Main --------------------
def main():
    seed_everything(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_ids = sorted([pid for pid in os.listdir("data") if pid not in [".gitkeep", "task3_quality_control.csv"]])
    dataset = ChimeraDataset(patient_ids, "features/features", "features/coordinates", "data", max_patches=1024)
    print("📦 Extracting features...")
    samples = extract_patient_features(dataset, device=device)
    print("🚀 Training ResSurv-MoE DeepSurv...")
    train_pipeline(samples, patient_ids, device=device)

if __name__ == "__main__":
    main()
