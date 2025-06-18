import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import ChimeraDataset
from utils import chimera_collate_fn

from pycox.models.loss import CoxPHLoss


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # [B, N, D]
        weights = self.attention(x).squeeze(-1)  # [B, N]
        attn = torch.softmax(weights, dim=1).unsqueeze(-1)  # [B, N, 1]
        out = torch.sum(attn * x, dim=1)  # [B, D]
        return out


class BaselineModel(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=13):
        super().__init__()
        self.attn_pool = AttentionPooling(in_dim=hist_dim)
        total_input = hist_dim + rna_dim + clinical_dim

        self.backbone = nn.Sequential(
            nn.Linear(total_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, hist_patches, rna_vec, clinical_vec):
        pooled_hist = self.attn_pool(hist_patches)  # [B, hist_dim]
        x = torch.cat([pooled_hist, rna_vec, clinical_vec], dim=1)
        return self.backbone(x).squeeze(-1)  # [B]


from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.utils


def train_model(model, train_loader, val_loader, fold_idx, writer, patience=5, max_epochs=100, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    # Scheduler: Warmup + Cosine
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - 5, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    # Optional: Use ReduceLROnPlateau instead
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    criterion = CoxPHLoss()
    best_val_cindex = 0
    patience_counter = 0
    best_model_path = f"checkpoints/fold_{fold_idx + 1}_best.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            hist = batch['hist_patches'].to(device)
            rna = batch['rna_vec'].to(device)
            clinical = batch['clinical_vec'].to(device)

            time = batch['time'].to(device).float()
            event = batch['event'].to(device).float()

            preds = model(hist, rna, clinical)
            loss = criterion(preds, time, event)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist_patches'].to(device)
                rna = batch['rna_vec'].to(device)
                clinical = batch['clinical_vec'].to(device)

                pred = model(hist, rna, clinical)
                val_preds.extend(pred.cpu().numpy())
                val_times.extend(batch['time'].cpu().numpy())
                val_events.extend(batch['event'].cpu().numpy())

        val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)],
                              dtype=[('event', '?'), ('time', '<f8')])
        val_cindex = concordance_index_censored(val_struct['event'], val_struct['time'], -np.array(val_preds))[0]

        writer.add_scalar(f"Fold{fold_idx}/TrainLoss", avg_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/ValCIndex", val_cindex, epoch)

        print(f"📊 Epoch {epoch + 1:03d} | Fold {fold_idx + 1} | Loss: {avg_loss:.4f} | C-Index: {val_cindex:.4f}")

        # Step scheduler
        scheduler.step()
        # If using ReduceLROnPlateau, replace with: scheduler.step(val_cindex)

        # Early stopping check
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved at {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1} for fold {fold_idx + 1}")
                break

    return best_val_cindex


def run_cross_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    patient_ids = sorted([
        pid for pid in os.listdir("data")
        if pid not in [".gitkeep", "task3_quality_control.csv"]
    ])

    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data",
        max_patches=32768
    )

    # 🧠 Stratify based on whether ID starts with "3A" or "3B"
    strat_labels = [0 if pid.startswith("3A") else 1 for pid in patient_ids]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    writer = SummaryWriter(log_dir="runs/attn_baseline_by_cohort")

    all_cindices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(strat_labels)), strat_labels)):
        print(f"Fold {fold + 1} | 3A in train: {sum(strat_labels[i]==0 for i in train_idx)} | 3B in train: {sum(strat_labels[i]==1 for i in train_idx)}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True,
                                   collate_fn=chimera_collate_fn, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False,
                                 collate_fn=chimera_collate_fn, pin_memory=True)

        model = BaselineModel().to(device)
        cindex = train_model(model, train_loader, val_loader, fold, writer, device=device)
        print(f"✅ Fold {fold + 1} C-Index: {cindex:.4f}")
        all_cindices.append(cindex)

    print("\n📈 Average C-Index across folds:", np.mean(all_cindices))
    writer.close()


if __name__ == "__main__":
    run_cross_validation()
