import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import ChimeraDataset  # assumes dataset is correctly implemented


class BaselineModel(nn.Module):
    def __init__(self, hist_dim=1024, rna_dim=19359, clinical_dim=13):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hist_dim + rna_dim + clinical_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # risk score
        )

    def forward(self, hist, rna, clinical):
        x = torch.cat([hist, rna, clinical], dim=1)
        return self.backbone(x).squeeze(-1)


def train_model(model, train_loader, val_loader, fold_idx, writer, patience=5, max_epochs=100, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # simplistic loss for baseline

    best_val_cindex = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move data to the device (GPU/CPU)
            hist = batch['hist_embedding'].to(device)
            rna = batch['rna_vec'].to(device)
            clinical = batch['clinical_vec'].to(device)
            target = batch['time'].float().to(device)

            preds = model(hist, rna, clinical)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist_embedding'].to(device)
                rna = batch['rna_vec'].to(device)
                clinical = batch['clinical_vec'].to(device)
                val_times.extend(batch['time'].cpu().numpy())
                val_events.extend(batch['event'].cpu().numpy())

                pred = model(hist, rna, clinical)
                val_preds.extend(pred.cpu().numpy())

        val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)],
                              dtype=[('event', '?'), ('time', '<f8')])
        val_cindex = concordance_index_censored(val_struct['event'], val_struct['time'], -np.array(val_preds))[0]

        writer.add_scalar(f"Fold{fold_idx}/TrainLoss", avg_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/ValCIndex", val_cindex, epoch)

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping on epoch {epoch} for fold {fold_idx}")
                break

    return best_val_cindex


def run_cross_validation():
    # Set device: use GPU if available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    patient_ids = sorted(os.listdir("data"))
    patient_ids = [pid for pid in patient_ids if pid != ".gitkeep" and pid != "task3_quality_control.csv"]

    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    writer = SummaryWriter(log_dir="runs/baseline")

    all_cindices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n==== Fold {fold + 1} ====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)

        model = BaselineModel().to(device)  # Move model to the GPU/CPU
        cindex = train_model(model, train_loader, val_loader, fold, writer, device=device)
        print(f"Fold {fold + 1} C-Index: {cindex:.4f}")
        all_cindices.append(cindex)

    print("\nAverage C-Index across folds:", np.mean(all_cindices))
    writer.close()


if __name__ == "__main__":
    run_cross_validation()
