import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sksurv.metrics import concordance_index_censored
from torch.utils.tensorboard import SummaryWriter

from dataset import ChimeraDataset
from utils import seed_everything


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cumsum)[event == 1])


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def train_model(model, train_loader, val_loader, fold_idx, writer,
                patience=10, max_epochs=100, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_cindex = 0
    patience_counter = 0
    best_model_path = f"checkpoints/fold_{fold_idx + 1}_best_mlp.pth"

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x = batch['input_vec'].to(device)
            time = batch['time'].to(device)
            event = batch['event'].to(device)

            optimizer.zero_grad()
            risk = model(x)
            loss = cox_ph_loss(risk, time, event)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_vec'].to(device)
                time = batch['time'].to(device)
                event = batch['event'].to(device)
                risk = model(x)
                val_preds.extend(risk.cpu().numpy())
                val_times.extend(time.cpu().numpy())
                val_events.extend(event.cpu().numpy())

        val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)], dtype=[('event', '?'), ('time', '<f8')])
        val_cindex = concordance_index_censored(val_struct['event'], val_struct['time'], -np.array(val_preds))[0]

        writer.add_scalar(f"Fold{fold_idx}/ValCIndex", val_cindex, epoch)
        print(f"Epoch {epoch + 1:03d} | Fold {fold_idx + 1} | C-Index: {val_cindex:.4f}")

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} for fold {fold_idx + 1}")
                break

    return best_val_cindex


def run_cross_validation():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("checkpoints", exist_ok=True)
    patient_ids = sorted([
        pid for pid in os.listdir("data")
        if pid.startswith("3")
    ])

    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        embeddings_dir="embeddings",
        data_dir="data",
        gene_list_file="test.txt"
    )

    # Prepare stratification labels
    progression_labels = []
    for pid in patient_ids:
        with open(os.path.join("data", pid, f"{pid}_CD.json"), 'r') as f:
            cd = json.load(f)
            progression_labels.append(int(cd.get("progression", 0)))

    strat_labels = shuffle(progression_labels, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    writer = SummaryWriter(log_dir="runs/mlp_survival")
    all_cindices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(strat_labels)), strat_labels)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)

        input_dim = next(iter(train_loader))['input_vec'].shape[1]
        model = MLP(input_dim=input_dim).to(device)

        cindex = train_model(model, train_loader, val_loader, fold, writer, device=device)
        print(f"Fold {fold + 1} C-Index: {cindex:.4f}")
        all_cindices.append(cindex)

    print("\nAverage C-Index across folds:", np.mean(all_cindices))
    writer.close()


if __name__ == "__main__":
    run_cross_validation()
