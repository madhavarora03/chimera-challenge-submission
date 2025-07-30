import os
import random
import pickle
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index

# -------------------
# Reproducibility
# -------------------
def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def worker_init_fn(worker_id):
    seed = 42
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

# -------------------
# Model Definition
# -------------------
class GCNSurvival(nn.Module):
    def __init__(self, in_dim, embedding_size=128, dropout=0.5):
        super().__init__()
        self.conv0 = GCNConv(in_dim, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch):
        x = self.get_embedding(x, edge_index, batch)
        return self.out(x).squeeze(-1)

    def get_embedding(self, x, edge_index, batch):
        x = self.conv0(x, edge_index); x = self.bn0(x); x = F.gelu(x); x = self.dropout(x)
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.gelu(x); x = self.dropout(x)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.gelu(x); x = self.dropout(x)
        x = self.conv3(x, edge_index); x = self.bn3(x); x = F.gelu(x); x = self.dropout(x)
        return global_add_pool(x, batch)

# -------------------
# Cox PH Loss
# -------------------
def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cumsum)[event == 1])

# -------------------
# Epoch Runner
# -------------------
def run_epoch(model, loader, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, risks, times, events = 0.0, [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = cox_ph_loss(out, batch.y[:, 0], batch.y[:, 1])

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        risks.append(out.detach().cpu().numpy())
        times.append(batch.y[:, 0].cpu().numpy())
        events.append(batch.y[:, 1].cpu().numpy())

    risks = np.concatenate(risks)
    times = np.concatenate(times)
    events = np.concatenate(events)
    ci = concordance_index(times, -risks, events)
    return total_loss / len(loader.dataset), ci

# -------------------
# Optuna Objective with 5-Fold CV
# -------------------
def objective(trial):
    base_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("batched_graphs.pkl", "rb") as f:
        data_list = pickle.load(f)

    kf = KFold(n_splits=5, shuffle=True, random_state=base_seed)
    fold_cis = []

    embedding_size = trial.suggest_categorical("embedding_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        seed_all(base_seed + fold_idx)

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        g = torch.Generator().manual_seed(base_seed + fold_idx)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=g, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)

        in_dim = data_list[0].x.size(1)
        model = GCNSurvival(in_dim, embedding_size, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_ci, best_epoch = 0.0, 0
        patience = 15

        for epoch in range(1, 101):
            train_loss, _ = run_epoch(model, train_loader, optimizer, device, is_train=True)
            val_loss, val_ci = run_epoch(model, val_loader, optimizer, device, is_train=False)
            scheduler.step(val_loss)

            if val_ci > best_ci:
                best_ci = val_ci
                best_epoch = epoch

            if epoch - best_epoch >= patience:
                break

        fold_cis.append(best_ci)

    return np.mean(fold_cis)

# -------------------
# Run Optuna
# -------------------
if __name__ == "__main__":
    seed_all(42)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n🏁 Best Trial:")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
    print(f"Best 5-Fold CV C-Index: {best_trial.value:.4f}")
