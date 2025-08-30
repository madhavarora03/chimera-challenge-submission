import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
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
    def __init__(self, in_dim, embedding_size=1280, n_layers=4, dropout=0.3167):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, embedding_size))
        self.bns.append(nn.BatchNorm1d(embedding_size))

        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
            self.bns.append(nn.BatchNorm1d(embedding_size))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch):
        x = self.get_embedding(x, edge_index, batch)
        return self.out(x).squeeze(-1)

    def get_embedding(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.gelu(x)
            x = self.dropout(x)
        return global_add_pool(x, batch)

# -------------------
# Cox PH Loss
# -------------------
def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    event_risk = risk[event == 1]
    event_log_cumsum = log_cumsum[event == 1]
    if len(event_risk) == 0:
        return torch.tensor(0.0, device=risk.device)
    return -torch.mean(event_risk - event_log_cumsum)

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

        if torch.isnan(loss) or torch.isinf(loss):
            return np.nan, np.nan

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
# Training with Best Hyperparameters
# -------------------
def train_with_best_hparams(data_list, device):
    seeds = [42, 121, 144, 245, 6901]
    mean_cis = []

    for seed in seeds:
        print(f"\n🌱 Training with seed {seed}")
        seed_all(seed)

        train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=seed)
        g = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=g, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)

        in_dim = data_list[0].x.size(1)
        model = GCNSurvival(in_dim).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004088, weight_decay=0.0001513)

        best_val_ci = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        max_epochs = 5

        for epoch in range(1, max_epochs + 1):
            train_loss, _ = run_epoch(model, train_loader, optimizer, device, is_train=True)
            val_loss, val_ci = run_epoch(model, val_loader, optimizer, device, is_train=False)

            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val CI: {val_ci:.4f}")

            if val_ci > best_val_ci:
                best_val_ci = val_ci
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch} for seed {seed}")
                break

        model_path = f"model_seed_{seed}.pt"
        torch.save(best_model_state, model_path)
        print(f"✅ Best model saved to: {model_path} with CI: {best_val_ci:.4f}")
        mean_cis.append(best_val_ci)

    overall_mean = np.mean(mean_cis)
    overall_std = np.std(mean_cis)
    print("\n🎯 Final Summary Across Seeds:")
    print(f"Mean CI: {overall_mean:.4f} | Std Dev: {overall_std:.4f}")

# -------------------
# Main Execution
# -------------------
def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("batched_graphs.pkl", "rb") as f:
        data_list = pickle.load(f)

    train_with_best_hparams(data_list, device)

if __name__ == "__main__":
    main()
