import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader

# -------------------
# GCNSurvival Model (for embedding only)
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

    def get_embedding(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.gelu(x)
            x = self.dropout(x)
        return global_add_pool(x, batch)

# -------------------
# Config & Setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

seeds = [42, 121, 144, 245, 1212]
n_models = len(seeds)

# -------------------
# Load Graph Data
# -------------------
with open("batches.pkl", "rb") as f:
    data_list = pickle.load(f)

in_dim = data_list[0].x.size(1)

# -------------------
# Create Untrained Models
# -------------------
models = []
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCNSurvival(in_dim).to(device)
    model.eval()
    models.append(model)

# -------------------
# Extract Embeddings
# -------------------
loader = DataLoader(data_list, batch_size=1, shuffle=False)

with torch.no_grad():
    for graph in tqdm(loader, desc="Extracting Embeddings"):
        graph = graph.to(device)
        pid = graph.pid[0] if hasattr(graph, "pid") else str(graph.batch[0].item())  # fallback

        embeddings = []
        for model in models:
            emb = model.get_embedding(graph.x, graph.edge_index, graph.batch)
            embeddings.append(emb.cpu())

        avg_emb = torch.mean(torch.stack(embeddings), dim=0)
        torch.save(avg_emb.squeeze(0), os.path.join(embedding_dir, f"{pid}.pt"))
