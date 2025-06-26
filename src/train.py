import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from loss import DifferentiableCIndexLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from dataset import ChimeraDataset
from utils import chimera_collate_fn, seed_everything
from pycox.models.loss import CoxPHLoss

torch.autograd.set_detect_anomaly(True)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, proj_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.attn = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, x, top_k=None):
        x_proj = self.proj(x)
        weights = self.attn(x_proj).squeeze(-1)
        if top_k is not None and top_k < x_proj.size(1):
            topk_vals, topk_idx = torch.topk(weights, top_k, dim=1)
            x_proj = torch.gather(x_proj, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x_proj.size(-1)))
            weights = torch.gather(weights, 1, topk_idx)
        attn_weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * x_proj, dim=1)
        return pooled

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class GatedFusion(nn.Module):
    def __init__(self, dims, fused_dim=512):
        super().__init__()
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, fused_dim) for name, dim in dims.items()
        })
        self.gate = nn.Sequential(
            nn.Linear(len(dims) * fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(dims)),
            nn.Softmax(dim=-1)
        )
        self.modalities = list(dims.keys())

    def forward(self, inputs):
        projected = {k: self.proj[k](v) for k, v in inputs.items()}
        stacked = torch.stack([projected[k] for k in self.modalities], dim=1)
        attn_input = torch.cat([projected[k] for k in self.modalities], dim=-1)
        weights = self.gate(attn_input).unsqueeze(-1)
        return torch.sum(weights * stacked, dim=1)

class BaselineEncoder(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=13,
                 hist_proj_dim=1024, hist_feat_dim=512,
                 rna_feat_dim=1024, clinical_feat_dim=128):
        super().__init__()
        self.attn_pool = AttentionPooling(hist_dim, hist_proj_dim)

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_proj_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            ResidualBlock(1024),
            nn.Linear(1024, hist_feat_dim),
            nn.ReLU()
        )

        self.rna_bottleneck = nn.Sequential(
            nn.Linear(rna_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.rna_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            ResidualBlock(1024),
            nn.Linear(1024, rna_feat_dim),
            nn.ReLU()
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, clinical_feat_dim),
            nn.ReLU()
        )

        self.gated_fusion = GatedFusion({
            'hist': hist_feat_dim,
            'rna': rna_feat_dim,
            'clinical': clinical_feat_dim
        }, fused_dim=512)

        self.out_dim = 512

    def forward(self, hist_patches, rna_vec, clinical_vec):
        pooled_hist = self.attn_pool(hist_patches, top_k=128)
        hist_repr = self.hist_encoder(pooled_hist)
        rna_repr = self.rna_encoder(self.rna_bottleneck(rna_vec))
        clinical_repr = self.clinical_encoder(clinical_vec)

        return self.gated_fusion({
            'hist': hist_repr,
            'rna': rna_repr,
            'clinical': clinical_repr
        })

class ProgressionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(x)

class ProgressionEmbedding(nn.Module):
    def __init__(self, num_classes=2, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, class_logits):
        class_probs = torch.softmax(class_logits, dim=1)
        pred_class = class_probs.argmax(dim=1)
        return self.embedding(pred_class)

class SurvivalRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            ResidualBlock(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)

class TwoStageModel(nn.Module):
    def __init__(self, hist_dim=1026, rna_dim=19359, clinical_dim=26):
        super().__init__()
        self.encoder = BaselineEncoder(hist_dim, rna_dim, clinical_dim)
        self.classifier = ProgressionClassifier(self.encoder.out_dim)
        self.prog_embed = ProgressionEmbedding()
        self.survival_regressor = SurvivalRegressor(self.encoder.out_dim + 32)

    def forward(self, hist_patches, rna_vec, clinical_vec):
        features = self.encoder(hist_patches, rna_vec, clinical_vec)
        class_logits = self.classifier(features)
        prog_embedding = self.prog_embed(class_logits)
        surv_input = torch.cat([features, prog_embedding], dim=1)
        risk_score = self.survival_regressor(surv_input)
        return class_logits, risk_score



def train_model_two_stage(model, train_loader, val_loader, fold_idx, writer, 
                          lambda_cls=1.0, lambda_surv=1.0, patience=10, max_epochs=100, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.01, total_iters=5), CosineAnnealingLR(optimizer, T_max=max_epochs - 5)],
        milestones=[5]
    )
    bce_loss = nn.CrossEntropyLoss()
    cindex_loss = DifferentiableCIndexLoss(sigma=0.1)

    best_val_cindex = 0
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/fold_{fold_idx + 1}_best_2stage.pth"

    for epoch in range(max_epochs):
        model.train()
        skipped_batches = 0

        for batch in train_loader:
            hist = batch['hist_patches'].to(device)
            rna = batch['rna_vec'].to(device)
            clinical = batch['clinical_vec'].to(device)
            time = batch['time'].to(device).float()
            event = batch['event'].to(device).float()
            progression = batch['event'].long().to(device)

            optimizer.zero_grad()
            class_logits, risk_score = model(hist, rna, clinical)
            risk_score = torch.clamp(risk_score, min=-30.0, max=30.0)

            loss_cls = bce_loss(class_logits, progression)
            loss_surv = cindex_loss(risk_score, time, event)
            loss = lambda_cls * loss_cls + lambda_surv * loss_surv

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_times, val_events = [], [], []
        val_loss_total = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                hist = batch['hist_patches'].to(device)
                rna = batch['rna_vec'].to(device)
                clinical = batch['clinical_vec'].to(device)
                time = batch['time'].to(device).float()
                event = batch['event'].to(device).float()
                progression = batch['event'].long().to(device)

                class_logits, risk_score = model(hist, rna, clinical)
                risk_score = torch.clamp(risk_score, min=-30.0, max=30.0)

                loss_cls = bce_loss(class_logits, progression)
                loss_surv = cindex_loss(risk_score, time, event)
                val_loss = lambda_cls * loss_cls + lambda_surv * loss_surv

                val_loss_total += val_loss.item()
                val_batches += 1

                val_preds.extend(risk_score.cpu().numpy())
                val_times.extend(time.cpu().numpy())
                val_events.extend(event.cpu().numpy())

        avg_val_loss = val_loss_total / max(val_batches, 1)
        val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)], dtype=[('event', '?'), ('time', '<f8')])
        val_cindex = concordance_index_censored(val_struct['event'], val_struct['time'], -np.array(val_preds))[0]

        writer.add_scalar(f"Fold{fold_idx}/ValLoss", avg_val_loss, epoch)
        writer.add_scalar(f"Fold{fold_idx}/ValCIndex", val_cindex, epoch)
        print(f"Epoch {epoch + 1:03d} | Fold {fold_idx + 1} | Val Loss: {avg_val_loss:.4f} | C-Index: {val_cindex:.4f}")

        scheduler.step()
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

    patient_ids = sorted([pid for pid in os.listdir("data") if pid not in [".gitkeep", "task3_quality_control.csv"]])
    dataset = ChimeraDataset(
        patient_ids=patient_ids,
        features_dir="features/features",
        coords_dir="features/coordinates",
        data_dir="data",
        max_patches=65536
    )

    cohorts = [0 if pid.startswith("3A") else 1 for pid in patient_ids]
    events, progression_labels = [], []
    for pid in patient_ids:
        with open(os.path.join("data", pid, f"{pid}_CD.json"), 'r') as f:
            cd = json.load(f)
            events.append(int(cd.get("event", 0)))
            progression_labels.append(float(cd.get("progression", 0)))

    strat_labels = [f"{c}_{e}" for c, e in zip(cohorts, progression_labels)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    writer = SummaryWriter(log_dir="runs/two_stage_model")
    all_cindices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(strat_labels)), strat_labels)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True, drop_last=True,
                                  collate_fn=chimera_collate_fn, pin_memory=True, num_workers=os.cpu_count())
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=4, shuffle=False, drop_last=True,
                                collate_fn=chimera_collate_fn, pin_memory=True, num_workers=os.cpu_count())

        model = TwoStageModel().to(device)
        cindex = train_model_two_stage(model, train_loader, val_loader, fold, writer,
                                       lambda_cls=1.0, lambda_surv=1.0, patience=10, device=device)
        print(f"Fold {fold + 1} C-Index: {cindex:.4f}")
        all_cindices.append(cindex)

    print("\nAverage C-Index across folds:", np.mean(all_cindices))
    writer.close()

if __name__ == "__main__":
    run_cross_validation()
