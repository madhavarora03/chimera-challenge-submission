#!/usr/bin/env python3
"""
Training script using optimized hyperparameters from Optuna results.
Trains all models with the best found parameters and saves them for testing.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import RobustScaler
from dataset import ChimeraDataset
from utils import seed_everything
import warnings
warnings.filterwarnings("ignore")

# Load optimized hyperparameters
OPTUNA_RESULTS = {
    "42": {
        "individual_models": {
            "rsf": 0.8207885304659498,
            "gbsa": 0.8154121863799283,
            "cox_pca": 0.6344086021505376,
            "coxnet": 0.6308243727598566,
            "survival_svm": 0.4767025089605735,
            "mlp": 0.5985663082437276
        },
        "ensemble_results": {
            "optimized_weighted": 0.8369175627240143,
            "simple_weighted": 0.8440860215053764,
            "best_single": 0.8207885304659498
        },
        "best_method": "simple_weighted",
        "best_cindex": 0.8440860215053764,
        "optimized_params": {
            "rsf": {
                "n_estimators": 550,
                "min_samples_split": 29,
                "min_samples_leaf": 13,
                "max_depth": 19,
                "max_features": 0.7,
                "bootstrap": True
            },
            "gbsa": {
                "n_estimators": 225,
                "learning_rate": 0.030952011306522923,
                "max_depth": 6,
                "min_samples_split": 24,
                "min_samples_leaf": 19,
                "subsample": 0.9692446710708374,
                "max_features": 0.7
            },
            "cox_pca": {
                "n_components": 36,
                "alpha": 0.0010300403995607632,
                "ties": "breslow"
            },
            "coxnet": {
                "n_components": 36,
                "l1_ratio": 0.2877173916670861,
                "alpha_min_ratio": 1.1723869841199362e-05,
                "n_alphas": 169,
                "max_iter": 1769
            },
            "survival_svm": {
                "alpha": 3.183289936074655e-06,
                "rank_ratio": 0.00019408089960514657,
                "fit_intercept": False,
                "max_iter": 726,
                "tol": 2.192494956744349e-05
            },
            "mlp": {
                "n_layers": 6,
                "base_hidden_dim": 320,
                "decay_factor": 0.36612770224805313,
                "lr": 0.00017439695738576791,
                "weight_decay": 2.2631505935844808e-05,
                "dropout": 0.32262150556352026,
                "batch_size": 13,
                "num_heads": 12,
                "use_attention": False,
                "activation": "gelu",
                "batch_norm": False,
                "use_concordance_loss": True
            },
            "ensemble_weights": {
                "rsf": 0.32933399081222986,
                "gbsa": 0.38121726770582504,
                "cox_pca": 0.16637024341367918,
                "coxnet": 0.11040051465368356,
                "survival_svm": 0.01267798341458236
            }
        }
    },
    "245": {
        "optimized_params": {
            "rsf": {
                "n_estimators": 200,
                "min_samples_split": 34,
                "min_samples_leaf": 16,
                "max_depth": 17,
                "max_features": 0.7,
                "bootstrap": True
            },
            "gbsa": {
                "n_estimators": 225,
                "learning_rate": 0.01776881861979314,
                "max_depth": 8,
                "min_samples_split": 9,
                "min_samples_leaf": 18,
                "subsample": 0.9355349112536552,
                "max_features": 0.7
            },
            "mlp": {
                "n_layers": 3,
                "base_hidden_dim": 512,
                "decay_factor": 0.7391963650868432,
                "lr": 0.0006251373574521745,
                "weight_decay": 4.2079886696066345e-06,
                "dropout": 0.17799726016810133,
                "batch_size": 11,
                "num_heads": 15,
                "use_attention": False,
                "activation": "gelu",
                "batch_norm": True,
                "use_concordance_loss": False
            }
        }
    }
}

# -------------------------
# Utility Functions
# -------------------------
def safe_makedirs(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def cox_ph_loss(risk, time, event, l2_reg=1e-4):
    """Cox proportional hazards loss function."""
    order = torch.argsort(time, descending=True)
    risk_sorted, event_sorted = risk[order].view(-1), event[order]
    
    eps = 1e-7
    log_cumsum = torch.logcumsumexp(risk_sorted + eps, dim=0)
    
    event_mask = event_sorted == 1
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    loss = -torch.mean((risk_sorted - log_cumsum)[event_mask])
    return loss

def concordance_loss(risk, time, event):
    """Direct concordance optimization loss."""
    n = len(risk)
    concordant_pairs = 0
    comparable_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 1 and time[i] < time[j]:
                comparable_pairs += 1
                if risk[i] > risk[j]:
                    concordant_pairs += 1
            elif event[j] == 1 and time[j] < time[i]:
                comparable_pairs += 1
                if risk[j] > risk[i]:
                    concordant_pairs += 1
    
    if comparable_pairs == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    concordance = concordant_pairs / comparable_pairs
    return torch.tensor(1.0 - concordance, requires_grad=True)

def extract_numpy_data(dataset, indices):
    """Extract numpy arrays from dataset."""
    X, T, E = [], [], []
    for i in indices:
        sample = dataset[i]
        X.append(sample['input_vec'].numpy())
        T.append(sample['time'])
        E.append(sample['event'])
    X = np.stack(X)
    y = np.array([(bool(e), t) for e, t in zip(E, T)], dtype=[('event', '?'), ('time', '<f8')])
    return X, y

def compute_cindex_from_scores(y_struct, scores):
    """Compute C-index with proper handling of edge cases."""
    if len(np.unique(scores)) == 1:
        return 0.5
    
    try:
        return concordance_index_censored(y_struct['event'], y_struct['time'], scores)[0]
    except Exception:
        return 0.5

def standardize_risk_scores(train_scores, val_scores, method='robust'):
    """Standardize risk scores for ensemble combination."""
    if method == 'robust':
        median = np.median(train_scores)
        mad = np.median(np.abs(train_scores - median))
        mad = max(mad, 1e-8)
        train_norm = (train_scores - median) / mad
        val_norm = (val_scores - median) / mad
        
    return train_norm, val_norm

# -------------------------
# Neural Network Models
# -------------------------
class OptimizedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            for nh in reversed(range(1, num_heads + 1)):
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_ = x.unsqueeze(1)
        attn_output, _ = self.attn(x_, x_, x_)
        x1 = self.norm1(x_ + attn_output)
        ffn_output = self.ffn(x1)
        x2 = self.norm2(x1 + ffn_output)
        return x2.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, act_fn=nn.GELU, batch_norm=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if batch_norm else nn.Identity(),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
    
    def forward(self, x):
        return self.norm(x + self.block(x))

class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2, num_heads=8, 
                 use_attention=True, activation='gelu', batch_norm=True):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim) if batch_norm else nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attn = OptimizedMultiHeadSelfAttention(input_dim, num_heads=num_heads, dropout=dropout)
        
        # Activation function selection
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'swish':
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU
        
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            
            if in_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim, dropout, act_fn, batch_norm))
            
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_norm(x)
        
        if self.use_attention:
            x_attn = self.attn(x)
            x = x + x_attn
        
        x = self.mlp(x)
        out = self.output_layer(x)
        return out

# -------------------------
# Training Functions
# -------------------------
def train_rsf(X_train, y_train, params, seed):
    """Train Random Survival Forest with optimized parameters."""
    print(f"🌲 Training Random Survival Forest...")
    
    model_params = params.copy()
    model_params['random_state'] = seed
    model_params['n_jobs'] = -1
    
    model = RandomSurvivalForest(**model_params)
    model.fit(X_train, y_train)
    
    return model

def train_gbsa(X_train, y_train, params, seed):
    """Train Gradient Boosting Survival Analysis with optimized parameters."""
    print(f"📈 Training Gradient Boosting Survival Analysis...")
    
    model_params = params.copy()
    model_params['random_state'] = seed
    
    model = GradientBoostingSurvivalAnalysis(**model_params)
    model.fit(X_train, y_train)
    
    return model

def train_cox_pca(X_train, y_train, params, seed):
    """Train Cox PH with PCA using optimized parameters."""
    print(f"📊 Training Cox PH with PCA...")
    
    # Apply PCA
    pca = PCA(n_components=params['n_components'], random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    
    # Train Cox model
    cox_model = CoxPHSurvivalAnalysis(
        alpha=params['alpha'],
        ties=params['ties']
    )
    cox_model.fit(X_train_pca, y_train)
    
    return cox_model, pca

def train_coxnet(X_train, y_train, params, seed):
    """Train CoxNet with optimized parameters."""
    print(f"🎯 Training CoxNet...")
    
    # Apply PCA
    pca = PCA(n_components=params['n_components'], random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    
    # Train CoxNet model
    coxnet_model = CoxnetSurvivalAnalysis(
        l1_ratio=params['l1_ratio'],
        alpha_min_ratio=params['alpha_min_ratio'],
        n_alphas=params['n_alphas'],
        max_iter=params['max_iter']
    )
    coxnet_model.fit(X_train_pca, y_train)
    
    return coxnet_model, pca

def train_survival_svm(X_train, y_train, params, seed):
    """Train Survival SVM with optimized parameters."""
    print(f"🔧 Training Survival SVM...")
    
    model_params = params.copy()
    model_params['random_state'] = seed
    
    model = FastSurvivalSVM(**model_params)
    model.fit(X_train, y_train)
    
    return model

def train_mlp(dataset, train_idx, val_idx, params, seed, device='cpu'):
    """Train MLP with optimized parameters."""
    print(f"🧠 Training Optimized MLP...")
    
    # Reconstruct hidden dimensions
    n_layers = params['n_layers']
    base_dim = params['base_hidden_dim']
    decay_factor = params['decay_factor']
    hidden_dims = []
    
    for i in range(n_layers):
        dim = max(16, int(base_dim * (decay_factor ** i)))
        hidden_dims.append(dim)
    
    print(f"   Architecture: {hidden_dims}")
    print(f"   Attention: {params['use_attention']}")
    print(f"   Activation: {params['activation']}")
    
    # Create model
    input_dim = dataset[0]['input_vec'].shape[0]
    model = OptimizedMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=params['dropout'],
        num_heads=params['num_heads'],
        use_attention=params['use_attention'],
        activation=params['activation'],
        batch_norm=params['batch_norm']
    ).to(device)
    
    # Create data loaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=params['batch_size'], shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_val_cindex = 0
    patience_counter = 0
    max_patience = 30
    
    print(f"   Training for up to 200 epochs...")
    
    for epoch in range(200):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x = batch['input_vec'].to(device).float()
            time = batch['time'].to(device).float()
            event = batch['event'].to(device).float()
            
            optimizer.zero_grad()
            risk = model(x).view(-1)
            
            if params['use_concordance_loss'] and epoch > 10:
                loss = concordance_loss(risk, time, event)
            else:
                loss = cox_ph_loss(risk, time, event)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())

        scheduler.step()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_preds, val_times, val_events = [], [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['input_vec'].to(device).float()
                    time = batch['time'].to(device).float()
                    event = batch['event'].to(device).float()
                    risk = model(x).view(-1)
                    val_preds.extend(risk.cpu().numpy().flatten())
                    val_times.extend(time.cpu().numpy())
                    val_events.extend(event.cpu().numpy())

            val_struct = np.array([(bool(e), t) for e, t in zip(val_events, val_times)],
                                  dtype=[('event', '?'), ('time', '<f8')])
            val_cindex = compute_cindex_from_scores(val_struct, np.array(val_preds))
            
            print(f"   Epoch {epoch:3d}: Loss={np.mean(train_losses):.4f}, Val C-index={val_cindex:.4f}")
            
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"optimized_models/mlp_seed_{seed}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= max_patience // 5:  # Adjusted for validation frequency
                    print(f"   Early stopping at epoch {epoch}")
                    break
    
    # Load best model
    model.load_state_dict(torch.load(f"optimized_models/mlp_seed_{seed}_best.pt", map_location=device))
    
    return model

def create_ensemble_predictor(models, ensemble_weights, scalers, seed):
    """Create ensemble predictor function."""
    
    def predict(X_new):
        """Predict using the optimized ensemble."""
        predictions = {}
        
        # Scale input data
        X_scaled = scalers['main'].transform(X_new)
        
        # Get predictions from each model
        if 'rsf' in models:
            predictions['rsf'] = models['rsf'].predict(X_scaled)
        
        if 'gbsa' in models:
            predictions['gbsa'] = models['gbsa'].predict(X_scaled)
        
        if 'cox_pca' in models:
            cox_model, pca = models['cox_pca']
            X_pca = pca.transform(X_scaled)
            predictions['cox_pca'] = cox_model.predict(X_pca)
        
        if 'coxnet' in models:
            coxnet_model, pca = models['coxnet']
            X_pca = pca.transform(X_scaled)
            predictions['coxnet'] = coxnet_model.predict(X_pca)
        
        if 'survival_svm' in models:
            predictions['survival_svm'] = models['survival_svm'].predict(X_scaled)
        
        if 'mlp' in models:
            # Handle MLP prediction
            mlp_model = models['mlp']
            mlp_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    mlp_model = mlp_model.cuda()
                mlp_preds = mlp_model(X_tensor).cpu().numpy().flatten()
            predictions['mlp'] = mlp_preds
        
        # Create weighted ensemble
        ensemble_pred = np.zeros(X_new.shape[0])
        total_weight = 0
        
        for model_name, weight in ensemble_weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred, predictions
    
    return predict

# -------------------------
# Main Training Function
# -------------------------
def train_optimized_models(seed=42, use_best_seed_params=True):
    """Train all models using optimized hyperparameters."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING OPTIMIZED MODELS - SEED {seed}")
    print(f"{'='*70}")
    
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory
    safe_makedirs("optimized_models")
    safe_makedirs("optimized_models/ensemble_components")
    
    # Load dataset
    patient_ids = sorted([pid for pid in os.listdir("data") if pid.startswith("3")])
    dataset = ChimeraDataset(patient_ids, "embeddings", "data", "top_2000_genes_by_brs.txt")
    print(f"Dataset size: {len(dataset)} patients")

    # Stratified split
    progression_labels = []
    for pid in patient_ids:
        with open(os.path.join("data", pid, f"{pid}_CD.json"), 'r') as f:
            cd = json.load(f)
            progression_labels.append(int(cd.get("progression", 0)))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(progression_labels)), progression_labels))
    
    print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

    # Data preparation
    X_train, y_train = extract_numpy_data(dataset, train_idx)
    X_val, y_val = extract_numpy_data(dataset, val_idx)

    # Preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Determine which parameters to use
    if use_best_seed_params:
        # Use parameters from the best performing seed (245 had the highest individual model scores)
        if str(seed) in OPTUNA_RESULTS:
            seed_params = OPTUNA_RESULTS[str(seed)]['optimized_params']
            print(f"Using optimized parameters for seed {seed}")
        else:
            # Use parameters from seed 42 as default
            seed_params = OPTUNA_RESULTS['42']['optimized_params']
            print(f"Using optimized parameters from seed 42 (default)")
    else:
        seed_params = OPTUNA_RESULTS['42']['optimized_params']
        print(f"Using default optimized parameters from seed 42")
    
    # Train all models
    trained_models = {}
    scalers = {'main': scaler}
    val_risks = {}
    train_risks = {}
    
    # 1. Train Random Survival Forest
    try:
        rsf_model = train_rsf(X_train_scaled, y_train, seed_params['rsf'], seed)
        trained_models['rsf'] = rsf_model
        
        train_risks['rsf'] = rsf_model.predict(X_train_scaled)
        val_risks['rsf'] = rsf_model.predict(X_val_scaled)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['rsf'])
        print(f"   ✅ RSF trained - Validation C-index: {val_cindex:.4f}")
        
        joblib.dump(rsf_model, f"optimized_models/rsf_seed_{seed}.pkl")
        
    except Exception as e:
        print(f"   ❌ RSF training failed: {e}")

    # 2. Train Gradient Boosting Survival Analysis
    try:
        gbsa_model = train_gbsa(X_train_scaled, y_train, seed_params['gbsa'], seed)
        trained_models['gbsa'] = gbsa_model
        
        train_risks['gbsa'] = gbsa_model.predict(X_train_scaled)
        val_risks['gbsa'] = gbsa_model.predict(X_val_scaled)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['gbsa'])
        print(f"   ✅ GBSA trained - Validation C-index: {val_cindex:.4f}")
        
        joblib.dump(gbsa_model, f"optimized_models/gbsa_seed_{seed}.pkl")
        
    except Exception as e:
        print(f"   ❌ GBSA training failed: {e}")

    # 3. Train Cox PH with PCA
    try:
        cox_model, pca_cox = train_cox_pca(X_train_scaled, y_train, seed_params['cox_pca'], seed)
        trained_models['cox_pca'] = (cox_model, pca_cox)
        
        X_train_pca = pca_cox.transform(X_train_scaled)
        X_val_pca = pca_cox.transform(X_val_scaled)
        
        train_risks['cox_pca'] = cox_model.predict(X_train_pca)
        val_risks['cox_pca'] = cox_model.predict(X_val_pca)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['cox_pca'])
        print(f"   ✅ Cox PCA trained - Validation C-index: {val_cindex:.4f}")
        
        joblib.dump((cox_model, pca_cox), f"optimized_models/cox_pca_seed_{seed}.pkl")
        
    except Exception as e:
        print(f"   ❌ Cox PCA training failed: {e}")

    # 4. Train CoxNet
    try:
        coxnet_model, pca_coxnet = train_coxnet(X_train_scaled, y_train, seed_params['coxnet'], seed)
        trained_models['coxnet'] = (coxnet_model, pca_coxnet)
        
        X_train_pca_coxnet = pca_coxnet.transform(X_train_scaled)
        X_val_pca_coxnet = pca_coxnet.transform(X_val_scaled)
        
        train_risks['coxnet'] = coxnet_model.predict(X_train_pca_coxnet)
        val_risks['coxnet'] = coxnet_model.predict(X_val_pca_coxnet)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['coxnet'])
        print(f"   ✅ CoxNet trained - Validation C-index: {val_cindex:.4f}")
        
        joblib.dump((coxnet_model, pca_coxnet), f"optimized_models/coxnet_seed_{seed}.pkl")
        
    except Exception as e:
        print(f"   ❌ CoxNet training failed: {e}")

    # 5. Train Survival SVM
    try:
        svm_model = train_survival_svm(X_train_scaled, y_train, seed_params['survival_svm'], seed)
        trained_models['survival_svm'] = svm_model
        
        train_risks['survival_svm'] = svm_model.predict(X_train_scaled)
        val_risks['survival_svm'] = svm_model.predict(X_val_scaled)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['survival_svm'])
        print(f"   ✅ Survival SVM trained - Validation C-index: {val_cindex:.4f}")
        
        joblib.dump(svm_model, f"optimized_models/survival_svm_seed_{seed}.pkl")
        
    except Exception as e:
        print(f"   ❌ Survival SVM training failed: {e}")

    # 6. Train MLP
    try:
        mlp_model = train_mlp(dataset, train_idx, val_idx, seed_params['mlp'], seed, device)
        trained_models['mlp'] = mlp_model
        
        # Get MLP predictions
        def get_mlp_predictions(dataloader):
            preds = []
            mlp_model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    x = batch['input_vec'].to(device).float()
                    r = mlp_model(x).view(-1)
                    preds.extend(r.cpu().numpy().flatten())
            return np.array(preds)
        
        train_loader_eval = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=False)
        val_loader_eval = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)
        
        train_risks['mlp'] = get_mlp_predictions(train_loader_eval)
        val_risks['mlp'] = get_mlp_predictions(val_loader_eval)
        
        val_cindex = compute_cindex_from_scores(y_val, val_risks['mlp'])
        print(f"   ✅ MLP trained - Validation C-index: {val_cindex:.4f}")
        
        # Save MLP model
        torch.save(mlp_model.state_dict(), f"optimized_models/mlp_seed_{seed}.pt")
        
        # Save MLP architecture info for loading later
        mlp_info = {
            'input_dim': dataset[0]['input_vec'].shape[0],
            'hidden_dims': [max(16, int(seed_params['mlp']['base_hidden_dim'] * 
                                       (seed_params['mlp']['decay_factor'] ** i))) 
                           for i in range(seed_params['mlp']['n_layers'])],
            'dropout': seed_params['mlp']['dropout'],
            'num_heads': seed_params['mlp']['num_heads'],
            'use_attention': seed_params['mlp']['use_attention'],
            'activation': seed_params['mlp']['activation'],
            'batch_norm': seed_params['mlp']['batch_norm']
        }
        
        with open(f"optimized_models/mlp_architecture_seed_{seed}.json", 'w') as f:
            json.dump(mlp_info, f, indent=2)
        
    except Exception as e:
        print(f"   ❌ MLP training failed: {e}")

    # -------------------- ENSEMBLE CREATION --------------------
    print(f"\n🎯 Creating Optimized Ensemble...")
    
    if len(val_risks) == 0:
        raise RuntimeError("No models were successfully trained!")
    
    # Print individual model performances
    print(f"\n📊 INDIVIDUAL MODEL PERFORMANCE:")
    individual_cindices = {}
    for name in val_risks.keys():
        val_cindex = compute_cindex_from_scores(y_val, val_risks[name])
        individual_cindices[name] = val_cindex
        print(f"  {name:15}: {val_cindex:.4f}")
    
    # Use optimized ensemble weights if available
    ensemble_weights = seed_params.get('ensemble_weights', {})
    
    # Filter weights to only include available models
    filtered_weights = {k: v for k, v in ensemble_weights.items() if k in val_risks.keys()}
    
    if len(filtered_weights) == 0:
        # Fall back to performance-based weights
        print("   Using performance-based weights (no optimized weights available)")
        weights = np.array([max(0, individual_cindices[name] - 0.5) for name in val_risks.keys()])
        weights = weights / (weights.sum() + 1e-8)
        filtered_weights = {name: weight for name, weight in zip(val_risks.keys(), weights)}
    
    print(f"\n   Ensemble weights: {filtered_weights}")
    
    # Create ensemble prediction
    standardized_preds = []
    model_names = list(val_risks.keys())
    
    for name in model_names:
        _, val_norm = standardize_risk_scores(
            train_risks[name], val_risks[name], method='robust'
        )
        standardized_preds.append(val_norm)
    
    # Weighted ensemble
    ensemble_pred = np.zeros(len(val_risks[model_names[0]]))
    for i, name in enumerate(model_names):
        weight = filtered_weights.get(name, 1.0 / len(model_names))
        ensemble_pred += weight * standardized_preds[i]
    
    ensemble_cindex = compute_cindex_from_scores(y_val, ensemble_pred)
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Best individual model: {max(individual_cindices, key=individual_cindices.get)} "
          f"({max(individual_cindices.values()):.4f})")
    print(f"   Optimized ensemble: {ensemble_cindex:.4f}")
    
    # -------------------- SAVE EVERYTHING --------------------
    print(f"\n💾 Saving all components...")
    
    # Save scalers
    joblib.dump(scaler, f"optimized_models/scaler_seed_{seed}.pkl")
    
    # Save ensemble predictor
    ensemble_predictor = create_ensemble_predictor(trained_models, filtered_weights, scalers, seed)
    
    # Save training metadata
    training_metadata = {
        'seed': seed,
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist(),
        'model_names': list(trained_models.keys()),
        'individual_performances': individual_cindices,
        'ensemble_performance': ensemble_cindex,
        'ensemble_weights': filtered_weights,
        'optimized_params': seed_params,
        'dataset_info': {
            'total_patients': len(dataset),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'input_dim': X_train.shape[1]
        }
    }
    
    with open(f"optimized_models/training_metadata_seed_{seed}.json", 'w') as f:
        json.dump(training_metadata, f, indent=2, default=str)
    
    # Save risk scores for analysis
    risk_data = {
        "train_risks": train_risks,
        "val_risks": val_risks,
        "y_train": {
            'event': y_train['event'].tolist(),
            'time': y_train['time'].tolist()
        },
        "y_val": {
            'event': y_val['event'].tolist(),
            'time': y_val['time'].tolist()
        },
        "ensemble_prediction": ensemble_pred.tolist(),
        "model_names": list(trained_models.keys())
    }
    
    joblib.dump(risk_data, f"optimized_models/risk_scores_seed_{seed}.pkl")
    
    print(f"   ✅ All models and components saved to 'optimized_models/' directory")
    
    return trained_models, ensemble_predictor, training_metadata

def create_inference_pipeline(seed=42):
    """Create a complete inference pipeline for new data."""
    
    class OptimizedSurvivalPipeline:
        def __init__(self, seed=42):
            self.seed = seed
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.models = {}
            self.scaler = None
            self.ensemble_weights = {}
            self.metadata = {}
            
        def load_models(self):
            """Load all trained models and components."""
            print(f"🔄 Loading optimized models for seed {self.seed}...")
            
            # Load metadata
            with open(f"optimized_models/training_metadata_seed_{self.seed}.json", 'r') as f:
                self.metadata = json.load(f)
            
            # Load scaler
            self.scaler = joblib.load(f"optimized_models/scaler_seed_{self.seed}.pkl")
            
            # Load individual models
            model_files = {
                'rsf': f"optimized_models/rsf_seed_{self.seed}.pkl",
                'gbsa': f"optimized_models/gbsa_seed_{self.seed}.pkl",
                'cox_pca': f"optimized_models/cox_pca_seed_{self.seed}.pkl",
                'coxnet': f"optimized_models/coxnet_seed_{self.seed}.pkl",
                'survival_svm': f"optimized_models/survival_svm_seed_{self.seed}.pkl"
            }
            
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[model_name] = joblib.load(file_path)
                    print(f"   ✅ Loaded {model_name}")
                else:
                    print(f"   ⚠️ Model file not found: {file_path}")
            
            # Load MLP if available
            mlp_path = f"optimized_models/mlp_seed_{self.seed}.pt"
            mlp_arch_path = f"optimized_models/mlp_architecture_seed_{self.seed}.json"
            
            if os.path.exists(mlp_path) and os.path.exists(mlp_arch_path):
                with open(mlp_arch_path, 'r') as f:
                    mlp_info = json.load(f)
                
                mlp_model = OptimizedMLP(
                    input_dim=mlp_info['input_dim'],
                    hidden_dims=mlp_info['hidden_dims'],
                    dropout=mlp_info['dropout'],
                    num_heads=mlp_info['num_heads'],
                    use_attention=mlp_info['use_attention'],
                    activation=mlp_info['activation'],
                    batch_norm=mlp_info['batch_norm']
                ).to(self.device)
                
                mlp_model.load_state_dict(torch.load(mlp_path, map_location=self.device))
                mlp_model.eval()
                self.models['mlp'] = mlp_model
                print(f"   ✅ Loaded MLP")
            
            # Load ensemble weights
            self.ensemble_weights = self.metadata['ensemble_weights']
            
            print(f"   📊 Pipeline ready with {len(self.models)} models")
            
        def predict(self, X_new):
            """Make predictions on new data."""
            if self.scaler is None:
                raise RuntimeError("Models not loaded. Call load_models() first.")
            
            # Scale input data
            X_scaled = self.scaler.transform(X_new)
            
            # Get predictions from each model
            predictions = {}
            
            if 'rsf' in self.models:
                predictions['rsf'] = self.models['rsf'].predict(X_scaled)
            
            if 'gbsa' in self.models:
                predictions['gbsa'] = self.models['gbsa'].predict(X_scaled)
            
            if 'cox_pca' in self.models:
                cox_model, pca = self.models['cox_pca']
                X_pca = pca.transform(X_scaled)
                predictions['cox_pca'] = cox_model.predict(X_pca)
            
            if 'coxnet' in self.models:
                coxnet_model, pca = self.models['coxnet']
                X_pca = pca.transform(X_scaled)
                predictions['coxnet'] = coxnet_model.predict(X_pca)
            
            if 'survival_svm' in self.models:
                predictions['survival_svm'] = self.models['survival_svm'].predict(X_scaled)
            
            if 'mlp' in self.models:
                mlp_model = self.models['mlp']
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                    mlp_preds = mlp_model(X_tensor).cpu().numpy().flatten()
                predictions['mlp'] = mlp_preds
            
            # Create weighted ensemble
            ensemble_pred = np.zeros(X_new.shape[0])
            total_weight = 0
            
            for model_name, weight in self.ensemble_weights.items():
                if model_name in predictions:
                    ensemble_pred += weight * predictions[model_name]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            return {
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': predictions,
                'ensemble_weights': self.ensemble_weights
            }
        
        def get_model_info(self):
            """Get information about loaded models."""
            return {
                'seed': self.seed,
                'loaded_models': list(self.models.keys()),
                'ensemble_weights': self.ensemble_weights,
                'training_performance': self.metadata.get('individual_performances', {}),
                'ensemble_performance': self.metadata.get('ensemble_performance', None)
            }
    
    return OptimizedSurvivalPipeline(seed)

def train_all_seeds():
    """Train models for all available seeds with their optimized parameters."""
    print(f"\n{'='*70}")
    print("TRAINING ALL OPTIMIZED MODELS")
    print(f"{'='*70}")
    
    seeds = [42, 121, 144, 245, 1212]
    all_results = {}
    
    for seed in seeds:
        print(f"\n🎯 Training models for seed {seed}")
        try:
            models, predictor, metadata = train_optimized_models(seed, use_best_seed_params=True)
            all_results[seed] = {
                'models': list(models.keys()),
                'ensemble_performance': metadata['ensemble_performance'],
                'individual_performances': metadata['individual_performances']
            }
            print(f"✅ Seed {seed} training completed")
            
        except Exception as e:
            print(f"❌ Seed {seed} training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary of all trained models
    with open("optimized_models/all_seeds_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ All seeds training completed!")
    print(f"📊 Summary saved to 'optimized_models/all_seeds_summary.json'")
    
    return all_results

def create_test_script():
    """Create a standalone testing script."""
    
    test_script = '''#!/usr/bin/env python3
"""
Testing script for optimized survival models.
Usage: python test_optimized_models.py --seed 42 --test_data path/to/test_data
"""

import argparse
import numpy as np
import torch
import joblib
import json
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
from optimized_training import OptimizedMLP, create_inference_pipeline

def load_test_data(test_path):
    """Load test data - implement according to your data format."""
    # This is a placeholder - implement based on your test data format
    pass

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    try:
        cindex = concordance_index_censored(y_true['event'], y_true['time'], y_pred)[0]
        return {
            'cindex': cindex,
            'n_samples': len(y_pred),
            'n_events': sum(y_true['event'])
        }
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {'cindex': 0.5, 'n_samples': len(y_pred), 'n_events': 0}

def main():
    parser = argparse.ArgumentParser(description='Test optimized survival models')
    parser.add_argument('--seed', type=int, default=42, help='Seed for model selection')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"🧪 Testing optimized models with seed {args.seed}")
    
    # Create inference pipeline
    pipeline = create_inference_pipeline(args.seed)
    pipeline.load_models()
    
    # Load test data
    X_test, y_test = load_test_data(args.test_data)
    
    # Make predictions
    results = pipeline.predict(X_test)
    
    # Evaluate results
    ensemble_eval = evaluate_model(y_test, results['ensemble_prediction'])
    
    print(f"\\n📊 TEST RESULTS:")
    print(f"   Ensemble C-index: {ensemble_eval['cindex']:.4f}")
    print(f"   Test samples: {ensemble_eval['n_samples']}")
    print(f"   Events: {ensemble_eval['n_events']}")
    
    # Individual model evaluation
    individual_evals = {}
    for model_name, preds in results['individual_predictions'].items():
        individual_evals[model_name] = evaluate_model(y_test, preds)
        print(f"   {model_name:15}: {individual_evals[model_name]['cindex']:.4f}")
    
    # Save test results
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_results = {
        'seed': args.seed,
        'ensemble_evaluation': ensemble_eval,
        'individual_evaluations': individual_evals,
        'model_info': pipeline.get_model_info(),
        'test_predictions': {
            'ensemble': results['ensemble_prediction'].tolist(),
            'individual': {k: v.tolist() for k, v in results['individual_predictions'].items()}
        }
    }
    
    with open(f"{args.output_dir}/test_results_seed_{args.seed}.json", 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\\n💾 Test results saved to '{args.output_dir}/test_results_seed_{args.seed}.json'")

if __name__ == "__main__":
    main()
'''
    
    with open("test_optimized_models.py", 'w') as f:
        f.write(test_script)
    
    print(f"📝 Test script created: 'test_optimized_models.py'")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train optimized survival models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--all_seeds', action='store_true', help='Train models for all seeds')
    parser.add_argument('--create_test_script', action='store_true', help='Create test script')
    
    args = parser.parse_args()
    
    try:
        if args.create_test_script:
            create_test_script()
        
        if args.all_seeds:
            print("🚀 Training models for all seeds...")
            results = train_all_seeds()
            
            # Create inference pipelines for all seeds
            print(f"\n🔧 Creating inference pipelines...")
            for seed in [42, 121, 144, 245, 1212]:
                if seed in results:
                    try:
                        pipeline = create_inference_pipeline(seed)
                        print(f"   ✅ Pipeline created for seed {seed}")
                    except Exception as e:
                        print(f"   ❌ Pipeline creation failed for seed {seed}: {e}")
        else:
            print(f"🚀 Training models for seed {args.seed}...")
            models, predictor, metadata = train_optimized_models(args.seed)
            
            print(f"\n✅ Training completed for seed {args.seed}")
            print(f"📊 Ensemble C-index: {metadata['ensemble_performance']:.4f}")
            print(f"🔧 Use create_inference_pipeline({args.seed}) to create predictor")
    
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()

# -------------------------
# Additional Utility Functions
# -------------------------
def load_trained_ensemble(seed=42):
    """Convenience function to load a trained ensemble."""
    pipeline = create_inference_pipeline(seed)
    pipeline.load_models()
    return pipeline

def compare_all_ensembles(X_test, y_test):
    """Compare ensembles from all trained seeds."""
    print(f"🔍 Comparing all trained ensembles...")
    
    results = {}
    seeds = [42, 121, 144, 245, 1212]
    
    for seed in seeds:
        try:
            pipeline = load_trained_ensemble(seed)
            predictions = pipeline.predict(X_test)
            
            ensemble_cindex = compute_cindex_from_scores(y_test, predictions['ensemble_prediction'])
            results[seed] = {
                'ensemble_cindex': ensemble_cindex,
                'individual_cindices': {}
            }
            
            for model_name, preds in predictions['individual_predictions'].items():
                model_cindex = compute_cindex_from_scores(y_test, preds)
                results[seed]['individual_cindices'][model_name] = model_cindex
            
            print(f"   Seed {seed}: Ensemble C-index = {ensemble_cindex:.4f}")
            
        except Exception as e:
            print(f"   ❌ Seed {seed} evaluation failed: {e}")
    
    return results

def get_model_explanations(seed=42):
    """Get explanations for model decisions (feature importance, etc.)."""
    print(f"📋 Generating model explanations for seed {seed}...")
    
    explanations = {}
    
    # Load models
    pipeline = load_trained_ensemble(seed)
    
    # Random Forest feature importance
    if 'rsf' in pipeline.models:
        rsf_model = pipeline.models['rsf']
        explanations['rsf_feature_importance'] = rsf_model.feature_importances_.tolist()
    
    # Gradient Boosting feature importance
    if 'gbsa' in pipeline.models:
        gbsa_model = pipeline.models['gbsa']
        explanations['gbsa_feature_importance'] = gbsa_model.feature_importances_.tolist()
    
    # Ensemble weights
    explanations['ensemble_weights'] = pipeline.ensemble_weights
    
    # Save explanations
    with open(f"optimized_models/model_explanations_seed_{seed}.json", 'w') as f:
        json.dump(explanations, f, indent=2)
    
    print(f"   ✅ Explanations saved to 'optimized_models/model_explanations_seed_{seed}.json'")
    
    return explanations

# Example usage and documentation
USAGE_EXAMPLES = '''
# Example Usage:

## 1. Train models for a specific seed
python optimized_training.py --seed 42

## 2. Train models for all seeds
python optimized_training.py --all_seeds

## 3. Create test script
python optimized_training.py --create_test_script

## 4. In Python code:
from optimized_training import load_trained_ensemble, compare_all_ensembles

# Load a specific ensemble
pipeline = load_trained_ensemble(seed=42)

# Make predictions on new data
predictions = pipeline.predict(X_new)
ensemble_pred = predictions['ensemble_prediction']
individual_preds = predictions['individual_predictions']

# Compare all ensembles
results = compare_all_ensembles(X_test, y_test)
'''

if __name__ == "__main__":
    print("📚 Optimized Survival Model Training Script")
    print("=" * 50)
    print(USAGE_EXAMPLES)