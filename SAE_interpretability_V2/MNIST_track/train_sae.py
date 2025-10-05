"""
Train MLP and SAE models, save checkpoints for later interpretability analysis.

This script trains an MNIST classifier and sparse autoencoder, then saves:
- Trained MLP model
- Trained SAE model
- Normalization parameters
- Configuration

Run once to train, then use interpret_sae.py for fast iteration on analysis.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

# Configuration
SEED = 42
pl.seed_everything(SEED, workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# FFN hyperparameters
FFN_TYPE = "ReLU"
HIDDEN_DIM = 128
LR_FFN = 1e-3
EPOCHS_FFN = 5
BATCH_FFN = 8

# SAE hyperparameters
SAE_LATENTS      = 32
TAU              = 0.1
INIT_THETA       = 0.5
LR_SAE_FINAL     = 1e-3
EPOCHS_SAE_FINAL = 5
BATCH_SAE        = 8
NORMALIZE_Y      = True
TARGET_ACTIVES   = [1, 2, 4]

OUT_DIR = "artifacts_sae_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================

def load_mnist(batch_size=BATCH_FFN):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_train_full = datasets.MNIST(root=".", train=True,  download=True, transform=tfm)
    ds_test       = datasets.MNIST(root=".", train=False, download=True, transform=tfm)

    train_len  = int(0.8 * len(ds_train_full))
    val_len    = len(ds_train_full) - train_len
    ds_train, ds_val = random_split(ds_train_full, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    pin = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return dl_train, dl_val, dl_test

print("Loading MNIST data...")
dl_train, dl_val, dl_test = load_mnist()

# ============================================================================
# Model Definitions
# ============================================================================

class FFN_ReLU(nn.Module):
    def __init__(self, d_i, d_h, d_o):
        super().__init__()
        self.W_in_ih  = nn.Parameter(torch.randn(d_i, d_h) * 0.02)
        self.W_out_ho = nn.Parameter(torch.randn(d_h, d_o) * 0.02)
    def forward(self, x_bi):
        z_bh = torch.einsum('bi,ih->bh', x_bi, self.W_in_ih)
        h_bh = F.relu(z_bh)
        y_bo = torch.einsum('bh,ho->bo', h_bh, self.W_out_ho)
        return y_bo

class MNIST_FFN(pl.LightningModule):
    def __init__(self, ffn_type="ReLU", d_h=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        d_i, d_o = 28*28, 10
        if ffn_type == "ReLU":
            self.ffn = FFN_ReLU(d_i, d_h, d_o)
        else:
            raise ValueError("Invalid ffn_type")
    def forward(self, x_bchw):
        x_bi = x_bchw.view(x_bchw.size(0), -1)
        y_bo = self.ffn(x_bi)
        return y_bo
    def training_step(self, batch, _):
        x_bchw, y_gt = batch
        y_bo = self(x_bchw)
        return F.cross_entropy(y_bo, y_gt)
    def validation_step(self, batch, _):
        x_bchw, y_gt = batch
        y_bo = self(x_bchw)
        acc = (y_bo.argmax(dim=1) == y_gt).float().mean()
        self.log("val_acc", acc, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class SAE_JumpReLU(pl.LightningModule):
    def __init__(self, d_in, d_latents=32, lambda_l0=1e-2, lr=1e-3, init_theta=0.5, tau=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(d_in, d_latents, bias=True)
        self.theta_h = nn.Parameter(torch.full((d_latents,), float(init_theta)))
        self.tau   = tau
        self.dec = nn.Linear(d_latents, d_in, bias=True)
    def forward(self, y_bO):
        u_bh = self.enc(y_bO)
        soft_bh = torch.sigmoid((u_bh - self.theta_h) / self.tau)
        hard_bh = (u_bh > self.theta_h).float()
        gate_bh = (hard_bh - soft_bh).detach() + soft_bh
        f_bh = u_bh * gate_bh
        y_hat_bO = self.dec(f_bh)
        return y_hat_bO, f_bh, u_bh, soft_bh, hard_bh
    def _step(self, batch):
        (y_bO,) = batch
        y_hat_bO, f_bh, u_bh, soft_bh, hard_bh = self(y_bO)
        recon = F.mse_loss(y_hat_bO, y_bO, reduction="mean")
        l0_soft = soft_bh.sum(dim=1).mean()
        loss = recon + self.hparams.lambda_l0 * l0_soft
        l0_hard = hard_bh.sum(dim=1).float().mean().detach()
        return loss, recon.detach(), l0_soft.detach(), l0_hard
    def training_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({"train/recon_mse": recon, "train/l0_soft": l0_soft, "train/l0_hard": l0_hard}, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({"val/recon_mse": recon, "val/l0_soft": l0_soft, "val/l0_hard": l0_hard}, prog_bar=True)
    def on_after_backward(self):
        with torch.no_grad():
            W_hO = self.dec.weight.data
            norms = W_hO.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.dec.weight.data = W_hO / norms
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ============================================================================
# Training Utilities
# ============================================================================

def _trainer(max_epochs):
    use_gpu = torch.cuda.is_available()
    has_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    precision = "bf16-mixed" if (use_gpu and has_bf16) else (16 if use_gpu else 32)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

def tloader(t, bs=BATCH_SAE, shuffle=True):
    pin = torch.cuda.is_available()
    return DataLoader(TensorDataset(t), batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=pin)

@torch.no_grad()
def baseline_accuracy(model, loader):
    model.eval().to(device)
    correct=total=0
    for x_bchw, y_gt in loader:
        y_bo = model(x_bchw.to(device))
        pred = y_bo.argmax(dim=1)
        correct += (pred == y_gt.to(device)).sum().item()
        total   += y_gt.numel()
    return correct / max(total,1)

@torch.no_grad()
def collect_logits_buffers(model, loader):
    model.eval().to(device)
    feats = []
    for x_bchw, _ in loader:
        x_bi = x_bchw.to(device).view(x_bchw.size(0), -1)
        y_bo = model.ffn(x_bi)
        feats.append(y_bo.cpu())
    return torch.cat(feats, dim=0)

@torch.no_grad()
def gate_counts(sae, Y_bO, batch=1024):
    sae.eval().to(device)
    tots = []
    for i in range(0, Y_bO.size(0), batch):
        u_bh = sae.enc(Y_bO[i:i+batch].to(device))
        hard_bh = (u_bh > sae.theta_h).float()
        tots.append(hard_bh.sum(dim=1).cpu())
    return float(torch.cat(tots).mean())

def train_sae_once(d_in, lam, Ytr, Yva, epochs):
    sae = SAE_JumpReLU(d_in=d_in, d_latents=SAE_LATENTS, lambda_l0=lam, lr=LR_SAE_FINAL, init_theta=INIT_THETA, tau=TAU)
    tr = _trainer(epochs)
    tr.fit(sae, tloader(Ytr, shuffle=True), tloader(Yva, shuffle=False))
    return sae

def calibrate_lambda(Ytr, Yva, target_actives, coarse_grid=np.geomspace(1e-6, 1e-1, 10), refine_factor=3, refine_steps=5):
    d_in = Ytr.shape[1]
    best = None
    for lam in coarse_grid:
        sae = train_sae_once(d_in, float(lam), Ytr, Yva, epochs=1)
        m_act = gate_counts(sae, Yva)
        gap = abs(m_act - target_actives)
        if (best is None) or (gap < best["gap"]):
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}
    lam_star = best["lam"]
    low = lam_star / (refine_factor**2)
    high = lam_star * (refine_factor**2)
    refine_grid = np.geomspace(max(low, 1e-8), min(high, 1.0), refine_steps)
    for lam in refine_grid:
        sae = train_sae_once(d_in, float(lam), Ytr, Yva, epochs=1)
        m_act = gate_counts(sae, Yva)
        gap = abs(m_act - target_actives)
        if gap < best["gap"]:
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}
    return best

# ============================================================================
# Train MLP
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Training MNIST Classifier")
print("="*70)

model = MNIST_FFN(FFN_TYPE, HIDDEN_DIM, LR_FFN)
_tr = _trainer(EPOCHS_FFN)
_tr.fit(model, dl_train, dl_val)

base_test_acc = baseline_accuracy(model, dl_test)
print(f"\nBaseline test accuracy: {base_test_acc:.4f}")

# ============================================================================
# Collect Logits & Train SAE
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Collecting Logits and Training SAE")
print("="*70)

Ytr_bo = collect_logits_buffers(model, dl_train)
Yva_bo = collect_logits_buffers(model, dl_val)
Yte_bo = collect_logits_buffers(model, dl_test)
print("Buffers (logits):", Ytr_bo.shape, Yva_bo.shape, Yte_bo.shape)

if NORMALIZE_Y:
    mu_bo  = Ytr_bo.mean(0, keepdim=True)
    std_bo = Ytr_bo.std(0, keepdim=True).clamp_min(1e-6)
    Ytr_n = (Ytr_bo - mu_bo) / std_bo
    Yva_n = (Yva_bo - mu_bo) / std_bo
    Yte_n = (Yte_bo - mu_bo) / std_bo
else:
    mu_bo, std_bo = torch.tensor(0.0), torch.tensor(1.0)
    Ytr_n, Yva_n, Yte_n = Ytr_bo, Yva_bo, Yte_bo

# Fast calibration
FAST_MODE = True
CAL_SAMPLES_TRAIN = 8000 if FAST_MODE else len(Ytr_n)
CAL_SAMPLES_VAL   = 2000 if FAST_MODE else len(Yva_n)
COARSE_GRID       = np.geomspace(1e-5, 1e-2, 4)
REFINE_STEPS      = 3
FINAL_EPOCHS      = min(EPOCHS_SAE_FINAL, 3) if FAST_MODE else EPOCHS_SAE_FINAL

Ytr_cal = Ytr_n[:CAL_SAMPLES_TRAIN].contiguous()
Yva_cal = Yva_n[:CAL_SAMPLES_VAL].contiguous()
print(f"Calibrate on train={len(Ytr_cal)}, val={len(Yva_cal)}")

# Train SAE for target=1 (best for interpretability)
print(f"\n=== Calibrating for target actives ~1 on LOGITS ===")
pick = calibrate_lambda(
    Ytr_cal, Yva_cal, target_actives=1,
    coarse_grid=COARSE_GRID,
    refine_steps=REFINE_STEPS
)
print(f"Picked lambda={pick['lam']:.2e}; achieved actives ~{pick.get('m_act', float('nan')):.2f}")

print("\nTraining final SAE...")
sae_final = train_sae_once(Ytr_cal.shape[1], pick["lam"], Ytr_cal, Yva_cal, epochs=FINAL_EPOCHS)

achieved_k = gate_counts(sae_final, Yva_n)
print(f"Final validation actives: {achieved_k:.2f}")

# ============================================================================
# Save Everything
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Saving Models and Configuration")
print("="*70)

checkpoint = {
    # Models
    'mlp_state_dict': model.state_dict(),
    'sae_state_dict': sae_final.state_dict(),
    
    # Normalization
    'mu_bo': mu_bo,
    'std_bo': std_bo,
    'normalize_y': NORMALIZE_Y,
    
    # Config
    'config': {
        'seed': SEED,
        'ffn_type': FFN_TYPE,
        'hidden_dim': HIDDEN_DIM,
        'sae_latents': SAE_LATENTS,
        'tau': TAU,
        'init_theta': INIT_THETA,
        'target_actives': 1,
        'achieved_actives': achieved_k,
        'lambda': pick['lam'],
        'baseline_acc': base_test_acc,
    },
    
    # Metadata
    'device_trained': str(device),
}

checkpoint_path = os.path.join(OUT_DIR, "trained_models.pt")
torch.save(checkpoint, checkpoint_path)
print(f"Saved checkpoint to: {checkpoint_path}")

# Save config as JSON for easy reading
config_path = os.path.join(OUT_DIR, "training_config.json")
with open(config_path, "w") as f:
    json.dump(checkpoint['config'], f, indent=2)
print(f"Saved config to: {config_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nCheckpoint: {checkpoint_path}")
print(f"Config: {config_path}")
print(f"\nBaseline accuracy: {base_test_acc:.4f}")
print(f"SAE latents: {SAE_LATENTS}")
print(f"Active latents (val): {achieved_k:.2f}")
print("\nNext: Run interpret_sae.py to analyze features")
print("="*70)

