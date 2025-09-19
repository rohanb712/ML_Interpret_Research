import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

SEED = 42
pl.seed_everything(SEED, workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Base classifier
FFN_TYPE     = "ReLU"     # "ReLU" or "GeGLU"
HIDDEN_DIM   = 128
LR_FFN       = 1e-3
EPOCHS_FFN   = 3
BATCH_FFN    = 256

# Buffer cleaning
NORMALIZE_Z  = True       # standardize pre-activations per-dimension

# SAE (JumpReLU) core
SAE_LATENTS        = 512
TAU                = 0.1  # STE temperature
INIT_THETA         = 0.5
LR_SAE_FINAL       = 1e-3
EPOCHS_SAE_FINAL   = 5
BATCH_SAE          = 512

# Sparsity targets (fixed #active features per sample)
TARGET_ACTIVES     = [20]  # add 10,40 later if you want more points

# Lambda calibration (fast grid; 1 epoch per lambda)
LAMBDA_GRID = np.geomspace(1e-5, 1e-1, 8).tolist()
EPOCHS_SAE_CAL = 1

OUT_DIR = "artifacts_mnist_sae"
os.makedirs(OUT_DIR, exist_ok=True)


def load_mnist(batch_size=BATCH_FFN):
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test       = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    train_len  = int(0.8 * len(train_full))
    val_len    = len(train_full) - train_len
    train_ds, val_ds = random_split(train_full, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test,     batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_mnist()


class FFN_GeGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W_in   = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_gate = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_out  = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.02)
    def forward(self, x_flat):
        x_proj = torch.einsum('bi,ih->bh', x_flat, self.W_in)
        gate   = F.gelu(torch.einsum('bi,ih->bh', x_flat, self.W_gate))
        h = x_proj * gate
        return torch.einsum('bh,ho->bo', h, self.W_out)

class FFN_ReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W_in  = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.02)
    def forward(self, x_flat):
        z = torch.einsum('bi,ih->bh', x_flat, self.W_in)
        h = F.relu(z)
        return torch.einsum('bh,ho->bo', h, self.W_out)

class MNIST_FFN(pl.LightningModule):
    def __init__(self, ffn_type="ReLU", hidden_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        input_dim, output_dim = 28*28, 10
        if ffn_type == "GeGLU":
            self.ffn = FFN_GeGLU(input_dim, hidden_dim, output_dim)
        elif ffn_type == "ReLU":
            self.ffn = FFN_ReLU(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError("Invalid ffn_type")
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.ffn(x_flat)
    def training_step(self, batch, _):
        x, y = batch
        return F.cross_entropy(self(x), y)
    def validation_step(self, batch, _):
        x, y = batch
        acc = (self(x).argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class JumpReLU_STE(nn.Module):
    """Forward: y = z * 1[z > theta]; Backward: sigmoid surrogate with temperature tau."""
    def __init__(self, n_latents, init_theta=0.5, tau=0.1):
        super().__init__()
        self.theta = nn.Parameter(torch.full((n_latents,), float(init_theta)))
        self.tau   = tau
    def forward(self, z):
        hard = (z > self.theta).float()
        soft = torch.sigmoid((z - self.theta) / self.tau)
        gate = (hard - soft).detach() + soft
        return z * gate

class SAE_JumpReLU(pl.LightningModule):
    """Loss = MSE + λ * E[L0]; L0 uses SOFT gate for gradients, HARD gate for reporting."""
    def __init__(self, n_in, n_latents=512, lambda_l0=1e-2, lr=1e-3, init_theta=0.5, tau=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(n_in, n_latents, bias=True)
        self.theta = nn.Parameter(torch.full((n_latents,), float(init_theta)))
        self.tau   = tau
        self.dec = nn.Linear(n_latents, n_in, bias=True)

    def forward(self, x):
        u = self.enc(x)
        # soft gate for STE/penalty
        soft = torch.sigmoid((u - self.theta) / self.tau)
        # hard gate for forward masking (STE trick)
        hard = (u > self.theta).float()
        gate = (hard - soft).detach() + soft
        f = u * gate
        xh = self.dec(f)
        return xh, f, u, soft, hard

    def _step(self, batch):
        (x,) = batch
        xh, f, u, soft, hard = self(x)
        recon = F.mse_loss(xh, x, reduction="mean")
        # L0 penalty: use SOFT gate so gradients flow sensibly
        l0_soft = soft.sum(dim=1).mean()
        loss = recon + self.hparams.lambda_l0 * l0_soft
        # reporting with HARD gate
        l0_hard = hard.sum(dim=1).float().mean().detach()
        return loss, recon.detach(), l0_soft.detach(), l0_hard

    def training_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({
            "train/recon_mse": recon,
            "train/l0_soft": l0_soft,
            "train/l0_hard": l0_hard
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, recon, l0_soft, l0_hard = self._step(batch)
        self.log_dict({
            "val/recon_mse": recon,
            "val/l0_soft": l0_soft,
            "val/l0_hard": l0_hard
        }, prog_bar=True)

    def on_after_backward(self):
        with torch.no_grad():
            W = self.dec.weight.data
            norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.dec.weight.data = W / norms

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def tloader(t, bs=BATCH_SAE, shuffle=True):
    return DataLoader(TensorDataset(t), batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True)

@torch.no_grad()
def gate_counts(sae, Z, batch=4096):
    """Mean #actives using HARD gate b=1[u>θ]."""
    sae.eval()
    dev = next(sae.parameters()).device
    tot = []
    for i in range(0, Z.size(0), batch):
        u = sae.enc(Z[i:i+batch].to(dev))
        hard = (u > sae.theta).float()
        tot.append(hard.sum(dim=1).cpu())
    return float(torch.cat(tot).mean())

def _get_theta(sae):
    return sae.theta if hasattr(sae, "theta") else sae.act.theta

def sae_forward_modes(sae, z_raw, mu, std, mode="jumprelu"):
    dev = z_raw.device
    sae.eval().to(dev)
    mu_t  = mu.to(dev)  if isinstance(mu,  torch.Tensor) else mu
    std_t = std.to(dev) if isinstance(std, torch.Tensor) else std
    z = (z_raw - mu_t) / std_t if isinstance(mu_t, torch.Tensor) else z_raw

    u = sae.enc(z)
    theta = _get_theta(sae)
    if mode == "jumprelu":
        b = (u > theta).float()
        f = u * b
    elif mode == "boolean":
        f = (u > theta).float()
    else:
        raise ValueError("mode must be 'jumprelu' or 'boolean'")

    xh = sae.dec(f)
    if isinstance(mu_t, torch.Tensor):
        xh = xh * std_t + mu_t
    return xh

@torch.no_grad()
def test_accuracy_with_mode(model, loader, ffn_type, sae, mu, std, mode="jumprelu"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    sae.eval().to(dev)

    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        x_flat = xb.view(xb.size(0), -1)
        z_raw  = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in)
        z_hat  = sae_forward_modes(sae, z_raw, mu, std, mode=mode)

        if ffn_type == "ReLU":
            h = F.relu(z_hat)
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        else:
            gate = F.gelu(torch.einsum('bi,ih->bh', x_flat, model.ffn.W_gate))
            h = z_hat * gate
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    return correct / max(total, 1)

@torch.no_grad()
def mean_active_on(sae, Z):
    sae.eval().to(device)
    acts = []
    for i in range(0, Z.size(0), 2048):
        batch = Z[i:i+2048].to(device)
        u = sae.enc(batch)
        theta = sae.theta if hasattr(sae, "theta") else sae.act.theta
        f = (u > theta).float() * u  # JumpReLU definition (nonzero where gate=1)
        acts.append((f > 0).float().sum(dim=1).cpu())
    return float(torch.cat(acts).mean())

def train_sae_once(n_in, lam, Ztr, Zva, epochs, tau=TAU, init_theta=INIT_THETA, latents=SAE_LATENTS, lr=LR_SAE_FINAL):
    sae = SAE_JumpReLU(n_in=n_in, n_latents=latents, lambda_l0=lam, lr=lr, init_theta=init_theta, tau=tau)
    tr = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=False, enable_checkpointing=False
    )
    tr.fit(sae, tloader(Ztr, shuffle=True), tloader(Zva, shuffle=False))
    return sae

def calibrate_lambda(Ztr, Zva, target_actives, coarse_grid=None, refine_factor=3, refine_steps=5):
    """
    1) coarse search over λ grid (train 1 epoch each),
    2) choose best by HARD gate count,
    3) local refine around the winner (geo-spaced).
    """
    n_in = Ztr.shape[1]
    if coarse_grid is None:
        # Wider range to ensure we can hit small #actives
        coarse_grid = np.geomspace(1e-6, 1e-0, 12)

    # coarse pass
    best = None
    for lam in coarse_grid:
        sae = train_sae_once(n_in, lam, Ztr, Zva, epochs=1)
        m_act = gate_counts(sae, Zva)
        gap = abs(m_act - target_actives)
        if (best is None) or (gap < best["gap"]):
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}

    # refine around best (geo range around lam*)
    lam_star = best["lam"]
    low = lam_star / (refine_factor**2)
    high = lam_star * (refine_factor**2)
    refine_grid = np.geomspace(max(low, 1e-8), min(high, 1.0), refine_steps)

    for lam in refine_grid:
        sae = train_sae_once(n_in, lam, Ztr, Zva, epochs=1)
        m_act = gate_counts(sae, Zva)
        gap = abs(m_act - target_actives)
        if gap < best["gap"]:
            best = {"lam": float(lam), "sae": sae, "m_act": float(m_act), "gap": float(gap)}

    return best  # dict: lam, sae, m_act (HARD), gap

@torch.no_grad()
def theta_shift_to_target(sae, Z_val_n, target, s_low=-2.0, s_high=2.0, steps=12):
    base = sae.theta.data.clone() if hasattr(sae, "theta") else sae.act.theta.data.clone()
    def set_theta(s):
        if hasattr(sae, "theta"):
            sae.theta.data = base + s
        else:
            sae.act.theta.data = base + s

    best_s, best_gap = 0.0, float("inf")
    lo, hi = s_low, s_high
    for _ in range(steps):
        s = 0.5 * (lo + hi)
        set_theta(s)
        m = gate_counts(sae, Z_val_n)
        gap = abs(m - target)
        if gap < best_gap:
            best_gap, best_s = gap, s
        # monotonic: increasing s raises θ → fewer actives
        if m < target:
            hi = s  # too sparse → lower θ → move left
        else:
            lo = s  # too dense → raise θ → move right
    set_theta(best_s)
    return float(best_s), float(gate_counts(sae, Z_val_n))

# All function definitions
@torch.no_grad()
def baseline_accuracy(model, loader):
    model.eval().to(device)
    correct=total=0
    for xb, yb in loader:
        preds = model(xb.to(device)).argmax(dim=1)
        correct += (preds == yb.to(device)).sum().item()
        total   += yb.numel()
    return correct / max(total,1)

@torch.no_grad()
def collect_preactivations(model, loader, ffn_type):
    model.eval().to(device)
    feats = []
    for xb, _ in loader:
        x_flat = xb.to(device).view(xb.size(0), -1)
        z = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in)  # main branch for both types
        feats.append(z.cpu())
    return torch.cat(feats, dim=0)

@torch.no_grad()
def sae_forward_modes(sae, z_raw, mu, std, mode="jumprelu"):
    dev = z_raw.device
    sae.eval().to(dev)
    mu_t  = mu.to(dev)  if isinstance(mu,  torch.Tensor) else mu
    std_t = std.to(dev) if isinstance(std, torch.Tensor) else std
    z = (z_raw - mu_t) / std_t if isinstance(mu_t, torch.Tensor) else z_raw

    u = sae.enc(z)
    theta = _get_theta(sae)
    if mode == "jumprelu":
        b = (u > theta).float()
        f = u * b
    elif mode == "boolean":
        f = (u > theta).float()
    else:
        raise ValueError("mode must be 'jumprelu' or 'boolean'")

    xh = sae.dec(f)
    if isinstance(mu_t, torch.Tensor):
        xh = xh * std_t + mu_t
    return xh

@torch.no_grad()
def test_accuracy_with_mode(model, loader, ffn_type, sae, mu, std, mode="jumprelu"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    sae.eval().to(dev)

    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        x_flat = xb.view(xb.size(0), -1)
        z_raw  = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in)
        z_hat  = sae_forward_modes(sae, z_raw, mu, std, mode=mode)

        if ffn_type == "ReLU":
            h = F.relu(z_hat)
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        else:
            gate = F.gelu(torch.einsum('bi,ih->bh', x_flat, model.ffn.W_gate))
            h = z_hat * gate
            logits = torch.einsum('bh,ho->bo', h, model.ffn.W_out)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    return correct / max(total, 1)

@torch.no_grad()
def collect_feature_bank(loader, model, sae, mu, std, topk=16):
    """
    For each latent k:
      - top activating examples (image tensor, label, score u_k)
      - label histogram when active (hard gate)
      - mean hard-activation rate
    """
    from collections import Counter

    # cache raw inputs and labels
    imgs, labels = [], []
    for xb, yb in loader:
        imgs.append(xb)
        labels.append(yb)
    imgs = torch.cat(imgs, 0)                  # [N,1,28,28]
    labels = torch.cat(labels, 0)              # [N]
    N = imgs.size(0)

    # compute z_raw then normalized z
    with torch.no_grad():
        dev = next(sae.parameters()).device
        x_flat = imgs.view(N, -1).to(dev)
        z_raw  = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in.to(dev))
        mu_t, std_t = mu.to(dev), std.to(dev)
        z = (z_raw - mu_t) / std_t

        u = sae.enc(z)                         # [N, m]
        theta = sae.theta if hasattr(sae,'theta') else sae.act.theta
        hard = (u > theta).float()             # [N, m]

    u_cpu    = u.cpu()
    hard_cpu = hard.cpu().bool()
    imgs_cpu = imgs
    labels_c = labels

    m = u_cpu.size(1)
    bank = {
        "top_indices": [None]*m,
        "top_scores":  [None]*m,
        "label_hist":  [None]*m,
        "mean_active": [None]*m
    }

    for k in range(m):
        uk = u_cpu[:, k]
        hk = hard_cpu[:, k]
        # Top-activating examples by u_k
        top_vals, top_idx = torch.topk(uk, k=min(topk, N))
        # Label histogram where hard gate is on
        lab_k = labels_c[hk].tolist()
        hist  = Counter(lab_k)
        bank["top_indices"][k] = top_idx.tolist()
        bank["top_scores"][k]  = top_vals.tolist()
        bank["label_hist"][k] = dict(sorted(hist.items()))
        bank["mean_active"][k] = float(hk.float().mean())
    return bank, imgs_cpu, labels_c

def feature_input_pattern(model, sae, mu, std, k):
    """Map SAE feature k to input-space pattern via gradient computation"""
    with torch.no_grad():
        W_in = model.ffn.W_in.detach().cpu()            # [784, 128]
        enc_k = sae.enc.weight[k, :].detach().cpu()     # [128] - row k of encoder weight
        pat = W_in @ (enc_k / std.squeeze(0).cpu())     # [784]
    return pat.view(28,28)

@torch.no_grad()
def feature_logit_effect(model, loader, sae, mu, std, k, alpha=1.0, n_eval=2000):
    """Measure how activating feature k affects model logits"""
    dev = next(sae.parameters()).device
    model.eval().to(dev)
    sae.eval().to(dev)

    deltas = []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(dev)
        B = xb.size(0)
        x_flat = xb.view(B, -1)
        z_raw = torch.einsum('bi,ih->bh', x_flat, model.ffn.W_in.to(dev))          # [B,128]

        # baseline path through SAE
        z_hat = sae_forward_modes(sae, z_raw, mu, std, mode="jumprelu")            # [B,128]

        # bump along decoder column k in normalized coordinates, then de-normalize
        dec_col = sae.dec.weight[:, k].unsqueeze(0).to(dev)                        # [1,128]
        z_hat_pert = z_hat + alpha * dec_col                                       # already in original z space (we returned de-normalized)

        # logits baseline
        if FFN_TYPE == "ReLU":
            h0 = F.relu(z_hat)
            h1 = F.relu(z_hat_pert)
            W_out = model.ffn.W_out.to(dev)
            log0 = torch.einsum('bh,ho->bo', h0, W_out)
            log1 = torch.einsum('bh,ho->bo', h1, W_out)
        else:
            # GeGLU path if needed
            gate = F.gelu(torch.einsum('bi,ih->bh', x_flat, model.ffn.W_gate.to(dev)))
            h0 = z_hat * gate
            h1 = z_hat_pert * gate
            W_out = model.ffn.W_out.to(dev)
            log0 = torch.einsum('bh,ho->bo', h0, W_out)
            log1 = torch.einsum('bh,ho->bo', h1, W_out)

        deltas.append((log1 - log0).detach().cpu())
        cnt += B
        if cnt >= n_eval:
            break

    deltas = torch.cat(deltas, 0)[:n_eval]     # [n_eval, 10]
    mean_delta = deltas.mean(dim=0)            # [10]
    top_class = int(mean_delta.argmax().item())
    return mean_delta.numpy(), top_class

def show_feature_gallery(k, bank, imgs, save_path=None, cols=10):
    """Visualize top-activating examples for a feature"""
    import matplotlib.pyplot as plt
    idxs = bank["top_indices"][k]
    n = len(idxs)
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*1.2, rows*1.2))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[idx,0].numpy(), cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Feature {k} top-activating examples | hist={bank['label_hist'][k]} | mean_active={bank['mean_active'][k]:.3f}")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_feature_pattern(heat, k, save_path=None):
    """Visualize feature input-space pattern"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.imshow(heat, cmap="bwr", vmin=-abs(heat).max(), vmax=abs(heat).max())
    plt.title(f"Feature {k} input-space pattern")
    plt.axis("off")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    model = MNIST_FFN(FFN_TYPE, HIDDEN_DIM, LR_FFN)
    trainer = pl.Trainer(
        max_epochs=EPOCHS_FFN,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=False, enable_checkpointing=False
    )
    trainer.fit(model, train_loader, val_loader)

    base_test_acc = baseline_accuracy(model, test_loader)
    print(f"Baseline test accuracy: {base_test_acc:.4f}")

    Z_train = collect_preactivations(model, train_loader, FFN_TYPE)
    Z_val   = collect_preactivations(model, val_loader,   FFN_TYPE)
    Z_test  = collect_preactivations(model, test_loader,  FFN_TYPE)
    print("Buffers:", Z_train.shape, Z_val.shape, Z_test.shape)

    # Optional cleaning (standardize per-dim)
    if NORMALIZE_Z:
        mu  = Z_train.mean(0, keepdim=True)
        std = Z_train.std(0, keepdim=True).clamp_min(1e-6)
        Z_train_n = (Z_train - mu) / std
        Z_val_n   = (Z_val   - mu) / std
        Z_test_n  = (Z_test  - mu) / std
    else:
        mu, std = 0.0, 1.0
        Z_train_n, Z_val_n, Z_test_n = Z_train, Z_val, Z_test

    # 🚀 Train calibrated SAE(s) and evaluate fidelity
    results = []

    for target_k in TARGET_ACTIVES:
        print(f"\n=== Calibrating for target actives ≈ {target_k} ===")
        # Calibrate λ quickly (1 epoch per λ), pick best, then train final SAE longer
        pick = calibrate_lambda(Z_train_n, Z_val_n, target_k)
        # NOTE: calibrate_lambda returns keys: 'lam', 'sae', 'm_act', 'gap'
        print(f"Picked λ={pick['lam']:.2e} with achieved actives ≈ {pick['m_act']:.2f} (cal)")

        # Train final SAE with chosen λ for more epochs
        sae_final = train_sae_once(Z_train_n.shape[1], pick["lam"], Z_train_n, Z_val_n, epochs=EPOCHS_SAE_FINAL)

        # Stats on validation buffer (HARD gate count)
        achieved_k = gate_counts(sae_final, Z_val_n)

        # Fidelity on test set
        acc_baseline = base_test_acc
        acc_jr   = test_accuracy_with_mode(model, test_loader, FFN_TYPE, sae_final, mu, std, mode="jumprelu")
        acc_bool = test_accuracy_with_mode(model, test_loader, FFN_TYPE, sae_final, mu, std, mode="boolean")

        row = {
            "target_actives": target_k,
            "achieved_actives_val": round(achieved_k, 2),
            "lambda": pick["lam"],
            "baseline_acc": round(acc_baseline, 4),
            "recon_acc_jumprelu": round(acc_jr, 4),
            "recon_acc_boolean": round(acc_bool, 4),
            "delta_acc_jumprelu": round(acc_baseline - acc_jr, 4),
            "delta_acc_boolean": round(acc_baseline - acc_bool, 4),
            "sae_latents": SAE_LATENTS,
            "tau": TAU,
            "normalize_z": NORMALIZE_Z
        }
        results.append(row)
        print(row)

    # Save summary
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(results)

    # Use theta shift adjustment
    target_k = 20
    shift, achieved = theta_shift_to_target(sae_final, Z_val_n, target_k)
    print(f"Applied θ shift: {shift:.3f}; achieved actives ≈ {achieved:.2f}")

    # Re-evaluate fidelity with the adjusted θ
    acc_jr_adj = test_accuracy_with_mode(model, test_loader, FFN_TYPE, sae_final, mu, std, mode="jumprelu")
    print(f"Re-eval JumpReLU accuracy after θ shift: {acc_jr_adj:.4f} (baseline {base_test_acc:.4f})")

    # 🚀 Feature Analysis and Hypothesis Generation
    print("\n=== Feature Analysis ===")
    print("Collecting feature bank...")
    feat_bank, bank_imgs, bank_labels = collect_feature_bank(test_loader, model, sae_final, mu, std, topk=20)
    print(f"Example: mean active first 10 features: {feat_bank['mean_active'][:10]}")

    # Generate feature report for hypothesis generation
    print("Generating feature report...")
    rows = []
    m = sae_final.dec.weight.shape[1]
    for k in range(min(m, 50)):  # Analyze first 50 features for speed
        hist = feat_bank["label_hist"][k]
        top_lab = max(hist.items(), key=lambda kv: kv[1])[0] if hist else None
        mean_act = feat_bank["mean_active"][k]
        md, top_cls = feature_logit_effect(model, test_loader, sae_final, mu, std, k, alpha=1.0, n_eval=1000)
        rows.append({
            "feature": k,
            "mean_active": round(mean_act, 4),
            "label_mode_when_active": top_lab,
            "logit_boost_class": int(top_cls),
            "logit_delta_vector": md.tolist()
        })

    # Sort by activation frequency for easy browsing
    import pandas as pd
    feat_df = pd.DataFrame(rows).sort_values("mean_active", ascending=False)
    print("Top 10 most active features:")
    print(feat_df.head(10)[["feature", "mean_active", "label_mode_when_active", "logit_boost_class"]])

    # Save feature report
    feat_df.to_json(os.path.join(OUT_DIR, "feature_report.json"), orient="records", indent=2)
    print(f"Feature report saved to {OUT_DIR}/feature_report.json")

    # Example analysis for a few interesting features
    print("\n=== Example Feature Analysis ===")
    interesting_features = feat_df.head(5)["feature"].tolist()

    for k in interesting_features:
        print(f"\nFeature {k}:")
        print(f"  Mean active: {feat_bank['mean_active'][k]:.4f}")
        print(f"  Label histogram: {feat_bank['label_hist'][k]}")
        md, top_cls = feature_logit_effect(model, test_loader, sae_final, mu, std, k, alpha=1.0, n_eval=1000)
        print(f"  Boosts class {top_cls} by {md[top_cls]:.4f}")
        print(f"  Top activating examples: {feat_bank['top_indices'][k][:5]}")

        # Generate and save input pattern
        heat = feature_input_pattern(model, sae_final, mu, std, k)
        pattern_file = os.path.join(OUT_DIR, f"feature_{k}_pattern.npy")
        np.save(pattern_file, heat.numpy())
        print(f"  Input pattern saved to {pattern_file}")

        # Save visualizations (gallery and pattern)
        gallery_file = os.path.join(OUT_DIR, f"feature_{k}_gallery.png")
        pattern_viz_file = os.path.join(OUT_DIR, f"feature_{k}_pattern.png")

        show_feature_gallery(k, feat_bank, bank_imgs, save_path=gallery_file)
        visualize_feature_pattern(heat, k, save_path=pattern_viz_file)
        print(f"  Gallery saved to {gallery_file}")
        print(f"  Pattern visualization saved to {pattern_viz_file}")

    print(f"\nFeature analysis complete! Check {OUT_DIR}/ for outputs.")
    print("Next steps for manual hypothesis generation:")
    print("1. Review feature_report.json for interesting features")
    print("2. Examine top-activating examples for each feature")
    print("3. Study input patterns (*.npy files) for visual insights")
    print("4. Write hypotheses based on label histograms and logit effects")
