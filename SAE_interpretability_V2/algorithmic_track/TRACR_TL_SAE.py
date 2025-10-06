"""
TRACR + TransformerLens + SAE Training Script

Setup Instructions for Windows:
1. Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt
   pip install git+https://github.com/neelnanda-io/Tracr.git@main

3. Run this script:
   python TRACR_TL_SAE.py

Note: JAX CPU version is used by default. For GPU support on Windows, use WSL2.
"""

import sys
import torch
import jax
import numpy

print("=" * 60)
print("Environment Check")
print("=" * 60)
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("JAX:", jax.__version__, "NumPy:", numpy.__version__)
print("=" * 60)

IN_COLAB = False


# ============================================================================
# Configuration and Setup
# ============================================================================
import os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tracr.rasp import rasp
from tracr.compiler import compiling
import pytorch_lightning as pl
import einops

SEED = 42
pl.seed_everything(SEED, workers=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# SAE config
SAE_LATENTS = 32
LR_SAE, EPOCHS_SAE = 1e-3, 5
BATCH_SAE = 64
LAMBDA_L0_LIST = [1e-2, 5e-3, 1e-3, 5e-4]
OUT_DIR = "sae_artifacts"; os.makedirs(OUT_DIR, exist_ok=True)

# ---- Transcoder (future use) ----
TX_LATENTS, TX_TAU, TX_INIT_TH = 128, 0.1, 0.5
TX_LAMBDA, TX_LR, TX_EPOCHS, TX_BS = 1e-2, 1e-3, 5, 256


# ============================================================================
# Compile RASP Program (Sequence Reversal)
# ============================================================================
def make_length():
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    return rasp.SelectorWidth(all_true_selector)


# ============================================================================
# Helper Functions (will be populated when model is initialized)
# ============================================================================
# These will be set in main()
model = None
tl_model = None
bos = None
INPUT_ENCODER = None
OUTPUT_ENCODER = None
n_layers = None

def create_model_input(input, input_encoder=None):
    """Create model input tensor from sequence."""
    if input_encoder is None:
        input_encoder = INPUT_ENCODER
    encoding = input_encoder.encode(input)
    return torch.tensor(encoding).unsqueeze(dim=0)

def decode_model_output(logits, output_encoder=None, bos_token=None):
    """Decode model output logits to sequence."""
    if output_encoder is None:
        output_encoder = OUTPUT_ENCODER
    if bos_token is None:
        bos_token = INPUT_ENCODER.bos_token
    max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
    decoded_output = output_encoder.decode(max_output_indices.tolist())
    decoded_output_with_bos = [bos_token] + decoded_output[1:]
    return decoded_output_with_bos

@torch.no_grad()
def parity_exact_match(samples=64):
    """Test parity between TransformerLens and Tracr models."""
    # Warm up Tracr JAX model once to avoid repeated compilation
    _ = model.apply([bos, 1, 2, 3])  
    
    ok = 0
    for _ in range(samples):
        L = np.random.randint(1, 5)
        seq = [bos] + list(np.random.choice([1, 2, 3], size=L))
        
        # TransformerLens forward
        tl_out = decode_model_output(tl_model(create_model_input(seq)))
        # Tracr forward (already compiled)
        tr_out = model.apply(seq).decoded
        
        ok += (tl_out == tr_out)
    
    print(f"TL vs Tracr exact-match: {ok}/{samples}")


# ============================================================================
# Data Collection: Sample Sequences & Collect Activations
# ============================================================================

@torch.no_grad()
def sample_sequences(n=8000, max_len=5):
    """Generate BOS + random content sequences, return token tensors + raw seqs."""
    X = []
    for _ in range(n):
        L = np.random.randint(1, max_len)  # 1..max_len-1 after BOS
        seq = [bos] + list(np.random.choice([1, 2, 3], size=L))
        X.append(seq)
    
    # Process sequences one by one to avoid tensor size mismatch
    token_tensors = []
    for seq in X:
        token_tensor = create_model_input(seq)[0]  # Remove batch dimension
        token_tensors.append(token_tensor)
    
    return token_tensors, X

@torch.no_grad()
def collect_activations(token_tensors, hook_point, batch=128):
    """Collect activations at a given hook_point for all sequences."""
    feats = []
    for i in range(0, len(token_tensors), batch):
        batch_tensors = token_tensors[i:i+batch]
        # Pad batch to same length
        max_len = max(t.size(0) for t in batch_tensors)
        padded_batch = []
        for t in batch_tensors:
            if t.size(0) < max_len:
                # Pad with zeros (or use attention mask later)
                pad_size = max_len - t.size(0)
                padded_t = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype, device=t.device)])
            else:
                padded_t = t
            padded_batch.append(padded_t)
        
        b = torch.stack(padded_batch, dim=0)
        _, cache = tl_model.run_with_cache(b)
        act = cache[hook_point]              # (B, T, d_model)
        feats.append(act.reshape(-1, act.size(-1)))  # flatten positions
    return torch.cat(feats, dim=0)


# ============================================================================
# SAE Model Definition and Training
# ============================================================================

class ActDataset(torch.utils.data.Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i]

class SAE(pl.LightningModule):
    def __init__(self, d_in, d_latent=SAE_LATENTS, l1=1e-3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(d_in, d_latent, bias=True)
        self.dec = nn.Linear(d_latent, d_in, bias=False)
        self.l1 = l1; self.lr = lr
        nn.init.kaiming_uniform_(self.enc.weight, a=np.sqrt(5))
        nn.init.zeros_(self.dec.weight)

    def forward(self, x):
        z = F.relu(self.enc(x))
        x_hat = self.dec(z)
        return x_hat, z

    def training_step(self, batch, _):
        x = batch
        x_hat, z = self(x)
        recon = F.mse_loss(x_hat, x)
        spars = z.abs().mean()
        loss = recon + self.l1 * spars
        self.log_dict({"train/recon": recon, "train/l1": spars, "train/loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def train_sae_for_layer(layer, l1, tag):
    """Train SAE for resid_post of a given layer."""
    hook_point = ("resid_post", layer)
    print(f"Training SAE for layer {layer} on {hook_point}")

    toks, _ = sample_sequences(n=8000, max_len=5)
    acts = collect_activations(toks, hook_point).to(device)
    print(f"Layer {layer} activations: {acts.shape}")

    # Split into train/val
    n = acts.size(0)
    idx = torch.randperm(n)
    train = acts[idx[: int(0.9*n)]]
    val   = acts[idx[int(0.9*n):]]

    train_loader = DataLoader(ActDataset(train), batch_size=BATCH_SAE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ActDataset(val),   batch_size=BATCH_SAE, shuffle=False, num_workers=0)

    model = SAE(d_in=acts.size(1), d_latent=SAE_LATENTS, l1=l1, lr=LR_SAE).to(device)
    ckpt_path = os.path.join(OUT_DIR, f"sae_layer{layer}_l1{l1}_{tag}.ckpt")

    trainer = pl.Trainer(max_epochs=EPOCHS_SAE, accelerator="auto", devices=1,
                         log_every_n_steps=10, enable_checkpointing=True, default_root_dir=OUT_DIR)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), ckpt_path)

    return ckpt_path, model, val


# ============================================================================
# Evaluation Function
# ============================================================================

def eval_sae(model, X, name="eval"):
    """Evaluate SAE reconstruction quality and sparsity."""
    model.eval()
    with torch.no_grad():
        x_hat, z = model(X.to(device))
        recon = F.mse_loss(x_hat, X.to(device)).item()
        active_frac = (z > 1e-9).float().mean().item()
    print(f"{name}: recon={recon:.4e} | active_frac={active_frac:.3f}")
    return recon, active_frac


# ============================================================================
# Model Initialization Function
# ============================================================================

def initialize_models():
    """Initialize TRACR and TransformerLens models."""
    global model, tl_model, bos, INPUT_ENCODER, OUTPUT_ENCODER, n_layers
    
    # Compile RASP program
    length = make_length()
    opp_index = length - rasp.indices - 1
    flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
    prog = rasp.Aggregate(flip, rasp.tokens)
    
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        prog,
        vocab={1, 2, 3},
        max_seq_len=5,
        compiler_bos=bos,
    )
    
    # Quick smoke test
    out = model.apply([bos, 1, 2, 3])
    print(f"[{prog}] Tracr output (decoded):", getattr(out, "decoded", None))
    
    # Map Tracr → TransformerLens
    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"
    
    n_ctx = model.params["pos_embed"]['embeddings'].shape[0]
    d_vocab = model.params["token_embed"]['embeddings'].shape[0]
    d_model = model.params["token_embed"]['embeddings'].shape[1]
    d_vocab_out = d_vocab - 2  # trim BOS + PAD
    
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
    )
    tl_model = HookedTransformer(cfg)
    
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)
    
    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head, n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]
        
        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]
    
    print(sd.keys())
    
    # Convert JAX → Torch tensors
    for k, v in sd.items():
        sd[k] = torch.tensor(np.array(v))
    
    tl_model.load_state_dict(sd, strict=False)
    
    # Set encoders
    INPUT_ENCODER = model.input_encoder
    OUTPUT_ENCODER = model.output_encoder
    
    # Parity test
    input = [bos, 1, 2, 3]
    out = model.apply(input)
    print("Original Decoding:", out.decoded)
    
    input_tokens_tensor = create_model_input(input)
    logits = tl_model(input_tokens_tensor)
    decoded_output = decode_model_output(logits)
    print("TransformerLens Replicated Decoding:", decoded_output)
    
    # Layer-by-layer parity checks
    logits, cache = tl_model.run_with_cache(input_tokens_tensor)
    for layer in range(tl_model.cfg.n_layers):
        print(f"Layer {layer} Attn Out Equality Check:",
              np.isclose(cache["attn_out", layer].detach().cpu().numpy(),
                         np.array(out.layer_outputs[2*layer])).all())
        print(f"Layer {layer} MLP Out Equality Check:",
              np.isclose(cache["mlp_out", layer].detach().cpu().numpy(),
                         np.array(out.layer_outputs[2*layer+1])).all())
    
    # Extended parity test
    parity_exact_match(64)
    
    return model, tl_model


# ============================================================================
# Main Training Loop
# ============================================================================

if __name__ == "__main__":
    # Initialize models
    model, tl_model = initialize_models()
    
    # Train SAEs for all layers
    print(f"\nTraining SAEs for {n_layers} layers on resid_post...")
    all_artifacts = []
    
    for layer in range(n_layers):
        layer_artifacts = []
        for i, lam in enumerate(LAMBDA_L0_LIST):
            ck, sae_model, val = train_sae_for_layer(layer, lam, f"id{i}")
            layer_artifacts.append((lam, ck, sae_model, val))
        all_artifacts.append(layer_artifacts)
    
    # Evaluate trained SAEs
    print("\n=== SAE Evaluation Results ===")
    for layer in range(n_layers):
        print(f"\n--- Layer {layer} ---")
        for lam, ck, sae_model, val in all_artifacts[layer]:
            eval_sae(sae_model, val[:1024], name=f"Layer {layer}, λ={lam:g}")
    
    print("\n✅ Training complete!")
    print(f"Artifacts saved to: {OUT_DIR}")