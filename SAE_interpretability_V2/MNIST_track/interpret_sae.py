"""
Load trained MLP and SAE models and generate interpretability visualizations.

This script loads previously trained models and creates:
- Activation database for all MNIST images
- Statistics for each SAE neuron
- Visualization grids showing top activating images
- Hypothesis template for manual annotation

Run this after train_sae.py to iterate quickly on analysis code.
"""

import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# Configuration
CHECKPOINT_PATH = "artifacts_sae_mnist/trained_models.pt"
OUT_DIR = "artifacts_sae_mnist"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================================
# Load Checkpoint
# ============================================================================

print("\n" + "="*70)
print("Loading Trained Models")
print("="*70)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Please run train_sae.py first to train and save models.")
    exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
config = checkpoint['config']

print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
print(f"Config: {json.dumps(config, indent=2)}")

# ============================================================================
# Model Definitions (must match training)
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
        self.ffn = FFN_ReLU(d_i, d_h, d_o)
    def forward(self, x_bchw):
        x_bi = x_bchw.view(x_bchw.size(0), -1)
        y_bo = self.ffn(x_bi)
        return y_bo

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

# ============================================================================
# Load Models
# ============================================================================

# Instantiate models
model = MNIST_FFN(
    ffn_type=config['ffn_type'],
    d_h=config['hidden_dim']
)
model.load_state_dict(checkpoint['mlp_state_dict'])
model.eval().to(device)

sae = SAE_JumpReLU(
    d_in=10,  # logits dimension
    d_latents=config['sae_latents']
)
sae.load_state_dict(checkpoint['sae_state_dict'])
sae.eval().to(device)

# Load normalization params
mu_bo = checkpoint['mu_bo'].to(device)
std_bo = checkpoint['std_bo'].to(device)

print(f"MLP: {config['hidden_dim']}-hidden ReLU")
print(f"SAE: {config['sae_latents']} latents, {config['achieved_actives']:.2f} active on avg")

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "="*70)
print("Loading MNIST Data")
print("="*70)

def load_mnist(batch_size=256):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_train_full = datasets.MNIST(root=".", train=True,  download=True, transform=tfm)
    ds_test       = datasets.MNIST(root=".", train=False, download=True, transform=tfm)

    train_len  = int(0.8 * len(ds_train_full))
    val_len    = len(ds_train_full) - train_len
    ds_train, ds_val = random_split(
        ds_train_full, [train_len, val_len],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    pin = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return dl_train, dl_val, dl_test

dl_train, dl_val, dl_test = load_mnist()
print("Data loaded successfully")

# ============================================================================
# Collect Activation Database
# ============================================================================

print("\n" + "="*70)
print("Collecting SAE Activation Database")
print("="*70)

@torch.no_grad()
def collect_sae_activations(model, sae, loader, mu_bo, std_bo, split_name="train"):
    """
    Collect SAE latent activations for all images in a dataloader.
    
    Returns:
        activations_db: list of dicts with keys:
            - 'image_idx': global index within the split
            - 'activations': (SAE_LATENTS,) numpy array of sparse latent values
            - 'label': ground truth digit
            - 'split': 'train', 'val', or 'test'
    
    Note: Images are NOT stored to save memory. Reload from MNIST when needed.
    """
    model.eval().to(device)
    sae.eval().to(device)

    activations_db = []
    global_idx = 0

    for x_bchw, y_gt in loader:
        x_bchw = x_bchw.to(device)
        y_gt = y_gt.to(device)

        # Get logits from base model
        x_bi = x_bchw.view(x_bchw.size(0), -1)
        y_raw_bO = model.ffn(x_bi)

        # Normalize logits
        y_n_bO = (y_raw_bO - mu_bo) / std_bo

        # Pass through SAE and get sparse latents (f_bh)
        y_hat_n_bO, f_bh, u_bh, soft_bh, hard_bh = sae(y_n_bO)

        # Store each image's data (convert to numpy to save memory)
        for i in range(x_bchw.size(0)):
            activations_db.append({
                'image_idx': global_idx,
                'activations': f_bh[i].cpu().numpy(),  # sparse latent activations as numpy
                'label': y_gt[i].cpu().item(),
                'split': split_name,
            })
            global_idx += 1

    return activations_db

# Collect activations for all splits
print("Collecting activations (this may take a minute)...")
db_train = collect_sae_activations(model, sae, dl_train, mu_bo, std_bo, split_name="train")
db_val   = collect_sae_activations(model, sae, dl_val,   mu_bo, std_bo, split_name="val")
db_test  = collect_sae_activations(model, sae, dl_test,  mu_bo, std_bo, split_name="test")

# Combine all splits
activation_database = db_train + db_val + db_test

print(f"Collected activations for {len(activation_database)} images")
print(f"  Train: {len(db_train)} images")
print(f"  Val:   {len(db_val)} images")
print(f"  Test:  {len(db_test)} images")

# Save to disk
db_path = os.path.join(OUT_DIR, "activation_database.pkl")
with open(db_path, "wb") as f:
    pickle.dump({
        'database': activation_database,
        'num_latents': config['sae_latents'],
        'config': config,
    }, f)

print(f"Saved activation database to: {db_path}")

# ============================================================================
# Compute Neuron Statistics
# ============================================================================

print("\n" + "="*70)
print("Computing Neuron Statistics")
print("="*70)

def compute_neuron_statistics(database, neuron_id, activation_threshold=0.0):
    """Compute statistics for a specific neuron."""
    activations = []
    firing_count = 0
    label_counts = {i: 0 for i in range(10)}

    for entry in database:
        act_val = entry['activations'][neuron_id]
        if isinstance(act_val, np.ndarray):
            act_val = float(act_val)
        
        activations.append(act_val)
        if act_val > activation_threshold:
            firing_count += 1
            label_counts[entry['label']] += 1

    activations = np.array(activations)

    return {
        'neuron_id': neuron_id,
        'mean_activation': float(activations.mean()),
        'max_activation': float(activations.max()),
        'firing_rate': firing_count / len(database) if len(database) > 0 else 0.0,
        'label_distribution': label_counts,
        'num_images': len(database),
    }

def analyze_all_neurons(database, num_latents):
    """Compute statistics for all neurons in the SAE."""
    stats = []
    for neuron_id in range(num_latents):
        neuron_stats = compute_neuron_statistics(database, neuron_id)
        stats.append(neuron_stats)
    return stats

neuron_stats = analyze_all_neurons(activation_database, num_latents=config['sae_latents'])

# Save stats
stats_path = os.path.join(OUT_DIR, "neuron_statistics.pkl")
with open(stats_path, "wb") as f:
    pickle.dump(neuron_stats, f)

print(f"Saved neuron statistics to: {stats_path}")

# Print summary
print("\n=== Neuron Statistics Summary ===")
print(f"{'Neuron':<8} {'Mean Act':<12} {'Max Act':<12} {'Fire Rate':<12} {'Top Label (Count)'}")
print("-" * 70)
for stats in neuron_stats[:10]:  # Show first 10 neurons
    nid = stats['neuron_id']
    mean_act = stats['mean_activation']
    max_act = stats['max_activation']
    fire_rate = stats['firing_rate']
    top_label = max(stats['label_distribution'].items(), key=lambda x: x[1])
    print(f"{nid:<8} {mean_act:<12.4f} {max_act:<12.4f} {fire_rate:<12.2%} {top_label[0]} ({top_label[1]})")

print(f"\n... (showing first 10 of {config['sae_latents']} neurons)")

# ============================================================================
# Generate Visualizations
# ============================================================================

print("\n" + "="*70)
print("Generating Visualizations")
print("="*70)

# Load full MNIST datasets for image retrieval
def load_full_mnist_datasets():
    """Load complete MNIST datasets for image retrieval by index."""
    tfm = transforms.Compose([transforms.ToTensor()])
    ds_train_full = datasets.MNIST(root=".", train=True, download=False, transform=tfm)
    ds_test = datasets.MNIST(root=".", train=False, download=False, transform=tfm)
    
    # Recreate the train/val split
    train_len = int(0.8 * len(ds_train_full))
    val_len = len(ds_train_full) - train_len
    ds_train, ds_val = random_split(
        ds_train_full, [train_len, val_len],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    return ds_train, ds_val, ds_test

mnist_train, mnist_val, mnist_test = load_full_mnist_datasets()
mnist_splits = {'train': mnist_train, 'val': mnist_val, 'test': mnist_test}

def get_image_by_index(split_name, idx):
    """Retrieve an MNIST image by split and index."""
    dataset = mnist_splits[split_name]
    image, label = dataset[idx]
    return image[0]  # Remove channel dimension

def get_top_k_activations(database, neuron_id, k=20):
    """Get the top-k images that most strongly activate a specific neuron."""
    results = []
    
    # Track indices within each split
    split_counters = {'train': 0, 'val': 0, 'test': 0}
    
    for entry in database:
        activation_val = entry['activations'][neuron_id]
        if isinstance(activation_val, np.ndarray):
            activation_val = float(activation_val)
        
        results.append({
            'image_idx': entry['image_idx'],
            'split_idx': split_counters[entry['split']],  # Index within split
            'activation_value': activation_val,
            'label': entry['label'],
            'split': entry['split'],
        })
        split_counters[entry['split']] += 1

    results.sort(key=lambda x: x['activation_value'], reverse=True)
    return results[:k]

def visualize_top_k_for_neuron(database, neuron_id, k=20, save_path=None):
    """Create a visualization grid showing the top-k images that activate a neuron."""
    top_images = get_top_k_activations(database, neuron_id, k=k)
    stats = compute_neuron_statistics(database, neuron_id)

    # Create figure
    n_cols = 5
    n_rows = (k + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(12, 2.5 * n_rows))

    # Add title with neuron statistics
    top_label = max(stats['label_distribution'].items(), key=lambda x: x[1])
    fig.suptitle(
        f"Neuron {neuron_id} | Mean Act: {stats['mean_activation']:.3f} | "
        f"Fire Rate: {stats['firing_rate']:.1%} | Top Label: {top_label[0]} ({top_label[1]} imgs)",
        fontsize=14, fontweight='bold'
    )

    # Plot each top image (load on-demand)
    for idx, img_data in enumerate(top_images):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Load image from MNIST by split and index
        image = get_image_by_index(img_data['split'], img_data['split_idx'])
        
        ax.imshow(image, cmap='gray')
        ax.set_title(
            f"#{idx+1}: {img_data['label']}\nAct={img_data['activation_value']:.3f}",
            fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

def visualize_all_neurons(database, num_latents, k=20, output_dir=None):
    """Create visualization grids for all neurons and save them."""
    if output_dir is None:
        output_dir = os.path.join(OUT_DIR, "neuron_visualizations")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations for {num_latents} neurons...")
    for neuron_id in range(num_latents):
        save_path = os.path.join(output_dir, f"neuron_{neuron_id:02d}_top{k}.png")
        fig = visualize_top_k_for_neuron(database, neuron_id, k=k, save_path=save_path)
        plt.close(fig)  # Close to avoid memory issues

        if (neuron_id + 1) % 5 == 0:
            print(f"  Completed {neuron_id + 1}/{num_latents} neurons")

    print(f"All visualizations saved to: {output_dir}")
    return output_dir

viz_dir = visualize_all_neurons(activation_database, num_latents=config['sae_latents'], k=20)

# ============================================================================
# Create Hypothesis Template
# ============================================================================

print("\n" + "="*70)
print("Creating Hypothesis Template")
print("="*70)

def create_hypothesis_template(neuron_stats, output_path=None):
    """Create a template file for manually recording hypotheses about each neuron."""
    if output_path is None:
        output_path = os.path.join(OUT_DIR, "hypothesis_template.md")

    with open(output_path, "w") as f:
        f.write("# SAE Neuron Interpretation Hypotheses\n\n")
        f.write(f"Total Neurons: {len(neuron_stats)}\n\n")
        f.write("---\n\n")

        for stats in neuron_stats:
            nid = stats['neuron_id']
            mean_act = stats['mean_activation']
            max_act = stats['max_activation']
            fire_rate = stats['firing_rate']
            top_label = max(stats['label_distribution'].items(), key=lambda x: x[1])

            f.write(f"## Neuron {nid}\n\n")
            f.write(f"**Statistics:**\n")
            f.write(f"- Mean Activation: {mean_act:.4f}\n")
            f.write(f"- Max Activation: {max_act:.4f}\n")
            f.write(f"- Firing Rate: {fire_rate:.2%}\n")
            f.write(f"- Most Common Label: {top_label[0]} ({top_label[1]} images)\n")
            f.write(f"- Label Distribution: {dict(stats['label_distribution'])}\n\n")

            f.write(f"**Hypothesis:**\n")
            f.write(f"- [ ] Monosemantic (represents one concept)\n")
            f.write(f"- [ ] Polysemantic (represents multiple concepts)\n")
            f.write(f"- [ ] Dead neuron (rarely/never activates)\n\n")

            f.write(f"**Interpretation:**\n")
            f.write(f"> _TODO: Describe what this neuron represents based on top activating images_\n\n")

            f.write(f"**Visual Features:**\n")
            f.write(f"- Primary digit(s): \n")
            f.write(f"- Key features detected: \n")
            f.write(f"- Notes: \n\n")

            f.write("---\n\n")

    return output_path

template_path = create_hypothesis_template(neuron_stats)
print(f"Created hypothesis template: {template_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("INTERPRETABILITY ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print(f"  1. {db_path}")
print(f"  2. {stats_path}")
print(f"  3. {viz_dir}/ (directory with {config['sae_latents']} images)")
print(f"  4. {template_path}")
print("\nNext steps for manual interpretation:")
print("  1. Review visualizations in neuron_visualizations/")
print("  2. Fill out hypothesis_template.md with your observations")
print("  3. Identify monosemantic vs polysemantic neurons")
print("  4. Document visual features each neuron detects")
print("="*70)

