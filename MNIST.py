import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import Callable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_lin = lambda x, w: torch.einsum("bd,dh->bh", x, w)

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

class FFN_ReLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, 10) * 0.02)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(_lin(x, self.W_in))
        return torch.einsum("bh,hc->bc", h, self.W_out)

class FFN_GeGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_gate = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, 10) * 0.02)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_proj = _lin(x, self.W_in)
        x_gate = F.gelu(_lin(x, self.W_gate))
        return torch.einsum("bh,hc->bc", x_proj * x_gate, self.W_out)

def train_and_validate_model(model, train_loader, val_loader, lr, max_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_accs = []
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            val_accs.append(acc)
    
    avg_acc = torch.stack(val_accs).mean()
    return avg_acc.item()

print("Phase 1: Hidden Dim Sweep")
hidden_dims = [2, 4, 8, 16]
acc_relu, acc_geglu = [], []

for h in hidden_dims:
    for model_class, accs, name in [(FFN_ReLU, acc_relu, "ReLU"), (FFN_GeGLU, acc_geglu, "GeGLU")]:
        print(f"{name} - Hidden Dim = {h}")
        model = model_class(h)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        acc = train_and_validate_model(model, train_loader, val_loader, lr=1e-3, max_epochs=1)
        accs.append(acc)

plt.figure()
plt.plot(hidden_dims, acc_relu, label="ReLU", marker="o")
plt.plot(hidden_dims, acc_geglu, label="GeGLU", marker="o")
plt.xlabel("Hidden Dimension")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs Hidden Dim")
plt.legend()
plt.grid(True)
plt.savefig("hidden_dim_sweep.png")
plt.show()

def run_k_trials(model_class, label, k):
    print(f"Running {label} Trials (k={k})")
    trials = []
    for i in range(k):
        bs = random.choice([8, 64])
        lr = random.choice([1e-1, 1e-2, 1e-3, 1e-4])
        model = model_class(8)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        acc = train_and_validate_model(model, train_loader, val_loader, lr=lr, max_epochs=1)
        print(f"Trial {i+1}: BS={bs}, LR={lr:.0e}, Acc={acc:.4f}")
        trials.append(acc)
    return trials

def bootstrap_ci(data, samples=10_000):
    sample_matrix = np.random.choice(data, size=(samples, len(data)), replace=True)
    max_per_sample = sample_matrix.max(axis=1)
    ci = np.percentile(max_per_sample, [2.5, 97.5])
    return max_per_sample, ci

for k in [2, 4, 8]:
    relu_trials = run_k_trials(FFN_ReLU, "ReLU", k)
    geglu_trials = run_k_trials(FFN_GeGLU, "GeGLU", k)

    relu_boot, relu_ci = bootstrap_ci(relu_trials)
    geglu_boot, geglu_ci = bootstrap_ci(geglu_trials)

    print(f"ReLU CI (95%): {relu_ci[0]:.4f} - {relu_ci[1]:.4f}")
    print(f"GeGLU CI (95%): {geglu_ci[0]:.4f} - {geglu_ci[1]:.4f}")

    plt.figure()
    plt.bar(["ReLU", "GeGLU"], [max(relu_trials), max(geglu_trials)], color=["skyblue", "orange"])
    plt.ylabel("Best Validation Accuracy")
    plt.title(f"Best Accuracy per Model (k={k})")
    plt.ylim(0.8, 1.0)
    plt.savefig(f"accuracy_vs_k_{k}.png")
    plt.show()

    plt.figure()
    sns.histplot(relu_boot, label="ReLU", color="skyblue", kde=False)
    sns.histplot(geglu_boot, label="GeGLU", color="orange", kde=False)
    plt.axvline(relu_ci[0], linestyle="--", color="blue")
    plt.axvline(relu_ci[1], linestyle="--", color="blue")
    plt.axvline(geglu_ci[0], linestyle="--", color="red")
    plt.axvline(geglu_ci[1], linestyle="--", color="red")
    plt.title(f"Bootstrap CI (k={k})")
    plt.xlabel("Max Validation Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bootstrap_ci_k{k}.png")
    plt.show()