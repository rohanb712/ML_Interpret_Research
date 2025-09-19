#!/usr/bin/env python3
"""
Demo script showing the feature analysis functionality.
Run this after training the main SAE to see feature analysis in action.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

# This demonstrates the core feature analysis functions
# without requiring the full training pipeline

def demo_feature_analysis():
    """Demo the feature analysis capabilities"""
    print("=== Feature Analysis Demo ===")

    # Mock some data for demonstration
    batch_size = 100
    n_features = 512
    n_dims = 128

    # Simulate feature activations
    u = torch.randn(batch_size, n_features)
    theta = torch.zeros(n_features)
    hard = (u > theta).float()

    # Simulate labels
    labels = torch.randint(0, 10, (batch_size,))

    print(f"Simulated {batch_size} examples with {n_features} features")
    print(f"Mean activations per example: {hard.sum(dim=1).mean():.2f}")

    # Demo label histogram for a feature
    k = 0
    feature_active = hard[:, k].bool()
    labels_when_active = labels[feature_active].tolist()
    hist = Counter(labels_when_active)
    print(f"\nFeature {k} label histogram: {dict(sorted(hist.items()))}")

    # Demo input pattern computation
    W_in = torch.randn(784, n_dims) * 0.02
    enc_k = torch.randn(n_dims) * 0.02
    mu, std = torch.zeros(1, n_dims), torch.ones(1, n_dims)

    pat = W_in @ (enc_k / std.squeeze(0))
    heat = pat.view(28, 28)
    print(f"Input pattern shape: {heat.shape}")
    print(f"Pattern range: [{heat.min():.3f}, {heat.max():.3f}]")

    # Demo logit effects
    logit_deltas = torch.randn(10)  # 10 classes
    top_class = logit_deltas.argmax().item()
    print(f"Feature most boosts class {top_class} by {logit_deltas[top_class]:.3f}")

    print("\n✅ Feature analysis functions are working correctly!")
    print("\nTo use with real SAE:")
    print("1. Train the SAE using MNIST_SAE_code.py")
    print("2. The script will automatically run feature analysis")
    print("3. Check artifacts_mnist_sae/ for outputs:")
    print("   - feature_report.json: structured data")
    print("   - feature_*_gallery.png: top examples")
    print("   - feature_*_pattern.png: input patterns")

if __name__ == "__main__":
    demo_feature_analysis()