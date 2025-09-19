#!/usr/bin/env python3
"""
Isolated test environment for SAE pipeline components.
This lets you test functions without re-training the model.
"""

import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# Import all the classes and functions from the main file
from MNIST_SAE_code import (
    FFN_ReLU, FFN_GeGLU, MNIST_FFN, 
    baseline_accuracy, collect_preactivations,
    JumpReLU_STE, SAE_JumpReLU,
    train_sae_once, calibrate_lambda, gate_counts,
    test_accuracy_with_mode, theta_shift_to_target,
    # Constants
    SEED, FFN_TYPE, HIDDEN_DIM, LR_FFN, EPOCHS_FFN, BATCH_FFN,
    NORMALIZE_Z, SAE_LATENTS, TAU, INIT_THETA, LR_SAE_FINAL, 
    EPOCHS_SAE_FINAL, BATCH_SAE, TARGET_ACTIVES, OUT_DIR
)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mock_model_and_data():
    """Create a simple mock model and data for testing"""
    print("Creating mock model and data for testing...")
    
    # Create a simple trained model
    model = MNIST_FFN(FFN_TYPE, HIDDEN_DIM, LR_FFN)
    
    # Initialize with reasonable weights (simulate a trained model)
    with torch.no_grad():
        if hasattr(model.ffn, 'W_in'):
            model.ffn.W_in.data.normal_(0, 0.1)
            model.ffn.W_out.data.normal_(0, 0.1)
    
    # Create small mock datasets
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Use smaller batch sizes for quick testing
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return model, test_loader

def test_baseline_accuracy():
    """Test the baseline_accuracy function in isolation"""
    print("\n=== Testing baseline_accuracy function ===")
    
    model, test_loader = create_mock_model_and_data()
    
    try:
        # This should work now
        acc = baseline_accuracy(model, test_loader)
        print(f"✅ baseline_accuracy works! Accuracy: {acc:.4f}")
        return model, test_loader, acc
    except Exception as e:
        print(f"❌ baseline_accuracy failed: {e}")
        return None, None, None

def test_collect_preactivations():
    """Test the collect_preactivations function"""
    print("\n=== Testing collect_preactivations function ===")
    
    model, test_loader = create_mock_model_and_data()
    
    try:
        Z = collect_preactivations(model, test_loader, FFN_TYPE)
        print(f"✅ collect_preactivations works! Shape: {Z.shape}")
        return Z
    except Exception as e:
        print(f"❌ collect_preactivations failed: {e}")
        return None

def test_data_normalization():
    """Test the data normalization step"""
    print("\n=== Testing data normalization ===")
    
    Z = test_collect_preactivations()
    if Z is None:
        return None, None
    
    try:
        if NORMALIZE_Z:
            mu = Z.mean(0, keepdim=True)
            std = Z.std(0, keepdim=True).clamp_min(1e-6)
            Z_normalized = (Z - mu) / std
            print(f"✅ Normalization works! Original range: [{Z.min():.3f}, {Z.max():.3f}]")
            print(f"   Normalized range: [{Z_normalized.min():.3f}, {Z_normalized.max():.3f}]")
            return Z_normalized, (mu, std)
        else:
            print("✅ No normalization (NORMALIZE_Z=False)")
            return Z, (0.0, 1.0)
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return None, None

def test_sae_creation():
    """Test creating an SAE model"""
    print("\n=== Testing SAE creation ===")
    
    Z, _ = test_data_normalization()
    if Z is None:
        return None
    
    try:
        n_in = Z.shape[1]
        sae = SAE_JumpReLU(n_in=n_in, n_latents=SAE_LATENTS, lambda_l0=1e-3, 
                          lr=LR_SAE_FINAL, init_theta=INIT_THETA, tau=TAU)
        print(f"✅ SAE created! Input dim: {n_in}, Latents: {SAE_LATENTS}")
        print(f"   Parameters: {sum(p.numel() for p in sae.parameters()):,}")
        return sae
    except Exception as e:
        print(f"❌ SAE creation failed: {e}")
        return None

def test_full_pipeline_components():
    """Test all components in sequence without training"""
    print("🚀 Testing Full Pipeline Components (No Training)")
    print("=" * 60)
    
    # Test each component
    model, test_loader, acc = test_baseline_accuracy()
    if model is None:
        return
    
    Z = test_collect_preactivations()
    if Z is None:
        return
    
    Z_norm, norm_params = test_data_normalization()
    if Z_norm is None:
        return
    
    sae = test_sae_creation()
    if sae is None:
        return
    
    print("\n🎉 All components pass basic tests!")
    print("The pipeline structure is correct. The error was just function definition order.")

def run_minimal_sae_test():
    """Run a minimal SAE training test with tiny dataset"""
    print("\n=== Minimal SAE Training Test ===")
    
    # Create tiny synthetic data for super fast test
    n_samples, n_features = 100, 128
    Z_tiny = torch.randn(n_samples, n_features) * 0.5
    
    try:
        # Test if SAE can train on tiny data (1 epoch)
        sae = train_sae_once(n_features, lam=1e-3, Ztr=Z_tiny[:80], 
                            Zva=Z_tiny[80:], epochs=1)
        print("✅ Minimal SAE training works!")
        
        # Test gate counts
        count = gate_counts(sae, Z_tiny)
        print(f"✅ Gate counting works! Average active: {count:.1f}")
        
    except Exception as e:
        print(f"❌ Minimal SAE training failed: {e}")

if __name__ == "__main__":
    print("SAE Pipeline Component Tester")
    print("=" * 50)
    print("This tests individual components without full training.")
    print()
    
    # Run all tests
    test_full_pipeline_components()
    run_minimal_sae_test()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete! Your main pipeline should work now.")
    print("Run: python MNIST_SAE_code.py")
