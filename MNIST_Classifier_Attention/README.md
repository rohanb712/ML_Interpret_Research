# MNIST Attention Transformer

A pure attention-based transformer for MNIST digit classification using vision transformer principles.

## Usage

```bash
python train_mnist_attn_only.py
```

## Model Overview

1. **Patches**: Converts 28×28 MNIST images into 49 non-overlapping 4×4 patches
2. **Embedding**: Projects each 16-pixel patch to 192-dimensional vectors  
3. **Self-Attention**: Single multi-head attention layer with 4 heads processes all patches globally
4. **Classification**: CLS token aggregates information for final digit prediction

## Key Features

- **Patch-based processing**: Treats images as sequences of spatial patches
- **Global context**: Every patch can attend to every other patch in one step
- **Learnable positional encoding**: Model learns spatial relationships between patches
- **CLS token classification**: Dedicated token accumulates global image information
- **Single attention layer**: Demonstrates core attention mechanism without complexity

## Architecture

```
Image [28×28] 
  ↓ patch into 4×4 pieces
Patches [49×16]
  ↓ linear embedding  
Embeddings [50×192]  # 49 patches + 1 CLS token
  ↓ self-attention (4 heads)
Attended [50×192]
  ↓ classify from CLS token
Logits [10]
```

## Expected accuracy

~85-90% in 5 epochs with ~160K parameters.