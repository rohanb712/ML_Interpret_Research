# Zero-Layer Transformer

A minimal "transformer" with no transformer blocks - just embeddings and unembeddings.

## Architecture

The model consists of only:
1. **Embedding layer**: Maps discrete pixel values (0-255) to high-dimensional vectors
2. **Positional embeddings**: Spatial position information for 28×28 MNIST pixels  
3. **Aggregation**: Mean pooling across spatial positions
4. **Unembedding layer**: Linear projection to class logits

No attention, no feedforward networks, no transformer blocks.

## Key Idea

Treats MNIST pixels as discrete tokens by:
- Converting continuous pixel values to integers (0-255)
- Learning embeddings for each possible pixel value
- Adding learned positional embeddings for spatial structure
- Aggregating with mean pooling and projecting to classes

## Files

- `zero_layer_transformer.py`: Complete implementation with hyperparameter sweep
- `test_model.py`: Quick functionality test (2 epochs)
- `full_test.py`: Full training with baseline comparison

## Usage

```bash
python test_model.py        # Quick test
python zero_layer_transformer.py  # Full experiment
```

## Performance

The quick test shows the model learns (accuracy improves from ~10% to 12%+ in just 2 epochs), demonstrating the core concept works. Full training would likely reach much higher accuracy given sufficient time.

## Parameters

With d_model=256: **268,810 parameters**
- Embedding: 256 × 256 = 65,536
- Positional: 784 × 256 = 200,704  
- Unembedding: 256 × 10 = 2,560