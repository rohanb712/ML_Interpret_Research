# SAE Feature Analysis and Hypothesis Generation

This document describes the integrated feature analysis system for understanding SAE (Sparse Autoencoder) learned representations.

## Overview

The system provides a rigorous, step-by-step framework for analyzing SAE features and generating hypotheses about what they encode. It follows the approach suggested in the ChatGPT framework with iterative testing and verification.

## Core Components

### 1. Feature Bank Collection (`collect_feature_bank`)
- **Purpose**: For each SAE feature, collect top-activating examples and statistics
- **Outputs**:
  - Top-k examples that maximally activate each feature
  - Label histograms showing which classes activate each feature
  - Mean activation rates across the dataset

### 2. Input Pattern Analysis (`feature_input_pattern`)
- **Purpose**: Map SAE features back to input space to understand what pixel patterns activate them
- **Method**: Computes gradient of feature activation w.r.t. input pixels via the chain rule
- **Output**: 28×28 heatmap showing which pixels increase/decrease feature activation

### 3. Causal Effect Measurement (`feature_logit_effect`)
- **Purpose**: Measure how artificially activating a feature affects model predictions
- **Method**: Perturb the latent space by adding the feature's decoder direction
- **Output**: Change in logits for each class when feature is activated

### 4. Visualization Tools
- `show_feature_gallery`: Creates visual galleries of top-activating examples
- `visualize_feature_pattern`: Visualizes input-space activation patterns

## Integration with Main Pipeline

The feature analysis is automatically executed after SAE training completes. The workflow:

1. **Train base classifier** (MNIST FFN)
2. **Collect preactivations** from hidden layer
3. **Train SAE** with lambda calibration for target sparsity
4. **Adjust thresholds** to hit exact target (e.g., ~20 active features)
5. **Run feature analysis**:
   - Collect feature bank from test set
   - Generate logit effect measurements
   - Create visualizations and reports
   - Save structured data for hypothesis generation

## Outputs

The system generates several files in `artifacts_mnist_sae/`:

### Structured Data
- `feature_report.json`: Complete feature analysis with statistics for each feature
- `summary.json`: SAE training and fidelity results

### Visualizations
- `feature_{k}_gallery.png`: Top-activating examples for feature k
- `feature_{k}_pattern.png`: Input-space activation pattern for feature k
- `feature_{k}_pattern.npy`: Raw numpy array of activation pattern

## Manual Hypothesis Generation Process

For each interesting feature, examine:

1. **Top-activating examples** (gallery): Look for visual patterns
2. **Label histogram**: Which classes most activate this feature?
3. **Input-space pattern**: What pixel locations matter most?
4. **Logit effects**: Which class does activating this feature boost?

### Example Analysis

```
Feature 42:
  Mean active: 0.0823
  Label histogram: {1: 45, 7: 23, 4: 12, 9: 8, ...}
  Boosts class 1 by 0.347
  Top activating examples: [1234, 5678, 9012, ...]
```

**Hypothesis**: "Vertical line detector for digit recognition"
**Evidence**: Mostly fires on 1s and 7s, boosts class 1 predictions, input pattern shows vertical stripe
**Confidence**: High

## Usage

### Running Full Analysis
```bash
python MNIST_SAE_code.py
```

This will train everything and run complete feature analysis.

### Testing Functions Only
```bash
python demo_feature_analysis.py
```

This demonstrates the analysis functions without training.

### Key Parameters

Configure in `MNIST_SAE_code.py`:

```python
TARGET_ACTIVES = [20]        # Target sparsity levels
SAE_LATENTS = 512           # Number of SAE features
EPOCHS_SAE_FINAL = 5        # SAE training epochs
```

## Advanced Usage

### Custom Feature Analysis

After training, you can analyze specific features:

```python
# Load trained models
# model = ... (trained MNIST classifier)
# sae_final = ... (trained SAE)
# mu, std = ... (normalization parameters)

# Analyze feature k=100
k = 100
heat = feature_input_pattern(model, sae_final, mu, std, k)
md, top_cls = feature_logit_effect(model, test_loader, sae_final, mu, std, k)

print(f"Feature {k} boosts class {top_cls} by {md[top_cls]:.3f}")
```

### Batch Analysis

The system automatically analyzes the first 50 features by default. To analyze all features:

```python
# Modify this line in the main script:
for k in range(min(m, 50)):  # Change 50 to m for all features
```

## Technical Details

### Normalization Handling
- Features work in normalized z-space for training
- Input patterns and logit effects properly denormalize
- Maintains consistency between training and analysis

### Device Compatibility
- Automatically detects and uses GPU if available
- Functions handle device placement correctly
- CPU fallback for analysis if needed

### Memory Efficiency
- Processes features in batches to avoid OOM
- Configurable batch sizes for large datasets
- Saves intermediate results to disk

## Next Steps

1. **Review `feature_report.json`** for high-level feature statistics
2. **Examine visualizations** for the most active/interesting features
3. **Write hypotheses** based on evidence from multiple sources
4. **Validate hypotheses** using held-out data or ablation studies
5. **Iterate** on SAE architecture if features aren't interpretable

## Troubleshooting

### Common Issues

**"Feature patterns look noisy"**
- Ensure `NORMALIZE_Z=True`
- Check that same mu/std used for training and analysis

**"Features seem redundant"**
- Try increasing `SAE_LATENTS`
- Adjust lambda slightly and retrain

**"Low fidelity with JumpReLU mode"**
- Expected - boolean mode typically has lower fidelity
- JumpReLU mode should maintain high fidelity (~0.94 vs baseline 0.94)

**"Analysis takes too long"**
- Reduce number of features analyzed
- Use smaller `n_eval` in logit effect measurement
- Reduce `topk` in feature bank collection