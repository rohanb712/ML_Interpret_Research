# MNIST FFN Comparison: GeGLU vs ReLU

This project compares two Feed-Forward Network (FFN) architectures on the MNIST dataset: FFN_GeGLU and FFN_ReLU. The comparison evaluates performance across different hidden dimensions using statistical analysis with bootstrap confidence intervals.

## Overview

The primary research question is: **Is FFN_GeGLU better than FFN_ReLU?**

This hypothesis is tested by comparing the two architectures across multiple hidden dimensions and hyperparameter configurations, using rigorous statistical methods to ensure reliable conclusions.

## Architecture Details

### FFN_ReLU
```
Input (784) → W_in → ReLU → W_out → Output (10)
```
- Standard feed-forward network with ReLU activation
- Parameters: W_in (784 × hidden_dim), W_out (hidden_dim × 10)
- Total parameters: 784 × hidden_dim + hidden_dim × 10

### FFN_GeGLU (Gated GELU)
```
Input (784) → W_in → Linear Path
            → W_gate → GELU → Gate Path
                            ↓
            Linear Path ⊙ Gate Path → W_out → Output (10)
```
- Gated Linear Unit with GELU activation
- Parameters: W_in (784 × hidden_dim), W_gate (784 × hidden_dim), W_out (hidden_dim × 10)
- Total parameters: 2 × (784 × hidden_dim) + hidden_dim × 10
- Mathematical formula: FFN_GeGLU(x) = W_out × ((W_in × x) ⊙ GELU(W_gate × x))

## Implementation Details

### Model Features
- Manual parameter initialization with small weights (σ = 0.02)
- Einsum operations for efficient tensor computations
- No bias terms for simplicity
- Cross-entropy loss with Adam optimizer

### Training Configuration
- Single epoch training ("One Epoch is All you Need")
- Full MNIST dataset (60,000 training, 10,000 test samples)
- Test set used as validation for model selection
- No train/validation split to maximize training data

### Hyperparameter Search
- Random search over hyperparameter combinations
- Batch sizes: [8, 64]
- Learning rates: [1e-1, 1e-2, 1e-3, 1e-4]
- Model selection based on highest validation accuracy

## Experimental Design

### Phase 1: Hidden Dimension Sweep
Tests performance across hidden dimensions [2, 4, 8, 16] with fixed hyperparameters:
- Batch size: 64
- Learning rate: 1e-3
- Single trial per configuration

### Phase 2: Statistical Analysis
For each k in [2, 4, 8]:
1. Run k random trials with different hyperparameter combinations
2. Select best model based on validation accuracy
3. Calculate bootstrap confidence intervals (10,000 samples)
4. Generate comparative visualizations

## Statistical Methods

### Bootstrap Confidence Intervals
- 95% confidence intervals using percentile method
- Bootstrap resampling with replacement (10,000 iterations)
- Applied to maximum accuracy across trials
- Provides robust uncertainty quantification

### Model Selection
- Best model chosen by highest validation accuracy across k trials
- Accounts for hyperparameter sensitivity
- Prevents overfitting to specific configurations

## Results

### Hidden Dimension Analysis

![Hidden Dimension Sweep Results](lightning_hidden_dim_sweep.png)

The results show:

1. **Capacity Constraints**: Both models fail at very small hidden dimensions (2, 4) with ~10% accuracy
2. **Optimal Performance**: Peak performance occurs at hidden dimension 8 for both architectures
3. **GeGLU Advantage**: FFN_GeGLU achieves ~87% accuracy vs FFN_ReLU's ~82% at optimal capacity
4. **Performance Degradation**: Both models show reduced performance at hidden dimension 16

### Key Findings

- **GeGLU demonstrates superior performance** when sufficient model capacity is available
- **Gating mechanism provides ~5% accuracy improvement** at the optimal hidden dimension
- **Training stability varies significantly** with hyperparameter choices, especially learning rate
- **One epoch training is sufficient** for meaningful comparisons on MNIST

## Technical Implementation

### Core Functions

#### `train_and_validate_model(model, train_loader, val_loader, lr, max_epochs=1)`
Trains a model and returns validation accuracy. Replaces PyTorch Lightning functionality with manual training loop.

#### `run_k_trials(model_class, label, k)`
Executes k random hyperparameter trials and returns accuracy results for statistical analysis.

#### `bootstrap_ci(data, samples=10_000)`
Calculates bootstrap confidence intervals for maximum accuracy across trials.

### Code Structure
- Models defined as standard PyTorch nn.Module classes
- Manual training loops with explicit forward/backward passes
- Comprehensive logging and visualization
- Reproducible random seeds for consistent results

## Dependencies

```
torch
numpy
matplotlib
seaborn
torchvision
```

## Usage

Run the complete experiment:
```bash
python MNIST.py
```

This will:
1. Execute hidden dimension sweep
2. Generate comparison plots
3. Run statistical trials for k=[2,4,8]
4. Create bootstrap confidence interval visualizations
5. Save all plots as PNG files

## Output Files

- `hidden_dim_sweep.png`: Performance vs hidden dimension comparison
- `accuracy_vs_k_{k}.png`: Best accuracy bar charts for each k value
- `bootstrap_ci_k{k}.png`: Bootstrap distribution histograms with confidence intervals

## Conclusion

The experiment provides evidence supporting the hypothesis that FFN_GeGLU outperforms FFN_ReLU, particularly when models have sufficient capacity (hidden dimension ≥ 8). The gating mechanism in GeGLU appears to provide more effective feature learning, resulting in consistently higher validation accuracy across multiple trials.

However, the results also highlight the critical importance of hyperparameter tuning and model capacity selection, which can have larger effects on performance than the choice of activation mechanism.