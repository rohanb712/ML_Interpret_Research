# Google Colab Instructions

## Quick Start

1. **Upload the file**:
   - Go to https://colab.research.google.com/
   - Click `File` → `Upload notebook`
   - Upload `TRACR_TL_SAE_colab.py`

2. **First run (Installation)**:
   - Run **Cell 0** (installs core dependencies)
   - Run **Cell 1** (installs transformer_lens and tracr)
   - Wait for installation to complete

3. **Restart Runtime**:
   - Click `Runtime` → `Restart runtime`
   - This is **CRITICAL** - the new package versions won't work without it

4. **Second run (Actual training)**:
   - After restart, **SKIP Cell 0 and Cell 1**
   - Run from **Cell 2** onwards
   - The script will now train SAEs on the TRACR model

## What to Expect

### Cell 0 Output (First run only)
```
Successfully uninstalled jax-0.7.2 jaxlib-0.7.2
...installing packages...
Python: 3.12.11 (main, Jun  4 2025, 08:56:18) [GCC 11.4.0]
Torch: 2.5.1+cu121 CUDA: True
JAX: 0.4.28 NumPy: 1.26.4
```

### Cell 1 Output (First run only)
```
Running as a Colab notebook
...installing transformer_lens and tracr...
✅ Installation complete!
⚠️ IMPORTANT: You must RESTART THE RUNTIME now for changes to take effect!
```

### Cell 2 Output (After restart)
```
Global seed set to 42
Using device: cuda
```

### Training Progress
- Compiles RASP program (sequence reversal)
- Converts to TransformerLens
- Runs parity checks (should pass)
- Collects activations from all layers
- Trains SAEs with different L1 penalties
- Shows training progress bars
- Prints final evaluation metrics

## Expected Runtime

- **Installation (Cell 0-1)**: ~3-5 minutes
- **Training (Cell 2+)**: ~10-20 minutes on free GPU
  - Depends on GPU type (T4, V100, etc.)
  - Can be faster on Colab Pro

## GPU Availability

### Free Tier
- May need to wait for GPU availability
- Usually gets T4 GPU
- Limited to ~12 hours/session

### Colab Pro
- Better GPU access (A100, V100)
- Longer sessions
- Faster training

## Troubleshooting

### "RESTART THE RUNTIME" Warning
This is normal! You **must** restart after Cell 0 and Cell 1:
1. Click `Runtime` → `Restart runtime`
2. Skip Cell 0 and Cell 1
3. Run from Cell 2

### Import Errors After Restart
If you see import errors:
1. Make sure you restarted the runtime
2. Make sure you're **skipping Cell 0 and Cell 1** after restart
3. If still failing, start over: `Runtime` → `Factory reset runtime`

### Dependency Conflicts
You may see some dependency warnings - these are usually harmless as long as the key versions are correct:
- `torch: 2.5.1+cu121`
- `jax: 0.4.28`
- `numpy: 1.26.4`

### Out of Memory
If you run out of memory:
- In Cell 2, reduce `SAE_LATENTS` from 32 to 16
- In `sample_sequences()`, reduce `n=8000` to `n=4000`

### Session Disconnected
Colab free tier may disconnect after inactivity:
- Keep the tab active
- Consider Colab Pro for longer sessions
- Save checkpoints frequently (already done automatically)

## Output Files

Results are saved in Colab's temporary storage at `/content/sae_artifacts/`:
- `sae_layer{X}_l1{Y}_id{Z}.ckpt` - SAE checkpoints
- PyTorch Lightning logs

**Note**: These files are **temporary**! Download them before your session ends:
```python
# Run this in a new cell to download artifacts
from google.colab import files
import shutil

# Create a zip file
shutil.make_archive('sae_artifacts', 'zip', 'sae_artifacts')
files.download('sae_artifacts.zip')
```

## Modifying Hyperparameters

Edit in Cell 2:
```python
# SAE config
SAE_LATENTS = 32              # Increase for more capacity
LR_SAE, EPOCHS_SAE = 1e-3, 5  # Adjust learning rate/epochs
BATCH_SAE = 64                # Reduce if OOM
LAMBDA_L0_LIST = [1e-2, 5e-3, 1e-3, 5e-4]  # L1 penalties to try
```

## Pro Tips

1. **Enable GPU**: `Runtime` → `Change runtime type` → `T4 GPU`
2. **Monitor GPU**: Click the RAM/Disk indicator in top-right
3. **Save checkpoints**: Artifacts are auto-saved but download them!
4. **Free tier limits**: ~12 hours, then you need to restart
5. **Multiple runs**: After restart, always skip Cell 0 and Cell 1

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify package versions in Cell 0 output
3. Try factory reset: `Runtime` → `Factory reset runtime`
4. Start fresh with Cell 0

## Next Steps

After training:
- Download the SAE checkpoints
- Analyze feature activations
- Visualize learned representations
- Try different RASP programs (edit Cell 5)

