"""
Run the full SAE interpretability pipeline: train models, then analyze.

This is a convenience script that runs:
1. train_sae.py - Train MLP and SAE models
2. interpret_sae.py - Generate interpretability visualizations

Use this for a complete end-to-end run, or run the scripts individually
if you want to iterate on just the interpretability analysis.
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        print("Please check the output above for error messages.")
        sys.exit(result.returncode)
    
    print(f"\n✓ {description} completed successfully")

def main():
    print("="*70)
    print("SAE INTERPRETABILITY FULL PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Train MLP and SAE models (~5-10 minutes)")
    print("  2. Generate interpretability visualizations (~1-2 minutes)")
    print()
    
    # Check if scripts exist
    if not os.path.exists("train_sae.py"):
        print("❌ ERROR: train_sae.py not found in current directory")
        sys.exit(1)
    
    if not os.path.exists("interpret_sae.py"):
        print("❌ ERROR: interpret_sae.py not found in current directory")
        sys.exit(1)
    
    # Run training
    run_script("train_sae.py", "Step 1: Training Models")
    
    # Run interpretation
    run_script("interpret_sae.py", "Step 2: Interpretability Analysis")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutput files are in: artifacts_sae_mnist/")
    print("\nNext steps:")
    print("  1. Review visualizations in artifacts_sae_mnist/neuron_visualizations/")
    print("  2. Fill out hypothesis_template.md with your interpretations")
    print("  3. Re-run interpret_sae.py anytime to regenerate visualizations")
    print("="*70)

if __name__ == "__main__":
    main()

