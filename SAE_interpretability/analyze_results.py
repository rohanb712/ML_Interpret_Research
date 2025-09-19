#!/usr/bin/env python3
"""
Analyze the feature analysis results from the completed run.
"""

import json
import pandas as pd
import numpy as np

def analyze_feature_results():
    """Analyze the feature analysis results"""
    print("=== SAE Feature Analysis Results ===")

    # Load the results
    with open("artifacts_mnist_sae/feature_report.json", "r") as f:
        features = json.load(f)

    with open("artifacts_mnist_sae/summary.json", "r") as f:
        summary = json.load(f)[0]

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(features)

    print(f"\n=== SAE Training Summary ===")
    print(f"  Target active features: {summary['target_actives']}")
    print(f"  Achieved active features: {summary['achieved_actives_val']:.2f}")
    print(f"  Baseline accuracy: {summary['baseline_acc']:.4f}")
    print(f"  SAE reconstruction (JumpReLU): {summary['recon_acc_jumprelu']:.4f}")
    print(f"  Fidelity loss: {summary['delta_acc_jumprelu']:.4f}")

    print(f"\n=== Feature Analysis Results ===")
    print(f"  Total features analyzed: {len(df)}")
    print(f"  Mean activation rate: {df['mean_active'].mean():.4f}")
    print(f"  Most active feature: {df['mean_active'].max():.4f}")

    # Top 10 most active features
    top_features = df.nlargest(10, 'mean_active')
    print(f"\n=== Top 10 Most Active Features ===")
    print("Feature | Active% | Favors Class | Boosts Class | Boost Amount")
    print("-" * 65)
    for _, row in top_features.iterrows():
        feat_id = int(row['feature'])
        active_pct = row['mean_active'] * 100
        fav_class = int(row['label_mode_when_active']) if row['label_mode_when_active'] is not None else "?"
        boost_class = int(row['logit_boost_class'])
        boost_amount = row['logit_delta_vector'][boost_class]
        print(f"   {feat_id:3d}  |  {active_pct:5.1f}%  |     {fav_class}      |     {boost_class}      |   {boost_amount:+.3f}")

    # Hypothesis examples for top features
    print(f"\n=== Example Hypotheses for Top Features ===")

    for i, (_, row) in enumerate(top_features.head(3).iterrows()):
        feat_id = int(row['feature'])
        active_pct = row['mean_active'] * 100
        fav_class = int(row['label_mode_when_active']) if row['label_mode_when_active'] is not None else "?"
        boost_class = int(row['logit_boost_class'])
        boost_amount = row['logit_delta_vector'][boost_class]

        print(f"\n-- Feature {feat_id}:")
        print(f"   Evidence: Active {active_pct:.1f}% of time, mainly on class {fav_class}, boosts class {boost_class} by {boost_amount:+.3f}")

        # Generate hypothesis based on patterns
        if fav_class == boost_class:
            confidence = "High"
            hypothesis = f"Detector for digit {fav_class} (consistent activation and boosting)"
        elif active_pct > 30:
            confidence = "Medium"
            hypothesis = f"General feature often active, but specifically helps classify digit {boost_class}"
        else:
            confidence = "Low"
            hypothesis = f"Sparse feature that weakly contributes to digit {boost_class} classification"

        print(f"   Hypothesis: {hypothesis}")
        print(f"   Confidence: {confidence}")

    print(f"\nGenerated Files:")
    print(f"  - artifacts_mnist_sae/feature_report.json: Complete analysis data")
    print(f"  - artifacts_mnist_sae/summary.json: SAE training summary")

    print(f"\nKey Insights:")
    high_fidelity = summary['delta_acc_jumprelu'] < 0.01
    good_sparsity = summary['achieved_actives_val'] < 25

    if high_fidelity:
        print(f"  + High fidelity: Only {summary['delta_acc_jumprelu']:.4f} accuracy loss")
    else:
        print(f"  - Moderate fidelity: {summary['delta_acc_jumprelu']:.4f} accuracy loss")

    if good_sparsity:
        print(f"  + Good sparsity: {summary['achieved_actives_val']:.1f} active features on average")
    else:
        print(f"  - High sparsity: {summary['achieved_actives_val']:.1f} active features (target was {summary['target_actives']})")

    # Check for interpretable patterns
    consistent_features = 0
    for _, row in df.iterrows():
        if row['label_mode_when_active'] is not None and row['label_mode_when_active'] == row['logit_boost_class']:
            consistent_features += 1

    interpretability_score = consistent_features / len(df)
    print(f"  Interpretability: {interpretability_score:.1%} of features show consistent class preference")

    if interpretability_score > 0.3:
        print(f"  + Many features appear interpretable")
    else:
        print(f"  - Features may need closer manual inspection")

if __name__ == "__main__":
    analyze_feature_results()