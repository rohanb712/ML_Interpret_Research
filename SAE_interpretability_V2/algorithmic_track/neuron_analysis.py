"""
Simple SAE Neuron Analysis and Hypothesis Generation
Analyzes trained SAE neurons to understand their role in the sequence reversal algorithm.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TRACR_TL_SAE import (
    device, SAE_LATENTS, LAMBDA_L0_LIST, OUT_DIR,
    create_model_input, decode_model_output, SAE, initialize_models
)
import TRACR_TL_SAE

# Initialize models on import
print("Initializing TRACR and TransformerLens models...")
TRACR_TL_SAE.model, TRACR_TL_SAE.tl_model = initialize_models()
tl_model = TRACR_TL_SAE.tl_model
bos = TRACR_TL_SAE.bos
print("✓ Models initialized\n")


@torch.no_grad()
def extract_neuron_activations(layer=0, l1_idx=3):
    """Extract activations for all neurons in a specific SAE."""
    # Load the SAE model
    lam = LAMBDA_L0_LIST[l1_idx]
    ckpt_path = os.path.join(OUT_DIR, f"sae_layer{layer}_l1{lam}_id{l1_idx}.ckpt")
    sae = SAE(d_in=tl_model.cfg.d_model, d_latent=SAE_LATENTS, l1=lam, lr=1e-3).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device))
    sae.eval()

    # Generate diverse test sequences
    test_seqs = []
    # All single tokens
    for tok in [1, 2, 3]:
        test_seqs.append([bos, tok])
    # All pairs
    for t1 in [1, 2, 3]:
        for t2 in [1, 2, 3]:
            test_seqs.append([bos, t1, t2])
    # All triples
    for t1 in [1, 2, 3]:
        for t2 in [1, 2, 3]:
            for t3 in [1, 2, 3]:
                test_seqs.append([bos, t1, t2, t3])

    # Get activations
    results = []
    for seq in test_seqs:
        inp = create_model_input(seq)
        _, cache = tl_model.run_with_cache(inp)
        resid = cache[("resid_post", layer)]  # (1, T, d_model)
        _, z = sae(resid.reshape(-1, resid.size(-1)))  # (T, SAE_LATENTS)

        results.append({
            'seq': seq,
            'activations': z.cpu().numpy(),  # (T, SAE_LATENTS)
            'output': decode_model_output(tl_model(inp))
        })

    return results, sae


def analyze_neuron(neuron_idx, activation_data):
    """Analyze activation patterns for a single neuron and generate hypothesis."""
    print(f"\n{'='*60}")
    print(f"Neuron {neuron_idx}")
    print(f"{'='*60}")

    # Collect activation info
    position_patterns = {0: [], 1: [], 2: [], 3: [], 4: []}  # by position
    token_patterns = {1: [], 2: [], 3: []}  # by token value

    for data in activation_data:
        seq = data['seq']
        acts = data['activations'][:, neuron_idx]  # activations for this neuron at each position

        for pos, act in enumerate(acts):
            if pos < len(seq):
                tok = seq[pos]
                position_patterns[pos].append(act)
                if tok in [1, 2, 3]:
                    token_patterns[tok].append(act)

    # Compute statistics
    print("\nActivation by Position:")
    position_stats = {}
    for pos, acts in position_patterns.items():
        if len(acts) > 0:
            mean_act = np.mean(acts)
            max_act = np.max(acts)
            active_frac = np.mean(np.array(acts) > 0.01)
            position_stats[pos] = {'mean': mean_act, 'max': max_act, 'active_frac': active_frac}
            print(f"  Pos {pos}: mean={mean_act:.3f}, max={max_act:.3f}, active={active_frac:.2%}")

    print("\nActivation by Token:")
    token_stats = {}
    for tok, acts in token_patterns.items():
        if len(acts) > 0:
            mean_act = np.mean(acts)
            max_act = np.max(acts)
            active_frac = np.mean(np.array(acts) > 0.01)
            token_stats[tok] = {'mean': mean_act, 'max': max_act, 'active_frac': active_frac}
            print(f"  Token {tok}: mean={mean_act:.3f}, max={max_act:.3f}, active={active_frac:.2%}")

    # Generate hypothesis
    hypothesis = generate_hypothesis(neuron_idx, position_stats, token_stats)
    print(f"\n💡 Hypothesis: {hypothesis}")

    return {
        'neuron_idx': neuron_idx,
        'position_stats': position_stats,
        'token_stats': token_stats,
        'hypothesis': hypothesis
    }


def generate_hypothesis(neuron_idx, position_stats, token_stats):
    """Generate simple hypothesis based on activation patterns."""

    # Check for position-specific activation
    if position_stats:
        pos_means = {p: s['mean'] for p, s in position_stats.items()}
        max_pos = max(pos_means.items(), key=lambda x: x[1])
        if max_pos[1] > 0.1:  # significant activation
            # Check if it's strongly position-specific
            other_means = [v for p, v in pos_means.items() if p != max_pos[0]]
            if len(other_means) > 0 and max_pos[1] > 2 * np.mean(other_means):
                if max_pos[0] == 0:
                    return "Detects BOS token position"
                else:
                    return f"Activates at position {max_pos[0]} (0-indexed)"

    # Check for token-specific activation
    if token_stats:
        tok_means = {t: s['mean'] for t, s in token_stats.items()}
        max_tok = max(tok_means.items(), key=lambda x: x[1])
        if max_tok[1] > 0.1:
            other_means = [v for t, v in tok_means.items() if t != max_tok[0]]
            if len(other_means) > 0 and max_tok[1] > 2 * np.mean(other_means):
                return f"Detects token value {max_tok[0]}"

    # Check overall activation level
    all_means = [s['mean'] for s in position_stats.values()]
    if len(all_means) > 0:
        overall_mean = np.mean(all_means)
        if overall_mean < 0.01:
            return "Mostly inactive (dead neuron)"
        elif overall_mean > 0.5:
            return "Broadly active across positions/tokens"

    return "No clear pattern detected"


def run_neuron_analysis(layer=0, l1_idx=3):
    """Run complete neuron analysis for a specific SAE."""
    print(f"\n{'#'*60}")
    print(f"# Analyzing SAE: Layer {layer}, λ={LAMBDA_L0_LIST[l1_idx]}")
    print(f"{'#'*60}")

    # Extract activations
    activation_data, sae = extract_neuron_activations(layer, l1_idx)

    # Analyze each neuron
    all_analyses = []
    for neuron_idx in range(SAE_LATENTS):
        analysis = analyze_neuron(neuron_idx, activation_data)
        all_analyses.append(analysis)

    # Save results
    output_file = os.path.join(OUT_DIR, f"neuron_analysis_layer{layer}_l1{LAMBDA_L0_LIST[l1_idx]}.json")
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable = []
        for a in all_analyses:
            serializable.append({
                'neuron_idx': a['neuron_idx'],
                'hypothesis': a['hypothesis'],
                'position_stats': {int(k): {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in a['position_stats'].items()},
                'token_stats': {int(k): {kk: float(vv) for kk, vv in v.items()}
                               for k, v in a['token_stats'].items()}
            })
        json.dump(serializable, f, indent=2)

    print(f"\n✅ Analysis saved to: {output_file}")
    return all_analyses


if __name__ == "__main__":
    # Analyze layer 0 with the lowest L1 penalty (most active neurons)
    run_neuron_analysis(layer=0, l1_idx=3)
