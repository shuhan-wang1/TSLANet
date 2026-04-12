"""
Generate a compact, publication-quality uncertainty heatmap for the paper.
Loads a saved model, runs MC Dropout inference on the test set,
and produces a compact 2x2 heatmap (error, aleatoric, epistemic, total).

Usage:
    python generate_paper_heatmap.py --model_dir saved_models/A8_pl96
    python generate_paper_heatmap.py --model_dir saved_models/A8_pl96 --variables 1,4,11,12,15,20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Import model & data utilities from test.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test import (ProbabilisticTSLANet, data_provider,
                  mc_dropout_predict, gaussian_predict, deterministic_predict)


# Weather dataset variable names (21 variables, indices 0-20)
WEATHER_VARS = [
    'p (mbar)', 'T (°C)', 'Tpot (K)', 'Tdew (°C)', 'rh (%)',
    'VPmax', 'VPact', 'VPdef', 'sh (g/kg)', 'H₂OC',
    'ρ (g/m³)', 'wv (m/s)', 'max wv', 'wd (°)', 'rain (mm)',
    'raining (s)', 'SWDR', 'PAR', 'max PAR', 'Tlog (°C)', 'OT'
]

# Indices of representative variables spanning diverse physical quantities
DEFAULT_VARS = [1, 4, 11, 12, 13, 15, 16, 20]
#                T  rh  wv  max.wv wd  raining SWDR OT


def load_and_infer(model_dir, mc_samples=50):
    """Load model and run inference, returning results dict."""
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # Load data
    test_data, test_loader = data_provider(args, 'test')

    # Load model
    model = ProbabilisticTSLANet(args)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'model_weights.pt'),
        map_location='cpu'))

    # Run inference
    if args.mc_dropout:
        print(f"Running MC Dropout inference ({mc_samples} samples)...")
        args.mc_samples = mc_samples
        results = mc_dropout_predict(model, test_loader, args.pred_len,
                                     num_samples=mc_samples)
    elif args.probabilistic:
        print("Running Gaussian inference (aleatoric only)...")
        results = gaussian_predict(model, test_loader, args.pred_len)
    else:
        print("Running deterministic inference...")
        results = deterministic_predict(model, test_loader, args.pred_len)

    return results, args


def plot_compact_heatmap(results, sample_idx=0, var_indices=None,
                         save_path='compact_heatmap.pdf'):
    """
    Generate a compact 2x2 uncertainty heatmap for the paper.

    Layout:
        (a) Prediction Error |y - ŷ|    (b) Aleatoric Uncertainty
        (c) Epistemic Uncertainty        (d) Total Uncertainty
    """
    if var_indices is None:
        var_indices = DEFAULT_VARS

    mu = results['mu_mean'][sample_idx].numpy()         # (T, C)
    tgt = results['targets'][sample_idx].numpy()         # (T, C)
    ale = results['aleatoric_var'][sample_idx].numpy()   # (T, C)
    epi = results['epistemic_var'][sample_idx].numpy()   # (T, C)

    # Select variables
    mu = mu[:, var_indices]
    tgt = tgt[:, var_indices]
    ale = ale[:, var_indices]
    epi = epi[:, var_indices]
    total = ale + epi
    error = np.abs(mu - tgt)

    T, M = mu.shape
    var_labels = [WEATHER_VARS[i] for i in var_indices]

    # Colormaps
    cmap_err = LinearSegmentedColormap.from_list(
        'error', ['#E8F5E9', '#FFF9C4', '#FFAB91', '#C62828'], N=256)
    cmap_ale = LinearSegmentedColormap.from_list(
        'aleatoric', ['#FFFFFF', '#BBDEFB', '#1565C0', '#0D47A1'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list(
        'epistemic', ['#FFFFFF', '#FFCCBC', '#E64A19', '#BF360C'], N=256)
    cmap_tot = LinearSegmentedColormap.from_list(
        'total', ['#FFFFFF', '#E1BEE7', '#7B1FA2', '#4A148C'], N=256)

    # --- Figure ---
    # Layout: 2 rows x 2 columns, each with a thin colorbar on the right.
    # Use 6 GridSpec columns: [panel_a, cbar_a, gap, panel_b, cbar_b_spacer, cbar_b]
    fig = plt.figure(figsize=(5.0, 3.6))
    gs = GridSpec(2, 6, width_ratios=[20, 0.8, 2.5, 20, 0.3, 0.8],
                  hspace=0.50, wspace=0.05,
                  left=0.14, right=0.98, top=0.93, bottom=0.10)

    panels = [
        (0, 0, 1, error.T,  cmap_err, '(a) Pred. Error $|y - \\hat{y}|$', 'black',  True),
        (0, 3, 5, ale.T,    cmap_ale, '(b) Aleatoric Uncertainty',         '#1565C0', False),
        (1, 0, 1, epi.T,    cmap_epi, '(c) Epistemic Uncertainty',         '#E64A19', True),
        (1, 3, 5, total.T,  cmap_tot, '(d) Total Uncertainty',             '#7B1FA2', False),
    ]

    for row, col, cb_col, data, cmap, title, color, show_ylabels in panels:
        ax = fig.add_subplot(gs[row, col])
        ax_cb = fig.add_subplot(gs[row, cb_col])

        im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_yticks(np.arange(M))
        if show_ylabels:
            ax.set_yticklabels(var_labels, fontsize=5.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=6.5, fontweight='bold', color=color, pad=3)
        ax.tick_params(axis='x', labelsize=5.5)

        # Only show x-label on bottom row
        if row == 1:
            ax.set_xlabel('Forecast timestep', fontsize=6)

        cb = fig.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=4.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved compact heatmap to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--sample', type=int, default=0,
                        help='Test sample index to visualise')
    parser.add_argument('--variables', type=str, default=None,
                        help='Comma-separated variable indices, e.g. "1,4,11,12,15,20"')
    parser.add_argument('--mc_samples', type=int, default=50)
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path (default: report_writing/images/compact_heatmap.pdf)')
    args = parser.parse_args()

    var_indices = DEFAULT_VARS
    if args.variables:
        var_indices = [int(x.strip()) for x in args.variables.split(',')]

    save_path = args.output
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'report_writing', 'images',
                                 'compact_heatmap.pdf')

    results, config = load_and_infer(args.model_dir, args.mc_samples)
    plot_compact_heatmap(results, sample_idx=args.sample,
                         var_indices=var_indices, save_path=save_path)


if __name__ == '__main__':
    main()
