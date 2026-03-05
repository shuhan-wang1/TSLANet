import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from metrics import compute_calibration


# ---------------------------------------------------------------------------
#  1. Prediction Interval Plot
# ---------------------------------------------------------------------------
def plot_prediction_intervals(mu, total_var, target, variable_idx=0, sample_idx=0,
                              save_path='prediction_intervals.pdf'):
    """Plot prediction with shaded 50%/90%/95% prediction intervals."""
    mu_s = mu[sample_idx, :, variable_idx].numpy()
    sigma_s = np.sqrt(total_var[sample_idx, :, variable_idx].numpy())
    target_s = target[sample_idx, :, variable_idx].numpy()
    T = len(mu_s)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    for coverage, alpha, label in [(0.95, 0.15, '95%'), (0.9, 0.25, '90%'), (0.5, 0.4, '50%')]:
        z = float(torch.erfinv(torch.tensor(coverage, dtype=torch.float64)).item()) * math.sqrt(2)
        lower = mu_s - z * sigma_s
        upper = mu_s + z * sigma_s
        ax.fill_between(t, lower, upper, alpha=alpha, color='steelblue', label=f'{label} PI')

    ax.plot(t, mu_s, color='steelblue', linewidth=1.5, label='Predicted mean')
    ax.plot(t, target_s, color='darkorange', linewidth=1.5, linestyle='--', label='Ground truth')

    ax.set_xlabel('Forecast Horizon (timestep)')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Intervals (Variable {variable_idx}, Sample {sample_idx})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction interval plot to {save_path}")


# ---------------------------------------------------------------------------
#  2. Calibration Plot
# ---------------------------------------------------------------------------
def plot_calibration(mu, total_var, target, save_path='calibration.pdf'):
    """Calibration plot: observed coverage vs nominal coverage."""
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal = compute_calibration(mu, total_var, target, quantiles=quantiles)

    nominal = list(cal.keys())
    observed = list(cal.values())

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(nominal, observed, 'o-', color='steelblue', linewidth=2,
            markersize=8, label='Model')
    ax.fill_between(nominal, nominal, observed, alpha=0.15, color='steelblue')

    ax.set_xlabel('Nominal Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration Plot')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {save_path}")


# ---------------------------------------------------------------------------
#  3. Uncertainty Decomposition (Stacked Area)
# ---------------------------------------------------------------------------
def plot_uncertainty_decomposition(epistemic_var, aleatoric_var, variable_idx=0,
                                   save_path='uncertainty_decomposition.pdf'):
    """Stacked area chart of epistemic vs aleatoric variance over forecast horizon."""
    epi = epistemic_var[:, :, variable_idx].mean(dim=0).numpy()
    ale = aleatoric_var[:, :, variable_idx].mean(dim=0).numpy()
    T = len(epi)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(t, 0, ale, alpha=0.6, color='#2196F3', label='Aleatoric')
    ax.fill_between(t, ale, ale + epi, alpha=0.6, color='#FF5722', label='Epistemic')

    ax.set_xlabel('Forecast Horizon (timestep)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Uncertainty Decomposition (Variable {variable_idx})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty decomposition plot to {save_path}")


# ---------------------------------------------------------------------------
#  4. Uncertainty Heatmap
# ---------------------------------------------------------------------------
def plot_uncertainty_heatmap(epistemic_var, aleatoric_var, target, mu,
                             sample_idx=0, max_variables=None,
                             variable_names=None,
                             save_path='uncertainty_heatmap.pdf'):
    """
    4-row heatmap: absolute error, aleatoric, epistemic, total uncertainty.
    X-axis = forecast horizon, Y-axis = variables.
    """
    M = epistemic_var.shape[2]
    if max_variables is not None:
        M = min(M, max_variables)

    epi = epistemic_var[sample_idx, :, :M].numpy()   # (T, M)
    ale = aleatoric_var[sample_idx, :, :M].numpy()    # (T, M)
    total = epi + ale                                  # (T, M)
    mu_s = mu[sample_idx, :, :M].numpy()              # (T, M)
    tgt_s = target[sample_idx, :, :M].numpy()         # (T, M)
    T = epi.shape[0]

    if variable_names is None:
        variable_names = [f'Var {i}' for i in range(M)]
    else:
        variable_names = variable_names[:M]

    # Custom colormaps
    cmap_ale = LinearSegmentedColormap.from_list(
        'aleatoric', ['#FFFFFF', '#BBDEFB', '#1565C0', '#0D47A1'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list(
        'epistemic', ['#FFFFFF', '#FFCCBC', '#E64A19', '#BF360C'], N=256)
    cmap_total = LinearSegmentedColormap.from_list(
        'total', ['#FFFFFF', '#E1BEE7', '#7B1FA2', '#4A148C'], N=256)
    cmap_err = LinearSegmentedColormap.from_list(
        'error', ['#E8F5E9', '#FFF9C4', '#FFAB91', '#C62828'], N=256)

    fig = plt.figure(figsize=(max(14, T * 0.12), 3 + M * 0.55 * 3))
    gs = GridSpec(4, 2, width_ratios=[50, 1], hspace=0.35, wspace=0.05)

    # Row 0: Absolute Prediction Error
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_pred_cb = fig.add_subplot(gs[0, 1])
    abs_error = np.abs(mu_s - tgt_s).T  # (M, T)
    im0 = ax_pred.imshow(abs_error, aspect='auto', cmap=cmap_err, interpolation='nearest')
    ax_pred.set_yticks(np.arange(M))
    ax_pred.set_yticklabels(variable_names, fontsize=8)
    ax_pred.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_pred.set_title('Absolute Prediction Error  |y - $\\hat{y}$|',
                      fontsize=11, fontweight='bold')
    cb0 = fig.colorbar(im0, ax=ax_pred_cb, fraction=0.9, pad=0.0)
    cb0.ax.tick_params(labelsize=7)

    # Row 1: Aleatoric
    ax_ale = fig.add_subplot(gs[1, 0])
    ax_ale_cb = fig.add_subplot(gs[1, 1])
    im1 = ax_ale.imshow(ale.T, aspect='auto', cmap=cmap_ale, interpolation='nearest')
    ax_ale.set_yticks(np.arange(M))
    ax_ale.set_yticklabels(variable_names, fontsize=8)
    ax_ale.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_ale.set_title('Aleatoric Uncertainty (Data Noise)',
                      fontsize=11, fontweight='bold', color='#1565C0')
    cb1 = fig.colorbar(im1, ax=ax_ale_cb, fraction=0.9, pad=0.0)
    cb1.ax.tick_params(labelsize=7)

    # Row 2: Epistemic
    ax_epi = fig.add_subplot(gs[2, 0])
    ax_epi_cb = fig.add_subplot(gs[2, 1])
    im2 = ax_epi.imshow(epi.T, aspect='auto', cmap=cmap_epi, interpolation='nearest')
    ax_epi.set_yticks(np.arange(M))
    ax_epi.set_yticklabels(variable_names, fontsize=8)
    ax_epi.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_epi.set_title('Epistemic Uncertainty (Model Uncertainty)',
                      fontsize=11, fontweight='bold', color='#E64A19')
    cb2 = fig.colorbar(im2, ax=ax_epi_cb, fraction=0.9, pad=0.0)
    cb2.ax.tick_params(labelsize=7)

    # Row 3: Total
    ax_tot = fig.add_subplot(gs[3, 0])
    ax_tot_cb = fig.add_subplot(gs[3, 1])
    im3 = ax_tot.imshow(total.T, aspect='auto', cmap=cmap_total, interpolation='nearest')
    ax_tot.set_yticks(np.arange(M))
    ax_tot.set_yticklabels(variable_names, fontsize=8)
    ax_tot.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_tot.set_title('Total Uncertainty (Aleatoric + Epistemic)',
                      fontsize=11, fontweight='bold', color='#7B1FA2')
    cb3 = fig.colorbar(im3, ax=ax_tot_cb, fraction=0.9, pad=0.0)
    cb3.ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty heatmap to {save_path}")


# ---------------------------------------------------------------------------
#  5. Metrics Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_metrics_comparison(metrics_list, model_names, save_path='metrics_comparison.pdf'):
    """Bar chart comparing metrics across models."""
    key_metrics = ['MSE', 'MAE', 'RMSE', 'NLL', 'CRPS', 'Calibration_Error', 'Sharpness_90']
    display_names = ['MSE', 'MAE', 'RMSE', 'NLL', 'CRPS', 'Cal. Error', 'Sharpness\n(90%)']

    n_models = len(model_names)
    n_metrics = len(key_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

    for i, (metrics, name) in enumerate(zip(metrics_list, model_names)):
        values = [metrics.get(m, 0.0) for m in key_metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison plot to {save_path}")


# ---------------------------------------------------------------------------
#  Master plot generator
# ---------------------------------------------------------------------------
def generate_all_plots(results_dict, output_dir, prefix=''):
    """Generate all standard plots from inference results."""
    os.makedirs(output_dir, exist_ok=True)

    mu = results_dict['mu_mean']
    total_var = results_dict['total_var']
    target = results_dict['targets']
    epistemic_var = results_dict['epistemic_var']
    aleatoric_var = results_dict['aleatoric_var']

    # 1. Prediction intervals (first 3 samples, first variable)
    for i in range(min(3, mu.shape[0])):
        plot_prediction_intervals(
            mu, total_var, target,
            variable_idx=0, sample_idx=i,
            save_path=os.path.join(output_dir, f'{prefix}pred_intervals_sample{i}.pdf')
        )

    # 2. Calibration
    plot_calibration(
        mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}calibration.pdf')
    )

    # 3. Uncertainty decomposition (first 3 variables)
    n_vars = min(3, mu.shape[2])
    for v in range(n_vars):
        plot_uncertainty_decomposition(
            epistemic_var, aleatoric_var,
            variable_idx=v,
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_decomp_var{v}.pdf')
        )

    # 4. Uncertainty heatmap (first sample, all variables)
    plot_uncertainty_heatmap(
        epistemic_var, aleatoric_var, target, mu,
        sample_idx=0,
        max_variables=min(21, mu.shape[2]),
        save_path=os.path.join(output_dir, f'{prefix}uncertainty_heatmap.pdf')
    )
