import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from prob_metrics import compute_calibration, compute_pit_values


# ---------------------------------------------------------------------------
#  1. Prediction Interval Plot (existing)
# ---------------------------------------------------------------------------
def plot_prediction_intervals(mu, total_var, target, variable_idx=0, sample_idx=0,
                              save_path='prediction_intervals.pdf'):
    """
    Plot prediction with shaded 50%, 90%, 95% prediction intervals.

    Args:
        mu: (N, T, M) predicted means
        total_var: (N, T, M) total variance
        target: (N, T, M) ground truth
        variable_idx: which variable to plot
        sample_idx: which sample to plot
        save_path: output file path
    """
    mu_s = mu[sample_idx, :, variable_idx].numpy()
    sigma_s = np.sqrt(total_var[sample_idx, :, variable_idx].numpy())
    target_s = target[sample_idx, :, variable_idx].numpy()
    T = len(mu_s)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Prediction intervals
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
#  2. Stacked Area Uncertainty Decomposition (existing)
# ---------------------------------------------------------------------------
def plot_uncertainty_decomposition(epistemic_var, aleatoric_var, variable_idx=0,
                                   save_path='uncertainty_decomposition.pdf'):
    """
    Stacked area chart of epistemic vs aleatoric variance over forecast horizon.
    Averaged across all test samples.

    Args:
        epistemic_var: (N, T, M) epistemic variance
        aleatoric_var: (N, T, M) aleatoric variance
        variable_idx: which variable to plot
        save_path: output file path
    """
    # Average across samples
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
#  3. Uncertainty Decomposition Heatmap (NEW)
# ---------------------------------------------------------------------------
def plot_uncertainty_heatmap(epistemic_var, aleatoric_var, target, mu,
                             sample_idx=0, max_variables=None,
                             variable_names=None,
                             save_path='uncertainty_heatmap.pdf'):
    """
    Heatmap-based uncertainty decomposition visualization.

    Produces a 4-row figure for a single test sample:
      Row 1: Ground truth vs predicted values (line overlay)
      Row 2: Aleatoric uncertainty heatmap  (data noise)
      Row 3: Epistemic uncertainty heatmap  (model uncertainty)
      Row 4: Total uncertainty heatmap

    X-axis = forecast horizon (time steps), Y-axis = variables.
    Color intensity encodes uncertainty magnitude.

    This directly satisfies the coursework requirement to output a
    probability distribution rather than a single predicted value:
    the heatmaps show the *spread* of the predictive distribution
    decomposed into its two fundamental sources.

    Args:
        epistemic_var: (N, T, M) epistemic variance
        aleatoric_var: (N, T, M) aleatoric variance
        target:        (N, T, M) ground truth values
        mu:            (N, T, M) predicted means
        sample_idx:    which test sample to visualize
        max_variables: maximum number of variables to display (None = all)
        variable_names: optional list of variable name strings
        save_path:     output file path
    """
    M = epistemic_var.shape[2]
    if max_variables is not None:
        M = min(M, max_variables)

    epi = epistemic_var[sample_idx, :, :M].numpy()   # (T, M)
    ale = aleatoric_var[sample_idx, :, :M].numpy()    # (T, M)
    total = (epi + ale)                                # (T, M)
    mu_s = mu[sample_idx, :, :M].numpy()              # (T, M)
    tgt_s = target[sample_idx, :, :M].numpy()         # (T, M)
    T = epi.shape[0]

    if variable_names is None:
        variable_names = [f'Var {i}' for i in range(M)]
    else:
        variable_names = variable_names[:M]

    # --- Custom colormaps ---
    cmap_ale = LinearSegmentedColormap.from_list(
        'aleatoric', ['#FFFFFF', '#BBDEFB', '#1565C0', '#0D47A1'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list(
        'epistemic', ['#FFFFFF', '#FFCCBC', '#E64A19', '#BF360C'], N=256)
    cmap_total = LinearSegmentedColormap.from_list(
        'total', ['#FFFFFF', '#E1BEE7', '#7B1FA2', '#4A148C'], N=256)

    # --- Figure layout: 4 rows ---
    fig = plt.figure(figsize=(max(14, T * 0.12), 3 + M * 0.55 * 3))
    gs = GridSpec(4, 2, width_ratios=[50, 1], hspace=0.35, wspace=0.05)

    # ====== Row 0: Prediction vs Ground Truth ======
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_pred_cb = fig.add_subplot(gs[0, 1])
    ax_pred_cb.axis('off')

    # Compute absolute prediction error as a heatmap
    abs_error = np.abs(mu_s - tgt_s).T                 # (M, T)
    cmap_err = LinearSegmentedColormap.from_list(
        'error', ['#E8F5E9', '#FFF9C4', '#FFAB91', '#C62828'], N=256)
    im0 = ax_pred.imshow(abs_error, aspect='auto', cmap=cmap_err,
                          interpolation='nearest')
    ax_pred.set_yticks(np.arange(M))
    ax_pred.set_yticklabels(variable_names, fontsize=8)
    ax_pred.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_pred.set_title('Absolute Prediction Error  |y - $\\hat{y}$|',
                      fontsize=11, fontweight='bold')
    cb0 = fig.colorbar(im0, ax=ax_pred_cb, fraction=0.9, pad=0.0)
    cb0.ax.tick_params(labelsize=7)

    # ====== Row 1: Aleatoric Heatmap ======
    ax_ale = fig.add_subplot(gs[1, 0])
    ax_ale_cb = fig.add_subplot(gs[1, 1])
    im1 = ax_ale.imshow(ale.T, aspect='auto', cmap=cmap_ale,
                         interpolation='nearest')
    ax_ale.set_yticks(np.arange(M))
    ax_ale.set_yticklabels(variable_names, fontsize=8)
    ax_ale.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_ale.set_title('Aleatoric Uncertainty  (Data Noise: $\\beta / (\\alpha - 1)$)',
                      fontsize=11, fontweight='bold', color='#1565C0')
    cb1 = fig.colorbar(im1, ax=ax_ale_cb, fraction=0.9, pad=0.0)
    cb1.ax.tick_params(labelsize=7)

    # ====== Row 2: Epistemic Heatmap ======
    ax_epi = fig.add_subplot(gs[2, 0])
    ax_epi_cb = fig.add_subplot(gs[2, 1])
    im2 = ax_epi.imshow(epi.T, aspect='auto', cmap=cmap_epi,
                         interpolation='nearest')
    ax_epi.set_yticks(np.arange(M))
    ax_epi.set_yticklabels(variable_names, fontsize=8)
    ax_epi.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_epi.set_title('Epistemic Uncertainty  (Model Uncertainty: $\\beta / ((\\alpha-1)\\nu)$)',
                      fontsize=11, fontweight='bold', color='#E64A19')
    cb2 = fig.colorbar(im2, ax=ax_epi_cb, fraction=0.9, pad=0.0)
    cb2.ax.tick_params(labelsize=7)

    # ====== Row 3: Total Uncertainty Heatmap ======
    ax_tot = fig.add_subplot(gs[3, 0])
    ax_tot_cb = fig.add_subplot(gs[3, 1])
    im3 = ax_tot.imshow(total.T, aspect='auto', cmap=cmap_total,
                         interpolation='nearest')
    ax_tot.set_yticks(np.arange(M))
    ax_tot.set_yticklabels(variable_names, fontsize=8)
    ax_tot.set_xlabel('Forecast Horizon (timestep)', fontsize=9)
    ax_tot.set_title('Total Uncertainty  (Aleatoric + Epistemic)',
                      fontsize=11, fontweight='bold', color='#7B1FA2')
    cb3 = fig.colorbar(im3, ax=ax_tot_cb, fraction=0.9, pad=0.0)
    cb3.ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty heatmap to {save_path}")


def plot_uncertainty_heatmap_comparison(results_list, model_names,
                                        sample_idx=0, variable_idx=0,
                                        save_path='uncertainty_heatmap_comparison.pdf'):
    """
    Side-by-side uncertainty heatmap comparison across multiple models.

    For a single variable and sample, shows how different models produce
    different aleatoric and epistemic uncertainty patterns. This is ideal
    for comparing:
      - A3 (MC Dropout) vs A7 (DER) vs A7+CovLoss

    Args:
        results_list: list of result dicts (from mc_dropout_predict / der_predict)
        model_names:  list of model name strings
        sample_idx:   which test sample to visualize
        variable_idx: which variable to focus on
        save_path:    output file path
    """
    n_models = len(results_list)

    fig, axes = plt.subplots(n_models, 3, figsize=(18, 3.5 * n_models),
                              squeeze=False)

    cmap_ale = LinearSegmentedColormap.from_list(
        'ale', ['#FFFFFF', '#BBDEFB', '#1565C0'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list(
        'epi', ['#FFFFFF', '#FFCCBC', '#E64A19'], N=256)
    cmap_total = LinearSegmentedColormap.from_list(
        'total', ['#FFFFFF', '#E1BEE7', '#7B1FA2'], N=256)

    # Compute global vmin/vmax for consistent color scales
    all_ale = [r['aleatoric_var'][sample_idx, :, variable_idx].numpy() for r in results_list]
    all_epi = [r['epistemic_var'][sample_idx, :, variable_idx].numpy() for r in results_list]
    vmax_ale = max(a.max() for a in all_ale) * 1.05
    vmax_epi = max(e.max() for e in all_epi) * 1.05
    vmax_total = max((a + e).max() for a, e in zip(all_ale, all_epi)) * 1.05

    for row, (res, name) in enumerate(zip(results_list, model_names)):
        ale = res['aleatoric_var'][sample_idx, :, variable_idx].numpy()
        epi = res['epistemic_var'][sample_idx, :, variable_idx].numpy()
        total = ale + epi
        T = len(ale)
        t = np.arange(T)

        # Reshape to (1, T) for imshow (single row heatmap)
        axes[row, 0].imshow(ale.reshape(1, -1), aspect='auto', cmap=cmap_ale,
                            vmin=0, vmax=vmax_ale, interpolation='nearest')
        axes[row, 0].set_yticks([0])
        axes[row, 0].set_yticklabels([name], fontsize=9, fontweight='bold')
        if row == 0:
            axes[row, 0].set_title('Aleatoric (Data Noise)', fontsize=11,
                                   fontweight='bold', color='#1565C0')

        axes[row, 1].imshow(epi.reshape(1, -1), aspect='auto', cmap=cmap_epi,
                            vmin=0, vmax=vmax_epi, interpolation='nearest')
        axes[row, 1].set_yticks([0])
        axes[row, 1].set_yticklabels([name], fontsize=9, fontweight='bold')
        if row == 0:
            axes[row, 1].set_title('Epistemic (Model Uncertainty)', fontsize=11,
                                   fontweight='bold', color='#E64A19')

        axes[row, 2].imshow(total.reshape(1, -1), aspect='auto', cmap=cmap_total,
                            vmin=0, vmax=vmax_total, interpolation='nearest')
        axes[row, 2].set_yticks([0])
        axes[row, 2].set_yticklabels([name], fontsize=9, fontweight='bold')
        if row == 0:
            axes[row, 2].set_title('Total Uncertainty', fontsize=11,
                                   fontweight='bold', color='#7B1FA2')

        # Only bottom row gets x-axis labels
        for col in range(3):
            if row == n_models - 1:
                axes[row, col].set_xlabel('Forecast Horizon', fontsize=9)
            else:
                axes[row, col].set_xticklabels([])

    plt.suptitle(f'Uncertainty Decomposition Comparison (Variable {variable_idx}, '
                 f'Sample {sample_idx})', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty heatmap comparison to {save_path}")


def plot_uncertainty_heatmap_temporal(epistemic_var, aleatoric_var, target, mu,
                                      variable_idx=0, max_samples=50,
                                      save_path='uncertainty_heatmap_temporal.pdf'):
    """
    Temporal uncertainty heatmap across multiple test samples.

    X-axis = forecast horizon, Y-axis = test sample index.
    Each row is one test sample; color encodes uncertainty magnitude.

    This reveals temporal patterns:
      - Whether epistemic uncertainty spikes for out-of-distribution samples
      - Whether aleatoric uncertainty grows with longer forecast horizons

    Args:
        epistemic_var: (N, T, M) epistemic variance
        aleatoric_var: (N, T, M) aleatoric variance
        target:        (N, T, M) ground truth values
        mu:            (N, T, M) predicted means
        variable_idx:  which variable to focus on
        max_samples:   maximum number of test samples to display
        save_path:     output file path
    """
    N = min(epistemic_var.shape[0], max_samples)
    T = epistemic_var.shape[1]

    epi = epistemic_var[:N, :, variable_idx].numpy()   # (N, T)
    ale = aleatoric_var[:N, :, variable_idx].numpy()    # (N, T)
    total = epi + ale
    err = np.abs(mu[:N, :, variable_idx].numpy() - target[:N, :, variable_idx].numpy())

    cmap_ale = LinearSegmentedColormap.from_list(
        'ale', ['#FFFFFF', '#BBDEFB', '#1565C0', '#0D47A1'], N=256)
    cmap_epi = LinearSegmentedColormap.from_list(
        'epi', ['#FFFFFF', '#FFCCBC', '#E64A19', '#BF360C'], N=256)
    cmap_err = LinearSegmentedColormap.from_list(
        'err', ['#E8F5E9', '#FFF9C4', '#FFAB91', '#C62828'], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, N * 0.12)),
                              sharey=True)

    # Panel 1: Aleatoric
    im0 = axes[0].imshow(ale, aspect='auto', cmap=cmap_ale, interpolation='nearest')
    axes[0].set_title('Aleatoric Uncertainty\n(Data Noise)',
                      fontsize=11, fontweight='bold', color='#1565C0')
    axes[0].set_ylabel('Test Sample Index', fontsize=10)
    axes[0].set_xlabel('Forecast Horizon', fontsize=10)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Panel 2: Epistemic
    im1 = axes[1].imshow(epi, aspect='auto', cmap=cmap_epi, interpolation='nearest')
    axes[1].set_title('Epistemic Uncertainty\n(Model Uncertainty)',
                      fontsize=11, fontweight='bold', color='#E64A19')
    axes[1].set_xlabel('Forecast Horizon', fontsize=10)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Panel 3: Absolute error (for correlation check)
    im2 = axes[2].imshow(err, aspect='auto', cmap=cmap_err, interpolation='nearest')
    axes[2].set_title('Absolute Error  |y - $\\hat{y}$|\n(For Correlation Check)',
                      fontsize=11, fontweight='bold', color='#C62828')
    axes[2].set_xlabel('Forecast Horizon', fontsize=10)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle(f'Temporal Uncertainty Patterns (Variable {variable_idx}, '
                 f'{N} samples)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal uncertainty heatmap to {save_path}")


# ---------------------------------------------------------------------------
#  4. Calibration Plot (existing)
# ---------------------------------------------------------------------------
def plot_calibration(mu, total_var, target, save_path='calibration.pdf'):
    """
    Calibration plot: observed coverage vs nominal coverage.
    A well-calibrated model falls on the diagonal.

    Args:
        mu: (N, T, M) predicted means
        total_var: (N, T, M) total variance
        target: (N, T, M) ground truth
        save_path: output file path
    """
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal = compute_calibration(mu, total_var, target, quantiles=quantiles)

    nominal = list(cal.keys())
    observed = list(cal.values())

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(nominal, observed, 'o-', color='steelblue', linewidth=2,
            markersize=8, label='Model')

    # Shade the region between model and perfect
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
#  5. PIT Histogram (existing)
# ---------------------------------------------------------------------------
def plot_pit_histogram(mu, total_var, target, save_path='pit_histogram.pdf'):
    """
    Probability Integral Transform histogram.
    Should be approximately uniform for a well-calibrated model.

    Args:
        mu: (N, T, M) predicted means
        total_var: (N, T, M) total variance
        target: (N, T, M) ground truth
        save_path: output file path
    """
    pit_values = compute_pit_values(mu, total_var, target).numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(pit_values, bins=20, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', linewidth=0.5)
    ax.axhline(y=1.0, color='darkorange', linestyle='--', linewidth=2,
               label='Uniform (ideal)')

    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title('PIT Histogram')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PIT histogram to {save_path}")


# ---------------------------------------------------------------------------
#  6. Metrics Comparison Bar Chart (existing)
# ---------------------------------------------------------------------------
def plot_metrics_comparison(metrics_list, model_names, save_path='metrics_comparison.pdf'):
    """
    Bar chart comparing metrics across models.

    Args:
        metrics_list: list of dicts from compute_all_metrics, one per model
        model_names: list of model name strings
        save_path: output file path
    """
    # Select key metrics for comparison
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
        # Add value labels on bars
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
#  7. Master plot generator
# ---------------------------------------------------------------------------
def generate_all_plots(results_dict, output_dir, prefix=''):
    """
    Generate all standard plots from inference results.

    Produces:
      1. Prediction interval plots (first 3 samples)
      2. Stacked-area uncertainty decomposition (first 3 variables)
      3. Uncertainty decomposition heatmap (single-sample, all variables)
      4. Temporal uncertainty heatmap (all samples, first variable)
      5. Calibration diagram
      6. PIT histogram

    Args:
        results_dict: dict from mc_dropout_predict / deep_ensemble_predict / der_predict
        output_dir: directory to save plots
        prefix: filename prefix (e.g., 'mc_dropout_' or 'ensemble_' or 'der_')
    """
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

    # 2. Stacked-area uncertainty decomposition (first 3 variables)
    n_vars = min(3, mu.shape[2])
    for v in range(n_vars):
        plot_uncertainty_decomposition(
            epistemic_var, aleatoric_var,
            variable_idx=v,
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_decomp_var{v}.pdf')
        )

    # 3. Uncertainty decomposition heatmap (single sample, all variables)
    for i in range(min(3, mu.shape[0])):
        plot_uncertainty_heatmap(
            epistemic_var, aleatoric_var, target, mu,
            sample_idx=i,
            max_variables=min(21, mu.shape[2]),
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_heatmap_sample{i}.pdf')
        )

    # 4. Temporal uncertainty heatmap (all samples, first variable)
    for v in range(n_vars):
        plot_uncertainty_heatmap_temporal(
            epistemic_var, aleatoric_var, target, mu,
            variable_idx=v,
            max_samples=50,
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_temporal_var{v}.pdf')
        )

    # 5. Calibration
    plot_calibration(
        mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}calibration.pdf')
    )

    # 6. PIT histogram
    plot_pit_histogram(
        mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}pit_histogram.pdf')
    )
