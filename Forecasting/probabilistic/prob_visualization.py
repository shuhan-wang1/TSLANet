import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prob_metrics import compute_calibration, compute_pit_values


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


def generate_all_plots(results_dict, output_dir, prefix=''):
    """
    Generate all 5 standard plots from inference results.

    Args:
        results_dict: dict from mc_dropout_predict / deep_ensemble_predict
        output_dir: directory to save plots
        prefix: filename prefix (e.g., 'mc_dropout_' or 'ensemble_')
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

    # 2. Uncertainty decomposition (first 3 variables)
    n_vars = min(3, mu.shape[2])
    for v in range(n_vars):
        plot_uncertainty_decomposition(
            epistemic_var, aleatoric_var,
            variable_idx=v,
            save_path=os.path.join(output_dir, f'{prefix}uncertainty_decomp_var{v}.pdf')
        )

    # 3. Calibration
    plot_calibration(
        mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}calibration.pdf')
    )

    # 4. PIT histogram
    plot_pit_histogram(
        mu, total_var, target,
        save_path=os.path.join(output_dir, f'{prefix}pit_histogram.pdf')
    )
