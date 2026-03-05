import math

import numpy as np
import torch
import torch.nn.functional as F


def compute_nll(mu, log_var, target):
    """
    Gaussian negative log-likelihood (with log(2*pi) constant for reporting).

    NLL = 0.5 * [log(var) + (y - mu)^2 / var + log(2*pi)]
    """
    precision = torch.exp(-log_var)
    nll = 0.5 * (log_var + (target - mu) ** 2 * precision + math.log(2 * math.pi))
    return nll.mean().item()


def compute_crps(mu, log_var, target):
    """
    Analytic Gaussian CRPS.

    CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    """
    sigma = torch.exp(0.5 * log_var)
    z = (target - mu) / (sigma + 1e-6)

    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))

    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean().item()


def compute_calibration(mu, total_var, target, quantiles=None):
    """
    For each nominal coverage level p, compute observed coverage.

    Returns:
        dict {nominal_coverage: observed_coverage}
    """
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    sigma = torch.sqrt(total_var + 1e-8)
    results = {}

    for q in quantiles:
        z = torch.erfinv(torch.tensor(q, dtype=torch.float64)).item() * math.sqrt(2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        covered = ((target >= lower) & (target <= upper)).float().mean().item()
        results[q] = covered

    return results


def compute_sharpness(total_var, confidence=0.9):
    """Average width of prediction interval at given confidence level."""
    z = torch.erfinv(torch.tensor(confidence, dtype=torch.float64)).item() * math.sqrt(2)
    sigma = torch.sqrt(total_var + 1e-8)
    width = 2.0 * z * sigma
    return width.mean().item()


def compute_all_metrics(results_dict):
    """
    Compute all probabilistic and point metrics from inference results.

    Args:
        results_dict: dict with keys: mu_mean, targets, total_var, epistemic_var, aleatoric_var

    Returns:
        dict of metric_name -> value
    """
    mu = results_dict['mu_mean']
    target = results_dict['targets']
    total_var = results_dict['total_var']
    epistemic_var = results_dict['epistemic_var']
    aleatoric_var = results_dict['aleatoric_var']

    metrics = {}

    # Point metrics
    metrics['MSE'] = F.mse_loss(mu, target).item()
    metrics['MAE'] = F.l1_loss(mu, target).item()
    metrics['RMSE'] = math.sqrt(metrics['MSE'])

    # Check if model has any uncertainty
    has_uncertainty = total_var.sum().item() > 0

    if has_uncertainty:
        log_total_var = torch.log(total_var + 1e-8)

        # Proper scoring rules
        metrics['NLL'] = compute_nll(mu, log_total_var, target)
        metrics['CRPS'] = compute_crps(mu, log_total_var, target)

        # Sharpness
        metrics['Sharpness_90'] = compute_sharpness(total_var, 0.9)
        metrics['Sharpness_50'] = compute_sharpness(total_var, 0.5)

        # Calibration
        calibration = compute_calibration(mu, total_var, target)
        for nom_cov, obs_cov in calibration.items():
            metrics[f'Cal_{nom_cov}'] = obs_cov

        # Average calibration error
        cal_error = np.mean([abs(nom - obs) for nom, obs in calibration.items()])
        metrics['Calibration_Error'] = float(cal_error)

        # Uncertainty decomposition
        metrics['Mean_Epistemic_Var'] = epistemic_var.mean().item()
        metrics['Mean_Aleatoric_Var'] = aleatoric_var.mean().item()
        total_mean = total_var.mean().item()
        metrics['Epistemic_Fraction'] = (
            epistemic_var.mean().item() / (total_mean + 1e-8)
        )

    return metrics
