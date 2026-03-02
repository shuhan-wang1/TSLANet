import math

import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood for heteroscedastic Gaussian distribution.

    L = 0.5 * [log_var + (y - mu)^2 / exp(log_var)]

    The constant 0.5*log(2*pi) is omitted during training (does not affect gradients)
    but can be included for proper NLL evaluation via `include_constant=True`.
    """

    def __init__(self, reduction='mean', include_constant=False):
        super().__init__()
        self.reduction = reduction
        self.include_constant = include_constant

    def forward(self, mu, log_var, target):
        """
        Args:
            mu: (B, T, M) predicted means
            log_var: (B, T, M) predicted log-variances
            target: (B, T, M) ground truth
        Returns:
            scalar loss
        """
        precision = torch.exp(-log_var)
        nll = 0.5 * (log_var + (target - mu) ** 2 * precision)

        if self.include_constant:
            nll = nll + 0.5 * math.log(2 * math.pi)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.

    Analytic form:
        CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = standard normal CDF, phi = standard normal PDF.

    CRPS is a strictly proper scoring rule that simultaneously rewards
    calibration and sharpness.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, log_var, target):
        """
        Args:
            mu: (B, T, M) predicted means
            log_var: (B, T, M) predicted log-variances
            target: (B, T, M) ground truth
        Returns:
            scalar CRPS
        """
        sigma = torch.exp(0.5 * log_var)
        z = (target - mu) / (sigma + 1e-6)

        # Standard normal PDF and CDF
        phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))

        crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))

        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        return crps
