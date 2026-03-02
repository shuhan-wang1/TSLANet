import math

import torch
import torch.nn as nn


class GaussianCoverageLoss(nn.Module):
    """
    Differentiable coverage loss for Gaussian predictive distributions.

    Optimizes: |PICP_target - PICP_actual|
    where PICP is the Prediction Interval Coverage Probability.

    For a Gaussian predictive distribution N(mu, sigma^2), the confidence
    interval at coverage level p is:
        CI = mu +/- z_{(1+p)/2} * sigma
    where z is the quantile of the standard normal distribution.

    Uses sigmoid soft approximation to make the indicator function
    differentiable (same technique as the NIG-based CoverageLoss in
    prob_der.py, adapted here for Gaussian outputs).

    Theoretical basis:
        ProbFM (Chinta et al., 2025) first integrated coverage loss with
        Deep Evidential Regression. We extend this idea to Gaussian
        predictive distributions, enabling coverage optimization for
        MC Dropout (A3) and Deep Ensemble (A4) configurations without
        requiring NIG parameterization.

        The key insight is that the coverage loss objective
        |PICP_target - PICP_actual| is distribution-agnostic: it only
        requires the ability to compute prediction intervals, which any
        parametric predictive distribution provides.
    """

    def __init__(self, target_coverage=0.9, sharpness=10.0):
        super().__init__()
        self.target_coverage = target_coverage
        self.sharpness = sharpness
        # z-score for Gaussian CI: Phi^{-1}((1 + coverage) / 2)
        # For 90% coverage: z ~ 1.6449
        self.register_buffer(
            'z_score',
            torch.tensor(self._normal_quantile(target_coverage))
        )

    @staticmethod
    def _normal_quantile(coverage):
        """Compute z = Phi^{-1}((1 + coverage) / 2) = sqrt(2) * erfinv(coverage)."""
        return math.sqrt(2) * torch.erfinv(torch.tensor(coverage)).item()

    def forward(self, mu, log_var, target):
        """
        Args:
            mu:      (B, T, M) predicted means
            log_var: (B, T, M) predicted log-variances
            target:  (B, T, M) ground truth

        Returns:
            scalar coverage loss = |PICP_target - PICP_actual|

        The Gaussian CI half-width is:
            ci_half = z * sigma = z * exp(0.5 * log_var)

        The soft indicator (differentiable approximation) is:
            soft_covered_i = sigmoid(s * (y_i - lower_i)) * sigmoid(s * (upper_i - y_i))
        where s is the sharpness parameter controlling sigmoid steepness.
        """
        sigma = torch.exp(0.5 * log_var)
        ci_half_width = self.z_score * sigma

        lower = mu - ci_half_width
        upper = mu + ci_half_width

        # Soft coverage via sigmoid approximation of indicator function
        in_lower = torch.sigmoid(self.sharpness * (target - lower))
        in_upper = torch.sigmoid(self.sharpness * (upper - target))
        soft_covered = in_lower * in_upper

        picp_actual = soft_covered.mean()
        return torch.abs(self.target_coverage - picp_actual)


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
