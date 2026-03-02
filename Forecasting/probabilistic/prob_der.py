"""
Deep Evidential Regression (DER) module for probabilistic TSLANet.

Implements the Normal-Inverse-Gamma (NIG) prior approach from:
  - Amini et al. (2020) "Deep Evidential Regression"
  - ProbFM (Chinta et al. 2025) adaptation for time series

Key advantage over Gaussian NLL + MC Dropout:
  Single forward pass provides explicit epistemic-aleatoric decomposition
  without requiring multiple stochastic passes or ensemble training.

NIG distribution: p(mu, sigma^2) = NIG(gamma, nu, alpha, beta)
  - p(mu | sigma^2) = N(mu; gamma, sigma^2 / nu)
  - p(sigma^2) = Inverse-Gamma(alpha, beta)

Uncertainty decomposition (closed-form):
  - Aleatoric: E[sigma^2] = beta / (alpha - 1)
  - Epistemic: Var[mu] = beta / ((alpha - 1) * nu)
  - Total: beta * (nu + 1) / ((alpha - 1) * nu)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NIGHead(nn.Module):
    """
    Normal-Inverse-Gamma output head for Deep Evidential Regression.

    Produces 4 NIG parameters from backbone features:
        gamma: unconstrained mean prediction (real-valued)
        nu:    > 0, precision parameter (virtual evidence for the mean)
        alpha: > 1, shape of inverse-gamma (required for finite variance)
        beta:  > 0, scale of inverse-gamma
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.gamma_head = nn.Linear(in_features, out_features)
        self.nu_head = nn.Linear(in_features, out_features)
        self.alpha_head = nn.Linear(in_features, out_features)
        self.beta_head = nn.Linear(in_features, out_features)

        # Initialization strategy:
        # - gamma: default Xavier init (learns the mean)
        # - nu: bias=1.0 -> softplus(1.0) ~ 1.31, moderate initial evidence
        # - alpha: bias=1.0 -> softplus(1.0)+1 ~ 2.31, safe shape param
        # - beta: bias=0.0 -> softplus(0.0) ~ 0.69, moderate initial scale
        nn.init.constant_(self.nu_head.bias, 1.0)
        nn.init.constant_(self.alpha_head.bias, 1.0)
        nn.init.constant_(self.beta_head.bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: (B*M, D) flattened backbone features
        Returns:
            gamma: (B*M, T) unconstrained mean
            nu:    (B*M, T) > 0 via Softplus + epsilon
            alpha: (B*M, T) > 1 via Softplus + 1 + epsilon
            beta:  (B*M, T) > 0 via Softplus + epsilon
        """
        eps = 1e-6
        gamma = self.gamma_head(features)
        nu = F.softplus(self.nu_head(features)) + eps
        alpha = F.softplus(self.alpha_head(features)) + 1.0 + eps
        beta = F.softplus(self.beta_head(features)) + eps
        return gamma, nu, alpha, beta


class EvidentialLoss(nn.Module):
    """
    Evidential loss for NIG distribution (Amini et al. 2020).

    L = L_NLL + lambda_evd * evidence_scale * L_reg

    L_NLL = 0.5*log(pi/nu) - alpha*log(Omega)
            + (alpha+0.5)*log((y-gamma)^2 * nu + Omega)
            + lgamma(alpha) - lgamma(alpha+0.5)
    where Omega = 2*beta*(1+nu)

    L_reg = |y - gamma| * (2*nu + alpha)
    Penalizes high evidence (confident predictions) when the model is wrong.
    """

    def __init__(self, lambda_evd=0.05, reduction='mean'):
        super().__init__()
        self.lambda_evd = lambda_evd
        self.reduction = reduction

    def forward(self, gamma, nu, alpha, beta, target, evidence_scale=1.0):
        """
        Args:
            gamma: (B, T, M) predicted means
            nu:    (B, T, M) precision parameters (> 0)
            alpha: (B, T, M) shape parameters (> 1)
            beta:  (B, T, M) scale parameters (> 0)
            target: (B, T, M) ground truth
            evidence_scale: float in [0, 1] for evidence annealing

        Returns:
            scalar loss
        """
        omega = 2.0 * beta * (1.0 + nu)

        # NIG negative log-likelihood
        nll = (0.5 * torch.log(math.pi / (nu + 1e-8))
               - alpha * torch.log(omega + 1e-8)
               + (alpha + 0.5) * torch.log((target - gamma).pow(2) * nu + omega + 1e-8)
               + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))

        # Regularization: penalize wrong predictions with high evidence
        reg = torch.abs(target - gamma) * (2.0 * nu + alpha)

        loss = nll + self.lambda_evd * evidence_scale * reg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CoverageLoss(nn.Module):
    """
    Differentiable coverage loss from ProbFM.

    Optimizes: |PICP_target - PICP_actual|
    where PICP is the prediction interval coverage probability.

    Uses sigmoid soft approximation to make the indicator function differentiable.
    CI width is derived from the NIG posterior (Student-t distribution).
    """

    def __init__(self, target_coverage=0.9, sharpness=10.0):
        super().__init__()
        self.target_coverage = target_coverage
        self.sharpness = sharpness
        # z-score for normal approximation at target coverage
        # For 90% coverage: z ~ 1.645
        self.register_buffer(
            'z_score',
            torch.tensor(self._normal_quantile(target_coverage))
        )

    @staticmethod
    def _normal_quantile(coverage):
        """Normal approximation for Student-t quantile (valid for alpha >> 1)."""
        import math
        # Phi^{-1}((1+coverage)/2)
        # Using the approximation: z = sqrt(2) * erfinv(coverage)
        p = coverage
        # erfinv is the inverse error function
        # For 0.9: erfinv(0.9) ~ 1.1631, z ~ 1.6449
        return math.sqrt(2) * torch.erfinv(torch.tensor(p)).item()

    def forward(self, gamma, nu, alpha, beta, target):
        """
        Args:
            gamma, nu, alpha, beta: (B, T, M) NIG parameters
            target: (B, T, M) ground truth
        Returns:
            scalar coverage loss
        """
        # Student-t CI half-width: z * sqrt(beta*(nu+1) / (alpha*nu))
        ci_half_width = self.z_score * torch.sqrt(
            beta * (nu + 1.0) / (alpha * nu + 1e-8) + 1e-8
        )

        lower = gamma - ci_half_width
        upper = gamma + ci_half_width

        # Soft coverage via sigmoid approximation of indicator
        in_lower = torch.sigmoid(self.sharpness * (target - lower))
        in_upper = torch.sigmoid(self.sharpness * (upper - target))
        soft_covered = in_lower * in_upper

        picp_actual = soft_covered.mean()
        return torch.abs(self.target_coverage - picp_actual)


def der_predict(model, dataloader, pred_len, device='cuda'):
    """
    Deep Evidential Regression inference: single-pass uncertainty decomposition.

    Performs ONE forward pass per batch (no sampling required).
    Computes epistemic and aleatoric uncertainty analytically from NIG parameters.

    Args:
        model: ProbabilisticTSLANet or GaussianLSTM with uncertainty_method='evidential'
        dataloader: test DataLoader
        pred_len: prediction length to slice outputs
        device: 'cuda' or 'cpu'

    Returns:
        dict with same format as mc_dropout_predict for downstream compatibility:
            mu_mean, targets, epistemic_var, aleatoric_var, total_var,
            mu_samples, log_var_samples (for metric compat),
            + DER-specific: der_gamma, der_nu, der_alpha, der_beta
    """
    model.eval()
    model = model.to(device)

    gamma_list, nu_list, alpha_list, beta_list, tgt_list = [], [], [], [], []

    for batch in dataloader:
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y[:, -pred_len:, :].float()

        with torch.no_grad():
            gamma, nu, alpha, beta = model(batch_x)
            gamma = gamma[:, -pred_len:, :]
            nu = nu[:, -pred_len:, :]
            alpha = alpha[:, -pred_len:, :]
            beta = beta[:, -pred_len:, :]

        gamma_list.append(gamma.cpu())
        nu_list.append(nu.cpu())
        alpha_list.append(alpha.cpu())
        beta_list.append(beta.cpu())
        tgt_list.append(batch_y.cpu())

    gamma_all = torch.cat(gamma_list, dim=0)   # (N, T, M)
    nu_all = torch.cat(nu_list, dim=0)
    alpha_all = torch.cat(alpha_list, dim=0)
    beta_all = torch.cat(beta_list, dim=0)
    targets = torch.cat(tgt_list, dim=0)

    # --- NIG uncertainty decomposition (closed-form, NO sampling) ---
    # Aleatoric: expected data noise = E[sigma^2] = beta / (alpha - 1)
    aleatoric_var = beta_all / (alpha_all - 1.0 + 1e-8)

    # Epistemic: model uncertainty = Var[mu] = beta / ((alpha-1) * nu)
    epistemic_var = beta_all / ((alpha_all - 1.0 + 1e-8) * (nu_all + 1e-8))

    # Total variance
    total_var = aleatoric_var + epistemic_var

    # Compute log_var equivalent for NLL/CRPS metric compatibility
    log_var_equiv = torch.log(total_var + 1e-8)

    return {
        # Standard dict format (compatible with compute_all_metrics)
        'mu_samples': gamma_all.unsqueeze(0),         # (1, N, T, M)
        'log_var_samples': log_var_equiv.unsqueeze(0), # (1, N, T, M)
        'targets': targets,
        'mu_mean': gamma_all,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_var': total_var,
        # DER-specific extras (for analysis and visualization)
        'der_gamma': gamma_all,
        'der_nu': nu_all,
        'der_alpha': alpha_all,
        'der_beta': beta_all,
    }
