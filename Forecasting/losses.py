import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """
    NLL = 0.5 * mean(log_var + (target - mu)^2 / exp(log_var))
    No constant term during training (doesn't affect gradients).
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var, target):
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + (target - mu) ** 2 * precision)
        return loss.mean()
