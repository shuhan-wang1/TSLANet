import sys
import os

import torch
import torch.nn as nn
from einops import rearrange

# Add parent directory so we can import original TSLANet components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TSLANet_Forecasting import ICB, Adaptive_Spectral_Block, TSLANet_layer
from utils import random_masking_3D


class ProbabilisticTSLANet(nn.Module):
    """
    TSLANet with Gaussian likelihood output heads for probabilistic forecasting.

    Instead of a single linear output layer producing point predictions,
    this model has two heads:
        - mu_head: predicts the mean of the Gaussian
        - log_var_head: predicts the log-variance (heteroscedastic noise)

    The backbone (patch embedding + TSLANet layers) is identical to the original.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.uncertainty_method = getattr(args, 'uncertainty_method', 'gaussian')
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        self.num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)

        # Backbone (identical to original TSLANet)
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)
        ])

        # Output heads depend on uncertainty method
        backbone_out_dim = args.emb_dim * self.num_patches

        if self.uncertainty_method == 'evidential':
            from prob_der import NIGHead
            self.nig_head = NIGHead(backbone_out_dim, args.pred_len)
        else:
            # Gaussian: two output heads (mu + log_var)
            self.mu_head = nn.Linear(backbone_out_dim, args.pred_len)

            self.log_var_head = nn.Linear(backbone_out_dim, args.pred_len)
            # Initialize log_var_head so initial sigma is small but nonzero (~0.37)
            # This prevents NLL explosion at the start of training
            nn.init.constant_(self.log_var_head.bias, -2.0)
            nn.init.zeros_(self.log_var_head.weight)

    def pretrain(self, x_in):
        """Masked patch reconstruction for self-supervised pretraining.
        Identical to original TSLANet.pretrain()."""
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=self.args.mask_ratio)
        self.mask = self.mask.bool()
        xb_mask = self.input_layer(xb_mask)

        for tsla_blk in self.tsla_blocks:
            xb_mask = tsla_blk(xb_mask)

        return xb_mask, self.input_layer(x_patched)

    def forward_backbone(self, x):
        """
        Shared backbone: instance norm -> patch -> embed -> TSLANet blocks -> flatten.

        Returns:
            features: (B*M, emb_dim * num_patches) flattened backbone output
            means: (B, 1, M) per-sample means for denormalization
            stdev: (B, 1, M) per-sample stdev for denormalization
            B: batch size
            M: number of variables
        """
        B, L, M = x.shape

        # Instance normalization (RevIN-style)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        # Patching
        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # Embedding
        x = self.input_layer(x)

        # TSLANet blocks
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        # Flatten
        features = x.reshape(B * M, -1)

        return features, means, stdev, B, M

    def forward(self, x):
        """
        Forward pass producing distribution parameters.

        For Gaussian mode:
            Returns (mu, log_var) both (B, pred_len, M) denormalized.

        For Evidential mode (DER):
            Returns (gamma, nu, alpha, beta) all (B, pred_len, M) denormalized.
            gamma = mean, nu = precision, alpha = shape, beta = scale (NIG params).

        Denormalization:
            mu/gamma: y_real = y_norm * stdev + mean
            log_var:  log_var_real = log_var_norm + 2*log(stdev)
            beta:     beta_real = beta_norm * stdev^2  (scale ~ variance)
            nu, alpha: dimensionless, no denormalization needed
        """
        features, means, stdev, B, M = self.forward_backbone(x)

        if self.uncertainty_method == 'evidential':
            gamma_norm, nu, alpha, beta_norm = self.nig_head(features)

            # Rearrange to (B, pred_len, M)
            gamma_norm = rearrange(gamma_norm, '(b m) l -> b l m', b=B)
            nu = rearrange(nu, '(b m) l -> b l m', b=B)
            alpha = rearrange(alpha, '(b m) l -> b l m', b=B)
            beta_norm = rearrange(beta_norm, '(b m) l -> b l m', b=B)

            # Denormalize gamma (same as mu)
            gamma = gamma_norm * stdev + means

            # Denormalize beta: beta_real = beta_norm * stdev^2
            # Because beta parameterizes the scale of sigma^2, which scales quadratically
            beta = beta_norm * stdev.pow(2)

            # nu and alpha are dimensionless -- no denormalization needed
            return gamma, nu, alpha, beta
        else:
            # Gaussian path (original)
            mu_norm = self.mu_head(features)  # (B*M, pred_len)
            log_var_norm = self.log_var_head(features)  # (B*M, pred_len)
            log_var_norm = torch.clamp(log_var_norm, min=-10.0, max=10.0)

            mu_norm = rearrange(mu_norm, '(b m) l -> b l m', b=B)
            log_var_norm = rearrange(log_var_norm, '(b m) l -> b l m', b=B)

            mu = mu_norm * stdev + means
            log_var = log_var_norm + 2.0 * torch.log(stdev + 1e-5)

            return mu, log_var
