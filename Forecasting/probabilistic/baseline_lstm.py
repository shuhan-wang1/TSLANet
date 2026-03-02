import sys
import os

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prob_losses import GaussianNLLLoss


class GaussianLSTM(nn.Module):
    """
    Channel-independent LSTM with Gaussian likelihood output heads.

    Processes each variable independently (like TSLANet) with the same
    instance normalization and denormalization scheme for fair comparison.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden if hasattr(args, 'lstm_hidden') else 128
        self.num_layers = args.lstm_layers if hasattr(args, 'lstm_layers') else 2
        dropout = args.dropout if hasattr(args, 'dropout') else 0.3

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

        # Dropout for MC Dropout inference
        self.mc_dropout = nn.Dropout(p=dropout)

        self.mu_head = nn.Linear(self.hidden_dim, args.pred_len)
        self.log_var_head = nn.Linear(self.hidden_dim, args.pred_len)

        # Same initialization as ProbabilisticTSLANet
        nn.init.constant_(self.log_var_head.bias, -2.0)
        nn.init.zeros_(self.log_var_head.weight)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, M)
        Returns:
            mu: (B, pred_len, M) denormalized means
            log_var: (B, pred_len, M) denormalized log-variances
        """
        B, L, M = x.shape

        # Instance normalization (same as TSLANet)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        # Process each variable independently
        x = rearrange(x, 'b l m -> (b m) l 1')

        lstm_out, _ = self.lstm(x)  # (B*M, L, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (B*M, hidden_dim)

        # Apply MC dropout
        last_hidden = self.mc_dropout(last_hidden)

        mu_norm = self.mu_head(last_hidden)          # (B*M, pred_len)
        log_var_norm = self.log_var_head(last_hidden) # (B*M, pred_len)
        log_var_norm = torch.clamp(log_var_norm, min=-10.0, max=10.0)

        # Rearrange to (B, pred_len, M)
        mu_norm = rearrange(mu_norm, '(b m) l -> b l m', b=B)
        log_var_norm = rearrange(log_var_norm, '(b m) l -> b l m', b=B)

        # Denormalize
        mu = mu_norm * stdev + means
        log_var = log_var_norm + 2.0 * torch.log(stdev + 1e-5)

        return mu, log_var


class LSTMModelTraining(L.LightningModule):
    """Lightning wrapper for GaussianLSTM training. Mirrors ProbModelTraining."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.model = GaussianLSTM(args)

        if args.probabilistic:
            self.criterion = GaussianNLLLoss(reduction='mean')
        else:
            self.criterion = nn.MSELoss()

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()

        self.test_mu = []
        self.test_log_var = []
        self.test_true = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr if hasattr(self.args, 'lr') else 1e-4,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 1e-6,
        )
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2
            ),
            'monitor': 'val_mse',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        mu, log_var = self.model(batch_x)
        mu = mu[:, -self.args.pred_len:, :]
        log_var = log_var[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :]

        if self.args.probabilistic:
            loss = self.criterion(mu, log_var, batch_y)
        else:
            loss = self.criterion(mu, batch_y)

        pred = mu.detach().cpu()
        true = batch_y.detach().cpu()

        mse = self.mse(pred.contiguous(), true.contiguous())
        mae = self.mae(pred, true)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float()
        batch_y = batch_y[:, -self.args.pred_len:, :].float()

        mu, log_var = self.model(batch_x)
        mu = mu[:, -self.args.pred_len:, :]
        log_var = log_var[:, -self.args.pred_len:, :]

        self.test_mu.append(mu.detach().cpu())
        self.test_log_var.append(log_var.detach().cpu())
        self.test_true.append(batch_y.detach().cpu())

    def on_test_epoch_end(self):
        mu_all = torch.cat(self.test_mu)
        true_all = torch.cat(self.test_true)

        mse = self.mse(mu_all.contiguous(), true_all.contiguous())
        mae = self.mae(mu_all, true_all)
        print(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")

        if self.args.probabilistic and len(self.test_log_var) > 0:
            log_var_all = torch.cat(self.test_log_var)
            nll_fn = GaussianNLLLoss(reduction='mean', include_constant=True)
            nll = nll_fn(mu_all, log_var_all, true_all)
            print(f"Test NLL: {nll:.6f}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
