import sys
import os

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prob_model import ProbabilisticTSLANet
from prob_losses import GaussianNLLLoss


class ProbModelPretraining(L.LightningModule):
    """Self-supervised pretraining via masked patch reconstruction.
    Identical to original model_pretraining but wraps ProbabilisticTSLANet."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = ProbabilisticTSLANet(args)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float()

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class ProbModelTraining(L.LightningModule):
    """
    Lightning module for probabilistic TSLANet training.

    When args.probabilistic=True, trains with Gaussian NLL loss and
    the model outputs (mu, log_var). Otherwise, falls back to the
    deterministic TSLANet with MSE loss for ablation.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.model = ProbabilisticTSLANet(args)

        if args.probabilistic:
            self.criterion = GaussianNLLLoss(reduction='mean')
        else:
            self.criterion = nn.MSELoss()

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()

        # Storage for test-time predictions
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

        if self.args.probabilistic:
            mu, log_var = self.model(batch_x)
            mu = mu[:, -self.args.pred_len:, :]
            log_var = log_var[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :]

            loss = self.criterion(mu, log_var, batch_y)
            pred = mu.detach().cpu()
        else:
            # Deterministic mode: only use mu_head output
            mu, _ = self.model(batch_x)
            mu = mu[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :]

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

        if self.args.probabilistic:
            mu, log_var = self.model(batch_x)
            mu = mu[:, -self.args.pred_len:, :]
            log_var = log_var[:, -self.args.pred_len:, :]
            self.test_mu.append(mu.detach().cpu())
            self.test_log_var.append(log_var.detach().cpu())
        else:
            mu, _ = self.model(batch_x)
            mu = mu[:, -self.args.pred_len:, :]
            self.test_mu.append(mu.detach().cpu())

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
