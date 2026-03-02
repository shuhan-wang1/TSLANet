import torch
import torch.nn as nn


def enable_mc_dropout(model):
    """
    Enable only nn.Dropout layers during inference for MC Dropout.

    Specifically targets Dropout in ICB blocks. Does NOT enable DropPath
    (stochastic depth), which would be too aggressive for uncertainty estimation.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, dataloader, pred_len, num_samples=50, device='cuda'):
    """
    MC Dropout inference: run K stochastic forward passes and decompose uncertainty.

    Args:
        model: ProbabilisticTSLANet model (raw nn.Module, not Lightning wrapper)
        dataloader: test DataLoader
        pred_len: prediction length to slice outputs
        num_samples: K — number of stochastic forward passes
        device: 'cuda' or 'cpu'

    Returns:
        dict with keys:
            mu_samples: (K, N, T, M)    mean predictions from each pass
            log_var_samples: (K, N, T, M) log-var predictions from each pass
            targets: (N, T, M)           ground truth
            mu_mean: (N, T, M)           ensemble mean prediction
            epistemic_var: (N, T, M)     variance of means across passes
            aleatoric_var: (N, T, M)     mean of predicted variances
            total_var: (N, T, M)         epistemic + aleatoric
    """
    model.eval()
    enable_mc_dropout(model)
    model = model.to(device)

    all_mu_samples = []
    all_log_var_samples = []
    all_targets = None

    for k in range(num_samples):
        mu_list, lv_list = [], []
        tgt_list = []

        for batch in dataloader:
            batch_x, batch_y, _, _ = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y[:, -pred_len:, :].float()

            with torch.no_grad():
                mu, log_var = model(batch_x)
                mu = mu[:, -pred_len:, :]
                log_var = log_var[:, -pred_len:, :]

            mu_list.append(mu.cpu())
            lv_list.append(log_var.cpu())

            if k == 0:
                tgt_list.append(batch_y.cpu())

        all_mu_samples.append(torch.cat(mu_list, dim=0))
        all_log_var_samples.append(torch.cat(lv_list, dim=0))

        if k == 0:
            all_targets = torch.cat(tgt_list, dim=0)

    # Stack: (K, N, T, M)
    mu_samples = torch.stack(all_mu_samples, dim=0)
    log_var_samples = torch.stack(all_log_var_samples, dim=0)

    # Uncertainty decomposition
    mu_mean = mu_samples.mean(dim=0)                         # (N, T, M)
    epistemic_var = mu_samples.var(dim=0)                    # (N, T, M)
    aleatoric_var = torch.exp(log_var_samples).mean(dim=0)   # (N, T, M)
    total_var = epistemic_var + aleatoric_var                 # (N, T, M)

    return {
        'mu_samples': mu_samples,
        'log_var_samples': log_var_samples,
        'targets': all_targets,
        'mu_mean': mu_mean,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_var': total_var,
    }


def deterministic_predict(model, dataloader, pred_len, device='cuda'):
    """
    Standard deterministic inference (no MC Dropout).
    For probabilistic models, uses the single-pass mu and log_var.

    Returns: same dict format as mc_dropout_predict
    """
    model.eval()
    model = model.to(device)

    mu_list, lv_list, tgt_list = [], [], []

    for batch in dataloader:
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y[:, -pred_len:, :].float()

        with torch.no_grad():
            mu, log_var = model(batch_x)
            mu = mu[:, -pred_len:, :]
            log_var = log_var[:, -pred_len:, :]

        mu_list.append(mu.cpu())
        lv_list.append(log_var.cpu())
        tgt_list.append(batch_y.cpu())

    mu_all = torch.cat(mu_list, dim=0)
    log_var_all = torch.cat(lv_list, dim=0)
    targets = torch.cat(tgt_list, dim=0)

    aleatoric_var = torch.exp(log_var_all)

    return {
        'mu_samples': mu_all.unsqueeze(0),
        'log_var_samples': log_var_all.unsqueeze(0),
        'targets': targets,
        'mu_mean': mu_all,
        'epistemic_var': torch.zeros_like(mu_all),
        'aleatoric_var': aleatoric_var,
        'total_var': aleatoric_var,
    }


def deep_ensemble_predict(model_class, model_paths, args, dataloader, pred_len, device='cuda'):
    """
    Deep Ensemble inference: load M independently-trained models,
    run each deterministically, combine as mixture of Gaussians.

    Args:
        model_class: the Lightning module class to load from checkpoint
        model_paths: list of checkpoint paths (one per ensemble member)
        args: configuration namespace
        dataloader: test DataLoader
        pred_len: prediction length
        device: 'cuda' or 'cpu'

    Returns: same dict format as mc_dropout_predict
    """
    all_mu = []
    all_log_var = []
    targets = None

    for path in model_paths:
        lit_model = model_class.load_from_checkpoint(path, args=args)
        model = lit_model.model.to(device)
        model.eval()

        mu_list, lv_list, tgt_list = [], [], []

        for batch in dataloader:
            batch_x, batch_y, _, _ = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y[:, -pred_len:, :].float()

            with torch.no_grad():
                mu, log_var = model(batch_x)
                mu = mu[:, -pred_len:, :]
                log_var = log_var[:, -pred_len:, :]

            mu_list.append(mu.cpu())
            lv_list.append(log_var.cpu())

            if targets is None:
                tgt_list.append(batch_y.cpu())

        all_mu.append(torch.cat(mu_list, dim=0))
        all_log_var.append(torch.cat(lv_list, dim=0))

        if targets is None:
            targets = torch.cat(tgt_list, dim=0)

    # Stack: (M, N, T, M_vars)
    mu_stack = torch.stack(all_mu, dim=0)
    lv_stack = torch.stack(all_log_var, dim=0)

    mu_mean = mu_stack.mean(dim=0)
    epistemic_var = mu_stack.var(dim=0)
    aleatoric_var = torch.exp(lv_stack).mean(dim=0)
    total_var = epistemic_var + aleatoric_var

    return {
        'mu_samples': mu_stack,
        'log_var_samples': lv_stack,
        'targets': targets,
        'mu_mean': mu_mean,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_var': total_var,
    }
