import torch
import torch.nn as nn


def deterministic_predict(model, dataloader, pred_len):
    """A1: Deterministic model, single forward pass."""
    model.eval()
    all_mu = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, _, _ in dataloader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -pred_len:, :].float()

            outputs = model(batch_x)
            outputs = outputs[:, -pred_len:, :]

            all_mu.append(outputs)
            all_targets.append(batch_y)

    mu_mean = torch.cat(all_mu, dim=0)
    targets = torch.cat(all_targets, dim=0)
    zeros = torch.zeros_like(mu_mean)

    return {
        'mu_mean': mu_mean,
        'targets': targets,
        'epistemic_var': zeros,
        'aleatoric_var': zeros,
        'total_var': zeros,
    }


def gaussian_predict(model, dataloader, pred_len):
    """A3: Gaussian head, single forward pass, aleatoric only."""
    model.eval()
    all_mu = []
    all_log_var = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, _, _ in dataloader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -pred_len:, :].float()

            mu, log_var = model(batch_x)
            mu = mu[:, -pred_len:, :]
            log_var = log_var[:, -pred_len:, :]

            all_mu.append(mu)
            all_log_var.append(log_var)
            all_targets.append(batch_y)

    mu_mean = torch.cat(all_mu, dim=0)
    log_var = torch.cat(all_log_var, dim=0)
    targets = torch.cat(all_targets, dim=0)

    aleatoric_var = torch.exp(log_var)
    zeros = torch.zeros_like(mu_mean)

    return {
        'mu_mean': mu_mean,
        'targets': targets,
        'epistemic_var': zeros,
        'aleatoric_var': aleatoric_var,
        'total_var': aleatoric_var,
    }


def mc_dropout_predict(model, dataloader, pred_len, num_samples=50):
    """A2 & A4: K stochastic forward passes with dropout enabled."""
    model.eval()
    # Enable ONLY nn.Dropout layers (not DropPath)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    is_probabilistic = hasattr(model, 'mu_head')

    all_targets = []
    # Collect targets once
    target_collected = False

    # Run K forward passes
    all_mu_samples = []
    all_log_var_samples = []

    for k in range(num_samples):
        batch_mus = []
        batch_log_vars = []
        batch_targets = []

        with torch.no_grad():
            for batch_x, batch_y, _, _ in dataloader:
                batch_x = batch_x.float()
                batch_y = batch_y[:, -pred_len:, :].float()

                if is_probabilistic:
                    mu, log_var = model(batch_x)
                    mu = mu[:, -pred_len:, :]
                    log_var = log_var[:, -pred_len:, :]
                    batch_mus.append(mu)
                    batch_log_vars.append(log_var)
                else:
                    outputs = model(batch_x)
                    outputs = outputs[:, -pred_len:, :]
                    batch_mus.append(outputs)

                if not target_collected:
                    batch_targets.append(batch_y)

        all_mu_samples.append(torch.cat(batch_mus, dim=0))
        if is_probabilistic:
            all_log_var_samples.append(torch.cat(batch_log_vars, dim=0))

        if not target_collected:
            all_targets = torch.cat(batch_targets, dim=0)
            target_collected = True

    # Stack: (K, N, T, M)
    mu_stack = torch.stack(all_mu_samples, dim=0)
    mu_mean = mu_stack.mean(dim=0)
    epistemic_var = mu_stack.var(dim=0)

    if is_probabilistic:
        log_var_stack = torch.stack(all_log_var_samples, dim=0)
        aleatoric_var = torch.exp(log_var_stack).mean(dim=0)
    else:
        aleatoric_var = torch.zeros_like(mu_mean)

    total_var = epistemic_var + aleatoric_var

    return {
        'mu_mean': mu_mean,
        'targets': all_targets,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_var': total_var,
    }
