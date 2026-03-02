"""
Testing script for Probabilistic TSLANet.

Loads saved models to produce final metrics and visual results.

Usage:
    # Test a saved model (reads config.json from model_dir)
    python test.py --model_dir saved_models/<run_desc>

    # Override data path (e.g., on a different machine)
    python test.py --model_dir saved_models/<run_desc> --root_path /path/to/data/weather

    # Override number of MC Dropout samples
    python test.py --model_dir saved_models/<run_desc> --mc_samples 100

    # Custom output directory
    python test.py --model_dir saved_models/<run_desc> --output_dir my_results
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import torch

from shared_config import load_config, setup_global_args


def parse_test_args():
    """Parse testing-specific arguments."""
    parser = argparse.ArgumentParser(description='Test Probabilistic TSLANet')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path to saved_models/<run_desc>/ directory '
                             'containing config.json and model weights')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory for results and plots (default: model_dir/results)')
    # Optional overrides
    parser.add_argument('--root_path', type=str, default=None,
                        help='override data root path from saved config')
    parser.add_argument('--data_path', type=str, default=None,
                        help='override data file path from saved config')
    parser.add_argument('--mc_samples', type=int, default=None,
                        help='override number of MC dropout samples')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch size for testing')
    return parser.parse_args()


def load_single_model(args, model_dir):
    """
    Load a single model from a saved directory.

    Tries Lightning checkpoint first, falls back to raw state_dict.

    Args:
        args: Namespace with model architecture config
        model_dir: directory containing best_checkpoint.ckpt and/or model_weights.pt

    Returns:
        model: nn.Module (raw model, not Lightning wrapper)
    """
    from prob_training import ProbModelTraining
    from baseline_lstm import LSTMModelTraining

    ckpt_path = os.path.join(model_dir, 'best_checkpoint.ckpt')
    weights_path = os.path.join(model_dir, 'model_weights.pt')

    if os.path.exists(ckpt_path):
        print(f"  Loading from Lightning checkpoint: {ckpt_path}")
        if args.model_type == 'tslanet':
            lit_model = ProbModelTraining.load_from_checkpoint(ckpt_path, args=args)
        else:
            lit_model = LSTMModelTraining.load_from_checkpoint(ckpt_path, args=args)
        return lit_model.model
    elif os.path.exists(weights_path):
        print(f"  Loading from raw state_dict: {weights_path}")
        from prob_model import ProbabilisticTSLANet
        from baseline_lstm import GaussianLSTM

        if args.model_type == 'tslanet':
            model = ProbabilisticTSLANet(args)
        else:
            model = GaussianLSTM(args)

        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        return model
    else:
        raise FileNotFoundError(
            f"No model found in {model_dir}. "
            f"Expected best_checkpoint.ckpt or model_weights.pt")


def load_ensemble_models(args, model_dir):
    """
    Load all ensemble member models from subdirectories.

    Returns:
        list of nn.Module models
    """
    ensemble_dirs = sorted([
        d for d in os.listdir(model_dir)
        if d.startswith('ensemble_') and os.path.isdir(os.path.join(model_dir, d))
    ])

    if not ensemble_dirs:
        raise FileNotFoundError(f"No ensemble_* directories found in {model_dir}")

    models = []
    for edir in ensemble_dirs:
        print(f"  Loading ensemble member: {edir}")
        model = load_single_model(args, os.path.join(model_dir, edir))
        models.append(model)

    return models


def ensemble_predict_from_models(models, dataloader, pred_len, device):
    """
    Deep ensemble inference from already-loaded model objects.

    Args:
        models: list of nn.Module models
        dataloader: test DataLoader
        pred_len: prediction length
        device: torch device

    Returns:
        dict with same structure as mc_dropout_predict output
    """
    all_mu = []
    all_log_var = []
    targets = None

    for model in models:
        model = model.to(device)
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


def run_testing(model_dir, args_overrides=None):
    """
    Core testing function (importable by run_probabilistic.py).

    Args:
        model_dir: path to saved model directory
        args_overrides: optional dict of arg overrides

    Returns:
        metrics: dict of metric_name -> value
    """
    # Load config
    args = load_config(model_dir)

    # Apply overrides
    if args_overrides:
        for key, val in args_overrides.items():
            if val is not None:
                setattr(args, key, val)

    # Setup global args BEFORE model imports
    setup_global_args(args)

    # Now import modules
    from data_factory import data_provider
    from prob_inference import mc_dropout_predict, deterministic_predict
    from prob_metrics import compute_all_metrics
    from prob_visualization import generate_all_plots

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_data, test_loader = data_provider(args, flag='test')
    print("Test dataset loaded ...")

    # Load model(s)
    is_ensemble = getattr(args, 'deep_ensemble', False)
    if is_ensemble:
        print(f"\n--- Loading ensemble models from {model_dir} ---")
        models = load_ensemble_models(args, model_dir)
        print(f"Loaded {len(models)} ensemble members")
    else:
        print(f"\n--- Loading model from {model_dir} ---")
        model = load_single_model(args, model_dir)

    # Run inference
    uncertainty_method = getattr(args, 'uncertainty_method', 'gaussian')

    if is_ensemble:
        print(f"\n--- Ensemble inference ---")
        results = ensemble_predict_from_models(models, test_loader, args.pred_len, device)
    elif uncertainty_method == 'evidential':
        from prob_der import der_predict
        print("\n--- Deep Evidential Regression inference (single-pass) ---")
        results = der_predict(model, test_loader, args.pred_len, device=device)
    elif getattr(args, 'probabilistic', True) and getattr(args, 'mc_dropout', True):
        mc_samples = getattr(args, 'mc_samples', 50)
        print(f"\n--- MC Dropout inference (K={mc_samples}) ---")
        results = mc_dropout_predict(
            model, test_loader, args.pred_len,
            num_samples=mc_samples, device=device
        )
    elif getattr(args, 'probabilistic', True):
        print("\n--- Deterministic probabilistic inference ---")
        results = deterministic_predict(model, test_loader, args.pred_len, device=device)
    else:
        print("\n--- Deterministic inference ---")
        results = deterministic_predict(model, test_loader, args.pred_len, device=device)

    # Compute metrics
    print("\n--- Computing metrics ---")
    metrics = compute_all_metrics(results)

    print("\n========== Results ==========")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")

    # Output directory
    output_dir = os.path.join(model_dir, 'results')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots
    if getattr(args, 'probabilistic', True) or uncertainty_method == 'evidential':
        print("\n--- Generating plots ---")
        if uncertainty_method == 'evidential':
            prefix = 'der_'
        elif getattr(args, 'mc_dropout', False):
            prefix = 'mc_'
        elif getattr(args, 'deep_ensemble', False):
            prefix = 'ens_'
        else:
            prefix = ''
        generate_all_plots(results, plots_dir, prefix=prefix)

    # Save metrics as JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save metrics as Excel
    df = pd.DataFrame([metrics])
    df.to_excel(os.path.join(output_dir,
                             f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"),
                index=False)

    # Text output
    os.makedirs("textOutput", exist_ok=True)
    data_path = getattr(args, 'data_path', 'unknown')
    with open(f"textOutput/Prob_{os.path.basename(data_path)}.txt", 'a') as f:
        f.write(f"[TEST] {model_dir}\n")
        for key, val in metrics.items():
            f.write(f'  {key}: {val:.6f}\n')
        f.write('\n')

    print(f"\nResults and plots saved to: {output_dir}")
    print("Done testing!")

    return metrics


def main():
    test_args = parse_test_args()

    # Build overrides dict from CLI
    overrides = {}
    if test_args.root_path:
        overrides['root_path'] = test_args.root_path
    if test_args.data_path:
        overrides['data_path'] = test_args.data_path
    if test_args.mc_samples:
        overrides['mc_samples'] = test_args.mc_samples
    if test_args.batch_size:
        overrides['batch_size'] = test_args.batch_size

    # Determine output dir
    output_dir = test_args.output_dir
    if output_dir:
        overrides['_output_dir'] = output_dir

    run_testing(model_dir=test_args.model_dir, args_overrides=overrides)


if __name__ == '__main__':
    main()
