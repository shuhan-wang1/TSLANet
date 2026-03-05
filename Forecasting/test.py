import argparse
import json
import os

import torch

from data_factory import data_provider
from inference import deterministic_predict, gaussian_predict, mc_dropout_predict
from metrics import compute_all_metrics
from model import ProbabilisticTSLANet
from visualization import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description='Test ProbabilisticTSLANet')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to saved model directory')
    # Optional overrides
    parser.add_argument('--mc_samples', type=int, default=None, help='Override MC samples')
    parser.add_argument('--root_path', type=str, default=None, help='Override data root path')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    return parser.parse_args()


def load_config(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)


def main():
    test_args = parse_args()

    # Load config
    args = load_config(test_args.model_dir)

    # Apply overrides
    if test_args.mc_samples is not None:
        args.mc_samples = test_args.mc_samples
    if test_args.root_path is not None:
        args.root_path = test_args.root_path
    if test_args.batch_size is not None:
        args.batch_size = test_args.batch_size

    # Load data
    test_data, test_loader = data_provider(args, 'test')

    # Load model
    model = ProbabilisticTSLANet(args)
    model.load_state_dict(torch.load(
        os.path.join(test_args.model_dir, 'model_weights.pt'),
        map_location='cpu'))

    # Run inference (auto-detect mode from args)
    if args.mc_dropout:
        print(f"Running MC Dropout inference ({args.mc_samples} samples)...")
        results = mc_dropout_predict(model, test_loader, args.pred_len,
                                     num_samples=args.mc_samples)
    elif args.probabilistic:
        print("Running Gaussian inference (aleatoric only)...")
        results = gaussian_predict(model, test_loader, args.pred_len)
    else:
        print("Running deterministic inference...")
        results = deterministic_predict(model, test_loader, args.pred_len)

    # Compute metrics
    metrics = compute_all_metrics(results)

    # Print metrics
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    output_dir = os.path.join(test_args.model_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Convert metrics to JSON-serializable format
    json_metrics = {}
    for k, v in metrics.items():
        json_metrics[k] = float(v) if isinstance(v, (int, float)) else v

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(json_metrics, f, indent=2)

    # Generate plots (only if model has uncertainty)
    if args.probabilistic or args.mc_dropout:
        print("\nGenerating plots...")
        generate_all_plots(results, os.path.join(output_dir, 'plots'), prefix='')

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
