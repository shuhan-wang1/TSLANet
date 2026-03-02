"""
Shared configuration utilities for train.py and test.py.

Contains the argument parser, global args setup, run description generation,
and config save/load functions used by both training and testing entry points.
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import str2bool


def create_argument_parser():
    """
    Create the base argument parser with all model/data/training/probabilistic args.

    Returns the parser object (not parsed args) so train.py and test.py
    can add their own extra arguments before calling parse_args().
    """
    parser = argparse.ArgumentParser(description='Probabilistic TSLANet Forecasting')

    # ========== Data args ==========
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../data/weather',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding [timeF, fixed, learned]')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature for S/MS')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features')

    # ========== Sequence lengths ==========
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # ========== Optimization ==========
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    # ========== TSLANet architecture ==========
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # ========== TSLANet components ==========
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # ========== Probabilistic args ==========
    parser.add_argument('--probabilistic', type=str2bool, default=True,
                        help='use Gaussian NLL loss (True) or MSE (False)')
    parser.add_argument('--mc_dropout', type=str2bool, default=True,
                        help='use MC Dropout for epistemic uncertainty')
    parser.add_argument('--mc_samples', type=int, default=50,
                        help='number of MC Dropout forward passes')
    parser.add_argument('--deep_ensemble', type=str2bool, default=False,
                        help='train multiple models with different seeds')
    parser.add_argument('--ensemble_size', type=int, default=5,
                        help='number of ensemble members')

    # ========== Uncertainty method ==========
    parser.add_argument('--uncertainty_method', type=str, default='gaussian',
                        choices=['gaussian', 'evidential'],
                        help='uncertainty estimation method: '
                             'gaussian=Gaussian NLL with optional MC/Ensemble, '
                             'evidential=Deep Evidential Regression (NIG prior)')

    # ========== DER-specific hyperparameters ==========
    parser.add_argument('--lambda_evd', type=float, default=0.05,
                        help='evidential regularization weight (DER only)')
    parser.add_argument('--anneal_epochs', type=int, default=5,
                        help='epochs to linearly anneal evidence regularization (DER only)')
    parser.add_argument('--use_coverage_loss', type=str2bool, default=False,
                        help='add coverage calibration loss during training '
                             '(works with both Gaussian and Evidential modes)')
    parser.add_argument('--coverage_target', type=float, default=0.9,
                        help='target prediction interval coverage probability')
    parser.add_argument('--lambda_coverage', type=float, default=0.1,
                        help='coverage loss weight')

    # ========== Model selection ==========
    parser.add_argument('--model_type', type=str, default='tslanet',
                        choices=['tslanet', 'lstm'],
                        help='backbone model type')
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--lstm_layers', type=int, default=2)

    return parser


def setup_global_args(args):
    """
    Inject args into the TSLANet_Forecasting module namespace.

    Required because the original TSLANet classes (Adaptive_Spectral_Block,
    TSLANet_layer) reference a module-level global `args`.
    Must be called BEFORE importing any model classes.
    """
    import TSLANet_Forecasting as tsla_module
    tsla_module.args = args


def make_run_description(args):
    """Generate a descriptive run name from args."""
    model_tag = args.model_type.upper()
    method = getattr(args, 'uncertainty_method', 'gaussian')
    if method == 'evidential':
        prob_tag = 'der'
    elif args.probabilistic:
        prob_tag = 'prob'
    else:
        prob_tag = 'det'
    mc_tag = f'mc{args.mc_samples}' if args.mc_dropout else 'nomc'
    ens_tag = f'ens{args.ensemble_size}' if args.deep_ensemble else 'noens'
    ts = datetime.datetime.now().strftime('%H_%M')
    desc = (f"{args.data_path.split('.')[0]}_{model_tag}_{prob_tag}_{mc_tag}_{ens_tag}_"
            f"emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}_"
            f"pl{args.pred_len}_bs{args.batch_size}_{ts}")
    return desc


def save_config(args, save_dir):
    """Save args as config.json to the given directory."""
    os.makedirs(save_dir, exist_ok=True)
    config = vars(args).copy()
    config['_training_timestamp'] = datetime.datetime.now().isoformat()
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return config


def load_config(model_dir):
    """
    Load config.json from a saved model directory and return as argparse.Namespace.

    Strips internal keys that are not model/data arguments.
    """
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Remove internal keys
    config.pop('_training_timestamp', None)
    config.pop('_ensemble_paths', None)
    config.pop('save_dir', None)
    return argparse.Namespace(**config)
