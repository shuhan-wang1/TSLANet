"""
Convenience wrapper that runs training followed by testing in one go.

This script calls train.py's run_training() then test.py's run_testing()
sequentially. For spec-compliant usage, use train.py and test.py directly.

Usage:
    # Train + test in one command (same args as train.py)
    python run_probabilistic.py --data custom --root_path ../data/weather --data_path weather.csv \
        --pred_len 96 --model_type tslanet --probabilistic True --mc_dropout True

    # For separate train/test (recommended for submission):
    #   python train.py --data custom --root_path ../data/weather --data_path weather.csv ...
    #   python test.py --model_dir saved_models/<run_desc>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_config import create_argument_parser, setup_global_args
from train import run_training
from test import run_testing


def parse_args():
    """Parse args using the shared parser + train-specific --save_dir."""
    parser = create_argument_parser()
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='directory for saved final model weights and config')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_global_args(args)

    # Train and get saved model directory
    save_dir = run_training(args)

    # Test using the saved model
    run_testing(model_dir=save_dir)


if __name__ == '__main__':
    main()
