"""
Training script for Probabilistic TSLANet.

Automates data retrieval, trains (or fine-tunes) the models, and saves the final weights.

Usage:
    # Probabilistic TSLANet with default settings
    python train.py --data custom --root_path ../data/weather --data_path weather.csv \
        --pred_len 96 --save_dir saved_models

    # Deep Ensemble (5 members)
    python train.py --data custom --root_path ../data/weather --data_path weather.csv \
        --deep_ensemble True --ensemble_size 5 --save_dir saved_models

    # LSTM baseline
    python train.py --data custom --root_path ../data/weather --data_path weather.csv \
        --model_type lstm --load_from_pretrained False --save_dir saved_models

    # Deterministic TSLANet (MSE loss, ablation baseline)
    python train.py --data custom --root_path ../data/weather --data_path weather.csv \
        --probabilistic False --save_dir saved_models
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from shared_config import (
    create_argument_parser,
    setup_global_args,
    make_run_description,
    save_config,
)


def parse_train_args():
    """Parse training-specific arguments."""
    parser = create_argument_parser()
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='directory for saved final model weights and config')
    return parser.parse_args()


def train_single_model(args, train_loader, val_loader, checkpoint_path, seed=None):
    """
    Train a single model (TSLANet or LSTM) and return the best checkpoint path.

    Handles both self-supervised pretraining (TSLANet only) and supervised fine-tuning.
    """
    if seed is not None:
        L.seed_everything(seed)

    # Deferred imports (after global args are set)
    from prob_training import ProbModelPretraining, ProbModelTraining
    from baseline_lstm import LSTMModelTraining

    # ---- Pretraining (TSLANet only) ----
    pretrained_path = None
    if args.model_type == 'tslanet' and args.load_from_pretrained:
        pretrain_cb = ModelCheckpoint(
            dirpath=checkpoint_path,
            save_top_k=1,
            filename=(f'pretrain-seed{seed}' + '-{epoch}') if seed else 'pretrain-{epoch}',
            monitor='val_loss',
            mode='min',
        )
        pretrain_trainer = L.Trainer(
            default_root_dir=checkpoint_path,
            accelerator='auto',
            devices=1,
            num_sanity_val_steps=0,
            max_epochs=args.pretrain_epochs,
            callbacks=[pretrain_cb, LearningRateMonitor('epoch'), TQDMProgressBar(refresh_rate=500)],
        )
        pretrain_trainer.logger._log_graph = False
        pretrain_trainer.logger._default_hp_metric = None

        pretrain_model = ProbModelPretraining(args)
        pretrain_trainer.fit(pretrain_model, train_loader, val_loader)
        pretrained_path = pretrain_cb.best_model_path

    # ---- Fine-tuning ----
    train_cb = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=1,
        filename=(f'train-seed{seed}' + '-{epoch}') if seed else 'train-{epoch}',
        monitor='val_mse',
        mode='min',
    )
    trainer = L.Trainer(
        default_root_dir=checkpoint_path,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=args.train_epochs,
        callbacks=[train_cb, LearningRateMonitor('epoch'), TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    if args.model_type == 'tslanet':
        if pretrained_path:
            lit_model = ProbModelTraining.load_from_checkpoint(pretrained_path, args=args)
        else:
            lit_model = ProbModelTraining(args)
    else:
        lit_model = LSTMModelTraining(args)

    trainer.fit(lit_model, train_loader, val_loader)

    best_path = train_cb.best_model_path
    print(f"Best model checkpoint: {best_path}")
    return best_path


def _save_single_weights(args, ckpt_path, dest_dir):
    """
    Extract raw model state_dict from a Lightning checkpoint and save both formats.

    Saves:
        dest_dir/best_checkpoint.ckpt  -- full Lightning checkpoint
        dest_dir/model_weights.pt      -- raw state_dict only
    """
    from prob_training import ProbModelTraining
    from baseline_lstm import LSTMModelTraining

    os.makedirs(dest_dir, exist_ok=True)

    # Copy Lightning checkpoint
    ckpt_dest = os.path.join(dest_dir, 'best_checkpoint.ckpt')
    shutil.copy2(ckpt_path, ckpt_dest)

    # Extract and save raw state_dict
    if args.model_type == 'tslanet':
        lit_model = ProbModelTraining.load_from_checkpoint(ckpt_path, args=args)
    else:
        lit_model = LSTMModelTraining.load_from_checkpoint(ckpt_path, args=args)

    weights_dest = os.path.join(dest_dir, 'model_weights.pt')
    torch.save(lit_model.model.state_dict(), weights_dest)
    print(f"  Saved weights to: {weights_dest}")


def save_final_model(args, best_path, save_dir, model_paths=None):
    """
    Save the final model weights and config to a standardized directory.

    For single models:
        save_dir/config.json + model_weights.pt + best_checkpoint.ckpt

    For deep ensembles:
        save_dir/config.json + ensemble_0/ + ensemble_1/ + ...
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config = save_config(args, save_dir)

    if model_paths:
        # Deep ensemble: save each member
        ensemble_paths = []
        for i, path in enumerate(model_paths):
            member_dir = os.path.join(save_dir, f'ensemble_{i}')
            print(f"Saving ensemble member {i}...")
            _save_single_weights(args, path, member_dir)
            ensemble_paths.append(f'ensemble_{i}')

        # Update config with ensemble info
        config['_ensemble_paths'] = ensemble_paths
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    else:
        # Single model
        _save_single_weights(args, best_path, save_dir)


def run_training(args):
    """
    Core training function (importable by run_probabilistic.py).

    Args:
        args: parsed Namespace with all configuration

    Returns:
        save_dir: path to directory containing saved model and config
    """
    from data_factory import data_provider

    run_desc = make_run_description(args)
    print(f"========== Training: {run_desc} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_desc}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(args.seed)

    # Load data
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    print("Dataset loaded ...")

    # Save directory
    save_dir = os.path.join(args.save_dir, run_desc)

    if args.deep_ensemble:
        print(f"\n--- Deep Ensemble: training {args.ensemble_size} members ---")
        model_paths = []
        for i in range(args.ensemble_size):
            seed_i = args.seed + i * 1000
            print(f"\n=== Ensemble member {i+1}/{args.ensemble_size} (seed={seed_i}) ===")
            ckpt_dir = os.path.join(CHECKPOINT_PATH, f'ensemble_{i}')
            os.makedirs(ckpt_dir, exist_ok=True)
            path = train_single_model(args, train_loader, val_loader, ckpt_dir, seed=seed_i)
            model_paths.append(path)

        save_final_model(args, best_path=None, save_dir=save_dir, model_paths=model_paths)
    else:
        best_path = train_single_model(args, train_loader, val_loader,
                                       CHECKPOINT_PATH, seed=args.seed)
        save_final_model(args, best_path=best_path, save_dir=save_dir)

    print(f"\n========== Training complete ==========")
    print(f"Final model saved to: {save_dir}")
    print(f"  config.json          -- model configuration")
    print(f"  model_weights.pt     -- raw state_dict")
    print(f"  best_checkpoint.ckpt -- Lightning checkpoint")

    return save_dir


def main():
    args = parse_train_args()
    setup_global_args(args)
    run_training(args)


if __name__ == '__main__':
    main()
