"""
Main entry point for probabilistic TSLANet forecasting with uncertainty quantification.

Usage examples:

    # Probabilistic TSLANet with MC Dropout
    python run_probabilistic.py --data custom --root_path ../data/weather --data_path weather.csv \
        --probabilistic True --mc_dropout True --mc_samples 50 \
        --seq_len 96 --pred_len 96 --model_type tslanet

    # Deterministic TSLANet (ablation baseline)
    python run_probabilistic.py --data custom --root_path ../data/weather --data_path weather.csv \
        --probabilistic False --model_type tslanet

    # LSTM baseline with MC Dropout
    python run_probabilistic.py --data custom --root_path ../data/weather --data_path weather.csv \
        --probabilistic True --mc_dropout True --model_type lstm

    # Deep Ensemble (5 members)
    python run_probabilistic.py --data custom --root_path ../data/weather --data_path weather.csv \
        --probabilistic True --deep_ensemble True --ensemble_size 5 --model_type tslanet
"""

import argparse
import datetime
import json
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from utils import save_copy_of_files, str2bool

# These will be imported AFTER args are set up (see below)
# from prob_model import ProbabilisticTSLANet
# from prob_training import ProbModelPretraining, ProbModelTraining
# from prob_inference import mc_dropout_predict, deterministic_predict, deep_ensemble_predict
# from prob_metrics import compute_all_metrics
# from prob_visualization import generate_all_plots, plot_metrics_comparison
# from baseline_lstm import LSTMModelTraining


def parse_args():
    parser = argparse.ArgumentParser(description='Probabilistic TSLANet Forecasting')

    # ========== Data args (same as original) ==========
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

    # ========== Probabilistic args (NEW) ==========
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

    # ========== Model selection ==========
    parser.add_argument('--model_type', type=str, default='tslanet',
                        choices=['tslanet', 'lstm'],
                        help='backbone model type')
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--lstm_layers', type=int, default=2)

    return parser.parse_args()


def setup_global_args(args):
    """
    Inject args into the TSLANet_Forecasting module namespace.
    Required because the original TSLANet classes (Adaptive_Spectral_Block,
    TSLANet_layer) reference a module-level global `args`.
    """
    import TSLANet_Forecasting as tsla_module
    tsla_module.args = args


def make_run_description(args):
    model_tag = args.model_type.upper()
    prob_tag = 'prob' if args.probabilistic else 'det'
    mc_tag = f'mc{args.mc_samples}' if args.mc_dropout else 'nomc'
    ens_tag = f'ens{args.ensemble_size}' if args.deep_ensemble else 'noens'
    ts = datetime.datetime.now().strftime('%H_%M')
    desc = (f"{args.data_path.split('.')[0]}_{model_tag}_{prob_tag}_{mc_tag}_{ens_tag}_"
            f"emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}_"
            f"pl{args.pred_len}_bs{args.batch_size}_{ts}")
    return desc


def train_single_model(args, train_loader, val_loader, test_loader, checkpoint_path, seed=None):
    """Train a single model (TSLANet or LSTM) and return the best checkpoint path."""
    if seed is not None:
        L.seed_everything(seed)

    # Import here after args are set up
    from prob_training import ProbModelPretraining, ProbModelTraining
    from baseline_lstm import LSTMModelTraining

    # ---- Pretraining (TSLANet only) ----
    pretrained_path = None
    if args.model_type == 'tslanet' and args.load_from_pretrained:
        pretrain_cb = ModelCheckpoint(
            dirpath=checkpoint_path,
            save_top_k=1,
            filename=f'pretrain-seed{seed}' + '-{epoch}' if seed else 'pretrain-{epoch}',
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
        filename=f'train-seed{seed}' + '-{epoch}' if seed else 'train-{epoch}',
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
    print(f"Best model saved to: {best_path}")

    return best_path


def main():
    args = parse_args()

    # Set up global args for original TSLANet classes
    setup_global_args(args)

    # Now safe to import probabilistic modules
    from data_factory import data_provider
    from prob_training import ProbModelTraining
    from prob_inference import mc_dropout_predict, deterministic_predict, deep_ensemble_predict
    from prob_metrics import compute_all_metrics
    from prob_visualization import generate_all_plots, plot_metrics_comparison
    from baseline_lstm import LSTMModelTraining

    # Run description and paths
    run_desc = make_run_description(args)
    print(f"========== {run_desc} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_desc}"
    PLOTS_DIR = os.path.join(CHECKPOINT_PATH, 'plots')
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Save config
    with open(os.path.join(CHECKPOINT_PATH, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(args.seed)

    # Load data
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    print("Dataset loaded ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Training ==========
    if args.deep_ensemble:
        print(f"\n--- Deep Ensemble: training {args.ensemble_size} members ---")
        model_paths = []
        for i in range(args.ensemble_size):
            seed_i = args.seed + i * 1000
            print(f"\n=== Ensemble member {i+1}/{args.ensemble_size} (seed={seed_i}) ===")
            ckpt_dir = os.path.join(CHECKPOINT_PATH, f'ensemble_{i}')
            os.makedirs(ckpt_dir, exist_ok=True)
            path = train_single_model(args, train_loader, val_loader, test_loader,
                                      ckpt_dir, seed=seed_i)
            model_paths.append(path)

        # Ensemble inference
        model_cls = ProbModelTraining if args.model_type == 'tslanet' else LSTMModelTraining
        results = deep_ensemble_predict(
            model_cls, model_paths, args, test_loader, args.pred_len, device=device
        )
    else:
        # Single model training
        best_path = train_single_model(args, train_loader, val_loader, test_loader,
                                       CHECKPOINT_PATH, seed=args.seed)

        # Load best model
        if args.model_type == 'tslanet':
            lit_model = ProbModelTraining.load_from_checkpoint(best_path, args=args)
        else:
            lit_model = LSTMModelTraining.load_from_checkpoint(best_path, args=args)
        model = lit_model.model.to(device)

        # Inference
        if args.probabilistic and args.mc_dropout:
            print(f"\n--- MC Dropout inference (K={args.mc_samples}) ---")
            results = mc_dropout_predict(
                model, test_loader, args.pred_len,
                num_samples=args.mc_samples, device=device
            )
        elif args.probabilistic:
            print("\n--- Deterministic probabilistic inference ---")
            results = deterministic_predict(model, test_loader, args.pred_len, device=device)
        else:
            print("\n--- Deterministic inference ---")
            results = deterministic_predict(model, test_loader, args.pred_len, device=device)

    # ========== Metrics ==========
    print("\n--- Computing metrics ---")
    metrics = compute_all_metrics(results)

    print("\n========== Results ==========")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")

    # ========== Visualization ==========
    if args.probabilistic:
        print("\n--- Generating plots ---")
        prefix = 'mc_' if args.mc_dropout else ('ens_' if args.deep_ensemble else '')
        generate_all_plots(results, PLOTS_DIR, prefix=prefix)

    # ========== Save results ==========
    # Excel
    df = pd.DataFrame([metrics])
    df.to_excel(os.path.join(CHECKPOINT_PATH,
                             f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"),
                index=False)

    # Text
    os.makedirs("textOutput", exist_ok=True)
    with open(f"textOutput/Prob_{os.path.basename(args.data_path)}.txt", 'a') as f:
        f.write(run_desc + "\n")
        for key, val in metrics.items():
            f.write(f'  {key}: {val:.6f}\n')
        f.write('\n')

    print(f"\nResults saved to {CHECKPOINT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()
