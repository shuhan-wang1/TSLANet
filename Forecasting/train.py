import argparse
import datetime
import json
import os

import torch
import torch.nn as nn

from data_factory import data_provider
from losses import GaussianNLLLoss
from model import ProbabilisticTSLANet
from utils import str2bool


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProbabilisticTSLANet')

    # Data
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='data/weather')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # Model
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # TSLANet components
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--train_epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    # Pretraining
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)

    # Probabilistic
    parser.add_argument('--probabilistic', type=str2bool, default=True)
    parser.add_argument('--mc_dropout', type=str2bool, default=True)
    parser.add_argument('--mc_samples', type=int, default=50)

    # Output
    parser.add_argument('--save_dir', type=str, default='saved_models')

    return parser.parse_args()


def make_save_dir(args):
    data_name = args.data_path.split('.')[0]
    prob_tag = 'prob' if args.probabilistic else 'det'
    mc_tag = 'mc' if args.mc_dropout else 'nomc'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    desc = f"{data_name}_{prob_tag}_{mc_tag}_pl{args.pred_len}_{timestamp}"
    save_dir = os.path.join(args.save_dir, desc)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_config(args, save_dir):
    config = vars(args).copy()
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def evaluate_pretrain(model, val_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float()
            preds, target = model.pretrain(batch_x)
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * model.mask).sum() / model.mask.sum()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, val_loader, args):
    model.eval()
    total_mse = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()

            if args.probabilistic:
                mu, log_var = model(batch_x)
                mu = mu[:, -args.pred_len:, :]
                mse = ((mu - batch_y) ** 2).mean()
            else:
                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                mse = ((outputs - batch_y) ** 2).mean()

            total_mse += mse.item()
            n_batches += 1
    return total_mse / max(n_batches, 1)


def pretrain(model, train_loader, val_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(args.pretrain_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            preds, target = model.pretrain(batch_x)
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * model.mask).sum() / model.mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = evaluate_pretrain(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs} | val_loss: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def train(model, train_loader, val_loader, args):
    criterion = GaussianNLLLoss() if args.probabilistic else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_mse = float('inf')
    best_state = None

    for epoch in range(args.train_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y[:, -args.pred_len:, :].float()

            if args.probabilistic:
                mu, log_var = model(batch_x)
                mu = mu[:, -args.pred_len:, :]
                log_var = log_var[:, -args.pred_len:, :]
                loss = criterion(mu, log_var, batch_y)
            else:
                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

        # Validation
        val_mse = evaluate(model, val_loader, args)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{args.train_epochs} | val_mse: {val_mse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')

    # Create model
    model = ProbabilisticTSLANet(args)

    # Phase 1: Self-supervised pretraining
    if args.load_from_pretrained:
        print("=== Phase 1: Self-supervised pretraining ===")
        pretrain(model, train_loader, val_loader, args)

    # Phase 2: Supervised fine-tuning
    print("=== Phase 2: Supervised fine-tuning ===")
    best_state = train(model, train_loader, val_loader, args)

    # Save
    save_dir = make_save_dir(args)
    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, 'model_weights.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    save_config(args, save_dir)

    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
