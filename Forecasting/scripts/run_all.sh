#!/bin/bash
# Run all 8 experiments: 4 configs x 2 pred_lens
# Usage: cd Forecasting && bash scripts/run_all.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42"

for pred_len in 96 336; do
  echo "===== pred_len=$pred_len ====="

  # A1: Deterministic
  echo "--- A1: Deterministic ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic False --mc_dropout False --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A2: Epistemic only (MSE + MC Dropout)
  echo "--- A2: Epistemic Only ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic False --mc_dropout True --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A3: Aleatoric only (Gaussian NLL, no MC)
  echo "--- A3: Aleatoric Only ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout False --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A4: Full (Gaussian NLL + MC Dropout)
  echo "--- A4: Full ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout True --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"
done

echo "All experiments completed!"
